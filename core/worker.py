"""Inference Worker Runtime with Exact Stateful KV-Cache Handover.

Explicit State Invariant:
-------------------------
Let S = [t_0, t_1, ..., t_{K-1}] be the sequence of tokens already processed by the model.
- `past_key_values` contains key-value states for exactly S[:K] (sequence length = K).
- `pending_token_id` is the token t_K chosen from logits at step K-1, ready to be processed.
  It is NOT YET present in `past_key_values`.
- On resumption, the destination worker feeds `pending_token_id` with `past_key_values`,
  ensuring no token is processed twice and no token is skipped.
"""

from typing import Dict, Any, Optional, List, Tuple
import io
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


class InferenceWorker:
    def __init__(
        self,
        worker_id: str,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.worker_id = worker_id
        self.model_name = model_name
        
        # Select device: Metal (mps) if available, else CPU
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        self.torch_dtype = torch_dtype
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def load_model(self):
        """Loads model and tokenizer onto target device using local files first."""
        print(f"[{self.worker_id}] Loading {self.model_name} onto {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=self.torch_dtype,
                local_files_only=True,
            ).to(self.device)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=self.torch_dtype,
            ).to(self.device)
        self.model.eval()
        print(f"[{self.worker_id}] Model loaded successfully.")

    def prefill(self, prompt: str) -> Tuple[torch.Tensor, Any, torch.Tensor, float]:
        """Runs the initial prompt prefill pass.

        Returns:
            prompt_token_ids: 2D tensor of prompt tokens (shape [1, P]).
            past_key_values: KV-cache containing all P tokens.
            first_generated_token_id: Token selected from prompt's final logits (shape [1, 1]).
            prefill_time_ms: Measured prefill wall-clock time in milliseconds.
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_token_ids = inputs["input_ids"].to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(input_ids=prompt_token_ids, use_cache=True)
            # Greedy selection of the first token directly from prefill logits
            next_token_logits = outputs.logits[:, -1, :]
            first_generated_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        prefill_time_ms = (time.perf_counter() - t0) * 1000.0

        return prompt_token_ids, outputs.past_key_values, first_generated_token_id, prefill_time_ms

    def generate(
        self,
        prompt: Optional[str] = None,
        cached_token_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        pending_token_id: Optional[torch.Tensor] = None,
        max_new_tokens: int = 20,
    ) -> Dict[str, Any]:
        """Generates tokens using an explicit state invariant.

        Args:
            prompt: Text prompt to start from (if starting cold).
            cached_token_ids: Tokens already represented in `past_key_values`.
            past_key_values: Key-value states for `cached_token_ids`.
            pending_token_id: Next token to process (not yet in `past_key_values`).
            max_new_tokens: Maximum number of new tokens to generate in this run.

        Returns:
            Dictionary containing token sequences, KV cache, pending token, and timings.
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        step_latencies: List[float] = []
        newly_generated_ids: List[int] = []

        if past_key_values is None:
            # Cold start: run prefill
            assert prompt is not None, "Prompt required for cold generation"
            cached_token_ids, past_key_values, pending_token, prefill_time_ms = self.prefill(prompt)
            print(f"[{self.worker_id}] Prefill for {cached_token_ids.shape[1]} tokens in {prefill_time_ms:.2f} ms")
        else:
            # Resuming from imported state
            assert cached_token_ids is not None and pending_token_id is not None
            prefill_time_ms = 0.0
            pending_token = pending_token_id.to(self.device)
            cached_token_ids = cached_token_ids.to(self.device)
            print(f"[{self.worker_id}] Resumed generation with {cached_token_ids.shape[1]} cached tokens.")

        # Autoregressive decoding loop:
        # Each iteration processes `pending_token`, caches it, and samples the next `pending_token`.
        with torch.no_grad():
            for _ in range(max_new_tokens):
                t_step_start = time.perf_counter()
                
                # Record the token being produced
                current_token_val = pending_token.item()
                newly_generated_ids.append(current_token_val)
                cached_token_ids = torch.cat([cached_token_ids, pending_token], dim=-1)

                outputs = self.model(
                    input_ids=pending_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                step_latencies.append((time.perf_counter() - t_step_start) * 1000.0)

                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                pending_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated_text = self.tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
        full_text = self.tokenizer.decode(cached_token_ids[0], skip_special_tokens=True)

        return {
            "cached_token_ids": cached_token_ids,
            "past_key_values": past_key_values,
            "pending_token_id": pending_token,
            "generated_token_ids": newly_generated_ids,
            "generated_text": generated_text,
            "full_text": full_text,
            "prefill_time_ms": prefill_time_ms,
            "step_latencies_ms": step_latencies,
            "mean_step_latency_ms": sum(step_latencies) / len(step_latencies) if step_latencies else 0.0,
        }

    def export_state(
        self,
        cached_token_ids: torch.Tensor,
        past_key_values: Any,
        pending_token_id: torch.Tensor,
        generated_token_ids: Optional[List[int]] = None,
    ) -> bytes:
        """Serializes the exact state tuple (cached_tokens, kv_cache, pending_token)."""
        t0 = time.perf_counter()

        cpu_kv = []
        if hasattr(past_key_values, "layers"):  # Transformers v5+ DynamicCache
            for layer in past_key_values.layers:
                cpu_kv.append((layer.keys.detach().to("cpu"), layer.values.detach().to("cpu")))
        elif hasattr(past_key_values, "key_cache"):  # Transformers v4 DynamicCache
            for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
                cpu_kv.append((k.detach().to("cpu"), v.detach().to("cpu")))
        elif isinstance(past_key_values, (tuple, list)):
            for layer_k, layer_v in past_key_values:
                cpu_kv.append((layer_k.detach().to("cpu"), layer_v.detach().to("cpu")))
        else:
            raise TypeError(f"Unsupported cache structure: {type(past_key_values)}")

        # Verify invariant: KV sequence length matches cached_token_ids length
        kv_seq_len = cpu_kv[0][0].shape[-2]
        assert kv_seq_len == cached_token_ids.shape[1], (
            f"Invariant violation: KV cache seq_len ({kv_seq_len}) != cached_token_ids ({cached_token_ids.shape[1]})"
        )

        state_dict = {
            "model_name": self.model_name,
            "cached_token_ids": cached_token_ids.detach().to("cpu"),
            "pending_token_id": pending_token_id.detach().to("cpu"),
            "generated_token_ids": generated_token_ids or [],
            "kv_cache": tuple(cpu_kv),
            "num_layers": len(cpu_kv),
            "kv_seq_len": kv_seq_len,
        }

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        payload = buffer.getvalue()

        export_time_ms = (time.perf_counter() - t0) * 1000.0
        payload_size_kb = len(payload) / 1024.0
        print(f"[{self.worker_id}] State exported: {payload_size_kb:.2f} KB (seq_len={kv_seq_len}) in {export_time_ms:.2f} ms")

        return payload

    def import_state(self, payload: bytes) -> Tuple[torch.Tensor, Any, torch.Tensor, List[int], float]:
        """Deserializes payload, validates model compatibility, and reconstructs KV cache.

        Returns:
            Tuple of (cached_token_ids, past_key_values, pending_token_id, prior_generated_ids, import_time_ms)
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        t0 = time.perf_counter()
        buffer = io.BytesIO(payload)
        state_dict = torch.load(buffer, weights_only=True)

        # 1. Compatibility validation
        payload_model = state_dict.get("model_name")
        if payload_model and payload_model != self.model_name:
            print(f"[{self.worker_id}] WARNING: Migrating state from {payload_model} to {self.model_name}")

        cached_token_ids = state_dict["cached_token_ids"].to(self.device)
        pending_token_id = state_dict["pending_token_id"].to(self.device)
        prior_generated_ids = state_dict.get("generated_token_ids", [])
        cpu_kv = state_dict["kv_cache"]

        # Validate layer count against destination model
        expected_layers = self.model.config.num_hidden_layers
        if len(cpu_kv) != expected_layers:
            raise ValueError(
                f"Layer mismatch: imported {len(cpu_kv)} layers, destination model requires {expected_layers}"
            )

        # Validate sequence length invariant
        kv_seq_len = cpu_kv[0][0].shape[-2]
        if kv_seq_len != cached_token_ids.shape[1]:
            raise ValueError(
                f"State corruption: KV length ({kv_seq_len}) != cached tokens ({cached_token_ids.shape[1]})"
            )

        # 2. Reconstruct KV cache according to destination runtime API
        if DynamicCache is not None:
            past_key_values = DynamicCache()
            for i, (layer_k, layer_v) in enumerate(cpu_kv):
                past_key_values.update(layer_k.to(self.device), layer_v.to(self.device), i)
        else:
            device_kv = []
            for layer_k, layer_v in cpu_kv:
                device_kv.append((layer_k.to(self.device), layer_v.to(self.device)))
            past_key_values = tuple(device_kv)

        import_time_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[{self.worker_id}] State imported: {cached_token_ids.shape[1]} tokens in {import_time_ms:.2f} ms")

        return cached_token_ids, past_key_values, pending_token_id, prior_generated_ids, import_time_ms
