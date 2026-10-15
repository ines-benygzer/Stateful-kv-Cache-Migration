"""Comprehensive Unit and Handover Continuation Test for KVCacheSerializer.

Validates:
1. Shape, dtype, and finite-value integrity for both FP16 and INT8.
2. Numerical error bounds on random and constant-valued tensors.
3. Full handover state serialization and restoration.
4. Actual generation continuation with the model for FP16 and INT8 (no unverified claims).
"""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.serializer import (
    quantize_tensor_int8,
    dequantize_tensor_int8,
    KVCacheSerializer,
)
from core.worker import InferenceWorker


def test_tensor_numerical_properties():
    print("=" * 75)
    print("TEST 1: TENSOR INTEGRITY & NUMERICAL ERROR BOUNDS")
    print("=" * 75)

    # 1. Random tensor test
    torch.manual_seed(42)
    x = torch.randn(2, 64, 128, dtype=torch.float32)
    meta = quantize_tensor_int8(x)
    x_hat = dequantize_tensor_int8(meta)

    assert x_hat.shape == x.shape, f"Shape mismatch: {x_hat.shape} vs {x.shape}"
    assert x_hat.dtype == x.dtype, f"Dtype mismatch: {x_hat.dtype} vs {x.dtype}"
    assert torch.isfinite(x_hat).all(), "Non-finite values found in restored tensor"

    max_err = (x_hat - x).abs().max().item()
    mean_err = (x_hat - x).abs().mean().item()
    theoretical_bound = meta["scale"] / 2.0

    print(f"[Random Tensor INT8] Shape: {tuple(x.shape)}")
    print(f"  Scale: {meta['scale']:.6f} | Zero-point: {meta['zero_point']}")
    print(f"  Max Absolute Error:  {max_err:.6f} (Theoretical upper bound: {theoretical_bound:.6f})")
    print(f"  Mean Absolute Error: {mean_err:.6f}")
    assert max_err <= theoretical_bound + 1e-5, f"Error {max_err} exceeds theoretical bound {theoretical_bound}"

    # 2. Constant-valued tensor safety test
    const_val = 17.5
    x_const = torch.full((1, 4, 32, 64), const_val, dtype=torch.float32)
    meta_const = quantize_tensor_int8(x_const)
    x_const_hat = dequantize_tensor_int8(meta_const)

    assert torch.equal(x_const, x_const_hat), "Constant tensor restoration failed"
    print(f"[Constant Tensor INT8] Value: {const_val} -> Exactly restored with zero error.")
    print(">> TEST 1 PASSED: Tensor integrity and error bounds confirmed.\n")


def test_generation_continuation():
    print("=" * 75)
    print("TEST 2: GENERATION CONTINUATION (FP16 vs INT8 vs BASELINE)")
    print("=" * 75)

    prompt = "The edge-to-cloud computing continuum presents unique challenges for distributed AI workloads, such as"
    total_steps = 30
    edge_steps = 15
    cloud_steps = total_steps - edge_steps

    # Reuse a single worker instance to stay well within 8GB RAM on M1
    worker = InferenceWorker(worker_id="M1-Worker")
    worker.load_model()

    # 1. Uninterrupted Baseline
    print("\n--- 1. Generating Uninterrupted Baseline (30 tokens) ---")
    base_res = worker.generate(prompt=prompt, max_new_tokens=total_steps)
    baseline_tokens = base_res["generated_token_ids"]
    print(f"Baseline Tokens: {baseline_tokens}")
    print(f"Baseline Text:\n\"{base_res['full_text']}\"")

    # 2. Phase 1: Generate first 15 tokens
    print(f"\n--- 2. Phase 1: Generating first {edge_steps} tokens ---")
    edge_res = worker.generate(prompt=prompt, max_new_tokens=edge_steps)
    edge_tokens = edge_res["generated_token_ids"]
    print(f"Edge Tokens: {edge_tokens}")

    # Build raw state dict
    raw_state_dict = {
        "model_name": worker.model_name,
        "cached_token_ids": edge_res["cached_token_ids"],
        "pending_token_id": edge_res["pending_token_id"],
        "generated_token_ids": edge_tokens,
        "kv_cache": edge_res["past_key_values"],
    }

    # 3. Test Handover with FP16 Serialization
    print("\n--- 3. Testing Handover with Mode='fp16' ---")
    payload_fp16, metrics_fp16 = KVCacheSerializer.serialize_handover_state(raw_state_dict, mode="fp16")
    restored_fp16_state, stats_fp16 = KVCacheSerializer.deserialize_handover_state(
        payload_fp16, target_device=worker.device
    )

    # Reconstruct KV cache into DynamicCache
    if hasattr(worker.model, "config") and hasattr(restored_fp16_state["kv_cache"], "__len__"):
        from transformers.cache_utils import DynamicCache
        cloud_kv_fp16 = DynamicCache()
        for i, (k, v) in enumerate(restored_fp16_state["kv_cache"]):
            cloud_kv_fp16.update(k.to(worker.device), v.to(worker.device), i)
    else:
        cloud_kv_fp16 = restored_fp16_state["kv_cache"]

    cloud_fp16_res = worker.generate(
        cached_token_ids=restored_fp16_state["cached_token_ids"],
        past_key_values=cloud_kv_fp16,
        pending_token_id=restored_fp16_state["pending_token_id"],
        max_new_tokens=cloud_steps,
    )
    fp16_combined_tokens = edge_tokens + cloud_fp16_res["generated_token_ids"]
    fp16_matches = (fp16_combined_tokens == baseline_tokens)
    print(f"FP16 Combined Tokens: {fp16_combined_tokens}")
    print(f"FP16 Payload Size:    {metrics_fp16['total_state_mb']:.4f} MB ({metrics_fp16['total_state_bytes']} bytes)")
    print(f"FP16 Serialization:   {metrics_fp16['total_serialization_time_ms']:.2f} ms")
    print(f"FP16 Deserialization: {stats_fp16['total_deserialization_time_ms']:.2f} ms")
    print(f">> FP16 Deterministic Match: {fp16_matches}")
    assert fp16_matches, "FP16 handover diverged from baseline!"

    # 4. Test Handover with INT8 Serialization
    print("\n--- 4. Testing Handover with Mode='int8' ---")
    payload_int8, metrics_int8 = KVCacheSerializer.serialize_handover_state(raw_state_dict, mode="int8")
    restored_int8_state, stats_int8 = KVCacheSerializer.deserialize_handover_state(
        payload_int8, target_device=worker.device
    )

    if hasattr(worker.model, "config") and hasattr(restored_int8_state["kv_cache"], "__len__"):
        from transformers.cache_utils import DynamicCache
        cloud_kv_int8 = DynamicCache()
        for i, (k, v) in enumerate(restored_int8_state["kv_cache"]):
            cloud_kv_int8.update(k.to(worker.device), v.to(worker.device), i)
    else:
        cloud_kv_int8 = restored_int8_state["kv_cache"]

    cloud_int8_res = worker.generate(
        cached_token_ids=restored_int8_state["cached_token_ids"],
        past_key_values=cloud_kv_int8,
        pending_token_id=restored_int8_state["pending_token_id"],
        max_new_tokens=cloud_steps,
    )
    int8_combined_tokens = edge_tokens + cloud_int8_res["generated_token_ids"]
    matching_tokens_count = sum(1 for b, q in zip(baseline_tokens, int8_combined_tokens) if b == q)
    token_match_pct = (matching_tokens_count / len(baseline_tokens)) * 100.0

    print(f"INT8 Combined Tokens: {int8_combined_tokens}")
    print(f"INT8 Payload Size:    {metrics_int8['total_state_mb']:.4f} MB ({metrics_int8['total_state_bytes']} bytes)")
    print(f"INT8 Serialization:   {metrics_int8['total_serialization_time_ms']:.2f} ms")
    print(f"INT8 Deserialization: {stats_int8['total_deserialization_time_ms']:.2f} ms")
    print(f"INT8 Max Tensor Error:  {metrics_int8['max_abs_error']:.6f}")
    print(f"INT8 Mean Tensor Error: {metrics_int8['mean_abs_error']:.6f}")
    print(f">> INT8 Token Match Rate against FP16 Baseline: {matching_tokens_count}/{len(baseline_tokens)} ({token_match_pct:.1f}%)")
    print(f"INT8 Resumed Output Text:\n\"{cloud_int8_res['full_text']}\"")
    print("=" * 75)


def main():
    test_tensor_numerical_properties()
    test_generation_continuation()


if __name__ == "__main__":
    main()
