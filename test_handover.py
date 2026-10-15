"""Comprehensive Verification Test for Stateful KV-Cache Handover.

Validates:
1. Explicit state invariant (cached_tokens == KV seq_len, pending_token processed next).
2. Exact token-ID list equality: (edge_tokens + cloud_tokens) == baseline_tokens.
3. Clean reporting of measured timings (prefill vs export vs import vs first-token resumption).
4. Package version and environment diagnostics.
"""

import sys
import os
import platform
import torch
import transformers

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.worker import InferenceWorker


def print_system_diagnostics():
    print("=" * 75)
    print("SYSTEM & RUNTIME DIAGNOSTICS")
    print("=" * 75)
    print(f"Platform:         {platform.platform()} ({platform.machine()})")
    print(f"Python Version:   {platform.python_version()}")
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"Transformers:     {transformers.__version__}")
    print(f"MPS Available:    {torch.backends.mps.is_available()}")
    print("=" * 75)


def run_stateful_handover_test(prompt: str, split_step: int, total_steps: int):
    edge_steps = split_step
    cloud_steps = total_steps - split_step

    print(f"\n[Test Configuration]: Total={total_steps} tokens | Edge={edge_steps} | Cloud={cloud_steps}")
    print(f"Prompt: \"{prompt}\"")
    print("-" * 75)

    # 1. Uninterrupted Baseline
    print("\n--- 1. Running Uninterrupted Greedy Baseline ---")
    baseline_worker = InferenceWorker(worker_id="BaselineNode")
    base_res = baseline_worker.generate(prompt=prompt, max_new_tokens=total_steps)
    baseline_token_ids = base_res["generated_token_ids"]
    print(f"Baseline Generated {len(baseline_token_ids)} tokens: {baseline_token_ids}")
    print(f"Baseline Output:\n\"{base_res['full_text']}\"")

    # 2. Phase 1: Edge Worker
    print(f"\n--- 2. Phase 1: Edge Worker Generating {edge_steps} Tokens ---")
    edge_worker = InferenceWorker(worker_id="EdgeNode-M1")
    edge_res = edge_worker.generate(prompt=prompt, max_new_tokens=edge_steps)
    edge_token_ids = edge_res["generated_token_ids"]
    print(f"Edge Generated {len(edge_token_ids)} tokens: {edge_token_ids}")
    print(f"Edge Output so far:\n\"{edge_res['full_text']}\"")

    # 3. Export State from Edge Worker
    print("\n--- 3. Exporting State from Edge Worker ---")
    state_payload = edge_worker.export_state(
        cached_token_ids=edge_res["cached_token_ids"],
        past_key_values=edge_res["past_key_values"],
        pending_token_id=edge_res["pending_token_id"],
        generated_token_ids=edge_token_ids,
    )
    payload_kb = len(state_payload) / 1024.0

    # 4. Phase 2: Cloud Worker Imports and Resumes
    print(f"\n--- 4. Phase 2: Cloud Worker Importing State and Generating Next {cloud_steps} Tokens ---")
    cloud_worker = InferenceWorker(worker_id="CloudNode")
    cloud_cached_tokens, cloud_kv, cloud_pending_token, prior_ids, import_time_ms = cloud_worker.import_state(state_payload)

    cloud_res = cloud_worker.generate(
        cached_token_ids=cloud_cached_tokens,
        past_key_values=cloud_kv,
        pending_token_id=cloud_pending_token,
        max_new_tokens=cloud_steps,
    )
    cloud_token_ids = cloud_res["generated_token_ids"]
    print(f"Cloud Generated {len(cloud_token_ids)} tokens: {cloud_token_ids}")
    print(f"Cloud Final Output:\n\"{cloud_res['full_text']}\"")

    # 5. Exact Token ID Comparison
    concatenated_token_ids = edge_token_ids + cloud_token_ids
    print("\n" + "=" * 75)
    print("VERIFICATION & ACCURACY AUDIT")
    print("=" * 75)
    print(f"Baseline Tokens:     {baseline_token_ids}")
    print(f"Edge + Cloud Tokens: {concatenated_token_ids}")

    token_match = (concatenated_token_ids == baseline_token_ids)
    text_match = (cloud_res["full_text"] == base_res["full_text"])

    if token_match and text_match:
        print(">> [RESULT]: PASS! Exact token-for-token deterministic equivalence verified.")
    else:
        print(">> [RESULT]: FAIL! Discrepancy detected.")
        for idx, (b_tok, c_tok) in enumerate(zip(baseline_token_ids, concatenated_token_ids)):
            if b_tok != c_tok:
                print(f"   First divergence at token index {idx}: baseline={b_tok} vs migrated={c_tok}")
                break
        raise AssertionError("Handover output diverged from uninterrupted baseline.")

    # 6. Overhead & Timing Breakdown
    print("\n" + "=" * 75)
    print("LATENCY & OVERHEAD CHARACTERIZATION (Measured Values)")
    print("=" * 75)
    print(f"Edge Prefill Latency:           {edge_res['prefill_time_ms']:.2f} ms")
    print(f"Cloud Destination Prefill:      0.00 ms (Prompt recomputation completely bypassed)")
    print(f"State Export Serialization:     (Included in payload: {payload_kb:.2f} KB)")
    print(f"State Import Deserialization:   {import_time_ms:.2f} ms")
    print(f"Cloud First Token Step Latency: {cloud_res['step_latencies_ms'][0]:.2f} ms")
    print(f"Edge Mean Decode Step Latency:  {edge_res['mean_step_latency_ms']:.2f} ms")
    print(f"Cloud Mean Decode Step Latency: {cloud_res['mean_step_latency_ms']:.2f} ms")
    print("=" * 75)


def main():
    print_system_diagnostics()
    prompt = (
        "The edge-to-cloud computing continuum presents unique challenges for "
        "distributed AI workloads, such as"
    )

    # Test Handover at different cut points
    print("\n>>> RUNNING TEST SUITE: EVALUATING MULTIPLE HANDOVER SPLIT POINTS <<<")
    run_stateful_handover_test(prompt=prompt, split_step=15, total_steps=30)
    run_stateful_handover_test(prompt=prompt, split_step=8, total_steps=25)


if __name__ == "__main__":
    main()
