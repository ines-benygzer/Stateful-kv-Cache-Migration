"""Benchmark Suite: Stateful KV Handover vs. Cold Re-prefill Crossover.

Empirically characterizes the crossover point (L*) where stateful KV-cache transfer
outperforms cold prompt recomputation on destination workers.
Evaluates:
- Actual prompt prefill latency across context lengths [128, 256, 512, 1024, 2048].
- Measured serialization & deserialization overhead (FP16 vs. INT8).
- Network transmission delay across continuum profiles (5G Edge Uplink vs. Datacenter Interconnect).
- Generates publication-grade figure: figures/handover_vs_reprefill.png.
"""

import sys
import os
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.worker import InferenceWorker
from core.serializer import KVCacheSerializer
from core.network_sim import NetworkSimulator


def run_crossover_benchmark(output_path: str = "figures/handover_vs_reprefill.png"):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lengths = [128, 256, 512, 1024, 2048]

    print("=" * 115)
    print("CONTINUUM-KV: STATEFUL HANDOVER VS. COLD RE-PREFILL CROSSOVER BENCHMARK")
    print("Model: Qwen/Qwen2.5-0.5B-Instruct on Apple Silicon M1 (ARM Neon Optimized CPU)")
    print("=" * 115)

    # Use CPU for deterministic, stable compute execution without MPS shader jitter
    worker = InferenceWorker(worker_id="BenchWorker", device="cpu")
    worker.load_model()
    sim = NetworkSimulator()

    profile_5g = sim.get_profile("5g_edge_uplink")
    profile_dc = sim.get_profile("datacenter_interconnect")

    base_text = (
        "The computing continuum spans distributed edge devices, 5G base stations, "
        "and centralized hyperscaler clouds. Optimizing stateful artificial intelligence "
        "workloads across this continuum requires understanding memory, energy, and network limits. "
    )
    tokens = worker.tokenizer.encode(base_text)

    # Warmup pass
    print("\nWarming up model kernels...")
    warmup_ids = torch.randint(0, 1000, (1, 64), device=worker.device)
    for _ in range(3):
        with torch.no_grad():
            _ = worker.model(input_ids=warmup_ids, use_cache=True)

    records = []

    for seq_len in lengths:
        print(f"\n[Evaluating Sequence Length: {seq_len} tokens]...")

        # 1. Prepare exact input sequence
        multiplier = (seq_len // len(tokens)) + 2
        input_tokens = (tokens * multiplier)[:seq_len]
        input_ids = torch.tensor([input_tokens], device=worker.device)

        # 2. Measure actual prefill latency (Cold Re-prefill on destination)
        t_prefill_start = time.perf_counter()
        with torch.no_grad():
            prefill_out = worker.model(input_ids=input_ids, use_cache=True)
            next_logits = prefill_out.logits[:, -1, :]
            pending_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        prefill_latency_ms = (time.perf_counter() - t_prefill_start) * 1000.0

        # Measure 1 decode step latency
        t_step_start = time.perf_counter()
        with torch.no_grad():
            _ = worker.model(input_ids=pending_token, past_key_values=prefill_out.past_key_values, use_cache=True)
        step_latency_ms = (time.perf_counter() - t_step_start) * 1000.0

        real_kv = prefill_out.past_key_values

        # 3. Measure actual FP16 Serialization & Deserialization
        payload_fp16, metrics_fp16 = KVCacheSerializer.serialize_cache(real_kv, mode="fp16")
        _, stats_fp16 = KVCacheSerializer.deserialize_cache(payload_fp16, target_device=worker.device)

        # 4. Measure actual INT8 Serialization & Deserialization
        payload_int8, metrics_int8 = KVCacheSerializer.serialize_cache(real_kv, mode="int8")
        _, stats_int8 = KVCacheSerializer.deserialize_cache(payload_int8, target_device=worker.device)

        # 5. Network delays across profiles
        # Profile 1: 5G Edge Uplink (50 Mbps, 25 ms)
        net_5g_fp16 = profile_5g.calculate_total_latency_ms(metrics_fp16["payload_bytes"])
        net_5g_int8 = profile_5g.calculate_total_latency_ms(metrics_int8["payload_bytes"])

        # Profile 2: Datacenter Direct Interconnect (1 Gbps, 2 ms)
        net_dc_fp16 = profile_dc.calculate_total_latency_ms(metrics_fp16["payload_bytes"])
        net_dc_int8 = profile_dc.calculate_total_latency_ms(metrics_int8["payload_bytes"])

        # Total Time to First Resumed Token (TTFRT)
        cold_ttfrt = prefill_latency_ms + step_latency_ms
        mig_5g_fp16_ttfrt = metrics_fp16["serialization_time_ms"] + net_5g_fp16 + stats_fp16["deserialization_time_ms"] + step_latency_ms
        mig_5g_int8_ttfrt = metrics_int8["serialization_time_ms"] + net_5g_int8 + stats_int8["deserialization_time_ms"] + step_latency_ms

        mig_dc_fp16_ttfrt = metrics_fp16["serialization_time_ms"] + net_dc_fp16 + stats_fp16["deserialization_time_ms"] + step_latency_ms
        mig_dc_int8_ttfrt = metrics_int8["serialization_time_ms"] + net_dc_int8 + stats_int8["deserialization_time_ms"] + step_latency_ms

        rec = {
            "seq_len": seq_len,
            "prefill_ms": prefill_latency_ms,
            "step_ms": step_latency_ms,
            "cold_ttfrt": cold_ttfrt,
            "mig_5g_fp16_ttfrt": mig_5g_fp16_ttfrt,
            "mig_5g_int8_ttfrt": mig_5g_int8_ttfrt,
            "mig_dc_fp16_ttfrt": mig_dc_fp16_ttfrt,
            "mig_dc_int8_ttfrt": mig_dc_int8_ttfrt,
            "fp16_mb": metrics_fp16["payload_mb"],
            "int8_mb": metrics_int8["payload_mb"],
        }
        records.append(rec)
        print(f"  Cold Prefill: {cold_ttfrt:.1f} ms | 5G INT8: {mig_5g_int8_ttfrt:.1f} ms | DC INT8: {mig_dc_int8_ttfrt:.1f} ms")

    # Print Summary Table
    print("\n" + "=" * 115)
    print("BENCHMARK AUDIT: TIME TO FIRST RESUMED TOKEN (TTFRT in milliseconds)")
    print("=" * 115)
    header = f"{'Seq Len':<8} | {'Cold Re-prefill':<16} | {'5G Mig (FP16)':<15} | {'5G Mig (INT8)':<15} | {'DC Mig (FP16)':<15} | {'DC Mig (INT8)':<15}"
    print(header)
    print("-" * 115)
    for r in records:
        print(
            f"{r['seq_len']:<8} | "
            f"{r['cold_ttfrt']:>13.2f} ms | "
            f"{r['mig_5g_fp16_ttfrt']:>12.2f} ms | "
            f"{r['mig_5g_int8_ttfrt']:>12.2f} ms | "
            f"{r['mig_dc_fp16_ttfrt']:>12.2f} ms | "
            f"{r['mig_dc_int8_ttfrt']:>12.2f} ms"
        )
    print("=" * 115)

    # Generate Publication Figure
    print(f"\nGenerating publication-quality figure: {output_path}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    seq_lens = [r["seq_len"] for r in records]
    cold_vals = [r["cold_ttfrt"] for r in records]
    
    # Subplot 1: 5G Edge Uplink (50 Mbps, 25 ms)
    mig_5g_fp16_vals = [r["mig_5g_fp16_ttfrt"] for r in records]
    mig_5g_int8_vals = [r["mig_5g_int8_ttfrt"] for r in records]

    ax1.plot(seq_lens, cold_vals, 'r--o', linewidth=2.4, markersize=8, label='Cold Re-prefill (Compute Bound)')
    ax1.plot(seq_lens, mig_5g_fp16_vals, 'b-s', linewidth=2.0, markersize=6, label='Stateful Migration (FP16 Raw)')
    ax1.plot(seq_lens, mig_5g_int8_vals, 'g-^', linewidth=2.4, markersize=8, label='ContinuumKV (Adaptive INT8)')

    ax1.set_title('5G Edge Uplink (50 Mbps, 25ms RTT)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Context Sequence Length (Tokens)', fontsize=11, labelpad=8)
    ax1.set_ylabel('Time to First Resumed Token (ms)', fontsize=11, labelpad=8)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10, loc='upper left')

    # Subplot 2: Datacenter / High-Speed Continuum (1 Gbps, 2 ms)
    mig_dc_fp16_vals = [r["mig_dc_fp16_ttfrt"] for r in records]
    mig_dc_int8_vals = [r["mig_dc_int8_ttfrt"] for r in records]

    ax2.plot(seq_lens, cold_vals, 'r--o', linewidth=2.4, markersize=8, label='Cold Re-prefill (Compute Bound)')
    ax2.plot(seq_lens, mig_dc_fp16_vals, 'b-s', linewidth=2.0, markersize=6, label='Stateful Migration (FP16 Raw)')
    ax2.plot(seq_lens, mig_dc_int8_vals, 'g-^', linewidth=2.4, markersize=8, label='ContinuumKV (Adaptive INT8)')

    ax2.set_title('High-Speed Continuum / Direct Interconnect (1 Gbps)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Context Sequence Length (Tokens)', fontsize=11, labelpad=8)
    ax2.set_ylabel('Time to First Resumed Token (ms)', fontsize=11, labelpad=8)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f">> Successfully generated and saved: {output_path}")


if __name__ == "__main__":
    run_crossover_benchmark()
