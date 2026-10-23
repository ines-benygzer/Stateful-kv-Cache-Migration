"""Reproducible Crossover Benchmark: Stateful KV Handover vs. Cold Re-prefill.

Empirically characterizes the crossover interval where stateful KV-cache transfer
outperforms cold prompt recomputation on destination workers.

Methodology & Protocol:
-----------------------
1. Model Warm-up: 3 warmup forward passes and 3 serialization/deserialization cycles.
2. Repeated Trials: Configurable trials (default 10) per sequence length.
3. Statistics: Reports Median and Interquartile Range (IQR = Q3 - Q1) for all components.
4. Component Separation:
   - Measured Compute: Prompt prefill latency, 1-step token decode latency.
   - Measured Serialization: Pure tensor quantization/conversion + packing (no diagnostic overhead).
   - Measured Deserialization: Buffer parsing + device tensor reconstruction.
   - Explicitly Simulated Network: 5G Edge Uplink (50 Mbps, 25ms RTT) and Datacenter Interconnect (1 Gbps, 2ms RTT).
5. Aggregated Metric: Labeled strictly as "Estimated TTFRT" (summed components),
   not end-to-end network measured.
6. Crossover Interval: Identified strictly within tested sequence-length intervals.
"""

import sys
import os
import time
import argparse
from typing import Dict, Any, List, Tuple
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.worker import InferenceWorker
from core.serializer import KVCacheSerializer
from core.network_sim import NetworkSimulator


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Computes median, Q1, Q3, and IQR for a list of measurements."""
    arr = np.array(values, dtype=np.float64)
    med = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = float(q3 - q1)
    return {"median": med, "q1": q1, "q3": q3, "iqr": iqr}


def run_crossover_benchmark(
    num_trials: int = 10,
    output_path: str = "figures/handover_vs_reprefill.png",
):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lengths = [128, 256, 512, 1024, 2048]

    print("=" * 115)
    print("CONTINUUM-KV: REPRODUCIBLE CROSSOVER BENCHMARK")
    print(f"Model: Qwen/Qwen2.5-0.5B-Instruct on CPU (ARM Neon Optimized) | Repeated Trials: {num_trials}")
    print("Metrics: Median and Interquartile Range (IQR = Q3 - Q1) across all trials")
    print("=" * 115)

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

    # 1. Warm-up Phase: Inference and Serialization/Deserialization
    print("\n[Warmup Phase] Warming up model inference and serializer...")
    warmup_ids = torch.randint(0, 1000, (1, 64), device=worker.device)
    for _ in range(3):
        with torch.no_grad():
            w_out = worker.model(input_ids=warmup_ids, use_cache=True)
            w_next = torch.argmax(w_out.logits[:, -1, :], dim=-1, keepdim=True)
            _ = worker.model(input_ids=w_next, past_key_values=w_out.past_key_values, use_cache=True)
        w_p16, _ = KVCacheSerializer.serialize_cache(w_out.past_key_values, mode="fp16", compute_error_metrics=False)
        _, _ = KVCacheSerializer.deserialize_cache(w_p16, target_device=worker.device)
        w_p8, _ = KVCacheSerializer.serialize_cache(w_out.past_key_values, mode="int8", compute_error_metrics=False)
        _, _ = KVCacheSerializer.deserialize_cache(w_p8, target_device=worker.device)
    print("[Warmup Phase] Completed.")

    aggregated_results = []

    for seq_len in lengths:
        print(f"\n[Benchmarking Sequence Length: {seq_len} tokens ({num_trials} trials)]...")

        multiplier = (seq_len // len(tokens)) + 2
        input_tokens = (tokens * multiplier)[:seq_len]
        input_ids = torch.tensor([input_tokens], device=worker.device)

        # Per-length warmup pass to eliminate first-time allocation jitter for this sequence length
        with torch.no_grad():
            _w_out = worker.model(input_ids=input_ids, use_cache=True)
            _w_next = torch.argmax(_w_out.logits[:, -1, :], dim=-1, keepdim=True)
            _ = worker.model(input_ids=_w_next, past_key_values=_w_out.past_key_values, use_cache=True)
        _w_p16, _ = KVCacheSerializer.serialize_cache(_w_out.past_key_values, mode="fp16", compute_error_metrics=False)
        _, _ = KVCacheSerializer.deserialize_cache(_w_p16, target_device=worker.device)
        _w_p8, _ = KVCacheSerializer.serialize_cache(_w_out.past_key_values, mode="int8", compute_error_metrics=False)
        _, _ = KVCacheSerializer.deserialize_cache(_w_p8, target_device=worker.device)

        prefill_times = []
        step_times = []
        cold_ttfrts = []

        fp16_ser_times = []
        fp16_deser_times = []
        fp16_bytes_list = []
        est_5g_fp16_ttfrts = []
        est_dc_fp16_ttfrts = []

        int8_ser_times = []
        int8_deser_times = []
        int8_bytes_list = []
        est_5g_int8_ttfrts = []
        est_dc_int8_ttfrts = []

        for trial in range(num_trials):
            # A. Measure actual prompt prefill latency
            t0 = time.perf_counter()
            with torch.no_grad():
                prefill_out = worker.model(input_ids=input_ids, use_cache=True)
                next_logits = prefill_out.logits[:, -1, :]
                pending_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            t_prefill = (time.perf_counter() - t0) * 1000.0

            # B. Measure 1 decode step latency
            t1 = time.perf_counter()
            with torch.no_grad():
                _ = worker.model(input_ids=pending_token, past_key_values=prefill_out.past_key_values, use_cache=True)
            t_step = (time.perf_counter() - t1) * 1000.0

            cold_ttfrt = t_prefill + t_step
            prefill_times.append(t_prefill)
            step_times.append(t_step)
            cold_ttfrts.append(cold_ttfrt)

            real_kv = prefill_out.past_key_values

            # C. Measure FP16 Serialization & Deserialization (pure timing, no error diagnostics)
            payload_fp16, metrics_fp16 = KVCacheSerializer.serialize_cache(
                real_kv, mode="fp16", compute_error_metrics=False
            )
            _, stats_fp16 = KVCacheSerializer.deserialize_cache(payload_fp16, target_device=worker.device)

            t_ser_fp16 = metrics_fp16["serialization_time_ms"]
            t_deser_fp16 = stats_fp16["deserialization_time_ms"]
            fp16_ser_times.append(t_ser_fp16)
            fp16_deser_times.append(t_deser_fp16)
            fp16_bytes_list.append(metrics_fp16["payload_bytes"])

            # Simulated network delays
            net_5g_fp16 = profile_5g.calculate_total_latency_ms(metrics_fp16["payload_bytes"])
            net_dc_fp16 = profile_dc.calculate_total_latency_ms(metrics_fp16["payload_bytes"])

            est_5g_fp16_ttfrt = t_ser_fp16 + net_5g_fp16 + t_deser_fp16 + t_step
            est_dc_fp16_ttfrt = t_ser_fp16 + net_dc_fp16 + t_deser_fp16 + t_step
            est_5g_fp16_ttfrts.append(est_5g_fp16_ttfrt)
            est_dc_fp16_ttfrts.append(est_dc_fp16_ttfrt)

            # D. Measure INT8 Serialization & Deserialization (pure timing, no error diagnostics)
            payload_int8, metrics_int8 = KVCacheSerializer.serialize_cache(
                real_kv, mode="int8", compute_error_metrics=False
            )
            _, stats_int8 = KVCacheSerializer.deserialize_cache(payload_int8, target_device=worker.device)

            t_ser_int8 = metrics_int8["serialization_time_ms"]
            t_deser_int8 = stats_int8["deserialization_time_ms"]
            int8_ser_times.append(t_ser_int8)
            int8_deser_times.append(t_deser_int8)
            int8_bytes_list.append(metrics_int8["payload_bytes"])

            # Simulated network delays
            net_5g_int8 = profile_5g.calculate_total_latency_ms(metrics_int8["payload_bytes"])
            net_dc_int8 = profile_dc.calculate_total_latency_ms(metrics_int8["payload_bytes"])

            est_5g_int8_ttfrt = t_ser_int8 + net_5g_int8 + t_deser_int8 + t_step
            est_dc_int8_ttfrt = t_ser_int8 + net_dc_int8 + t_deser_int8 + t_step
            est_5g_int8_ttfrts.append(est_5g_int8_ttfrt)
            est_dc_int8_ttfrts.append(est_dc_int8_ttfrt)

        row = {
            "seq_len": seq_len,
            "cold_ttfrt": compute_stats(cold_ttfrts),
            "prefill": compute_stats(prefill_times),
            "step": compute_stats(step_times),
            "fp16_mb": float(np.median(fp16_bytes_list)) / 1_000_000.0,
            "int8_mb": float(np.median(int8_bytes_list)) / 1_000_000.0,
            "fp16_ser": compute_stats(fp16_ser_times),
            "fp16_deser": compute_stats(fp16_deser_times),
            "int8_ser": compute_stats(int8_ser_times),
            "int8_deser": compute_stats(int8_deser_times),
            "est_5g_fp16": compute_stats(est_5g_fp16_ttfrts),
            "est_5g_int8": compute_stats(est_5g_int8_ttfrts),
            "est_dc_fp16": compute_stats(est_dc_fp16_ttfrts),
            "est_dc_int8": compute_stats(est_dc_int8_ttfrts),
        }
        aggregated_results.append(row)
        print(
            f"  Cold TTFRT: {row['cold_ttfrt']['median']:.1f} ms (IQR {row['cold_ttfrt']['iqr']:.1f} ms) | "
            f"5G INT8: {row['est_5g_int8']['median']:.1f} ms (IQR {row['est_5g_int8']['iqr']:.1f} ms) | "
            f"DC INT8: {row['est_dc_int8']['median']:.1f} ms (IQR {row['est_dc_int8']['iqr']:.1f} ms)"
        )

    # 2. Print Comprehensive Aggregated Table
    print("\n" + "=" * 145)
    print("AGGREGATED BENCHMARK AUDIT: MEDIAN & IQR (ms) [Simulated Network + Measured Compute/Serialization]")
    print("=" * 145)
    header = (
        f"{'Seq Len':<8} | "
        f"{'Cold Re-prefill Median (IQR)':<28} | "
        f"{'5G Mig FP16 Median (IQR)':<25} | "
        f"{'5G Mig INT8 Median (IQR)':<25} | "
        f"{'DC Mig FP16 Median (IQR)':<25} | "
        f"{'DC Mig INT8 Median (IQR)':<25}"
    )
    print(header)
    print("-" * 145)
    for r in aggregated_results:
        c_str = f"{r['cold_ttfrt']['median']:>7.1f} ms (IQR {r['cold_ttfrt']['iqr']:>5.1f} ms)"
        g5_fp_str = f"{r['est_5g_fp16']['median']:>7.1f} ms (IQR {r['est_5g_fp16']['iqr']:>5.1f} ms)"
        g5_i8_str = f"{r['est_5g_int8']['median']:>7.1f} ms (IQR {r['est_5g_int8']['iqr']:>5.1f} ms)"
        dc_fp_str = f"{r['est_dc_fp16']['median']:>7.1f} ms (IQR {r['est_dc_fp16']['iqr']:>5.1f} ms)"
        dc_i8_str = f"{r['est_dc_int8']['median']:>7.1f} ms (IQR {r['est_dc_int8']['iqr']:>5.1f} ms)"
        print(f"{r['seq_len']:<8} | {c_str:<28} | {g5_fp_str:<25} | {g5_i8_str:<25} | {dc_fp_str:<25} | {dc_i8_str:<25}")
    print("=" * 145)

    # 3. Component Breakdown Table
    print("\n" + "=" * 125)
    print("MEASURED COMPONENT BREAKDOWN (Median ms, Serialization Excludes Error Diagnostics)")
    print("=" * 125)
    header_comp = (
        f"{'Seq Len':<8} | "
        f"{'Prefill':<12} | "
        f"{'1-Token Step':<14} | "
        f"{'FP16 Ser':<12} | "
        f"{'FP16 Deser':<12} | "
        f"{'INT8 Ser':<12} | "
        f"{'INT8 Deser':<12} | "
        f"{'FP16 Size':<10} | "
        f"{'INT8 Size':<10}"
    )
    print(header_comp)
    print("-" * 125)
    for r in aggregated_results:
        print(
            f"{r['seq_len']:<8} | "
            f"{r['prefill']['median']:>8.1f} ms  | "
            f"{r['step']['median']:>10.1f} ms  | "
            f"{r['fp16_ser']['median']:>8.1f} ms  | "
            f"{r['fp16_deser']['median']:>8.1f} ms  | "
            f"{r['int8_ser']['median']:>8.1f} ms  | "
            f"{r['int8_deser']['median']:>8.1f} ms  | "
            f"{r['fp16_mb']:>7.3f} MB | "
            f"{r['int8_mb']:>7.3f} MB"
        )
    print("=" * 125)

    # 4. Measured Crossover Interval Analysis (strictly based on tested boundaries)
    print("\n" + "=" * 115)
    print("MEASURED CROSSOVER INTERVAL ANALYSIS (Tested Intervals: [128, 256], [256, 512], [512, 1024], [1024, 2048])")
    print("=" * 115)

  
    def find_crossover_interval(metric_name: str, label: str):
        # Positive difference means migration is faster than cold re-prefill.
        diffs = [
            r["cold_ttfrt"]["median"] - r[metric_name]["median"]
            for r in aggregated_results
        ]

        seq_lens = [r["seq_len"] for r in aggregated_results]

        if all(d > 0 for d in diffs):
            print(
                f">> {label} is faster at every tested sequence length "
                f"[{seq_lens[0]}, {seq_lens[-1]}] tokens.\n"
            )
            return

        if all(d < 0 for d in diffs):
            print(
                f">> {label} is slower at every tested sequence length "
                f"[{seq_lens[0]}, {seq_lens[-1]}] tokens.\n"
            )
            return

        if all(d == 0 for d in diffs):
            print(
                f">> {label} ties Cold Re-prefill at every tested sequence length.\n"
            )
            return

        # Report every adjacent interval where the relative performance changes.
        found_transition = False

        for i in range(len(diffs) - 1):
            d1, d2 = diffs[i], diffs[i + 1]

            if d1 * d2 < 0:
                before = "faster" if d1 > 0 else "slower"
                after = "faster" if d2 > 0 else "slower"

                print(
                    f">> {label} changes from {before} to {after} relative "
                    f"to Cold Re-prefill between {seq_lens[i]} and "
                    f"{seq_lens[i + 1]} tokens.\n"
                    f"   Cold-minus-migration differences: "
                    f"{d1:.1f} ms -> {d2:.1f} ms.\n"
                    f"   More measurements are needed to locate the crossover.\n"
                )
                found_transition = True

            elif d1 == 0 or d2 == 0:
                idx = i if d1 == 0 else i + 1
                print(
                    f">> {label} ties Cold Re-prefill at "
                    f"{seq_lens[idx]} tokens.\n"
                )
                found_transition = True

        if not found_transition:
            print(
                f">> {label} has mixed results across tested lengths, "
                "but no strict adjacent sign change was detected. "
                "Inspect the medians for ties or non-monotonic behavior.\n"
            )


    find_crossover_interval("est_5g_fp16", "5G Edge Uplink (FP16 Raw)")
    find_crossover_interval("est_5g_int8", "5G Edge Uplink (ContinuumKV INT8)")
    find_crossover_interval("est_dc_fp16", "Datacenter Interconnect (FP16 Raw)")
    find_crossover_interval("est_dc_int8", "Datacenter Interconnect (ContinuumKV INT8)")
    print("=" * 115)

    # 5. Generate Aggregated Publication Figure with Error Bands
    print(f"\nRegenerating publication figure with aggregated trial medians and IQR: {output_path}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    seq_lens = [r["seq_len"] for r in aggregated_results]

    cold_med = [r["cold_ttfrt"]["median"] for r in aggregated_results]
    cold_q1 = [r["cold_ttfrt"]["q1"] for r in aggregated_results]
    cold_q3 = [r["cold_ttfrt"]["q3"] for r in aggregated_results]

    # Subplot 1: 5G Edge Uplink (Simulated 50 Mbps, 25 ms RTT)
    g5_fp16_med = [r["est_5g_fp16"]["median"] for r in aggregated_results]
    g5_fp16_q1 = [r["est_5g_fp16"]["q1"] for r in aggregated_results]
    g5_fp16_q3 = [r["est_5g_fp16"]["q3"] for r in aggregated_results]

    g5_int8_med = [r["est_5g_int8"]["median"] for r in aggregated_results]
    g5_int8_q1 = [r["est_5g_int8"]["q1"] for r in aggregated_results]
    g5_int8_q3 = [r["est_5g_int8"]["q3"] for r in aggregated_results]

    ax1.plot(seq_lens, cold_med, 'r--o', linewidth=2.2, markersize=7, label='Cold Re-prefill (Compute Bound)')
    ax1.fill_between(seq_lens, cold_q1, cold_q3, color='red', alpha=0.12)

    ax1.plot(seq_lens, g5_fp16_med, 'b-s', linewidth=2.0, markersize=6, label='Stateful Migration (FP16 Raw)')
    ax1.fill_between(seq_lens, g5_fp16_q1, g5_fp16_q3, color='blue', alpha=0.10)

    ax1.plot(seq_lens, g5_int8_med, 'g-^', linewidth=2.4, markersize=7, label='ContinuumKV (Adaptive INT8)')
    ax1.fill_between(seq_lens, g5_int8_q1, g5_int8_q3, color='green', alpha=0.12)

    ax1.set_title('5G Edge Uplink (Simulated 50 Mbps, 25ms RTT)', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel('Context Sequence Length (Tokens)\n[Shaded bands indicate IQR (Q1-Q3)]', fontsize=10, labelpad=8)
    ax1.set_ylabel('Estimated TTFRT (ms) [Simulated Net + Measured Compute]', fontsize=10, labelpad=8)
    ax1.grid(True, linestyle='--', alpha=0.55)
    ax1.legend(fontsize=9.5, loc='upper left')

    # Subplot 2: Datacenter Interconnect (Simulated 1 Gbps, 2 ms RTT)
    dc_fp16_med = [r["est_dc_fp16"]["median"] for r in aggregated_results]
    dc_fp16_q1 = [r["est_dc_fp16"]["q1"] for r in aggregated_results]
    dc_fp16_q3 = [r["est_dc_fp16"]["q3"] for r in aggregated_results]

    dc_int8_med = [r["est_dc_int8"]["median"] for r in aggregated_results]
    dc_int8_q1 = [r["est_dc_int8"]["q1"] for r in aggregated_results]
    dc_int8_q3 = [r["est_dc_int8"]["q3"] for r in aggregated_results]

    ax2.plot(seq_lens, cold_med, 'r--o', linewidth=2.2, markersize=7, label='Cold Re-prefill (Compute Bound)')
    ax2.fill_between(seq_lens, cold_q1, cold_q3, color='red', alpha=0.12)

    ax2.plot(seq_lens, dc_fp16_med, 'b-s', linewidth=2.0, markersize=6, label='Stateful Migration (FP16 Raw)')
    ax2.fill_between(seq_lens, dc_fp16_q1, dc_fp16_q3, color='blue', alpha=0.10)

    ax2.plot(seq_lens, dc_int8_med, 'g-^', linewidth=2.4, markersize=7, label='ContinuumKV (Adaptive INT8)')
    ax2.fill_between(seq_lens, dc_int8_q1, dc_int8_q3, color='green', alpha=0.12)

    ax2.set_title('Datacenter Interconnect (Simulated 1 Gbps, 2ms RTT)', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel('Context Sequence Length (Tokens)\n[Shaded bands indicate IQR (Q1-Q3)]', fontsize=10, labelpad=8)
    ax2.set_ylabel('Estimated TTFRT (ms) [Simulated Net + Measured Compute]', fontsize=10, labelpad=8)
    ax2.grid(True, linestyle='--', alpha=0.55)
    ax2.legend(fontsize=9.5, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f">> Successfully regenerated figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run reproducible crossover benchmark.")
    parser.add_argument("--trials", type=int, default=10, help="Number of repeated trials per sequence length (default: 10)")
    parser.add_argument("--output", type=str, default="figures/handover_vs_reprefill.png", help="Output figure path")
    args = parser.parse_args()

    run_crossover_benchmark(num_trials=args.trials, output_path=args.output)


if __name__ == "__main__":
    main()
