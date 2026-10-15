"""Benchmark Suite for KV-Cache Serialization and Dynamic INT8 Quantization.

Measures actual model cache tensors across sequence lengths: [128, 256, 512, 1024, 2048].
Reports:
- Actual serialized payload size (bytes and decimal MB)
- Raw tensor data size vs metadata overhead
- Serialization & Deserialization latency (ms)
- Maximum and Mean Absolute Reconstruction Error
- Compression ratio
All results use real cache tensors from Qwen/Qwen2.5-0.5B-Instruct running on Apple Silicon.
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.serializer import KVCacheSerializer
from core.worker import InferenceWorker


def generate_real_kv_cache(worker: InferenceWorker, target_seq_len: int):
    """Generates a real KV-cache of exact sequence length using the model."""
    base_text = (
        "The computing continuum spans distributed edge devices, 5G base stations, "
        "and centralized hyperscaler clouds. Optimizing stateful artificial intelligence "
        "workloads across this continuum requires understanding memory, energy, and network limits. "
    )
    tokens = worker.tokenizer.encode(base_text)
    
    # Repeat tokens to exceed target length, then slice exact length
    multiplier = (target_seq_len // len(tokens)) + 2
    full_tokens = (tokens * multiplier)[:target_seq_len]
    input_ids = torch.tensor([full_tokens], device=worker.device)

    with torch.no_grad():
        outputs = worker.model(input_ids=input_ids, use_cache=True)
        
    return outputs.past_key_values


def run_benchmark():
    lengths = [128, 256, 512, 1024, 2048]
    print("=" * 115)
    print("CONTINUUM-KV: REAL-CACHE SERIALIZATION & QUANTIZATION BENCHMARK")
    print("Model: Qwen/Qwen2.5-0.5B-Instruct (24 Layers, 2 KV Heads, Head Dim 64)")
    print("All measurements use 100% REAL model cache tensors on Apple Silicon M1 (MPS). No estimations.")
    print("=" * 115)

    worker = InferenceWorker(worker_id="BenchWorker")
    worker.load_model()

    results = []

    for seq_len in lengths:
        print(f"\n[Benchmarking Sequence Length: {seq_len} tokens]...")
        t_gen_start = time.perf_counter()
        real_kv = generate_real_kv_cache(worker, seq_len)
        gen_time_sec = time.perf_counter() - t_gen_start
        print(f"  Real cache generated in {gen_time_sec:.2f}s")

        # 1. FP16 Benchmark
        payload_fp16, metrics_fp16 = KVCacheSerializer.serialize_cache(real_kv, mode="fp16")
        _, stats_fp16 = KVCacheSerializer.deserialize_cache(payload_fp16, target_device=worker.device)

        # 2. INT8 Benchmark
        payload_int8, metrics_int8 = KVCacheSerializer.serialize_cache(real_kv, mode="int8")
        _, stats_int8 = KVCacheSerializer.deserialize_cache(payload_int8, target_device=worker.device)

        compression_ratio = metrics_fp16["payload_bytes"] / metrics_int8["payload_bytes"]

        row = {
            "seq_len": seq_len,
            "fp16_bytes": metrics_fp16["payload_bytes"],
            "fp16_mb": metrics_fp16["payload_mb"],
            "fp16_raw_mb": metrics_fp16["raw_tensor_mb"],
            "fp16_meta_kb": metrics_fp16["metadata_overhead_bytes"] / 1000.0,
            "fp16_ser_ms": metrics_fp16["serialization_time_ms"],
            "fp16_des_ms": stats_fp16["deserialization_time_ms"],
            "int8_bytes": metrics_int8["payload_bytes"],
            "int8_mb": metrics_int8["payload_mb"],
            "int8_raw_mb": metrics_int8["raw_tensor_mb"],
            "int8_meta_kb": metrics_int8["metadata_overhead_bytes"] / 1000.0,
            "int8_ser_ms": metrics_int8["serialization_time_ms"],
            "int8_des_ms": stats_int8["deserialization_time_ms"],
            "compression_ratio": compression_ratio,
            "int8_max_err": metrics_int8["max_abs_error"],
            "int8_mean_err": metrics_int8["mean_abs_error"],
        }
        results.append(row)

    print("\n" + "=" * 115)
    print("BENCHMARK SUMMARY: SIZE & METADATA OVERHEAD (Decimal MB = Bytes / 1,000,000)")
    print("=" * 115)
    header1 = f"{'Seq Len':<8} | {'FP16 Total':<11} | {'FP16 Raw':<10} | {'FP16 Meta':<10} | {'INT8 Total':<11} | {'INT8 Raw':<10} | {'INT8 Meta':<10} | {'Compression':<11}"
    print(header1)
    print("-" * 115)
    for r in results:
        print(
            f"{r['seq_len']:<8} | {r['fp16_mb']:>7.3f} MB | {r['fp16_raw_mb']:>6.3f} MB | {r['fp16_meta_kb']:>6.2f} KB | "
            f"{r['int8_mb']:>7.3f} MB | {r['int8_raw_mb']:>6.3f} MB | {r['int8_meta_kb']:>6.2f} KB | {r['compression_ratio']:>9.2f}x"
        )

    print("\n" + "=" * 115)
    print("BENCHMARK SUMMARY: LATENCY & NUMERICAL ERROR AUDIT")
    print("=" * 115)
    header2 = f"{'Seq Len':<8} | {'FP16 Ser':<10} | {'FP16 Deser':<10} | {'INT8 Ser':<10} | {'INT8 Deser':<10} | {'INT8 Max Err':<13} | {'INT8 Mean Err':<13}"
    print(header2)
    print("-" * 115)
    for r in results:
        print(
            f"{r['seq_len']:<8} | {r['fp16_ser_ms']:>7.2f} ms | {r['fp16_des_ms']:>7.2f} ms | "
            f"{r['int8_ser_ms']:>7.2f} ms | {r['int8_des_ms']:>7.2f} ms | "
            f"{r['int8_max_err']:>13.6f} | {r['int8_mean_err']:>13.6f}"
        )
    print("=" * 115)


if __name__ == "__main__":
    run_benchmark()
