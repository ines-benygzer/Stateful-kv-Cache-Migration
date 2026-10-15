"""Verification & Characterization Benchmark for Continuum Network Link Emulator.

Compares transmission latency across continuum profiles:
1. 5G Edge Uplink (50 Mbps, 25 ms RTT)
2. Constrained WAN / Fog (15 Mbps, 60 ms RTT)
3. Datacenter Interconnect (1 Gbps, 2 ms RTT)

Evaluates actual measured FP16 and INT8 KV-cache payloads from Step 2
for sequence lengths: [128, 256, 512, 1024, 2048] tokens.
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.network_sim import NetworkSimulator, DEFAULT_PROFILES


# Measured payload sizes from Step 2 (Qwen2.5-0.5B-Instruct)
MEASURED_PAYLOADS = [
    {"seq_len": 128,  "fp16_bytes": 1_587_000, "int8_bytes": 803_000},
    {"seq_len": 256,  "fp16_bytes": 3_160_000, "int8_bytes": 1_590_000},
    {"seq_len": 512,  "fp16_bytes": 6_306_000, "int8_bytes": 3_163_000},
    {"seq_len": 1024, "fp16_bytes": 12_597_000, "int8_bytes": 6_308_000},
    {"seq_len": 2048, "fp16_bytes": 25_180_000, "int8_bytes": 12_600_000},
]


def run_network_characterization():
    sim = NetworkSimulator()
    profiles_to_test = [
        "5g_edge_uplink",
        "constrained_edge_wan",
        "datacenter_interconnect",
    ]

    print("=" * 115)
    print("CONTINUUM NETWORK LINK EMULATOR: LATENCY CHARACTERIZATION")
    print("Model: Latency = RTT + (Payload Size (MB) / Bandwidth (MB/s)) * 1000")
    print("=" * 115)

    for profile_name in profiles_to_test:
        profile = sim.get_profile(profile_name)
        print(f"\n>>> PROFILE: {profile.description}")
        print(f"    Bandwidth: {profile.bandwidth_mbps:.1f} Mbps ({profile.bandwidth_mb_per_sec:.2f} MB/s) | RTT: {profile.rtt_ms:.1f} ms")
        print("-" * 115)
        header = f"{'Seq Len':<8} | {'FP16 Size':<10} | {'FP16 Trans':<11} | {'FP16 Total':<11} | {'INT8 Size':<10} | {'INT8 Trans':<11} | {'INT8 Total':<11} | {'Time Saved':<11} | {'Speedup':<8}"
        print(header)
        print("-" * 115)

        for item in MEASURED_PAYLOADS:
            seq_len = item["seq_len"]
            fp16_est = sim.estimate_latency(item["fp16_bytes"], profile_name=profile_name)
            int8_est = sim.estimate_latency(item["int8_bytes"], profile_name=profile_name)

            time_saved_ms = fp16_est["total_latency_ms"] - int8_est["total_latency_ms"]
            speedup = fp16_est["total_latency_ms"] / int8_est["total_latency_ms"]

            print(
                f"{seq_len:<8} | "
                f"{fp16_est['payload_mb']:>6.3f} MB  | "
                f"{fp16_est['transmission_delay_ms']:>8.1f} ms | "
                f"{fp16_est['total_latency_ms']:>8.1f} ms | "
                f"{int8_est['payload_mb']:>6.3f} MB  | "
                f"{int8_est['transmission_delay_ms']:>8.1f} ms | "
                f"{int8_est['total_latency_ms']:>8.1f} ms | "
                f"{time_saved_ms:>8.1f} ms | "
                f"{speedup:>6.2f}x"
            )

    print("\n" + "=" * 115)


async def test_async_transfer():
    print("\nTesting async transfer emulator with real payload sleep...")
    sim = NetworkSimulator()
    dummy_payload = b"X" * 100_000  # 100 KB
    
    t0 = asyncio.get_event_loop().time()
    _, metrics = await sim.transfer_async(dummy_payload, profile_name="datacenter_interconnect")
    elapsed_ms = (asyncio.get_event_loop().time() - t0) * 1000.0

    print(f"Async test: Simulated 100 KB over datacenter_interconnect in {elapsed_ms:.2f} ms (Expected: {metrics['total_latency_ms']:.2f} ms)")
    assert abs(elapsed_ms - metrics["total_latency_ms"]) < 20.0, "Async transfer timing divergence"
    print(">> Async transmission simulation verified.\n")


def main():
    run_network_characterization()
    asyncio.run(test_async_transfer())


if __name__ == "__main__":
    main()
