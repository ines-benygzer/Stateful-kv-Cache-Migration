"""Continuum Network Link Emulator and Latency Model.

Simulates transmission latency, bandwidth limits, and round-trip times (RTT)
across the computing continuum:
- 5G Edge Uplink (e.g., Telecom Private 5G)
- Constrained WAN / Fog (e.g., IoT Gateway over cellular/DSL)
- Datacenter Interconnect (e.g., Metro Optical Fabric)

Mathematical Delay Model:
-------------------------
Bandwidth (MB/s) = Bandwidth (Mbps) / 8.0
Transmission Delay (ms) = (Payload Size (MB) / Bandwidth (MB/s)) * 1000
Total Latency (ms) = RTT (ms) + Transmission Delay (ms)
"""

from typing import Dict, Any, Optional, Union, Tuple
import json
import os
import random
import asyncio
from dataclasses import dataclass, asdict


@dataclass
class NetworkProfile:
    name: str
    bandwidth_mbps: float
    rtt_ms: float
    jitter_ms: float = 0.0
    description: str = ""

    @property
    def bandwidth_mb_per_sec(self) -> float:
        """Bandwidth in decimal Megabytes per second (MB/s)."""
        return self.bandwidth_mbps / 8.0

    def calculate_transmission_delay_ms(self, payload_bytes: int) -> float:
        """Calculates transmission delay in milliseconds based on payload size."""
        payload_mb = payload_bytes / 1_000_000.0
        # transmission_time_sec = payload_mb / bandwidth_mb_per_sec
        return (payload_mb / self.bandwidth_mb_per_sec) * 1000.0

    def calculate_total_latency_ms(self, payload_bytes: int, with_jitter: bool = False) -> float:
        """Calculates total latency (RTT + transmission delay + optional jitter)."""
        trans_delay = self.calculate_transmission_delay_ms(payload_bytes)
        jitter = random.gauss(0, self.jitter_ms) if with_jitter and self.jitter_ms > 0 else 0.0
        rtt = max(0.1, self.rtt_ms + jitter)
        return rtt + trans_delay


# Standard Continuum Presets
DEFAULT_PROFILES: Dict[str, NetworkProfile] = {
    "5g_edge_uplink": NetworkProfile(
        name="5g_edge_uplink",
        bandwidth_mbps=50.0,
        rtt_ms=25.0,
        jitter_ms=5.0,
        description="5G Edge Uplink (Telekom 5G Private Edge)",
    ),
    "constrained_edge_wan": NetworkProfile(
        name="constrained_edge_wan",
        bandwidth_mbps=15.0,
        rtt_ms=60.0,
        jitter_ms=10.0,
        description="Constrained WAN / Fog (Remote Edge Gateway to Cloud)",
    ),
    "datacenter_interconnect": NetworkProfile(
        name="datacenter_interconnect",
        bandwidth_mbps=1000.0,
        rtt_ms=2.0,
        jitter_ms=0.5,
        description="Datacenter Interconnect (Metro Optical Fabric)",
    ),
}


class NetworkSimulator:
    """Simulates continuum network transfers across heterogeneous link profiles."""

    def __init__(self, default_profile: str = "5g_edge_uplink", custom_profiles_path: Optional[str] = None):
        self.profiles: Dict[str, NetworkProfile] = dict(DEFAULT_PROFILES)

        # Load from json if available
        if custom_profiles_path and os.path.exists(custom_profiles_path):
            self.load_profiles(custom_profiles_path)
        else:
            default_json_path = os.path.join(
                os.path.dirname(__file__), "..", "traces", "network_profiles.json"
            )
            if os.path.exists(default_json_path):
                self.load_profiles(default_json_path)

        if default_profile not in self.profiles:
            raise KeyError(f"Default profile '{default_profile}' not in available profiles: {list(self.profiles.keys())}")
        self.default_profile = default_profile

    def load_profiles(self, filepath: str):
        """Loads network profiles from a JSON configuration file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        for key, val in data.items():
            self.profiles[key] = NetworkProfile(
                name=key,
                bandwidth_mbps=float(val.get("bandwidth_mbps", 50.0)),
                rtt_ms=float(val.get("rtt_ms", 25.0)),
                jitter_ms=float(val.get("jitter_ms", 0.0)),
                description=val.get("description", ""),
            )

    def get_profile(self, profile_name: Optional[str] = None) -> NetworkProfile:
        name = profile_name or self.default_profile
        if name not in self.profiles:
            raise KeyError(f"Unknown network profile: {name}. Available: {list(self.profiles.keys())}")
        return self.profiles[name]

    def estimate_latency(
        self,
        payload: Union[bytes, int],
        profile_name: Optional[str] = None,
        with_jitter: bool = False,
    ) -> Dict[str, Any]:
        """Calculates precise transmission metrics for a given payload without blocking."""
        payload_bytes = len(payload) if isinstance(payload, bytes) else int(payload)
        profile = self.get_profile(profile_name)

        payload_mb = payload_bytes / 1_000_000.0
        trans_delay_ms = profile.calculate_transmission_delay_ms(payload_bytes)
        total_latency_ms = profile.calculate_total_latency_ms(payload_bytes, with_jitter=with_jitter)

        return {
            "profile_name": profile.name,
            "description": profile.description,
            "bandwidth_mbps": profile.bandwidth_mbps,
            "bandwidth_mb_per_sec": profile.bandwidth_mb_per_sec,
            "rtt_ms": profile.rtt_ms,
            "payload_bytes": payload_bytes,
            "payload_mb": payload_mb,
            "transmission_delay_ms": trans_delay_ms,
            "total_latency_ms": total_latency_ms,
        }

    async def transfer_async(
        self,
        payload: bytes,
        profile_name: Optional[str] = None,
        with_jitter: bool = False,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Asynchronously simulates transmission delay by awaiting the calculated latency."""
        metrics = self.estimate_latency(payload, profile_name=profile_name, with_jitter=with_jitter)
        delay_sec = metrics["total_latency_ms"] / 1000.0
        await asyncio.sleep(delay_sec)
        return payload, metrics
