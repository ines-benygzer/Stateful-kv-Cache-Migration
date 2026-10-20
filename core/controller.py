"""ContinuumKV: Policy Engine and Time-Budget SLA Controller.

Formulates and evaluates the decision boundary for stateful LLM session mobility
across heterogeneous edge-to-cloud continuums.

Key Principles:
1. Time-Budgeted SLA Enforcement:
   Migration downtime (Time-to-First-Resumed-Token, TTFRT) must not exceed
   the application SLA budget:
     TTFRT(L, B, RTT, mode) <= SLA_budget_ms

2. Net-Energy & Carbon Hypothesis:
   Migrating state to a cleaner or more efficient node yields a net carbon benefit
   only if the saved compute emissions exceed the cost of state transfer:
     Delta Carbon = Carbon(Stay Local) - Carbon(Migrate) > 0
   where:
     Carbon(Stay Local) = E_edge_decode * Carbon_edge
     Carbon(Migrate)    = (E_ser_deser * Carbon_local) + (E_net * Carbon_net) + (E_cloud_decode * Carbon_cloud)
   If remaining tokens are short, or network bandwidth is constrained, naive
   migration increases emissions and causes SLA violations.

3. Candidate Strategies:
   - STAY_LOCAL: Complete remaining tokens on edge node.
   - MIGRATE_INT8: Quantize KV cache to 8-bit affine, transfer, resume on cloud.
   - MIGRATE_FP16: Transfer raw FP16 KV cache, resume on cloud.
   - COLD_REPREFILL: Discard cache, recompute prompt on cloud.
"""

from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
import json
import os
import math

from core.network_sim import NetworkSimulator, NetworkProfile, DEFAULT_PROFILES


@dataclass
class DeviceEnergyProfile:
    """Power and compute performance characteristics of a continuum compute node."""
    name: str
    active_power_w: float       # Active inference power draw (Watts)
    idle_power_w: float         # Idle baseline power draw (Watts)
    decode_ms_per_token: float  # Generation latency per token (ms)
    prefill_ms_per_token: float # Prefill processing latency per token (ms)
    description: str = ""


# Calibrated profiles for M1 Edge Node vs Datacenter Cloud Server
DEFAULT_DEVICE_PROFILES: Dict[str, DeviceEnergyProfile] = {
    "edge_m1": DeviceEnergyProfile(
        name="edge_m1",
        active_power_w=12.5,        # Apple Silicon M1 SoC under active inference
        idle_power_w=1.8,          # M1 idle power draw
        decode_ms_per_token=80.0,   # ~12.5 tokens/sec autoregressive decode
        prefill_ms_per_token=2.2,   # Measured CPU prefill rate
        description="Apple Silicon M1 Edge Node (Local Device / Private Edge)",
    ),
    "cloud_server": DeviceEnergyProfile(
        name="cloud_server",
        active_power_w=160.0,       # Datacenter server + accelerator active power
        idle_power_w=40.0,         # Server idle power
        decode_ms_per_token=16.0,   # ~62.5 tokens/sec high-throughput decode
        prefill_ms_per_token=0.18,  # Fast parallel prefill on server accelerator
        description="Datacenter Server Accelerator (Cloud Hyperscaler)",
    ),
}

# Network transmission energy model (Joules per Megabyte transmitted)
# Sourced from 3GPP cellular edge and datacenter optical interconnect literature
DEFAULT_NETWORK_ENERGY_J_PER_MB: Dict[str, float] = {
    "5g_edge_uplink": 0.145,           # ~40 mWh/GB over 5G radio access
    "constrained_edge_wan": 0.280,     # Higher retransmission/buffer overhead over WAN
    "datacenter_interconnect": 0.006,  # High-efficiency optical fiber interconnect
}


@dataclass
class CandidateAction:
    """Evaluation result for a specific candidate execution action."""
    action_name: str                  # STAY_LOCAL, MIGRATE_INT8, MIGRATE_FP16, COLD_REPREFILL
    meets_sla: bool                   # True if TTFRT <= SLA budget
    estimated_ttfrt_ms: float         # Time-to-first-resumed-token / handover downtime (ms)
    remaining_decode_ms: float        # Remaining tokens generation time (ms)
    total_remaining_latency_ms: float # TTFRT + remaining_decode_ms
    total_energy_joules: float        # Total energy consumed across all nodes & link (Joules)
    total_carbon_mg: float            # Total carbon emissions in milligrams of CO2 (mgCO2)
    net_carbon_savings_mg: float      # Delta carbon relative to STAY_LOCAL (> 0 means cleaner)
    net_energy_savings_joules: float  # Delta energy relative to STAY_LOCAL (> 0 means less energy)
    payload_mb: float                 # Transmitted payload size in MB (0 if stay local or cold text)
    rationale: str                    # Detailed explanation of decision


class ContinuumController:
    """Joint SLA-Budget and Net-Carbon Aware Transition Controller."""

    def __init__(
        self,
        edge_profile: Optional[DeviceEnergyProfile] = None,
        cloud_profile: Optional[DeviceEnergyProfile] = None,
        network_sim: Optional[NetworkSimulator] = None,
        network_energy_model: Optional[Dict[str, float]] = None,
    ):
        self.edge_device = edge_profile or DEFAULT_DEVICE_PROFILES["edge_m1"]
        self.cloud_device = cloud_profile or DEFAULT_DEVICE_PROFILES["cloud_server"]
        self.net_sim = network_sim or NetworkSimulator()
        self.net_energy_map = network_energy_model or dict(DEFAULT_NETWORK_ENERGY_J_PER_MB)

    @staticmethod
    def estimate_cache_payload_bytes(seq_len: int, mode: str = "fp16") -> int:
        """Estimates serialized KV-cache payload size in bytes for Qwen2.5-0.5B.
        
        Architecture: 24 layers, 2 KV heads, 64 head dim.
        FP16: seq_len * 24 * 2 * 64 * 2 * 2 bytes = seq_len * 12,288 bytes + metadata (~1KB).
        INT8: seq_len * 6,144 bytes + scale/zp metadata (~3KB).
        """
        if mode == "fp16":
            return int(seq_len * 12288 + 1200)
        elif mode == "int8":
            return int(seq_len * 6144 + 3200)
        elif mode == "prompt_text":
            return int(seq_len * 4 + 128)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def estimate_ser_deser_ms(seq_len: int, mode: str) -> Tuple[float, float]:
        """Estimates serialization and deserialization time (ms) based on empirical benchmarks."""
        if mode == "fp16":
            ser_ms = max(1.5, seq_len * 0.016)
            deser_ms = max(2.5, seq_len * 0.015)
        elif mode == "int8":
            ser_ms = max(3.0, seq_len * 0.018)
            deser_ms = max(3.5, seq_len * 0.016)
        else:
            ser_ms, deser_ms = 0.5, 0.5
        return ser_ms, deser_ms

    @staticmethod
    def joules_to_carbon_mg(energy_joules: float, carbon_intensity_g_per_kwh: float) -> float:
        """Converts energy in Joules to carbon emissions in milligrams of CO2 (mgCO2).
        
        1 kWh = 3,600,000 Joules.
        Emissions (gCO2) = (Joules / 3,600,000) * (gCO2/kWh)
        Emissions (mgCO2) = Emissions (gCO2) * 1,000 = (Joules * gCO2/kWh) / 3600.0
        """
        return (energy_joules * carbon_intensity_g_per_kwh) / 3600.0

    def evaluate_all_actions(
        self,
        current_seq_len: int,
        remaining_tokens: int,
        sla_budget_ms: float,
        network_profile_name: str,
        edge_carbon_intensity: float,
        cloud_carbon_intensity: float,
    ) -> Dict[str, CandidateAction]:
        """Evaluates all candidate actions for given sequence length and environment."""
        profile = self.net_sim.get_profile(network_profile_name)
        net_j_per_mb = self.net_energy_map.get(network_profile_name, 0.15)
        actions: Dict[str, CandidateAction] = {}

        # 1. ACTION: STAY_LOCAL (Edge runs remaining tokens to completion)
        stay_ttfrt = 0.0
        stay_decode_ms = remaining_tokens * self.edge_device.decode_ms_per_token
        stay_total_latency = stay_decode_ms
        stay_energy_j = (stay_decode_ms / 1000.0) * self.edge_device.active_power_w
        stay_carbon_mg = self.joules_to_carbon_mg(stay_energy_j, edge_carbon_intensity)

        actions["STAY_LOCAL"] = CandidateAction(
            action_name="STAY_LOCAL",
            meets_sla=True,
            estimated_ttfrt_ms=stay_ttfrt,
            remaining_decode_ms=stay_decode_ms,
            total_remaining_latency_ms=stay_total_latency,
            total_energy_joules=stay_energy_j,
            total_carbon_mg=stay_carbon_mg,
            net_carbon_savings_mg=0.0,
            net_energy_savings_joules=0.0,
            payload_mb=0.0,
            rationale="Remain on edge device. Zero handover downtime.",
        )

        # 2. ACTION: MIGRATE_INT8 (Adaptive Quantization & State Transfer)
        int8_bytes = self.estimate_cache_payload_bytes(current_seq_len, mode="int8")
        int8_mb = int8_bytes / 1_000_000.0
        int8_ser_ms, int8_deser_ms = self.estimate_ser_deser_ms(current_seq_len, mode="int8")
        int8_net_est = self.net_sim.estimate_latency(int8_bytes, profile_name=network_profile_name)
        int8_trans_ms = int8_net_est["total_latency_ms"]
        cloud_1st_step_ms = self.cloud_device.decode_ms_per_token

        int8_ttfrt = int8_ser_ms + int8_trans_ms + int8_deser_ms + cloud_1st_step_ms
        int8_meets_sla = (int8_ttfrt <= sla_budget_ms)

        cloud_rem_decode_ms = max(0, remaining_tokens - 1) * self.cloud_device.decode_ms_per_token
        int8_total_latency = int8_ttfrt + cloud_rem_decode_ms

        ser_energy_j = (int8_ser_ms / 1000.0) * self.edge_device.active_power_w
        deser_energy_j = (int8_deser_ms / 1000.0) * self.cloud_device.active_power_w
        net_energy_j = int8_mb * net_j_per_mb
        cloud_decode_energy_j = ((cloud_1st_step_ms + cloud_rem_decode_ms) / 1000.0) * self.cloud_device.active_power_w

        int8_total_energy = ser_energy_j + deser_energy_j + net_energy_j + cloud_decode_energy_j
        net_carbon_intensity = (edge_carbon_intensity + cloud_carbon_intensity) / 2.0
        int8_carbon_mg = (
            self.joules_to_carbon_mg(ser_energy_j, edge_carbon_intensity)
            + self.joules_to_carbon_mg(net_energy_j, net_carbon_intensity)
            + self.joules_to_carbon_mg(deser_energy_j + cloud_decode_energy_j, cloud_carbon_intensity)
        )

        actions["MIGRATE_INT8"] = CandidateAction(
            action_name="MIGRATE_INT8",
            meets_sla=int8_meets_sla,
            estimated_ttfrt_ms=int8_ttfrt,
            remaining_decode_ms=cloud_1st_step_ms + cloud_rem_decode_ms,
            total_remaining_latency_ms=int8_total_latency,
            total_energy_joules=int8_total_energy,
            total_carbon_mg=int8_carbon_mg,
            net_carbon_savings_mg=stay_carbon_mg - int8_carbon_mg,
            net_energy_savings_joules=stay_energy_j - int8_total_energy,
            payload_mb=int8_mb,
            rationale=(
                f"Stateful INT8 migration ({int8_mb:.2f} MB). TTFRT: {int8_ttfrt:.1f}ms "
                f"({'Meets' if int8_meets_sla else 'Exceeds'} SLA {sla_budget_ms:.0f}ms)."
            ),
        )

        # 3. ACTION: MIGRATE_FP16 (Raw Uncompressed State Transfer)
        fp16_bytes = self.estimate_cache_payload_bytes(current_seq_len, mode="fp16")
        fp16_mb = fp16_bytes / 1_000_000.0
        fp16_ser_ms, fp16_deser_ms = self.estimate_ser_deser_ms(current_seq_len, mode="fp16")
        fp16_net_est = self.net_sim.estimate_latency(fp16_bytes, profile_name=network_profile_name)
        fp16_trans_ms = fp16_net_est["total_latency_ms"]

        fp16_ttfrt = fp16_ser_ms + fp16_trans_ms + fp16_deser_ms + cloud_1st_step_ms
        fp16_meets_sla = (fp16_ttfrt <= sla_budget_ms)
        fp16_total_latency = fp16_ttfrt + cloud_rem_decode_ms

        fp16_ser_energy_j = (fp16_ser_ms / 1000.0) * self.edge_device.active_power_w
        fp16_deser_energy_j = (fp16_deser_ms / 1000.0) * self.cloud_device.active_power_w
        fp16_net_energy_j = fp16_mb * net_j_per_mb
        fp16_total_energy = fp16_ser_energy_j + fp16_deser_energy_j + fp16_net_energy_j + cloud_decode_energy_j

        fp16_carbon_mg = (
            self.joules_to_carbon_mg(fp16_ser_energy_j, edge_carbon_intensity)
            + self.joules_to_carbon_mg(fp16_net_energy_j, net_carbon_intensity)
            + self.joules_to_carbon_mg(fp16_deser_energy_j + cloud_decode_energy_j, cloud_carbon_intensity)
        )

        actions["MIGRATE_FP16"] = CandidateAction(
            action_name="MIGRATE_FP16",
            meets_sla=fp16_meets_sla,
            estimated_ttfrt_ms=fp16_ttfrt,
            remaining_decode_ms=cloud_1st_step_ms + cloud_rem_decode_ms,
            total_remaining_latency_ms=fp16_total_latency,
            total_energy_joules=fp16_total_energy,
            total_carbon_mg=fp16_carbon_mg,
            net_carbon_savings_mg=stay_carbon_mg - fp16_carbon_mg,
            net_energy_savings_joules=stay_energy_j - fp16_total_energy,
            payload_mb=fp16_mb,
            rationale=(
                f"Stateful FP16 migration ({fp16_mb:.2f} MB). TTFRT: {fp16_ttfrt:.1f}ms "
                f"({'Meets' if fp16_meets_sla else 'Exceeds'} SLA {sla_budget_ms:.0f}ms)."
            ),
        )

        # 4. ACTION: COLD_REPREFILL (Prompt Recomputation on Cloud)
        prompt_bytes = self.estimate_cache_payload_bytes(current_seq_len, mode="prompt_text")
        prompt_mb = prompt_bytes / 1_000_000.0
        prompt_net_est = self.net_sim.estimate_latency(prompt_bytes, profile_name=network_profile_name)
        prompt_trans_ms = prompt_net_est["total_latency_ms"]

        cloud_prefill_ms = current_seq_len * self.cloud_device.prefill_ms_per_token
        cold_ttfrt = prompt_trans_ms + cloud_prefill_ms + cloud_1st_step_ms
        cold_meets_sla = (cold_ttfrt <= sla_budget_ms)
        cold_total_latency = cold_ttfrt + cloud_rem_decode_ms

        cold_net_energy_j = prompt_mb * net_j_per_mb
        cold_compute_energy_j = ((cloud_prefill_ms + cloud_1st_step_ms + cloud_rem_decode_ms) / 1000.0) * self.cloud_device.active_power_w
        cold_total_energy = cold_net_energy_j + cold_compute_energy_j

        cold_carbon_mg = (
            self.joules_to_carbon_mg(cold_net_energy_j, net_carbon_intensity)
            + self.joules_to_carbon_mg(cold_compute_energy_j, cloud_carbon_intensity)
        )

        actions["COLD_REPREFILL"] = CandidateAction(
            action_name="COLD_REPREFILL",
            meets_sla=cold_meets_sla,
            estimated_ttfrt_ms=cold_ttfrt,
            remaining_decode_ms=cloud_1st_step_ms + cloud_rem_decode_ms,
            total_remaining_latency_ms=cold_total_latency,
            total_energy_joules=cold_total_energy,
            total_carbon_mg=cold_carbon_mg,
            net_carbon_savings_mg=stay_carbon_mg - cold_carbon_mg,
            net_energy_savings_joules=stay_energy_j - cold_total_energy,
            payload_mb=prompt_mb,
            rationale=(
                f"Cold prompt recomputation on cloud. TTFRT: {cold_ttfrt:.1f}ms "
                f"({'Meets' if cold_meets_sla else 'Exceeds'} SLA {sla_budget_ms:.0f}ms)."
            ),
        )

        return actions

    def decide(
        self,
        current_seq_len: int,
        remaining_tokens: int,
        sla_budget_ms: float,
        network_profile_name: str,
        edge_carbon_intensity: float,
        cloud_carbon_intensity: float,
        carbon_savings_threshold_mg: float = 0.5,
    ) -> CandidateAction:
        """Determines the optimal action balancing SLA time budget and net carbon emissions."""
        actions = self.evaluate_all_actions(
            current_seq_len=current_seq_len,
            remaining_tokens=remaining_tokens,
            sla_budget_ms=sla_budget_ms,
            network_profile_name=network_profile_name,
            edge_carbon_intensity=edge_carbon_intensity,
            cloud_carbon_intensity=cloud_carbon_intensity,
        )

        # 1. Filter candidates that satisfy the SLA downtime budget
        feasible_actions = {name: act for name, act in actions.items() if act.meets_sla}

        migration_candidates = [
            act for name, act in feasible_actions.items() if name != "STAY_LOCAL"
        ]

        if not migration_candidates:
            selected = actions["STAY_LOCAL"]
            selected.rationale = (
                f"MIGRATION REJECTED: All migration candidates exceed SLA budget ({sla_budget_ms:.0f} ms). "
                f"Remaining locally on edge to preserve user responsiveness."
            )
            return selected

        # 2. Check Net Carbon Savings among SLA-compliant migration candidates
        best_mig = max(migration_candidates, key=lambda a: a.net_carbon_savings_mg)

        if best_mig.net_carbon_savings_mg < carbon_savings_threshold_mg:
            selected = actions["STAY_LOCAL"]
            selected.rationale = (
                f"MIGRATION REJECTED: Net-carbon negative or marginal ({best_mig.net_carbon_savings_mg:.2f} mgCO2 saved vs "
                f"threshold {carbon_savings_threshold_mg:.2f} mgCO2). Transmission overhead outweighs compute savings."
            )
            return selected

        # 3. Migration is SLA-compliant and yields proven net-carbon reduction
        best_mig.rationale = (
            f"MIGRATION APPROVED ({best_mig.action_name}): Satisfies SLA budget ({best_mig.estimated_ttfrt_ms:.1f}ms <= {sla_budget_ms:.0f}ms) "
            f"and achieves {best_mig.net_carbon_savings_mg:.1f} mgCO2 net savings ({best_mig.net_energy_savings_joules:.1f} J saved)."
        )
        return best_mig


class NaiveCarbonController:
    """Baseline naive controller that migrates whenever cloud is greener, ignoring SLA & network."""

    def __init__(self, continuum_controller: ContinuumController):
        self.cc = continuum_controller

    def decide(
        self,
        current_seq_len: int,
        remaining_tokens: int,
        sla_budget_ms: float,
        network_profile_name: str,
        edge_carbon_intensity: float,
        cloud_carbon_intensity: float,
    ) -> CandidateAction:
        """Blindly migrates (using raw FP16) if cloud_carbon < edge_carbon, ignoring SLA."""
        actions = self.cc.evaluate_all_actions(
            current_seq_len=current_seq_len,
            remaining_tokens=remaining_tokens,
            sla_budget_ms=sla_budget_ms,
            network_profile_name=network_profile_name,
            edge_carbon_intensity=edge_carbon_intensity,
            cloud_carbon_intensity=cloud_carbon_intensity,
        )

        if cloud_carbon_intensity < edge_carbon_intensity:
            chosen = actions["MIGRATE_FP16"]
            chosen.rationale = (
                f"NAIVE MIGRATION: Cloud carbon ({cloud_carbon_intensity:.1f}) is cleaner than edge ({edge_carbon_intensity:.1f}). "
                f"Triggered raw FP16 migration unconditionally."
            )
            return chosen
        else:
            return actions["STAY_LOCAL"]


def load_carbon_trace(trace_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads dynamic carbon intensity trace data."""
    if not trace_path:
        trace_path = os.path.join(
            os.path.dirname(__file__), "..", "traces", "sample_carbon_trace.json"
        )
    with open(trace_path, "r") as f:
        return json.load(f)

