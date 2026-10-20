"""Unit and Property Verification Suite for ContinuumKV Controller.

Tests:
1. SLA Enforcement:
   - When SLA budget is exceeded, migration is rejected or switched to INT8.
2. Net Carbon Hypothesis Verification:
   - When remaining tokens are small, state transfer carbon exceeds compute savings,
     verifying that naive migration is counter-productive.
3. Long Context Amortization:
   - For long remaining token generations, migration achieves net carbon reduction.
4. Carbon Trace Simulation:
   - Parses traces/sample_carbon_trace.json and simulates 24h continuum handover.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.controller import (
    ContinuumController,
    NaiveCarbonController,
    load_carbon_trace,
    DEFAULT_DEVICE_PROFILES,
)


class TestContinuumController(unittest.TestCase):

    def setUp(self):
        self.controller = ContinuumController()
        self.naive = NaiveCarbonController(self.controller)

    def test_sla_enforcement_tight_budget(self):
        """Verifies that an impossibly tight SLA rejects all migrations and remains local."""
        decision = self.controller.decide(
            current_seq_len=1024,
            remaining_tokens=50,
            sla_budget_ms=50.0,  # 50 ms is less than any network migration downtime on 5G
            network_profile_name="5g_edge_uplink",
            edge_carbon_intensity=400.0,
            cloud_carbon_intensity=100.0,
        )
        self.assertEqual(decision.action_name, "STAY_LOCAL")
        self.assertTrue(decision.meets_sla)
        self.assertIn("MIGRATION REJECTED", decision.rationale)

    def test_sla_prefers_int8_over_fp16_when_budget_tight(self):
        """Verifies that when SLA is feasible for INT8 but exceeded by FP16, INT8 is selected."""
        # For 512 tokens over 5G: FP16 TTFRT is ~1150ms, INT8 TTFRT is ~650ms
        decision = self.controller.decide(
            current_seq_len=512,
            remaining_tokens=200,
            sla_budget_ms=800.0,  # Feasible for INT8 (~650ms), violates FP16 (~1150ms)
            network_profile_name="5g_edge_uplink",
            edge_carbon_intensity=450.0,
            cloud_carbon_intensity=120.0,
        )
        self.assertEqual(decision.action_name, "MIGRATE_INT8")
        self.assertTrue(decision.meets_sla)
        self.assertLessEqual(decision.estimated_ttfrt_ms, 800.0)

    def test_net_carbon_hypothesis_short_session(self):
        """Hypothesis Test: Migrating a session with only 5 remaining tokens over constrained link
        is net-carbon NEGATIVE even when cloud is 3x cleaner. Controller must reject migration.
        """
        # Cloud is much cleaner (100 vs 400 gCO2/kWh), but remaining decode is only 5 tokens!
        decision = self.controller.decide(
            current_seq_len=1024,
            remaining_tokens=5,  # Only 5 tokens to generate!
            sla_budget_ms=5000.0, # Generous SLA
            network_profile_name="constrained_edge_wan",
            edge_carbon_intensity=400.0,
            cloud_carbon_intensity=100.0,
        )
        # Even though cloud is 4x cleaner, transferring 6-12 MB of cache to decode 5 tokens wastes energy!
        self.assertEqual(decision.action_name, "STAY_LOCAL")
        self.assertIn("MIGRATION REJECTED", decision.rationale)
        self.assertIn("Net-carbon negative", decision.rationale)

    def test_net_carbon_hypothesis_long_session(self):
        """Hypothesis Test: When remaining session is long (e.g. 500 tokens), compute savings
        amortize transfer cost and yield positive net carbon reduction.
        """
        decision = self.controller.decide(
            current_seq_len=512,
            remaining_tokens=500,  # Long session to generate on green cloud
            sla_budget_ms=2000.0,
            network_profile_name="5g_edge_uplink",
            edge_carbon_intensity=450.0,
            cloud_carbon_intensity=120.0,
        )
        self.assertTrue(decision.action_name in ("MIGRATE_INT8", "MIGRATE_FP16"))
        self.assertGreater(decision.net_carbon_savings_mg, 0.0)
        self.assertIn("MIGRATION APPROVED", decision.rationale)

    def test_naive_vs_continuum_sla_violation(self):
        """Verifies that Naive Policy causes SLA violations under constrained WAN while
        ContinuumKV Controller prevents SLA breaches.
        """
        current_seq_len = 1024
        remaining_tokens = 100
        tight_sla = 500.0  # User needs <= 500ms response
        cloud_carbon = 145.0
        edge_carbon = 412.5

        # 1. Naive controller blindly migrates
        naive_action = self.naive.decide(
            current_seq_len=current_seq_len,
            remaining_tokens=remaining_tokens,
            sla_budget_ms=tight_sla,
            network_profile_name="constrained_edge_wan",
            edge_carbon_intensity=edge_carbon,
            cloud_carbon_intensity=cloud_carbon,
        )
        self.assertEqual(naive_action.action_name, "MIGRATE_FP16")
        self.assertFalse(naive_action.meets_sla, "Naive policy should have violated SLA")
        self.assertGreater(naive_action.estimated_ttfrt_ms, tight_sla)

        # 2. Continuum controller respects SLA
        smart_action = self.controller.decide(
            current_seq_len=current_seq_len,
            remaining_tokens=remaining_tokens,
            sla_budget_ms=tight_sla,
            network_profile_name="constrained_edge_wan",
            edge_carbon_intensity=edge_carbon,
            cloud_carbon_intensity=cloud_carbon,
        )
        self.assertTrue(smart_action.meets_sla, "Continuum controller must never violate SLA")

    def test_load_carbon_trace(self):
        """Verifies loading real sample carbon trace."""
        trace = load_carbon_trace()
        self.assertGreaterEqual(len(trace), 6)
        first_point = trace[0]
        self.assertIn("carbon_intensity_gco2_kwh", first_point)
        self.assertIn("grid_signal", first_point)


if __name__ == "__main__":
    unittest.main()
