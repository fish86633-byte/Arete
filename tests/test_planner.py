"""Integration checks for the relocated planner, not formal performance tests."""

from pathlib import Path
import json
import unittest

import numpy as np

from arete import paper_config, plan, verify_plan
from arete.demo import demo_inputs
from arete._frozen.pcqr.support import state_hash


def fingerprint(result):
    trace = result["trace"]
    construction = result["proposal"]["construction"]
    return {"active_assignments": np.argwhere(result["plan"]).tolist(),
            "initial_assignments": np.argwhere(result["initial_plan"]).tolist(),
            "accepted": result["accepted"], "stop": result["stop"],
            "F": [trace.F.numerator, trace.F.denominator],
            "W_RT": [trace.W.numerator, trace.W.denominator],
            "bits": trace.bits, "weighted_wait": trace.weighted_wait,
            "end_state_hash": state_hash(trace.end_state),
            "addition_decisions": [[{
                "satellite": a["global_satellite_id"], "cell": a["cell"],
                "delta": [a["delta_proxy"].numerator, a["delta_proxy"].denominator]}
                for a in slot["additions"]] for slot in construction["slots"]]}


class PlannerIntegrationTests(unittest.TestCase):
    def test_original_decisions_and_independent_audit(self):
        state, arrivals, positions, cells, ids, config, constants = demo_inputs()
        old_rt, old_nrt = state.rt.copy(), state.nrt_batches.to_json()
        old_arrivals, old_positions = arrivals.copy(), positions.copy()
        result = plan(state, arrivals, positions, cells, ids,
                      config=config, constants=constants)
        self.assertEqual(result["run_status"], "OK", result.get("error"))
        expected = json.loads((Path(__file__).parent / "fixtures/original_demo.json").read_text())
        self.assertEqual(fingerprint(result), expected["fingerprint"])
        self.assertEqual(result["proposal"]["construction"]["private_implementation"], "UB")
        self.assertGreater(result["trace"].F, result["original_own_TPG"]["F"])
        audit = verify_plan(result["plan"], state, arrivals, result["channel"],
                            expected=result["trace"])
        self.assertEqual(audit.status, "PASS")
        self.assertEqual(audit.physical_input_mode, "scalar_geometry_airy")
        np.testing.assert_array_equal(state.rt, old_rt)
        self.assertEqual(state.nrt_batches.to_json(), old_nrt)
        np.testing.assert_array_equal(arrivals, old_arrivals)
        np.testing.assert_array_equal(positions, old_positions)

    def test_zero_budget_only_publishes_registered_silence(self):
        state, arrivals, positions, cells, ids, config, constants = demo_inputs()
        config["planning_budget_s"] = 0.0
        result = plan(state, arrivals, positions, cells, ids, config=config)
        self.assertEqual(result["run_status"], "OK")
        self.assertEqual(result["stop"], "TIME_LIMIT")
        self.assertEqual(result["reliable_plan_source"], "ALL_SILENT_REGISTERED")
        self.assertFalse(np.any(result["plan"]))
        self.assertIsNone(result["trace"])
        self.assertTrue(result["fallback_used"])
        audit = verify_plan(result["plan"], state, arrivals, result["channel"])
        self.assertEqual(audit.delivered_packets, 0)
        self.assertGreater(int(audit.expired.sum()), 0)

    def test_two_horizons_keep_fifo_and_absolute_slots(self):
        state, arrivals, positions, cells, ids, config, constants = demo_inputs()
        first = plan(state, arrivals, positions, cells, ids, config=config)
        first_audit = verify_plan(first["plan"], state, arrivals, first["channel"],
                                  expected=first["trace"])
        next_state = first["trace"].end_state
        second = plan(next_state, arrivals, positions, cells, ids, config=config, cycle=1)
        second_audit = verify_plan(second["plan"], next_state, arrivals, second["channel"],
                                   start_slot=config["H"], expected=second["trace"],
                                   previous_end=first_audit.final_state)
        self.assertEqual(second_audit.status, "PASS")
        self.assertEqual(second["trace"].abs_start, config["H"])

    def test_configuration_copies_and_load_scaling(self):
        config = paper_config(0.8)
        self.assertEqual(config["R_total"], 0.8 * config["C_ref"])
        self.assertEqual((config["H"], config["G_exec"], config["K"], config["ttl"]),
                         (20, 20, 4, 60))
        self.assertEqual(config["planning_budget_s"], 2.0)
        config["K"] = 99
        self.assertEqual(paper_config()["K"], 4)
        with self.assertRaises(ValueError):
            paper_config(float("nan"))

    def test_paper_parameters_match_registered_source(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "provenance/source_manifest.json").read_text())
        original = json.loads((root / "provenance/original_rho020_seed0_config.json").read_text())
        for key in manifest["configuration"]["removed_driver_metadata"]:
            original.pop(key, None)
        self.assertEqual(paper_config(0.2), original)

    def test_geometry_horizon_mismatch_is_rejected(self):
        state, arrivals, positions, cells, ids, config, constants = demo_inputs()
        with self.assertRaisesRegex(ValueError, "configured H"):
            plan(state, arrivals, positions[:-1], cells, ids, config=config)


if __name__ == "__main__":
    unittest.main()
