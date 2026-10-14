"""Regression coverage for the actual RocketPy stability gate and search errors."""
import contextlib
import copy
import io
import json
import math
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from core import optimization
from core.simulation import SimulationInputError, simulate_once, simulate_once_mc


class TestSimulationFailures(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "fixtures" / "simulation_params.json"
        self.params = json.loads(path.read_text(encoding="utf-8"))
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def optimize(self, stop=None):
        return optimization.optimize_launch_angle(
            mode="自由", base_params=self.params, r_max=1000, landing_prob=90,
            wind_uncertainty=0, thrust_uncertainty=0,
            stop_flag=stop or threading.Event(), progress_cb=lambda *_: None)

    def test_actual_stability_gate_reports_geometry_in_both_simulators(self):
        before = copy.deepcopy(self.params)
        for run in (lambda: simulate_once(84, 0, self.params),
                    lambda: simulate_once_mc(84, 0, self.params, backfire_alt=50)):
            with self.subTest(run=run), patch("core.simulation.Flight") as flight:
                result = run()
                self.assertFalse(result["ok"])
                self.assertEqual(result["error_code"], "unstable_rocket")
                self.assertGreater(result["stability"]["cg_m"], result["stability"]["cp_m"])
                self.assertAlmostEqual(result["stability"]["cg_m"], 0.3155, delta=0.0005)
                self.assertAlmostEqual(result["stability"]["cp_m"], 0.3056, delta=0.0005)
                self.assertAlmostEqual(result["stability"]["propellant_mass_kg"], 100 / (80 * 9.80665), places=5)
                self.assertIn("Isp = 80 s", result["error"])
                flight.assert_not_called()
        self.assertEqual(self.params, before)

    def test_stable_model_still_runs_real_flight(self):
        self.params.update(thrust_data=[[0, 0], [0.1, 12], [0.3, 7], [1.1, 0]], motor_burn_time=1.1)
        result = simulate_once(84, 0, self.params)
        self.assertTrue(result["ok"], result.get("error"))
        self.assertGreater(result["apogee_m"], 0)
        self.assertGreater(result["hang_time"], 0)
        for key in ("x_vals", "y_vals", "z_vals"):
            self.assertTrue(all(math.isfinite(v) for v in result[key]))

    def test_search_stops_after_first_angle_independent_stability_failure(self):
        with patch.object(optimization, "simulate_once", wraps=simulate_once) as simulate:
            with self.assertRaises(SimulationInputError) as error:
                self.optimize()
        self.assertEqual(simulate.call_count, 1)
        self.assertIn("CG with motor/propellant", str(error.exception))
        self.assertIn("Static margin", str(error.exception))

    def test_all_candidate_failure_keeps_distinct_underlying_causes(self):
        def fail(elev, azi, params):
            return {"ok": False, "error": "Bad wind data" if azi < 180 else "Numerical failure"}
        with patch.object(optimization, "simulate_once", side_effect=fail) as simulate:
            with self.assertRaises(SimulationInputError) as error:
                self.optimize()
        self.assertEqual(simulate.call_count, 48)
        bad_wind_count = sum(call.args[1] < 180 for call in simulate.call_args_list)
        self.assertIn(f"{bad_wind_count}/48: Bad wind data", str(error.exception))
        self.assertIn(f"{48-bad_wind_count}/48: Numerical failure", str(error.exception))

    def test_angle_dependent_failure_does_not_prevent_later_success(self):
        self.params.update(thrust_data=[[0, 0], [0.1, 12], [0.3, 7], [1.1, 0]], motor_burn_time=1.1)
        valid = simulate_once(84, 0, self.params)
        self.assertTrue(valid["ok"])
        count = 0
        def sometimes_fails(*_args, **_kwargs):
            nonlocal count
            count += 1
            return {"ok": False, "error": "integration failed"} if count < 3 else valid
        # Stop immediately after annealing: this regression only checks
        # that an ordinary per-angle failure does not abort that search.
        with patch.object(optimization, "simulate_once", side_effect=sometimes_fails), \
             patch.object(optimization, "_monte_carlo_r90", side_effect=RuntimeError("Reached MC verification")):
            with self.assertRaisesRegex(RuntimeError, "Reached MC verification"):
                self.optimize()
        self.assertEqual(count, 48)

    def test_cancelled_search_does_not_evaluate_candidates(self):
        stop = threading.Event()
        stop.set()
        with patch.object(optimization, "simulate_once") as simulate:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.optimize(stop)
            simulate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
