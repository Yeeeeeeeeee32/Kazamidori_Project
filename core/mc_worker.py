"""
core/mc_worker.py

Picklable worker functions for ProcessPoolExecutor.
Must not import PySide6 or any Qt modules, as doing so inside a subprocess
can cause long hangs on Windows.
"""
from __future__ import annotations
import random as _random

from core.simulation import simulate_once, simulate_once_mc
from core.monte_carlo import _perturb_wind_profile
from core.optimization import p1_objective_score

def _noop_warmup():
    """Dummy function to force subprocesses to spawn and import modules."""
    pass

def _mc_worker_chunk(
    seeds: list[int],
    trials: list[int],
    sim_params: dict,
    wind_unc: float,
    gust_sigma: float,
    tu: float,
    raw_thrust: list,
    elev: float,
    azi: float,
    base_u: list,
    base_v: list,
    flight_mode: str,
    target_radius: float,
    backfire_alt: float = 0.0,
    target_x: float = 0.0,
    target_y: float = 0.0,
) -> list[dict | None]:
    """
    Top-level, picklable worker function for ProcessPoolExecutor.
    Runs a chunk of perturbed Monte Carlo simulations.

    If *backfire_alt* > 0 (provided by the nominal run), the fast
    single-pass ``simulate_once_mc`` is used, halving ODE integrations.
    Falls back to the full two-pass ``simulate_once`` otherwise.
    """
    results = []
    for seed, trial_idx in zip(seeds, trials):
        rng = _random.Random(seed)

        # ── Wind perturbation
        u_prof, v_prof, _ = _perturb_wind_profile(
            base_u, base_v, rng,
            wind_unc, gust_intensity=gust_sigma,
        )

        # ── Thrust perturbation
        thrust_scale = max(0.1, 1.0 + rng.gauss(0.0, tu))
        perturbed_thrust = [[t, T * thrust_scale] for t, T in raw_thrust]

        trial_p = {
            **sim_params,
            "wind_u_prof": u_prof,
            "wind_v_prof": v_prof,
            "thrust_data": perturbed_thrust,
        }

        # Use fast single-pass variant when the nominal backfire_alt is known.
        # Falls back to the full two-pass simulate_once if it was not provided.
        if backfire_alt > 0.0:
            r = simulate_once_mc(elev, azi, trial_p, backfire_alt, trial_idx=trial_idx)
        else:
            r = simulate_once(elev, azi, trial_p, trial_idx=trial_idx)

        if r["ok"]:
            score = p1_objective_score(r, flight_mode, target_radius, target_x, target_y)

            h_time = float(r["hang_time"])
            bf_t = float(r.get("bf_abs_time", 0.0))
            if '有翼' in flight_mode or 'winged' in flight_mode.lower() or 'wing' in flight_mode.lower():
                h_time = max(0.0, h_time - bf_t)

            # Only return the scalars (no trajectory arrays — minimise IPC payload)
            res = {
                "x":           float(r["impact_x"]),
                "y":           float(r["impact_y"]),
                "apogee":      float(r["apogee_m"]),
                "hang_time":   h_time,
                # bf_abs_time: time of ejection charge firing (backfire).
                # Required by p1_objective_score for Winged Hover mode:
                #   score = hang_time - bf_abs_time  (payload airborne duration)
                "bf_abs_time": bf_t,
                "score": score,
            }
        else:
            res = None

        # Memory optimization
        del r, trial_p, u_prof, v_prof, perturbed_thrust
        results.append(res)

    return results
