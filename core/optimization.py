"""
core/optimization.py
Two-layer launch-angle optimisation and Phase-1 limit-margin search.

Public API
----------
p1_objective_score(res, mode) -> float
    Compute the scalar objective from a simulation result dict.

optimize_launch_angle(mode, base_params, r_max, sim_fn, mc_r90_fn,
                      landing_prob, stop_flag, progress_cb) -> dict
    Coarse grid-search + Monte-Carlo verification (the original
    _optimize_worker logic).  Returns a result dict or raises on error.

p1_params_at_wind(base_params, mu_surf) -> dict
    Return a copy of params with the wind speed scaled to mu_surf.

p1_mc_points(elev, azi, base_params, mu, sigma, n, sim_fn,
             stop_flag=None) -> list[(x, y)]
    Run n Monte Carlo sims at the given wind statistics.

p1_ellipse_params(points) -> (cx, cy, eigvals, eigvecs)
    Fit a 2-D covariance ellipse to the landing scatter.

p1_ellipse_breaches_circle(cx, cy, eigvals, eigvecs, R, n_pts=180) -> bool
    True if the 90 % error ellipse extends beyond radius R.

run_phase1(base_params, target_r, mode, stop_flag,
           progress_cb) -> Phase1Result
    Full 5-step Phase-1 analysis (grid search → nominal MC → sensitivity
    → μ_max binary search → σ_max binary search).
    Raises RuntimeError with a user-readable message on failure.

Wind-profile helper
-------------------
build_perturbed_wind_prof(params, rng, wu) -> (u_prof, v_prof,
                                               surf_spd, up_spd,
                                               spd_profile)
    Stochastic wind profile used by both Monte-Carlo passes.
"""

from __future__ import annotations

import math
import random as _random_mod
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import os
import concurrent.futures
import numpy as np

# Relative imports within the package
from .simulation import simulate_once
from .constants  import CHI2_90, OBS_ALT, BLEND_ALT


# ── Wind profile builder (self-contained copy to avoid circular deps) ─────────
# (The canonical copy lives in main.py as WindProfileBuilder; mirrored here
#  so core/ has no dependency on any UI module.)

def _hellmann_alpha(v_lo: float, z_lo: float,
                    v_hi: float, z_hi: float) -> float:
    try:
        if v_lo < 1e-6 or z_lo <= 0 or z_hi <= z_lo:
            return 0.14
        return math.log(max(v_hi, 1e-9) / v_lo) / math.log(z_hi / z_lo)
    except (ValueError, ZeroDivisionError):
        return 0.14


def build_wind_profile(
    v_surf: float, dir_surf_deg: float, z_surf: float,
    v_upper: float, dir_upper_deg: float, z_upper: float,
    alpha: float = None
) -> tuple[list, list]:
    """Return (u_prof, v_prof) for RocketPy custom_atmosphere.

    Altitude 0 is forced to zero wind (below the anemometer).
    """
    if alpha is None:
        alpha = _hellmann_alpha(v_surf, z_surf, v_upper, z_upper)

    # Pre-calculate common terms to avoid redundant math inside the loop
    dir_surf_rad = math.radians(dir_surf_deg)
    dir_upper_rad = math.radians(dir_upper_deg)
    diff_deg = ((dir_upper_deg - dir_surf_deg + 180.0) % 360.0) - 180.0
    diff_rad = math.radians(diff_deg)

    z_diff_inv = 1.0 / (z_upper - z_surf) if z_upper != z_surf else 0.0
    z_surf_inv = 1.0 / z_surf if z_surf != 0 else 0.0

    alts = sorted({0, 3, z_surf, 30, 100, 300, z_upper, 1000, 5000})
    u_prof: list = [(0, 0.0)]
    v_prof: list = [(0, 0.0)]

    for z in alts:
        if z == 0:
            continue

        if z <= z_surf:
            spd = v_surf * (z * z_surf_inv) ** alpha
            rad = dir_surf_rad
        elif z >= z_upper:
            spd = v_upper
            rad = dir_upper_rad
        else:
            spd = v_surf * (z * z_surf_inv) ** alpha
            rad = dir_surf_rad + (z - z_surf) * z_diff_inv * diff_rad

        u_prof.append((z, -spd * math.sin(rad)))
        v_prof.append((z, -spd * math.cos(rad)))

    return u_prof, v_prof


# ── Perturbed wind profile for MC ────────────────────────────────────────────

def build_perturbed_wind_prof(
    params: dict,
    rng: _random_mod.Random,
    wu: float,
) -> tuple[list, list, float, float, list]:
    """Build a Monte-Carlo perturbed wind profile for Phase 1.

    Args:
        params: dict from ui_qt (must contain wind_u_prof, wind_v_prof).
        rng:    seeded PRNG.
        wu:     wind_uncertainty (fraction).

    Returns:
        (u_prof, v_prof, surf_spd, up_spd, spd_profile)
    """
    base_u = params.get('wind_u_prof', [(0, 0.0)])
    base_v = params.get('wind_v_prof', [(0, 0.0)])

    # 0 gust intensity for optimization MC (phase 1)
    u_prof, v_prof, spd_prof = _perturb_wind_profile(base_u, base_v, rng, wu, gust_intensity=0.0)

    surf_spd = math.hypot(u_prof[0][1], v_prof[0][1]) if u_prof else 0.0
    up_spd = math.hypot(u_prof[-1][1], v_prof[-1][1]) if u_prof else 0.0

    return u_prof, v_prof, surf_spd, up_spd, spd_prof


# ── Objective helpers ─────────────────────────────────────────────────────────

def p1_objective_score(res: dict, mode: str, r_max: float = float('inf')) -> float:
    """Return the scalar objective for a simulation result in the given mode.

    Higher is always better (even for Precision Landing where we return
    the negative landing radius).

    Implements hard constraint:
    If mode is not 'Free' (自由) and r_horiz > r_max, return -inf.

    Score definitions
    -----------------
    定点滞空 (Precision Landing) : (r_max - r_horiz) + hang_time
        Minimise landing radius; hang_time tie-breaks equal-radius results.
    高度 (Altitude Competition)  : apogee_m
        Maximise peak altitude.
    有翼 (Winged Hover)          : hang_time - bf_abs_time
        Maximise payload hangtime — time from backfire ejection to landing.
        bf_abs_time is the absolute time at which the ejection charge fires,
        so (hang_time - bf_abs_time) is the duration the payload is airborne
        after being released from the rocket body.
    自由 (Free)                  : apogee_m (default fallback)
    """
    if not res.get('ok', False):
        return float('-inf')

    # Hard constraint: disqualify if r > r_max and not Free mode
    is_free = "free" in mode.lower() or "自由" in mode
    if not is_free and res['r_horiz'] > r_max:
        return float('-inf')

    # Task 2 Mode Objective Scores
    if mode == 'Precision Landing' or '定点滞空' in mode:
        return (r_max - res['r_horiz']) + res['hang_time']
    elif mode == 'Altitude Competition' or '高度' in mode:
        return res['apogee_m']
    elif mode == 'Winged Hover' or '有翼' in mode:
        # Payload hangtime: from ejection charge to landing.
        # bf_abs_time is the moment the backfire fires and the payload is
        # released; hang_time is total flight time.
        # Falls back to total hang_time if bf_abs_time is unavailable
        # (e.g. older cached result dicts from before this change).
        bf_t = res.get('bf_abs_time', 0.0)
        return float(res['hang_time']) - float(bf_t)
    else:
        # Free mode fallback
        return res['apogee_m']

# ── Optimiser (from _optimize_worker) ────────────────────────────────────────

def _grid_search_chunk(chunk_configs: list[tuple[float, float]], base_params: dict) -> list[tuple[float, float, dict]]:
    """Module-level worker for parallel grid search.
    Strips large trajectory arrays from result to minimise IPC overhead.
    """
    results = []
    for e_, a_ in chunk_configs:
        res = simulate_once(e_, a_, base_params)
        if res['ok']:
            light_res = {
                'ok': True,
                'apogee_m': res['apogee_m'],
                'hang_time': res['hang_time'],
                'impact_x': res['impact_x'],
                'impact_y': res['impact_y'],
                'r_horiz': res['r_horiz'],
                'backfire_alt': res['backfire_alt'],
                'bf_abs_time': res.get('bf_abs_time', 0.0),
            }
            results.append((e_, a_, light_res))
        else:
            results.append((e_, a_, {'ok': False, 'error': res.get('error', 'unknown error')}))
    return results


def optimize_launch_angle(
    mode: str,
    base_params: dict,
    r_max: float,
    landing_prob: int,
    wind_uncertainty: float,
    thrust_uncertainty: float,
    stop_flag: threading.Event,
    progress_cb: Callable[[str, float], None],
) -> dict:
    """Coarse grid-search + MC verification optimiser.

    Phase 1: grid search over (elev, azi) to find feasible candidates.
    Phase 2: MC r90 check on the top-5 candidates.
    Phase 3: Final MC analysis on the winner.
    """
    is_free = "free" in mode.lower() or "自由" in mode

    if mode == "Precision Landing" or '定点滞空' in mode:
        elev_grid = [60, 66, 72, 78, 84, 90]
        azi_grid  = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    elif mode == "Altitude Competition" or '高度' in mode:
        elev_grid = [60, 66, 72, 78, 84, 90]
        azi_grid  = [0, 45, 90, 135, 180, 225, 270, 315]
    elif mode == "Winged Hover" or '有翼' in mode:
        elev_grid = [60, 66, 72, 78, 84, 90]
        azi_grid  = [0, 45, 90, 135, 180, 225, 270, 315]
    else:
        # Free mode fallback
        elev_grid = [60, 66, 72, 78, 84, 90]
        azi_grid  = [0, 45, 90, 135, 180, 225, 270, 315]

    def objective(res, mc_r=None):
        if not res.get('ok', False):
            return float('-inf')
        r = res['r_horiz']

        # Hard constraint check incorporating MC radius if available
        # Phase 2 passes mc_r to verify the 90% ellipse stays within target.
        if not is_free:
            check_r = r if mc_r is None else r + mc_r
            if check_r > r_max:
                return float('-inf')

        return p1_objective_score(res, mode, r_max)

    candidates = []
    N       = len(elev_grid) * len(azi_grid)
    done        = 0
    phase1_weight = 0.6

    progress_cb(f"Phase 1: Coarse search (0/{N})", 0.0)

    import os
    import time

    _t_start = time.perf_counter()

    from .pool_manager import get_global_pool
    executor = get_global_pool()

    configs = []
    for e_ in elev_grid:
        for a_ in azi_grid:
            configs.append((e_, a_))

    _n_workers = max(1, (os.cpu_count() or 2) - 2)
    chunksize = max(1, N // _n_workers)
    chunks = [configs[i:i + chunksize] for i in range(0, N, chunksize)]
    futures = [executor.submit(_grid_search_chunk, chunk, base_params) for chunk in chunks]

    while futures:
        if stop_flag.is_set():
            for f in futures:
                f.cancel()
            raise RuntimeError('cancelled')
            
        done_futures, futures = concurrent.futures.wait(
            futures, timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
        )

        for future in done_futures:
            chunk_res = future.result()
            for e_, a_, res in chunk_res:
                done += 1
                if res['ok']:
                    score = objective(res, mc_r=None)
                    candidates.append((score, e_, a_, res))
                frac = (done / N) * phase1_weight
                progress_cb(
                    f"Phase 1: Coarse search ({done}/{N}) "
                    f"elev={e_:.0f}° azi={a_:.0f}°", frac)

        time.sleep(0.005)

    elapsed = time.perf_counter() - _t_start
    print(f"[BENCHMARK] Phase 1 Grid Search evaluated {N} combinations in {elapsed:.3f} seconds using {os.cpu_count()} workers.")

    if not candidates:
        raise ValueError(
            'Simulation failed for all candidates.\n'
            'Please check your parameters.')

    candidates.sort(key=lambda x: -x[0] if math.isfinite(x[0]) else float('inf'))

    # Phase 2: MC verification on top-5
    top_n    = min(5, len(candidates))
    mc_trials = 8
    best      = None

    for i in range(top_n):
        if stop_flag.is_set():
            raise RuntimeError('cancelled')
        _, e_, a_, res = candidates[i]
        mc_r, succ = _monte_carlo_r90(
            e_, a_, base_params,
            n_trials=mc_trials,
            landing_prob=landing_prob,
            wind_uncertainty=wind_uncertainty,
            thrust_uncertainty=thrust_uncertainty,
            stop_flag=stop_flag)
        score       = objective(res, mc_r=mc_r)
        phase2_span = (1 - phase1_weight) * 0.75
        prog_frac   = phase1_weight + (i + 1) / top_n * phase2_span
        progress_cb(
            f"Phase 2: MC verification ({i+1}/{top_n}) "
            f"elev={e_:.0f}° azi={a_:.0f}°  "
            f"MC r={mc_r:.1f}m (≤{r_max:.1f}m?)", prog_frac)
        if math.isfinite(score):
            if best is None or score > best[0]:
                best = (score, e_, a_, res, mc_r)

    if best is None:
        raise ValueError(
            f'No candidate satisfies constraint '
            f'(r + MC {landing_prob}% circle ≤ {r_max:.1f} m).\n'
            'Try increasing r_max or adjusting wind / airframe settings.')

    score, best_e, best_a, _, best_mc_r = best

    if stop_flag.is_set():
        raise RuntimeError('cancelled')

    # Retrieve full trajectory arrays for the best result
    # since we stripped them out during the parallel grid search
    full_best_res = simulate_once(best_e, best_a, base_params)

    # Phase 3: final MC
    progress_cb(
        f"Phase 3: Final MC analysis (elev={best_e:.1f}° azi={best_a:.1f}°)", 0.9)
    final_mc_trials = 16
    final_mc_r, final_mc_succ = _monte_carlo_r90(
        best_e, best_a, base_params,
        n_trials=final_mc_trials,
        landing_prob=landing_prob,
        wind_uncertainty=wind_uncertainty,
        thrust_uncertainty=thrust_uncertainty,
        stop_flag=stop_flag)
    if stop_flag.is_set():
        raise RuntimeError('cancelled')

    reported_mc_r = final_mc_r if math.isfinite(final_mc_r) else best_mc_r
    progress_cb('Phase 3: Complete', 1.0)

    return {
        'mode':        mode,
        'r_max':       r_max,
        'elev':        best_e,
        'azi':         best_a,
        'score':       score,
        'result':      full_best_res,
        'mc_r':        reported_mc_r,
        'mc_success':  final_mc_succ,
        'mc_trials':   final_mc_trials,
    }


def _monte_carlo_r90(
    elev: float, azi: float,
    base_params: dict,
    n_trials: int,
    landing_prob: int,
    wind_uncertainty: float,
    thrust_uncertainty: float,
    stop_flag: Optional[threading.Event] = None,
) -> tuple[float, float]:
    """Run n_trials perturbed simulations; return (r_p, success_rate).

    r_p is the ``landing_prob``-th percentile of impact distances.
    Returns (inf, 0) if all trials fail.
    """
    distances: list[float] = []
    succeeded = 0
    rng = _random_mod.Random()
    wu  = max(wind_uncertainty, 0.0)
    tu  = max(thrust_uncertainty, 0.0)
    raw_thrust = base_params['thrust_data']

    for _ in range(n_trials):
        if stop_flag is not None and stop_flag.is_set():
            break
        u_prof, v_prof, _, _, _ = build_perturbed_wind_prof(base_params, rng, wu)
        thrust_scale     = max(0.1, 1.0 + rng.gauss(0.0, tu))
        perturbed_thrust = [[t, T * thrust_scale] for (t, T) in raw_thrust]

        p = dict(base_params)
        p['wind_u_prof'] = u_prof
        p['wind_v_prof'] = v_prof
        p['thrust_data'] = perturbed_thrust

        r = simulate_once(elev, azi, p)
        if r['ok']:
            distances.append(math.hypot(r['impact_x'], r['impact_y']))
            succeeded += 1

    if not distances:
        return float('inf'), 0.0
    distances.sort()
    p_idx = max(0, min(
        len(distances) - 1,
        int(round((landing_prob / 100.0) * len(distances))) - 1))
    return distances[p_idx], succeeded / n_trials


# ── Phase-1 helpers ───────────────────────────────────────────────────────────

def p1_params_at_wind(base_params: dict, mu_surf: float) -> dict:
    """Return a params dict where the wind profile is scaled so surface speed = mu_surf."""
    base_u = base_params.get('wind_u_prof', [(0, 0.0)])
    base_v = base_params.get('wind_v_prof', [(0, 0.0)])

    spd0 = math.hypot(base_u[0][1], base_v[0][1]) if base_u else 0.0
    ratio = mu_surf / max(spd0, 1e-6)

    u_prof = [(z, u * ratio) for z, u in base_u]
    v_prof = [(z, v * ratio) for z, v in base_v]

    p = dict(base_params)
    p['wind_u_prof'] = u_prof
    p['wind_v_prof'] = v_prof
    return p


def p1_mc_points(
    elev: float, azi: float,
    base_params: dict,
    mu: float, sigma: float,
    n: int,
    stop_flag: Optional[threading.Event] = None,
) -> list[tuple[float, float]]:
    """Run n Monte Carlo sims and return landing scatter points.

    Wind speed is drawn from N(mu, sigma); the upper-level speed is
    scaled proportionally from the nominal ratio.

    Returns list of (impact_x, impact_y) for successful runs only.
    """
    rng        = _random_mod.Random()
    points: list[tuple[float, float]] = []

    # Optimization: hoist invariant scaling parameters outside the loop
    # Scale the nominal profile to mu
    p_scaled = p1_params_at_wind(base_params, mu)

    # Perturb with sigma as fraction of mu
    wu = sigma / max(mu, 1e-6)

    for _ in range(n):
        if stop_flag is not None and stop_flag.is_set():
            break
        
        u_prof, v_prof, _, _, _ = build_perturbed_wind_prof(p_scaled, rng, wu)
        
        p = dict(base_params)
        p['wind_u_prof'] = u_prof
        p['wind_v_prof'] = v_prof
        r = simulate_once(elev, azi, p)
        if r['ok']:
            points.append((r['impact_x'], r['impact_y']))

    return points


def p1_ellipse_params(
    points: list[tuple[float, float]],
) -> tuple[float, float, Any, Any]:
    """Fit a 2-D covariance ellipse to the MC landing scatter.

    Returns:
        (cx, cy, eigvals, eigvecs)
        eigvals / eigvecs are the output of np.linalg.eigh(cov)
        (ascending eigenvalue order).
    """
    pts = np.array(points)
    cx  = float(np.mean(pts[:, 0]))
    cy  = float(np.mean(pts[:, 1]))
    cov = np.cov(pts.T)
    # Regularise: guarantee strict positive-definiteness for collinear scatter
    cov = cov + np.eye(2) * 1e-6
    eigvals, eigvecs = np.linalg.eigh(cov)
    return cx, cy, eigvals, eigvecs


def p1_ellipse_breaches_circle(
    cx: float, cy: float,
    eigvals: Any, eigvecs: Any,
    R: float,
    n_pts: int = 180,
) -> bool:
    """Return True if the 90 % error ellipse extends beyond circle radius R.

    Uses ``CHI2_90`` (chi²(2, 90 %)) for the ellipse scale factor.
    The check is done by sampling n_pts boundary points of the ellipse
    and testing whether any fall outside the circle.

    Args:
        cx, cy:   Ellipse centre offset from origin (metres).
        eigvals:  Eigenvalues from p1_ellipse_params (ascending order).
        eigvecs:  Eigenvectors from p1_ellipse_params.
        R:        Target circle radius (metres).
        n_pts:    Number of boundary samples (default 180).
    """
    K   = math.sqrt(CHI2_90)
    a   = K * math.sqrt(max(float(eigvals[1]), 0.0))   # major semi-axis
    b   = K * math.sqrt(max(float(eigvals[0]), 0.0))   # minor semi-axis
    ang = math.atan2(float(eigvecs[1, 1]), float(eigvecs[0, 1]))
    ca, sa = math.cos(ang), math.sin(ang)

    for i in range(n_pts):
        t  = 2.0 * math.pi * i / n_pts
        xe = a * math.cos(t) * ca - b * math.sin(t) * sa
        ye = a * math.cos(t) * sa + b * math.cos(t) * ca
        if math.hypot(cx + xe, cy + ye) > R:
            return True
    return False


# ── Phase-1 main worker ───────────────────────────────────────────────────────

@dataclass
class Phase1Result:
    """Immutable result container for a completed Phase-1 analysis."""

    best_elev:             float
    best_azi:              float
    apogee_m:              float
    nominal_cx:            float
    nominal_cy:            float
    mu_nominal:            float
    mu_max:                float
    sigma_max:             float
    ellipse_a:             float
    ellipse_b:             float
    ellipse_angle_rad:     float
    ellipse_scale_per_sigma: float
    dcx_dmu:               float
    dcy_dmu:               float
    target_radius_m:       float
    best_score:            float
    mode:                  str


def run_phase1(
    base_params: dict,
    target_r: float,
    mode: str,
    stop_flag: threading.Event,
    progress_cb: Callable[[str, float], None],
) -> Phase1Result:
    """Run the full 5-step Phase-1 analysis.

    Step 1 — Grid search (elev 60-90°/6° step, azi 0-345°/15° step).
    Step 2 — Nominal MC: 40-run 90 % error ellipse at nominal wind.
    Step 3 — Landing sensitivity d(cx,cy)/dmu via central difference.
    Step 4 — Binary search for mu_max (deterministic, sigma=0).
    Step 5 — Binary search for sigma_max (MC ellipse containment).

    Args:
        base_params:  Simulation params dict.
        target_r:     Target landing-zone radius (metres).
        mode:         'Altitude Competition', 'Precision Landing', or
                      'Winged Hover'.
        stop_flag:    threading.Event; set to cancel.
        progress_cb:  Callable(message, fraction[0..1]).

    Returns:
        :class:`Phase1Result` on success.

    Raises:
        RuntimeError: on cancellation (message == 'cancelled').
        ValueError:   on search failure with a user-readable message.
    """

    def prog(msg: str, frac: float) -> None:
        progress_cb(msg, frac)

    base_u = base_params.get('wind_u_prof', [(0, 0.0)])
    base_v = base_params.get('wind_v_prof', [(0, 0.0)])
    mu_nom = math.hypot(base_u[0][1], base_v[0][1]) if base_u else 0.0
    wu = 0.08
    sigma_nom = wu * max(mu_nom, 1.0)

    # ── Step 1: Grid search ───────────────────────────────────────────────────
    elev_grid    = list(range(60, 91, 6))   # 60, 66, …, 90
    azi_grid     = list(range(0, 360, 15))  # 24 azimuths
    use_r_filter = (mode != 'Precision Landing')
    N        = len(elev_grid) * len(azi_grid)
    done, cands  = 0, []
    prog(f'Step 1/5  Grid search (0/{N})', 0.0)

    import concurrent.futures
    import os
    import time

    _t_start = time.perf_counter()

    from .pool_manager import get_global_pool
    executor = get_global_pool()
    p_nom = p1_params_at_wind(base_params, mu_nom)
    
    configs = []
    for e in elev_grid:
        for a in azi_grid:
            configs.append((e, a))

    _n_workers = max(1, (os.cpu_count() or 2) - 2)
    chunksize = max(1, N // _n_workers)
    chunks = [configs[i:i + chunksize] for i in range(0, N, chunksize)]
    futures = [executor.submit(_grid_search_chunk, chunk, p_nom) for chunk in chunks]

    while futures:
        if stop_flag.is_set():
            for f in futures:
                f.cancel()
            raise RuntimeError('cancelled')
            
        done_futures, futures = concurrent.futures.wait(
            futures, timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
        )

        for future in done_futures:
            try:
                chunk_res = future.result()
                for e_, a_, res in chunk_res:
                    done += 1
                    if res['ok']:
                        if not use_r_filter or res['r_horiz'] <= target_r:
                            score = p1_objective_score(res, mode)
                            cands.append((score, e_, a_, res))
                    prog(f'Step 1/5  Grid ({done}/{N})  e={e_}° a={a_}°',
                         done / N * 0.25)
            except Exception as e:
                print(f"Optimization Grid Chunk Error: {e}", flush=True)

        time.sleep(0.005)

    elapsed = time.perf_counter() - _t_start
    print(f"[BENCHMARK] Phase 1 Grid Search evaluated {N} combinations in {elapsed:.3f} seconds using {os.cpu_count()} workers.")

    if not cands:
        raise ValueError(
            f'No trajectory satisfies r_horiz ≤ {target_r:.0f} m.\n'
            'Check parameters (r_max, wind speed, airframe specs).')

    cands.sort(key=lambda x: -x[0])
    _, best_e, best_a, best_res = cands[0]
    best_apogee = best_res['apogee_m']
    prog(f'Step 1/5  done  best elev={best_e}° azi={best_a}°'
         f'  apogee={best_apogee:.1f} m', 0.26)

    # ── Step 2: Nominal MC ────────────────────────────────────────────────────
    N_NOM     = 40
    sigma_nom = max(mu_nom * 0.08, 0.3)
    prog(f'Step 2/5  Nominal MC  ({N_NOM} runs, σ={sigma_nom:.2f} m/s)…', 0.28)

    pts_nom = p1_mc_points(
        best_e, best_a, base_params, mu_nom, sigma_nom,
        n=N_NOM, stop_flag=stop_flag)
    if stop_flag.is_set():
        raise RuntimeError('cancelled')
    if len(pts_nom) < 6:
        raise ValueError(
            'Nominal MC: insufficient samples (< 6). Check parameters.')

    cx_nom, cy_nom, eig_v, eig_vc = p1_ellipse_params(pts_nom)
    K              = math.sqrt(CHI2_90)
    a_nom          = K * math.sqrt(max(float(eig_v[1]), 0.0))
    b_nom          = K * math.sqrt(max(float(eig_v[0]), 0.0))
    angle_rad      = math.atan2(float(eig_vc[1, 1]), float(eig_vc[0, 1]))
    scale_per_sigma = (a_nom / sigma_nom) if sigma_nom > 0 else 10.0
    prog('Step 2/5  Nominal MC done', 0.42)

    # ── Step 3: Wind sensitivity d(cx, cy)/dmu ────────────────────────────────
    prog('Step 3/5  Wind sensitivity…', 0.44)
    dmu  = max(mu_nom * 0.15, 0.5)
    p_hi = p1_params_at_wind(base_params, mu_nom + dmu)
    p_lo = p1_params_at_wind(base_params, max(mu_nom - dmu, 0.1))
    r_hi = simulate_once(best_e, best_a, p_hi)
    r_lo = simulate_once(best_e, best_a, p_lo)
    if r_hi['ok'] and r_lo['ok']:
        dcx_dmu = (r_hi['impact_x'] - r_lo['impact_x']) / (2 * dmu)
        dcy_dmu = (r_hi['impact_y'] - r_lo['impact_y']) / (2 * dmu)
    else:
        dcx_dmu = dcy_dmu = 0.0
    prog('Step 3/5  Sensitivity done', 0.50)

    # ── Step 4: Binary search μ_max (deterministic, σ = 0) ───────────────────
    prog('Step 4/5  μ_max search…', 0.52)
    mu_lo_s, mu_hi_s = mu_nom, mu_nom * 8.0
    for _ in range(22):
        if stop_flag.is_set():
            raise RuntimeError('cancelled')
        if mu_hi_s - mu_lo_s < 0.05:
            break
        mu_mid = (mu_lo_s + mu_hi_s) / 2.0
        p_m    = p1_params_at_wind(base_params, mu_mid)
        r_m    = simulate_once(best_e, best_a, p_m)
        if r_m['ok'] and r_m['r_horiz'] <= target_r:
            mu_lo_s = mu_mid
        else:
            mu_hi_s = mu_mid
    mu_max = mu_lo_s
    prog(f'Step 4/5  μ_max = {mu_max:.2f} m/s', 0.70)

    # ── Step 5: Binary search σ_max (MC ellipse containment) ─────────────────
    prog('Step 5/5  σ_max search (MC)…', 0.72)
    N_SIG = 20
    sig_lo, sig_hi = 0.0, max(mu_nom * 3.0, 5.0)

    def _sigma_ok(sig: float) -> bool:
        if stop_flag.is_set():
            return False
        pts = p1_mc_points(
            best_e, best_a, base_params, mu_nom, sig,
            n=N_SIG, stop_flag=stop_flag)
        if len(pts) < 6:
            return False
        cx_m = float(np.mean([p[0] for p in pts]))
        cy_m = float(np.mean([p[1] for p in pts]))
        _, _, ev, evc = p1_ellipse_params(pts)
        # Constraint: landing centre + 90 % error ellipse must fit inside target_r
        return not p1_ellipse_breaches_circle(cx_m, cy_m, ev, evc, target_r)

    if _sigma_ok(sig_hi):
        sigma_max = sig_hi
    else:
        for _ in range(15):
            if stop_flag.is_set():
                raise RuntimeError('cancelled')
            if sig_hi - sig_lo < 0.05:
                break
            sig_mid = (sig_lo + sig_hi) / 2.0
            if _sigma_ok(sig_mid):
                sig_lo = sig_mid
            else:
                sig_hi = sig_mid
        sigma_max = sig_lo

    prog(f'Step 5/5  σ_max = {sigma_max:.2f} m/s', 0.99)

    # ── Compile result ────────────────────────────────────────────────────────
    if mode == 'Precision Landing':
        display_score = best_res['r_horiz']
    elif mode == 'Winged Hover':
        display_score = best_res['hang_time']
    else:
        display_score = best_res['apogee_m']

    prog('Phase 1 complete ✓', 1.0)

    return Phase1Result(
        best_elev             = float(best_e),
        best_azi              = float(best_a),
        apogee_m              = float(best_apogee),
        nominal_cx            = float(cx_nom),
        nominal_cy            = float(cy_nom),
        mu_nominal            = float(mu_nom),
        mu_max                = float(mu_max),
        sigma_max             = float(sigma_max),
        ellipse_a             = float(a_nom),
        ellipse_b             = float(b_nom),
        ellipse_angle_rad     = float(angle_rad),
        ellipse_scale_per_sigma = float(scale_per_sigma),
        dcx_dmu               = float(dcx_dmu),
        dcy_dmu               = float(dcy_dmu),
        target_radius_m       = float(target_r),
        best_score            = float(display_score),
        mode                  = mode,
    )
