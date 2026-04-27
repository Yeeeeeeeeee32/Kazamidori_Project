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

run_phase1(base_params, target_r, mode, sim_fn, stop_flag,
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

import numpy as np

# Relative imports within the package
from .simulation import simulate_once


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
) -> tuple[list, list]:
    """Return (u_prof, v_prof) for RocketPy custom_atmosphere.

    Altitude 0 is forced to zero wind (below the anemometer).
    """
    alpha = _hellmann_alpha(v_surf, z_surf, v_upper, z_upper)

    def _speed(z: float) -> float:
        if z <= 0:
            return 0.0
        if z <= z_surf:
            return v_surf * (z / z_surf) ** alpha
        if z >= z_upper:
            return v_upper
        return v_surf * (z / z_surf) ** alpha

    def _dir(z: float) -> float:
        if z <= z_surf:
            return dir_surf_deg
        if z >= z_upper:
            return dir_upper_deg
        frac = (z - z_surf) / (z_upper - z_surf)
        diff = ((dir_upper_deg - dir_surf_deg + 180.0) % 360.0) - 180.0
        return dir_surf_deg + frac * diff

    alts = sorted({0, 3, z_surf, 30, 100, 300, z_upper, 1000, 5000})
    u_prof: list = [(0, 0.0)]
    v_prof: list = [(0, 0.0)]
    for z in alts:
        if z == 0:
            continue
        spd = _speed(z)
        rad = math.radians(_dir(z))
        u_prof.append((z, -spd * math.sin(rad)))
        v_prof.append((z, -spd * math.cos(rad)))
    return u_prof, v_prof


# ── Perturbed wind profile for MC ────────────────────────────────────────────

def build_perturbed_wind_prof(
    params: dict,
    rng: _random_mod.Random,
    wu: float,
) -> tuple[list, list, float, float, list]:
    """Build a stochastically perturbed wind profile for one MC trial.

    Applies global speed/direction uncertainty to both surface and
    upper-level anchors, then adds independent per-layer Gaussian noise
    to model upper-level turbulence.

    Args:
        params: Simulation params dict (must have surf_spd, up_spd,
                surf_dir, up_dir).
        rng:    Random instance (caller-owned so seeds are reproducible).
        wu:     Wind-speed fractional uncertainty (e.g. 0.10 = ±10 %).

    Returns:
        (u_prof, v_prof, surf_spd, up_spd, spd_profile)
        where spd_profile is [(alt_m, speed_m_s), …] for spaghetti plots.
    """
    base_surf  = max(params['surf_spd'], 0.1)
    base_up    = max(params['up_spd'],   0.1)
    dir_sigma  = wu * 60.0

    surf_spd = max(0.0, rng.gauss(params['surf_spd'], wu * base_surf))
    up_spd   = max(0.0, rng.gauss(params['up_spd'],   wu * base_up))
    surf_dir = params['surf_dir'] + rng.gauss(0.0, dir_sigma)
    up_dir   = params['up_dir']   + rng.gauss(0.0, dir_sigma)

    u_prof, v_prof = build_wind_profile(
        surf_spd, surf_dir, 3.0, up_spd, up_dir, 100.0)

    layer_sigma = wu * base_surf * 0.35
    if layer_sigma > 1e-6:
        u_prof = [(z, u + rng.gauss(0.0, layer_sigma)) for z, u in u_prof]
        v_prof = [(z, v + rng.gauss(0.0, layer_sigma)) for z, v in v_prof]

    spd_prof = [(z_u, math.sqrt(u ** 2 + v ** 2))
                for (z_u, u), (_, v) in zip(u_prof, v_prof)]

    return u_prof, v_prof, surf_spd, up_spd, spd_prof


# ── Objective helpers ─────────────────────────────────────────────────────────

def p1_objective_score(res: dict, mode: str) -> float:
    """Return the scalar objective for a simulation result in the given mode.

    Higher is always better (even for Precision Landing where we return
    the negative landing radius).
    """
    if mode == 'Altitude Competition':
        return res['apogee_m']
    elif mode == 'Precision Landing':
        return -res['r_horiz']           # lower radius → higher score
    elif mode == 'Winged Hover':
        return res['hang_time']
    else:
        return res['apogee_m']


# ── Optimiser (from _optimize_worker) ────────────────────────────────────────

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

    Args:
        mode:               One of 'Precision Landing', 'Altitude Competition',
                            'Winged Hover'.
        base_params:        Params dict as produced by _gather_sim_params.
        r_max:              Maximum landing radius constraint (metres).
        landing_prob:       Confidence percentile (e.g. 90).
        wind_uncertainty:   Fractional wind speed uncertainty.
        thrust_uncertainty: Fractional thrust uncertainty.
        stop_flag:          threading.Event; set to request cancellation.
        progress_cb:        Callback(message: str, fraction: float).

    Returns:
        dict with keys: mode, r_max, elev, azi, score, result (sim dict),
        mc_r, mc_success, mc_trials.

    Raises:
        ValueError: if mode is unknown or no valid candidate is found.
        RuntimeError: on cancellation.
    """
    if mode == "Precision Landing":
        elev_grid = [60, 66, 72, 78, 84, 90]
        azi_grid  = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

        def objective(res, mc_r=None):
            if not res['ok']:
                return float('-inf')
            r = res['r_horiz']
            if mc_r is None:
                return float('-inf') if r > r_max else (r_max - r) + res['hang_time']
            return float('-inf') if r + mc_r > r_max else (r_max - r) + res['hang_time']

    elif mode == "Altitude Competition":
        elev_grid = [60, 66, 72, 78, 84, 90]
        azi_grid  = [0, 45, 90, 135, 180, 225, 270, 315]

        def objective(res, mc_r=None):
            if not res['ok']:
                return float('-inf')
            r = res['r_horiz']
            if mc_r is None:
                return float('-inf') if r > r_max else res['apogee_m']
            return float('-inf') if r + mc_r > r_max else res['apogee_m']

    elif mode == "Winged Hover":
        elev_grid = [60, 66, 72, 78, 84, 90]
        azi_grid  = [0, 45, 90, 135, 180, 225, 270, 315]

        def objective(res, mc_r=None):
            if not res['ok']:
                return float('-inf')
            r = res['r_horiz']
            if mc_r is None:
                return float('-inf') if r > r_max else res['hang_time']
            return float('-inf') if r + mc_r > r_max else res['hang_time']

    else:
        raise ValueError(f'Unknown mode: {mode}')

    candidates = []
    total       = len(elev_grid) * len(azi_grid)
    done        = 0
    phase1_weight = 0.6

    progress_cb(f"Phase 1: Coarse search (0/{total})", 0.0)

    for e_ in elev_grid:
        for a_ in azi_grid:
            if stop_flag.is_set():
                raise RuntimeError('cancelled')
            res = simulate_once(e_, a_, base_params)
            done += 1
            if res['ok']:
                score = objective(res, mc_r=None)
                candidates.append((score, e_, a_, res))
            frac = (done / total) * phase1_weight
            progress_cb(
                f"Phase 1: Coarse search ({done}/{total}) "
                f"elev={e_:.0f}° azi={a_:.0f}°", frac)

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

    score, best_e, best_a, best_res, best_mc_r = best

    if stop_flag.is_set():
        raise RuntimeError('cancelled')

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
        'result':      best_res,
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
    """Return a copy of base_params with the surface wind speed set to mu_surf.

    The upper-level wind speed is scaled proportionally.
    """
    ratio    = mu_surf / max(base_params['surf_spd'], 1e-6)
    mu_upper = base_params['up_spd'] * ratio
    u_prof, v_prof = build_wind_profile(
        mu_surf, base_params['surf_dir'], 3.0,
        mu_upper, base_params['up_dir'], 100.0,
    )
    p = dict(base_params)
    p['wind_u_prof'] = u_prof
    p['wind_v_prof'] = v_prof
    p['surf_spd']    = mu_surf
    p['up_spd']      = mu_upper
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
    mu_nominal = max(base_params['surf_spd'], 1e-6)
    points: list[tuple[float, float]] = []

    for _ in range(n):
        if stop_flag is not None and stop_flag.is_set():
            break
        surf_spd = max(0.0, rng.gauss(mu, sigma))
        ratio    = surf_spd / mu_nominal
        up_spd   = max(0.0, rng.gauss(base_params['up_spd'] * ratio, sigma * 0.5))
        u_prof, v_prof = build_wind_profile(
            surf_spd, base_params['surf_dir'], 3.0,
            up_spd,   base_params['up_dir'],   100.0,
        )
        p = dict(base_params)
        p['wind_u_prof'] = u_prof
        p['wind_v_prof'] = v_prof
        p['surf_spd']    = surf_spd
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
    eigvals, eigvecs = np.linalg.eigh(cov)
    return cx, cy, eigvals, eigvecs


def p1_ellipse_breaches_circle(
    cx: float, cy: float,
    eigvals: Any, eigvecs: Any,
    R: float,
    n_pts: int = 180,
) -> bool:
    """Return True if the 90 % error ellipse extends beyond circle radius R.

    Uses chi²(2, 90 %) = 4.605 for the ellipse scale factor.
    The check is done by sampling n_pts boundary points of the ellipse
    and testing whether any fall outside the circle.

    Args:
        cx, cy:   Ellipse centre offset from origin (metres).
        eigvals:  Eigenvalues from p1_ellipse_params (ascending order).
        eigvecs:  Eigenvectors from p1_ellipse_params.
        R:        Target circle radius (metres).
        n_pts:    Number of boundary samples (default 180).
    """
    K   = math.sqrt(4.605)   # chi²(2, 90 %)
    a   = K * math.sqrt(max(float(eigvals[1]), 0.0))   # major semi-axis
    b   = K * math.sqrt(max(float(eigvals[0]), 0.0))   # minor semi-axis
    ang = math.atan2(float(eigvecs[1, 1]), float(eigvecs[0, 1]))
    ca, sa = math.cos(ang), math.sin(ang)

    for i in range(n_pts):
        t  = 2.0 * math.pi * i / n_pts
        xe = a * math.cos(t) * ca - b * math.sin(t) * sa
        ye = a * math.cos(t) * sa + b * math.sin(t) * ca
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

    mu_nom = base_params['surf_spd']

    # ── Step 1: Grid search ───────────────────────────────────────────────────
    elev_grid    = list(range(60, 91, 6))   # 60, 66, …, 90
    azi_grid     = list(range(0, 360, 15))  # 24 azimuths
    use_r_filter = (mode != 'Precision Landing')
    total        = len(elev_grid) * len(azi_grid)
    done, cands  = 0, []
    prog(f'Step 1/5  Grid search (0/{total})', 0.0)

    for e in elev_grid:
        for a in azi_grid:
            if stop_flag.is_set():
                raise RuntimeError('cancelled')
            p   = p1_params_at_wind(base_params, mu_nom)
            res = simulate_once(e, a, p)
            done += 1
            if res['ok']:
                if not use_r_filter or res['r_horiz'] <= target_r:
                    score = p1_objective_score(res, mode)
                    cands.append((score, e, a, res))
            prog(f'Step 1/5  Grid ({done}/{total})  e={e}° a={a}°',
                 done / total * 0.25)

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
    K              = math.sqrt(4.605)
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
