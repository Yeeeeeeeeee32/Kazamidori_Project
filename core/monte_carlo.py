"""
core/monte_carlo.py
Monte-Carlo statistical analysis for landing-zone visualisation.

Public API
----------
CHI2_2DOF : dict[int, float]
    Chi-squared (df=2) quantiles keyed by integer percentage.

chi2_scale(prob_pct) -> float
    Return sqrt(chi²(2, prob_pct/100)) for error-ellipse scaling.

run_mc_scatter(params, n_runs, wind_uncertainty, thrust_uncertainty,
               stop_flag=None) -> (scatter, wind_profiles)
    Run n_runs perturbed simulations; return landing scatter and
    spaghetti wind profiles.

compute_error_ellipse(scatter, prob_pct=90) -> dict | None
    Fit a 2-D covariance error ellipse to the scatter.
    Returns {'cx', 'cy', 'a', 'b', 'angle_rad'} or None if too few
    points.

compute_cep(scatter) -> float
    Compute the 50th-percentile distance from the origin (CEP).

compute_kde_contours(scatter, launch_lat, launch_lon,
                     conf_pct=90) -> list[(latlons, color, width)]
    Compute KDE probability-mass contours and convert to (lat, lon)
    polygons.  Returns an empty list if scipy is unavailable or the
    scatter is too small.

All functions are pure — no tkinter, no matplotlib rendering.
geo_math.offset_to_latlon is used for coordinate conversion.
"""

from __future__ import annotations

import math
import random as _random_mod
import threading
from typing import Any, Optional

import numpy as np

from .simulation import simulate_once
from .optimization import build_perturbed_wind_prof
from utils.geo_math import offset_to_latlon


# ── Chi-squared 2-DOF quantile table ─────────────────────────────────────────

# Values: chi2.ppf(p, df=2)  for p in {0.50, 0.68, 0.80, 0.85, 0.90, 0.95, 0.99}
CHI2_2DOF: dict[int, float] = {
    50: 1.386,
    68: 2.296,
    80: 3.219,
    85: 3.794,
    90: 4.605,
    95: 5.991,
    99: 9.210,
}


def chi2_scale(prob_pct: int) -> float:
    """Return sqrt(chi²(2, prob_pct/100)) for error-ellipse axis scaling.

    Falls back to the 90 % value for unknown percentages.
    """
    return math.sqrt(CHI2_2DOF.get(int(prob_pct), 4.605))


# ── MC scatter worker ─────────────────────────────────────────────────────────

def run_mc_scatter(
    params: dict,
    n_runs: int,
    wind_uncertainty: float,
    thrust_uncertainty: float,
    stop_flag: Optional[threading.Event] = None,
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    """Run n_runs Monte-Carlo simulations and return results.

    Each trial perturbs the wind profile and motor thrust independently.

    Args:
        params:              Simulation params dict (must contain elev, azi,
                             thrust_data, surf_spd, up_spd, surf_dir, up_dir).
        n_runs:              Number of MC trials.
        wind_uncertainty:    Fractional wind-speed uncertainty (e.g. 0.10).
        thrust_uncertainty:  Fractional thrust uncertainty (e.g. 0.05).
        stop_flag:           Optional threading.Event; set to abort early.

    Returns:
        (scatter, wind_profiles)
        scatter:       list of (impact_x, impact_y) — successful runs only.
        wind_profiles: list of [(alt_m, speed_m_s), …] — one per trial
                       (includes failed runs, for spaghetti visualisation).
    """
    scatter:       list[tuple[float, float]]       = []
    wind_profiles: list[list[tuple[float, float]]] = []

    rng        = _random_mod.Random()
    wu         = max(wind_uncertainty,  0.0)
    tu         = max(thrust_uncertainty, 0.0)
    raw_thrust = params['thrust_data']
    elev       = params['elev']
    azi        = params['azi']

    for _ in range(n_runs):
        if stop_flag is not None and stop_flag.is_set():
            break
        u_prof, v_prof, _, _, spd_prof = build_perturbed_wind_prof(
            params, rng, wu)
        thrust_scale = max(0.1, 1.0 + rng.gauss(0.0, tu))
        perturbed    = [[t, T * thrust_scale] for (t, T) in raw_thrust]

        p = dict(params)
        p['wind_u_prof'] = u_prof
        p['wind_v_prof'] = v_prof
        p['thrust_data'] = perturbed

        r = simulate_once(elev, azi, p)
        if r['ok']:
            scatter.append((r['impact_x'], r['impact_y']))
        wind_profiles.append(spd_prof)

    return scatter, wind_profiles


# ── Error ellipse ─────────────────────────────────────────────────────────────

def compute_error_ellipse(
    scatter: list[tuple[float, float]],
    prob_pct: int = 90,
) -> dict[str, float] | None:
    """Fit a 2-D covariance error ellipse to the MC landing scatter.

    Uses np.linalg.eigh on the full 2×2 covariance matrix.  A floor is
    applied to the minor semi-axis to prevent a degenerate line when the
    scatter is nearly collinear.

    Args:
        scatter:  list of (x, y) landing positions.
        prob_pct: Confidence percentage (default 90).

    Returns:
        dict with keys 'cx', 'cy', 'a', 'b', 'angle_rad', or None if
        fewer than 4 points are available.
    """
    if len(scatter) < 4:
        return None

    arr = np.array(scatter, dtype=float)
    cx  = float(arr[:, 0].mean())
    cy  = float(arr[:, 1].mean())
    cov = np.cov(arr[:, 0], arr[:, 1])

    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns eigenvalues in ascending order; index 1 = largest
    lam2, lam1 = float(eigvals[0]), float(eigvals[1])
    major_vec  = eigvecs[:, 1]
    angle_rad  = float(math.atan2(float(major_vec[1]), float(major_vec[0])))

    k = chi2_scale(prob_pct)
    a = k * math.sqrt(max(lam1, 0.0))
    b = k * math.sqrt(max(lam2, 0.0))
    # Floor: prevent degenerate near-zero minor axis
    b = max(b, max(0.5, a * 0.05))

    return {'cx': cx, 'cy': cy, 'a': a, 'b': b, 'angle_rad': angle_rad}


# ── CEP ───────────────────────────────────────────────────────────────────────

def compute_cep(scatter: list[tuple[float, float]]) -> float:
    """Return the CEP: 50th-percentile distance from the origin.

    The origin is the nominal launch site, not the mean landing point.
    Linear interpolation is used when the 50th percentile falls between
    two samples.

    Returns 0.0 if scatter is empty.
    """
    if not scatter:
        return 0.0
    dists = sorted(math.hypot(x, y) for x, y in scatter)
    mid   = (len(dists) - 1) / 2.0
    lo    = int(mid)
    hi    = min(lo + 1, len(dists) - 1)
    return dists[lo] + (mid - lo) * (dists[hi] - dists[lo])


# ── KDE contours ──────────────────────────────────────────────────────────────

def compute_kde_contours(
    scatter: list[tuple[float, float]],
    launch_lat: float,
    launch_lon: float,
    conf_pct: int = 90,
) -> list[tuple[list[tuple[float, float]], str, int]]:
    """Compute KDE probability-mass contours and convert to lat/lon polygons.

    Three contour levels are drawn: 50 %, 70 %, and *conf_pct* %.
    The outermost level always uses the configured confidence percentage.

    This function requires scipy.  If scipy is not installed, or if fewer
    than 5 points are provided, an empty list is returned.

    Args:
        scatter:    list of (x, y) landing positions (metres, East/North).
        launch_lat: Reference latitude for offset_to_latlon conversion.
        launch_lon: Reference longitude for offset_to_latlon conversion.
        conf_pct:   Outer contour confidence percentage (default 90).

    Returns:
        list of (latlons, hex_colour, border_width) tuples ready for
        direct use with tkintermapview set_polygon.
        latlons is a list of (lat, lon) tuples.
    """
    try:
        from scipy.stats import gaussian_kde
        import matplotlib.pyplot as _plt
        import numpy as _np
    except ImportError:
        return []

    if len(scatter) < 5:
        return []

    xs = _np.array([p[0] for p in scatter])
    ys = _np.array([p[1] for p in scatter])

    try:
        kde = gaussian_kde(_np.vstack([xs, ys]))
    except Exception:
        return []

    pad = max(xs.ptp(), ys.ptp(), 1.0) * 0.5
    gx  = _np.linspace(xs.min() - pad, xs.max() + pad, 120)
    gy  = _np.linspace(ys.min() - pad, ys.max() + pad, 120)
    GX, GY = _np.meshgrid(gx, gy)
    Z = kde(_np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)

    # Probability-mass thresholds: always 50 %, 70 %, + configured conf_pct
    z_flat   = Z.ravel()
    z_sorted = _np.sort(z_flat)[::-1]
    cumsum   = _np.cumsum(z_sorted)
    cumsum  /= cumsum[-1]

    outer_frac = max(min(conf_pct / 100.0, 0.999), 0.501)
    levels_pm  = sorted({0.50, 0.70, outer_frac})

    level_vals: list[float] = []
    for pm in levels_pm:
        idx = _np.searchsorted(cumsum, pm)
        idx = min(idx, len(z_sorted) - 1)
        level_vals.append(float(z_sorted[idx]))

    # Deduplicate while preserving order
    seen: set = set()
    unique_vals: list[float] = []
    for v in level_vals:
        key = round(v, 12)
        if key not in seen:
            seen.add(key)
            unique_vals.append(v)

    if len(unique_vals) < 2:
        return []

    # Extract contour paths via a temporary off-screen figure
    fig_tmp, ax_tmp = _plt.subplots()
    cs = ax_tmp.contour(GX, GY, Z, levels=sorted(unique_vals))
    _plt.close(fig_tmp)

    sorted_lv = sorted(unique_vals)
    # Innermost (50 %) = brightest; outermost (conf_pct %) = dimmest
    palette = ['#ff6600', '#ff9900', '#ffcc00', '#ffe066']
    widths  = [3, 2, 1, 1]
    lv_style = {
        lv: (palette[min(i, len(palette) - 1)],
             widths[min(i, len(widths) - 1)])
        for i, lv in enumerate(sorted_lv[::-1])
    }

    contours: list[tuple[list[tuple[float, float]], str, int]] = []
    for collection, lv in zip(cs.collections, sorted_lv):
        col, bw = lv_style.get(lv, ('#ffcc00', 1))
        for path in collection.get_paths():
            verts = path.vertices
            if len(verts) < 3:
                continue
            latlons = [
                offset_to_latlon(launch_lat, launch_lon,
                                 float(v[0]), float(v[1]))
                for v in verts
            ]
            contours.append((latlons, col, bw))

    return contours
