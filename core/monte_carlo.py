"""
core/monte_carlo.py
Monte-Carlo statistical analysis for landing-zone dispersion.

Pure mathematical engine — no geographic coordinates, no rendering
concerns.  All inputs and outputs use the local metric East-North frame
with the launch point at the origin (0, 0).

Public API
----------
CHI2_2DOF : dict[int, float]
    Chi-squared (df=2) quantiles keyed by integer percentage.

chi2_scale(prob_pct) -> float
    Return sqrt(chi²(2, prob_pct/100)) for error-ellipse axis scaling.

run_mc_scatter(params, n_runs, wind_uncertainty, thrust_uncertainty,
               stop_flag=None) -> (scatter, wind_profiles)
    Run n_runs perturbed simulations; return landing scatter and
    spaghetti wind profiles.  The entire upper-air wind profile is
    perturbed at every altitude level for each trial.

compute_error_ellipse(scatter, prob_pct=90) -> dict | None
    Fit a 2-D covariance error ellipse to the scatter.
    Returns {'cx', 'cy', 'a', 'b', 'angle_rad'} in metres/radians,
    or None if fewer than 4 points are available.

compute_cep(scatter) -> float
    Return the CEP: 50th-percentile distance from the scatter centroid.
    Returns 0.0 for empty scatter.

compute_cep_circle(scatter, n=36) -> dict | None
    Compute the CEP circle and return it as a metric polygon.
    Returns {'cx_m', 'cy_m', 'radius_m', 'points_m'} or None.

compute_kde_contours(scatter, conf_pct=90) -> list[dict]
    Compute KDE probability-mass contours entirely in the metric frame.
    Returns a list of contour dicts (outer → inner), each containing:
        'points_m'  — list of (x_east_m, y_north_m) polygon vertices
        'prob_frac' — probability mass fraction (e.g. 0.90)
        'label'     — str like '90%' for the primary polygon at each
                       level, None for secondary disconnected islands

COORDINATE CONTRACT
-------------------
All functions receive *scatter* as a list of (x_east_m, y_north_m)
pairs in **metres** measured from the launch point.  This matches the
impact_x / impact_y values returned directly by simulate_once.

Geographic conversion (metres → lat/lon) is entirely the responsibility
of the UI layer.  This module has zero knowledge of geographic coordinates
and zero knowledge of how results are rendered.
"""

from __future__ import annotations

import math
import random as _random_mod
import threading
import warnings
from typing import Optional

import numpy as np

from .simulation import simulate_once


# ── Chi-squared 2-DOF quantile table ─────────────────────────────────────────

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


# ── Internal geometry helpers ─────────────────────────────────────────────────

def _circle_points_m(
    cx: float,
    cy: float,
    radius: float,
    n: int = 36,
) -> list[tuple[float, float]]:
    """Return *n* (x, y) vertices approximating a circle in metric space."""
    step = 2.0 * math.pi / n
    return [
        (cx + radius * math.cos(step * i),
         cy + radius * math.sin(step * i))
        for i in range(n)
    ]


# ── Wind profile perturbation ─────────────────────────────────────────────────

def _perturb_wind_profile(
    u_prof: list[tuple[float, float]],
    v_prof: list[tuple[float, float]],
    rng: _random_mod.Random,
    wind_uncertainty: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
    """Perturb a wind profile at every altitude level with two-component noise.

    *Global layer* (synoptic variability):
        A single speed-scale factor and direction rotation are sampled
        once per trial and applied uniformly across all altitude levels.

    *Local layer* (mesoscale turbulence):
        Independent additive Gaussian noise at each level, scaled to the
        local wind speed so jet-stream layers are perturbed proportionally.

    Args:
        u_prof:           list of (alt_m, u_m_s) — east wind component.
        v_prof:           list of (alt_m, v_m_s) — north wind component.
        rng:              seeded Random instance.
        wind_uncertainty: fractional 1-σ uncertainty (e.g. 0.10 = ±10 %).

    Returns:
        (u_perturbed, v_perturbed, speed_profile)
        speed_profile is [(alt_m, speed_m_s), …] for spaghetti plots.
    """
    if not u_prof or not v_prof:
        return list(u_prof), list(v_prof), []

    wu = max(wind_uncertainty, 0.0)

    speed_factor = max(0.05, 1.0 + rng.gauss(0.0, wu))
    dir_rot      = rng.gauss(0.0, wu * math.pi / 6.0)
    cos_r, sin_r = math.cos(dir_rot), math.sin(dir_rot)

    u_new:   list[tuple[float, float]] = []
    v_new:   list[tuple[float, float]] = []
    spd_out: list[tuple[float, float]] = []

    for (alt_u, u_nom), (_, v_nom) in zip(u_prof, v_prof):
        u_g = (u_nom * cos_r - v_nom * sin_r) * speed_factor
        v_g = (u_nom * sin_r + v_nom * cos_r) * speed_factor

        local_spd = math.hypot(u_nom, v_nom)
        sigma     = wu * max(local_spd, 1.0) * 0.30
        u_p = u_g + rng.gauss(0.0, sigma)
        v_p = v_g + rng.gauss(0.0, sigma)

        u_new.append((alt_u, u_p))
        v_new.append((alt_u, v_p))
        spd_out.append((alt_u, math.hypot(u_p, v_p)))

    return u_new, v_new, spd_out


# ── MC scatter ────────────────────────────────────────────────────────────────

def run_mc_scatter(
    params: dict,
    n_runs: int,
    wind_uncertainty: float,
    thrust_uncertainty: float,
    stop_flag: Optional[threading.Event] = None,
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    """Run n_runs Monte-Carlo simulations and return landing scatter.

    Each trial independently perturbs the complete upper-air wind profile
    (every altitude level) and the motor thrust curve.

    Args:
        params:              Simulation params dict.  Must contain
                             wind_u_prof, wind_v_prof, thrust_data,
                             elev, azi.
        n_runs:              Number of MC trials.
        wind_uncertainty:    Fractional 1-σ wind uncertainty (e.g. 0.10).
        thrust_uncertainty:  Fractional 1-σ thrust uncertainty (e.g. 0.05).
        stop_flag:           Optional threading.Event; set to abort early.

    Returns:
        (scatter, wind_profiles)
        scatter:       list of (x_east_m, y_north_m) — successful runs only,
                       in metres from the launch-point origin.
        wind_profiles: list of [(alt_m, speed_m_s), …] — one per trial.
    """
    scatter:       list[tuple[float, float]]       = []
    wind_profiles: list[list[tuple[float, float]]] = []

    rng        = _random_mod.Random()
    wu         = max(wind_uncertainty,   0.0)
    tu         = max(thrust_uncertainty, 0.0)
    raw_thrust = params['thrust_data']
    elev       = params['elev']
    azi        = params['azi']
    base_u: list[tuple[float, float]] = params.get('wind_u_prof', [])
    base_v: list[tuple[float, float]] = params.get('wind_v_prof', [])

    for _ in range(n_runs):
        if stop_flag is not None and stop_flag.is_set():
            break

        u_prof, v_prof, spd_prof = _perturb_wind_profile(base_u, base_v, rng, wu)

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

    All calculations are in the metric East-North frame (metres from the
    launch-point origin).  The UI layer is responsible for converting the
    returned metric parameters to geographic coordinates for display.

    Args:
        scatter:  list of (x_east_m, y_north_m) landing positions.
        prob_pct: Confidence percentage; must be a key in CHI2_2DOF
                  (50, 68, 80, 85, 90, 95, 99).  Falls back to 90.

    Returns:
        dict with keys:
            cx, cy     — ellipse centre (metres East/North from origin)
            a          — semi-major axis (metres)
            b          — semi-minor axis (metres)
            angle_rad  — major-axis angle from East (radians)
        or None if fewer than 4 scatter points are available.
    """
    if len(scatter) < 4:
        return None

    arr = np.array(scatter, dtype=float)
    cx  = float(arr[:, 0].mean())
    cy  = float(arr[:, 1].mean())
    cov = np.cov(arr[:, 0], arr[:, 1])
    # Regularise: 1e-6 m² on the diagonal prevents zero minor axis for
    # perfectly collinear scatter (e.g. zero crosswind variance).
    cov = cov + np.eye(2) * 1e-6

    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending eigenvalue order
    lam1      = float(eigvals[1])             # major-axis variance
    lam2      = float(eigvals[0])             # minor-axis variance
    major_vec = eigvecs[:, 1]
    angle_rad = float(math.atan2(float(major_vec[1]), float(major_vec[0])))

    k = chi2_scale(prob_pct)
    a = k * math.sqrt(max(lam1, 0.0))
    b = k * math.sqrt(max(lam2, 0.0))
    b = max(b, max(0.5, a * 0.05))   # floor: prevent degenerate near-zero b

    return {'cx': cx, 'cy': cy, 'a': a, 'b': b, 'angle_rad': angle_rad}


# ── CEP ───────────────────────────────────────────────────────────────────────

def compute_cep(scatter: list[tuple[float, float]]) -> float:
    """Return the CEP: 50th-percentile distance from the scatter centroid.

    The centroid is the mean (x, y) of all landing positions — the bias
    point of the distribution.  Linear interpolation is used when the
    50th percentile falls between two samples.

    Returns 0.0 if scatter is empty.
    """
    if not scatter:
        return 0.0
    arr   = np.array(scatter, dtype=float)
    cx    = float(arr[:, 0].mean())
    cy    = float(arr[:, 1].mean())
    dists = sorted(math.hypot(x - cx, y - cy) for x, y in scatter)
    mid   = (len(dists) - 1) / 2.0
    lo    = int(mid)
    hi    = min(lo + 1, len(dists) - 1)
    return dists[lo] + (mid - lo) * (dists[hi] - dists[lo])


def compute_cep_circle(
    scatter: list[tuple[float, float]],
    n: int = 36,
) -> dict[str, object] | None:
    """Compute the CEP circle as a metric polygon.

    The circle is centred on the scatter centroid (mean landing position),
    not the launch origin.  All output is in the metric East-North frame.

    Args:
        scatter: list of (x_east_m, y_north_m) landing positions.
        n:       Number of polygon vertices (default 36).

    Returns:
        dict with keys:
            cx_m, cy_m — circle centre (metres East/North from origin)
            radius_m   — CEP radius (metres)
            points_m   — list of (x_east_m, y_north_m) polygon vertices
        or None if scatter is empty.
    """
    if not scatter:
        return None
    arr    = np.array(scatter, dtype=float)
    cx_m   = float(arr[:, 0].mean())
    cy_m   = float(arr[:, 1].mean())
    dists  = sorted(math.hypot(x - cx_m, y - cy_m) for x, y in scatter)
    mid    = (len(dists) - 1) / 2.0
    lo     = int(mid)
    hi     = min(lo + 1, len(dists) - 1)
    radius = dists[lo] + (mid - lo) * (dists[hi] - dists[lo])
    radius = max(radius, 1.0)   # prevent degenerate zero-radius circle

    return {
        'cx_m':     cx_m,
        'cy_m':     cy_m,
        'radius_m': radius,
        'points_m': _circle_points_m(cx_m, cy_m, radius, n),
    }


# ── KDE contours ──────────────────────────────────────────────────────────────

def compute_kde_contours(
    scatter: list[tuple[float, float]],
    conf_pct: int = 90,
) -> list[dict]:
    """Compute KDE probability-mass contours in the metric East-North frame.

    All KDE fitting, grid evaluation, and contour extraction are performed
    entirely in metres.  The function returns raw mathematical data with
    no geographic coordinates and no rendering attributes (no colours,
    no line widths, no alpha values).

    Three probability levels are computed: 50 %, 70 %, and *conf_pct* %.

    This function requires scipy.  If scipy or matplotlib is unavailable,
    or if fewer than 5 points are provided, an empty list is returned.

    Uses matplotlib.figure.Figure() directly (no pyplot / TkAgg canvas)
    so it is safe to call from any thread, including background workers.

    Args:
        scatter:   list of (x_east_m, y_north_m) landing positions.
        conf_pct:  Outer contour confidence percentage (default 90).

    Returns:
        list of contour dicts sorted outer → inner, each containing:
            'points_m'  — list of (x_east_m, y_north_m) polygon vertices
            'prob_frac' — probability mass fraction (e.g. 0.90)
            'label'     — str like '90%' for the primary (largest) polygon
                           at each level; None for secondary island polygons
    """
    try:
        from scipy.stats import gaussian_kde
        from matplotlib.figure import Figure as _MplFigure
        import numpy as _np
    except ImportError:
        return []

    if len(scatter) < 5:
        return []

    xs = _np.array([p[0] for p in scatter], dtype=float)
    ys = _np.array([p[1] for p in scatter], dtype=float)

    try:
        kde = gaussian_kde(_np.vstack([xs, ys]))
    except Exception:
        return []

    x_range = float(xs.max() - xs.min())
    y_range = float(ys.max() - ys.min())
    pad     = max(x_range, y_range, 1.0) * 0.5

    gx     = _np.linspace(float(xs.min()) - pad, float(xs.max()) + pad, 120)
    gy     = _np.linspace(float(ys.min()) - pad, float(ys.max()) + pad, 120)
    GX, GY = _np.meshgrid(gx, gy)
    Z      = kde(_np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)

    # Convert probability-mass fractions to KDE density thresholds
    z_flat   = Z.ravel()
    z_sorted = _np.sort(z_flat)[::-1]
    cumsum   = _np.cumsum(z_sorted)
    cumsum  /= cumsum[-1]

    outer_frac = max(min(conf_pct / 100.0, 0.999), 0.501)
    levels_pm  = sorted({0.50, 0.70, outer_frac})

    # Compute density thresholds; record pm → threshold for labels
    level_vals: list[float] = []
    lv_to_pm:   dict        = {}   # keyed by round(lv, 10)
    for pm in levels_pm:
        idx = int(_np.searchsorted(cumsum, pm))
        idx = min(idx, len(z_sorted) - 1)
        lv  = float(z_sorted[idx])
        level_vals.append(lv)
        key = round(lv, 10)
        if key not in lv_to_pm:
            lv_to_pm[key] = pm

    seen:        set         = set()
    unique_vals: list[float] = []
    for v in level_vals:
        key = round(v, 12)
        if key not in seen:
            seen.add(key)
            unique_vals.append(v)

    if len(unique_vals) < 2:
        return []

    # Figure() is not registered with pyplot — no TkAgg canvas, thread-safe
    _fig = _MplFigure()
    _ax  = _fig.add_subplot(111)
    try:
        cs = _ax.contour(GX, GY, Z, levels=sorted(unique_vals))
    except Exception:
        return []

    sorted_lv = sorted(unique_vals)

    # Extract contour path segments.
    # allsegs removed in mpl 3.10; collections deprecated 3.8 — suppress both.
    segs_by_level: list = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        try:
            segs_by_level = list(cs.allsegs)
        except AttributeError:
            try:
                segs_by_level = [
                    [p.vertices for p in c.get_paths()]
                    for c in cs.collections
                ]
            except Exception:
                pass

    # _fig is not in pyplot's figure registry; GC handles cleanup.

    contours: list[dict] = []
    for seg_group, lv in zip(segs_by_level, sorted_lv):
        pm         = lv_to_pm.get(round(lv, 10))
        base_label = f'{int(round(pm * 100))}%' if pm is not None else None
        first      = True
        for verts in seg_group:
            if len(verts) < 3:
                continue
            contours.append({
                'points_m':  [(float(v[0]), float(v[1])) for v in verts],
                'prob_frac': pm if pm is not None else 0.0,
                'label':     base_label if first else None,
            })
            first = False

    return contours
