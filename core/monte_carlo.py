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
    Fit a 2-D covariance error ellipse to the scatter (list-of-tuples input,
    integer percentile).  Returns math + UI keys, or None if < 4 points.

compute_cep_ellipse(points, probability=0.90) -> dict | None
    Standalone ellipse calculator for UI-driven updates (list-of-tuples or
    numpy input, float probability in (0, 1)).  Explicit descending eigen-sort;
    analytic chi²(2) scale factor.  Returns safe default dict (not None) on
    degenerate input; None only when probability is outside (0, 1).

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
from .wind_model import apply_gust
from .constants  import CHI2_90


# ── Chi-squared 2-DOF quantile table ─────────────────────────────────────────

CHI2_2DOF: dict[int, float] = {
    50: 1.386,
    68: 2.296,
    80: 3.219,
    85: 3.794,
    90: CHI2_90,
    95: 5.991,
    99: 9.210,
}


def chi2_scale(prob_pct: int) -> float:
    """Return sqrt(chi²(2, prob_pct/100)) for error-ellipse axis scaling.

    Falls back to the 90 % value for unknown percentages.
    """
    return math.sqrt(CHI2_2DOF.get(int(prob_pct), CHI2_90))


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
    gust_intensity: float = 0.0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
    """Perturb a wind profile at every altitude level with three-layer noise.

    *Global layer* (synoptic variability):
        A single speed-scale factor and direction rotation are sampled
        once per trial and applied uniformly across all altitude levels.

    *Local layer* (mesoscale turbulence):
        Independent additive Gaussian noise at each level, scaled to the
        local wind speed so jet-stream layers are perturbed proportionally.

    *Gust layer* (sub-grid turbulence):
        Independent absolute Gaussian noise at each level with 1-σ =
        *gust_intensity* m/s.  Applied after the synoptic and mesoscale
        layers.  Disabled when gust_intensity ≤ 0.

    Args:
        u_prof:           list of (alt_m, u_m_s) — east wind component.
        v_prof:           list of (alt_m, v_m_s) — north wind component.
        rng:              seeded Random instance.
        wind_uncertainty: fractional 1-σ uncertainty (e.g. 0.10 = ±10 %).
        gust_intensity:   absolute 1-σ gust noise in m/s (default 0 = off).

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

    has_gust = gust_intensity > 0.0
    gust_sigma = float(gust_intensity)

    rng_gauss = rng.gauss
    math_hypot = math.hypot

    n = len(u_prof)
    u_new: list[tuple[float, float]] = [None] * n  # type: ignore
    v_new: list[tuple[float, float]] = [None] * n  # type: ignore
    spd_out: list[tuple[float, float]] = [None] * n  # type: ignore

    for i, ((alt_u, u_nom), (_, v_nom)) in enumerate(zip(u_prof, v_prof)):
        # 1. Global (synoptic) rotation & scaling
        u_g = (u_nom * cos_r - v_nom * sin_r) * speed_factor
        v_g = (u_nom * sin_r + v_nom * cos_r) * speed_factor

        # 2. Local (mesoscale) turbulence
        local_spd = math_hypot(u_nom, v_nom)
        sigma     = wu * max(local_spd, 1.0) * 0.30
        u_val = u_g + rng_gauss(0.0, sigma)
        v_val = v_g + rng_gauss(0.0, sigma)

        # 3. Gust layer
        if has_gust:
            u_val += rng_gauss(0.0, gust_sigma)
            v_val += rng_gauss(0.0, gust_sigma)

        u_new[i] = (alt_u, u_val)
        v_new[i] = (alt_u, v_val)
        spd_out[i] = (alt_u, math_hypot(u_val, v_val))

    return u_new, v_new, spd_out


# ── MC scatter ────────────────────────────────────────────────────────────────

def run_mc_scatter(
    params: dict,
    n_runs: int,
    wind_uncertainty: float,
    thrust_uncertainty: float,
    gust_intensity: float = 0.0,
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
        gust_intensity:      Absolute 1-σ per-level gust noise in m/s
                             (default 0 = disabled).
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
    gi         = max(gust_intensity,     0.0)
    raw_thrust = params['thrust_data']
    elev       = params['elev']
    azi        = params['azi']
    base_u: list[tuple[float, float]] = params.get('wind_u_prof', [])
    base_v: list[tuple[float, float]] = params.get('wind_v_prof', [])

    for _ in range(n_runs):
        if stop_flag is not None and stop_flag.is_set():
            break

        u_prof, v_prof, spd_prof = _perturb_wind_profile(
            base_u, base_v, rng, wu, gust_intensity=gi
        )

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

    The scale factor applied to the standard-deviation axes is:
        k = sqrt(chi²(2, prob_pct/100)) = sqrt(-2 × ln(1 - prob_pct/100))

    Args:
        scatter:  list of (x_east_m, y_north_m) landing positions.
        prob_pct: Confidence percentage; must be a key in CHI2_2DOF
                  (50, 68, 80, 85, 90, 95, 99).  Falls back to 90.

    Returns:
        dict with keys:
            Math representation (semi-axes):
                cx, cy     — ellipse centre (metres East/North from origin)
                a          — semi-major axis length (metres)
                b          — semi-minor axis length (metres)
                angle_rad  — major-axis angle from East (radians)
            UI-ready representation (full extents, degrees):
                x, y       — ellipse centre (same as cx, cy)
                width      — full major-axis extent = 2 × a  (metres)
                height     — full minor-axis extent = 2 × b  (metres)
                angle_deg  — major-axis angle from East (degrees)
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

    return {
        # Math representation — semi-axes, radians
        'cx': cx, 'cy': cy, 'a': a, 'b': b, 'angle_rad': angle_rad,
        # UI-ready representation — full extents, degrees, aliased centre
        'x': cx, 'y': cy,
        'width':     2.0 * a,
        'height':    2.0 * b,
        'angle_deg': math.degrees(angle_rad),
    }


def compute_cep_ellipse(
    points,
    probability: float = 0.90,
) -> "dict[str, float] | None":
    """Fit a covariance error ellipse to landing scatter for a given probability.

    Standalone utility designed for UI-driven updates: when the operator moves
    the CEP-probability slider, the controller calls this directly on the cached
    scatter without re-running the physics simulation.

    Accepts both a plain list of (x, y) tuples and an (N, 2) numpy array.

    Scale factor
    ------------
    The 2-D bivariate normal containment probability ``p`` maps to axis scale
    ``k`` via the inverse chi-squared CDF with 2 degrees of freedom:

        k = sqrt(-2 × ln(1 − p))

    This is the analytic formula — no lookup table, so any probability in (0, 1)
    is supported continuously, not just discrete table values.

    Args:
        points:      Landing positions — list of (x, y) tuples or an (N, 2)
                     float array.  Column 0 is East (m), column 1 is North (m),
                     measured from the launch point.
        probability: Containment probability in the open interval (0, 1).
                     Pass 0.90 for a 90 % ellipse, 0.50 for CEP50, etc.
                     Defaults to 0.90.

    Returns:
        dict with keys:
            Math keys (consumed by _render_overlays):
                cx, cy     — centroid in metres East/North from launch origin
                a          — semi-major axis (metres)
                b          — semi-minor axis (metres)
                angle_rad  — major-axis angle from East axis (radians)
            UI-ready keys (full extents, degrees):
                x, y       — centroid (same as cx, cy)
                width      — 2 × a  (metres)
                height     — 2 × b  (metres)
                angle_deg  — degrees (alias: angle)
            Metadata:
                probability — echo of the input probability
        Safe default dict (all numeric fields 0) when fewer than 2 points.
        None when probability is outside (0, 1) — caller error, not data error.
    """
    _zero = {
        'cx': 0.0, 'cy': 0.0, 'a': 0.0, 'b': 0.0, 'angle_rad': 0.0,
        'x':  0.0, 'y':  0.0,
        'width': 0.0, 'height': 0.0,
        'angle_deg': 0.0, 'angle': 0.0,
        'probability': float(probability),
    }

    if probability <= 0.0 or probability >= 1.0:
        return None

    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        return _zero

    cx = float(arr[:, 0].mean())
    cy = float(arr[:, 1].mean())

    # Need at least 2 distinct rows for np.cov to return a 2×2 matrix.
    if arr.shape[0] < 3:
        return dict(_zero, cx=cx, cy=cy, x=cx, y=cy)

    cov = np.cov(arr[:, 0], arr[:, 1])
    # Tiny regularisation: prevents singular matrix for perfectly collinear
    # scatter (e.g. zero crosswind in a pure-headwind simulation).
    cov = cov + np.eye(2) * 1e-9

    eigvals, eigvecs = np.linalg.eigh(cov)   # returns ascending eigenvalue order

    # Explicit descending sort so major axis is always index 0.
    order   = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Analytic chi²(2, p) inverse CDF — scale factor for each axis.
    k = float(np.sqrt(-2.0 * np.log(1.0 - probability)))

    a         = k * float(np.sqrt(max(float(eigvals[0]), 0.0)))   # semi-major
    b         = k * float(np.sqrt(max(float(eigvals[1]), 0.0)))   # semi-minor
    angle_rad = float(np.arctan2(float(eigvecs[1, 0]), float(eigvecs[0, 0])))
    angle_deg = float(np.degrees(angle_rad))

    return {
        # Math keys — consumed by _render_overlays
        'cx': cx, 'cy': cy, 'a': a, 'b': b, 'angle_rad': angle_rad,
        # UI-ready keys — full extents, degrees
        'x': cx, 'y': cy,
        'width':       2.0 * a,
        'height':      2.0 * b,
        'angle_deg':   angle_deg,
        'angle':       angle_deg,   # alias for Folium/external consumers
        'probability': probability,
    }


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
    n = len(scatter)
    sum_x = 0.0
    sum_y = 0.0
    for x, y in scatter:
        sum_x += x
        sum_y += y
    cx = sum_x / n
    cy = sum_y / n

    math_hypot = math.hypot
    dists = sorted(math_hypot(x - cx, y - cy) for x, y in scatter)
    mid   = (n - 1) / 2.0
    lo    = int(mid)
    hi    = min(lo + 1, n - 1)
    return dists[lo] + (mid - lo) * (dists[hi] - dists[lo])


def compute_cep_circle(scatter: list[tuple[float, float]], n: int = 36) -> "dict | None":
    """Compute the CEP circle and return it as a metric polygon."""
    if not scatter:
        return None
    r = compute_cep(scatter)
    if r <= 0:
        return None

    arr   = np.array(scatter, dtype=float)
    cx    = float(arr[:, 0].mean())
    cy    = float(arr[:, 1].mean())

    pts = []
    for i in range(n):
        ang = 2 * math.pi * i / n
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))

    return {
        "cx_m": cx,
        "cy_m": cy,
        "radius_m": r,
        "points_m": pts
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


# ── KDE density grid ─────────────────────────────────────────────────────────

def compute_kde_grid(
    scatter: list[tuple[float, float]],
    grid_size: int = 100,
    padding_frac: float = 0.50,
) -> "dict | None":
    """Evaluate a Gaussian KDE on a 2-D metric grid and return the density field.

    The UI can use the returned X_m / Y_m / Z arrays directly — as a heatmap,
    as custom contour input, or as any other density visualisation.  No further
    statistical computation is required on the UI side.

    Grid resolution
    ---------------
    100 × 100 = 10 000 cells per axis pair.  This is the sweet spot between
    smooth gradient rendering and payload size (100² × 3 arrays ≈ 240 KB as
    JSON floats).  Halving to 50 saves ~75 % of the data but produces visibly
    blocky contours; doubling to 200 quadruples the payload with marginal
    visual improvement at typical map zoom levels.

    Grid cells are spaced uniformly in metres.  The bounding box is expanded by
    *padding_frac* × max(x_range, y_range) on all four sides to avoid edge
    artefacts at the KDE bandwidth boundary.

    Normalisation
    -------------
    Z is peak-normalised to [0, 1] by dividing by Z.max() so the UI can map Z
    directly to a colour intensity without knowing the absolute density scale.
    The shape of the density field (relative heights, contour positions) is
    preserved exactly.

    Args:
        scatter:      list of (x_east_m, y_north_m) landing positions.
        grid_size:    Number of grid points along each axis (default 100).
        padding_frac: Fractional padding around the scatter bounding box.

    Returns:
        dict with keys:
            X_m        — list[list[float]]  east coords of each grid cell (m)
            Y_m        — list[list[float]]  north coords of each grid cell (m)
            Z          — list[list[float]]  normalised density in [0, 1]
            x_min_m    — float  western edge of the grid (m)
            x_max_m    — float  eastern edge of the grid (m)
            y_min_m    — float  southern edge of the grid (m)
            y_max_m    — float  northern edge of the grid (m)
        or None if scipy is unavailable or fewer than 5 points are provided.

    All arrays are returned as Python lists of lists so the dict is safe to
    pass through Qt queued signals and to serialise to JSON.
    """
    try:
        from scipy.stats import gaussian_kde
        import numpy as _np
    except ImportError:
        return None

    if len(scatter) < 5:
        return None

    xs = _np.array([p[0] for p in scatter], dtype=float)
    ys = _np.array([p[1] for p in scatter], dtype=float)

    try:
        kde = gaussian_kde(_np.vstack([xs, ys]))
    except Exception:
        return None

    span    = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
    pad     = span * padding_frac
    x_min   = float(xs.min()) - pad
    x_max   = float(xs.max()) + pad
    y_min   = float(ys.min()) - pad
    y_max   = float(ys.max()) + pad

    gx      = _np.linspace(x_min, x_max, grid_size)
    gy      = _np.linspace(y_min, y_max, grid_size)
    GX, GY  = _np.meshgrid(gx, gy)
    Z       = kde(_np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)

    # Peak-normalise to [0, 1]: preserves relative density shape; lets the
    # UI apply any colour map by treating Z as a direct intensity value.
    z_max = float(Z.max())
    if z_max > 0.0:
        Z = Z / z_max

    return {
        "X_m":     GX.tolist(),
        "Y_m":     GY.tolist(),
        "Z":       Z.tolist(),
        "x_min_m": x_min,
        "x_max_m": x_max,
        "y_min_m": y_min,
        "y_max_m": y_max,
    }


# ── Phase B O(1) wind evaluation ─────────────────────────────────────────────

def evaluate_wind_within_bounds(
    live_u:  float,
    live_v:  float,
    mu_u:    float,
    mu_v:    float,
    sigma:   float,
    k:       float = 2.0,
) -> bool:
    """Return True if the live wind vector is within the Phase A k-sigma envelope.

    Computes the Euclidean distance between the live wind vector and the
    Phase A baseline (locked_mu), then compares it against k × σ × |μ|.
    A minimum absolute bound of 1.0 m/s is applied so a near-calm baseline
    does not shrink the acceptance region to zero.

    Args:
        live_u, live_v: Current East/North wind components (m/s).
        mu_u, mu_v:     Phase A locked baseline East/North components (m/s).
        sigma:          Fractional 1-σ wind uncertainty from Phase A (e.g. 0.20).
        k:              Sigma multiplier (default 2.0 → 95 % single-trial bound).

    Returns:
        True  → within bounds (🟢 GO).
        False → outside bounds (🔴 NO-GO).
    """
    delta    = math.hypot(live_u - mu_u, live_v - mu_v)
    mu_speed = math.hypot(mu_u, mu_v)
    bound    = k * sigma * max(mu_speed, 1.0)
    return delta <= bound
