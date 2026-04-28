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
    spaghetti wind profiles.  The entire upper-air wind profile is
    perturbed at every altitude level for each trial.

compute_error_ellipse(scatter, prob_pct=90) -> dict | None
    Fit a 2-D covariance error ellipse to the scatter in the metric
    East-North coordinate system (metres from launch point).
    Returns {'cx', 'cy', 'a', 'b', 'angle_rad'} or None if too few
    points.

compute_error_ellipse_polygon(scatter, launch_lat, launch_lon,
                              prob_pct=90) -> list | None
    Convenience wrapper: compute the error ellipse in the metric frame
    and convert to a (lat, lon) polygon ready for map display.  This is
    the preferred entry-point for UI code because it guarantees the
    coordinate conversion is performed exactly once and in the right
    direction.

compute_cep(scatter) -> float
    Compute the CEP: 50th-percentile distance from the scatter centroid.

compute_cep_polygon(scatter, launch_lat, launch_lon) -> dict | None
    Compute the CEP circle and return metric dimensions plus a (lat, lon)
    polygon ready for simultaneous use on graphs (metres) and map displays.

compute_kde_contours_metric(scatter, conf_pct=90)
        -> list[(points_m, color, width, pct_label)]
    Compute KDE probability-mass contours in the metric East-North frame.
    All output coordinates are (x_east_m, y_north_m) in metres.  No
    geographic conversion is performed here.

compute_kde_contours(scatter, launch_lat, launch_lon,
                     conf_pct=90) -> list[(latlons, color, width, pct_label)]
    Geographic wrapper around compute_kde_contours_metric.  Converts each
    metric point to (lat, lon) via offset_to_latlon.  Returns an empty
    list if scipy is unavailable or the scatter is too small.

COORDINATE CONTRACT
-------------------
All statistical functions (compute_error_ellipse, compute_cep,
compute_kde_contours) receive scatter as a list of
(x_east_m, y_north_m) positions in **metres** measured from the
launch point.  This matches the impact_x / impact_y values returned
directly by simulate_once (RocketPy East-North frame).

Conversion to geographic coordinates is performed only at the final
output stage via geo_math.offset_to_latlon / geo_math.ellipse_polygon.
Never pass lat/lon pairs as scatter; the covariance and KDE
calculations would operate on degree-scale values (~1e-4 to 1e-8)
and produce results that cannot be meaningfully converted back.
"""

from __future__ import annotations

import math
import random as _random_mod
import threading
import warnings
from typing import Optional

import numpy as np

from .simulation import simulate_once
from utils.geo_math import circle_polygon, offset_to_latlon, ellipse_polygon


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


# ── Wind profile perturbation ─────────────────────────────────────────────────
def _perturb_wind_profile(
    u_prof: list[tuple[float, float]],
    v_prof: list[tuple[float, float]],
    rng: _random_mod.Random,
    wind_uncertainty: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
    """Perturb a wind profile at every altitude level with realistic noise.

    Two-component perturbation model applied to the full vertical profile:

    *Global layer* (large-scale synoptic variability):
        A single speed-scale factor and direction rotation are sampled
        once and applied uniformly across all altitude levels.  This
        represents broad forecast error in the overall wind magnitude
        and backing/veering.

    *Local layer* (mesoscale turbulence / small-scale variability):
        Independent additive Gaussian noise is added at each level,
        scaled proportionally to the *local* wind speed at that
        altitude rather than the surface speed.  This avoids
        under-perturbing the jet-stream layer and over-perturbing
        near-calm surface layers.

    Args:
        u_prof:           list of (alt_m, u_m_s) — east wind component.
        v_prof:           list of (alt_m, v_m_s) — north wind component.
        rng:              seeded Random instance for reproducibility.
        wind_uncertainty: fractional 1-σ uncertainty
                          (e.g. 0.10 = ±10 % of the local wind speed).

    Returns:
        (u_perturbed, v_perturbed, speed_profile)
        speed_profile is [(alt_m, speed_m_s), …] for spaghetti plots.
    """
    if not u_prof or not v_prof:
        return list(u_prof), list(v_prof), []

    wu = max(wind_uncertainty, 0.0)

    # Global synoptic perturbation — sampled once per trial
    speed_factor = max(0.05, 1.0 + rng.gauss(0.0, wu))
    # Direction rotation: ~±30° σ when wu = 1.0, ±3° σ at wu = 0.1
    dir_rot      = rng.gauss(0.0, wu * math.pi / 6.0)
    cos_r, sin_r = math.cos(dir_rot), math.sin(dir_rot)

    u_new:   list[tuple[float, float]] = []
    v_new:   list[tuple[float, float]] = []
    spd_out: list[tuple[float, float]] = []

    for (alt_u, u_nom), (_, v_nom) in zip(u_prof, v_prof):
        # Apply global rotation + speed scaling
        u_g = (u_nom * cos_r - v_nom * sin_r) * speed_factor
        v_g = (u_nom * sin_r + v_nom * cos_r) * speed_factor

        # Per-level turbulence: 30 % of the global fraction × local speed
        # (minimum 1 m/s reference so calm layers still see small noise)
        local_spd = math.hypot(u_nom, v_nom)
        sigma     = wu * max(local_spd, 1.0) * 0.30
        u_p = u_g + rng.gauss(0.0, sigma)
        v_p = v_g + rng.gauss(0.0, sigma)

        u_new.append((alt_u, u_p))
        v_new.append((alt_u, v_p))
        spd_out.append((alt_u, math.hypot(u_p, v_p)))

    return u_new, v_new, spd_out


# ── MC scatter worker ─────────────────────────────────────────────────────────

def run_mc_scatter(
    params: dict,
    n_runs: int,
    wind_uncertainty: float,
    thrust_uncertainty: float,
    stop_flag: Optional[threading.Event] = None,
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    """Run n_runs Monte-Carlo simulations and return results.

    Each trial independently perturbs the complete upper-air wind
    profile (every altitude level) and the motor thrust curve.
    Wind perturbation uses the pre-computed wind_u_prof / wind_v_prof
    arrays already stored in *params*, so the MC ensemble is always
    centred on the same nominal trajectory.

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
        scatter:       list of (impact_x_m, impact_y_m) — successful runs
                       only, in metres East/North from the launch point.
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
    # Use the already-computed full-altitude wind profiles as the baseline
    base_u: list[tuple[float, float]] = params.get('wind_u_prof', [])
    base_v: list[tuple[float, float]] = params.get('wind_v_prof', [])

    for _ in range(n_runs):
        if stop_flag is not None and stop_flag.is_set():
            break

        # Perturb the entire upper-air wind profile at every altitude level
        u_prof, v_prof, spd_prof = _perturb_wind_profile(
            base_u, base_v, rng, wu)

        # Perturb motor thrust
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

    All calculations are performed in the metric East-North frame
    (metres from launch point).  The returned values are also in metres
    and radians; they must be converted to geographic coordinates via
    geo_math functions before display.

    For a ready-to-use (lat, lon) polygon, call
    compute_error_ellipse_polygon instead.

    Args:
        scatter:  list of (x_east_m, y_north_m) landing positions.
        prob_pct: Confidence percentage.  Must be a key in CHI2_2DOF
                  (50, 68, 80, 85, 90, 95, 99); falls back to 90.

    Returns:
        dict with keys:
            cx, cy      — ellipse centre (metres East/North from origin)
            a           — semi-major axis (metres)
            b           — semi-minor axis (metres)
            angle_rad   — major-axis azimuth from East (radians)
        or None if fewer than 4 scatter points are available.
    """
    if len(scatter) < 4:
        return None

    arr = np.array(scatter, dtype=float)
    cx  = float(arr[:, 0].mean())
    cy  = float(arr[:, 1].mean())
    cov = np.cov(arr[:, 0], arr[:, 1])
    # Regularise: adds 1e-6 m² to the diagonal so eigenvalues are always
    # strictly positive even when scatter is perfectly collinear.
    cov = cov + np.eye(2) * 1e-6

    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending order; index 1 = largest eigenvalue
    lam2      = float(eigvals[0])   # minor-axis variance
    lam1      = float(eigvals[1])   # major-axis variance
    major_vec = eigvecs[:, 1]
    angle_rad = float(math.atan2(float(major_vec[1]), float(major_vec[0])))

    k = chi2_scale(prob_pct)
    a = k * math.sqrt(max(lam1, 0.0))
    b = k * math.sqrt(max(lam2, 0.0))
    # Floor: prevent degenerate near-zero minor axis
    b = max(b, max(0.5, a * 0.05))

    return {'cx': cx, 'cy': cy, 'a': a, 'b': b, 'angle_rad': angle_rad}


def compute_error_ellipse_polygon(
    scatter: list[tuple[float, float]],
    launch_lat: float,
    launch_lon: float,
    prob_pct: int = 90,
) -> list[tuple[float, float]] | None:
    """Compute the confidence error ellipse and return it as a (lat, lon) polygon.

    This is the preferred entry-point for UI code.  It guarantees that:
      1. All statistics are computed in the metric East-North frame.
      2. The coordinate conversion to geographic coordinates is
         performed exactly once, using the launch point as the origin.

    Args:
        scatter:    list of (x_east_m, y_north_m) landing positions.
        launch_lat: Launch-point latitude in decimal degrees.
        launch_lon: Launch-point longitude in decimal degrees.
        prob_pct:   Confidence percentage (default 90).

    Returns:
        list of (lat, lon) tuples (polygon vertices, closed loop), or
        None if fewer than 4 scatter points are available.
    """
    ell = compute_error_ellipse(scatter, prob_pct)
    if ell is None:
        return None
    return ellipse_polygon(
        launch_lat, launch_lon,
        ell['cx'], ell['cy'],
        ell['a'],  ell['b'],
        ell['angle_rad'],
    )


# ── CEP ───────────────────────────────────────────────────────────────────────

def compute_cep(scatter: list[tuple[float, float]]) -> float:
    """Return the CEP: 50th-percentile distance from the scatter centroid.

    The centroid is the mean East/North position of all landing points
    (the bias point of the distribution).  Linear interpolation is used
    when the 50th percentile falls between two samples.

    Returns 0.0 if scatter is empty.
    """
    if not scatter:
        return 0.0
    arr  = np.array(scatter, dtype=float)
    cx   = float(arr[:, 0].mean())
    cy   = float(arr[:, 1].mean())
    dists = sorted(math.hypot(x - cx, y - cy) for x, y in scatter)
    mid   = (len(dists) - 1) / 2.0
    lo    = int(mid)
    hi    = min(lo + 1, len(dists) - 1)
    return dists[lo] + (mid - lo) * (dists[hi] - dists[lo])


def compute_cep_polygon(
    scatter: list[tuple[float, float]],
    launch_lat: float,
    launch_lon: float,
) -> dict[str, object] | None:
    """Compute the CEP circle and return metric + geographic polygon data.

    The circle is centred on the mean landing position (the bias point of
    the scatter cloud), not on the launch origin.  All calculations are
    performed in the metric East-North frame; the geographic polygon is
    produced only at the final step via geo_math utilities.

    Args:
        scatter:    list of (x_east_m, y_north_m) landing positions.
        launch_lat: Launch-point latitude in decimal degrees.
        launch_lon: Launch-point longitude in decimal degrees.

    Returns:
        dict with keys:
            cx_m, cy_m  — circle centre in the metric frame (metres East/North
                          from the launch point); useful for graph overlays.
            radius_m    — CEP radius (metres).
            latlons     — list of (lat, lon) tuples (polygon vertices, not
                          closed); pass directly to a map widget.
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

    centre_lat, centre_lon = offset_to_latlon(launch_lat, launch_lon, cx_m, cy_m)
    latlons = circle_polygon(centre_lat, centre_lon, radius)

    return {
        'cx_m':     cx_m,
        'cy_m':     cy_m,
        'radius_m': radius,
        'latlons':  latlons,
    }


# ── KDE contours ──────────────────────────────────────────────────────────────

def compute_kde_contours_metric(
    scatter: list[tuple[float, float]],
    conf_pct: int = 90,
) -> list[tuple[list[tuple[float, float]], str, int, Optional[str]]]:
    """Compute KDE probability-mass contours in the metric East-North frame.

    All computation and all output coordinates are in metres from the
    launch point.  No geographic conversion is performed here; callers
    that need (lat, lon) output should use compute_kde_contours instead.

    Three probability levels are computed: 50 %, 70 %, and *conf_pct* %.
    The first (largest) polygon at each level is tagged with a
    percentage label string; subsequent disconnected islands at the same
    level receive None.

    This function requires scipy.  If scipy or matplotlib is not
    installed, or if fewer than 5 points are provided, an empty list is
    returned.

    Args:
        scatter:   list of (x_east_m, y_north_m) landing positions.
        conf_pct:  Outer contour confidence percentage (default 90).

    Returns:
        list of (points_m, hex_colour, border_width, pct_label) tuples.
        points_m is a list of (x_east_m, y_north_m) tuples in metres.
        pct_label is e.g. '90%' for the first polygon per level, None
        for subsequent polygons at the same level.
    """
    try:
        from scipy.stats import gaussian_kde
        import matplotlib.pyplot as _plt
        import numpy as _np
    except ImportError:
        return []

    if len(scatter) < 5:
        return []

    # All computation in metres (East/North from launch point)
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

    z_flat   = Z.ravel()
    z_sorted = _np.sort(z_flat)[::-1]
    cumsum   = _np.cumsum(z_sorted)
    cumsum  /= cumsum[-1]

    outer_frac = max(min(conf_pct / 100.0, 0.999), 0.501)
    levels_pm  = sorted({0.50, 0.70, outer_frac})

    # Compute density thresholds; track pm→threshold for percentage labels
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

    fig_tmp, ax_tmp = _plt.subplots()
    try:
        cs = ax_tmp.contour(GX, GY, Z, levels=sorted(unique_vals))
    except Exception:
        _plt.close(fig_tmp)
        return []

    sorted_lv = sorted(unique_vals)
    # Innermost (50 %) = brightest colour; outermost (conf_pct %) = dimmest
    palette  = ['#ff6600', '#ff9900', '#ffcc00', '#ffe066']
    widths   = [3, 2, 1, 1]
    lv_style = {
        lv: (palette[min(i, len(palette) - 1)],
             widths[min(i, len(widths) - 1)])
        for i, lv in enumerate(sorted_lv[::-1])
    }

    # Extract contour segments before closing the figure.
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

    _plt.close(fig_tmp)

    contours: list[tuple[list[tuple[float, float]], str, int, Optional[str]]] = []
    for seg_group, lv in zip(segs_by_level, sorted_lv):
        col, bw    = lv_style.get(lv, ('#ffcc00', 1))
        pm         = lv_to_pm.get(round(lv, 10))
        base_label = (f'{int(round(pm * 100))}%'
                      if pm is not None else None)
        first = True
        for verts in seg_group:
            if len(verts) < 3:
                continue
            # Output in metres — no geographic conversion here
            pts = [(float(v[0]), float(v[1])) for v in verts]
            contours.append((pts, col, bw, base_label if first else None))
            first = False

    return contours


def compute_kde_contours(
    scatter: list[tuple[float, float]],
    launch_lat: float,
    launch_lon: float,
    conf_pct: int = 90,
) -> list[tuple[list[tuple[float, float]], str, int, Optional[str]]]:
    """Compute KDE probability-mass contours as (lat, lon) polygons.

    Geographic wrapper around compute_kde_contours_metric.  All KDE
    fitting, grid evaluation, and contour extraction are performed in
    the metric East-North frame; the conversion to geographic coordinates
    is applied here in a single final pass via offset_to_latlon.

    Args:
        scatter:    list of (x_east_m, y_north_m) landing positions.
        launch_lat: Launch latitude in decimal degrees (conversion origin).
        launch_lon: Launch longitude in decimal degrees.
        conf_pct:   Outer contour confidence percentage (default 90).

    Returns:
        list of (latlons, hex_colour, border_width, pct_label) tuples.
        latlons is a list of (lat, lon) tuples ready for map display.
        pct_label is e.g. '90%' for the first polygon per level, None
        for subsequent polygons at the same level.
    """
    metric = compute_kde_contours_metric(scatter, conf_pct)
    return [
        (
            [offset_to_latlon(launch_lat, launch_lon, x, y) for x, y in pts],
            col, bw, label,
        )
        for pts, col, bw, label in metric
    ]
