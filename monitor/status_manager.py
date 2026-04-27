"""
monitor/status_manager.py
Phase-2 GO/NO-GO judgement engine.

This module is the pure-logic counterpart to the UI's go_nogo_label.
It accepts Phase-1 limits and current wind statistics, evaluates three
independent safety conditions, and returns a self-contained status
result dict.  No tkinter, no RocketPy, no matplotlib.

Public API
----------
GoNoGoStatus (dataclass, frozen)
    The complete result of one evaluation cycle.
    Fields documented below.

evaluate(ph1, tracker, window_sec=None) -> GoNoGoStatus
    The main entry point.  Takes a Phase1Result-like object and a
    WindTracker, returns a GoNoGoStatus.

build_live_ellipse(ph1, mu_cur, sigma_cur) -> dict
    Compute the live ellipse geometry from Phase-1 limits and current
    wind statistics.  Returns {'cx', 'cy', 'a', 'b', 'angle_rad'}.

Condition definitions
---------------------
A  μ_cur  ≤ ph1.mu_max          (mean wind within tolerable range)
B  σ_cur  ≤ ph1.sigma_max       (wind variability within tolerance)
C  ellipse ⊂ target circle       (projected landing zone fits inside r_max)

Final verdict: GO  iff  A ∧ B ∧ C
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# The ellipse-breach checker lives in core/optimization.py
from core.optimization import p1_ellipse_breaches_circle
from monitor.wind_tracker import WindTracker, WindStats


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GoNoGoStatus:
    """Immutable result of one Phase-2 evaluation cycle.

    Attributes:
        go          True = GO (all three conditions satisfied).
        cond_a      Condition A: μ_cur ≤ μ_max.
        cond_b      Condition B: σ_cur ≤ σ_max.
        cond_c      Condition C: ellipse ⊂ target circle.
        mu_cur      Current mean surface wind speed (m/s).
        sigma_cur   Current surface wind std-dev (m/s).
        mu_max      Phase-1 limit (m/s).
        sigma_max   Phase-1 limit (m/s).
        target_r    Target landing radius (m).
        ellipse     Live ellipse geometry dict:
                    {'cx', 'cy', 'a', 'b', 'angle_rad'}.
        color       Hex colour string: '#007700' (GO) or '#cc0000' (NO-GO).
        verdict     Short display string, e.g. '●  GO  — Ready for Launch'.
        detail      Multi-line string suitable for a small label widget.
        n_samples   Number of wind samples used in the evaluation.
    """
    go:        bool
    cond_a:    bool
    cond_b:    bool
    cond_c:    bool
    mu_cur:    float
    sigma_cur: float
    mu_max:    float
    sigma_max: float
    target_r:  float
    ellipse:   dict
    color:     str
    verdict:   str
    detail:    str
    n_samples: int


# ── Ellipse geometry builder ──────────────────────────────────────────────────

def build_live_ellipse(ph1, mu_cur: float, sigma_cur: float) -> dict:
    """Compute the live 90 % error ellipse from Phase-1 data and current wind.

    The ellipse centre is shifted from the nominal landing position using
    the linear wind-sensitivity gradient stored in ph1.  The semi-axes
    are scaled proportionally with sigma_cur via the per-sigma scale
    factor computed during Phase 1.

    Args:
        ph1:       Phase1Result (or compatible object with the attributes
                   nominal_cx, nominal_cy, dcx_dmu, dcy_dmu, mu_nominal,
                   ellipse_scale_per_sigma, ellipse_a, ellipse_b,
                   ellipse_angle_rad).
        mu_cur:    Current mean surface wind speed (m/s).
        sigma_cur: Current surface wind std-dev (m/s).

    Returns:
        dict with keys 'cx', 'cy', 'a', 'b', 'angle_rad' (all floats).
    """
    dmu   = mu_cur - ph1.mu_nominal
    cx    = ph1.nominal_cx + ph1.dcx_dmu * dmu
    cy    = ph1.nominal_cy + ph1.dcy_dmu * dmu

    scale = ph1.ellipse_scale_per_sigma
    a     = max(ph1.ellipse_a, scale * sigma_cur)
    ratio = ph1.ellipse_b / max(ph1.ellipse_a, 1e-6)
    b     = max(ph1.ellipse_b, scale * sigma_cur * ratio)

    return {
        'cx':        cx,
        'cy':        cy,
        'a':         a,
        'b':         b,
        'angle_rad': ph1.ellipse_angle_rad,
    }


# ── Main evaluator ────────────────────────────────────────────────────────────

def evaluate(
    ph1,
    tracker: WindTracker,
    window_sec: Optional[float] = None,
) -> Optional[GoNoGoStatus]:
    """Evaluate the current GO/NO-GO status.

    This is the single entry point called at ~1 Hz from the UI tick.
    It is fast: no simulations, no file I/O — only arithmetic on the
    rolling wind statistics.

    Args:
        ph1:        Phase1Result object (from core.optimization or
                    core.optimization.Phase1Result).  Must have attributes:
                    mu_nominal, mu_max, sigma_max, nominal_cx, nominal_cy,
                    dcx_dmu, dcy_dmu, ellipse_a, ellipse_b,
                    ellipse_angle_rad, ellipse_scale_per_sigma,
                    target_radius_m.
        tracker:    WindTracker instance with at least one sample.
        window_sec: Statistics window in seconds (None = full history).
                    The Phase-2 tick uses the full deque (≤300 samples =
                    5 min at 1 Hz), which is the same window used in the
                    original _phase2_tick implementation.

    Returns:
        GoNoGoStatus on success, or None if the tracker has no samples.
    """
    stats: Optional[WindStats] = tracker.stats(window_sec=window_sec)
    if stats is None:
        return None

    mu_cur    = stats.mu_surf
    sigma_cur = stats.sigma_surf

    # ── Three independent conditions ─────────────────────────────────────────

    # A: Mean wind speed within Phase-1 limit
    cond_a = mu_cur <= ph1.mu_max

    # B: Wind variability within Phase-1 limit
    cond_b = sigma_cur <= ph1.sigma_max

    # C: Projected error ellipse fits inside target circle
    #    Reconstruct synthetic eigenvalues / eigenvectors from the live
    #    semi-axes so we can reuse the p1_ellipse_breaches_circle checker.
    ellipse = build_live_ellipse(ph1, mu_cur, sigma_cur)
    K        = math.sqrt(4.605)   # chi²(2, 90 %)
    ev_cur   = np.array([(ellipse['b'] / K) ** 2,
                         (ellipse['a'] / K) ** 2])
    ang      = ellipse['angle_rad']
    c, s     = math.cos(ang), math.sin(ang)
    evc_cur  = np.array([[c, -s], [s, c]])
    cond_c = not p1_ellipse_breaches_circle(
        ellipse['cx'], ellipse['cy'],
        ev_cur, evc_cur,
        ph1.target_radius_m)

    go = cond_a and cond_b and cond_c

    # ── Display strings (computed here; UI layer just renders them) ───────────
    color   = '#007700' if go else '#cc0000'
    verdict = ('●  GO  — Ready for Launch' if go
               else '●  NO-GO  — Hold')

    def _m(ok: bool) -> str:
        return '✓' if ok else '✗'

    detail = (
        f'{_m(cond_a)} A  μ = {mu_cur:.2f}  /  μ_max = {ph1.mu_max:.2f} m/s\n'
        f'{_m(cond_b)} B  σ = {sigma_cur:.2f}  /  σ_max = {ph1.sigma_max:.2f} m/s\n'
        f'{_m(cond_c)} C  Ellipse ⊂ circle  (r = {ph1.target_radius_m:.0f} m)'
    )

    return GoNoGoStatus(
        go        = go,
        cond_a    = cond_a,
        cond_b    = cond_b,
        cond_c    = cond_c,
        mu_cur    = mu_cur,
        sigma_cur = sigma_cur,
        mu_max    = ph1.mu_max,
        sigma_max = ph1.sigma_max,
        target_r  = ph1.target_radius_m,
        ellipse   = ellipse,
        color     = color,
        verdict   = verdict,
        detail    = detail,
        n_samples = stats.n,
    )
