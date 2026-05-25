"""
core/wind_model.py
Multi-level wind profile factory for RocketPy atmospheric model input.

Public API
----------
WIND_SAMPLE_ALTS : list[float]
    Fixed diagnostic altitudes (m AGL) used by sample_wind_nodes.

ANEMOMETER_ALT : float
    Altitude (m AGL) of the surface anemometer node (3.0 m).
    Matches obs_alt in create_wind_profile and WIND_SAMPLE_ALTS[0].

WindLevel
    NamedTuple: (alt_m, speed_ms, dir_deg).

speed_dir_to_uv(speed, dir_deg) -> (u, v)
    Meteorological speed/direction → East/North vector components.

uv_to_speed_dir(u, v) -> (speed, dir_deg)
    East/North vector components → meteorological speed/direction.
    Returns (0.0, 0.0) for zero or near-zero vectors.

create_wind_profile(gpv_levels, obs_speed, obs_dir,
                    obs_alt=3.0, blend_alt=100.0) -> (u_prof, v_prof)
    Build a smooth, physically consistent wind profile from multi-level
    GPV data blended with a surface observation.  All interpolation is
    performed on U/V vector components to eliminate the 0°/360° crossover
    discontinuity that appears when interpolating meteorological directions.

sample_wind_nodes(u_prof, v_prof, alts=None) -> list[dict]
    Extract wind speed, direction, U, and V at five fixed diagnostic
    altitude nodes so the UI can accurately display vertical wind shear.

COORDINATE CONTRACT
-------------------
All wind components are in the RocketPy East-North convention:
  U — positive eastward  (m/s)
  V — positive northward (m/s)

The returned (u_prof, v_prof) lists are ready for direct use::

    env.set_atmospheric_model(
        type='custom_atmosphere', pressure=None, temperature=None,
        wind_u=u_prof, wind_v=v_prof,
    )

0°/360° CROSSOVER FIX
---------------------
Interpolating meteorological directions directly is unsafe near North.
Example: naive interpolation between 350° and 10° yields 180° (South)
instead of 0° (North).  This module avoids the problem entirely by
converting all levels to (U, V) vectors before any interpolation,
then converting back at the end.  atan2 handles the zero-vector case.
"""

from __future__ import annotations

import math
import random as _random_mod
from typing import NamedTuple, Union


# ── Diagnostic sample altitudes ───────────────────────────────────────────────

# Five fixed nodes that span the boundary layer through the mid-troposphere.
#
# Data source split:
#   3 m  → 自作風速計 (custom on-site anemometer) — surface truth
#   10 m, 150 m, 300 m, 600 m → upper-wind API (GPV / JMA MSM)
#
# The 3 m node is intentionally kept separate because its value is measured
# directly at the launch site; the upper nodes are model/API-derived and may
# carry larger uncertainty, which is reflected in the MC perturbation budget.
WIND_SAMPLE_ALTS: list[float] = [3.0, 10.0, 150.0, 300.0, 600.0]

# Altitude (m AGL) of the surface anemometer observation.
# Matches obs_alt passed to create_wind_profile and the first WIND_SAMPLE_ALTS entry.
ANEMOMETER_ALT: float = 3.0


# ── Data type ─────────────────────────────────────────────────────────────────

class WindLevel(NamedTuple):
    """Single altitude level in a GPV wind profile."""
    alt_m:    float   # altitude above ground level (metres AGL)
    speed_ms: float   # wind speed (m/s)
    dir_deg:  float   # meteorological direction (degrees FROM which wind blows)


# Accept WindLevel, plain tuple, or list
_LevelLike = Union[WindLevel, tuple, list]

# ── Vector conversion helpers ─────────────────────────────────────────────────

def speed_dir_to_uv(speed: float, dir_deg: float) -> tuple[float, float]:
    """Convert meteorological speed/direction to East/North vector components.

    System Directive ENU System: *dir_deg* 0 is North, increasing clockwise.
    Calculates explicitly as: u = speed * sin(direction), v = speed * cos(direction)

    Args:
        speed:    Wind speed (m/s); must be ≥ 0.
        dir_deg:  Wind direction in degrees [0, 360).

    Returns:
        (u, v) — East (positive = eastward) and North (positive = northward)
        components in m/s.
    """
    rad = math.radians(dir_deg)
    return (speed * math.sin(rad),
            speed * math.cos(rad))


def uv_to_speed_dir(u: float, v: float) -> tuple[float, float]:
    """Convert East/North vector components to meteorological speed/direction.

    Safe for zero or near-zero vectors: returns (0.0, 0.0) rather than
    raising ZeroDivisionError or producing NaN.

    Args:
        u:  East component (m/s, positive = eastward).
        v:  North component (m/s, positive = northward).

    Returns:
        (speed, dir_deg) — speed in m/s, direction in degrees [0, 360).
    """
    speed = math.hypot(u, v)
    if speed < 1e-9:
        return 0.0, 0.0
    dir_deg = math.degrees(math.atan2(u, v)) % 360.0
    return speed, dir_deg


# ── Profile interpolation helper ─────────────────────────────────────────────

def _interp_profile(prof: list[tuple[float, float]], alt: float) -> float:
    """Linearly interpolate a sorted (alt_m, value) profile at *alt*.

    Clamps to the first/last value outside the profiled range.
    """
    if not prof:
        return 0.0
    if alt <= prof[0][0]:
        return prof[0][1]
    if alt >= prof[-1][0]:
        return prof[-1][1]
    for i in range(len(prof) - 1):
        a0, v0 = prof[i]
        a1, v1 = prof[i + 1]
        if a0 <= alt <= a1:
            span = a1 - a0
            if span < 1e-9:
                return v0
            return v0 + (v1 - v0) * (alt - a0) / span
    return prof[-1][1]


# ── Fixed-node wind extraction ────────────────────────────────────────────────

def sample_wind_nodes(
    u_prof: list[tuple[float, float]],
    v_prof: list[tuple[float, float]],
    alts: list[float] | None = None,
) -> list[dict]:
    """Sample U/V wind profile at fixed diagnostic altitude nodes.

    Provides structured per-layer data ready for the UI's vertical wind
    shear display.  Each node carries a ``source`` tag that records whether
    the value originates from the on-site anemometer or the upper-wind API:

    * ``"anemometer"`` — 3 m AGL (自作風速計, custom launch-site hardware).
      This is surface truth; it is entered directly as ``obs_speed`` /
      ``obs_dir`` in :func:`create_wind_profile` and is NOT inferred from
      any model.
    * ``"api"`` — 10 m, 150 m, 300 m, 600 m AGL.  Values are derived from
      the GPV / JMA-MSM upper-wind API and may carry larger uncertainty,
      which drives the Monte Carlo perturbation budget.

    Args:
        u_prof: list of (alt_m, u_m_s) — east wind component, sorted.
        v_prof: list of (alt_m, v_m_s) — north wind component, sorted.
        alts:   Altitudes (m AGL) to sample.  Defaults to
                :data:`WIND_SAMPLE_ALTS` = [3, 10, 150, 300, 600] m.

    Returns:
        List of dicts (one per altitude, order preserved), each with:
            ``alt_m``    — altitude (m AGL)
            ``u``        — east component (m/s, + = eastward)
            ``v``        — north component (m/s, + = northward)
            ``speed_ms`` — wind speed (m/s)
            ``dir_deg``  — meteorological direction (° FROM which wind blows)
            ``source``   — ``"anemometer"`` for 3 m; ``"api"`` for all others
    """
    # Enforce the 5-node contract: always sample exactly these altitudes
    # unless the caller explicitly overrides for testing purposes.
    sample_alts: list[float] = WIND_SAMPLE_ALTS if alts is None else list(alts)

    nodes: list[dict] = []
    for alt in sample_alts:
        u = _interp_profile(u_prof, alt)
        v = _interp_profile(v_prof, alt)
        speed, dir_deg = uv_to_speed_dir(u, v)
        nodes.append({
            "alt_m":    float(alt),
            "u":        float(u),
            "v":        float(v),
            "speed_ms": float(speed),
            "dir_deg":  float(dir_deg),
            # Distinguishes on-site hardware truth (3 m) from API-derived levels
            "source":   "anemometer" if alt == ANEMOMETER_ALT else "api",
        })

    # Hard invariant: callers depend on exactly 5 nodes in WIND_SAMPLE_ALTS order.
    if alts is None and len(nodes) != 5:
        raise RuntimeError(
            f"sample_wind_nodes: expected 5 nodes, produced {len(nodes)}. "
            f"WIND_SAMPLE_ALTS={WIND_SAMPLE_ALTS!r}"
        )

    return nodes


# ── Gust layer ────────────────────────────────────────────────────────────────

def apply_gust(
    u_prof: list[tuple[float, float]],
    v_prof: list[tuple[float, float]],
    gust_intensity: float,
    rng: _random_mod.Random,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Apply independent per-level Gaussian gust noise to a U/V wind profile.

    Unlike the synoptic perturbation in ``_perturb_wind_profile`` (which
    applies a single coherent speed-scale and direction rotation across all
    altitudes), this function adds *independent* additive noise at every
    level.  This models sub-grid turbulence that a rocket experiences as it
    passes through each altitude band.

    The noise is disabled when *gust_intensity* is zero or negative so that
    callers can unconditionally call this function without a guard.

    Args:
        u_prof:         list of (alt_m, u_m_s) — east wind component.
        v_prof:         list of (alt_m, v_m_s) — north wind component.
        gust_intensity: 1-σ absolute gust noise in m/s.  Values ≤ 0 are
                        treated as "no gust" and the input profiles are
                        returned unchanged.
        rng:            Seeded :class:`random.Random` instance.

    Returns:
        ``(u_out, v_out)`` — new profile lists with gust noise applied.
        The altitude coordinates are preserved exactly.
    """
    if gust_intensity <= 0.0 or not u_prof:
        return list(u_prof), list(v_prof)

    sigma  = float(gust_intensity)
    u_out: list[tuple[float, float]] = []
    v_out: list[tuple[float, float]] = []

    for (alt_u, u), (_, v) in zip(u_prof, v_prof):
        u_out.append((alt_u, u + rng.gauss(0.0, sigma)))
        v_out.append((alt_u, v + rng.gauss(0.0, sigma)))

    return u_out, v_out


# ── Wind profile factory ───────────────────────────────────────────────────────

import numpy as np

def create_wind_profile(
    wind_profile: list[dict],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if not wind_profile:
        return [(0.0, 0.0)], [(0.0, 0.0)]

    nodes = []
    for n in wind_profile:
        try:
            alt = float(n.get("alt_m", n.get("alt", 0.0)))
            spd = float(n.get("speed_ms", n.get("spd", 0.0)))
            dir_deg = float(n.get("dir_deg", n.get("dir", 0.0)))
            nodes.append((alt, spd, dir_deg))
        except (ValueError, TypeError):
            continue

    nodes.sort(key=lambda t: t[0])

    if not nodes:
        return [(0.0, 0.0)], [(0.0, 0.0)]

    u_pts, v_pts, alts = [], [], []
    for alt, spd, dir_deg in nodes:
        u, v = speed_dir_to_uv(spd, dir_deg)
        u_pts.append(u)
        v_pts.append(v)
        alts.append(alt)

    alt_set: set[float] = set(alts)
    if not alt_set:
        alt_set.add(0.0)

    max_alt = max(alt_set)
    if max_alt < 5000.0:
        alt_set.add(5000.0)

    for _diag_alt in WIND_SAMPLE_ALTS:
        alt_set.add(float(_diag_alt))

    eval_alts = sorted(alt_set)
    u_interp = np.interp(eval_alts, alts, u_pts)
    v_interp = np.interp(eval_alts, alts, v_pts)

    u_prof = [(float(a), float(u)) for a, u in zip(eval_alts, u_interp)]
    v_prof = [(float(a), float(v)) for a, v in zip(eval_alts, v_interp)]

    return u_prof, v_prof


# ── Phase B O(1) interlock ────────────────────────────────────────────────────

def check_wind_limits(
    current_wind_ms: float,
    locked_mu: float,
    locked_sigma: float,
    k: float = 3.0,
) -> bool:
    """O(1) Phase B GO/NO-GO wind safety check.

    Returns True (GO) when the current surface wind speed is within the
    safety envelope established during Phase A.  The envelope is:

        current_wind ≤ locked_mu + k × max(locked_sigma, 0.1)

    The 0.1 m/s floor on sigma prevents a perfectly-calm Phase A baseline
    from creating an impossibly tight envelope the instant the first gust
    arrives.

    Args:
        current_wind_ms: Live surface wind speed (m/s).
        locked_mu:       Mean surface wind speed recorded during Phase A (m/s).
        locked_sigma:    Standard deviation of surface wind during Phase A (m/s).
        k:               Safety multiplier (default 3.0 = 3-sigma boundary).

    Returns:
        True  → GO   — within the Phase A safety envelope
        False → NO-GO — exceeds the envelope
    """
    threshold = locked_mu + k * max(locked_sigma, 0.1)
    return float(current_wind_ms) <= threshold
