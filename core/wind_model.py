"""
core/wind_model.py
Multi-level wind profile factory for RocketPy atmospheric model input.

Public API
----------
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

COORDINATE CONTRACT
-------------------
All wind components are in the RocketPy East-North convention:
  U — positive eastward  (m/s)
  V — positive northward (m/s)

The returned (u_prof, v_prof) lists are ready for direct use::

    env.set_atmospheric_model(
        type='custom_atmosphere', pressure=None, temperature=300,
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
from typing import NamedTuple, Union


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

    Meteorological convention: *dir_deg* is the direction FROM which the
    wind blows (e.g. 270° = westerly wind, moves toward the east).

    Args:
        speed:    Wind speed (m/s); must be ≥ 0.
        dir_deg:  Wind direction in degrees [0, 360).

    Returns:
        (u, v) — East (positive = eastward) and North (positive = northward)
        components in m/s.
    """
    rad = math.radians(dir_deg)
    return (-speed * math.sin(rad),
            -speed * math.cos(rad))


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
    dir_deg = math.degrees(math.atan2(-u, -v)) % 360.0
    return speed, dir_deg


# ── Wind profile factory ───────────────────────────────────────────────────────

def create_wind_profile(
    gpv_levels:  list[_LevelLike],
    obs_speed:   float,
    obs_dir:     float,
    obs_alt:     float = 3.0,
    blend_alt:   float = 100.0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Build a smooth, blended wind profile from multi-level GPV data.

    All interpolation is performed on U/V vector components to prevent
    the 0°/360° crossover discontinuity.

    The profile is built in three vertical zones:

    * **Ground → obs_alt**: wind ramps linearly from zero at z = 0 to the
      observed value at *obs_alt*.  Models surface-layer friction.

    * **obs_alt → blend_alt** (blend zone): linear transition from the
      observed U/V to the GPV-interpolated U/V.

    * **blend_alt and above**: pure multi-level GPV linear interpolation
      on U/V components.

    Zero-wind failsafe
    ~~~~~~~~~~~~~~~~~~
    GPV levels where the reported speed is below **0.5 m/s** are treated as
    calm (zero vector) rather than using a potentially noise-dominated
    direction.  This prevents spurious directional forcing from unreliable
    light-wind model output.

    Args:
        gpv_levels:  Multi-level GPV data.  Each element may be a
                     :class:`WindLevel` or any sequence
                     ``(alt_m, speed_ms, dir_deg)``.  Levels are sorted
                     internally; order is irrelevant.
        obs_speed:   Observed surface wind speed (m/s).
        obs_dir:     Observed surface wind direction (degrees FROM).
        obs_alt:     Height of the surface observation (m AGL).
                     Default 3 m (standard anemometer height).
        blend_alt:   Altitude above which GPV has full weight (m AGL).
                     Default 100 m.

    Returns:
        ``(u_prof, v_prof)`` — two lists of ``(alt_m, value_m_s)`` tuples,
        sorted by ascending altitude, ready for RocketPy's
        ``set_atmospheric_model(type='custom_atmosphere', ...)``.
    """
    # ── Parse and normalise GPV levels into (alt_m, u, v) ────────────────────
    gpv_uv: list[tuple[float, float, float]] = []
    for lvl in gpv_levels:
        if isinstance(lvl, (list, tuple)):
            alt_m    = float(lvl[0])
            speed_ms = float(lvl[1])
            dir_deg  = float(lvl[2])
        else:
            alt_m    = float(lvl.alt_m)
            speed_ms = float(lvl.speed_ms)
            dir_deg  = float(lvl.dir_deg)

        # Zero-wind failsafe: GPV speeds below 0.5 m/s have unreliable
        # direction data; treat them as calm rather than injecting noise.
        if speed_ms < 0.5:
            gpv_uv.append((alt_m, 0.0, 0.0))
        else:
            u, v = speed_dir_to_uv(speed_ms, dir_deg)
            gpv_uv.append((alt_m, u, v))

    gpv_uv.sort(key=lambda t: t[0])

    # Observed surface U/V at obs_alt
    obs_u, obs_v = speed_dir_to_uv(max(obs_speed, 0.0), obs_dir)

    # ── GPV interpolation closure (linear on U/V) ─────────────────────────────
    def _gpv_at(alt: float) -> tuple[float, float]:
        if not gpv_uv:
            return obs_u, obs_v
        if alt <= gpv_uv[0][0]:
            return gpv_uv[0][1], gpv_uv[0][2]
        if alt >= gpv_uv[-1][0]:
            return gpv_uv[-1][1], gpv_uv[-1][2]
        for i in range(len(gpv_uv) - 1):
            lo_alt, lo_u, lo_v = gpv_uv[i]
            hi_alt, hi_u, hi_v = gpv_uv[i + 1]
            if lo_alt <= alt <= hi_alt:
                span = hi_alt - lo_alt
                if span < 1e-9:
                    return lo_u, lo_v
                frac = (alt - lo_alt) / span
                return (lo_u + frac * (hi_u - lo_u),
                        lo_v + frac * (hi_v - lo_v))
        return gpv_uv[-1][1], gpv_uv[-1][2]

    # ── Build altitude grid ───────────────────────────────────────────────────
    alt_set: set[float] = {0.0, float(obs_alt), float(blend_alt)}
    for alt_m, _, _ in gpv_uv:
        alt_set.add(float(alt_m))
    # Guarantee coverage to at least 5 000 m AGL for long-apogee flights
    max_gpv_alt = gpv_uv[-1][0] if gpv_uv else blend_alt
    if max_gpv_alt < 5_000.0:
        alt_set.add(5_000.0)

    # ── Evaluate U/V at every grid altitude ──────────────────────────────────
    u_prof: list[tuple[float, float]] = []
    v_prof: list[tuple[float, float]] = []

    for alt in sorted(alt_set):
        if alt <= 0.0:
            u, v = 0.0, 0.0                               # surface: no wind
        elif alt <= obs_alt:
            # Linear ramp from zero (ground) to observation
            frac = alt / obs_alt
            u, v = obs_u * frac, obs_v * frac
        elif alt <= blend_alt:
            # Smooth transition: observation → GPV
            frac   = (alt - obs_alt) / max(blend_alt - obs_alt, 1e-9)
            gu, gv = _gpv_at(alt)
            u = obs_u + frac * (gu - obs_u)
            v = obs_v + frac * (gv - obs_v)
        else:
            # Pure GPV interpolation in U/V space — no crossover bug
            u, v = _gpv_at(alt)

        u_prof.append((alt, u))
        v_prof.append((alt, v))

    return u_prof, v_prof
