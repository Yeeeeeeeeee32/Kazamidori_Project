"""
monitor/wind_tracker.py
Real-time wind data accumulator and statistics provider.

All operations are O(n) or O(1) over the internal deque — no heavy
computation, no I/O, no tkinter.  Safe to call from the main thread at
1 Hz without blocking the UI event loop.

Public API
----------
WindTracker(maxlen=300)
    Stateful accumulator.  Append one sample per second with push().

WindTracker.push(t_sec, speed_mps, surf_dir_deg, up_spd_mps, up_dir_deg)
    Add a new wind observation.

WindTracker.stats(window_sec=None) -> WindStats
    Return rolling statistics for the past window_sec seconds
    (or the full history if window_sec is None).

WindTracker.recent_avg(window_sec=10.0) -> float
    Fast helper: surface-speed mean over the last window_sec seconds.
    Returns the latest single speed value if the window is empty.

WindTracker.snapshot() -> dict | None
    Return the last wind observation as a plain dict, or None.

WindTracker.gust() -> float
    Peak surface speed seen in the entire history (0.0 if empty).

WindTracker.clear()
    Reset all history.

WindStats (dataclass)
    mu_surf       float   mean surface speed (m/s)
    sigma_surf    float   std-dev of surface speed (m/s)
    mu_up         float   mean upper-level speed (m/s)
    sigma_up      float   std-dev of upper-level speed (m/s)
    surf_dir_mean float   circular mean of surface direction (deg)
    up_dir_mean   float   circular mean of upper direction (deg)
    n             int     number of samples in window
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WindSample:
    """One wind observation at a specific elapsed time."""
    t_sec:        float   # seconds since wind tracking started
    speed_mps:    float   # surface wind speed (m/s)
    surf_dir_deg: float   # surface wind direction (degrees, met convention)
    up_spd_mps:   float   # upper-level wind speed (m/s)
    up_dir_deg:   float   # upper-level wind direction (degrees)


@dataclass(frozen=True)
class WindStats:
    """Descriptive statistics over a window of WindSample objects."""
    mu_surf:       float   # mean surface speed (m/s)
    sigma_surf:    float   # std-dev of surface speed (m/s)
    mu_up:         float   # mean upper-level speed (m/s)
    sigma_up:      float   # std-dev of upper-level speed (m/s)
    surf_dir_mean: float   # circular mean of surface direction (deg)
    up_dir_mean:   float   # circular mean of upper direction (deg)
    n:             int     # number of samples in the window


# ── Circular-mean helper ──────────────────────────────────────────────────────

def _circular_mean_deg(angles_deg: list[float]) -> float:
    """Return the circular mean of a list of angles (degrees).

    Uses the standard unit-vector approach so wrap-around (e.g. 350° and 10°)
    is handled correctly.  Returns 0.0 for an empty list.
    """
    if not angles_deg:
        return 0.0
    sin_sum = sum(math.sin(math.radians(a)) for a in angles_deg)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles_deg)
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0


def _std(values: list[float]) -> float:
    """Population std-dev with Bessel's correction (ddof=1).

    Returns 0.0 for a list with fewer than two elements.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def _angle_diff(a: float, b: float) -> float:
    """Return the unsigned angular difference between two compass bearings.

    Result is in [0, 180] degrees.
    """
    return abs(((a - b) + 180.0) % 360.0 - 180.0)


# ── WindTracker ───────────────────────────────────────────────────────────────

class WindTracker:
    """Rolling wind-data accumulator with fast statistics.

    Designed to be called at approximately 1 Hz from the tkinter main loop.
    All methods complete in O(n) time where n ≤ maxlen (default 300 samples
    = 5 minutes at 1 Hz).

    Args:
        maxlen: Maximum number of samples to retain (oldest are discarded).
    """

    # Speed and direction tolerances for wind-change detection
    SPEED_TOLERANCE_MPS: float = 2.0   # m/s absolute
    DIR_TOLERANCE_DEG:   float = 15.0  # degrees

    def __init__(self, maxlen: int = 300) -> None:
        self._maxlen  = maxlen
        self._history: deque[WindSample] = deque(maxlen=maxlen)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def push(
        self,
        t_sec:        float,
        speed_mps:    float,
        surf_dir_deg: float  = 0.0,
        up_spd_mps:   float  = 0.0,
        up_dir_deg:   float  = 0.0,
    ) -> None:
        """Append one wind observation to the rolling buffer.

        Args:
            t_sec:        Elapsed seconds since tracking started.
            speed_mps:    Surface wind speed in m/s (non-negative).
            surf_dir_deg: Surface wind direction in degrees (0–360).
            up_spd_mps:   Upper-level wind speed in m/s.
            up_dir_deg:   Upper-level wind direction in degrees.
        """
        self._history.append(WindSample(
            t_sec        = float(t_sec),
            speed_mps    = max(0.0, float(speed_mps)),
            surf_dir_deg = float(surf_dir_deg) % 360.0,
            up_spd_mps   = max(0.0, float(up_spd_mps)),
            up_dir_deg   = float(up_dir_deg) % 360.0,
        ))

    def clear(self) -> None:
        """Discard all stored history."""
        self._history.clear()

    # ── Queries ───────────────────────────────────────────────────────────────

    def stats(self, window_sec: Optional[float] = None) -> Optional[WindStats]:
        """Return rolling statistics for the past window_sec seconds.

        If *window_sec* is None, all stored samples are used.

        Returns None if there are no samples in the requested window.
        """
        samples = self._window_samples(window_sec)
        if not samples:
            return None

        speeds    = [s.speed_mps    for s in samples]
        up_speeds = [s.up_spd_mps   for s in samples]
        s_dirs    = [s.surf_dir_deg for s in samples]
        u_dirs    = [s.up_dir_deg   for s in samples]

        return WindStats(
            mu_surf       = sum(speeds)    / len(speeds),
            sigma_surf    = _std(speeds),
            mu_up         = sum(up_speeds) / len(up_speeds),
            sigma_up      = _std(up_speeds),
            surf_dir_mean = _circular_mean_deg(s_dirs),
            up_dir_mean   = _circular_mean_deg(u_dirs),
            n             = len(samples),
        )

    def recent_avg(self, window_sec: float = 10.0) -> float:
        """Return the mean surface speed over the past window_sec seconds.

        Falls back to the most recent single sample if the window contains
        no samples.  Returns 0.0 if the history is completely empty.
        """
        if not self._history:
            return 0.0
        samples = self._window_samples(window_sec)
        if not samples:
            return float(self._history[-1].speed_mps)
        return sum(s.speed_mps for s in samples) / len(samples)

    def snapshot(self) -> Optional[dict]:
        """Return the most recent observation as a plain dict, or None."""
        if not self._history:
            return None
        s = self._history[-1]
        return {
            't_sec':        s.t_sec,
            'speed_mps':    s.speed_mps,
            'surf_dir_deg': s.surf_dir_deg,
            'up_spd_mps':   s.up_spd_mps,
            'up_dir_deg':   s.up_dir_deg,
        }

    def gust(self) -> float:
        """Return the peak surface speed recorded in the entire history."""
        if not self._history:
            return 0.0
        return max(s.speed_mps for s in self._history)

    def time_series(self) -> list[tuple[float, float]]:
        """Return the full surface-speed time series as [(t_sec, speed), …]."""
        return [(s.t_sec, s.speed_mps) for s in self._history]

    def check_drift(
        self,
        baseline: dict,
    ) -> tuple[bool, dict]:
        """Compare the current 10-second rolling average against a baseline.

        Uses the same 10-second window on both sides so only sustained
        changes trigger an alert (single-tick noise is suppressed).

        Args:
            baseline: dict with keys surf_spd, surf_dir, up_spd, up_dir
                      (as produced by capture_baseline()).

        Returns:
            (exceeded, deltas) where:
            exceeded — True if any tolerance was crossed.
            deltas   — dict with keys surf_spd_diff, up_spd_diff,
                       surf_dir_diff, up_dir_diff plus the raw current
                       values (cur_surf_spd, cur_up_spd).
        """
        cur_spd = self.recent_avg(window_sec=10.0)
        snap    = self.snapshot() or {}
        cur_surf_dir = snap.get('surf_dir_deg', baseline.get('surf_dir', 0.0))
        cur_up_spd   = snap.get('up_spd_mps',  baseline.get('up_spd',   0.0))
        cur_up_dir   = snap.get('up_dir_deg',  baseline.get('up_dir',   0.0))

        surf_spd_diff = abs(cur_spd      - baseline.get('surf_spd', cur_spd))
        up_spd_diff   = abs(cur_up_spd   - baseline.get('up_spd',   cur_up_spd))
        surf_dir_diff = _angle_diff(cur_surf_dir, baseline.get('surf_dir', cur_surf_dir))
        up_dir_diff   = _angle_diff(cur_up_dir,   baseline.get('up_dir',   cur_up_dir))

        exceeded = (
            surf_spd_diff > self.SPEED_TOLERANCE_MPS or
            up_spd_diff   > self.SPEED_TOLERANCE_MPS or
            surf_dir_diff > self.DIR_TOLERANCE_DEG   or
            up_dir_diff   > self.DIR_TOLERANCE_DEG
        )
        deltas = {
            'cur_surf_spd':   cur_spd,
            'cur_up_spd':     cur_up_spd,
            'cur_surf_dir':   cur_surf_dir,
            'cur_up_dir':     cur_up_dir,
            'surf_spd_diff':  surf_spd_diff,
            'up_spd_diff':    up_spd_diff,
            'surf_dir_diff':  surf_dir_diff,
            'up_dir_diff':    up_dir_diff,
        }
        return exceeded, deltas

    def capture_baseline(self) -> dict:
        """Snapshot the current 10-second rolling average as a baseline dict.

        Returns a dict with keys surf_spd, surf_dir, up_spd, up_dir
        ready to be stored and later passed to check_drift().
        """
        snap = self.snapshot() or {}
        return {
            'surf_spd': self.recent_avg(window_sec=10.0),
            'surf_dir': snap.get('surf_dir_deg', 0.0),
            'up_spd':   snap.get('up_spd_mps',  0.0),
            'up_dir':   snap.get('up_dir_deg',  0.0),
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _window_samples(
        self, window_sec: Optional[float]
    ) -> list[WindSample]:
        """Return samples within the past window_sec seconds.

        If window_sec is None, return all samples.
        """
        history = list(self._history)
        if not history:
            return []
        if window_sec is None:
            return history
        t_latest = history[-1].t_sec
        cutoff   = t_latest - window_sec
        return [s for s in history if s.t_sec >= cutoff]
