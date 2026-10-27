"""Team landing constraint; ground-relative speed, not pendulum dynamics."""
import math
import numpy as np


DEFAULTS = dict(max_speed_ms=3.0, window_s=3.0, max_variation_ms=0.5)


def is_montblanc(mode):
    return 'モンブラン' in str(mode) or 'montblanc' in str(mode).lower()


def normalize_options(options=None):
    values = dict(DEFAULTS)
    if options is not None:
        if not isinstance(options, dict) or set(options) - set(DEFAULTS):
            raise ValueError('Invalid Montblanc settings')
        values.update(options)
    for key, value in values.items():
        if isinstance(value, bool):
            raise ValueError(f'Montblanc {key} must be numeric')
        value = float(value)
        lo, hi = (0.01, 120.) if key == 'window_s' else (0., 1000.)
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'Montblanc {key} must be between {lo} and {hi}')
        values[key] = value
    return values


def landing_metrics(times, vx, vy, opening_time, options):
    """Evaluate all solver knots and the exact window boundary.

    Velocity components are linearly interpolated between knots. Speed maxima
    occur at endpoints; minima also include the closest point on each vector
    segment, so rotating vectors cannot hide a dip between solver samples.
    The full window must be after canopy opening. No zero-speed sample after
    ground contact is introduced.
    """
    opts = normalize_options(options)
    t, x, y = (np.asarray(a, dtype=float) for a in (times, vx, vy))
    if (t.ndim != 1 or len(t) < 2 or x.shape != t.shape or y.shape != t.shape
            or not all(np.all(np.isfinite(a)) for a in (t, x, y))
            or np.any(np.diff(t) <= 0) or not math.isfinite(opening_time)):
        raise ValueError('Invalid horizontal velocity history')
    start = float(t[-1]) - opts['window_s']
    covered = start >= max(float(t[0]), float(opening_time)) - 1e-9
    sample_t = np.r_[max(start, float(t[0])), t[t > max(start, float(t[0]))]]
    vectors = np.column_stack((np.interp(sample_t, t, x), np.interp(sample_t, t, y)))
    speeds = np.linalg.norm(vectors, axis=1)
    delta = np.diff(vectors, axis=0)
    denom = np.sum(delta * delta, axis=1)
    alpha = np.clip(np.divide(-np.sum(vectors[:-1] * delta, axis=1), denom,
                              out=np.zeros_like(denom), where=denom > 0), 0, 1)
    minima = np.linalg.norm(vectors[:-1] + alpha[:, None] * delta, axis=1)
    maximum = float(speeds.max())
    variation = maximum - float(min(speeds.min(), minima.min()))
    return dict(landing_horizontal_speed_ms=float(speeds[-1]),
                landing_window_max_speed_ms=maximum,
                landing_speed_variation_ms=variation,
                landing_window_covered=bool(covered),
                montblanc_pass=bool(covered and maximum <= opts['max_speed_ms'] + 1e-9
                                    and variation <= opts['max_variation_ms'] + 1e-9))
