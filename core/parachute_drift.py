"""Prescribed horizontal advection after canopy opening; vertical flight is unchanged."""
import math
import numpy as np


def apply_parachute_drift(times, east, north, altitude, opening_time, u_profile, v_profile, ratio):
    """Integrate v_horizontal = ratio * local ENU wind after opening.

    This is an operator-selected kinematic approximation, not a canopy force
    model. None retains the original RocketPy horizontal dynamics.
    """
    if ratio is None:
        return east, north
    ratio = float(ratio)
    if not math.isfinite(ratio) or not 0 <= ratio <= 2:
        raise ValueError('Parachute wind drift ratio must be between 0 and 2')
    t, z = np.asarray(times), np.asarray(altitude)
    x, y = np.array(east, dtype=float, copy=True), np.array(north, dtype=float, copy=True)
    if len(t) < 2 or opening_time >= t[-1]:
        return x, y
    start = max(float(t[0]), float(opening_time))
    for out, profile in ((x, u_profile), (y, v_profile)):
        nodes = np.asarray(profile, dtype=float)
        velocities = ratio * np.interp(z, nodes[:, 0], nodes[:, 1])
        initial = float(np.interp(start, t, out))
        previous_t = start
        previous_v = float(np.interp(start, t, velocities))
        for i in np.flatnonzero(t > start):
            initial += .5 * (previous_v + velocities[i]) * (t[i] - previous_t)
            out[i] = initial
            previous_t, previous_v = t[i], velocities[i]
    return x, y
