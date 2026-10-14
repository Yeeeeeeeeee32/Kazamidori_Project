"""Pressure levels to metres above the station, using a lapse-rate atmosphere."""
import math


def pressure_altitude_agl(pressure_hpa, station_pressure_pa, temperature_c=15.0):
    """Return estimated AGL height, not MSL altitude or measured geopotential.

    Station pressure is the actual sensor pressure, NOT sea-level QNH.
    Uses dry air R=287.05 J/(kg K), g=9.80665 m/s², lapse=0.0065 K/m.
    """
    p, p0, t = float(pressure_hpa) * 100, float(station_pressure_pa), float(temperature_c) + 273.15
    if not all(math.isfinite(v) for v in (p, p0, t)) or not (0 < p <= 120000 and 0 < p0 <= 120000 and 150 < t < 350):
        raise ValueError('Invalid pressure level, station pressure or temperature')
    return t / .0065 * (1 - (p / p0) ** (287.05 * .0065 / 9.80665))
