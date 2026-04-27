"""
utils/geo_math.py
Coordinate conversions and geometric helpers.

All functions are pure (no class state, no tkinter dependencies).
Units: meters for distances, degrees for lat/lon, radians for angles
where noted.
"""

import math


# ── Geodetic helpers ─────────────────────────────────────────────────────────

def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """Return (m_per_deg_lat, m_per_deg_lon) at the given geodetic latitude.

    Uses the WGS-84 approximation.
    """
    phi = math.radians(lat_deg)
    m_per_deg_lat = (111132.92
                     - 559.82  * math.cos(2 * phi)
                     + 1.175   * math.cos(4 * phi)
                     - 0.0023  * math.cos(6 * phi))
    m_per_deg_lon = (111412.84 * math.cos(phi)
                     - 93.5    * math.cos(3 * phi)
                     + 0.118   * math.cos(5 * phi))
    return m_per_deg_lat, m_per_deg_lon


def offset_to_latlon(
    lat0: float, lon0: float,
    dx_east: float, dy_north: float,
) -> tuple[float, float]:
    """Convert a local Cartesian offset (metres, East/North) to (lat, lon).

    Args:
        lat0, lon0: Reference origin in decimal degrees.
        dx_east:    East displacement in metres (positive = East).
        dy_north:   North displacement in metres (positive = North).

    Returns:
        (lat, lon) in decimal degrees.
    """
    m_lat, m_lon = meters_per_degree(lat0)
    return (lat0 + dy_north / m_lat,
            lon0 + dx_east  / m_lon)


def latlon_to_offset(
    lat0: float, lon0: float,
    lat: float,  lon: float,
) -> tuple[float, float]:
    """Convert (lat, lon) to local Cartesian offset (metres) from origin.

    Returns:
        (dx_east, dy_north) in metres.
    """
    m_lat, m_lon = meters_per_degree(lat0)
    return ((lon - lon0) * m_lon,
            (lat - lat0) * m_lat)


# ── Polygon builders ─────────────────────────────────────────────────────────

def circle_polygon(
    lat: float, lon: float,
    radius_m: float,
    n: int = 36,
) -> list[tuple[float, float]]:
    """Return a list of (lat, lon) vertices approximating a circle.

    Uses a simple spherical approximation (good to <0.1 % for r < 10 km).

    Args:
        lat, lon:  Centre in decimal degrees.
        radius_m:  Radius in metres.
        n:         Number of vertices (default 36).
    """
    R_EARTH = 6_378_137.0
    coords = []
    for i in range(n):
        angle = math.pi * 2 * i / n
        dx = radius_m * math.cos(angle)
        dy = radius_m * math.sin(angle)
        d_lat = (dy / R_EARTH) * (180.0 / math.pi)
        d_lon = (dx / (R_EARTH * math.cos(math.pi * lat / 180.0))) * (180.0 / math.pi)
        coords.append((lat + d_lat, lon + d_lon))
    return coords


def ellipse_polygon(
    ref_lat: float, ref_lon: float,
    cx: float, cy: float,
    a: float, b: float,
    angle_rad: float,
    n: int = 60,
) -> list[tuple[float, float]]:
    """Return (lat, lon) vertices for a rotated ellipse in local coords.

    The ellipse is defined in the local East-North plane centred at
    (ref_lat, ref_lon).  The local origin is the reference point;
    cx/cy shift the ellipse centre (metres, East/North).

    Args:
        ref_lat, ref_lon: Reference origin in decimal degrees.
        cx, cy:   Ellipse centre offset from origin (metres, East/North).
        a:        Semi-major axis (metres).
        b:        Semi-minor axis (metres).
        angle_rad: Rotation of the major axis from East (radians).
        n:        Number of vertices (default 60).
    """
    coords = []
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    for i in range(n):
        t  = 2.0 * math.pi * i / n
        xe = a * math.cos(t) * ca - b * math.sin(t) * sa
        ye = a * math.cos(t) * sa + b * math.sin(t) * ca
        lat, lon = offset_to_latlon(ref_lat, ref_lon, cx + xe, cy + ye)
        coords.append((lat, lon))
    return coords


# ── Bounding-box helpers ─────────────────────────────────────────────────────

def ring_extents(
    lat: float, lon: float,
    radius_m: float,
) -> tuple[float, float, float, float]:
    """Return (lat_min, lat_max, lon_min, lon_max) for a circle at (lat, lon).

    Useful for computing map fit bounds.
    """
    m_lat, m_lon = meters_per_degree(lat)
    dlat = radius_m / m_lat
    dlon = radius_m / m_lon
    return (lat - dlat, lat + dlat, lon - dlon, lon + dlon)
