#!/usr/bin/env python3
"""
utils/map_downloader.py

Phase 1: Online Downloader Tool
Downloads Web Mercator tiles for a 500x500m bounding box
around a target latitude/longitude, calculates magnetic declination,
and saves the tiles and metadata for offline use.
"""

import argparse
import math
import os
import json
import time

try:
    import geomag
except ImportError:
    geomag = None

try:
    import requests
except ImportError:
    requests = None


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[float, float]:
    """Convert lat/lon to precise floating-point tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return (xtile, ytile)


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """Return (m_per_deg_lat, m_per_deg_lon) at the given geodetic latitude."""
    phi = math.radians(lat_deg)
    m_per_deg_lat = (111132.92
                     - 559.82  * math.cos(2 * phi)
                     + 1.175   * math.cos(4 * phi)
                     - 0.0023  * math.cos(6 * phi))
    m_per_deg_lon = (111412.84 * math.cos(phi)
                     - 93.5    * math.cos(3 * phi)
                     + 0.118   * math.cos(5 * phi))
    return m_per_deg_lat, m_per_deg_lon

def offset_to_latlon(lat0: float, lon0: float, dx_east: float, dy_north: float) -> tuple[float, float]:
    m_lat, m_lon = meters_per_degree(lat0)
    return (lat0 + dy_north / m_lat, lon0 + dx_east / m_lon)


def calculate_bounds(lat: float, lon: float, size_m: float = 500.0) -> dict:
    """Calculate the bounding box in Web Mercator tiles for a region."""
    half_size = size_m / 2.0

    # Calculate corner lat/lons
    north_lat, _ = offset_to_latlon(lat, lon, 0, half_size)
    south_lat, _ = offset_to_latlon(lat, lon, 0, -half_size)
    _, east_lon = offset_to_latlon(lat, lon, half_size, 0)
    _, west_lon = offset_to_latlon(lat, lon, -half_size, 0)

    return {
        "north": north_lat,
        "south": south_lat,
        "east": east_lon,
        "west": west_lon
    }

def main():
    parser = argparse.ArgumentParser(description="Download offline map tiles and calculate magnetic declination.")
    parser.add_argument("--lat", type=float, required=True, help="Target Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Target Longitude")
    parser.add_argument("--zoom", type=int, default=18, help="Zoom level (default: 18)")

    args = parser.parse_args()

    print(f"Target Coordinates: {args.lat}, {args.lon}")
    bounds = calculate_bounds(args.lat, args.lon, 500.0)
    print(f"Bounding Box (Lat/Lon): {bounds}")

    zoom = args.zoom

    # Get exact tile float coords for the corners
    x_min, y_max = deg2num(bounds['south'], bounds['west'], zoom) # y increases southwards
    x_max, y_min = deg2num(bounds['north'], bounds['east'], zoom)

    # Convert to integer tile indices
    tile_x_min = int(math.floor(x_min))
    tile_x_max = int(math.floor(x_max))
    tile_y_min = int(math.floor(y_min))
    tile_y_max = int(math.floor(y_max))

    print(f"Tile Range X: {tile_x_min} to {tile_x_max}")
    print(f"Tile Range Y: {tile_y_min} to {tile_y_max}")

    output_dir = "assets/offline_map"
    os.makedirs(output_dir, exist_ok=True)

    if requests is None:
        print("Error: 'requests' module is missing. Please install it with 'pip install requests'")
        return

    session = requests.Session()
    # OSM requires a User-Agent
    session.headers.update({
        "User-Agent": "KazamidoriProject/1.0 (offline map downloader)"
    })

    print("Downloading tiles...")
    downloaded_tiles = []

    for x in range(tile_x_min, tile_x_max + 1):
        for y in range(tile_y_min, tile_y_max + 1):
            tile_path = os.path.join(output_dir, str(zoom), str(x), f"{y}.png")
            os.makedirs(os.path.dirname(tile_path), exist_ok=True)

            if not os.path.exists(tile_path):
                url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                try:
                    response = session.get(url, timeout=10)
                    response.raise_for_status()
                    with open(tile_path, "wb") as f:
                        f.write(response.content)
                    print(f"  Downloaded: {url}")
                    # Be nice to OSM servers
                    time.sleep(0.1)
                except Exception as e:
                    print(f"  Failed to download {url}: {e}")
            else:
                print(f"  Cached: {zoom}/{x}/{y}.png")

            downloaded_tiles.append({"x": x, "y": y, "z": zoom})

    declination = 0.0
    if geomag is not None:
        try:
            declination = geomag.declination(args.lat, args.lon)
            print(f"Magnetic Declination: {declination:.4f} degrees")
        except Exception as e:
            print(f"Error calculating magnetic declination: {e}")
    else:
        print("Warning: 'geomag' module not available. Magnetic declination set to 0.0.")

    meta = {
        "center_lat": args.lat,
        "center_lon": args.lon,
        "magnetic_declination": declination,
        "zoom_level": zoom,
        "extent_meters": [-250.0, 250.0, -250.0, 250.0],
        "tile_bounds": {
            "x_min": tile_x_min,
            "x_max": tile_x_max,
            "y_min": tile_y_min,
            "y_max": tile_y_max
        }
    }

    meta_path = os.path.join(output_dir, "map_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    main()
