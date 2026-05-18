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

    from PIL import Image
    import io

    # To stitch tiles, we need a base image
    width_tiles = tile_x_max - tile_x_min + 1
    height_tiles = tile_y_max - tile_y_min + 1
    tile_size = 256

    stitched_image = Image.new("RGB", (width_tiles * tile_size, height_tiles * tile_size))

    for x in range(tile_x_min, tile_x_max + 1):
        for y in range(tile_y_min, tile_y_max + 1):
            tile_path = os.path.join(output_dir, str(zoom), str(x), f"{y}.png")
            os.makedirs(os.path.dirname(tile_path), exist_ok=True)

            img_data = None
            if not os.path.exists(tile_path):
                url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                try:
                    response = session.get(url, timeout=10)
                    response.raise_for_status()
                    img_data = response.content
                    with open(tile_path, "wb") as f:
                        f.write(img_data)
                    print(f"  Downloaded: {url}")
                    # Be nice to OSM servers
                    time.sleep(0.1)
                except Exception as e:
                    print(f"  Failed to download {url}: {e}")
            else:
                print(f"  Cached: {zoom}/{x}/{y}.png")
                with open(tile_path, "rb") as f:
                    img_data = f.read()

            if img_data:
                try:
                    tile_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    px = (x - tile_x_min) * tile_size
                    py = (y - tile_y_min) * tile_size
                    stitched_image.paste(tile_img, (px, py))
                except Exception as e:
                    print(f"  Failed to process tile image {x}/{y}: {e}")

            downloaded_tiles.append({"x": x, "y": y, "z": zoom})

    # Save stitched image
    # We must crop it so that the 500x500m area is exactly centered and bounded correctly
    # But wait, it's easier to just save it and let map_view scale it.
    # However, to be precise, the tiles cover more than 500x500.
    # We need to map the tile boundaries to ENU coordinates relative to the center,
    # and save that actual extent in map_meta.json, OR we crop it to exactly 500x500m.
    # Let's save the actual extent in map_meta.json based on the exact lat/lons of the stitched image edges.

    n_tiles = 2.0 ** zoom
    lon_left = tile_x_min / n_tiles * 360.0 - 180.0
    lon_right = (tile_x_max + 1) / n_tiles * 360.0 - 180.0

    lat_rad_top = math.atan(math.sinh(math.pi * (1 - 2 * tile_y_min / n_tiles)))
    lat_top = math.degrees(lat_rad_top)

    lat_rad_bottom = math.atan(math.sinh(math.pi * (1 - 2 * (tile_y_max + 1) / n_tiles)))
    lat_bottom = math.degrees(lat_rad_bottom)

    def latlon_to_offset_inline(lat0, lon0, lat, lon):
        phi = math.radians(lat0)
        m_lat = (111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi) - 0.0023 * math.cos(6 * phi))
        m_lon = (111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi) + 0.118 * math.cos(5 * phi))
        return ((lon - lon0) * m_lon, (lat - lat0) * m_lat)

    dx_left, dy_bottom = latlon_to_offset_inline(args.lat, args.lon, lat_bottom, lon_left)
    dx_right, dy_top = latlon_to_offset_inline(args.lat, args.lon, lat_top, lon_right)

    actual_extent = [dx_left, dx_right, dy_bottom, dy_top]

    background_path = os.path.join(output_dir, "background.png")
    stitched_image.save(background_path)
    print(f"Saved stitched background map to {background_path}")

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
        "extent_meters": actual_extent,
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
