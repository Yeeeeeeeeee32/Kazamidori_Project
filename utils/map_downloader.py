import os
import math
import json
import argparse

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
    import io
except ImportError:
    Image = None

try:
    import geomag
except ImportError:
    geomag = None

def latlon_to_tile(lat_deg, lon_deg, zoom):
    """Returns the tile coordinates (x, y) for a given lat/lon and zoom level."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def tile_to_latlon(xtile, ytile, zoom):
    """Returns the NW-corner lat/lon of the tile."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def latlon_to_enu(lat, lon, clat, clon):
    """Convert a lat/lon to an ENU offset from a center lat/lon."""
    R_EARTH = 6378137.0
    dlat = lat - clat
    dlon = lon - clon
    dy = dlat * (math.pi / 180.0) * R_EARTH
    dx = dlon * (math.pi / 180.0) * R_EARTH * math.cos(math.radians(clat))
    return dx, dy

def enu_to_latlon(dx, dy, clat, clon):
    """Convert an ENU offset from a center lat/lon back to lat/lon."""
    R_EARTH = 6378137.0
    dlat = (dy / R_EARTH) * (180.0 / math.pi)
    dlon = (dx / (R_EARTH * math.cos(math.radians(clat)))) * (180.0 / math.pi)
    lat = clat + dlat
    lon = clon + dlon
    return lat, lon

def main():
    parser = argparse.ArgumentParser(description="Download offline map tiles and calculate magnetic declination.")
    parser.add_argument("--lat", type=float, required=True, help="Target Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Target Longitude")
    parser.add_argument("--radius", type=float, default=250.0, help="Target radius in meters (default: 250m for 500x500 box)")
    parser.add_argument("--zoom", type=int, default=18, help="Zoom level (default: 18)")

    args = parser.parse_args()

    center_lat = args.lat
    center_lon = args.lon
    radius = args.radius
    zoom = args.zoom

    print(f"Target Coordinates: {center_lat}, {center_lon}")
    print(f"Target Radius: {radius}m (Bounding Box: {-radius} to {radius})")

    # ENU bounds
    extent_meters = [-radius, radius, -radius, radius]

    # Northwest corner (x_min, y_max in ENU -> -radius, radius)
    nw_lat, nw_lon = enu_to_latlon(-radius, radius, center_lat, center_lon)

    # Southeast corner (x_max, y_min in ENU -> radius, -radius)
    se_lat, se_lon = enu_to_latlon(radius, -radius, center_lat, center_lon)

    # Get tile bounds
    min_x_tile, min_y_tile = latlon_to_tile(nw_lat, nw_lon, zoom)
    max_x_tile, max_y_tile = latlon_to_tile(se_lat, se_lon, zoom)

    if min_x_tile > max_x_tile: min_x_tile, max_x_tile = max_x_tile, min_x_tile
    if min_y_tile > max_y_tile: min_y_tile, max_y_tile = max_y_tile, min_y_tile

    total_tiles = (max_x_tile - min_x_tile + 1) * (max_y_tile - min_y_tile + 1)
    print(f"Tile Range X: {min_x_tile} to {max_x_tile}")
    print(f"Tile Range Y: {min_y_tile} to {max_y_tile}")
    print(f"Total Tiles: {total_tiles}")

    if total_tiles > 100:
        print(f"Error: Requested bounding box requires too many tiles ({total_tiles}). Please use a smaller radius or lower zoom level.")
        return

    out_dir = "assets/offline_map"
    os.makedirs(out_dir, exist_ok=True)

    if requests is None or Image is None:
        print("Error: Missing required libraries: 'requests' or 'Pillow'. Please install them.")
        return

    session = requests.Session()
    # OSM requires a User-Agent
    session.headers.update({
        "User-Agent": "KazamidoriProject/1.0 (offline map downloader)"
    })

    print("Downloading tiles...")
    tiles_data = {}
    downloaded = 0

    tile_size = 256
    import time

    for x in range(min_x_tile, max_x_tile + 1):
        for y in range(min_y_tile, max_y_tile + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            try:
                response = session.get(url, timeout=10)
                response.raise_for_status()

                img = Image.open(io.BytesIO(response.content)).convert("RGBA")
                tiles_data[(x, y)] = img

                downloaded += 1
                print(f"  Downloaded tile {x},{y} ({downloaded}/{total_tiles})")
                time.sleep(0.1) # Be nice to OSM
            except Exception as e:
                print(f"  Failed to download tile {x},{y}: {e}")
                downloaded += 1

    if not tiles_data:
        print("Error: Failed to download any map tiles.")
        return

    # Stitch the tiles together
    stitched_w = (max_x_tile - min_x_tile + 1) * tile_size
    stitched_h = (max_y_tile - min_y_tile + 1) * tile_size

    stitched_img = Image.new('RGBA', (stitched_w, stitched_h), (0, 0, 0, 0))

    for (x, y), img in tiles_data.items():
        px = (x - min_x_tile) * tile_size
        py = (y - min_y_tile) * tile_size
        stitched_img.paste(img, (px, py))

    # Calculate ENU coordinates of the stitched image boundaries relative to the center
    stitched_nw_lat, stitched_nw_lon = tile_to_latlon(min_x_tile, min_y_tile, zoom)
    stitched_se_lat, stitched_se_lon = tile_to_latlon(max_x_tile + 1, max_y_tile + 1, zoom)

    img_nw_x, img_nw_y = latlon_to_enu(stitched_nw_lat, stitched_nw_lon, center_lat, center_lon)
    img_se_x, img_se_y = latlon_to_enu(stitched_se_lat, stitched_se_lon, center_lat, center_lon)

    # Scale from ENU coordinates to pixels on the stitched image
    x_scale = stitched_w / (img_se_x - img_nw_x)
    y_scale = stitched_h / (img_nw_y - img_se_y) # Since Y decreases from top to bottom

    # Target bounding box in ENU: Left=-radius, Right=radius, Bottom=-radius, Top=radius
    crop_left = int(( -radius - img_nw_x ) * x_scale)
    crop_right = int(( radius - img_nw_x ) * x_scale)

    crop_top = int(( img_nw_y - radius ) * y_scale)
    crop_bottom = int(( img_nw_y - -radius ) * y_scale)

    # Ensure crop boundaries are within image
    crop_left = max(0, crop_left)
    crop_top = max(0, crop_top)
    crop_right = min(stitched_w, crop_right)
    crop_bottom = min(stitched_h, crop_bottom)

    final_img = stitched_img.crop((crop_left, crop_top, crop_right, crop_bottom))
    background_path = os.path.join(out_dir, "background.png")
    final_img.save(background_path)
    print(f"Saved correctly cropped map to {background_path}")

    # Calculate magnetic declination using geomag
    mag_declination = 0.0
    if geomag is not None:
        try:
            mag = geomag.declination(center_lat, center_lon)
            mag_declination = float(mag)
            print(f"Calculated magnetic declination: {mag_declination}")
        except Exception as e:
            print(f"Failed to calculate magnetic declination: {e}")
    else:
        print("geomag library not found. Defaulting magnetic declination to 0.0")

    meta = {
        "lat": center_lat,
        "lon": center_lon,
        "magnetic_declination": mag_declination,
        "extent_meters": extent_meters
    }

    meta_path = os.path.join(out_dir, "map_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Saved metadata to {meta_path}")

if __name__ == "__main__":
    main()
