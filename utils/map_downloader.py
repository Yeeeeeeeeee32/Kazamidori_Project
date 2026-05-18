"""
utils/map_downloader.py
Downloads offline map tiles from OpenStreetMap and stitches them into a background image.
"""

import os
import math
import json

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
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

def download_offline_map(center_lat: float, center_lon: float, zoom: int = 18, progress_callback=None):
    """
    Downloads OSM tiles for a 500m x 500m bounding box centered at the given lat/lon.
    Stitches the tiles into a single background.png and saves metadata.
    """
    if requests is None or Image is None:
        raise RuntimeError("Missing required libraries: 'requests' or 'Pillow'. Please install them.")

    out_dir = "assets/offline_map"
    os.makedirs(out_dir, exist_ok=True)

    # 500m x 500m bounding box means +/- 250m from center.
    # We need to find the lat/lon of the corners.
    # Using a simple spherical approximation.
    R_EARTH = 6378137.0

    # Offset in meters (East and North)
    half_size = 250.0

    # Northwest corner (x_min, y_max in ENU -> -250, 250)
    dy_nw = half_size
    dx_nw = -half_size

    dlat_nw = (dy_nw / R_EARTH) * (180.0 / math.pi)
    dlon_nw = (dx_nw / (R_EARTH * math.cos(math.radians(center_lat)))) * (180.0 / math.pi)
    nw_lat = center_lat + dlat_nw
    nw_lon = center_lon + dlon_nw

    # Southeast corner (x_max, y_min in ENU -> 250, -250)
    dy_se = -half_size
    dx_se = half_size

    dlat_se = (dy_se / R_EARTH) * (180.0 / math.pi)
    dlon_se = (dx_se / (R_EARTH * math.cos(math.radians(center_lat)))) * (180.0 / math.pi)
    se_lat = center_lat + dlat_se
    se_lon = center_lon + dlon_se

    # Get tile bounds
    min_x_tile, min_y_tile = latlon_to_tile(nw_lat, nw_lon, zoom)
    max_x_tile, max_y_tile = latlon_to_tile(se_lat, se_lon, zoom)

    # Ensure min < max
    if min_x_tile > max_x_tile: min_x_tile, max_x_tile = max_x_tile, min_x_tile
    if min_y_tile > max_y_tile: min_y_tile, max_y_tile = max_y_tile, min_y_tile

    total_tiles = (max_x_tile - min_x_tile + 1) * (max_y_tile - min_y_tile + 1)

    # Let's limit the number of tiles to avoid abuse / huge memory usage
    if total_tiles > 100:
        raise ValueError(f"Requested bounding box requires too many tiles ({total_tiles}).")

    tiles_data = {}
    downloaded = 0

    headers = {
        'User-Agent': 'KazamidoriOfflineMapDownloader/1.0'
    }

    for x in range(min_x_tile, max_x_tile + 1):
        for y in range(min_y_tile, max_y_tile + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            try:
                if progress_callback:
                    progress_callback(downloaded, total_tiles, f"Downloading tile {x},{y}")
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                # Save temporarily or keep in memory. We'll keep in memory using Pillow.
                from io import BytesIO
                img = Image.open(BytesIO(response.content)).convert("RGBA")
                tiles_data[(x, y)] = img

                downloaded += 1
            except Exception as e:
                # If a tile fails, we'll just ignore it or log it, leaving a blank spot
                print(f"Failed to download tile {x},{y}: {e}")
                downloaded += 1 # advance progress anyway

    if not tiles_data:
        raise RuntimeError("Failed to download any map tiles.")

    # Stitch the tiles together
    tile_w = 256
    tile_h = 256

    stitched_w = (max_x_tile - min_x_tile + 1) * tile_w
    stitched_h = (max_y_tile - min_y_tile + 1) * tile_h

    stitched_img = Image.new('RGBA', (stitched_w, stitched_h), (0, 0, 0, 0))

    for (x, y), img in tiles_data.items():
        px = (x - min_x_tile) * tile_w
        py = (y - min_y_tile) * tile_h
        stitched_img.paste(img, (px, py))

    # Now we have the stitched map image. It covers from tile_to_latlon(min_x_tile, min_y_tile) to tile_to_latlon(max_x_tile+1, max_y_tile+1)
    # The user requires exactly the 500x500 ENU bounding box from the center point, which we will achieve during plotting
    # using Matplotlib's `extent=[-250, 250, -250, 250]`.
    # To do this correctly, we must crop the stitched image to exactly match the 500x500 box, or we can just save it
    # and map it to a larger extent, BUT the requirements say:
    # "strictly requirement to map a 500m x 500m bounding box to our local ENU coordinate system (extent=[-250, 250, -250, 250])"
    # To ensure it aligns correctly, we should crop the image so its edges exactly match the 500x500m box, OR we just save the stitched image
    # and let the plot clip it or warp it?
    # Actually, the simplest approach for Matplotlib `imshow(..., extent=[-250,250,-250,250])` is to crop the stitched image
    # to exactly the pixels that correspond to the 500x500m bounding box.

    stitched_nw_lat, stitched_nw_lon = tile_to_latlon(min_x_tile, min_y_tile, zoom)
    stitched_se_lat, stitched_se_lon = tile_to_latlon(max_x_tile + 1, max_y_tile + 1, zoom)

    # Calculate ENU coordinates of the stitched image boundaries relative to the center
    # using the same formula
    def latlon_to_enu(lat, lon, clat, clon):
        dlat = lat - clat
        dlon = lon - clon
        dy = dlat * (math.pi / 180.0) * R_EARTH
        dx = dlon * (math.pi / 180.0) * R_EARTH * math.cos(math.radians(clat))
        return dx, dy

    img_nw_x, img_nw_y = latlon_to_enu(stitched_nw_lat, stitched_nw_lon, center_lat, center_lon)
    img_se_x, img_se_y = latlon_to_enu(stitched_se_lat, stitched_se_lon, center_lat, center_lon)

    # img_nw_x is left, img_se_x is right.
    # img_nw_y is top, img_se_y is bottom.

    # Calculate the pixel coordinates for the -250, 250 box
    # stitched_w corresponds to (img_se_x - img_nw_x)
    # stitched_h corresponds to (img_nw_y - img_se_y) # since y is top-down

    x_scale = stitched_w / (img_se_x - img_nw_x)
    y_scale = stitched_h / (img_nw_y - img_se_y)

    # Target bounding box in ENU: Left=-250, Right=250, Bottom=-250, Top=250
    crop_left = int(( -250 - img_nw_x ) * x_scale)
    crop_right = int(( 250 - img_nw_x ) * x_scale)

    crop_top = int(( img_nw_y - 250 ) * y_scale)
    crop_bottom = int(( img_nw_y - -250 ) * y_scale)

    # Ensure crop boundaries are within image
    crop_left = max(0, crop_left)
    crop_top = max(0, crop_top)
    crop_right = min(stitched_w, crop_right)
    crop_bottom = min(stitched_h, crop_bottom)

    final_img = stitched_img.crop((crop_left, crop_top, crop_right, crop_bottom))
    final_img.save(os.path.join(out_dir, "background.png"))

    # Calculate magnetic declination using geomag
    mag_declination = 0.0
    if geomag is not None:
        try:
            mag = geomag.declination(center_lat, center_lon)
            mag_declination = float(mag)
        except Exception as e:
            print(f"Failed to calculate magnetic declination: {e}")
    else:
        print("geomag library not found. Defaulting magnetic declination to 0.0")

    meta = {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "zoom_level": zoom,
        "magnetic_declination": mag_declination,
        "bounds": {
            "x_min": -250,
            "x_max": 250,
            "y_min": -250,
            "y_max": 250
        }
    }

    with open(os.path.join(out_dir, "map_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    if progress_callback:
        progress_callback(total_tiles, total_tiles, "Download complete")
