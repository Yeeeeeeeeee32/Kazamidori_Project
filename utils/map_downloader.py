"""
utils/map_downloader.py
──────────────────────────────────────────────────────────────────────────────
Standalone offline map downloader for the Kazamidori Ground Control Station.

Downloads OSM XYZ raster tiles for a given centre coordinate and search
radius, stitches them into a single background image, calculates the Earth
magnetic declination, and writes an atomic metadata JSON alongside the image
into ``assets/offline_map/``.

Projection Guarantee
────────────────────
The crop pipeline operates entirely in **Web Mercator (EPSG:3857) global
pixel coordinates**.  The centre lat/lon is converted to a float64 global
pixel position, a symmetric pixel-radius is derived from the exact
metres-per-pixel resolution at the target latitude, and the image is
cropped from integer-symmetric bounds around that centre pixel.  This
guarantees the visual centre of the output image aligns perfectly with the
ENU Cartesian origin (0, 0).

Usage
─────
    python utils/map_downloader.py --lat 35.123 --lon 136.789
    python utils/map_downloader.py --lat 35.123 --lon 136.789 --radius 500

Requirements
────────────
    pip install requests Pillow pygeomag
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import tempfile
import time
import uuid
from datetime import date
from pathlib import Path

# ── Optional dependencies ────────────────────────────────────────────────────

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

try:
    from pygeomag import GeoMag
except ImportError:
    GeoMag = None  # type: ignore[assignment]

# ── Constants ────────────────────────────────────────────────────────────────

R_EARTH = 6_378_137.0          # WGS-84 semi-major axis (m)
TILE_SIZE = 256                # Standard OSM tile size in pixels
ZOOM = 16                      # Fixed zoom level
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "KazamidoriGCS/3.0"
MAX_TILES = 400                # Safety guard — abort if BBox is too large
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "assets" / "offline_map"
MAX_MERCATOR_LAT = 85.05112878


def validate_map_coordinates(lat, lon) -> tuple[float, float]:
    """Validate the coordinates supported by the raster tile projection."""
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter valid latitude and longitude first.") from exc
    if not (math.isfinite(lat) and abs(lat) < MAX_MERCATOR_LAT
            and math.isfinite(lon) and -180 <= lon <= 180):
        raise ValueError("Map latitude must be between -85.0511 and 85.0511; longitude between -180 and 180.")
    return lat, lon


def read_offline_map() -> tuple[dict, "Image.Image"]:
    """Read a complete saved map, including the original background.png format."""
    with (OUT_DIR / "map_meta.json").open(encoding="utf-8-sig") as fp:
        meta = json.load(fp)
    if not isinstance(meta, dict):
        raise ValueError("Invalid map metadata.")
    meta["center_lat"], meta["center_lon"] = validate_map_coordinates(meta.get("center_lat"), meta.get("center_lon"))
    extent = meta.get("extent_meters")
    if (not isinstance(extent, list) or len(extent) != 4
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in extent)
            or extent[0] >= extent[1] or extent[2] >= extent[3]):
        raise ValueError("Invalid map extent.")
    filename = meta.get("image_file", "background.png")
    if (not isinstance(filename, str) or not filename
            or Path(filename).name != filename or "/" in filename or "\\" in filename):
        raise ValueError("Invalid map image filename.")
    with Image.open(OUT_DIR / filename) as source:
        image = source.convert("RGBA")
    if "image_size_px" in meta and list(image.size) != meta["image_size_px"]:
        raise ValueError("Map image dimensions do not match the metadata.")
    if image.getextrema()[3][0] != 255:
        raise ValueError("Saved map contains missing tiles. Download it again.")
    return meta, image


def map_extent_at_launch(meta: dict, launch_lat: float, launch_lon: float) -> list[float]:
    """Place a saved raster in the launch site's local East/North frame."""
    from utils.geo_math import latlon_to_offset

    validate_map_coordinates(launch_lat, launch_lon)
    xmin, xmax, ymin, ymax = meta["extent_meters"]
    south, west = enu_to_latlon(xmin, ymin, meta["center_lat"], meta["center_lon"])
    north, east = enu_to_latlon(xmax, ymax, meta["center_lat"], meta["center_lon"])
    left, bottom = latlon_to_offset(launch_lat, launch_lon, south, west)
    right, top = latlon_to_offset(launch_lat, launch_lon, north, east)
    return [left, right, bottom, top]

# ── Web Mercator Exact Resolution ────────────────────────────────────────────
#
# The equatorial circumference of the EPSG:3857 sphere divided by the number
# of pixels at zoom 0 (one 256×256 tile covers the whole world):
#   C_eq / 256  =  2π × 6 378 137 / 256  ≈  156 543.033 928 041
_EPSG3857_RESOLUTION_Z0 = 156543.03392804097


def _resolution_at_lat(lat_deg: float, zoom: int) -> float:
    """Return the exact ground resolution (metres / pixel) at *lat_deg* and *zoom*.

    Formula: ``C_eq / 256 × cos(φ) / 2^zoom``  (standard Web Mercator).
    """
    return _EPSG3857_RESOLUTION_Z0 * math.cos(math.radians(lat_deg)) / (2.0 ** zoom)


def _latlon_to_global_pixel(
    lat_deg: float, lon_deg: float, zoom: int,
) -> tuple[float, float]:
    """Convert (lat, lon) to **float64** global pixel coordinates at *zoom*.

    The global pixel space covers ``[0, 2^zoom × 256)`` on both axes.
    X increases eastward; Y increases southward (standard Web Mercator).
    """
    n = 2.0 ** zoom
    px_x = (lon_deg + 180.0) / 360.0 * n * TILE_SIZE
    lat_rad = math.radians(lat_deg)
    px_y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    return px_x, px_y


# ── Geodetic / Tile Maths (retained for backward compatibility) ──────────────


def latlon_to_tile(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    """Convert (lat, lon) → tile index (x, y) at *zoom*."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_to_latlon(xtile: int, ytile: int, zoom: int) -> tuple[float, float]:
    """Return the NW-corner (lat, lon) of the tile."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def enu_to_latlon(
    dx_east: float, dy_north: float, clat: float, clon: float,
) -> tuple[float, float]:
    """Convert an ENU offset (metres) to geodetic coordinates.

    Applies the cos(lat) correction on the longitude axis so that the
    resulting BBox is geometrically accurate in the local East-North plane.
    """
    dlat = (dy_north / R_EARTH) * (180.0 / math.pi)
    dlon = (dx_east / (R_EARTH * math.cos(math.radians(clat)))) * (180.0 / math.pi)
    return clat + dlat, clon + dlon


def latlon_to_enu(
    lat: float, lon: float, clat: float, clon: float,
) -> tuple[float, float]:
    """Convert geodetic coordinates to ENU offset (metres) from a centre."""
    dy = (lat - clat) * (math.pi / 180.0) * R_EARTH
    dx = (lon - clon) * (math.pi / 180.0) * R_EARTH * math.cos(math.radians(clat))
    return dx, dy


# ── Magnetic Declination ─────────────────────────────────────────────────────


def compute_magnetic_declination(lat: float, lon: float) -> float:
    """Return the theoretical magnetic declination (degrees) for today's date.

    Uses pygeomag (World Magnetic Model).  Returns 0.0 on failure / missing lib.
    """
    if GeoMag is None:
        print("[WARN] pygeomag not installed -- defaulting declination to 0.0 deg")
        return 0.0
    try:
        geo_mag = GeoMag()
        today = date.today()
        year_start = date(today.year, 1, 1)
        next_year = date(today.year + 1, 1, 1)
        decimal_year = today.year + (today - year_start).days / (next_year - year_start).days
        result = geo_mag.calculate(glat=lat, glon=lon, alt=0, time=decimal_year)
        dec = float(result.d)
        print(f"  Magnetic declination (WMM): {dec:+.2f} deg")
        return dec
    except Exception as exc:
        print(f"[WARN] Magnetic declination calculation failed: {exc}")
        return 0.0


# ── Tile Download ────────────────────────────────────────────────────────────


def _download_tile(
    x: int, y: int, zoom: int, session: "requests.Session",
) -> "Image.Image | None":
    """Download a single OSM tile and return it as a PIL Image (or None)."""
    url = TILE_URL.format(z=zoom, x=x, y=y)
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        with Image.open(io.BytesIO(resp.content)) as source:
            image = source.convert("RGBA")
        if image.size != (TILE_SIZE, TILE_SIZE) or image.getextrema()[3][0] != 255:
            raise ValueError("Incomplete map tile")
        return image
    except Exception as exc:
        print(f"  [FAIL] tile ({x},{y}): {exc}")
        return None


# ── Public API (importable by sim_controller / MapDownloadWorker) ────────────


def download_offline_map(
    center_lat: float,
    center_lon: float,
    radius: float = 1000.0,
) -> dict:
    """Download, stitch, and save an offline map for *center_lat/lon*.

    Pipeline
    --------
    1. Compute exact metres-per-pixel at the target latitude & zoom.
    2. Convert centre (lat, lon) to float64 global pixel coordinates.
    3. Derive a symmetric pixel-radius from the requested search radius.
    4. Build a pixel bounding box that is **perfectly bilateral-symmetric**
       around the centre pixel.
    5. Download the required tiles, stitch, and crop.
    6. Hardcode ``extent_meters`` to ``[-radius, +radius, -radius, +radius]``
       so the visual centre aligns exactly with ENU origin (0, 0).
    7. Compute magnetic declination and write assets atomically.

    Parameters
    ----------
    center_lat : float
        Centre latitude in decimal degrees (IEEE 754 float64).
    center_lon : float
        Centre longitude in decimal degrees (IEEE 754 float64).
    radius : float, optional
        Half-width of the square bounding box in metres (default 1 000 m).

    Returns
    -------
    dict
        The metadata dict written to ``assets/offline_map/map_meta.json``.

    Raises
    ------
    RuntimeError
        If required dependencies are missing or no tiles could be downloaded.
    """
    print(f"[DIAG] [COORD TRACE] download_offline_map ENTERED with lat={center_lat} (type: {type(center_lat)}), lon={center_lon} (type: {type(center_lon)})")
    # ── Enforce float64 precision ─────────────────────────────────────────
    center_lat, center_lon = validate_map_coordinates(center_lat, center_lon)
    radius     = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Map radius must be a positive finite number.")

    if requests is None:
        raise RuntimeError("'requests' library not installed.  Run: pip install requests")
    if Image is None:
        raise RuntimeError("'Pillow' library not installed.  Run: pip install Pillow")

    print("=" * 60)
    print("  Kazamidori Offline Map Downloader v3.0")
    print("=" * 60)
    print(f"  Centre:  {center_lat:.8f}, {center_lon:.8f}")
    print(f"  Radius:  {radius:.0f} m  (BBox: +/-{radius:.0f} m)")
    print(f"  Zoom:    {ZOOM}")
    print()

    # ── 1. Exact resolution & global pixel centre ────────────────────────
    resolution = _resolution_at_lat(center_lat, ZOOM)
    center_px_x, center_px_y = _latlon_to_global_pixel(center_lat, center_lon, ZOOM)
    radius_px = radius / resolution

    print(f"  Resolution:   {resolution:.6f} m/px")
    print(f"  Centre pixel: ({center_px_x:.4f}, {center_px_y:.4f})")
    print(f"  Radius (px):  {radius_px:.2f}")
    print()

    # ── 2. Symmetric pixel bounding box ──────────────────────────────────
    bbox_left   = center_px_x - radius_px
    bbox_right  = center_px_x + radius_px
    bbox_top    = center_px_y - radius_px
    bbox_bottom = center_px_y + radius_px

    # ── 3. Tile range from pixel BBox ────────────────────────────────────
    #   min_t = floor(bbox_min / 256)     → first tile overlapping the bbox
    #   max_t = ceil(bbox_max / 256) - 1  → last tile overlapping the bbox
    min_tx = int(math.floor(bbox_left   / TILE_SIZE))
    max_tx = int(math.ceil(bbox_right   / TILE_SIZE)) - 1
    min_ty = int(math.floor(bbox_top    / TILE_SIZE))
    max_ty = int(math.ceil(bbox_bottom  / TILE_SIZE)) - 1

    # Guard: when bbox_max lands exactly on a tile boundary, ceil(x/256)-1
    # may undershoot.  A floor-based fallback guarantees correctness.
    max_tx = max(max_tx, int(math.floor((bbox_right  - 1e-9) / TILE_SIZE)))
    max_ty = max(max_ty, int(math.floor((bbox_bottom - 1e-9) / TILE_SIZE)))

    n_cols = max_tx - min_tx + 1
    n_rows = max_ty - min_ty + 1
    total_tiles = n_cols * n_rows
    print(f"  Tile range X: {min_tx} -> {max_tx}  ({n_cols} tiles)")
    print(f"  Tile range Y: {min_ty} -> {max_ty}  ({n_rows} tiles)")
    print(f"  Total tiles:  {total_tiles}")

    if total_tiles > MAX_TILES:
        raise RuntimeError(
            f"BBox requires {total_tiles} tiles (limit {MAX_TILES}). "
            "Use a smaller radius or lower zoom."
        )

    if min_tx < 0 or min_ty < 0 or max_tx >= 2 ** ZOOM or max_ty >= 2 ** ZOOM:
        raise ValueError("Requested map crosses the Web Mercator map boundary.")

    # Keep the current offline map usable until the replacement is complete.

    # ── 5. Download tiles sequentially ───────────────────────────────────
    tiles: dict[tuple[int, int], "Image.Image"] = {}
    downloaded = 0

    print(f"\n  Downloading {total_tiles} tiles ...")
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                img = _download_tile(tx, ty, ZOOM, session)
                downloaded += 1
                if img is None:
                    # Offline connections must not wait for every tile to time
                    # out. A partial image cannot replace the saved map either.
                    raise RuntimeError("Map tile download failed. Existing map was preserved.")
                tiles[(tx, ty)] = img
                print(f"    [{downloaded}/{total_tiles}]  tile ({tx},{ty}) OK")

                # Rate-limit: 0.5 – 1.0 s random delay between requests
                if downloaded < total_tiles:
                    time.sleep(random.uniform(0.5, 1.0))

    # ── 6. Stitch tiles ──────────────────────────────────────────────────
    print("\n  Stitching tiles ...")
    stitched_w = n_cols * TILE_SIZE
    stitched_h = n_rows * TILE_SIZE
    stitched = Image.new("RGBA", (stitched_w, stitched_h), (0, 0, 0, 0))

    for (tx, ty), tile_img in tiles.items():
        px = (tx - min_tx) * TILE_SIZE
        py = (ty - min_ty) * TILE_SIZE
        stitched.paste(tile_img, (px, py))

    # ── 7. Symmetric crop via global pixel coordinates ───────────────────
    #
    # The stitched image's top-left corner corresponds to global pixel
    # (min_tx × 256, min_ty × 256).  We translate the centre and crop
    # bounds into stitched-local pixel space, then build an integer crop
    # box that is bilaterally symmetric around the centre.
    print("  Cropping to symmetric BBox ...")

    stitch_origin_x = float(min_tx * TILE_SIZE)
    stitch_origin_y = float(min_ty * TILE_SIZE)

    center_in_stitch_x = center_px_x - stitch_origin_x
    center_in_stitch_y = center_px_y - stitch_origin_y

    # Integer half-width: round(radius_px) so the crop is as close to the
    # requested radius as possible (sub-pixel error < 0.5 px ≈ 1.2 m at z16).
    # Using a single half_w for both axes enforces a square, matching the
    # square [-radius, +radius] extent metadata.
    half_w = max(1, int(round(radius_px)))
    crop_size = 2 * half_w                          # guaranteed even

    crop_left_i   = int(round(center_in_stitch_x)) - half_w
    crop_top_i    = int(round(center_in_stitch_y)) - half_w
    crop_right_i  = crop_left_i + crop_size
    crop_bottom_i = crop_top_i  + crop_size

    # Clamp to stitched image bounds (should never trigger with correct tiles)
    crop_left_i   = max(0, crop_left_i)
    crop_top_i    = max(0, crop_top_i)
    crop_right_i  = min(stitched_w, crop_right_i)
    crop_bottom_i = min(stitched_h, crop_bottom_i)

    final_img = stitched.crop((crop_left_i, crop_top_i, crop_right_i, crop_bottom_i))

    print(f"  Crop box (stitch-local): ({crop_left_i}, {crop_top_i}, "
          f"{crop_right_i}, {crop_bottom_i})")
    print(f"  Final image size: {final_img.size[0]}x{final_img.size[1]} px")

    # ── 8. Magnetic declination ──────────────────────────────────────────
    print("  Calculating magnetic declination ...")
    mag_declination = compute_magnetic_declination(center_lat, center_lon)

    # ── 9. Strict ENU extent ─────────────────────────────────────────────
    # The image is bilaterally symmetric around the exact centre pixel of
    # (center_lat, center_lon), so mapping to [-radius, +radius] on both
    # axes is mathematically correct.  The sub-pixel rounding (<0.5 px)
    # is absorbed into the extent → pixel mapping as a <0.05 % stretch.
    extent_meters = [-radius, radius, -radius, radius]  # [xmin, xmax, ymin, ymax]

    # Publish a new image, then atomically switch the metadata pointer so that
    # readers never combine a replacement image with the previous bounds.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_name = f"background-{uuid.uuid4().hex}.png"
    final_img_path = OUT_DIR / image_name
    final_meta_path = OUT_DIR / "map_meta.json"

    meta = {
        "center_lat":           center_lat,
        "center_lon":           center_lon,
        "magnetic_declination": mag_declination,
        "manual_offset":        0.0,
        "declination_model":    "WMM",
        "image_file":           image_name,
        "extent_meters":        extent_meters,
        # ── Diagnostic fields (transparency / debuggability) ─────────────
        "zoom":                 ZOOM,
        "resolution_m_per_px":  resolution,
        "image_size_px":        [final_img.size[0], final_img.size[1]],
        "search_radius_m":      radius,
    }

    with tempfile.TemporaryDirectory(prefix=".download-", dir=OUT_DIR) as staging:
        tmp_img_path = Path(staging) / image_name
        tmp_meta_path = Path(staging) / "map_meta.json"
        final_img.save(tmp_img_path)
        tmp_meta_path.write_text(json.dumps(meta, indent=4, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_img_path, final_img_path)
        try:
            os.replace(tmp_meta_path, final_meta_path)
        except OSError:
            final_img_path.unlink(missing_ok=True)
            raise

    for old_image in OUT_DIR.glob("background-*.png"):
        token = old_image.stem.removeprefix("background-")
        if old_image != final_img_path and len(token) == 32 and all(c in "0123456789abcdef" for c in token):
            try:
                old_image.unlink()
            except OSError:
                pass  # A reader may still have the previous image open.

    # ── Done ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  [OK] Download complete")
    print(f"  Image:       {final_img_path}  ({final_img.size[0]}x{final_img.size[1]} px)")
    print(f"  Metadata:    {final_meta_path}")
    print(f"  Declination: {mag_declination:+.2f} deg")
    print(f"  Extent (ENU): {extent_meters}")
    print(f"  Resolution:  {resolution:.6f} m/px")
    print("=" * 60)

    return meta


# ── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kazamidori Offline Map Downloader - download, stitch, and "
                    "crop OSM tiles for a given coordinate and search radius.",
    )
    parser.add_argument(
        "--lat", type=float, required=True,
        help="Centre latitude (decimal degrees)",
    )
    parser.add_argument(
        "--lon", type=float, required=True,
        help="Centre longitude (decimal degrees)",
    )
    parser.add_argument(
        "--radius", type=float, default=1000.0,
        help="Search radius in metres (default: 1000 m). "
             "Defines a square BBox of +/-radius around the centre.",
    )
    args = parser.parse_args()

    try:
        download_offline_map(
            center_lat=float(args.lat),
            center_lon=float(args.lon),
            radius=float(args.radius),
        )
    except (RuntimeError, ValueError) as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
