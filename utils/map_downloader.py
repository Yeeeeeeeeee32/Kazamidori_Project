"""
utils/map_downloader.py
──────────────────────────────────────────────────────────────────────────────
Standalone offline map downloader for the Kazamidori Ground Control Station.

Downloads OSM XYZ raster tiles for a given centre coordinate and search
radius, stitches them into a single background image, calculates the Earth
magnetic declination, and writes an atomic metadata JSON alongside the image
into ``assets/offline_map/``.

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
import shutil
import sys
import time
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
USER_AGENT = "KazamidoriGCS/2.0"
MAX_TILES = 400                # Safety guard — abort if BBox is too large
OUT_DIR = Path("assets/offline_map")

# ── Geodetic / Tile Maths ────────────────────────────────────────────────────


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
        result = geo_mag.calculate(glat=lat, glon=lon, alt=0, time=date.today())
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
        return Image.open(io.BytesIO(resp.content)).convert("RGBA")
    except Exception as exc:
        print(f"  [FAIL] tile ({x},{y}): {exc}")
        return None


# ── Main Pipeline ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kazamidori Offline Map Downloader - download, stitch, and "
                    "crop OSM tiles for a given coordinate and search radius.",
    )
    parser.add_argument("--lat", type=float, required=True, help="Centre latitude (decimal degrees)")
    parser.add_argument("--lon", type=float, required=True, help="Centre longitude (decimal degrees)")
    parser.add_argument(
        "--radius", type=float, default=1000.0,
        help="Search radius in metres (default: 1000 m). "
             "Defines a square BBox of +/-radius around the centre.",
    )
    args = parser.parse_args()

    center_lat: float = args.lat
    center_lon: float = args.lon
    radius: float = args.radius

    # ── Dependency checks ────────────────────────────────────────────────
    if requests is None:
        print("[ERROR] 'requests' library not installed.  pip install requests")
        sys.exit(1)
    if Image is None:
        print("[ERROR] 'Pillow' library not installed.  pip install Pillow")
        sys.exit(1)

    print("=" * 60)
    print("  Kazamidori Offline Map Downloader v2.0")
    print("=" * 60)
    print(f"  Centre:  {center_lat:.6f}, {center_lon:.6f}")
    print(f"  Radius:  {radius:.0f} m  (BBox: +/-{radius:.0f} m)")
    print(f"  Zoom:    {ZOOM}")
    print()

    # ── ENU extent (the authoritative metadata) ──────────────────────────
    extent_meters = [-radius, radius, -radius, radius]  # [xmin, xmax, ymin, ymax]

    # ── BBox corners in geodetic ─────────────────────────────────────────
    # NW = (−radius East, +radius North), SE = (+radius East, −radius North)
    nw_lat, nw_lon = enu_to_latlon(-radius,  radius, center_lat, center_lon)
    se_lat, se_lon = enu_to_latlon( radius, -radius, center_lat, center_lon)

    # ── Tile range ───────────────────────────────────────────────────────
    min_tx, min_ty = latlon_to_tile(nw_lat, nw_lon, ZOOM)
    max_tx, max_ty = latlon_to_tile(se_lat, se_lon, ZOOM)

    # Normalise (latlon_to_tile may swap axes near the equator / antimeridian)
    if min_tx > max_tx:
        min_tx, max_tx = max_tx, min_tx
    if min_ty > max_ty:
        min_ty, max_ty = max_ty, min_ty

    n_cols = max_tx - min_tx + 1
    n_rows = max_ty - min_ty + 1
    total_tiles = n_cols * n_rows
    print(f"  Tile range X: {min_tx} -> {max_tx}  ({n_cols} tiles)")
    print(f"  Tile range Y: {min_ty} -> {max_ty}  ({n_rows} tiles)")
    print(f"  Total tiles:  {total_tiles}")

    if total_tiles > MAX_TILES:
        print(f"[ERROR] BBox requires {total_tiles} tiles (limit {MAX_TILES}). "
              f"Use a smaller radius or lower zoom.")
        sys.exit(1)

    # ── Step 1 — Clear existing assets ───────────────────────────────────
    if OUT_DIR.exists():
        print(f"\n  Clearing {OUT_DIR}/ ...")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 2 — Download tiles sequentially ─────────────────────────────
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    tiles: dict[tuple[int, int], "Image.Image"] = {}
    downloaded = 0
    failed = 0

    print(f"\n  Downloading {total_tiles} tiles ...")
    for ty in range(min_ty, max_ty + 1):
        for tx in range(min_tx, max_tx + 1):
            img = _download_tile(tx, ty, ZOOM, session)
            downloaded += 1
            if img is not None:
                tiles[(tx, ty)] = img
                print(f"    [{downloaded}/{total_tiles}]  tile ({tx},{ty}) OK")
            else:
                failed += 1
                print(f"    [{downloaded}/{total_tiles}]  tile ({tx},{ty}) FAIL")

            # Rate-limit: 0.5 – 1.0 s random delay between requests
            if downloaded < total_tiles:
                delay = random.uniform(0.5, 1.0)
                time.sleep(delay)

    if not tiles:
        print("[ERROR] No tiles downloaded -- aborting.")
        sys.exit(1)

    if failed > 0:
        print(f"\n  WARNING: {failed}/{total_tiles} tiles failed -- proceeding with gaps.")

    # ── Step 3 — Stitch tiles ────────────────────────────────────────────
    print("\n  Stitching tiles ...")
    stitched_w = n_cols * TILE_SIZE
    stitched_h = n_rows * TILE_SIZE
    stitched = Image.new("RGBA", (stitched_w, stitched_h), (0, 0, 0, 0))

    for (tx, ty), tile_img in tiles.items():
        px = (tx - min_tx) * TILE_SIZE
        py = (ty - min_ty) * TILE_SIZE
        stitched.paste(tile_img, (px, py))

    # ── Step 4 — Crop to exact BBox ──────────────────────────────────────
    print("  Cropping to BBox ...")

    # Stitched image boundary in geodetic
    stitch_nw_lat, stitch_nw_lon = tile_to_latlon(min_tx,     min_ty,     ZOOM)
    stitch_se_lat, stitch_se_lon = tile_to_latlon(max_tx + 1, max_ty + 1, ZOOM)

    # … converted to ENU
    img_nw_x, img_nw_y = latlon_to_enu(stitch_nw_lat, stitch_nw_lon, center_lat, center_lon)
    img_se_x, img_se_y = latlon_to_enu(stitch_se_lat, stitch_se_lon, center_lat, center_lon)

    # Pixel scale
    x_scale = stitched_w / (img_se_x - img_nw_x)
    y_scale = stitched_h / (img_nw_y - img_se_y)  # Y is inverted (top → bottom)

    # Crop rectangle in pixel space
    crop_left   = int((-radius  - img_nw_x) * x_scale)
    crop_right  = int(( radius  - img_nw_x) * x_scale)
    crop_top    = int((img_nw_y -  radius ) * y_scale)
    crop_bottom = int((img_nw_y - (-radius)) * y_scale)

    # Clamp to image bounds
    crop_left   = max(0, crop_left)
    crop_top    = max(0, crop_top)
    crop_right  = min(stitched_w, crop_right)
    crop_bottom = min(stitched_h, crop_bottom)

    final_img = stitched.crop((crop_left, crop_top, crop_right, crop_bottom))

    # ── Step 5 — Magnetic declination ────────────────────────────────────
    print("  Calculating magnetic declination ...")
    mag_declination = compute_magnetic_declination(center_lat, center_lon)

    # ── Step 6 — Atomic write (tmp → rename) ─────────────────────────────
    print("  Writing assets (atomic transaction) ...")

    tmp_img_path  = OUT_DIR / "background.tmp.png"
    tmp_meta_path = OUT_DIR / "map_meta.tmp.json"
    final_img_path  = OUT_DIR / "background.png"
    final_meta_path = OUT_DIR / "map_meta.json"

    meta = {
        "center_lat":           center_lat,
        "center_lon":           center_lon,
        "magnetic_declination": mag_declination,
        "manual_offset":        0.0,
        "extent_meters":        extent_meters,
    }

    # Write temporary files
    final_img.save(str(tmp_img_path))
    with open(tmp_meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=4, ensure_ascii=False)

    # Atomic rename
    os.replace(str(tmp_img_path),  str(final_img_path))
    os.replace(str(tmp_meta_path), str(final_meta_path))

    # ── Done ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  [OK] Download complete")
    print(f"  Image:     {final_img_path}  ({final_img.size[0]}x{final_img.size[1]} px)")
    print(f"  Metadata:  {final_meta_path}")
    print(f"  Declination: {mag_declination:+.2f} deg")
    print(f"  Extent (ENU): {extent_meters}")
    print("=" * 60)


# ── Public API (importable by sim_controller / MapDownloadWorker) ────────────


def download_offline_map(
    center_lat: float,
    center_lon: float,
    radius: float = 1000.0,
) -> dict:
    """Download, stitch, and save an offline map for *center_lat/lon*.

    This is the importable counterpart of ``main()``.  It runs the same
    pipeline (tile download → stitch → crop → magnetic declination → atomic
    write) but raises exceptions instead of calling ``sys.exit()``, making it
    safe to call from a background QThread.

    Parameters
    ----------
    center_lat : float
        Centre latitude in decimal degrees.
    center_lon : float
        Centre longitude in decimal degrees.
    radius : float, optional
        Half-width of the square bounding box in metres (default 1 000 m).

    Returns
    -------
    dict
        The metadata dict that was written to ``assets/offline_map/map_meta.json``.

    Raises
    ------
    RuntimeError
        If required optional dependencies (``requests``, ``Pillow``) are missing
        or if no tiles could be downloaded.
    """
    if requests is None:
        raise RuntimeError("'requests' library not installed.  Run: pip install requests")
    if Image is None:
        raise RuntimeError("'Pillow' library not installed.  Run: pip install Pillow")

    print("=" * 60)
    print("  Kazamidori Offline Map Downloader v2.0")
    print("=" * 60)
    print(f"  Centre:  {center_lat:.6f}, {center_lon:.6f}")
    print(f"  Radius:  {radius:.0f} m  (BBox: +/-{radius:.0f} m)")
    print(f"  Zoom:    {ZOOM}")
    print()

    extent_meters = [-radius, radius, -radius, radius]

    nw_lat, nw_lon = enu_to_latlon(-radius,  radius, center_lat, center_lon)
    se_lat, se_lon = enu_to_latlon( radius, -radius, center_lat, center_lon)

    min_tx, min_ty = latlon_to_tile(nw_lat, nw_lon, ZOOM)
    max_tx, max_ty = latlon_to_tile(se_lat, se_lon, ZOOM)

    if min_tx > max_tx:
        min_tx, max_tx = max_tx, min_tx
    if min_ty > max_ty:
        min_ty, max_ty = max_ty, min_ty

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

    if OUT_DIR.exists():
        print(f"\n  Clearing {OUT_DIR}/ ...")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    tiles: dict[tuple[int, int], "Image.Image"] = {}
    downloaded = 0
    failed = 0

    print(f"\n  Downloading {total_tiles} tiles ...")
    for ty in range(min_ty, max_ty + 1):
        for tx in range(min_tx, max_tx + 1):
            img = _download_tile(tx, ty, ZOOM, session)
            downloaded += 1
            if img is not None:
                tiles[(tx, ty)] = img
                print(f"    [{downloaded}/{total_tiles}]  tile ({tx},{ty}) OK")
            else:
                failed += 1
                print(f"    [{downloaded}/{total_tiles}]  tile ({tx},{ty}) FAIL")

            if downloaded < total_tiles:
                time.sleep(random.uniform(0.5, 1.0))

    if not tiles:
        raise RuntimeError("No tiles downloaded -- aborting.")

    if failed > 0:
        print(f"\n  WARNING: {failed}/{total_tiles} tiles failed -- proceeding with gaps.")

    print("\n  Stitching tiles ...")
    stitched_w = n_cols * TILE_SIZE
    stitched_h = n_rows * TILE_SIZE
    stitched = Image.new("RGBA", (stitched_w, stitched_h), (0, 0, 0, 0))
    for (tx, ty), tile_img in tiles.items():
        px = (tx - min_tx) * TILE_SIZE
        py = (ty - min_ty) * TILE_SIZE
        stitched.paste(tile_img, (px, py))

    print("  Cropping to BBox ...")
    stitch_nw_lat, stitch_nw_lon = tile_to_latlon(min_tx,     min_ty,     ZOOM)
    stitch_se_lat, stitch_se_lon = tile_to_latlon(max_tx + 1, max_ty + 1, ZOOM)

    img_nw_x, img_nw_y = latlon_to_enu(stitch_nw_lat, stitch_nw_lon, center_lat, center_lon)
    img_se_x, img_se_y = latlon_to_enu(stitch_se_lat, stitch_se_lon, center_lat, center_lon)

    x_scale = stitched_w / (img_se_x - img_nw_x)
    y_scale = stitched_h / (img_nw_y - img_se_y)

    crop_left   = int((-radius  - img_nw_x) * x_scale)
    crop_right  = int(( radius  - img_nw_x) * x_scale)
    crop_top    = int((img_nw_y -  radius ) * y_scale)
    crop_bottom = int((img_nw_y - (-radius)) * y_scale)

    crop_left   = max(0, crop_left)
    crop_top    = max(0, crop_top)
    crop_right  = min(stitched_w, crop_right)
    crop_bottom = min(stitched_h, crop_bottom)

    final_img = stitched.crop((crop_left, crop_top, crop_right, crop_bottom))

    print("  Calculating magnetic declination ...")
    mag_declination = compute_magnetic_declination(center_lat, center_lon)

    print("  Writing assets (atomic transaction) ...")
    tmp_img_path    = OUT_DIR / "background.tmp.png"
    tmp_meta_path   = OUT_DIR / "map_meta.tmp.json"
    final_img_path  = OUT_DIR / "background.png"
    final_meta_path = OUT_DIR / "map_meta.json"

    meta = {
        "center_lat":           center_lat,
        "center_lon":           center_lon,
        "magnetic_declination": mag_declination,
        "manual_offset":        0.0,
        "extent_meters":        extent_meters,
    }

    final_img.save(str(tmp_img_path))
    with open(tmp_meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=4, ensure_ascii=False)

    os.replace(str(tmp_img_path),  str(final_img_path))
    os.replace(str(tmp_meta_path), str(final_meta_path))

    print()
    print("=" * 60)
    print("  [OK] Download complete")
    print(f"  Image:       {final_img_path}  ({final_img.size[0]}x{final_img.size[1]} px)")
    print(f"  Metadata:    {final_meta_path}")
    print(f"  Declination: {mag_declination:+.2f} deg")
    print(f"  Extent (ENU): {extent_meters}")
    print("=" * 60)

    return meta


if __name__ == "__main__":
    main()
