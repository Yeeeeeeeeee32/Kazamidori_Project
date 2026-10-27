"""Versioned result archives, statistical migrations and CSVs; no Qt dependency."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class ResultFileError(ValueError):
    """The selected file is not a supported, drawable result archive."""


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        # Workers use -inf for unavailable scores. JSON represents these as
        # null; invalid trajectory/grid values will still fail validation.
        return None
    return value


def atomic_write(path, data: bytes) -> None:
    """Replace a destination only after the complete export has been written."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ResultFileError(f"{name}: a finite number is required.")
    return value


def _array(value, name):
    if not isinstance(value, list):
        raise ResultFileError(f"{name}: an array is required.")
    for item in value:
        _number(item, name)
    return value


def _parallel_arrays(data, keys):
    arrays = [_array(data.get(key, []), key) for key in keys]
    if len({len(a) for a in arrays}) != 1:
        raise ResultFileError(f"{', '.join(keys)}: array lengths must match.")
    return arrays


def _validate_payload(data):
    if not isinstance(data, dict):
        raise ResultFileError("Result data must be an object.")
    snapshot = data.get('input_snapshot')
    if snapshot is not None:
        if not isinstance(snapshot,dict):
            raise ResultFileError('input_snapshot must be an object')
        if 'flight_mode' in snapshot and not isinstance(snapshot['flight_mode'],str):
            raise ResultFileError('input_snapshot.flight_mode must be a string')
    for keys in (("trajectory_x", "trajectory_y", "trajectory_z"),
                 ("x_vals", "y_vals", "z_vals"),
                 ("mc_scatter_x", "mc_scatter_y"),
                 ("mc_scatter_x_filtered", "mc_scatter_y_filtered")):
        _parallel_arrays(data, keys)
    for time_key, x_key in (("t_hist", "trajectory_x"), ("t_vals", "x_vals")):
        if time_key in data:
            times = _array(data[time_key], time_key)
            if len(times) != len(data.get(x_key, [])) or any(a > b for a, b in zip(times, times[1:])):
                raise ResultFileError(f"{time_key}: times must match the trajectory and be ordered.")
    for key in ("scatter_points", "scatter"):
        points = data.get(key, [])
        if not isinstance(points, list):
            raise ResultFileError(f"{key}: an array is required.")
        for point in points:
            if len(_array(point, key)) != 2:
                raise ResultFileError(f"{key}: each impact must have East and North coordinates.")
    metrics = ("impact_x", "impact_y", "land_x", "land_y", "apogee_m", "hang_time",
               "horizontal_distance_m", "elev", "azi", "score", "r_N_radius", "cep", "n_runs")
    for key, value in data.items():
        if key in metrics or key.startswith("mc_avg_") or key.startswith("mc_min_") or key.startswith("mc_max_"):
            if key in ("impact_x", "impact_y", "land_x", "land_y", "apogee_m", "hang_time", "r_N_radius", "cep"):
                _number(value, key)
            elif value is not None and value != "N/A":
                _number(value, key)
        if key == "landing_prob" and not 1 <= _number(value, key) <= 99:
            raise ResultFileError("landing_prob must be between 1 and 99.")
    for key in ("ellipse", "cep_ellipse"):
        ellipse = data.get(key)
        if ellipse is not None:
            if not isinstance(ellipse, dict):
                raise ResultFileError(f"{key}: an object is required.")
            for field in ("cx", "cy", "angle_rad", "a", "b"):
                _number(ellipse.get(field), f"{key}.{field}")
            if ellipse["a"] < 0 or ellipse["b"] < 0:
                raise ResultFileError("Ellipse axes must be nonnegative.")
    for key in ("phases", "events"):
        if data.get(key) is not None and not isinstance(data[key], dict):
            raise ResultFileError(f"{key}: an object is required.")
    for phase in (data.get("phases") or {}).values():
        if not isinstance(phase, dict):
            raise ResultFileError("Each phase must be an object.")
        _parallel_arrays(phase, ("x", "y", "z"))
    for event in (data.get("events") or {}).values():
        if event is not None and len(_array(event, "event")) < 3:
            raise ResultFileError("Each event must have East, North and altitude coordinates.")
    nodes = data.get("wind_nodes", [])
    if not isinstance(nodes, list):
        raise ResultFileError("wind_nodes: an array is required.")
    for node in nodes:
        if not isinstance(node, dict):
            raise ResultFileError("Each wind node must be an object.")
        for key in ("alt_m", "speed_ms", "dir_deg"):
            _number(node.get(key), key)
    kde = data.get("kde")
    if kde is not None:
        if not isinstance(kde, dict):
            raise ResultFileError("kde: an object is required.")
        shapes = []
        for key in ("X_m", "Y_m", "Z"):
            grid = kde.get(key)
            if not isinstance(grid, list) or len(grid) < 2:
                raise ResultFileError(f"kde.{key}: a 2D grid is required.")
            lengths = {len(_array(row, key)) for row in grid}
            if len(lengths) != 1 or min(lengths) < 2:
                raise ResultFileError("KDE grid rows must have equal lengths (at least 2).")
            shapes.append((len(grid), lengths.pop()))
        if len(set(shapes)) != 1:
            raise ResultFileError("KDE grid dimensions must match.")
        if any(v < 0 for row in kde["Z"] for v in row) or not any(v > 0 for row in kde["Z"] for v in row):
            raise ResultFileError("KDE density must be nonnegative and nonzero.")
        for key in ("x_min_m", "x_max_m", "y_min_m", "y_max_m"):
            _number(kde.get(key), key)


def validate_archive(archive):
    if not isinstance(archive, dict) or archive.get("format") != "kazamidori-results":
        raise ResultFileError("Select a Kazamidori result archive exported from File > Export Results.")
    if archive.get("version") != 1 or archive.get("coordinates") != "ENU_m":
        raise ResultFileError("Unsupported result format version or coordinate system.")
    context = archive.get("context")
    if not isinstance(context, dict) or context.get("flight_mode") not in ("定点滞空", "高度", "有翼", "自由"):
        raise ResultFileError("Missing or invalid flight mode.")
    lat, lon = context.get("launch_lat"), context.get("launch_lon")
    if (lat is None) != (lon is None):
        raise ResultFileError("Both launch coordinates are required.")
    if lat is not None and not (-90 <= _number(lat, "latitude") <= 90 and -180 <= _number(lon, "longitude") <= 180):
        raise ResultFileError("Launch coordinates are outside their valid range.")
    radius = context.get("target_radius")
    for key in ('target_x','target_y'):
        if key in context:
            _number(context[key],key)
    if radius is not None and _number(radius, "maximum radius") < 0:
        raise ResultFileError("Maximum radius must be nonnegative.")
    _validate_payload(archive.get("result"))
    _validate_payload(archive.get("nominal", {}))
    result = archive["result"]
    if not result.get("trajectory_x"):
        raise ResultFileError("The archive has no trajectory to restore.")
    times = result.get("t_hist", result.get("t_vals"))
    if times is not None and len(times) != len(result["trajectory_x"]):
        raise ResultFileError("Time samples must match the displayed trajectory.")
    return archive


def make_archive(result, nominal, context):
    archive = {"format": "kazamidori-results", "version": 1, "coordinates": "ENU_m",
               "exported_at": datetime.now(timezone.utc).isoformat(),
               "context": context, "nominal": nominal or {}, "result": result}
    # Round-trip to detach from mutable state and convert numpy values.
    try:
        archive = json.loads(json.dumps(_json_safe(archive), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ResultFileError(str(exc)) from exc
    return validate_archive(archive)


def save_archive(path, archive):
    validate_archive(archive)
    atomic_write(path, json.dumps(archive, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8"))


def load_archive(path):
    def reject_constant(value):
        raise ResultFileError(f"Invalid numeric constant: {value}")
    try:
        with open(path, encoding="utf-8-sig") as stream:
            archive = json.load(stream, parse_constant=reject_constant)
    except (ValueError, UnicodeError) as exc:
        raise ResultFileError(f"Cannot read result JSON: {exc}") from exc
    validate_archive(archive)
    # Older archives stored a north-clockwise angle in a field consumed by
    # Matplotlib as east-counterclockwise. Refit from the original samples;
    # rotating alone would retain the old artificial minimum minor axis.
    from core.monte_carlo import compute_error_ellipse
    result = archive['result']
    points = result.get('scatter_points') or result.get('scatter') or list(zip(
        result.get('mc_scatter_x', []), result.get('mc_scatter_y', [])))
    for key in ('ellipse', 'cep_ellipse'):
        ellipse = result.get(key)
        if ellipse and ellipse.get('angle_convention') != 'east_ccw':
            result[key] = compute_error_ellipse(points, result.get('landing_prob',90))
            archive.setdefault('migration_notes', []).append(
                f'{key}: recalculated from ENU samples with east_ccw angle convention'
                if result[key] else f'{key}: hidden because fewer than four samples are available')
    return archive


def save_csv(path, archive):
    """One row per trajectory sample or MC impact; distances in local ENU metres."""
    validate_archive(archive)
    result = archive["result"]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("record_type", "index", "time_s", "east_m", "north_m", "altitude_m"))
    times = result.get("t_hist", result.get("t_vals", []))
    for i, (x, y, z) in enumerate(zip(result["trajectory_x"], result["trajectory_y"], result["trajectory_z"])):
        writer.writerow(("trajectory", i, times[i] if times else "", x, y, z))
    for i, (x, y) in enumerate(zip(result.get("mc_scatter_x", []), result.get("mc_scatter_y", []))):
        writer.writerow(("mc_impact", i, "", x, y, 0))
    atomic_write(path, stream.getvalue().encode("utf-8-sig"))
