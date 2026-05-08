"""
utils/data_loader.py
File I/O and data-parsing helpers: JSON config (airframe/parachute) and
motor CSV loading.

All functions are pure (no tkinter, no class state).  The UI layer is
responsible for picking file paths via filedialog and showing error
messages; these helpers only raise ValueError/OSError on bad input.

Unit contract
-------------
All values are in SI throughout: metres (m), kilograms (kg), seconds (s),
square metres (m²).  JSON files store SI values; parsers return SI values;
no conversions are applied at any layer.
"""

from __future__ import annotations

import json
import os
from typing import Any


# ── Config schema ────────────────────────────────────────────────────────────

_AIRFRAME_DEFAULTS: dict[str, Any] = {
    "mass":           0.0872,   # kg
    "cg":             0.21,     # m from nose
    "length":         0.383,    # m
    "radius":         0.015,    # m (body radius)
    "nose_length":    0.08,     # m
    "fin_root":       0.04,     # m
    "fin_tip":        0.02,     # m
    "fin_span":       0.03,     # m
    "fin_pos":        0.35,     # m from nose
    "motor_pos":      0.38,     # m from nose
    "motor_dry_mass": 0.015,    # kg
    "backfire_delay": 0.0,      # s
}

_PARACHUTE_DEFAULTS: dict[str, Any] = {
    "cd":   1.5,     # dimensionless
    "area": 0.196,   # m²
    "lag":  0.5,     # s
}


# ── Airframe helpers ─────────────────────────────────────────────────────────

def parse_airframe(raw: dict) -> dict[str, Any]:
    """Return a validated airframe dict, filling missing keys with SI defaults.

    Raises:
        ValueError: if a value cannot be converted to float.
    """
    result: dict[str, Any] = {}
    for key, default in _AIRFRAME_DEFAULTS.items():
        raw_val = raw.get(key, default)
        try:
            result[key] = float(raw_val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"airframe['{key}'] = {raw_val!r} is not a number") from exc
    return result


def parse_parachute(raw: dict) -> dict[str, Any]:
    """Return a validated parachute dict, filling missing keys with SI defaults.

    Raises:
        ValueError: if a value cannot be converted to float.
    """
    result: dict[str, Any] = {}
    for key, default in _PARACHUTE_DEFAULTS.items():
        raw_val = raw.get(key, default)
        try:
            result[key] = float(raw_val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"parachute['{key}'] = {raw_val!r} is not a number") from exc
    return result


# ── JSON config I/O ──────────────────────────────────────────────────────────

def save_config(filepath: str, airframe: dict, parachute: dict) -> None:
    """Serialise airframe + parachute to a versioned JSON file (SI units).

    Args:
        filepath:  Destination path (including filename).
        airframe:  Dict produced by :func:`parse_airframe` or equivalent.
        parachute: Dict produced by :func:`parse_parachute` or equivalent.

    Raises:
        OSError: on file-system errors.
    """
    payload = {
        "version":   2,
        "airframe":  airframe,
        "parachute": parachute,
    }
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=4, ensure_ascii=False)


def load_config(filepath: str) -> tuple[dict | None, dict | None]:
    """Load and parse a JSON config file (SI units).

    Supports both the v2 envelope format (``{"version": 2, "airframe": …,
    "parachute": …}``) and legacy flat files where the top-level dict
    contains either airframe or parachute keys directly.

    Returns:
        (airframe_dict | None, parachute_dict | None)
        Each element is ``None`` if the corresponding section was absent.

    Raises:
        ValueError: if the file contains neither airframe nor parachute data.
        OSError / json.JSONDecodeError: on I/O or parse errors.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    af_raw = data.get("airframe")
    pa_raw = data.get("parachute")

    # Legacy flat format
    if af_raw is None and pa_raw is None:
        af_keys = {"mass", "fin_root", "cg", "length", "radius"}
        pa_keys = {"cd", "area", "lag"}
        has_af = bool(af_keys & data.keys())
        has_pa = bool(pa_keys & data.keys())
        if not has_af and not has_pa:
            raise ValueError(
                "JSON contains neither airframe nor parachute data.")
        af_raw = data if has_af else None
        pa_raw = data if has_pa else None

    airframe  = parse_airframe(af_raw)  if af_raw is not None else None
    parachute = parse_parachute(pa_raw) if pa_raw is not None else None
    return airframe, parachute


# ── Motor CSV loader ─────────────────────────────────────────────────────────

class MotorData:
    """Plain-data container for a loaded motor thrust curve."""

    __slots__ = (
        "name", "filepath",
        "thrust_points",   # list of [time_s, thrust_N]
        "burn_time",       # float, seconds
        "max_thrust",      # float, Newtons
        "avg_thrust",      # float, Newtons
        "total_impulse",   # float, N·s
    )

    def __init__(
        self,
        name: str,
        filepath: str,
        thrust_points: list[list[float]],
        burn_time: float,
        max_thrust: float,
        avg_thrust: float,
        total_impulse: float,
    ) -> None:
        self.name          = name
        self.filepath      = filepath
        self.thrust_points = thrust_points
        self.burn_time     = burn_time
        self.max_thrust    = max_thrust
        self.avg_thrust    = avg_thrust
        self.total_impulse = total_impulse

    def __repr__(self) -> str:  # pragma: no cover
        return (f"MotorData({self.name!r}, burn={self.burn_time:.3f}s, "
                f"avg={self.avg_thrust:.1f}N, max={self.max_thrust:.1f}N)")

    def to_dict(self) -> dict[str, Any]:
        """Return all attributes as a plain dict (JSON-serialisable)."""
        return {
            "name":          self.name,
            "filepath":      self.filepath,
            "thrust_points": self.thrust_points,
            "burn_time":     self.burn_time,
            "max_thrust":    self.max_thrust,
            "avg_thrust":    self.avg_thrust,
            "total_impulse": self.total_impulse,
        }


# ── Rocket JSON loader ────────────────────────────────────────────────────────

class RocketConfigError(ValueError):
    """Raised when a Rocket.json config file cannot be loaded or parsed."""


class AirframeConfigError(ValueError):
    """Raised when an airframe JSON config file cannot be loaded or parsed."""


_AIRFRAME_REQUIRED_KEYS: frozenset[str] = frozenset({
    "mass", "cg", "length", "radius", "nose_length",
    "fin_root", "fin_tip", "fin_span", "fin_pos",
    "motor_pos", "motor_dry_mass", "backfire_delay",
})

_PARACHUTE_REQUIRED_KEYS: frozenset[str] = frozenset({"cd", "area", "lag"})


def load_rocket_config(filepath: str) -> dict[str, dict[str, float]]:
    """Load a Rocket.json config file and return SI-unit sections.

    Expected JSON structure (version 2)::

        {
            "version": 2,
            "airframe": {
                "mass":           <kg>,
                "cg":             <m from nose>,
                "length":         <m>,
                "radius":         <m, body radius>,
                "nose_length":    <m>,
                "fin_root":       <m>,
                "fin_tip":        <m>,
                "fin_span":       <m>,
                "motor_pos":      <m from nose>,
                "motor_dry_mass": <kg>,
                "backfire_delay": <s>
            },
            "parachute": {
                "cd":   <dimensionless>,
                "area": <m²>,
                "lag":  <s>
            }
        }

    All values are returned exactly as stored — no unit conversion is applied.
    The JSON file is the authoritative source of truth and must be in SI.

    Args:
        filepath: Path to the Rocket.json file.

    Returns:
        ``{"airframe": <si_dict>, "parachute": <si_dict>}``

    Raises:
        RocketConfigError: on missing file, invalid JSON, missing sections
                           or keys, non-numeric values, or negative physical
                           quantities.
    """
    # ── Load ────────────────────────────────────────────────────────────────
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise RocketConfigError(
            f"Rocket config not found: {filepath!r}"
        ) from None
    except json.JSONDecodeError as exc:
        raise RocketConfigError(
            f"Invalid JSON in {filepath!r}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise RocketConfigError(
            f"Expected a JSON object at top level, got {type(data).__name__!r}"
        )

    # ── Require both top-level sections ─────────────────────────────────────
    for section in ("airframe", "parachute"):
        if section not in data or not isinstance(data[section], dict):
            raise RocketConfigError(
                f"Missing or invalid '{section}' section in {filepath!r}"
            )

    # ── Parse airframe ───────────────────────────────────────────────────────
    af_raw     = data["airframe"]
    missing_af = _AIRFRAME_REQUIRED_KEYS - af_raw.keys()
    if missing_af:
        raise RocketConfigError(
            f"airframe section in {filepath!r} is missing keys: {sorted(missing_af)}"
        )

    airframe: dict[str, float] = {}
    for key in _AIRFRAME_REQUIRED_KEYS:
        raw = af_raw[key]
        try:
            val = float(raw)
        except (TypeError, ValueError):
            raise RocketConfigError(
                f"airframe['{key}'] in {filepath!r} is not numeric: {raw!r}"
            ) from None
        if key != "backfire_delay" and val < 0.0:
            raise RocketConfigError(
                f"airframe['{key}'] in {filepath!r} must be non-negative, got {val}"
            )
        airframe[key] = val  # SI pass-through — no conversion

    # ── Parse parachute ──────────────────────────────────────────────────────
    pa_raw     = data["parachute"]
    missing_pa = _PARACHUTE_REQUIRED_KEYS - pa_raw.keys()
    if missing_pa:
        raise RocketConfigError(
            f"parachute section in {filepath!r} is missing keys: {sorted(missing_pa)}"
        )

    parachute: dict[str, float] = {}
    for key in _PARACHUTE_REQUIRED_KEYS:
        raw = pa_raw[key]
        try:
            val = float(raw)
        except (TypeError, ValueError):
            raise RocketConfigError(
                f"parachute['{key}'] in {filepath!r} is not numeric: {raw!r}"
            ) from None
        if val < 0.0:
            raise RocketConfigError(
                f"parachute['{key}'] in {filepath!r} must be non-negative, got {val}"
            )
        parachute[key] = val  # SI pass-through — no conversion

    return {"airframe": airframe, "parachute": parachute}


def load_motor_csv(filepath: str) -> MotorData:
    """Parse a RockSim-format motor CSV file and return a :class:`MotorData`.

    The parser is lenient: it skips non-numeric rows (header, comments)
    and extracts the motor name from a ``Motor: <name>`` field when present.

    Args:
        filepath: Path to the CSV file.

    Returns:
        :class:`MotorData` with the parsed thrust curve and derived metrics.

    Raises:
        ValueError: if no valid numeric data is found.
        OSError: on file-system errors.
    """
    motor_name = os.path.basename(filepath).replace(".csv", "")
    thrust_points: list[list[float]] = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().replace('"', "").split(",")
            if len(parts) >= 2:
                try:
                    t = float(parts[0])
                    T = float(parts[1])
                    thrust_points.append([t, T])
                except ValueError:
                    # Header row or metadata — check for motor name field
                    if parts[0].strip().lower() in ("motor:", "motor"):
                        motor_name = parts[1].strip()

    if not thrust_points:
        raise ValueError(
            "No valid numeric data found in the file. "
            "Please verify it is a RockSim-format CSV.")

    # Ensure t=0 is present (extrapolate from first point if needed)
    if thrust_points[0][0] != 0.0:
        thrust_points.insert(0, [0.0, thrust_points[0][1]])

    burn_time    = thrust_points[-1][0]
    max_thrust   = max(p[1] for p in thrust_points)
    total_impulse = sum(
        (thrust_points[i - 1][1] + thrust_points[i][1]) * 0.5
        * (thrust_points[i][0] - thrust_points[i - 1][0])
        for i in range(1, len(thrust_points))
    )
    avg_thrust = (total_impulse / burn_time) if burn_time > 0 else 0.0

    return MotorData(
        name          = motor_name,
        filepath      = filepath,
        thrust_points = thrust_points,
        burn_time     = burn_time,
        max_thrust    = max_thrust,
        avg_thrust    = avg_thrust,
        total_impulse = total_impulse,
    )
