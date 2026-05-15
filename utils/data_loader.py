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

import io
import json
import math as _math
import os
import xml.etree.ElementTree as _ET
import zipfile as _zipfile
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


# ── RockSim .rkt XML parser ───────────────────────────────────────────────────
#
# RockSim stores all linear dimensions in MILLIMETRES and masses in GRAMS.
# Every value is converted to SI (m, kg) before being stored or passed to the
# geometry_math MoI helpers.
#
# File layout (ZIP-wrapped OR raw XML):
#   <RockSimDocument>
#     <DesignInformation>
#       <RocketDesign>
#         <Stage3Parts>           ← top-level sequential components
#           <NoseCone/>
#           <BodyTube>
#             <AttachedParts>    ← fins, parachute, rings, mass objects, motor
#               <TrapezoidalFinSet/>
#               <Parachute/>
#               <Ring/>
#               <MassObject/>
#               <Motor/>
#             </AttachedParts>
#           </BodyTube>
#         </Stage3Parts>
#       </RocketDesign>
#     </DesignInformation>
#   </RockSimDocument>

def _rkt_float(elem: _ET.Element, tag: str, default: float = 0.0) -> float:
    """Return float value of a child element, or *default* if absent/blank."""
    child = elem.find(tag)
    if child is None or not (child.text or "").strip():
        return default
    try:
        return float(child.text.strip())
    except ValueError:
        return default


def _rkt_mass_kg(elem: _ET.Element) -> float:
    """Return component mass in kg.  Prefers <KnownMass> over <CalcMass> (g)."""
    known = _rkt_float(elem, "KnownMass", 0.0)
    if known > 0.0:
        return known / 1000.0
    return _rkt_float(elem, "CalcMass", 0.0) / 1000.0


def _load_rkt_bytes(filepath: str) -> bytes:
    """Return the XML payload from a .rkt file (ZIP-wrapped or raw XML)."""
    try:
        with _zipfile.ZipFile(filepath) as zf:
            # Security: filter out entries with path traversal (Zip Slip)
            names = [n for n in zf.namelist()
                     if not (n.startswith("/") or ".." in n.replace("\\", "/").split("/"))]

            if not names:
                raise RocketConfigError(f"No safe entries found in ZIP: {filepath}")

            xml_entries = [n for n in names
                           if n.lower().endswith((".ork", ".xml", ".rkt"))]
            entry = xml_entries[0] if xml_entries else names[0]
            return zf.read(entry)
    except _zipfile.BadZipFile:
        with open(filepath, "rb") as fh:
            return fh.read()


def _parse_rocksim_attached(
    attached:      _ET.Element,
    tube_top_mm:   float,
    body_radius_m: float,
    components:    list,
    extracted:     dict,
) -> None:
    """Parse every element inside a <BodyTube>'s <AttachedParts>."""
    from utils.geometry_math import (
        Component, moi_hollow_cylinder, moi_point_mass, moi_trapezoidal_fin_set,
    )
    for child in attached:
        tag      = child.tag
        xb_mm    = _rkt_float(child, "Xb",  0.0)
        len_mm   = _rkt_float(child, "Len", 0.0)
        top_mm   = tube_top_mm + xb_mm
        top_m    = top_mm  / 1000.0
        len_m    = len_mm  / 1000.0
        mass_kg  = _rkt_mass_kg(child)

        if tag in ("TrapezoidalFinSet", "EllipticalFinSet",
                   "FinSet", "CustomFinSet"):
            fin_n  = max(1, int(_rkt_float(child, "FinCount",  4.0)))
            root_m = _rkt_float(child, "RootChord", 0.0) / 1000.0
            # TrapezoidalFinSet uses TipChord; EllipticalFinSet has no tip chord
            if tag == "EllipticalFinSet":
                tip_m = 0.0
            else:
                tip_m = _rkt_float(child, "TipChord", 0.0) / 1000.0
            # Try multiple span tag names used across RockSim versions
            span_mm = (_rkt_float(child, "SemiSpan", 0.0) or
                       _rkt_float(child, "Span",     0.0) or
                       _rkt_float(child, "Height",   0.0))
            span_m = span_mm / 1000.0
            # root_m must be > 0; derive from Len (fin axial length) as last resort
            if root_m <= 0.0:
                root_m = len_m if len_m > 0.0 else _AIRFRAME_DEFAULTS["fin_root"]
            cg_m   = top_m + root_m / 2.0   # axial CG ≈ mid-chord
            moi    = moi_trapezoidal_fin_set(mass_kg, fin_n, body_radius_m,
                                             root_m, tip_m, span_m)
            components.append(Component("FinSet", mass_kg, cg_m, moi))
            extracted.update({"fin_root_m": root_m, "fin_tip_m": tip_m,
                               "fin_span_m": span_m, "fin_pos_m": top_m,
                               "fin_count": fin_n})

        elif tag == "Parachute":
            dia_mm = _rkt_float(child, "Dia", 0.0)
            cd     = _rkt_float(child, "Cd",  1.5)
            area   = _math.pi * (dia_mm / 2000.0) ** 2
            cg_m   = top_m + len_m / 2.0
            components.append(Component("Parachute", mass_kg, cg_m,
                                        moi_point_mass(mass_kg)))
            if extracted.get("para_area_m2") is None:   # first parachute wins
                extracted["para_cd"]      = cd
                extracted["para_area_m2"] = area

        elif tag in ("Ring", "BodyTubeCoupler", "Coupler"):
            r_out = _rkt_float(child, "DiaOuter", 0.0) / 2000.0
            r_inn = _rkt_float(child, "DiaInner", 0.0) / 2000.0
            cg_m  = top_m + len_m / 2.0
            moi   = moi_hollow_cylinder(mass_kg, r_out,
                                        min(r_inn, r_out * 0.98), len_m)
            components.append(Component(tag, mass_kg, cg_m, moi))

        elif tag == "Motor":
            # RockSim Motor: InitialWt = full mass (g), PropWt = propellant (g)
            init_g  = _rkt_float(child, "InitialWt", 0.0)
            prop_g  = _rkt_float(child, "PropWt",    0.0)
            dry_kg  = max(0.0, init_g - prop_g) / 1000.0
            delay_s = _rkt_float(child, "Delays",    0.0)
            r_mot   = _rkt_float(child, "Dia", 0.0) / 2000.0
            cg_m    = top_m + len_m / 2.0
            moi     = (moi_hollow_cylinder(dry_kg, r_mot, r_mot * 0.85, len_m)
                       if r_mot > 0.0 else moi_point_mass(dry_kg))
            components.append(Component("Motor", dry_kg, cg_m, moi))
            extracted["motor_pos_m"]       = cg_m
            extracted["motor_dry_mass_kg"] = dry_kg
            extracted["backfire_delay_s"]  = delay_s

        elif tag in ("MassObject", "LaunchLug"):
            cg_m = top_m + len_m / 2.0
            components.append(Component(tag, mass_kg, cg_m,
                                        moi_point_mass(mass_kg)))
            # Heuristic: treat largest MassObject as motor placeholder when
            # no explicit <Motor> element is present.
            if tag == "MassObject":
                if mass_kg > extracted.get("motor_dry_mass_kg", 0.0):
                    extracted["motor_pos_m"]       = cg_m
                    extracted["motor_dry_mass_kg"] = mass_kg


def _parse_rocksim_xml(root: _ET.Element) -> dict:
    """Traverse <RockSimDocument> and return the AppState-ready result dict."""
    from utils.geometry_math import (
        Component, calculate_system_cg, calculate_total_moi,
        moi_hollow_cone, moi_hollow_cylinder, moi_point_mass,
    )

    design = root.find(".//RocketDesign")
    if design is None:
        raise RocketConfigError("No <RocketDesign> element found in .rkt file")

    # Locate stage container (single-stage → Stage3Parts; multi-stage → highest)
    stage_parts = None
    for stag in ("Stage3Parts", "Stage2Parts", "Stage1Parts"):
        stage_parts = design.find(stag)
        if stage_parts is not None:
            break
    if stage_parts is None:
        stage_parts = design   # fall back to direct children

    components: list = []
    extracted: dict = {
        "total_length_m":    0.0,
        "body_radius_m":     0.0,
        "nose_length_m":     0.0,
        "fin_root_m":        None,
        "fin_tip_m":         None,
        "fin_span_m":        None,
        "fin_pos_m":         None,
        "fin_count":         4,
        "motor_pos_m":       None,
        "motor_dry_mass_kg": 0.0,
        "backfire_delay_s":  0.0,
        "para_cd":           1.5,
        "para_area_m2":      None,
        "para_lag_s":        0.5,
    }

    # ── Sequential traversal of top-level stage components ───────────────────
    # Components are stacked nose→tail; each starts where the previous ends.
    # Xb on Stage3Parts children is typically 0 and is intentionally ignored
    # here to avoid ambiguity — sequential stacking is the authoritative model.
    cursor_mm = 0.0
    _SEQ_TAGS = ("NoseCone", "BodyTube", "BodyTubeCoupler",
                 "Transition", "TailCone")
    for child in stage_parts:
        if child.tag not in _SEQ_TAGS:
            continue

        len_mm   = _rkt_float(child, "Len", 0.0)
        top_mm   = cursor_mm
        top_m    = top_mm  / 1000.0
        len_m    = len_mm  / 1000.0
        mass_kg  = _rkt_mass_kg(child)
        r_out    = _rkt_float(child, "DiaOuter", 0.0) / 2000.0
        r_inn    = _rkt_float(child, "DiaInner", 0.0) / 2000.0

        if child.tag == "NoseCone":
            # Solid-cone CG ≈ 2/3 from tip; hollow cone is close enough.
            cg_m = top_m + len_m * 2.0 / 3.0
            moi  = moi_hollow_cone(mass_kg, r_out, len_m)
            components.append(Component("NoseCone", mass_kg, cg_m, moi))
            extracted["nose_length_m"] = len_m
            if extracted["body_radius_m"] == 0.0:
                extracted["body_radius_m"] = r_out

        else:   # BodyTube / Transition / TailCone
            cg_m      = top_m + len_m / 2.0
            r_inn_use = min(r_inn, r_out * 0.98) if r_inn > 0.0 else r_out * 0.90
            moi       = moi_hollow_cylinder(mass_kg, r_out, r_inn_use, len_m)
            components.append(Component(child.tag, mass_kg, cg_m, moi))
            if extracted["body_radius_m"] == 0.0:
                extracted["body_radius_m"] = r_out

            attached = child.find("AttachedParts")
            if attached is not None:
                _parse_rocksim_attached(
                    attached, top_mm, extracted["body_radius_m"],
                    components, extracted,
                )

        cursor_mm = top_mm + len_mm

    extracted["total_length_m"] = cursor_mm / 1000.0

    # ── MoI summation via geometry_math engine ────────────────────────────────
    valid      = [c for c in components if c.mass > 0.0]
    total_mass = sum(c.mass for c in valid)
    system_cg  = calculate_system_cg(valid) if valid else 0.0
    system_moi = calculate_total_moi(valid, system_cg) if valid else None

    para_area = (extracted["para_area_m2"]
                 if extracted["para_area_m2"] is not None
                 else _math.pi * 0.15 ** 2)   # 300 mm diameter fallback

    return {
        "airframe": {
            "mass":           total_mass,
            "cg":             system_cg,
            "length":         extracted["total_length_m"],
            "radius":         extracted["body_radius_m"],
            "nose_length":    extracted["nose_length_m"],
            "fin_root":       extracted["fin_root_m"]        or _AIRFRAME_DEFAULTS["fin_root"],
            "fin_tip":        extracted["fin_tip_m"]         or _AIRFRAME_DEFAULTS["fin_tip"],
            "fin_span":       extracted["fin_span_m"]        or _AIRFRAME_DEFAULTS["fin_span"],
            "fin_pos":        extracted["fin_pos_m"]         or _AIRFRAME_DEFAULTS["fin_pos"],
            "motor_pos":      extracted["motor_pos_m"]       or system_cg,
            "motor_dry_mass": extracted["motor_dry_mass_kg"],
            "backfire_delay": extracted["backfire_delay_s"],
        },
        "parachute": {
            "cd":   extracted["para_cd"],
            "area": para_area,
            "lag":  extracted["para_lag_s"],
        },
        "moi": {
            "ixx": system_moi.Ixx if system_moi else 0.0,
            "iyy": system_moi.Iyy if system_moi else 0.0,
            "izz": system_moi.Izz if system_moi else 0.0,
        },
    }


def parse_rkt_file(filepath: str) -> dict:
    """Parse a RockSim .rkt file into AppState-ready SI parameters.

    The file is first attempted as a ZIP archive (newer OpenRocket / RockSim
    variants wrap the XML in a ZIP); if that fails the file is read as raw XML.

    Returns:
        {
            "airframe":  { mass, cg, length, radius, nose_length, fin_root,
                           fin_tip, fin_span, fin_pos, motor_pos,
                           motor_dry_mass, backfire_delay },
            "parachute": { cd, area, lag },
            "moi":       { ixx, iyy, izz },      # kg·m²
        }
        All values are in SI (m, kg, s).

    Raises:
        RocketConfigError: on any I/O or parse failure.
    """
    try:
        xml_bytes = _load_rkt_bytes(filepath)
        root      = _ET.fromstring(xml_bytes)
    except _ET.ParseError as exc:
        raise RocketConfigError(f".rkt XML parse error: {exc}") from exc
    except Exception as exc:
        raise RocketConfigError(f"Cannot read .rkt file: {exc}") from exc

    if root.tag == "RockSimDocument":
        return _parse_rocksim_xml(root)

    if root.tag == "openrocket":
        raise RocketConfigError(
            "OpenRocket native (.ork) format is not supported.\n"
            "Please export from OpenRocket using File → Export → RockSim (.rkt)."
        )

    raise RocketConfigError(
        f"Unrecognised file format: root element <{root.tag}>.\n"
        "Expected a RockSim <RockSimDocument>."
    )


# ── Parachute-only JSON parser ────────────────────────────────────────────────

def parse_parachute_json(filepath: str) -> dict:
    """Parse a Parachute.json file → {'cd': float, 'area': float, 'lag': float}.

    Accepted keys
    -------------
    cd          Drag coefficient (dimensionless, required > 0)
    area        Reference area (m²)          ← used if present
    diameter    Canopy diameter (m)           ← used to compute area if 'area' absent
    lag         Deployment lag (s)

    Missing keys fall back to *_PARACHUTE_DEFAULTS*.

    Raises:
        RocketConfigError: on file I/O, JSON decode, or invalid value.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError as exc:
        raise RocketConfigError(f"File not found: {filepath}") from exc
    except json.JSONDecodeError as exc:
        raise RocketConfigError(f"Invalid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise RocketConfigError("Parachute JSON must be a top-level object { … }")

    try:
        cd  = float(raw.get("cd",  _PARACHUTE_DEFAULTS["cd"]))
        lag = float(raw.get("lag", _PARACHUTE_DEFAULTS["lag"]))

        if "area" in raw:
            area = float(raw["area"])
        elif "diameter" in raw:
            d    = float(raw["diameter"])
            area = _math.pi * (d / 2.0) ** 2
        else:
            area = float(_PARACHUTE_DEFAULTS["area"])
    except (TypeError, ValueError) as exc:
        raise RocketConfigError(f"Non-numeric value in parachute file: {exc}") from exc

    if cd   <= 0.0:
        raise RocketConfigError(f"Parachute Cd must be > 0, got {cd}")
    if area <= 0.0:
        raise RocketConfigError(f"Parachute area must be > 0, got {area}")

    return {"cd": cd, "area": area, "lag": lag}


# ── Mach-dependent Cd curve parser (Phase C) ──────────────────────────────────

def parse_cd_curve_csv(filepath: str) -> list[tuple[float, float]]:
    """Parse a 2-column ``(Mach, Cd)`` CSV file into a sorted list of tuples.

    The returned list is consumed directly by RocketPy's ``Rocket(...)``
    constructor through ``power_on_drag`` / ``power_off_drag``, which accept
    a list of ``(Mach, Cd)`` pairs and interpolate linearly between them.

    Format
    ------
    *   Two columns, comma-separated. Either column may contain a header
        label as long as the first row is *purely textual* (i.e. neither
        cell parses as a float).  All-numeric rows are taken as data.
    *   Empty lines and whitespace-only lines are ignored at any position.
    *   Cells are stripped of surrounding whitespace before parsing.
    *   The UTF-8 BOM (``\\ufeff``), if present, is consumed automatically.

    Example accepted layouts (any of these is fine)::

        Mach, Cd                ← header (skipped)
        0.0,  0.50
        0.5,  0.48
        1.0,  0.62

        0.00, 0.50              ← no header
        0.50, 0.48
        1.00, 0.62

    Validation
    ----------
    Raises :class:`ValueError` when:
        *   The file contains no parsable ``(Mach, Cd)`` rows.
        *   Any row has fewer than two cells.
        *   A non-numeric row appears **after** valid data (= corrupt body).
        *   Mach or Cd is negative.
        *   The same Mach number appears twice (ambiguous Cd).

    Raises :class:`OSError` (or subclasses) when the file cannot be opened.

    Parameters
    ----------
    filepath : str
        Path to the CSV file on disk.

    Returns
    -------
    list[tuple[float, float]]
        Non-empty, sorted by ascending Mach; ready for RocketPy.
    """
    import csv as _csv

    rows: list[tuple[float, float]] = []

    # ``utf-8-sig`` transparently strips the BOM that some spreadsheet apps
    # (e.g. Excel) prepend when exporting CSV.  ``newline=""`` is the
    # documented best practice for the ``csv`` module.
    with open(filepath, "r", encoding="utf-8-sig", newline="") as fh:
        reader = _csv.reader(fh)
        for line_no, raw in enumerate(reader, start=1):
            if not raw:
                continue
            stripped = [c.strip() for c in raw]
            # Skip blank lines (all cells empty after stripping).
            if all(c == "" for c in stripped):
                continue
            if len(stripped) < 2:
                raise ValueError(
                    f"{filepath}:{line_no}: expected 2 columns (Mach, Cd), "
                    f"got {len(stripped)}"
                )

            try:
                mach = float(stripped[0])
                cd   = float(stripped[1])
            except ValueError:
                # A non-numeric row is acceptable ONLY as a header before any
                # data has been collected — after that it indicates corruption.
                if rows:
                    raise ValueError(
                        f"{filepath}:{line_no}: non-numeric data "
                        f"{stripped[:2]!r} after valid rows"
                    ) from None
                continue

            if mach < 0.0 or cd < 0.0:
                raise ValueError(
                    f"{filepath}:{line_no}: Mach and Cd must be non-negative "
                    f"(got mach={mach}, cd={cd})"
                )

            rows.append((mach, cd))

    if not rows:
        raise ValueError(
            f"{filepath}: no valid (Mach, Cd) data rows found"
        )

    rows.sort(key=lambda r: r[0])

    # Strict-increase check: RocketPy interpolation would silently misbehave
    # on duplicate Mach values.
    for i in range(1, len(rows)):
        if rows[i][0] == rows[i - 1][0]:
            raise ValueError(
                f"{filepath}: duplicate Mach value {rows[i][0]} — "
                "Cd curve must have strictly increasing Mach numbers"
            )

    return rows
