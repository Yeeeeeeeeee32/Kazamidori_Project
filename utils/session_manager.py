"""
utils/session_manager.py

Round-trip persistence layer for the operator-facing pieces of
:class:`ui_qt.app_state.AppState`.

Scope
-----
Only the **Advanced Settings** parameters introduced in Phase B / C are
persisted: aerodynamic drag coefficients, Mach-dependent drag curves, and
motor propellant defaults.  The MVVM contract is strict — this module
talks to :class:`AppState` only; it never imports or instantiates Qt
widgets, and never touches the simulation core directly.

JSON layout (``schema_version = 1``)
------------------------------------
::

    {
      "schema_version": 1,
      "aerodynamics": {
        "power_on_cd":  0.45,
        "power_off_cd": 0.40,
        "cd_curve_power_on":  null | [[mach, cd], [mach, cd], ...],
        "cd_curve_power_off": null | [[mach, cd], [mach, cd], ...]
      },
      "motor": {
        "isp":               80.0,
        "propellant_density": 1700.0
      }
    }

Tuple round-trip pitfall
------------------------
``json.dump`` silently converts Python ``tuple`` objects to JSON arrays,
and ``json.load`` then returns plain Python ``list`` objects.  RocketPy
and our own ``simulate_once`` accept ``list[(Mach, Cd)]`` semantically,
but type-checkers and equality comparisons (used by ``AppState`` change
detection) treat ``[1, 2]`` and ``(1, 2)`` as distinct.  This module
re-tuples every entry on load so the AppState payload is exactly what
Phase B / C handed us originally.
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional


# ── Schema metadata ──────────────────────────────────────────────────────────

_SCHEMA_VERSION: int = 1


class SessionError(Exception):
    """Raised for unrecoverable session-file format errors (caller-friendly)."""


# ── Cd curve tuple helpers ───────────────────────────────────────────────────

def _curve_to_jsonable(
    curve: Optional[list[tuple[float, float]]],
) -> Optional[list[list[float]]]:
    """Convert a Cd curve (list of tuples) to JSON-friendly nested lists.

    ``None`` is preserved verbatim — JSON's ``null`` round-trips cleanly.
    """
    if curve is None:
        return None
    return [[float(m), float(cd)] for (m, cd) in curve]


def _curve_from_jsonable(
    raw: Any,
) -> Optional[list[tuple[float, float]]]:
    """Convert a JSON-decoded Cd curve back into ``list[tuple[float, float]]``.

    Accepts:
      *  ``None``                                              → returns ``None``
      *  ``[]`` (empty list, e.g. cleared by external tooling) → returns ``None``
      *  ``[[mach, cd], [mach, cd], ...]``                     → list of tuples

    Anything else raises :class:`SessionError` so the caller can surface a
    descriptive warning instead of letting an invalid payload reach the
    physics core.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise SessionError(
            f"Cd curve must be a list or null, got {type(raw).__name__}"
        )
    if len(raw) == 0:
        return None

    result: list[tuple[float, float]] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise SessionError(
                f"Cd curve entry #{i} must be a 2-element list [mach, cd]; "
                f"got {entry!r}"
            )
        try:
            mach = float(entry[0])
            cd   = float(entry[1])
        except (TypeError, ValueError) as exc:
            raise SessionError(
                f"Cd curve entry #{i} is not numeric: {entry!r}"
            ) from exc
        result.append((mach, cd))
    if (len(result) < 2 or any(not math.isfinite(v) or v < 0 for p in result for v in p)
            or any(a[0] >= b[0] for a,b in zip(result,result[1:]))):
        raise SessionError('Cd curve requires finite non-negative values and increasing Mach')
    return result


# ── Public API ───────────────────────────────────────────────────────────────

def state_to_dict(state: "object") -> dict[str, Any]:
    """Snapshot the operator-facing AppState properties as a JSON-safe dict.

    The returned dict mirrors the schema documented at the top of this
    module.  Float fields are coerced explicitly so a future ``Decimal``
    or ``numpy.float64`` slipped into AppState would not poison JSON
    encoding downstream.

    Parameters
    ----------
    state
        An ``AppState`` instance (duck-typed; only the named properties
        are read so unit tests can pass a simple namespace mock).
    """
    return {
        "schema_version": _SCHEMA_VERSION,
        "phase4": dict(getattr(state, 'phase4_options', {})),
        "montblanc": dict(getattr(state, 'montblanc_options', {})),
        "aerodynamics": {
            "power_on_cd":         float(state.power_on_cd),
            "power_off_cd":        float(state.power_off_cd),
            "cd_curve_power_on":   _curve_to_jsonable(state.cd_curve_power_on),
            "cd_curve_power_off":  _curve_to_jsonable(state.cd_curve_power_off),
        },
        "motor": {
            "isp":                float(state.motor_isp),
            "propellant_density": float(state.motor_propellant_density),
        },
    }


def _session_changes(state: "object", data: dict[str, Any]) -> dict:
    """Validate the complete draft before changing any live property."""
    if not isinstance(data,dict):
        raise SessionError('Session must be an object')
    schema = data.get('schema_version',_SCHEMA_VERSION)
    if isinstance(schema,bool) or not isinstance(schema,int) or not 1 <= schema <= _SCHEMA_VERSION:
        raise SessionError('Unsupported session schema_version')
    changes = {}
    if 'montblanc' in data:
        from core.montblanc import normalize_options
        try:
            changes['montblanc_options'] = normalize_options(data['montblanc'])
        except (TypeError, ValueError) as exc:
            raise SessionError(str(exc)) from exc
    try:
        for section,fields in {
            'aerodynamics':{'power_on_cd':('power_on_cd',0,10),'power_off_cd':('power_off_cd',0,10)},
            'motor':{'isp':('motor_isp',40,300),'propellant_density':('motor_propellant_density',500,2500)}
        }.items():
            raw = data.get(section,{})
            if not isinstance(raw,dict):
                raise SessionError(f'{section} must be an object')
            for key,(prop,lo,hi) in fields.items():
                if key in raw:
                    value = float(raw[key])
                    if isinstance(raw[key],bool) or not math.isfinite(value) or not lo <= value <= hi:
                        raise SessionError(f'{section}.{key} outside its valid range')
                    changes[prop] = value
        for key in ('cd_curve_power_on','cd_curve_power_off'):
            if key in data.get('aerodynamics',{}):
                changes[key] = _curve_from_jsonable(data['aerodynamics'][key])
        phase4 = data.get('phase4',{})
        if not isinstance(phase4,dict):
            raise SessionError('phase4 must be an object')
        if phase4:
            from utils.phase4_settings import normalize_phase4
            changes['phase4_options'] = normalize_phase4(phase4,state.phase4_options)
    except (ValueError,TypeError,AttributeError) as exc:
        raise SessionError(str(exc)) from exc
    return changes


def dict_to_state(state: "object", data: dict[str, Any]) -> None:
    for key,value in _session_changes(state,data).items():
        setattr(state,key,value)


def save_session(state: "object", filepath: str) -> None:
    """Serialise ``state`` to *filepath* as pretty-printed JSON (UTF-8).

    Raises :class:`OSError` if the file cannot be written.
    """
    payload = state_to_dict(state)
    _session_changes(state,payload)
    from utils.result_io import atomic_write
    atomic_write(filepath,(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))


def load_session(state: "object", filepath: str) -> None:
    """Read a session JSON from *filepath* and apply it to ``state``.

    Raises:
        OSError       – if the file cannot be opened.
        SessionError  – on JSON decode error or schema/shape violations.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SessionError(f"Invalid JSON in {filepath}: {exc}") from exc
    dict_to_state(state, data)
