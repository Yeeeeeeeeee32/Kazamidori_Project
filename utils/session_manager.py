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


def dict_to_state(state: "object", data: dict[str, Any]) -> None:
    """Apply a previously-saved snapshot to the live AppState.

    Each field is written through the AppState property *setter* so the
    change signals fire — UI bindings, plot views, and (any future)
    persisted snapshot subscribers see the load just as they would see
    a manual edit in the Advanced Settings dialog.

    Missing keys are silently kept at their current value: this makes
    partial/legacy session files forward-compatible (e.g. a session
    saved before Cd curves existed still loads cleanly).

    Raises
    ------
    SessionError
        If the data dict has the wrong top-level shape, an unknown
        ``schema_version``, or a Cd curve that cannot be re-tupled.
    """
    if not isinstance(data, dict):
        raise SessionError(
            f"Session file must contain a JSON object, got {type(data).__name__}"
        )

    schema = data.get("schema_version", _SCHEMA_VERSION)
    if not isinstance(schema, int) or schema > _SCHEMA_VERSION:
        raise SessionError(
            f"Unsupported session schema_version={schema!r} "
            f"(this build understands ≤ {_SCHEMA_VERSION})"
        )

    aero = data.get("aerodynamics") or {}
    if not isinstance(aero, dict):
        raise SessionError("`aerodynamics` section must be an object")

    if "power_on_cd" in aero:
        state.power_on_cd = float(aero["power_on_cd"])
    if "power_off_cd" in aero:
        state.power_off_cd = float(aero["power_off_cd"])

    # Cd curves: critical tuple re-conversion happens inside _curve_from_jsonable.
    if "cd_curve_power_on" in aero:
        state.cd_curve_power_on = _curve_from_jsonable(aero["cd_curve_power_on"])
    if "cd_curve_power_off" in aero:
        state.cd_curve_power_off = _curve_from_jsonable(aero["cd_curve_power_off"])

    motor = data.get("motor") or {}
    if not isinstance(motor, dict):
        raise SessionError("`motor` section must be an object")

    if "isp" in motor:
        state.motor_isp = float(motor["isp"])
    if "propellant_density" in motor:
        state.motor_propellant_density = float(motor["propellant_density"])


def save_session(state: "object", filepath: str) -> None:
    """Serialise ``state`` to *filepath* as pretty-printed JSON (UTF-8).

    Raises :class:`OSError` if the file cannot be written.
    """
    payload = state_to_dict(state)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


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
