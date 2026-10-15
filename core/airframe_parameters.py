"""Resolve design parameters in order: OpenRocket, calculated, manual."""
from copy import deepcopy
import math


def calculate_cp(airframe):
    """Mach-zero CP using the same surfaces as the flight model (no Flight run)."""
    from rocketpy import Rocket
    from .simulation import _add_fin_set
    rocket = Rocket(radius=float(airframe['radius']), mass=1,
                    inertia=(1, 1, 1), power_off_drag=0, power_on_drag=0,
                    center_of_mass_without_motor=0,
                    coordinate_system_orientation='nose_to_tail')
    rocket.add_nose(length=float(airframe['nose_length']),
                    kind=airframe['nose_shape'], position=0)
    _add_fin_set(rocket, airframe, float(airframe['fin_pos']),
                 float(airframe['fin_root']), float(airframe['fin_tip']),
                 float(airframe['fin_span']))
    cp = float(rocket.cp_position(0))
    if not math.isfinite(cp) or not 0 <= cp <= float(airframe['length']):
        raise ValueError('Calculated CP is outside the supported airframe range')
    return cp


def resolve_airframe(data):
    """Preserve provenance and enforce precedence, including on saved JSON reload.

    Legacy/manual-only JSON keeps its existing semantics. An OpenRocket import
    carries its immutable source values and calculated estimates in `provenance`.
    CP is recalculated from resolved geometry, never taken from stale flight data.
    """
    result = deepcopy(data)
    provenance = result.get('provenance')
    if provenance is None:
        return result
    if not isinstance(provenance, dict):
        raise ValueError('provenance must be an object')
    for layer in ('ork', 'calculated'):
        if not isinstance(provenance.get(layer, {}), dict):
            raise ValueError(f'provenance.{layer} must be an object')
        allowed = {'mass','cg','cp','length','radius','nose_length','nose_shape',
                   'fin_root','fin_tip','fin_span','fin_pos','fin_count',
                   'fin_shape','fin_points','fin_sweep','fin_cant',
                   'motor_pos','motor_dry_mass','backfire_delay'}
        if set(provenance.get(layer, {}))-allowed:
            raise ValueError(f'Unknown fields in provenance.{layer}')
    if not isinstance(provenance.get('warnings', []), list) or not all(
            isinstance(v,str) for v in provenance.get('warnings', [])):
        raise ValueError('provenance.warnings must be a list of strings')
    parachute = provenance.get('parachute', {})
    if not isinstance(parachute, dict) or set(parachute)-{'area','cd'}:
        raise ValueError('Invalid parachute metadata')
    for key,value in parachute.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid parachute {key}')
    calculated = provenance.setdefault('calculated', {})
    ork = provenance.setdefault('ork', {})
    for layer in (calculated,ork):
        for key in list(layer):
            if layer[key] is None and key not in ('fin_points','fin_sweep'):
                del layer[key]
    result.update(calculated)
    result.update(ork)
    if 'cp' not in ork:
        # Previous calculated CP cannot survive geometry changes or failures.
        calculated.pop('cp', None)
        result['cp'] = (data.get('cp') if provenance.get('sources', {}).get('cp')
                        not in ('ork','calculated') else None)
        try:
            result['cp'] = calculated['cp'] = calculate_cp(result)
        except (ValueError, TypeError, KeyError, ZeroDivisionError):
            pass
    provenance['sources'] = {key: 'manual' for key in result
                             if key not in ('provenance',) and result[key] is not None}
    provenance['sources'].update({key: 'calculated' for key in calculated})
    provenance['sources'].update({key: 'ork' for key in ork})
    return result
