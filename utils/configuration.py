"""Validated operator configuration, independent of Qt and result archives."""
import json
import math
from pathlib import Path

from core.constants import OBS_ALT
from utils.result_io import atomic_write

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config.json'
SECTIONS = {
    'montblanc': {
        'max_speed_ms': ('montblanc_max_speed_ms', 0, 1000),
        'window_s': ('montblanc_window_s', .01, 120),
        'max_variation_ms': ('montblanc_max_variation_ms', 0, 1000),
    },
    'simulation': {
        'wind_uncertainty': ('wind_uncertainty', 0, 1),
        'thrust_uncertainty': ('thrust_uncertainty', 0, 1),
        'landing_prob': ('landing_prob', 50, 99),
        'mc_n_runs': ('mc_n_runs', 10, 5000),
    },
    'motor': {
        'nozzle_pos_m': ('motor_cg_pos', 0, 1000),
        'dry_mass_kg': ('motor_dry_mass', .000001, 50),
        'backfire_delay_s': ('backfire_delay', 0, 60),
        'isp_s': ('motor_isp', 40, 300),
        'propellant_density': ('motor_propellant_density', 500, 2500),
    },
    'parachute': {
        'cd': ('parachute_cd', .000001, 100),
        'area_m2': ('parachute_area', .000001, 100),
        'lag_s': ('parachute_lag', 0, 60),
        'wind_drift_pct': ('parachute_wind_drift_pct', 0, 200),
    },
    'aerodynamics': {
        'power_on_cd': ('power_on_cd', 0, 10),
        'power_off_cd': ('power_off_cd', 0, 10),
    },
}


def validate_config(data):
    if not isinstance(data, dict) or data.get('version', 1) != 1:
        raise ValueError('Configuration must be a version 1 object')
    unknown = set(data) - set(SECTIONS) - {'version', 'practice_wind'}
    if unknown:
        raise ValueError('Unknown config sections: ' + ', '.join(sorted(unknown)))
    result = {'version': 1}
    for section, fields in SECTIONS.items():
        raw = data.get(section, {})
        if not isinstance(raw, dict) or set(raw)-set(fields):
            raise ValueError(f'Invalid {section} configuration fields')
        result[section] = {}
        for key, value in raw.items():
            if value is not None:
                _, lo, hi = fields[key]
                if isinstance(value, bool):
                    raise ValueError(f'{section}.{key} must be numeric')
                value = float(value)
                if not math.isfinite(value) or not lo <= value <= hi:
                    raise ValueError(f'{section}.{key} must be between {lo} and {hi}')
                if key in ('landing_prob', 'mc_n_runs'):
                    if not value.is_integer():
                        raise ValueError(f'{key} must be an integer')
                    value = int(value)
            result[section][key] = value
    if 'practice_wind' in data:
        nodes = data['practice_wind']
        if not isinstance(nodes, list) or not nodes:
            raise ValueError('practice_wind must contain altitude/speed/direction nodes')
        result['practice_wind'] = []
        for n in nodes:
            if not isinstance(n, dict) or set(n) != {'alt_m', 'speed_ms', 'dir_deg'}:
                raise ValueError('Invalid practice wind node')
            node = {k:float(v) for k,v in n.items()}
            if not all(math.isfinite(v) for v in node.values()) or node['alt_m'] < 0 or node['speed_ms'] < 0 or not 0 <= node['dir_deg'] <= 360:
                raise ValueError('Invalid practice wind values')
            result['practice_wind'].append(node)
        if len({n['alt_m'] for n in result['practice_wind']}) != len(nodes):
            raise ValueError('Duplicate practice wind altitude')
        # Older configs stored the Koinobori surface node at 3 m. Preserve
        # its speed/direction while updating the installation height.
        if min(n['alt_m'] for n in result['practice_wind']) == 3.0:
            for node in result['practice_wind']:
                if node['alt_m'] == 3.0:
                    node['alt_m'] = OBS_ALT
    return result


def read_config(path=CONFIG_PATH):
    return validate_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def apply_config(state, data):
    data = validate_config(data)  # Validate every section before changing state.
    for section, fields in SECTIONS.items():
        for key, value in data[section].items():
            if section == 'montblanc':
                if value is not None:
                    state.montblanc_options = {key: value}
                continue
            prop = fields[key][0]
            provenance = getattr(state, 'airframe_provenance', {})
            source_key = {'nozzle_pos_m': 'motor_pos', 'backfire_delay_s': 'backfire_delay'}.get(key)
            if section == 'motor' and source_key and provenance.get('sources', {}).get(source_key) in ('ork', 'calculated'):
                continue
            if section == 'parachute' and {'area_m2': 'area', 'cd': 'cd'}.get(key) in provenance.get('parachute', {}):
                continue
            if value is not None:
                setattr(state, prop, value)
            elif prop in ('motor_cg_pos', 'motor_dry_mass', 'backfire_delay',
                           'parachute_cd', 'parachute_area', 'parachute_lag'):
                setattr(state, prop, None)
    if 'practice_wind' in data:
        state.phase4_options = dict(wind_csv='', wind_time_utc='')
        state.wind_profile_data = data['practice_wind']


def state_config(state):
    data = {'version': 1, 'practice_wind': state.wind_profile_data}
    for section, fields in SECTIONS.items():
        if section == 'montblanc':
            data[section] = state.montblanc_options
            continue
        data[section] = {key:getattr(state, prop) for key,(prop,_,_) in fields.items()}
    return validate_config(data)


def save_config(state, path=CONFIG_PATH):
    data = state_config(state)
    atomic_write(path, (json.dumps(data, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))


def load_motor_config(state, path):
    raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    apply_config(state, {'motor': raw})


def save_motor_config(state, path):
    data = state_config(state)['motor']
    atomic_write(path, (json.dumps(data, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
