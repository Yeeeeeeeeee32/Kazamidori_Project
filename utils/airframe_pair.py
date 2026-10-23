"""Resolve an OpenRocket design and a small JSON supplement as one airframe."""
from pathlib import Path
import json

from core.airframe_parameters import resolve_airframe
from utils.openrocket_airframe import load_openrocket_airframe


PARAMETER_FIELDS = {
    'mass', 'cg', 'cp', 'length', 'radius', 'nose_length', 'nose_shape',
    'fin_root', 'fin_tip', 'fin_span', 'fin_pos', 'fin_count', 'fin_shape',
    'fin_points', 'fin_sweep', 'fin_cant', 'motor_pos', 'motor_dry_mass',
    'backfire_delay',
}


def read_json(path):
    with open(path, encoding='utf-8-sig') as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError('Airframe JSON must contain an object at the top level.')
    return data


def load_pair(ork_path, supplement_path=None, supplement=None):
    ork_path = Path(ork_path).resolve()
    data = load_openrocket_airframe(ork_path)
    provenance = data['provenance']
    provenance['source_path'] = str(ork_path)
    if supplement_path is not None:
        supplement_path = Path(supplement_path).resolve()
        supplement = read_json(supplement_path) if supplement is None else supplement
        # Legacy ork_file metadata never selects or restricts the chosen design.
        unknown = set(supplement) - PARAMETER_FIELDS - {'ork_file'}
        if unknown:
            raise ValueError('Unknown supplement fields: ' + ', '.join(sorted(unknown)))
        values = {key: value for key, value in supplement.items() if key != 'ork_file'}
        data.update(values)
        provenance['supplement_path'] = str(supplement_path)
        provenance['supplement_values'] = values
        duplicates = set(values) & (set(provenance['ork']) | set(provenance['calculated']))
        if duplicates:
            provenance['warnings'].append(
                'JSONの重複項目は.ork・自動計算を優先します: ' + ', '.join(sorted(duplicates)))
    return resolve_airframe(data)


def supplement_for_save(filepath, validated):
    """Write only unresolved parameters; the two files are selected manually."""
    provenance = validated.get('provenance', {})
    source = provenance.get('source_path')
    if not source:
        # Legacy snapshots without a locatable source remain self-contained.
        return validated
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError(f'OpenRocket source is missing: {source}. Reload the design before saving.')
    destination = Path(filepath).resolve()
    if destination == source:
        raise ValueError('The supplement must not overwrite the OpenRocket file.')
    supplied = set(provenance.get('ork', {})) | set(provenance.get('calculated', {}))
    return {
        key: value for key, value in validated.items()
        if key in PARAMETER_FIELDS and key not in supplied and value is not None
        and key not in ('motor_pos', 'motor_dry_mass', 'backfire_delay')
    }
