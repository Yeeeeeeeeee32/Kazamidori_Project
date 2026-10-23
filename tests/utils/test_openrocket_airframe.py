import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from core.airframe_parameters import resolve_airframe, calculate_cp
from utils.data_loader import load_airframe_config, save_airframe_config, AirframeConfigError
from utils.openrocket_airframe import _position

FIXTURE = Path(__file__).parents[1] / 'fixtures' / 'single_stage.ork'


def test_full_import_positions_overrides_recovery_and_default_motor():
    data = load_airframe_config(FIXTURE)
    assert data['length'] == pytest.approx(.4)
    assert data['fin_pos'] == pytest.approx(.33)
    assert data['motor_pos'] == pytest.approx(.405)
    assert data['backfire_delay'] == 3
    assert data['motor_dry_mass'] is None
    assert data['mass'] == .25 and data['cg'] == .2
    assert data['provenance']['sources']['mass'] == 'ork'
    assert data['provenance']['sources']['cp'] == 'calculated'
    assert data['cp'] == pytest.approx(calculate_cp(data))
    assert data['provenance']['motor'] == 'Estes C5'
    assert data['provenance']['parachute'] == dict(cd=1,area=math.pi*.25**2)
    assert 'lag' not in data['provenance']['parachute']


@pytest.mark.parametrize('method,expected', [('top',.12),('middle',.24),('bottom',.36),('absolute',.02),('after',.32)])
def test_component_reference_positions(method,expected):
    e=ET.fromstring(f'<trapezoidfinset><rootchord>.06</rootchord><axialoffset method="{method}">.02</axialoffset></trapezoidfinset>')
    assert _position(e,.1,.3,.3) == pytest.approx(expected)


def test_priority_survives_conflicts_and_json_roundtrip(tmp_path):
    data = load_airframe_config(FIXTURE)
    original = copy.deepcopy(data)
    data.update(length=9, mass=9, cg=9, cp=.01, motor_dry_mass=.016)
    data['provenance']['calculated']['mass'] = 8
    path=tmp_path/'airframe.json'
    save_airframe_config(path,data)
    result=load_airframe_config(FIXTURE, path)
    assert result['mass']==original['mass']
    assert result['cg']==original['cg']
    assert result['length']==original['length']
    assert result['cp']==pytest.approx(original['cp'])
    assert result['motor_dry_mass'] is None
    assert 'motor_dry_mass' not in result['provenance']['sources']
    # A CP explicitly provided by the source outranks the geometry calculation.
    result['provenance']['ork']['cp']=.3
    assert resolve_airframe(result)['cp']==.3


def test_incomplete_import_is_saveable_without_fake_motor_mass(tmp_path):
    data=load_airframe_config(FIXTURE)
    path=tmp_path/'partial.json'
    save_airframe_config(path,data)
    assert load_airframe_config(FIXTURE, path)['motor_dry_mass'] is None


def mutate(tmp_path, change):
    root=ET.parse(FIXTURE).getroot()
    change(root)
    path=tmp_path/'changed.ork'
    ET.ElementTree(root).write(path)
    return path


def remove_overrides(root):
    stage=root.find('./rocket/subcomponents/stage')
    for tag in ('overridemass','overridecg','overridesubcomponentsmass','overridesubcomponentscg'):
        stage.remove(stage.find(tag))


def test_component_mass_estimation_excludes_motor_and_uses_weighted_cg(tmp_path):
    path=mutate(tmp_path,remove_overrides)
    data=load_airframe_config(path)
    assert .03 < data['mass'] < .05
    assert .1 < data['cg'] < .3
    assert data['provenance']['sources']['mass']=='calculated'
    assert data['provenance']['sources']['cg']=='calculated'
    # Changing the motor designation / length does not add motor to dry mass.
    root=ET.parse(path)
    root.find('.//motor[@configid="chosen"]/designation').text='Other motor'
    root.write(path)
    assert load_airframe_config(path)['mass']==data['mass']


def test_subtree_mass_override_does_not_shift_unmodified_cg(tmp_path):
    data=load_airframe_config(mutate(tmp_path,remove_overrides))
    def change(root):
        stage=root.find('./rocket/subcomponents/stage')
        stage.remove(stage.find('overridecg'))
        stage.remove(stage.find('overridesubcomponentscg'))
    result=load_airframe_config(mutate(tmp_path,change))
    assert result['mass']==.25
    assert result['cg']==pytest.approx(data['cg'])


def test_manual_cp_is_last_resort_if_calculation_is_unavailable():
    result=resolve_airframe(dict(cp=.2,provenance=dict(ork={},calculated={},sources={'cp':'manual'})))
    assert result['cp']==.2
    assert result['provenance']['sources']['cp']=='manual'


def test_unknown_mass_is_not_silently_omitted(tmp_path):
    def change(root):
        remove_overrides(root)
        body=root.find('.//bodytube/subcomponents')
        ET.SubElement(body,'unknowncomponent')
    data=load_airframe_config(mutate(tmp_path,change))
    assert data['mass'] is None and data['cg'] is None
    assert any('unknowncomponent' in x for x in data['provenance']['warnings'])
    data['mass']=.4
    data['cg']=.2
    assert resolve_airframe(data)['mass']==.4


def test_multistage_design_is_rejected_instead_of_flattened(tmp_path):
    def change(root):
        subs=root.find('rocket/subcomponents')
        subs.append(copy.deepcopy(subs[0]))
    with pytest.raises(AirframeConfigError,match='単段'):
        load_airframe_config(mutate(tmp_path,change))


def test_inactive_stage_is_excluded(tmp_path):
    def change(root):
        subs=root.find('rocket/subcomponents')
        subs.append(copy.deepcopy(subs[0]))
        ET.SubElement(root.find('rocket/motorconfiguration'),'stage',number='1',active='false')
    assert load_airframe_config(mutate(tmp_path,change))['mass']==.25


def test_unknown_fields_in_provenance_are_rejected():
    with pytest.raises(ValueError,match='Unknown fields'):
        resolve_airframe({'provenance':{'ork':{'provenance':{}}}})
