"""Two-file designs: authoritative ORK plus editable missing-parameter JSON."""
import gzip
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from utils.data_loader import load_airframe_config, save_airframe_config, AirframeConfigError

FIXTURE = Path(__file__).parents[1] / 'fixtures' / 'single_stage.ork'


def pair(tmp_path, **extras):
    ork = tmp_path / '機体.ork'
    shutil.copyfile(FIXTURE, ork)
    supplement = tmp_path / 'Airframe_Test.json'
    supplement.write_text(json.dumps(dict(ork_file=ork.name, motor_dry_mass=.016, **extras)), encoding='utf-8')
    return ork, supplement


def test_explicit_pair_loads_both_and_source_wins_conflicts(tmp_path):
    ork, supplement = pair(tmp_path, length=99, mass=99, cp=.01)
    from_json = load_airframe_config(ork, supplement)
    assert load_airframe_config(ork)["motor_dry_mass"] is None
    assert from_json['length'] == pytest.approx(.4)
    assert from_json['mass'] == .25
    assert from_json['cp'] > .1
    assert from_json['motor_dry_mass'] == .016
    assert from_json['provenance']['sources']['motor_dry_mass'] == 'json'
    assert from_json['provenance']['supplement_path'] == str(supplement)
    assert any('JSON' in w for w in from_json['provenance']['warnings'])


def test_save_removes_duplicates_and_manual_pair_survives_directory_move(tmp_path):
    folder = tmp_path / 'original'
    folder.mkdir()
    ork, supplement = pair(folder, length=9, cg=9)
    data = load_airframe_config(ork, supplement)
    data['motor_dry_mass'] = .018
    save_airframe_config(supplement, data)
    assert json.loads(supplement.read_text(encoding='utf-8')) == {}
    moved = tmp_path / 'moved'
    shutil.copytree(folder, moved)
    restored = load_airframe_config(moved / ork.name, moved / supplement.name)
    assert restored['motor_dry_mass'] is None
    assert restored['provenance']['source_path'] == str(moved / ork.name)


def test_json_reload_uses_updated_ork_not_saved_snapshot(tmp_path):
    ork, supplement = pair(tmp_path)
    data = load_airframe_config(ork, supplement)
    save_airframe_config(supplement, data)
    tree = ET.parse(ork)
    tree.find('.//stage/overridemass').text = '.3'
    tree.write(ork)
    assert load_airframe_config(ork, supplement)['mass'] == .3


def test_json_supplies_mass_and_cg_when_calculation_unavailable(tmp_path):
    ork, supplement = pair(tmp_path, mass=.4, cg=.2)
    tree = ET.parse(ork)
    stage = tree.find('./rocket/subcomponents/stage')
    for tag in ('overridemass', 'overridecg', 'overridesubcomponentsmass', 'overridesubcomponentscg'):
        stage.remove(stage.find(tag))
    ET.SubElement(tree.find('.//bodytube/subcomponents'), 'unknowncomponent')
    tree.write(ork)
    data = load_airframe_config(ork, supplement)
    assert data['mass'] == .4 and data['cg'] == .2
    assert data['provenance']['sources']['mass'] == 'json'
    save_airframe_config(supplement, data)
    saved = json.loads(supplement.read_text(encoding='utf-8'))
    assert saved['mass'] == .4 and saved['cg'] == .2
    assert 'length' not in saved


@pytest.mark.parametrize('change', [dict(motor_dry_mass=-1), dict(motor_dry_mas=.1)])
def test_invalid_pair_rejected(tmp_path, change):
    ork, supplement = pair(tmp_path)
    data = json.loads(supplement.read_text())
    data.update(change)
    supplement.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(AirframeConfigError):
        load_airframe_config(ork, supplement)


def test_only_selected_json_is_used_even_when_multiple_candidates_exist(tmp_path):
    ork, supplement = pair(tmp_path)
    second = tmp_path / 'arbitrary.json'
    second.write_text(json.dumps(dict(motor_dry_mass=.02)), encoding='utf-8')
    assert load_airframe_config(ork)['motor_dry_mass'] is None
    assert load_airframe_config(ork, supplement)['motor_dry_mass'] == .016
    assert load_airframe_config(ork, second)['motor_dry_mass'] == .02


def test_legacy_reference_does_not_override_explicit_selection(tmp_path):
    ork, supplement = pair(tmp_path)
    supplement.write_text(json.dumps(dict(ork_file='missing.ork', motor_dry_mass=.016)))
    assert load_airframe_config(ork, supplement)['motor_dry_mass'] == .016
    with pytest.raises(AirframeConfigError):
        load_airframe_config(supplement)


def test_gzip_with_differently_named_json_in_another_folder(tmp_path):
    ork = tmp_path / 'design.ork.gz'
    ork.write_bytes(gzip.compress(FIXTURE.read_bytes()))
    subfolder = tmp_path / 'other'
    subfolder.mkdir()
    supplement = subfolder / 'extras.json'
    supplement.write_text(json.dumps(dict(motor_dry_mass=.016)))
    assert load_airframe_config(ork, supplement)['motor_dry_mass'] == .016


def test_missing_source_on_save_does_not_overwrite_existing_supplement(tmp_path):
    ork, supplement = pair(tmp_path)
    data = load_airframe_config(ork, supplement)
    before = supplement.read_bytes()
    ork.unlink()
    with pytest.raises(AirframeConfigError, match='missing'):
        save_airframe_config(supplement, data)
    assert supplement.read_bytes() == before
