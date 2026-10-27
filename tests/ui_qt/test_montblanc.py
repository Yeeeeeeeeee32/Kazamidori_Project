import os
import json
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PySide6.QtWidgets import QApplication
from main_qt import DEFAULT_CONFIG
from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.sim_controller import SimController
from ui_qt.workers import SimulationWorker
from utils.configuration import apply_config, state_config
from utils.session_manager import state_to_dict, dict_to_state


@pytest.fixture
def ui():
    app = QApplication.instance() or QApplication([])
    state = AppState(DEFAULT_CONFIG)
    with patch('ui_qt.app_window.MapView.request_render'):
        window = AppWindow(state)
        controller = SimController(window,state)
        controller._wind_timer.stop()
        window._wind_refresh_timer.stop()
        yield state,window,controller
        controller._tray_icon.hide()
        window.close()
        controller.deleteLater()
        window.deleteLater()
        app.processEvents()


def test_mode_controls_invalidate_and_persist(ui):
    state, window, controller = ui
    window.mode_combo.setCurrentText('モンブラン')
    assert state.flight_mode == 'モンブラン'
    assert not window.montblanc_group.isHidden()
    revision = controller._input_revision
    window.montblanc_inputs['max_speed_ms'].setValue(1.25)
    assert state.montblanc_options['max_speed_ms'] == 1.25
    assert controller._input_revision > revision
    state.montblanc_options = dict(window_s=5)
    assert window.montblanc_inputs['window_s'].value() == 5
    restored = AppState(DEFAULT_CONFIG)
    apply_config(restored, state_config(state))
    assert restored.montblanc_options == state.montblanc_options
    dict_to_state(restored, state_to_dict(state))
    assert restored.montblanc_options == state.montblanc_options
    window.mode_combo.setCurrentText('高度')
    assert window.montblanc_group.isHidden()


def test_joint_stats_and_final_go_do_not_ignore_speed(ui):
    state, window, controller = ui
    window.mode_combo.setCurrentText('モンブラン')
    points = [dict(x=i,y=0,score=-2 if i==0 else -float('inf'),apogee=80,
                   hang_time=20,montblanc_pass=i==0) for i in range(4)]
    stats = SimulationWorker._compute_stats(points,90)
    assert stats['mc_montblanc_joint_count'] == 1
    controller._nominal_payload = dict(montblanc_pass=True)
    payload = dict(scatter_points=[(i,0) for i in range(4)], n_attempted=5,
                   n_runs=4, mc_montblanc_joint_count=1, montblanc_pass=True,
                   apogee_m=80,hang_time=20,impact_x=0,impact_y=0, landing_prob=90)
    with patch.object(window,'update_map_plot'), patch.object(window,'update_profile_plot'):
        controller._on_mc_done(payload)
    assert 'NO-GO' in window._go_nogo_label.text()
    assert not state.phase2_active
    payload['n_attempted'] = 4
    payload['mc_montblanc_joint_count'] = 4
    with patch.object(window,'update_map_plot'), patch.object(window,'update_profile_plot'):
        controller._on_mc_done(payload)
    assert window._go_nogo_label.text().startswith('GO')
    assert not state.phase2_active


def test_result_import_export_restores_execution_limits(ui, tmp_path):
    from utils.result_io import load_archive, make_archive, save_archive
    state, window, controller = ui
    source = json.loads(Path('tests/fixtures/result_archive.json').read_text(encoding='utf-8'))
    limits = dict(max_speed_ms=1.25, window_s=5, max_variation_ms=.2)
    source['context'].update(flight_mode='モンブラン', montblanc_options=limits)
    source['nominal'].update(montblanc_pass=False, landing_horizontal_speed_ms=2.,
                             landing_window_max_speed_ms=2.1, landing_speed_variation_ms=.1,
                             landing_window_covered=True)
    source['result'].update(source['nominal'])
    source['result']['input_snapshot'] = dict(source['context'])
    archive = make_archive(source['result'],source['nominal'],source['context'])
    path = tmp_path / 'montblanc.json'
    save_archive(path, archive)
    with patch('ui_qt.sim_controller.QFileDialog.getOpenFileName',return_value=(str(path),'')):
        controller._on_import_results()
    assert state.flight_mode == 'モンブラン'
    assert state.montblanc_options == limits
    assert window.montblanc_inputs['window_s'].value() == 5
    assert '2.00' in window.montblanc_result_label.text()
    window.montblanc_inputs['max_speed_ms'].setValue(7)
    output = tmp_path / 'export.json'
    with patch('ui_qt.app_window.QFileDialog.getSaveFileName',return_value=(str(output),'')):
        window._on_export_results('json')
    restored = load_archive(output)
    assert restored['context']['montblanc_options'] == limits
