import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QPushButton, QToolBox, QHBoxLayout
from main_qt import DEFAULT_CONFIG
from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.sim_controller import SimController
from ui_qt.wind_data_dialog import WindDataWorker
from utils.configuration import save_config, read_config, apply_config, save_motor_config, load_motor_config
from ui_qt.workers import OptimizationWorker, SimulationWorker


@pytest.fixture
def ui():
    app = QApplication.instance() or QApplication([])
    state = AppState(DEFAULT_CONFIG)
    with patch('ui_qt.app_window.MapView.request_render'):
        win = AppWindow(state)
        controller = SimController(win, state)
        controller._wind_timer.stop()
        win._wind_refresh_timer.stop()
        yield state, win, controller
        controller._tray_icon.hide()
        win.close()
        controller.deleteLater()
        win.deleteLater()
        app.processEvents()


def test_motor_has_own_page_and_filename_beside_button(ui):
    _, win, _ = ui
    box = next(b for b in win.findChildren(QToolBox) if any(b.itemText(i).endswith('Airframe') for i in range(b.count())))
    motor_page = next(box.widget(i) for i in range(box.count()) if box.itemText(i).endswith('Motor Specifications'))
    button = motor_page.findChild(QPushButton, 'btn_load_thrust')
    assert button is not None
    group = button.parentWidget()
    form = group.layout()
    row, _ = form.getWidgetPosition(win.motor_cg_input)
    assert row >= 0
    assert win.motor_label.parentWidget() is group
    assert any(isinstance(form.itemAt(i), QHBoxLayout) and form.itemAt(i).indexOf(button) == 0
               and form.itemAt(i).indexOf(win.motor_label) == 1 for i in range(form.count()))
    fields = [win._manual_dialog.airframe_sources_table.item(i,0).text()
              for i in range(win._manual_dialog.airframe_sources_table.rowCount())]
    assert 'motor_dry_mass' not in fields


def test_external_compass_tracks_camera_and_does_not_overlap_trajectory(ui):
    _, win, _ = ui
    win.update_profile_plot()
    assert win.orientation_ax.get_position().x0 >= win.profile_ax.get_position().x1
    win._on_azim_changed(15)
    assert win.orientation_ax.azim == win.profile_ax.azim == 15
    assert {t.get_text() for t in win.orientation_ax.texts} == {'E','N','Up'}


def test_progress_is_monotonic_and_messages_agree(ui):
    state, win, controller = ui
    state.is_calculating = True
    controller._on_worker_status('Searching launch angle')
    controller._on_progress(60)
    controller._on_progress_updated(80,100,'Monte Carlo (5/10)')
    assert state.progress_percentage == win._progress.value() == 80
    assert win._phase_label.text() == win._status_label.text() == 'Monte Carlo (5/10)'
    assert 'Monte Carlo (5/10)' in win._progress.format()
    controller._on_progress(25)
    assert win._progress.value() == 80


def test_practice_wind_edits_preserve_text_and_random_samples(ui):
    state, win, controller = ui
    d = win._adv_dialog
    d.wind_inputs[0][1].setText('10')
    d.wind_inputs[0][2].setText('90')
    assert d.wind_inputs[0][1].text() == '10'
    assert state.wind_profile_data[0]['speed_ms'] == 10
    baseline = [dict(n) for n in state.wind_profile_data]
    with patch('ui_qt.sim_controller.random.gauss', return_value=.5):
        controller._on_wind_tick()
    sample = state.wind_history_for_alt(1.5)[-1]
    assert sample['speed_ms'] == 10.5 and sample['dir_deg'] == 90.5
    assert state.wind_profile_data == baseline


def test_config_and_motor_settings_roundtrip(ui, tmp_path):
    state, win, _ = ui
    state.motor_dry_mass = .018
    state.parachute_wind_drift_pct = 45
    path = tmp_path/'config.json'
    save_config(state, path)
    state.motor_dry_mass = .02
    state.parachute_wind_drift_pct = 100
    apply_config(state, read_config(path))
    assert state.motor_dry_mass == .018
    assert state.parachute_wind_drift_pct == win.para_drift_input.value() == 45
    motor = tmp_path/'Motor.json'
    save_motor_config(state, motor)
    state.motor_dry_mass = .03
    load_motor_config(state, motor)
    assert state.motor_dry_mass == .018


def test_predownload_does_not_require_current_simulation_time():
    worker = WindDataWorker('future.csv', {'download_only':True, 'latitude':35.,'longitude':139.,
                           'start':'2026-10-22T00:00:00Z','end':'2026-10-23T00:00:00Z'}, fetch=True)
    ready, errors = [], []
    worker.ready.connect(ready.append)
    worker.error.connect(errors.append)
    with patch('ui_qt.wind_data_dialog.fetch_wind_csv') as fetch, \
         patch('ui_qt.wind_data_dialog.read_wind_snapshot', side_effect=AssertionError('Should only download')):
        worker.run()
    fetch.assert_called_once()
    assert ready == ['future.csv'] and not errors


def test_optimization_worker_progress_covers_entire_pipeline(ui):
    worker = OptimizationWorker({'is_free_mode':True})
    percentages, errors = [], []
    worker.progress.connect(percentages.append)
    worker.sig_progress.connect(lambda value,total,msg: percentages.append(value*100/total))
    worker.error.connect(errors.append)
    def optimize(**kwargs):
        for fraction in (0, .5, 1):
            kwargs['progress_cb']('Searching', fraction)
        return {'elev':84, 'azi':0, 'score':1}
    def mc(dummy, *args, **kwargs):
        for value in (25, 50, 90):
            dummy.progress.emit(value)
            dummy.sig_progress.emit(value,100,'MC')
        return []
    with patch('core.optimization.optimize_launch_angle', side_effect=optimize), \
         patch('core.simulation.simulate_once', return_value={'ok':True}), \
         patch.object(SimulationWorker, '_build_wind_profiles', return_value=([(0,0),(1000,0)],[(0,0),(1000,0)])), \
         patch.object(SimulationWorker, '_build_sim_params', return_value={}), \
         patch.object(SimulationWorker, '_package_nominal', return_value={}), \
         patch.object(SimulationWorker, '_run_mc_loop', new=mc), \
         patch.object(SimulationWorker, '_package_result', return_value={}):
        worker.run()
    assert not errors
    assert percentages == sorted(percentages)
    assert percentages[-1] == 100 and percentages[:-1].count(100) == 0
