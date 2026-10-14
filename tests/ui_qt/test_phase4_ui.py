import os
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtWidgets import QApplication
from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.sim_controller import SimController
from ui_qt.workers import SimulationWorker
from ui_qt.wind_data_dialog import WindDataDialog, WindDataWorker
from utils.session_manager import save_session, load_session


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    state = AppState({})
    with patch('ui_qt.app_window.MapView.request_render'):
        win = AppWindow(state)
        win._wind_refresh_timer.stop()
        yield win
        win.close()
        win.deleteLater()
        app.processEvents()


def test_score_settings_and_manual_declination_round_trip(window, tmp_path):
    d = window._adv_dialog
    d.score_a_input.setValue(2.5)
    d.score_b_input.setValue(0)
    window.declination_mode_input.setCurrentIndex(1)
    window.declination_input.setValue(-8.2)
    window.azimuth_reference_input.setCurrentIndex(1)
    state = window.state
    assert state.phase4_options['score_a'] == 2.5
    assert state.effective_declination == -8.2
    path = tmp_path/'session.json'
    save_session(state, str(path))
    d.score_a_input.setValue(10)
    window.declination_input.setValue(0)
    load_session(state, str(path))
    assert d.score_a_input.value() == 2.5
    assert d.score_b_input.value() == 0
    assert window.declination_input.value() == -8.2
    assert state.effective_declination == -8.2


def test_history_marks_equal_receptions_and_holds_silent_data(window):
    state = window.state
    now = time.monotonic()
    with patch('ui_qt.app_state.time.monotonic', side_effect=[now-20, now-10]):
        state.append_wind_reading(4, 90)
        state.append_wind_reading(4, 90)
    window.update_wind_plot()
    ax = window.wind_profile_ax
    assert len(ax.collections[0].get_offsets()) == 2
    assert ax.lines[0].get_linestyle() == '--'
    assert ax.lines[0].get_drawstyle() == 'steps-post'
    assert ax.lines[0].get_xdata()[-1] == 0
    with patch('time.monotonic', return_value=now+100):
        window.update_wind_plot()
    assert list(ax.lines[0].get_ydata()) == [4, 4]
    assert len(ax.collections) == 0  # no invented reception marker at now


def test_compass_uses_live_reading_instead_of_old_simulation(window):
    window.state.simulation_result = {'wind_nodes': [dict(alt_m=3, speed_ms=1, dir_deg=0)]}
    window.state.append_wind_reading(8, 90)
    with patch('scipy.stats.gaussian_kde', side_effect=AssertionError('KDE in redraw')):
        window.update_wind_plot()
    assert '8.0 m/s' in window.wind_ax.get_legend_handles_labels()[1][0]


def test_csv_pressure_conversion_is_used_in_worker(tmp_path):
    path = tmp_path/'wind.csv'
    path.write_text('time_utc,pressure_hpa,speed_ms,dir_deg\n2026-10-14T03:00:00Z,900,10,0\n')
    params = dict(wind_csv=str(path), wind_time_utc='2026-10-14T03:00:00Z',
                  wind_profile_data=[dict(alt_m=3, speed_ms=2, dir_deg=0)], env_pressure=101325.)
    u1, v1 = SimulationWorker._build_wind_profiles(params)
    params['env_pressure'] = 100000
    u2, v2 = SimulationWorker._build_wind_profiles(params)
    assert next(v for z,v in v2 if z == 600) < next(v for z,v in v1 if z == 600)
    assert v1[0][1] == -2  # ground measurement retained


def test_weather_worker_reports_load_failure():
    worker = WindDataWorker('missing-wind-file.csv', dict(target='2026-10-14T03:00:00Z',
        pressure=101325, temperature=15, latitude=35, longitude=139))
    errors, successes = [], []
    worker.error.connect(errors.append)
    worker.ready.connect(successes.append)
    worker.run()
    assert errors and not successes


def test_wind_data_dialog_uses_utc_and_manual_reset(window):
    window.state.phase4_options = dict(wind_csv='saved.csv', wind_time_utc='2026-10-14T03:00:00Z')
    dialog = WindDataDialog(window.state, window)
    assert dialog._iso(dialog.target) == '2026-10-14T03:00:00Z'
    dialog._clear()
    assert not window.state.phase4_options['wind_csv']
    dialog.close()
