import os
import time
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication
from main_qt import DEFAULT_CONFIG
from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.sim_controller import SimController
from ui_qt.workers import SimulationWorker


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


def test_physical_edits_invalidate_go_and_stale_completion_is_discarded(ui):
    state,window,controller = ui
    state.set_wind_lock(0,-4,.2)
    state.phase2_active = True
    old_revision = controller._input_revision
    state.rocket_dry_mass = .8
    assert not state.phase2_active
    state.is_calculating = True
    controller._on_mc_done({'input_snapshot':{'input_revision':old_revision},'cancelled':False})
    assert not state.has_sim_result and not state.is_calculating
    assert state.simulation_result is None
    assert 'stale result discarded' in window._status_label.text()


def test_forecast_upper_nodes_survive_practice_tick(ui):
    state,_,controller = ui
    state.phase4_options = dict(wind_csv='forecast.csv',wind_time_utc='2026-10-24T00:00:00Z')
    controller._forecast_nodes = {600.:dict(alt_m=600.,speed_ms=20.,dir_deg=90.)}
    with patch('ui_qt.sim_controller.random.gauss',return_value=0):
        controller._on_wind_tick()
    assert state.get_wind_zoh(600)['speed_ms'] == 20
    assert state.get_wind_zoh(1.5) is not None


def test_weather_source_switch_does_not_average_old_forecast(ui):
    state,_,controller = ui
    state.phase4_options = dict(wind_csv='one.csv',wind_time_utc='2026-10-24T00:00:00Z')
    controller._forecast_nodes = {600.:dict(alt_m=600.,speed_ms=20.,dir_deg=90.)}
    state.append_wind_nodes(list(controller._forecast_nodes.values()))
    state.phase4_options = dict(wind_csv='two.csv')
    assert not controller._forecast_nodes
    assert not state.wind_history_for_alt(600)
    assert state.get_wind_zoh(600) is None
    assert 'Pending' in state.gpv_last_fetch_time


def test_statistics_are_complete_and_score_denominator_is_visible(ui):
    state,window,controller = ui
    state.flight_mode = '高度'
    points = [dict(x=x,y=y,apogee=height,hang_time=10,bf_abs_time=2,score=score)
              for x,y,height,score in [(0,0,100,100),(1,0,120,100),(0,1,80,100),(1,1,100,float('-inf'))]]
    stats = SimulationWorker._compute_stats(points,90)
    stats['n_attempted'] = 5
    payload = SimulationWorker._package_mc(points,stats,90)
    controller.populate_results({},payload)
    assert window.lbl_res_mc_avg_apo.text() == '100.0 m'
    assert window.lbl_res_mc_min_apo.text() == '80.0 m'
    assert '1 disqualified' in window.mc_quality_label.text()
    assert '1 failed' in window.mc_quality_label.text()
    assert 'Valid' in window.res_labels['mc_min_score'][0].text()


def test_mc_seed_repeats_trial_inputs_and_worker_copies_nested_data(ui):
    _,_,_ = ui
    p = dict(mc_runs=4,random_seed=123,wind_unc=.2,thrust_unc=.05,flight_mode='自由')
    sim = dict(elev=84,azi=0,thrust_data=[[0,0],[.1,12],[1,0]],
               wind_u_prof=[(1.5,2),(600,5)],wind_v_prof=[(1.5,0),(600,1)])
    p['wind_profile_data'] = [dict(alt_m=1.5,speed_ms=4,dir_deg=0)]
    worker = SimulationWorker(p)
    p['wind_profile_data'][0]['speed_ms'] = 999
    assert worker._params['wind_profile_data'][0]['speed_ms'] == 4
    recorded = []
    def flight(elev,azi,params,**kw):
        recorded.append(deepcopy(params))
        return dict(ok=True,impact_x=1,impact_y=2,apogee_m=100,hang_time=10,bf_abs_time=2)
    with patch('ui_qt.workers.simulate_once',side_effect=flight):
        worker._run_mc_loop(sim,p)
        worker._run_mc_loop(sim,p)
    assert recorded[:4] == recorded[4:]


def test_close_waits_for_live_thread_without_blocking_gui(ui):
    _,window,controller = ui
    class SlowWorker(QThread):
        stopped = False
        def stop(self):
            self.stopped = True
            self.requestInterruption()
        def run(self):
            while not self.isInterruptionRequested():
                self.msleep(5)
    thread = SlowWorker()
    controller._worker = thread
    thread.start()
    try:
        before = time.monotonic()
        assert window.close() is False  # The real close event is deferred.
        assert time.monotonic()-before < .2
        assert thread.stopped
    finally:
        thread.requestInterruption()
        assert thread.wait(1000)
        controller._worker = None
    assert controller.request_shutdown() is True


def test_motor_curve_and_target_change_also_invalidate_go(ui):
    state,window,_ = ui
    state.phase2_active = True
    window.sig_thrust_curve_changed.emit()
    assert not state.phase2_active
    state.phase2_active = True
    state.target_x = 20
    assert not state.phase2_active
    state.target_y = -10
    state.target_radius = 30
    circles = [p for p in window.map_view.ax.patches if hasattr(p,'get_radius') and p.get_radius() == 30]
    assert circles and circles[0].center == (20.,-10.)


def test_export_keeps_execution_context_after_edit(ui,tmp_path):
    state,window,controller = ui
    archive = json.loads((Path(__file__).parents[1]/'fixtures/result_archive.json').read_text())
    result = archive['result']
    snapshot = dict(flight_mode='高度',launch_lat=35.,launch_lon=139.,target_radius=30.,target_x=20.,target_y=-10.)
    result['input_snapshot'] = dict(snapshot)
    state.simulation_result = result
    state.launch_lat = 36
    state.target_radius = 100
    destination = tmp_path/'result.json'
    with patch('ui_qt.app_window.QFileDialog.getSaveFileName',return_value=(str(destination),'')):
        window._on_export_results('json')
    saved = json.loads(destination.read_text(encoding='utf-8'))
    assert saved['context'] == snapshot
    assert saved['result']['input_snapshot'] == snapshot
    with patch('ui_qt.sim_controller.QFileDialog.getOpenFileName',return_value=(str(destination),'')):
        controller._on_import_results()
    assert state.flight_mode == '高度'
    assert state.target_radius == 30
    assert (state.target_x,state.target_y) == (20.,-10.)
