"""Offline audit probes. Does not modify production code or call external services.

Run from the repository root:
    python -B -X utf8 archive/kazamidori_audit_20261024_probe.py
"""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import warnings
from importlib.metadata import version
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import binomtest
from core.simulation import simulate_once
from core.optimization import _monte_carlo_r90, _perturb_wind_profile as search_wind
from core.monte_carlo import _perturb_wind_profile as final_wind
from ui_qt.workers import SimulationWorker


def run():
    evidence = {'versions': {name:version(name) for name in ('rocketpy','PySide6','numpy','scipy')}}
    fixture = json.loads((ROOT/'tests/fixtures/simulation_params.json').read_text())
    fixture.update(thrust_data=[[0, 0], [.1, 12], [.3, 7], [1.1, 0]], motor_burn_time=1.1)
    from core.simulation import Flight as original_flight
    flights = []
    def capture_flight(*args, **kwargs):
        flight = original_flight(*args, **kwargs)
        flights.append({'end_s':float(flight.t_final),
            'parachute_events_s':[float(t) for t, _ in flight.parachute_events],
            'pressure_round_trip_m': [float(flight.env.barometric_height(flight.env.pressure(h)))
                                      for h in (0,50,93)]})
        return flight
    with patch('core.simulation.Flight',side_effect=capture_flight):
        slow = simulate_once(84, 0, dict(fixture, para_area=10, para_cd=1.5))
    evidence['parachute_not_deployed'] = {
        'ok': slow['ok'], 'time_s': float(slow['t_vals'][-1]),
        'height_m': float(slow['z_vals'][-1]),
        'reported_open_time_s': slow['para_open_time'], 'actual_flights':flights,
        'reported_impact_m': [slow['impact_x'], slow['impact_y']],
    }
    # Counterfactual diagnostic only: use true state altitude for the trigger,
    # without changing production code. This exposes the separate timeout gate.
    def true_height_trigger(alt):
        return lambda pressure,height,y: y[5] < 0 and y[2] <= alt
    with patch('core.simulation.make_backfire_trigger',side_effect=true_height_trigger):
        capped = simulate_once(84,0,dict(fixture,para_area=10,para_cd=1.5))
    evidence['timeout_with_diagnostic_true_height_trigger'] = {
        'ok':capped['ok'], 'end_s':float(capped['t_vals'][-1]),
        'height_at_end_m':float(capped['z_vals'][-1]),
        'note':'Trigger substituted only for this diagnostic; production code unchanged'}
    delayed = simulate_once(84, 0, dict(fixture, backfire_delay=100))
    evidence['ejection_delay'] = {
        'requested_ejection_time_s': 101.1,
        'reported_ejection_time_s': delayed['bf_abs_time'],
        'reported_open_time_s': delayed['para_open_time'],
        'flight_end_s': delayed['hang_time'],
    }
    outcomes = [dict(ok=True, impact_x=0, impact_y=0)] + [
        dict(ok=True, impact_x=i, impact_y=0) for i in range(1, 16)
    ] + [dict(ok=False, error='injected failure')]
    with patch('core.optimization.simulate_once', side_effect=outcomes):
        radius, success = _monte_carlo_r90(84, 0, fixture, 16, 90, .2, .05, threading.Event())
    evidence['search_acceptance'] = {
        'returned_radius_m': radius, 'returned_success_rate': success,
        'target_radius_m': 14, 'acceptance_conditions_satisfied': radius <= 14 and success >= .9,
        'actual_fraction_within_target': 14/16,
    }
    evidence['wind_models_at_zero_mean'] = {}
    for name, fn in [('search', search_wind), ('final', final_wind)]:
        u, v, _ = fn([(1.5, 0.), (10., 0.)], [(1.5, 0.), (10., 0.)], random.Random(1), .2)
        evidence['wind_models_at_zero_mean'][name] = [[z, a, b] for (z, a), (_, b) in zip(u, v)]
    points = [dict(x=x, y=y, apogee=100, hang_time=10, bf_abs_time=2, score=score)
              for x, y, score in [(0,0,100), (1,0,100), (0,1,100), (1,1,-math.inf)]]
    stats = SimulationWorker._compute_stats(points, 90)
    evidence['conditional_scores'] = {'trial_scores': [100,100,100,'disqualified'],
        'reported_mean': stats['mc_avg_score'], 'reported_minimum': stats['mc_min_score']}
    evidence['binomial_confidence'] = {
        '45_of_50_two_sided_95_exact_lower': binomtest(45,50).proportion_ci().low,
        '50_of_50_one_sided_95_exact_lower': binomtest(50,50,alternative='greater').proportion_ci().low,
    }
    # Positive nodes can produce a reversed wind between observations.
    altitudes = [1.5,10.,150.,300.,600.,5000.]
    values = [1.,10.,1.,1.,1.,1.]
    z = np.linspace(1.5,600,10000)
    interpolated = CubicSpline(altitudes, values, bc_type='natural')(z)
    evidence['spline_overshoot'] = {'node_values': values,
        'minimum_between_nodes': float(interpolated.min()),
        'maximum_between_nodes': float(interpolated.max())}

    from PySide6.QtWidgets import QApplication
    from main_qt import DEFAULT_CONFIG
    from ui_qt.app_state import AppState
    from ui_qt.app_window import AppWindow
    from ui_qt.sim_controller import SimController
    from utils.result_io import load_archive
    app = QApplication.instance() or QApplication([])
    state = AppState(DEFAULT_CONFIG)
    with patch('ui_qt.app_window.MapView.request_render'):
        window = AppWindow(state)
        controller = SimController(window, state)
        controller._wind_timer.stop()
        window._wind_refresh_timer.stop()
        state.flight_mode = '高度'
        state.target_radius = 100
        state.set_wind_lock(0,-4,.2)
        state.phase2_active = True
        state.rocket_dry_mass = .8
        state.parachute_area = .9
        evidence['edit_keeps_monitor_active'] = state.phase2_active
        controller.populate_results({'apogee_m':100}, stats)
        evidence['results_panel'] = {'worker_mc_avg_alt':stats['mc_avg_alt'],
                                    'displayed_mc_avg_apogee':window.lbl_res_mc_avg_apo.text()}
        # A currently running calculation uses old settings, but completion
        # judges its points using the now-edited maximum radius.
        sample = load_archive(ROOT/'tests/fixtures/result_archive.json')['result']
        sample.update(x_vals=sample['trajectory_x'],y_vals=sample['trajectory_y'],
                      z_vals=sample['trajectory_z'],t_vals=[0.,1.,2.,3.])
        sample.update(scatter_points=[(50,0),(51,0),(50,1),(51,1)], n_attempted=4,
                      nominal_surf_spd=4, nominal_surf_dir=0)
        state.target_radius = 100
        controller._on_mc_done(sample)
        controller._wind_timer.stop()
        go_at_100 = state.phase2_active
        state.target_radius = 10
        still_go_after_edit = state.phase2_active
        # Repeat completion with a different current input; same points.
        controller._on_mc_done(sample)
        controller._wind_timer.stop()
        evidence['mutable_completion_context'] = {
            'go_at_current_radius_100':go_at_100,
            'monitor_still_active_after_edit_to_10':still_go_after_edit,
            'go_after_reprocessing_same_payload_at_radius_10':state.phase2_active,
        }
        # Forecast node is appended as a new sample on nominal completion;
        # the mock timer then returns to the manually configured upper wind.
        state.wind_profile_data = [dict(alt_m=1.5,speed_ms=4,dir_deg=0),
                                   dict(alt_m=600,speed_ms=2,dir_deg=0)]
        state.append_wind_nodes([dict(alt_m=600,speed_ms=20,dir_deg=90)])
        before = state.get_wind_zoh(600)['speed_ms']
        with patch('ui_qt.sim_controller.random.gauss',return_value=0):
            controller._on_wind_tick()
        evidence['forecast_overwritten_by_mock_tick'] = {
            'before_ms':before, 'after_ms':state.get_wind_zoh(600)['speed_ms']}
        from utils.session_manager import dict_to_state
        previous_cd = state.power_on_cd
        try:
            dict_to_state(state,{'aerodynamics':{'power_on_cd':.9},'motor':{'isp':'invalid'}})
        except (TypeError,ValueError) as exc:
            evidence['session_partial_mutation'] = {'before_cd':previous_cd,
                'after_failed_load_cd':state.power_on_cd,'error':str(exc)}
        controller._tray_icon.hide()
        window.close()
        controller.deleteLater()
        window.deleteLater()
        app.processEvents()
    return evidence


if __name__ == '__main__':
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = run()
    output = ROOT/'assets/verification/kazamidori_audit_20261024.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
