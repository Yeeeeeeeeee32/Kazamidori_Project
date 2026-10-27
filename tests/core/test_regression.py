"""Numerical baselines with actual timed deployment (RocketPy 1.13).

These tolerances detect unit/coordinate regressions, not flight certification.
"""
import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from core.optimization import optimize_launch_angle, p1_objective_score
from core.simulation import simulate_once
from core.wind_model import create_wind_profile


@pytest.mark.physics
def test_reference_flight():
    p = json.loads((Path(__file__).parents[1] / 'fixtures/simulation_params.json').read_text())
    p.update(thrust_data=[[0, 0], [.1, 12], [.3, 7], [1.1, 0]], motor_burn_time=1.1)
    r = simulate_once(84, 0, p)
    assert r['ok'], r.get('error')
    assert r['apogee_m'] == pytest.approx(93.4254, abs=.5)
    assert r['parachute_deployed']
    assert r['hang_time'] == pytest.approx(27.5436, abs=.1)
    assert r['impact_y'] == pytest.approx(25.5181, abs=.5)
    assert abs(r['impact_x']) < .2


@pytest.mark.parametrize('mode,score', [('定点滞空', 107), ('高度', 80), ('有翼', 8), ('自由', 12)])
def test_scoring_reference(mode, score):
    r = dict(ok=True, impact_x=103, impact_y=204, hang_time=12, apogee_m=80, bf_abs_time=4)
    assert p1_objective_score(r, mode, 100, 100, 200) == score
    if mode != '自由':
        assert p1_objective_score(r, mode, 4, 100, 200) == float('-inf')


def test_north_wind_flows_south():
    u, v = create_wind_profile([dict(alt_m=3, speed_ms=5, dir_deg=0)])
    assert u[0][1] == pytest.approx(0, abs=1e-12)
    assert v[0][1] < 0
    assert all(a[0] < b[0] for a, b in zip(u, u[1:]))


def test_optimizer_keeps_best_feasible_result():
    def flight(elev, azi, params):
        return dict(ok=True, impact_x=0, impact_y=0, hang_time=elev, apogee_m=elev)
    with patch('core.optimization.simulate_once', side_effect=flight), \
         patch('core.optimization._monte_carlo_r90', return_value=(0, 1)):
        r = optimize_launch_angle('高度', {}, 100, 90, 0, 0, threading.Event(), lambda *_: None)
    assert r['elev'] >= 89
    assert 0 <= r['azi'] < 360
    assert r['score'] == r['elev']
