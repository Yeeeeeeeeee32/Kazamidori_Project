import math
import threading
from unittest.mock import patch

import pytest

from core.annealing import anneal
from core.atmosphere import pressure_altitude_agl
from core.optimization import _monte_carlo_r90, optimize_launch_angle, p1_objective_score
from core.simulation import SimulationInputError


def test_weighted_score_and_disqualification():
    r = dict(ok=True, impact_x=3, impact_y=4, hang_time=10)
    assert p1_objective_score(r, '定点滞空', 100, score_a=2, score_b=3) == 220
    assert p1_objective_score(r, '定点滞空', 4, score_a=-2, score_b=0) == -math.inf


def test_annealing_reproducible_bounded_and_handles_nonconvex_objective():
    visited = []
    def objective(e, a):
        visited.append((e, a))
        # Several local maxima in azimuth, dominant maximum at e=78,a=210.
        d = (a-210+180) % 360-180
        return 100 - (e-78)**2 - .015*d*d + 3*math.cos(math.radians(a*6)), None
    first = anneal(objective, budget=200, seed=42)
    assert first == anneal(objective, budget=200, seed=42)
    assert all(60 <= e <= 90 and 0 <= a < 360 for e, a in visited)
    assert abs(first[0][1]-78) < 2
    assert abs(first[0][2]-210) < 20
    assert any(score < first[0][0] for score, *_ in first)


def test_annealing_cancellation_checked_each_step():
    calls = []
    with pytest.raises(RuntimeError, match='cancelled'):
        anneal(lambda *p: (calls.append(p) or 1, None), cancelled=lambda: len(calls) == 3)
    assert len(calls) == 3


def test_annealing_escapes_initial_infeasible_region():
    result = anneal(lambda e, a: ((a if a > 180 else -math.inf), None), seed=0)
    assert result and result[0][0] > 300


def test_final_mc_failure_is_not_replaced_by_earlier_good_result():
    result = dict(ok=True, impact_x=0, impact_y=0, apogee_m=50, hang_time=8)
    with patch('core.optimization.simulate_once', return_value=result), \
         patch('core.optimization._monte_carlo_r90', side_effect=[(0, 1)]*5+[(math.inf, 0)]):
        with pytest.raises(SimulationInputError, match='Final Monte Carlo'):
            optimize_launch_angle('高度', {}, 100, 90, 0, 0, threading.Event(), lambda *_: None)


def test_mc_radius_measures_spread_not_distance_from_launch_pad():
    # Perfect repeatability at a target 400 m away has zero spread.
    p = dict(thrust_data=[[0, 1], [1, 0]], wind_u_prof=[(0, 0), (1000, 0)], wind_v_prof=[(0, 0), (1000, 0)])
    with patch('core.optimization.simulate_once', return_value=dict(ok=True, impact_x=400, impact_y=0)):
        radius, success = _monte_carlo_r90(85, 0, p, 8, 90, 0, 0)
    assert radius == 0
    assert success == 1


def test_pressure_conversion_is_relative_to_station_and_changes_with_pressure():
    assert pressure_altitude_agl(1013.25, 101325) == 0
    assert pressure_altitude_agl(900, 101325) == pytest.approx(988.49, abs=1)
    assert pressure_altitude_agl(900, 100000) < pressure_altitude_agl(900, 101325)
    assert pressure_altitude_agl(1000, 95000) < 0  # below ground; caller excludes it


@pytest.mark.parametrize('p,p0,t', [(0, 101325, 15), (math.nan, 101325, 15), (900, 0, 15), (900, 101325, -273)])
def test_invalid_pressure_rejected(p, p0, t):
    with pytest.raises(ValueError):
        pressure_altitude_agl(p, p0, t)
