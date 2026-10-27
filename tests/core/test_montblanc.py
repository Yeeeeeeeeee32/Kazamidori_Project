import json
import math
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from core.montblanc import landing_metrics, normalize_options
from core.optimization import _monte_carlo_r90, p1_objective_score, optimize_launch_angle


def metrics(vx, vy=None, **options):
    return landing_metrics(list(range(len(vx))), vx, vy or [0]*len(vx), 0,
                           dict(max_speed_ms=3, window_s=2, max_variation_ms=.5, **options))


def test_entire_window_and_ground_speed():
    stable = metrics([9, 2, 2, 2])
    assert stable['montblanc_pass']
    assert stable['landing_horizontal_speed_ms'] == 2
    # Slow at contact does not excuse a fast or changing approach.
    assert not metrics([2, 4, 2, 1])['montblanc_pass']
    assert not metrics([2, 2, 1, 1])['montblanc_pass']
    # A vector rotation can contain an internal minimum in speed.
    rotating = metrics([2, 2, -2, -2])
    assert rotating['landing_speed_variation_ms'] == 2
    assert not rotating['montblanc_pass']


def test_exact_boundary_and_full_post_deployment_window():
    result = landing_metrics([0, 1, 2, 3], [4, 3, 2, 2], [0]*4, 0,
                             dict(max_speed_ms=2.5, window_s=1.5, max_variation_ms=.5))
    assert result['landing_window_max_speed_ms'] == 2.5
    assert result['montblanc_pass']
    short = landing_metrics([0, 1, 2, 3], [2]*4, [0]*4, 2, dict(window_s=2))
    assert not short['landing_window_covered'] and not short['montblanc_pass']


@pytest.mark.parametrize('options', [{'window_s':0}, {'max_speed_ms':float('nan')},
                                    {'max_variation_ms':-1}, {'window_s':True}])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        normalize_options(options)


def test_score_rejects_constraints_and_existing_modes_unchanged():
    r = dict(ok=True, impact_x=0, impact_y=0, apogee_m=100, hang_time=20,
             **metrics([2]*4))
    assert p1_objective_score(r, 'モンブラン', 50) == -2
    assert p1_objective_score(dict(r, montblanc_pass=False), 'モンブラン', 50) == -math.inf
    assert p1_objective_score(dict(r, impact_x=51), 'モンブラン', 50) == -math.inf
    assert p1_objective_score(dict(r, montblanc_pass=False), '高度', 50) == 100
    assert p1_objective_score(dict(r, impact_x=999), '自由', 50) == 20


def test_mc_joint_constraint_counts_failures():
    p = dict(flight_mode='モンブラン', verification_radius=50, thrust_data=[[0,1],[1,0]])
    good = dict(ok=True, impact_x=1, impact_y=0, montblanc_pass=True)
    results = [good, good, dict(good, montblanc_pass=False), dict(ok=False), good]
    with patch('core.optimization.simulate_once', side_effect=results), patch(
        'core.optimization.build_perturbed_wind_prof', return_value=([],[],0,0,0)):
        radius, fraction = _monte_carlo_r90(85, 0, p, 4, 90, 0, 0, threading.Event())
    assert fraction == .5
    assert radius == math.inf


def test_angle_search_excludes_fast_candidate_and_verifies_winner():
    good = dict(ok=True, impact_x=0, impact_y=0, apogee_m=80, hang_time=20,
                **metrics([2]*4))
    bad = dict(good, montblanc_pass=False, landing_window_max_speed_ms=10)
    def candidates(evaluate, **kwargs):
        assert evaluate(80,0)[0] == -math.inf
        score, result = evaluate(85,0)
        assert math.isfinite(score)
        return [(score,85,0,result)]
    with patch('core.annealing.anneal',side_effect=candidates), patch(
        'core.optimization.simulate_once',side_effect=lambda e,a,p: good if e==85 else bad), patch(
        'core.optimization._monte_carlo_r90',return_value=(0,1)) as mc:
        result = optimize_launch_angle('モンブラン',{},50,90,0,0,threading.Event(),lambda *args:None)
    assert result['elev'] == 85
    assert mc.call_count == 2
    assert mc.call_args.args[2]['flight_mode'] == 'モンブラン'


@pytest.mark.physics
def test_real_flight_uses_corrected_velocity_and_insufficient_window():
    from core.simulation import simulate_once
    p = json.loads(Path('tests/fixtures/simulation_params.json').read_text())
    p.update(thrust_data=[[0,0],[.1,12],[.3,7],[1.1,0]], motor_burn_time=1.1,
             flight_mode='モンブラン', para_wind_drift_ratio=.5,
             wind_u_prof=[(0,4),(5000,4)], wind_v_prof=[(0,0),(5000,0)],
             montblanc_options=dict(max_speed_ms=2, window_s=2, max_variation_ms=0))
    r = simulate_once(84,0,p)
    assert r['ok'], r
    assert r['montblanc_pass']
    assert r['landing_horizontal_speed_ms'] == pytest.approx(2)
    assert r['landing_window_max_speed_ms'] == pytest.approx(2)
    assert r['landing_speed_variation_ms'] == pytest.approx(0)
    p['montblanc_options']['window_s'] = 120
    short = simulate_once(84,0,p)
    assert short['ok']
    assert not short['montblanc_pass'] and not short['landing_window_covered']
