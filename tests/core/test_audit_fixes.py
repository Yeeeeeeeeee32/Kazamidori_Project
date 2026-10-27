"""Independent event, termination and probability checks for the October audit."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from core import simulation
from core.optimization import _monte_carlo_r90, _perturb_wind_profile
from core.monte_carlo import _perturb_wind_profile as canonical_wind
from utils.session_manager import dict_to_state, save_session, SessionError


@pytest.fixture
def params():
    p = json.loads((Path(__file__).parents[1]/'fixtures/simulation_params.json').read_text())
    p.update(thrust_data=[[0,0],[.1,12],[.3,7],[1.1,0]],motor_burn_time=1.1)
    return p


def test_pressure_height_round_trip(params):
    env = simulation.build_environment(params)
    for height in (0,1.5,50,93,1000,5000):
        assert env.barometric_height(env.pressure(height)) == pytest.approx(height,abs=.002)
    # Linear interpolation cannot create a reversed wind between positive nodes.
    params['wind_u_prof'] = [(1.5,1),(10,10),(150,1),(300,1),(600,1),(5000,1)]
    env = simulation.build_environment(params)
    assert 1 <= env.wind_velocity_x(50) <= 10


@pytest.mark.physics
def test_real_deployment_event_and_canopy_response(params):
    actual = simulation.Flight
    flights = []
    def capture(**kwargs):
        flight = actual(**kwargs)
        flights.append(flight)
        return flight
    with patch('core.simulation.Flight',side_effect=capture):
        result = simulation.simulate_once(84,0,params)
    assert result['ok'], result
    assert result['termination_reason'] == 'ground_impact'
    assert result['parachute_deployed']
    events = flights[-1].parachute_events
    assert len(events) == 1
    assert events[0][0] == pytest.approx(5.1,abs=1/105)
    assert result['para_open_time'] == pytest.approx(events[0][0]+params['para_lag'])
    assert result['bf_abs_time'] == events[0][0]
    assert abs(result['z_vals'][-1]) < .05
    assert result['hang_time'] > 20  # Previously ballistic at ~9.2 s.
    small = simulation.simulate_once(84,0,dict(params,para_area=params['para_area']/2))
    assert small['ok'], small
    assert small['hang_time'] < result['hang_time']


@pytest.mark.physics
def test_late_ejection_and_timeout_are_not_successful_landings(params):
    late = simulation.simulate_once(84,0,dict(params,backfire_delay=100))
    assert not late['ok'] and late['error_code'] == 'parachute_not_deployed'
    assert 'para_open_time' not in late
    slow = simulation.simulate_once(84,0,dict(params,para_area=10,para_cd=1.5))
    assert not slow['ok'] and slow['error_code'] == 'time_limit'
    assert 'impact_x' not in slow


def test_mc_gate_counts_failures_and_target_directly(params):
    nominal = dict(ok=True,impact_x=100,impact_y=0)
    outcomes = [nominal] + [dict(ok=True,impact_x=100+i,impact_y=0) for i in range(1,16)] + [dict(ok=False)]
    p = dict(params,target_x=100,target_y=0,verification_radius=14)
    with patch('core.optimization.simulate_once',side_effect=outcomes):
        radius,fraction = _monte_carlo_r90(84,0,p,16,90,0,0)
    assert radius == 15
    assert fraction == 14/16 < .9


def test_search_uses_identical_wind_model():
    import random
    args = ([(1.5,0),(600,10)],[(1.5,0),(600,-5)])
    assert _perturb_wind_profile(*args,random.Random(123),.2,.5) == canonical_wind(*args,random.Random(123),.2,.5)


def test_small_mass_is_not_inflated_and_bad_position_is_rejected(params):
    p = dict(params,airframe_mass=.005)
    simulation.validate_simulation_inputs(p)
    rocket,_ = simulation._build_flight_rocket(p)
    assert float(rocket.mass) == .005
    result = simulation.simulate_once(84,0,dict(params,fin_pos=0))
    assert not result['ok'] and 'Fin' in result['error']


def test_failed_session_load_is_transactional():
    state = SimpleNamespace(power_on_cd=.45,motor_isp=80)
    with pytest.raises(SessionError):
        dict_to_state(state,dict(aerodynamics={'power_on_cd':.9},motor={'isp':'bad'}))
    assert state.power_on_cd == .45 and state.motor_isp == 80


def test_failed_session_write_preserves_previous_file(tmp_path):
    state = SimpleNamespace(power_on_cd=.45,power_off_cd=.4,motor_isp=80,
                            motor_propellant_density=1700,cd_curve_power_on=None,cd_curve_power_off=None)
    destination = tmp_path/'session.json'
    destination.write_text('previous session')
    with patch('utils.result_io.os.replace',side_effect=OSError('write failure')):
        with pytest.raises(OSError):
            save_session(state,destination)
    assert destination.read_text() == 'previous session'
