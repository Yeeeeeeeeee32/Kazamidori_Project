import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.parachute_drift import apply_parachute_drift
from utils.configuration import validate_config, apply_config, read_config, CONFIG_PATH


@pytest.mark.parametrize('ratio', [0, .25, .5, 1, 2])
def test_parachute_constant_wind_integral_and_fractional_opening(ratio):
    t = np.array([0., 1., 2., 3., 4.])
    x, y = apply_parachute_drift(t, t*3, t*2, [20,15,10,5,0], 1.5,
                               [(0, 8), (100,8)], [(0,-4),(100,-4)], ratio)
    np.testing.assert_allclose(x[:2], [0,3])
    np.testing.assert_allclose(y[:2], [0,2])
    assert x[-1] == pytest.approx(4.5 + 8*ratio*2.5)
    assert y[-1] == pytest.approx(3 - 4*ratio*2.5)


def test_parachute_uses_altitude_dependent_wind_and_preserves_original_arrays():
    original = np.array([0., 1., 2.])
    x, y = apply_parachute_drift([0,1,2], original, original, [10,5,0], 0,
                               [(0,0),(10,10)], [(0,0),(10,0)], .5)
    np.testing.assert_allclose(x, [0,3.75,5])
    np.testing.assert_allclose(y, [0,0,0])
    np.testing.assert_allclose(original, [0,1,2])


def test_invalid_config_is_rejected_before_mutation():
    state = SimpleNamespace(motor_isp=80)
    with pytest.raises(ValueError):
        apply_config(state, {'motor': {'isp_s':90}, 'parachute': {'wind_drift_pct':float('nan')}})
    assert state.motor_isp == 80
    with pytest.raises(ValueError):
        validate_config({'practice_wind': [{'alt_m':3,'speed_ms':-1,'dir_deg':0}]})


def test_shipped_config_validates_and_separates_motor_and_recovery():
    config = read_config(CONFIG_PATH)
    assert config['parachute']['wind_drift_pct'] == 100
    assert config['simulation']['mc_n_runs'] == 50
    assert len(config['practice_wind']) == 5
    assert config['practice_wind'][0]['alt_m'] == 1.5
    assert 'dry_mass_kg' in config['motor']


def test_legacy_surface_height_migrates_without_changing_wind_or_input():
    legacy = {'practice_wind': [dict(alt_m=3, speed_ms=7, dir_deg=359),
                                dict(alt_m=10, speed_ms=9, dir_deg=1)]}
    state = SimpleNamespace()
    apply_config(state, legacy)
    assert state.wind_profile_data == [dict(alt_m=1.5, speed_ms=7, dir_deg=359),
                                       dict(alt_m=10, speed_ms=9, dir_deg=1)]
    assert legacy['practice_wind'][0]['alt_m'] == 3
    assert validate_config({'practice_wind': state.wind_profile_data})['practice_wind'] == state.wind_profile_data


@pytest.mark.physics
def test_nominal_and_fast_mc_apply_same_prescribed_canopy_velocity():
    from pathlib import Path
    from core.simulation import simulate_once, simulate_once_mc
    params = json.loads((Path(__file__).parents[1]/'fixtures/simulation_params.json').read_text())
    params.update(thrust_data=[[0,0],[.1,12],[.3,7],[1.1,0]],motor_burn_time=1.1,
                  para_wind_drift_ratio=.25, wind_u_prof=[[0,4],[1000,4]])
    nominal = simulate_once(84,0,params)
    assert nominal['ok'], nominal.get('error')
    fast = simulate_once_mc(84,0,params,nominal['backfire_alt'])
    assert fast['ok'], fast.get('error')
    for result in (nominal,fast):
        after = np.flatnonzero(np.asarray(result['t_vals']) > result['para_open_time'])
        assert len(after) > 2
        # 25% of 4 m/s east wind = exactly 1 m/s after opening.
        x,t = np.asarray(result['x_vals']),np.asarray(result['t_vals'])
        np.testing.assert_allclose(np.diff(x[after]),np.diff(t[after]),atol=1e-10)
