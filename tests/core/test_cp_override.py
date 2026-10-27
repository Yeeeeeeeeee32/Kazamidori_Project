"""Manual CP must change Flight's moment arms as well as stability diagnostics."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from rocketpy import Rocket

from core.simulation import _apply_cp_override, SimulationInputError, simulate_once, simulate_once_mc


def rocket():
    model = Rocket(radius=.015, mass=.0872, inertia=(.001, .001, .00001),
                   power_off_drag=.45, power_on_drag=.45,
                   center_of_mass_without_motor=.21, coordinate_system_orientation='nose_to_tail')
    model.add_nose(length=.08, kind='vonKarman', position=0)
    model.add_trapezoidal_fins(n=4, root_chord=.04, tip_chord=.02, span=.03, position=.35)
    return model


def test_manual_cp_changes_actual_surface_moment_arms_and_preserves_lift():
    model = rocket()
    machs = (0., .5, 1.2)
    original = [model.cp_position(m) for m in machs]
    lift = [model.total_lift_coeff_der(m) for m in machs]
    arms = dict(model.surfaces_cp_to_cdm)
    geometry = list(model.aerodynamic_surfaces)
    _apply_cp_override(model, dict(cp_override=.32, airframe_len=.383))
    shift = .32 - original[0]
    assert model.cp_position(0) == pytest.approx(.32)
    for mach, old_cp, old_lift in zip(machs, original, lift):
        assert model.cp_position(mach) == pytest.approx(old_cp + shift)
        assert model.total_lift_coeff_der(mach) == pytest.approx(old_lift)
    assert list(model.aerodynamic_surfaces) == geometry
    for surface, arm in arms.items():
        assert model.surfaces_cp_to_cdm[surface].z == pytest.approx(arm.z - shift)
    assert model.static_margin(0) == pytest.approx((.32 - model.center_of_mass(0)) / .03)


def test_blank_cp_preserves_automatic_model():
    model = rocket()
    before_cp, before_arms = model.cp_position, dict(model.surfaces_cp_to_cdm)
    _apply_cp_override(model, {})
    _apply_cp_override(model, dict(cp_override=None))
    assert model.cp_position is before_cp
    assert model.surfaces_cp_to_cdm == before_arms


@pytest.mark.parametrize('cp', [float('nan'), float('inf'), -.1, .5, True])
def test_invalid_manual_cp_rejected(cp):
    with pytest.raises(SimulationInputError, match='CP'):
        _apply_cp_override(rocket(), dict(cp_override=cp, airframe_len=.383))


def params():
    p = json.loads((Path(__file__).parents[1] / 'fixtures/simulation_params.json').read_text())
    p.update(thrust_data=[[0, 0], [.1, 12], [.3, 7], [1.1, 0]], motor_burn_time=1.1)
    return p


def test_manual_unstable_cp_blocks_nominal_and_mc_before_flight():
    p = params()
    p['cp_override'] = .1
    for simulate in (lambda: simulate_once(84, 0, p),
                     lambda: simulate_once_mc(84, 0, p, 90)):
        with patch('core.simulation.Flight') as flight:
            result = simulate()
        assert not result['ok']
        assert result['error_code'] == 'unstable_rocket'
        assert result['stability']['cp_m'] == pytest.approx(.1)
        flight.assert_not_called()


@pytest.mark.physics
@pytest.mark.parametrize('fin_settings', [{}, dict(fin_shape='freeform',
    fin_points=[[0,0],[.02,.03],[.04,.03],[.04,0]])])
def test_cp_applied_to_both_nominal_passes_and_mc_real_flights(fin_settings):
    p = params()
    p.update(fin_settings)
    p['cp_override'] = .32
    applied = []
    def apply(model, config):
        _apply_cp_override(model, config)
        applied.append(model.cp_position(0))
    with patch('core.simulation._apply_cp_override', side_effect=apply):
        nominal = simulate_once(84, 0, p)
        assert nominal['ok'], nominal.get('error')
        mc = simulate_once_mc(84, 0, p, nominal['backfire_alt'])
        assert mc['ok'], mc.get('error')
    assert applied == pytest.approx([.32]*4)  # Two time-continuation segments per run, including MC.
