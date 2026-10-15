from unittest.mock import Mock

from rocketpy import Rocket
from rocketpy.rocket.aero_surface import FreeFormFins

from core.fin_geometry import validate_fin_options
from core.simulation import _add_fin_set


def test_legacy_fin_keeps_automatic_sweep():
    rocket = Mock()
    _add_fin_set(rocket, dict(fin_count=4), .35, .04, .02, .03)
    rocket.add_trapezoidal_fins.assert_called_once_with(n=4, position=.35, cant_angle=0,
                                                     root_chord=.04, tip_chord=.02, span=.03)


def test_sweep_and_cant_reach_physics():
    rocket=Mock()
    _add_fin_set(rocket, dict(fin_sweep=.01, fin_cant=2), .35, .04, .02, .03)
    assert rocket.add_trapezoidal_fins.call_args.kwargs['sweep_length'] == .01
    assert rocket.add_trapezoidal_fins.call_args.kwargs['cant_angle'] == 2


def test_freeform_vertices_reach_real_rocketpy_surface():
    r=Rocket(radius=.015,mass=.1,inertia=(.001,.001,.0001),power_off_drag=.4,power_on_drag=.4,
             center_of_mass_without_motor=.2,coordinate_system_orientation='nose_to_tail')
    points=[[0,0],[.01,.03],[.04,.03],[.04,0]]
    fin=_add_fin_set(r, dict(fin_shape='freeform', fin_points=points, fin_count=3), .3, .04, .03, .03)
    assert isinstance(fin,FreeFormFins)
    assert r.cp_position(0) > .3
    assert fin.shape_points == points


def test_freeform_dimensions_are_derived_instead_of_stale_input():
    result=validate_fin_options(dict(fin_shape='freeform',fin_points=[[0,0],[.01,.03],[.04,.03],[.04,0]],
                                     fin_root=9,fin_span=9))
    assert result['fin_root'] == .04
    assert result['fin_span'] == .03
    assert result['fin_tip'] == .03
