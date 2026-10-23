"""Independent analytic checks, not baselines copied from the implementation."""
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import chi2

from core.monte_carlo import (compute_error_ellipse, compute_cep_ellipse,
    ellipse_points, compute_kde_grid, kde_density_threshold, chi2_scale,
    landing_region_fraction)
from core.wind_model import speed_dir_to_uv, uv_to_speed_dir
from core.geometry_math import (Component, MoITriplet, moi_solid_cylinder,
    moi_hollow_cone, calculate_system_cg, calculate_total_moi)
from core.simulation import build_motor_from_curve, simulate_once
from core.atmosphere import pressure_altitude_agl


@pytest.mark.parametrize('degrees',[0,15,45,90,120,165])
def test_covariance_ellipse_and_kde_have_consistent_axes(degrees):
    theta=math.radians(degrees)
    rotation=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
    points=np.random.default_rng(716).normal(size=(1000,2))*[8,1]
    points=points@rotation.T+[30,-12]
    ellipse=compute_error_ellipse(points)
    assert ellipse==compute_cep_ellipse(points,.9)
    values,vectors=np.linalg.eigh(np.cov(points,rowvar=False))
    expected=vectors[:,-1]
    actual=np.array([math.cos(ellipse['angle_rad']),math.sin(ellipse['angle_rad'])])
    assert abs(actual@expected)==pytest.approx(1,abs=1e-12)
    boundary=ellipse_points(ellipse)-points.mean(axis=0)
    mahal=np.einsum('ni,ij,nj->n',boundary,np.linalg.inv(np.cov(points,rowvar=False)),boundary)
    np.testing.assert_allclose(mahal,chi2.ppf(.9,2),atol=1e-10)
    grid=compute_kde_grid(points,grid_size=100)
    x,y,z=(np.asarray(grid[k]) for k in ('X_m','Y_m','Z'))
    xy=np.column_stack((x.ravel(),y.ravel()))
    weights=z.ravel()/z.sum()
    delta=xy-np.sum(weights[:,None]*xy,axis=0)
    _,kde_axes=np.linalg.eigh((delta.T*weights)@delta)
    assert abs(actual@kde_axes[:,-1])> .9999


@pytest.mark.parametrize('percent',[1,50,68,75,90,93,99])
def test_all_percentages_use_the_correct_chi_squared_quantile(percent):
    assert chi2_scale(percent)**2==pytest.approx(chi2.ppf(percent/100,2))


def test_gaussian_population_coverage_and_degenerate_axes():
    points=np.random.default_rng(721).multivariate_normal([10,-30],[[9,5],[5,4]],size=50000)
    e=compute_error_ellipse(points)
    angle=e['angle_rad']
    rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    local=(points-[e['cx'],e['cy']])@rotation
    inside=np.mean(np.sum((local/[e['a'],e['b']])**2,axis=1)<=1)
    assert inside==pytest.approx(.9,abs=.005)
    point=compute_error_ellipse([[10,20]]*8)
    assert point['a']==point['b']==0
    line=compute_error_ellipse([[i/1000,0] for i in range(8)])
    assert line['b']==0 and line['a']<.01
    assert compute_kde_grid([[10,20]]*8) is None
    assert compute_cep_ellipse([[0,0]]*4,float('nan')) is None


def test_kde_probability_threshold_is_integrated_mass_not_density_at_samples():
    axis=np.linspace(-6,6,401)
    x,y=np.meshgrid(axis,axis)
    z=np.exp(-(x*x+y*y)/2)
    grid={'Z':z}
    for probability in (.5,.7,.9,.95):
        threshold=kde_density_threshold(grid,probability)
        # For an isotropic 2-D normal, peak-relative HDR threshold is 1-p.
        assert threshold==pytest.approx(1-probability,abs=.002)
        assert z[z>=threshold].sum()/z.sum()==pytest.approx(probability,abs=.002)


def test_landing_acceptance_includes_bias_target_offset_and_failed_trials():
    points=np.array([[100+i/100,100] for i in range(10)])
    assert landing_region_fraction(points,10)==0
    assert landing_region_fraction(points,10,target=(100,100))==1
    assert landing_region_fraction(points,10,target=(100,100),attempted=20)==.5
    assert landing_region_fraction([],10)==0


@pytest.mark.parametrize('direction,expected',[(0,(0,-5)),(90,(-5,0)),(180,(0,5)),(270,(5,0))])
def test_meteorological_wind_cardinal_directions_and_roundtrip(direction,expected):
    uv=speed_dir_to_uv(5,direction)
    np.testing.assert_allclose(uv,expected,atol=1e-12)
    speed,back=uv_to_speed_dir(*uv)
    assert speed==pytest.approx(5)
    assert back==pytest.approx(direction)


def test_motor_impulse_propellant_and_mass_balance():
    thrust=[[0,0],[.1,12],[.3,7],[1.1,0]]
    motor=build_motor_from_curve(thrust,1.1,.015,.015,isp_s=80)
    impulse=.1*12/2+.2*(12+7)/2+.8*7/2
    assert float(motor.total_impulse)==pytest.approx(impulse,rel=1e-5)
    assert float(motor.propellant_mass(0))==pytest.approx(impulse/(80*9.80665),rel=1e-5)
    assert float(motor.propellant_mass(1.1))==pytest.approx(0,abs=1e-7)
    assert float(motor.total_mass(0))-float(motor.total_mass(1.1))==pytest.approx(float(motor.propellant_mass(0)),abs=1e-7)


def test_mass_centroid_and_parallel_axis_translation_invariance():
    components=[Component('A',2,1,MoITriplet(0,0,0)),Component('B',1,4,MoITriplet(0,0,0))]
    cg=calculate_system_cg(components)
    assert cg==2
    assert calculate_total_moi(components,cg)==MoITriplet(0,6,6)
    moved=[Component(c.name,c.mass,c.local_cg+100,c.local_moi) for c in components]
    assert calculate_total_moi(moved,calculate_system_cg(moved))==MoITriplet(0,6,6)
    cylinder=moi_solid_cylinder(2,.1,1)
    assert cylinder.Ixx==pytest.approx(.01)
    assert cylinder.Iyy==pytest.approx(2*(3*.1**2+1)/12)
    # Independent quadrature of a thin conical shell: dm proportional to x dx.
    x=(np.arange(100000)+.5)/100000
    dm=2*x/100000
    expected_roll=np.sum((.1*x)**2*dm)
    expected_pitch=np.sum(((.1*x)**2/2+(x-2/3)**2)*dm)
    cone=moi_hollow_cone(1,.1,1)
    assert cone.Ixx==pytest.approx(expected_roll,abs=1e-10)
    assert cone.Iyy==pytest.approx(expected_pitch,abs=1e-10)


def test_pressure_altitude_is_inverse_of_hydrostatic_lapse_relation():
    for height in (0,100,1000,5000):
        pressure=100000*(1-.0065*height/293.15)**(9.80665/(287.05*.0065))
        assert pressure_altitude_agl(pressure/100,100000,20)==pytest.approx(height,abs=1e-8)


@pytest.mark.physics
def test_heading_rotation_in_calm_air_preserves_flight():
    from rocketpy import Environment, Flight
    p=json.loads((Path(__file__).parents[1]/'fixtures/simulation_params.json').read_text())
    p.update(thrust_data=[[0,0],[.1,12],[.3,7],[1.1,0]],motor_burn_time=1.1)
    # Rotational symmetry holds for a nonrotating Earth. Real RocketPy includes
    # Coriolis acceleration; do not remove it from the production flight model.
    def environment(**kwargs):
        env = Environment(**kwargs)
        env.earth_rotation_vector = (0.,0.,0.)
        return env
    def flight(**kwargs):
        kwargs.update(rtol=1e-9, atol=1e-9, max_time_step=.005)
        return Flight(**kwargs)
    with patch('core.simulation.Environment',side_effect=environment), \
         patch('core.simulation.Flight',side_effect=flight):
        north=simulate_once(84,0,p)
        east=simulate_once(84,90,p)
    assert north['ok'] and east['ok']
    assert east['impact_x']==pytest.approx(north['impact_y'],abs=.001)
    assert east['impact_y']==pytest.approx(-north['impact_x'],abs=.001)
    assert east['apogee_m']==pytest.approx(north['apogee_m'],abs=.001)
    assert east['hang_time']==pytest.approx(north['hang_time'],abs=.001)
