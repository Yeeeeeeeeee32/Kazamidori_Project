"""
core/simulation.py
Single-run 6-DOF rocket flight simulation via RocketPy.

Public API
----------
make_backfire_trigger(backfire_alt)
    Legacy altitude-trigger helper; the timed pipeline does not use it.

simulate_once(elev, azi, params) -> dict
    Run time-continuous pre-ejection and recovery segments and return
    a result dict.  Always returns a dict; never raises.
    On success:  {'ok': True,  ...trajectory arrays and event indices...}
    On failure:  {'ok': False, 'error': <str>}

params dict keys expected by simulate_once
-------------------------------------------
  launch_lat               float   geodetic degrees  (optional, default 35.0)
  launch_lon               float   geodetic degrees  (optional, default 135.0)
    Note: lat/lon are used only to initialise RocketPy's Environment
    atmospheric model.  All trajectory outputs (impact_x, impact_y,
    x_vals, y_vals, z_vals) are in the metric East-North-Up frame
    centred on the launch point.
  elev, azi                float   degrees
  rail                     float   metres
  airframe_mass            float   kg
  airframe_cg              float   metres from nose tip  (positive toward tail)
  cp_override              float|None  Mach-0 CP from nose (m); None uses geometry
  airframe_len             float   metres
  radius                   float   metres
  nose_len                 float   metres
  fin_root, fin_tip        float   metres
  fin_span                 float   metres
  fin_pos                  float   metres from nose tip  (positive toward tail)
  fin_shape                str     'trapezoid' (default) or 'freeform'
  fin_points               list    freeform [[x_aft_m, y_outward_m], ...]
  fin_sweep                float|None  trapezoid sweep (m); None uses RocketPy default
  fin_cant                 float   fin cant angle, degrees (default 0)
  motor_pos                float   metres from nose tip  (nozzle exit, positive toward tail)
  motor_dry_mass           float   kg
  backfire_delay           float   seconds after burn-out (absolute timed event)
  power_on_cd              float   airframe drag coefficient during boost (default 0.45)
  power_off_cd             float   airframe drag coefficient during coast (default 0.40)
  cd_curve_power_on        list|None  Optional [(Mach, Cd), ...] curve used during
                                   boost; overrides power_on_cd when present and
                                   contains ≥ 2 points.  None → use scalar.
  cd_curve_power_off       list|None  Same as above for the coast phase.
  motor_isp                float   motor specific impulse, s (default 80, Black Powder)
  motor_propellant_density float   propellant bulk density, kg/m³ (default 1700, BP)
  I_z                      float   roll/longitudinal MoI, kg·m² (optional; falls back
                                   to a solid-cylinder approximation when omitted)
  I_xy                     float   lateral pitch/yaw MoI, kg·m²  (optional; same fallback)
  para_cd                  float   parachute drag coefficient (used as Cd in CdS product)
  para_area                float   parachute reference area m²
  para_lag                 float   deployment lag seconds
  wind_u_prof              list[(alt_m, u_m_s)]   east component profile
  wind_v_prof              list[(alt_m, v_m_s)]   north component profile
  thrust_data              list[[t, T]]
  motor_burn_time          float   seconds

Coordinate system
-----------------
The Rocket is instantiated with ``coordinate_system_orientation="nose_to_tail"``.
Position 0 is the nose tip; positive values increase toward the tail.
UI parameters (airframe_cg, motor_pos, fin_pos) are all measured from the
nose tip and map directly to RocketPy positions without sign inversion.
"""

import os
import math
import sys
from typing import Any

import numpy as np
from rocketpy import Environment, SolidMotor, Rocket, Flight

from .constants import G0, RHO_0

DEBUG_PHYSICS = (os.environ.get("DEBUG_PHYSICS") == "1")


# ── Motor builder ────────────────────────────────────────────────────────────

def build_motor_from_curve(
    thrust_data:   list,
    burn_time:     float,
    dry_mass:      float,
    rocket_radius: float,
    *,
    isp_s:         float = 80.0,
    grain_density: float = 1700.0,
) -> SolidMotor:
    """Build a physically consistent SolidMotor from a thrust curve.

    Instead of using fixed grain dimensions, this function derives propellant
    mass from the thrust curve's total impulse and an assumed specific impulse,
    then back-calculates the BATES-cylinder grain geometry to match that mass.

    Grain dimensions are estimates; actual casing/mount fit is not verified.

    Parameters
    ----------
    thrust_data   : [[t_s, T_N], ...] — thrust curve in SI
    burn_time     : float  — supplied burn duration (s); actual SolidMotor burn_time is taken
                    from the final time point in the (normalised) curve.
    dry_mass      : float  — motor dry mass after burnout (kg)
    rocket_radius : float  — rocket body radius (m); motor must fit inside
    isp_s         : float  — specific impulse (s)
                    Black-powder Estes motors : ~80–100 s  ← default (80)
                    Low-power APCP            : ~130–170 s
                    Mid/high-power APCP       : ~180–230 s
    grain_density : float  — propellant bulk density (kg/m³)
                    Black Powder              : ~1700 kg/m³  ← default
                    APCP                      : ~1815 kg/m³

    Raises
    ------
    ValueError
        If total impulse is non-positive, grain geometry would be non-physical,
        or the implied exhaust velocity is implausibly low (< 100 m/s).
    """
    def _diag(*args, **kwargs):
        pass

    if not thrust_data or len(thrust_data) < 2:
        raise ValueError("thrust_data must contain at least two points")
    curve = np.asarray(thrust_data, dtype=float)
    if (curve.ndim != 2 or curve.shape[1] != 2 or not np.all(np.isfinite(curve))
            or np.any(np.diff(curve[:,0]) <= 0) or np.any(curve < 0)):
        raise ValueError('Thrust curve requires finite increasing times and non-negative thrust')
    if not all(math.isfinite(float(v)) and v > 0 for v in (dry_mass,rocket_radius,isp_s,grain_density)):
        raise ValueError('Motor mass, radius, Isp and density must be finite and positive')

    # ── Bug C: ensure curve is anchored at t=0 ───────────────────────────────
    # RocketPy requires the thrust curve to start at t=0.  If the first data
    # point has a non-zero time, prepend a zero-thrust point so the integrator
    # and burn_time derivation are both anchored to the true ignition instant.
    thrust_data = [list(pt) for pt in thrust_data]
    if float(thrust_data[0][0]) > 0:
        thrust_data = [[0.0, 0.0]] + thrust_data
    elif float(thrust_data[0][0]) < 0:
        raise ValueError("Thrust curve cannot start at a negative time")

    # ── Total impulse (trapezoidal integration of the thrust curve) ───────────
    total_impulse = sum(
        (thrust_data[i - 1][1] + thrust_data[i][1]) * 0.5
        * (thrust_data[i][0] - thrust_data[i - 1][0])
        for i in range(1, len(thrust_data))
    )
    if total_impulse <= 0.0:
        raise ValueError(
            f"Thrust curve total impulse is non-positive: {total_impulse:.3f} Ns. "
            "Check that the thrust data contains valid (time_s, thrust_N) pairs."
        )

    # ── Propellant mass: m_prop = J / (Isp × g0) ─────────────────────────────
    propellant_mass = total_impulse / (isp_s * G0)
    implied_v_e     = total_impulse / propellant_mass   # = isp_s × g0

    if implied_v_e < 100.0:
        raise ValueError(
            f"Implied exhaust velocity {implied_v_e:.1f} m/s < 100 m/s — "
            f"isp_s={isp_s} s gives a physically impossible propellant model. "
            "Increase isp_s or verify the thrust curve."
        )

    # ── Grain geometry: BATES hollow cylinder sized to fit inside rocket ──────
    # Motor outer radius constrained to 80 % of rocket body radius (casing wall).
    grain_r_out = rocket_radius * 0.80
    # Assumed core/outer ratio 0.40 (port area = 16 % of face area).
    grain_r_in  = grain_r_out * 0.40

    cross_area = math.pi * (grain_r_out ** 2 - grain_r_in ** 2)
    if cross_area <= 0.0:
        raise ValueError(
            f"Grain cross-sectional area is zero or negative "
            f"(r_out={grain_r_out:.4f} m, r_in={grain_r_in:.4f} m)"
        )

    total_volume = propellant_mass / grain_density

    # Number of segments: target one outer diameter in height.
    ideal_h      = 2.0 * grain_r_out
    grain_number = max(1, math.ceil(total_volume / (cross_area * ideal_h)))
    grain_height = total_volume / (grain_number * cross_area)

    if grain_height < 1e-4:
        raise ValueError(
            f"Computed grain height {grain_height*1000:.2f} mm is non-physical. "
            f"total_impulse={total_impulse:.2f} Ns, "
            f"propellant_mass={propellant_mass*1000:.1f} g, "
            f"grain_r_out={grain_r_out*1000:.1f} mm. "
            "Check isp_s and rocket_radius."
        )

    # Nozzle and throat: scale from grain core dimensions
    nozzle_r = grain_r_in * 0.70
    throat_r = nozzle_r  * 0.50

    _diag("BUILD_MOTOR",
          total_impulse_Ns=round(total_impulse, 3),
          propellant_mass_g=round(propellant_mass * 1000, 2),
          implied_v_exhaust_m_s=round(implied_v_e, 1),
          grain_r_out_mm=round(grain_r_out * 1000, 2),
          grain_r_in_mm=round(grain_r_in  * 1000, 2),
          grain_height_mm=round(grain_height * 1000, 2),
          grain_number=grain_number)

    # Bug C: SolidMotor burn_time must match the final time in the curve exactly.
    # Using the last data point avoids rounding mismatches between safe_burn_time
    # (which is max(0.1, motor_burn_time)) and the actual end of the thrust curve.
    curve_burn_time = float(thrust_data[-1][0])

    return SolidMotor(
        thrust_source=thrust_data,
        burn_time=curve_burn_time,
        grain_number=grain_number,
        grain_density=grain_density,
        grain_outer_radius=grain_r_out,
        grain_initial_inner_radius=grain_r_in,
        grain_initial_height=grain_height,
        nozzle_radius=nozzle_r,
        throat_radius=throat_r,
        interpolation_method="linear",
        nozzle_position=0,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
        dry_mass=dry_mass,
        dry_inertia=(1e-5, 1e-5, 1e-6),
        grain_separation=0.0,
        grains_center_of_mass_position=0.0,
        center_of_dry_mass_position=0.0,
    )


# ── Trigger factory ───────────────────────────────────────────────────────────

def make_backfire_trigger(backfire_alt: float):
    """Return a one-shot altitude trigger for RocketPy parachute deployment.

    Fires the first time the rocket is descending (vz < 0) and has fallen
    back to *backfire_alt* metres.  The trigger is stateful (closure) so
    it can only fire once per simulation run.

    Args:
        backfire_alt: Deployment altitude in metres AGL.

    Returns:
        Callable matching the RocketPy trigger signature
        ``trigger(pressure, height, state_vector) -> bool``.
    """
    triggered = [False]

    def trigger(p, h, y):
        if triggered[0]:
            return True
        # y[5] is vz; h is height AGL
        if y[5] < 0 and h <= backfire_alt:
            triggered[0] = True
            return True
        return False

    return trigger


# ── Core simulation ───────────────────────────────────────────────────────────

def _apply_cp_override(rocket, params):
    """Calibrate the nose-to-tail aerodynamic model to a supplied Mach-0 CP.

    Translate every component's aerodynamic centre by the same axial offset.
    This preserves geometry, lift slopes, and the model's Mach-dependent CP
    variation, while updating the moment arms actually consumed by Flight.
    Setting only rocket.cp_position would change the stability diagnostic but
    leave the flight dynamics unchanged.
    """
    raw = params.get('cp_override')
    if raw is None:
        return
    cp = float(raw)
    length = float(params['airframe_len'])
    if isinstance(raw, bool) or not math.isfinite(cp) or not 0 <= cp <= length:
        raise SimulationInputError('CP must be between 0 and the rocket length [m] from the nose')
    automatic_cp = float(rocket.cp_position(0))
    if not math.isfinite(automatic_cp):
        raise SimulationInputError('Automatic CP is invalid; cannot apply manual CP')
    offset = cp - automatic_cp
    for surface, _ in rocket.aerodynamic_surfaces:
        # In our nose_to_tail model, positive local cpz is toward the tail.
        surface.cpz += offset
        surface.cp = (surface.cpx, surface.cpy, surface.cpz)
    rocket.evaluate_center_of_pressure()
    rocket.evaluate_surfaces_cp_to_cdm()
    rocket.evaluate_stability_margin()
    rocket.evaluate_static_margin()


def _add_fin_set(rocket, params, position, root, tip, span):
    from .fin_geometry import validate_fin_options
    options = validate_fin_options(params)
    common = dict(n=params.get('fin_count', 4), position=position,
                  cant_angle=options.get('fin_cant', 0.0))
    if options.get('fin_shape') == 'freeform':
        return rocket.add_free_form_fins(shape_points=options['fin_points'], **common)
    if options.get('fin_sweep') is not None:
        common['sweep_length'] = options['fin_sweep']
    return rocket.add_trapezoidal_fins(root_chord=root, tip_chord=tip, span=span, **common)


class SimulationInputError(ValueError):
    """A simulation could not run with the supplied model/conditions."""


def _unstable_rocket_result(cg: float, cp: float, radius: float, motor, params: dict) -> dict:
    """Explain a launch stability failure without changing the supplied model."""
    margin = (cp - cg) / (2.0 * radius)
    propellant_mass = float(motor.propellant_mass(0))
    isp = float(params.get("motor_isp", 80.0))
    return {
        "ok": False,
        "error_code": "unstable_rocket",
        "error": (
            "Unstable Rocket: launch CG is behind the aerodynamic CP.\n"
            f"CG with motor/propellant: {cg:.4f} m; CP: {cp:.4f} m (from nose).\n"
            f"Static margin: {margin:.3f} cal (negative).\n"
            f"Airframe dry mass: {params['airframe_mass']:.4f} kg; "
            f"motor dry mass: {params['motor_dry_mass']:.4f} kg.\n"
            f"Estimated propellant mass: {propellant_mass:.4f} kg "
            f"from the thrust curve and motor Isp = {isp:g} s.\n"
            "Check the actual airframe CG/mass, fins, motor position/dry mass, "
            "and motor-specific Isp. Changing launch angles does not resolve this condition."
        ),
        "stability": {"cg_m": cg, "cp_m": cp, "static_margin_cal": margin,
                      "propellant_mass_kg": propellant_mass, "motor_isp_s": isp},
    }



def validate_simulation_inputs(params):
    """Validate physical inputs without silently resizing the supplied rocket."""
    positive = ('airframe_mass','airframe_len','radius','nose_len','fin_root','fin_span',
                'motor_dry_mass','para_cd','para_area','rail','motor_burn_time')
    nonnegative = ('fin_tip','airframe_cg','fin_pos','motor_pos','backfire_delay','para_lag')
    for key in positive + nonnegative:
        value = params.get(key)
        if isinstance(value, bool) or value is None or not math.isfinite(float(value)):
            raise SimulationInputError(f'{key} must be a finite number')
        if float(value) < 0 or (key in positive and float(value) == 0):
            raise SimulationInputError(f'{key} must be {"positive" if key in positive else "non-negative"}')
    if not params['nose_len'] <= params['fin_pos'] < params['airframe_len']:
        raise SimulationInputError('Fin leading edge must be on the body, within the rocket length')
    if params['nose_len'] >= params['airframe_len'] or params['airframe_cg'] > params['airframe_len']:
        raise SimulationInputError('Nose length / airframe CG is outside the rocket')
    for key in ('I_z','I_xy'):
        if params.get(key) is not None and (not math.isfinite(float(params[key])) or params[key] <= 0):
            raise SimulationInputError(f'{key} must be finite and positive')
    if not 3 <= int(params.get('fin_count',4)) <= 8 or int(params.get('fin_count',4)) != params.get('fin_count',4):
        raise SimulationInputError('fin_count must be an integer between 3 and 8')


def solver_options(mode):
    """Numerical settings actually supplied to every Flight segment."""
    mode = str(mode).lower()
    if '高度' in mode or 'altitude' in mode:
        return dict(rtol=1e-6, atol=1e-6, max_time_step=.02)
    if '有翼' in mode or 'wing' in mode:
        return dict(rtol=1e-5, atol=1e-5, max_time_step=.03)
    if '定点' in mode or 'precision' in mode:
        return dict(rtol=1e-4, atol=1e-4, max_time_step=.05)
    return dict(rtol=1e-6, atol=1e-6, max_time_step=.02)


def build_environment(params):
    """AGL environment with an invertible tabulated station-pressure model.

    Pressure arrays avoid RocketPy 1.13's callable inverse discretization in
    the wrong input domain. Wind uses the same endpoint-clamped linear
    interpolation as the profile factory and canopy advection.
    """
    p0 = float(params.get('env_pressure',101325.))
    t0 = float(params.get('env_temp',15.)) + 273.15
    if not 0 < p0 <= 120000 or not 150 < t0 < 350:
        raise SimulationInputError('Invalid station pressure or temperature')
    profiles = []
    for key in ('wind_u_prof','wind_v_prof'):
        a = np.asarray(params[key], dtype=float)
        if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.all(np.isfinite(a)) or np.any(np.diff(a[:,0]) <= 0):
            raise SimulationInputError(f'Invalid {key}: finite increasing altitude nodes required')
        profiles.append(a)
    height = np.linspace(0,20000,2001)
    pressure = p0 * (1-.0065*height/t0)**(G0/(287.05*.0065))
    env = Environment(latitude=float(params.get('launch_lat',35)),
                      longitude=float(params.get('launch_lon',135)), elevation=0)
    env.set_atmospheric_model(type='custom_atmosphere',
        pressure=np.column_stack((height,pressure)),
        temperature=lambda h: t0-.0065*h,
        wind_u=lambda h: float(np.interp(h,profiles[0][:,0],profiles[0][:,1])),
        wind_v=lambda h: float(np.interp(h,profiles[1][:,0],profiles[1][:,1])))
    for h in (0.,50.,1000.):
        if abs(float(env.barometric_height(env.pressure(h)))-h) > .1:
            raise SimulationInputError('Atmosphere pressure/height inverse failed')
    return env


def _build_flight_rocket(params):
    p = params
    motor = build_motor_from_curve(p['thrust_data'],p['motor_burn_time'],
        p['motor_dry_mass'],p['radius'],isp_s=float(p.get('motor_isp',80)),
        grain_density=float(p.get('motor_propellant_density',1700)))
    mass, radius, length = p['airframe_mass'],p['radius'],p['airframe_len']
    i_z = p.get('I_z') if p.get('I_z') is not None else .5*mass*radius**2
    i_xy = p.get('I_xy') if p.get('I_xy') is not None else mass*(3*radius**2+length**2)/12
    def drag(key, scalar, default):
        curve = p.get(key)
        return curve if curve is not None else float(p.get(scalar,default))
    rocket = Rocket(radius=radius,mass=mass,inertia=(i_xy,i_xy,i_z),
        power_on_drag=drag('cd_curve_power_on','power_on_cd',.45),
        power_off_drag=drag('cd_curve_power_off','power_off_cd',.4),
        center_of_mass_without_motor=p['airframe_cg'],coordinate_system_orientation='nose_to_tail')
    rocket.add_motor(motor,position=p['motor_pos'])
    rocket.add_nose(length=p['nose_len'],kind=p.get('nose_kind','vonKarman'),position=0)
    _add_fin_set(rocket,p,p['fin_pos'],p['fin_root'],p['fin_tip'],p['fin_span'])
    _apply_cp_override(rocket,p)
    return rocket,motor


def simulate_once(elev: float, azi: float, params: dict[str, Any], trial_idx: int = 0) -> dict:
    """Integrate to the scheduled ejection time, then continue the full state.

    Both segments use absolute flight time. Deployment events come exclusively
    from RocketPy, never from an inferred altitude crossing. Only a completed
    ground impact with an opened recovery system is a successful result.
    """
    try:
        validate_simulation_inputs(params)
        if not math.isfinite(float(elev)) or not 0 < elev <= 90 or not math.isfinite(float(azi)):
            raise SimulationInputError('Invalid launch angle')
        env = build_environment(params)
        rocket,motor = _build_flight_rocket(params)
        cg,cp = float(rocket.center_of_mass(0)),float(rocket.cp_position(0))
        if not math.isfinite(cg) or not math.isfinite(cp):
            raise SimulationInputError('Unable to evaluate launch stability')
        if cp < cg:
            return _unstable_rocket_result(cg,cp,params['radius'],motor,params)
        burn_time = float(motor.burn_out_time)
        ejection_time = burn_time + float(params['backfire_delay'])
        limit = float(params.get('simulation_max_time',120.))
        if not math.isfinite(limit) or limit <= 0:
            raise SimulationInputError('simulation_max_time must be finite and positive')
        common = dict(environment=env,rail_length=params['rail'],inclination=elev,
                      heading=azi,terminate_on_apogee=False,**solver_options(params.get('flight_mode','')))
        ascent = Flight(rocket=rocket,max_time=min(ejection_time,limit),**common)
        if len(ascent.impact_state) > 1:
            return dict(ok=False,error_code='parachute_not_deployed',
                        error='Ground impact occurred before the scheduled ejection. Check motor delay.')
        if ejection_time >= limit:
            return dict(ok=False,error_code='time_limit',error='Simulation reached its time limit before ejection')
        if float(ascent.out_of_rail_time) <= 0:
            return dict(ok=False,error_code='no_liftoff',error='Rocket did not leave the launch rail')
        # Rebuild the same model to keep the preceding segment immutable.
        recovery,_ = _build_flight_rocket(params)
        recovery.add_parachute('Main',cd_s=params['para_cd']*params['para_area'],
            trigger=lambda pressure,height,state: True,sampling_rate=105,lag=params['para_lag'])
        descent = Flight(rocket=recovery,initial_solution=list(ascent.solution[-1]),max_time=limit,**common)
        if len(descent.impact_state) <= 1:
            return dict(ok=False,error_code='time_limit',error='Simulation ended before ground impact (time limit)')
        events = descent.parachute_events
        if not events:
            return dict(ok=False,error_code='parachute_not_deployed',error='No actual parachute deployment event')
        bf_time = float(events[0][0])
        open_time = bf_time + float(events[0][1].lag)
        if open_time >= float(descent.t_final):
            return dict(ok=False,error_code='parachute_not_deployed',error='Ground impact occurred before canopy opening')
        solution = np.vstack((np.asarray(ascent.solution), np.asarray(descent.solution)[1:]))
        t,x,y,z,vz = solution[:,0],solution[:,1],solution[:,2],solution[:,3],solution[:,6]
        if not np.all(np.isfinite(solution)) or abs(z[-1]) > .05:
            return dict(ok=False,error_code='invalid_impact',error='Invalid ground-impact state')
        from .parachute_drift import apply_parachute_drift
        x,y = apply_parachute_drift(t,x,y,z,open_time,params['wind_u_prof'],params['wind_v_prof'],
                                   params.get('para_wind_drift_ratio'))
        from .montblanc import is_montblanc, landing_metrics
        landing = {}
        if is_montblanc(params.get('flight_mode', '')):
            vx, vy = solution[:,4].copy(), solution[:,5].copy()
            ratio = params.get('para_wind_drift_ratio')
            if ratio is not None:
                # Differentiate the prescribed advection analytically, rather
                # than the discretized positions or the pre-correction state.
                deployed = t >= open_time
                for velocity, profile in ((vx, params['wind_u_prof']), (vy, params['wind_v_prof'])):
                    nodes = np.asarray(profile, dtype=float)
                    velocity[deployed] = float(ratio) * np.interp(z[deployed], nodes[:,0], nodes[:,1])
            landing = landing_metrics(t, vx, vy, open_time, params.get('montblanc_options'))
        index = lambda time: int(np.argmin(np.abs(t-time)))
        return dict(ok=True,termination_reason='ground_impact',parachute_deployed=True,**landing,
            apogee_m=float(z.max()),hang_time=float(t[-1]),impact_x=float(x[-1]),impact_y=float(y[-1]),
            r_horiz=math.hypot(x[-1],y[-1]),t_vals=t,x_vals=x,y_vals=y,z_vals=z,vz_vals=vz,
            idx_burnout=index(burn_time),idx_bf=index(bf_time),idx_para=index(open_time),
            apogee_idx=int(np.argmax(z)),bf_abs_time=bf_time,para_open_time=open_time,
            backfire_alt=float(np.interp(bf_time,t,z)),burn_time_s=burn_time,elev=elev,azi=azi,
            solver_settings=solver_options(params.get('flight_mode','')))
    except Exception as exc:
        return dict(ok=False,error_code='invalid_simulation',error=str(exc))


def simulate_once_mc(elev,azi,params,backfire_alt=None,trial_idx=0):
    """Compatibility entry point: MC now uses the same time/event model."""
    return simulate_once(elev,azi,params,trial_idx=trial_idx)
