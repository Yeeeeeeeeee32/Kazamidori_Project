"""
core/simulation.py
Single-run 6-DOF rocket flight simulation via RocketPy.

Public API
----------
make_backfire_trigger(backfire_alt)
    Build the altitude-based parachute trigger closure used in Pass 2.

simulate_once(elev, azi, params) -> dict
    Run the two-pass simulation (apogee-only then full flight) and return
    a result dict.  Always returns a dict; never raises.
    On success:  {'ok': True,  ...trajectory arrays and event indices...}
    On failure:  {'ok': False, 'error': <str>}

SimResult (TypedDict, informational)
    Documents every key returned by simulate_once on success.

params dict keys expected by simulate_once
-------------------------------------------
  launch_lat, launch_lon   float   geodetic degrees
  elev, azi                float   degrees
  rail                     float   metres
  airframe_mass            float   kg
  airframe_cg              float   metres from nose
  airframe_len             float   metres
  radius                   float   metres
  nose_len                 float   metres
  fin_root, fin_tip        float   metres
  fin_span                 float   metres
  fin_pos                  float   metres from nose
  motor_pos                float   metres from nose
  motor_dry_mass           float   kg
  backfire_delay           float   seconds after burn-out
  para_cd                  float   drag coefficient
  para_area                float   reference area m²
  para_lag                 float   deployment lag seconds
  wind_u_prof              list[(alt_m, u_m_s)]   east component profile
  wind_v_prof              list[(alt_m, v_m_s)]   north component profile
  thrust_data              list[[t, T]]
  motor_burn_time          float   seconds
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from rocketpy import Environment, SolidMotor, Rocket, Flight


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

def simulate_once(elev: float, azi: float, params: dict[str, Any]) -> dict:
    """Run a two-pass RocketPy simulation and return a result dict.

    Pass 1 (terminate_on_apogee=True) finds the altitude at which the
    backfire charge fires so that the parachute trigger altitude can be
    set deterministically.

    Pass 2 (full flight) runs with the altitude-triggered parachute and
    produces the full trajectory arrays.

    Returns a dict with 'ok' True/False.  On success the following keys
    are present:

        ok, apogee_m, hang_time, impact_x, impact_y, r_horiz,
        t_vals, x_vals, y_vals, z_vals, vz_vals,
        idx_bf, idx_para, bf_abs_time, para_open_time,
        backfire_alt, apogee_idx, elev, azi

    Never raises — exceptions are caught and returned as {'ok': False,
    'error': <message>}.
    """
    try:
        # ── unpack params ────────────────────────────────────────────────────
        airframe_mass  = max(0.01,  params['airframe_mass'])
        airframe_len   = max(0.01,  params['airframe_len'])
        radius         = max(0.001, params['radius'])
        airframe_cg    = params['airframe_cg']
        nose_len       = params['nose_len']
        fin_root       = params['fin_root']
        fin_tip        = params['fin_tip']
        fin_span       = params['fin_span']
        fin_pos        = params['fin_pos']
        motor_pos      = params['motor_pos']
        motor_dry_mass = params['motor_dry_mass']
        backfire_delay = params['backfire_delay']
        para_cd        = params['para_cd']
        para_area      = params['para_area']
        para_lag       = params['para_lag']
        rail           = params['rail']
        launch_lat     = params['launch_lat']
        launch_lon     = params['launch_lon']
        wind_u_prof    = params['wind_u_prof']
        wind_v_prof    = params['wind_v_prof']
        thrust_data    = params['thrust_data']
        motor_burn_time = params['motor_burn_time']

        if not thrust_data:
            return {'ok': False, 'error': 'No thrust data'}

        safe_burn_time = max(0.1, motor_burn_time)
        backfire_time  = safe_burn_time + backfire_delay

        I_z  = 0.5  * airframe_mass * (radius ** 2)
        I_xy = (1 / 12) * airframe_mass * (3 * (radius ** 2) + airframe_len ** 2)

        env = Environment(
            latitude=launch_lat, longitude=launch_lon, elevation=0)
        env.set_atmospheric_model(
            type="custom_atmosphere", pressure=None, temperature=300,
            wind_u=wind_u_prof, wind_v=wind_v_prof,
        )

        def _build_rocket() -> Rocket:
            motor = SolidMotor(
                thrust_source=thrust_data,
                burn_time=safe_burn_time,
                grain_number=1, grain_density=1815,
                grain_outer_radius=radius * 0.8,
                grain_initial_inner_radius=0.005,
                grain_initial_height=0.1,
                nozzle_radius=radius * 0.8, throat_radius=0.005,
                interpolation_method="linear",
                nozzle_position=0,
                coordinate_system_orientation="nozzle_to_combustion_chamber",
                dry_mass=motor_dry_mass,
                dry_inertia=(1e-5, 1e-5, 1e-6),
                grain_separation=0.0,
                grains_center_of_mass_position=0.0,
                center_of_dry_mass_position=0.0,
            )
            rk = Rocket(
                radius=radius, mass=airframe_mass,
                inertia=(I_xy, I_xy, I_z),
                power_off_drag=para_cd, power_on_drag=para_cd,
                center_of_mass_without_motor=-airframe_cg,
            )
            rk.add_motor(motor, position=-motor_pos)
            rk.add_nose(length=nose_len, kind="vonKarman", position=0.0)
            rk.add_trapezoidal_fins(
                n=4, root_chord=fin_root, tip_chord=fin_tip,
                span=fin_span, position=-fin_pos,
            )
            return rk

        # ── Pass 1: find backfire altitude ───────────────────────────────────
        rk1 = _build_rocket()
        fl1 = Flight(
            rocket=rk1, environment=env,
            rail_length=rail, inclination=elev, heading=azi,
            terminate_on_apogee=True,
        )
        t1_arr = fl1.z[:, 0]
        z1_arr = fl1.z[:, 1]
        if backfire_time >= t1_arr[-1]:
            backfire_alt = float(z1_arr[-1])
        else:
            idx_bf_p1   = int((np.abs(t1_arr - backfire_time)).argmin())
            backfire_alt = float(z1_arr[idx_bf_p1])
        backfire_alt = max(backfire_alt, 1.0)

        # ── Pass 2: full flight with parachute ───────────────────────────────
        rk2  = _build_rocket()
        trig = make_backfire_trigger(backfire_alt)
        rk2.add_parachute(
            "Main",
            cd_s=para_cd * para_area,
            trigger=trig,
            sampling_rate=105,
            lag=para_lag,
        )
        fl2 = Flight(
            rocket=rk2, environment=env,
            rail_length=rail, inclination=elev, heading=azi,
            terminate_on_apogee=False,
        )

        t_vals  = fl2.z[:, 0]
        x_vals  = fl2.x[:, 1]
        y_vals  = fl2.y[:, 1]
        z_vals  = fl2.z[:, 1]
        vz_vals = fl2.vz[:, 1]

        # ── Event indices ────────────────────────────────────────────────────
        descending = vz_vals < 0
        below_alt  = z_vals <= backfire_alt
        bf_cands   = np.where(descending & below_alt)[0]
        idx_bf     = int(bf_cands[0]) if len(bf_cands) > 0 else int(np.argmax(z_vals))

        bf_abs_time    = float(t_vals[idx_bf])
        para_open_time = bf_abs_time + para_lag
        if para_open_time <= t_vals[-1]:
            idx_para = int((np.abs(t_vals - para_open_time)).argmin())
        else:
            idx_para = -1

        apogee_idx = int(np.argmax(z_vals))
        apogee_m   = float(z_vals[apogee_idx])
        impact_x   = float(x_vals[-1])
        impact_y   = float(y_vals[-1])
        r_horiz    = math.hypot(impact_x, impact_y)
        hang_time  = float(t_vals[-1])

        return {
            'ok':             True,
            'apogee_m':       apogee_m,
            'hang_time':      hang_time,
            'impact_x':       impact_x,
            'impact_y':       impact_y,
            'r_horiz':        r_horiz,
            't_vals':         t_vals,
            'x_vals':         x_vals,
            'y_vals':         y_vals,
            'z_vals':         z_vals,
            'vz_vals':        vz_vals,
            'idx_bf':         idx_bf,
            'idx_para':       idx_para,
            'bf_abs_time':    bf_abs_time,
            'para_open_time': para_open_time,
            'backfire_alt':   backfire_alt,
            'apogee_idx':     apogee_idx,
            'elev':           elev,
            'azi':            azi,
        }

    except ZeroDivisionError:
        return {'ok': False,
                'error': 'ZeroDivisionError (launch failure or unstable attitude)'}
    except Exception as exc:
        return {'ok': False, 'error': str(exc)}
