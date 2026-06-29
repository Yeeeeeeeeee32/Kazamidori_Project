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
  airframe_len             float   metres
  radius                   float   metres
  nose_len                 float   metres
  fin_root, fin_tip        float   metres
  fin_span                 float   metres
  fin_pos                  float   metres from nose tip  (positive toward tail)
  motor_pos                float   metres from nose tip  (nozzle exit, positive toward tail)
  motor_dry_mass           float   kg
  backfire_delay           float   seconds after burn-out
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
from scipy.interpolate import CubicSpline

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

    The motor is guaranteed to physically fit inside the rocket body:
    ``grain_outer_radius ≤ rocket_radius × 0.80``.

    Parameters
    ----------
    thrust_data   : [[t_s, T_N], ...] — thrust curve in SI
    burn_time     : float  — burn duration (s); used only for propellant-mass
                    back-calculation — the actual SolidMotor burn_time is taken
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
    # BATES standard: core/outer ratio 0.40 (port area ≈ 35 % of face area)
    grain_r_in  = grain_r_out * 0.40

    cross_area = math.pi * (grain_r_out ** 2 - grain_r_in ** 2)
    if cross_area <= 0.0:
        raise ValueError(
            f"Grain cross-sectional area is zero or negative "
            f"(r_out={grain_r_out:.4f} m, r_in={grain_r_in:.4f} m)"
        )

    total_volume = propellant_mass / grain_density

    # Number of segments: aim for height ≈ 2 × outer diameter (BATES aspect).
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

def simulate_once(elev: float, azi: float, params: dict[str, Any], trial_idx: int = 0) -> dict:
    """Run a two-pass RocketPy simulation and return a result dict.

    Pass 1 (terminate_on_apogee=True) finds the altitude at which the
    backfire charge fires so that the parachute trigger altitude can be
    set deterministically.

    Pass 2 (full flight) runs with the altitude-triggered parachute and
    produces the full trajectory arrays.

    The Rocket is built with ``coordinate_system_orientation="nose_to_tail"``
    so all position parameters map directly from the UI (measured from the
    nose tip, positive toward the tail) with no sign inversions.

    Returns a dict with 'ok' True/False.  On success the following keys
    are present:

        ok, apogee_m, hang_time, impact_x, impact_y, r_horiz,
        t_vals, x_vals, y_vals, z_vals, vz_vals,
        idx_bf, idx_para, bf_abs_time, para_open_time,
        backfire_alt, apogee_idx, elev, azi

    Never raises — exceptions are caught and returned as {'ok': False,
    'error': <message>}.
    """
    diag_buffer = []

    import os
    DEBUG_PHYSICS = os.environ.get("DEBUG_PHYSICS") == "1"

    def _diag(tag: str, **kw) -> None:
        if not DEBUG_PHYSICS:
            return

        diag_buffer.append((tag, kw))
        if len(diag_buffer) > 2000:
            lines = []
            for t_tag, t_kw in diag_buffer:
                vals = "  ".join(f"{k}={v}" for k, v in t_kw.items())
                lines.append(f"[Trial {trial_idx}] {t_tag}  {vals}")
            with open("mc_diagnostics.log", "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            diag_buffer.clear()

    try:
        # ── unpack params ────────────────────────────────────────────────────
        airframe_mass  = max(0.01,  params['airframe_mass'])
        airframe_len   = max(0.01,  params['airframe_len'])
        radius         = max(0.001, params['radius'])
        airframe_cg    = params['airframe_cg']
        nose_len       = params['nose_len']
        fin_root       = max(0.001, params['fin_root'])   # must be > 0 for RocketPy Af calc
        fin_tip        = max(0.0,   params['fin_tip'])
        fin_span       = max(0.001, params['fin_span'])   # must be > 0 for RocketPy AR calc
        fin_pos        = params['fin_pos']
        motor_pos      = params['motor_pos']
        motor_dry_mass = params['motor_dry_mass']
        backfire_delay = params['backfire_delay']
        # ── Drag coefficients — boost vs coast phase split (Phase B) ─────────
        # ``power_on_cd``  applies while the motor is producing thrust (jet
        # entrainment lowers base drag slightly versus the coast value).
        # ``power_off_cd`` applies after burnout while the rocket is coasting
        # to apogee.  Both come from the AppState advanced-settings panel.
        power_on_cd  = float(params.get('power_on_cd',  0.45))
        power_off_cd = float(params.get('power_off_cd', 0.40))
        # ``body_cd`` retained as a single-value diagnostic reference (used by
        # the legacy _diag() drag-estimate prints below).  It does NOT feed
        # the Rocket constructor any more — the split values do.
        body_cd      = power_off_cd
        para_cd        = params['para_cd']             # parachute Cd (used only for CdS product)
        para_area      = params['para_area']
        para_lag       = params['para_lag']
        rail           = params['rail']
        # lat/lon are optional: only used for RocketPy's atmospheric model
        # initialisation, not in any trajectory calculation.
        launch_lat     = params.get('launch_lat', 35.0)
        launch_lon     = params.get('launch_lon', 135.0)
        wind_u_prof    = params['wind_u_prof']
        wind_v_prof    = params['wind_v_prof']
        thrust_data    = params['thrust_data']
        motor_burn_time = params['motor_burn_time']

        if not thrust_data:
            return {'ok': False, 'error': 'No thrust data'}

        safe_burn_time = max(0.1, motor_burn_time)
        backfire_time  = safe_burn_time + backfire_delay

        # ── ODE solver tolerances — mode-aware (Proposal F) ──────────────────
        # Looser tolerances reduce ODE step count and speed up simulation.
        # Required accuracy scales with expected apogee:
        #   高度 (Altitude Competition)  ~400 m  → tight   (1e-6)
        #   有翼 (Winged Hover)          ~100 m  → medium  (1e-5)
        #   定点滞空 (Precision Landing)  ~50 m   → relaxed (1e-4)
        #   自由 (Free Mode)             → unconstrained, default to tight (1e-6) for safety
        _fmode = str(params.get('flight_mode', ''))
        if '高度' in _fmode or 'altitude' in _fmode.lower():
            _ode_rtol, _ode_atol, _ode_max_dt = 1e-6, 1e-6, 0.02
        elif '有翼' in _fmode or 'winged' in _fmode.lower() or 'wing' in _fmode.lower():
            _ode_rtol, _ode_atol, _ode_max_dt = 1e-5, 1e-5, 0.03
        elif '定点' in _fmode or 'precision' in _fmode.lower():
            _ode_rtol, _ode_atol, _ode_max_dt = 1e-4, 1e-4, 0.05
        else:
            # 自由 (Free) or default fallback — unconstrained flight profile, use tight tolerances for accuracy
            _ode_rtol, _ode_atol, _ode_max_dt = 1e-6, 1e-6, 0.02

        # ── Moment of inertia ────────────────────────────────────────────────
        # Prefer the precise MoI values computed by ``core/geometry_math.py``
        # and passed in via ``params`` (keys ``I_z`` for the longitudinal/roll
        # axis and ``I_xy`` for the lateral pitch/yaw axes — symmetric for
        # axisymmetric airframes).  Only fall back to the homogeneous solid-
        # cylinder approximation when the caller has not supplied them.
        _I_z_param  = params.get('I_z')
        _I_xy_param = params.get('I_xy')
        if _I_z_param is not None and _I_xy_param is not None:
            I_z  = float(_I_z_param)
            I_xy = float(_I_xy_param)
        else:
            I_z  = 0.5 * airframe_mass * (radius ** 2)
            I_xy = (1 / 12) * airframe_mass * (3 * (radius ** 2) + airframe_len ** 2)

        # ── [GUARD 2] Mass & inertia validation ──────────────────────────────
        _dry_mass_total = airframe_mass + motor_dry_mass
        for _mname, _mval in (
            ('airframe_mass', airframe_mass),
            ('motor_dry_mass', motor_dry_mass),
            ('dry_mass_total', _dry_mass_total),
            ('I_11 (I_xy)', I_xy),
            ('I_22 (I_xy)', I_xy),
            ('I_33 (I_z)',  I_z),
        ):
            if _mval is None or not math.isfinite(_mval) or _mval <= 0.0:
                print(f"[FATAL] Bad Parameter: {_mname}={_mval} — must be finite and > 0.",
                      flush=True)
                return {"ok": False, "error": f"Bad Parameter: {_mname}={_mval}"}

        # ── Atmosphere ───────────────────────────────────────────────────────
        # Use RocketPy's standard atmosphere (ISA) for pressure AND temperature
        # so air-density / drag calculations respect the lapse rate at apogee.
        # ``type="custom_atmosphere"`` is retained as the carrier so the user-
        # supplied wind profile is honoured; passing ``pressure=None`` and
        # ``temperature=None`` tells RocketPy to fall back to the ISA model
        # for those two variables (equivalent to ``standard_atmosphere`` plus
        # custom wind).  The previous hard-coded ``temperature=300`` flattened
        # the entire column to 26.85 °C and corrupted high-altitude density.

        u_alts = [pt[0] for pt in wind_u_prof]
        u_vals = [pt[1] for pt in wind_u_prof]
        v_alts = [pt[0] for pt in wind_v_prof]
        v_vals = [pt[1] for pt in wind_v_prof]

        # ── [GUARD 1] Wind profile validation ────────────────────────────────
        # CubicSpline / LSODA will hang forever on empty, NaN, or unsorted data.
        for _wlabel, _walts in (('u_profile', u_alts), ('v_profile', v_alts)):
            if len(_walts) < 2:
                print(f"[FATAL] Bad Wind Data: {_wlabel} has fewer than 2 altitude points.",
                      flush=True)
                return {"ok": False, "error": "Bad Wind Data: too few altitude points"}
            if any(math.isnan(a) for a in _walts):
                print(f"[FATAL] Bad Wind Data: {_wlabel} contains NaN altitudes.",
                      flush=True)
                return {"ok": False, "error": "Bad Wind Data: NaN altitude"}
            for _i in range(1, len(_walts)):
                if _walts[_i] <= _walts[_i - 1]:
                    print("[FATAL] Bad Wind Data: Altitudes must be strictly increasing.",
                          flush=True)
                    return {"ok": False, "error": "Bad Wind Data"}
        for _wlabel, _wvals in (('u_profile_vals', u_vals), ('v_profile_vals', v_vals)):
            if any(math.isnan(v) for v in _wvals):
                print(f"[FATAL] Bad Wind Data: {_wlabel} contains NaN wind-speed values.",
                      flush=True)
                return {"ok": False, "error": "Bad Wind Data: NaN wind value"}

        u_spline = CubicSpline(u_alts, u_vals, bc_type='natural')
        v_spline = CubicSpline(v_alts, v_vals, bc_type='natural')

        env = Environment(
            latitude=launch_lat, longitude=launch_lon, elevation=0)
            
        p0 = params.get('env_pressure', 101325.0)
        t0 = params.get('env_temp', 15.0)
        
        # GPV Synthesis / Koinobori Baseline (Hybrid ISA Model)
        # Using Koinobori's surface reading (p0, t0) and standard ISA lapse rates 
        # for upper atmosphere approximation.
        t0_k = t0 + 273.15
        lapse_div_t0 = 0.0065 / t0_k

        def _calc_temp(h):
            return t0_k - 0.0065 * h

        def _calc_pres(h):
            return p0 * (1 - lapse_div_t0 * h) ** 5.2561

        env.set_atmospheric_model(
            type="custom_atmosphere",
            pressure=_calc_pres,
            temperature=_calc_temp,
            wind_u=lambda h: float(u_spline(h)),
            wind_v=lambda h: float(v_spline(h)),
        )

        def _build_rocket() -> tuple:
            """Return (Rocket, SolidMotor) for physics inspection."""
            _diag("THRUST_INPUT",
                  n_points=len(thrust_data),
                  t0=thrust_data[0][0] if thrust_data else None,
                  t_end=thrust_data[-1][0] if thrust_data else None,
                  T_max=max(pt[1] for pt in thrust_data) if thrust_data else None,
                  burn_time_param=safe_burn_time)

            # Derive grain geometry from thrust curve + Isp — no hard-coded values.
            # Both Isp and propellant bulk density are now sourced from the
            # AppState advanced-settings panel (defaults match Black Powder).
            isp_s         = float(params.get("motor_isp",                80.0))
            grain_density = float(params.get("motor_propellant_density", 1700.0))
            motor = build_motor_from_curve(
                thrust_data=thrust_data,
                burn_time=safe_burn_time,
                dry_mass=motor_dry_mass,
                rocket_radius=radius,
                isp_s=isp_s,
                grain_density=grain_density,
            )

            # ── motor post-build diagnostics ──────────────────────────────────
            try:
                _mot_impulse = float(motor.total_impulse)
            except Exception:
                _mot_impulse = float("nan")
            try:
                _mot_burnout = float(motor.burn_out_time)
            except Exception:
                _mot_burnout = float("nan")
            try:
                _mot_max_T   = float(motor.max_thrust)
            except Exception:
                _mot_max_T   = float("nan")
            try:
                _mot_prop_m  = float(motor.propellant_mass(0))
            except Exception:
                _mot_prop_m  = float("nan")
            _diag("MOTOR_BUILT",
                  total_impulse_Ns=round(_mot_impulse, 3),
                  burn_out_time_s=round(_mot_burnout, 4),
                  max_thrust_N=round(_mot_max_T, 2),
                  propellant_mass_kg=round(_mot_prop_m, 5))

            # coordinate_system_orientation="nose_to_tail":
            #   Position 0 is the nose tip; positive values run toward the tail.
            #   All UI position parameters (measured from the nose) map here
            #   directly — no minus signs needed.
            _ref_area = math.pi * radius ** 2
            _diag("ROCKET_PARAMS",
                  airframe_mass_kg=round(airframe_mass, 4),
                  radius_m=round(radius, 4),
                  ref_area_m2=round(_ref_area, 6),
                  body_cd=round(body_cd, 3),
                  cd_type=type(body_cd).__name__,
                  expected_drag_at_60ms=round(
                      0.5 * RHO_0 * 60**2 * _ref_area * body_cd, 2))

            # ── Drag model selection (Phase C) ───────────────────────────────
            # RocketPy accepts either a scalar Cd or a list of
            # ``(Mach, Cd)`` tuples for ``power_on_drag`` / ``power_off_drag``.
            # If the operator has loaded a curve via the Advanced Settings
            # dialog it lives in ``params['cd_curve_power_on/off']`` as a list;
            # otherwise the entry is ``None`` and we fall back to the scalar.
            # A curve is considered usable only when it has at least two
            # interpolation points — anything else degrades silently to the
            # scalar so a malformed payload can never break the simulation.
            def _pick_drag(curve, scalar):
                if isinstance(curve, list) and len(curve) >= 2:
                    return curve
                return scalar

            power_on_drag  = _pick_drag(
                params.get("cd_curve_power_on"),  power_on_cd)
            power_off_drag = _pick_drag(
                params.get("cd_curve_power_off"), power_off_cd)

            rk = Rocket(
                radius=radius,
                mass=airframe_mass,
                inertia=(I_xy, I_xy, I_z),
                power_off_drag=power_off_drag,
                power_on_drag=power_on_drag,
                center_of_mass_without_motor=airframe_cg,
                coordinate_system_orientation="nose_to_tail",
            )
            rk.add_motor(motor, position=motor_pos)
            rk.add_nose(length=nose_len, kind=params.get("nose_kind", "vonKarman"), position=0.0)
            # ── Fin position sanity guard ─────────────────────────────────────
            # In nose_to_tail orientation `position` is the distance from the
            # nose tip to the fin root-chord leading edge (m).
            # A valid fin position MUST be:
            #   (a) greater than nose_len  — fins cannot be inside the nose cone
            #   (b) less than airframe_len — fins cannot extend beyond the tail
            # If either bound fails the .rkt parser most likely returned the
            # body-tube top coordinate instead of the Xb-offset fin position.
            # Auto-correct to the aft default: leading edge = tail - root_chord.
            _fin_pos_raw  = fin_pos
            _fin_pos_safe = _fin_pos_raw

            if _fin_pos_safe <= nose_len or _fin_pos_safe >= airframe_len:
                # Place fin leading edge so trailing edge is flush with the tail
                _fin_pos_safe = max(nose_len + 0.001,
                                    airframe_len - fin_root)
                print(
                    f"[WARN] FIN POSITION OUT-OF-RANGE: raw={_fin_pos_raw:.4f} m "
                    f"(nose_len={nose_len:.4f} m, airframe_len={airframe_len:.4f} m). "
                    f"Auto-corrected to {_fin_pos_safe:.4f} m (aft default).",
                    flush=True,
                )

            print(
                f"[DIAG] FIN_BUILD: n={params.get('fin_count', 4)}  "
                f"root={fin_root:.4f} m  tip={fin_tip:.4f} m  "
                f"span={fin_span:.4f} m  pos={_fin_pos_safe:.4f} m  "
                f"(raw_pos={_fin_pos_raw:.4f} m)",
                flush=True,
            )

            rk.add_trapezoidal_fins(
                n=params.get("fin_count", 4),
                root_chord=fin_root,
                tip_chord=fin_tip,
                span=fin_span,
                position=_fin_pos_safe,
            )

            try:
                _wet_mass = float(rk.total_mass(0))
            except Exception:
                _wet_mass = float("nan")
            try:
                _dry_mass_at_burnout = float(rk.total_mass(safe_burn_time))
            except Exception:
                _dry_mass_at_burnout = float("nan")
            _diag("ROCKET_BUILT",
                  total_wet_mass_kg=round(_wet_mass, 4),
                  mass_at_burnout_kg=round(_dry_mass_at_burnout, 4))

            return rk, motor

        # ── Pass 1: find backfire altitude ───────────────────────────────────
        rk1, _motor1 = _build_rocket()
        _diag("PASS1_START", elev_deg=elev, azi_deg=azi, rail_m=rail)

        # ── [GUARD 3] Pre-flight diagnostic dump ──────────────────────────────
        try:
            _pf_mass  = float(rk1.total_mass(0))
        except Exception:
            _pf_mass  = float('nan')
        _pf_wind_pts = len(u_alts)   # same count for u and v profiles
        _pf_para_cds = para_cd * para_area   # CdS product used in Pass 2
        print(
            f"[DIAG] PRE-FLIGHT CHECK (Pass 1): "
            f"Mass={_pf_mass:.4f} kg, "
            f"Wind points={_pf_wind_pts}, "
            f"Parachute Cd*S={_pf_para_cds:.4f} m², "
            f"elev={elev}° azi={azi}° rail={rail} m, "
            f"I_xy={I_xy:.6f} I_z={I_z:.6f} kg·m²",
            flush=True,
        )

        # ── [GUARD 4] Aerodynamic stability check (Pass 1) ───────────────────
        # A negative static margin means CP is ahead of CG — the rocket will
        # tumble violently, driving LSODA to picosecond steps and hanging.
        # In nose_to_tail orientation: CP > CG (numerically) → stable.
        try:
            _p1_cg = float(rk1.center_of_mass(0))
            _p1_cp = float(rk1.cp_position(0))
        except Exception as _sm_exc:
            _p1_cg = float('nan')
            _p1_cp = float('nan')
            print(f"[WARN] Could not evaluate CG/CP for stability check: {_sm_exc}",
                  flush=True)
        print(
            f"[DIAG] STABILITY (Pass 1): CG={_p1_cg:.4f} m, CP={_p1_cp:.4f} m "
            f"(nose_to_tail; CP>CG → stable)",
            flush=True,
        )
        _p1_cal = (2.0 * radius)  # calibre = 1 body diameter
        if math.isfinite(_p1_cg) and math.isfinite(_p1_cp) and _p1_cal > 0:
            _p1_sm = (_p1_cp - _p1_cg) / _p1_cal   # static margin in calibres
            if _p1_sm < 0.0:
                print(
                    f"[FATAL] Rocket is aerodynamically unstable "
                    f"(Static Margin={_p1_sm:.3f} cal). "
                    "Check CG/CP/Motor position.",
                    flush=True,
                )
                return {"ok": False, "error": "Unstable Rocket"}

        # Anti-hang ODE limiters: force a minimum step size and loose tolerances
        # so LSODA cannot stall on chaotic tumbling or extreme AoA transients.
        fl1 = Flight(
            rocket=rk1, environment=env,
            rail_length=rail, inclination=elev, heading=azi,
            terminate_on_apogee=True,
            max_time=120,
            max_time_step=0.05, rtol=1e-3, atol=1e-3,
        )
        t1_arr = fl1.z[:, 0]
        z1_arr = fl1.z[:, 1]

        # ── Burnout state from Pass 1 ─────────────────────────────────────────
        _idx_bo1 = int((np.abs(t1_arr - safe_burn_time)).argmin())
        _bo1_t   = float(t1_arr[_idx_bo1])
        _bo1_z   = float(z1_arr[_idx_bo1])
        _bo1_vz  = float(fl1.vz[_idx_bo1, 1])
        try:
            _bo1_az = float(fl1.az(_bo1_t))   # Bug A: RocketPy Function — call with time
        except Exception:
            _bo1_az = float("nan")
        _diag("PASS1_BURNOUT",
              t_s=round(_bo1_t, 3),
              alt_m=round(_bo1_z, 2),
              vz_m_s=round(_bo1_vz, 2),
              az_m_s2=round(_bo1_az, 2),
              NOTE="if vz>300 m/s → motor too big; if az≈-9.81 → zero drag")

        # ── Sample coasting deceleration from Pass 1 ─────────────────────────
        # Expected a_z during coast ≈ -(g + drag/m).  If ≈ -9.81 → drag = 0.
        _vz1  = fl1.vz[:, 1]
        _t1   = fl1.vz[:, 0]
        _coast_mask = _t1 > (safe_burn_time + 0.05)
        if np.any(_coast_mask):
            _ct = _t1[_coast_mask]
            _cv = _vz1[_coast_mask]
            for _sample_t in [_ct[0], _ct[len(_ct)//4], _ct[len(_ct)//2]]:
                _si = int((np.abs(_ct - _sample_t)).argmin())
                try:
                    _az_s = float(fl1.az(float(_ct[_si])))  # Bug A: RocketPy Function call
                except Exception:
                    _az_s = float("nan")
                _diag("PASS1_COAST_SAMPLE",
                      t_s=round(float(_ct[_si]), 3),
                      vz_m_s=round(float(_cv[_si]), 2),
                      az_m_s2=round(_az_s, 3),
                      expected_az_with_drag="<-9.81")

        _p1_apogee = float(z1_arr.max())
        _diag("PASS1_DONE",
              apogee_m=round(_p1_apogee, 1),
              flight_time_s=round(float(t1_arr[-1]), 2))

        if backfire_time >= t1_arr[-1]:
            backfire_alt = float(z1_arr[-1])
        else:
            idx_bf_p1    = int((np.abs(t1_arr - backfire_time)).argmin())
            backfire_alt = float(z1_arr[idx_bf_p1])
        backfire_alt = max(backfire_alt, 1.0)

        # ── Pass 2: full flight with parachute ───────────────────────────────
        rk2, _motor2 = _build_rocket()

        # ── [GUARD 4b] Aerodynamic stability check (Pass 2) ──────────────────
        try:
            _p2_cg = float(rk2.center_of_mass(0))
            _p2_cp = float(rk2.cp_position(0))
        except Exception as _sm2_exc:
            _p2_cg = float('nan')
            _p2_cp = float('nan')
            print(f"[WARN] Could not evaluate CG/CP (Pass 2): {_sm2_exc}", flush=True)
        print(
            f"[DIAG] STABILITY (Pass 2): CG={_p2_cg:.4f} m, CP={_p2_cp:.4f} m "
            f"(nose_to_tail; CP>CG → stable)",
            flush=True,
        )
        if math.isfinite(_p2_cg) and math.isfinite(_p2_cp) and (2.0 * radius) > 0:
            _p2_sm = (_p2_cp - _p2_cg) / (2.0 * radius)
            if _p2_sm < 0.0:
                print(
                    f"[FATAL] Rocket is aerodynamically unstable "
                    f"(Static Margin={_p2_sm:.3f} cal). "
                    "Check CG/CP/Motor position.",
                    flush=True,
                )
                return {"ok": False, "error": "Unstable Rocket"}

        trig = make_backfire_trigger(backfire_alt)
        rk2.add_parachute(
            "Main",
            cd_s=para_cd * para_area,
            trigger=trig,
            sampling_rate=105,
            lag=para_lag,
        )

        # Anti-hang ODE limiters (mirrors Pass 1)
        fl2 = Flight(
            rocket=rk2, environment=env,
            rail_length=rail, inclination=elev, heading=azi,
            terminate_on_apogee=False,
            max_time=120,
            max_time_step=0.05, rtol=1e-3, atol=1e-3,
        )

        t_vals  = fl2.z[:, 0]
        x_vals  = fl2.x[:, 1]
        y_vals  = fl2.y[:, 1]
        z_vals  = fl2.z[:, 1]
        vz_vals = fl2.vz[:, 1]

        # All arrays are in SI units (metres, seconds) — RocketPy native output.
        # No scaling is applied here.  Consumers must display as-is.

        # ── Event indices ────────────────────────────────────────────────────
        # Motor burnout (燃焼終了点): the time-step closest to safe_burn_time.
        idx_burnout = int((np.abs(t_vals - safe_burn_time)).argmin())

        # Backfire / ejection charge: rocket is descending and at backfire altitude.
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

        # Apogee (最高点) and landing
        apogee_idx = int(np.argmax(z_vals))
        apogee_m   = float(z_vals[apogee_idx])
        impact_x   = float(x_vals[-1])
        impact_y   = float(y_vals[-1])
        r_horiz    = math.hypot(impact_x, impact_y)
        hang_time  = float(t_vals[-1])

        # ── Pass 2 burnout state ──────────────────────────────────────────────
        _idx_bo2 = idx_burnout
        _bo2_t   = float(t_vals[_idx_bo2])
        _bo2_z   = float(z_vals[_idx_bo2])
        _bo2_vz  = float(vz_vals[_idx_bo2])
        try:
            _bo2_az = float(fl2.az(float(_bo2_t)))   # Bug A: RocketPy Function — call with time
        except Exception:
            _bo2_az = float("nan")
        _diag("PASS2_BURNOUT",
              t_s=round(_bo2_t, 3),
              alt_m=round(_bo2_z, 2),
              vz_m_s=round(_bo2_vz, 2),
              az_m_s2=round(_bo2_az, 2))

        # ── Coast sample: measured deceleration vs gravity-only expectation ───
        _coast_start = _idx_bo2 + 1
        _coast_end   = apogee_idx
        if _coast_end > _coast_start + 4:
            _n = _coast_end - _coast_start
            for _frac in (0.05, 0.25, 0.50):
                _ci = _coast_start + int(_frac * _n)
                _vz_c = float(vz_vals[_ci])
                _vz_prev = float(vz_vals[_ci - 1])
                _dt = float(t_vals[_ci]) - float(t_vals[_ci - 1])
                _measured_az = (_vz_c - _vz_prev) / _dt if _dt > 0 else float("nan")
                # Expected: a_z = -(g + C_D * 0.5*rho*v^2*A/m)
                _rho = RHO_0
                _coast_mass = airframe_mass + motor_dry_mass
                _a_drag = (body_cd * 0.5 * _rho * _vz_c**2 * (math.pi * radius**2)
                           / max(_coast_mass, 0.001))
                _expected_az = -(G0 + abs(_a_drag))
                _diag("COAST_DECEL",
                      t_s=round(float(t_vals[_ci]), 3),
                      vz_m_s=round(_vz_c, 2),
                      measured_az=round(_measured_az, 3),
                      expected_az=round(_expected_az, 3),
                      drag_contribution_m_s2=round(-abs(_a_drag), 3))

        _diag("PASS2_RESULT",
              apogee_m=round(apogee_m, 1),
              hang_time_s=round(hang_time, 1),
              impact_x_m=round(impact_x, 1),
              impact_y_m=round(impact_y, 1))

        if DEBUG_PHYSICS and diag_buffer:
            lines = []
            for t_tag, t_kw in diag_buffer:
                vals = "  ".join(f"{k}={v}" for k, v in t_kw.items())
                lines.append(f"[Trial {trial_idx}] {t_tag}  {vals}")
            with open("mc_diagnostics.log", "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            diag_buffer.clear()

        return {
            'ok':             True,
            # ── Scalar summaries (SI: m, s) ──────────────────────────────────
            'apogee_m':       apogee_m,
            'hang_time':      hang_time,
            'impact_x':       impact_x,
            'impact_y':       impact_y,
            'r_horiz':        r_horiz,
            'backfire_alt':   backfire_alt,
            'burn_time_s':    float(safe_burn_time),
            # ── Full trajectory arrays (SI: m, s) ────────────────────────────
            't_vals':         t_vals,
            'x_vals':         x_vals,
            'y_vals':         y_vals,
            'z_vals':         z_vals,
            'vz_vals':        vz_vals,
            # ── Phase-boundary indices into the trajectory arrays ────────────
            # idx_burnout : motor burns out        (推進 → 滑空 boundary)
            # apogee_idx  : peak altitude          (最高点)
            # idx_bf      : ejection charge fires  (backfire trigger)
            # idx_para    : parachute fully open   (滑空 → パラシュート boundary)
            #               -1 if deployment time exceeds total flight time
            'idx_burnout':    idx_burnout,
            'idx_bf':         idx_bf,
            'idx_para':       idx_para,
            'apogee_idx':     apogee_idx,
            # ── Absolute times of key events (s) ─────────────────────────────
            'bf_abs_time':    bf_abs_time,
            'para_open_time': para_open_time,
            # ── Launch config (echoed for traceability) ──────────────────────
            'elev':           elev,
            'azi':            azi,
        }

    except ZeroDivisionError as _zdx:
        import traceback as _tb
        _tb_str = _tb.format_exc()
        _diag("ERROR", msg=f"ZeroDivisionError: {_zdx}")
        if DEBUG_PHYSICS and diag_buffer:
            lines = []
            for t_tag, t_kw in diag_buffer:
                vals = "  ".join(f"{k}={v}" for k, v in t_kw.items())
                lines.append(f"[Trial {trial_idx}] {t_tag}  {vals}")
            with open("mc_diagnostics.log", "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        print(f"[simulate_once] ZeroDivisionError:\n{_tb_str}", flush=True)
        return {'ok': False, 'error': f'ZeroDivisionError: {_zdx}\n{_tb_str}'}
    except Exception as exc:
        import traceback as _tb
        _diag("ERROR", msg=f"{type(exc).__name__}: {exc}")
        if DEBUG_PHYSICS and diag_buffer:
            lines = []
            for t_tag, t_kw in diag_buffer:
                vals = "  ".join(f"{k}={v}" for k, v in t_kw.items())
                lines.append(f"[Trial {trial_idx}] {t_tag}  {vals}")
            with open("mc_diagnostics.log", "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        print(f"[simulate_once] {type(exc).__name__}: {exc}\n{_tb.format_exc()}", flush=True)
        return {'ok': False, 'error': str(exc)}


# ── Fast MC variant ────────────────────────────────────────────────────────────

def simulate_once_mc(
    elev: float,
    azi: float,
    params: dict[str, Any],
    backfire_alt: float,
    trial_idx: int = 0,
) -> dict:
    """Single-pass flight simulation for Monte Carlo trials.

    Identical to :func:`simulate_once` but **skips Pass 1** by accepting a
    pre-computed *backfire_alt* from the nominal (deterministic) run.

    Rationale
    ---------
    The two-pass structure in ``simulate_once`` exists to find the altitude at
    which the ejection charge fires so it can be used as the parachute trigger
    in Pass 2.  For Monte Carlo perturbations (±20 % wind, ±5 % thrust) the
    variation in backfire altitude is small (typically < 0.5 m for
    competition-class rockets at 10–20 m apogee).  Reusing the nominal value
    across all MC trials is a valid engineering approximation that halves the
    number of expensive RocketPy ``Flight`` ODE integrations.

    Parameters
    ----------
    elev, azi    : launch rail orientation (degrees)
    params       : same dict as ``simulate_once`` — must include all keys
    backfire_alt : ejection-charge trigger altitude (m AGL), pre-computed by
                   the nominal ``simulate_once`` call
    trial_idx    : for diagnostic log labelling only

    Returns
    -------
    Same result dict as ``simulate_once``, with 'ok' True/False.
    """
    try:
        airframe_mass  = max(0.01,  params['airframe_mass'])
        airframe_len   = max(0.01,  params['airframe_len'])
        radius         = max(0.001, params['radius'])
        airframe_cg    = params['airframe_cg']
        nose_len       = params['nose_len']
        fin_root       = max(0.001, params['fin_root'])
        fin_tip        = max(0.0,   params['fin_tip'])
        fin_span       = max(0.001, params['fin_span'])
        fin_pos        = params['fin_pos']
        motor_pos      = params['motor_pos']
        motor_dry_mass = params['motor_dry_mass']
        power_on_cd    = float(params.get('power_on_cd',  0.45))
        power_off_cd   = float(params.get('power_off_cd', 0.40))
        para_cd        = params['para_cd']
        para_area      = params['para_area']
        para_lag       = params['para_lag']
        rail           = params['rail']
        launch_lat     = params.get('launch_lat', 35.0)
        launch_lon     = params.get('launch_lon', 135.0)
        wind_u_prof    = params['wind_u_prof']
        wind_v_prof    = params['wind_v_prof']
        thrust_data    = params['thrust_data']
        motor_burn_time = params['motor_burn_time']

        if not thrust_data:
            return {'ok': False, 'error': 'No thrust data'}

        safe_burn_time = max(0.1, motor_burn_time)

        # ── ODE solver tolerances — mode-aware (mirrors simulate_once) ───────
        _fmode = str(params.get('flight_mode', ''))
        if '高度' in _fmode or 'altitude' in _fmode.lower():
            _ode_rtol, _ode_atol, _ode_max_dt = 1e-6, 1e-6, 0.02
        elif '有翼' in _fmode or 'winged' in _fmode.lower() or 'wing' in _fmode.lower():
            _ode_rtol, _ode_atol, _ode_max_dt = 1e-5, 1e-5, 0.03
        elif '定点' in _fmode or 'precision' in _fmode.lower():
            _ode_rtol, _ode_atol, _ode_max_dt = 1e-4, 1e-4, 0.05
        else:
            _ode_rtol, _ode_atol, _ode_max_dt = 1e-6, 1e-6, 0.02

        _I_z_param  = params.get('I_z')
        _I_xy_param = params.get('I_xy')
        if _I_z_param is not None and _I_xy_param is not None:
            I_z  = float(_I_z_param)
            I_xy = float(_I_xy_param)
        else:
            I_z  = 0.5 * airframe_mass * (radius ** 2)
            I_xy = (1 / 12) * airframe_mass * (3 * (radius ** 2) + airframe_len ** 2)

        # ── [GUARD 2-MC] Mass & inertia validation ────────────────────────────
        _dry_mass_total = airframe_mass + motor_dry_mass
        for _mname, _mval in (
            ('airframe_mass', airframe_mass),
            ('motor_dry_mass', motor_dry_mass),
            ('dry_mass_total', _dry_mass_total),
            ('I_11 (I_xy)', I_xy),
            ('I_22 (I_xy)', I_xy),
            ('I_33 (I_z)',  I_z),
        ):
            if _mval is None or not math.isfinite(_mval) or _mval <= 0.0:
                print(f"[FATAL] Bad Parameter: {_mname}={_mval} — must be finite and > 0.",
                      flush=True)
                return {"ok": False, "error": f"Bad Parameter: {_mname}={_mval}"}

        # ── Atmosphere ────────────────────────────────────────────────────────
        u_alts = [pt[0] for pt in wind_u_prof]
        u_vals = [pt[1] for pt in wind_u_prof]
        v_alts = [pt[0] for pt in wind_v_prof]
        v_vals = [pt[1] for pt in wind_v_prof]

        # ── [GUARD 1-MC] Wind profile validation ──────────────────────────────
        for _wlabel, _walts in (('u_profile', u_alts), ('v_profile', v_alts)):
            if len(_walts) < 2:
                print(f"[FATAL] Bad Wind Data: {_wlabel} has fewer than 2 altitude points.",
                      flush=True)
                return {"ok": False, "error": "Bad Wind Data: too few altitude points"}
            if any(math.isnan(a) for a in _walts):
                print(f"[FATAL] Bad Wind Data: {_wlabel} contains NaN altitudes.",
                      flush=True)
                return {"ok": False, "error": "Bad Wind Data: NaN altitude"}
            for _i in range(1, len(_walts)):
                if _walts[_i] <= _walts[_i - 1]:
                    print("[FATAL] Bad Wind Data: Altitudes must be strictly increasing.",
                          flush=True)
                    return {"ok": False, "error": "Bad Wind Data"}
        for _wlabel, _wvals in (('u_profile_vals', u_vals), ('v_profile_vals', v_vals)):
            if any(math.isnan(v) for v in _wvals):
                print(f"[FATAL] Bad Wind Data: {_wlabel} contains NaN wind-speed values.",
                      flush=True)
                return {"ok": False, "error": "Bad Wind Data: NaN wind value"}

        u_spline = CubicSpline(u_alts, u_vals, bc_type='natural')
        v_spline = CubicSpline(v_alts, v_vals, bc_type='natural')

        env = Environment(latitude=launch_lat, longitude=launch_lon, elevation=0)

        p0 = params.get('env_pressure', 101325.0)
        t0 = params.get('env_temp', 15.0)

        t0_k = t0 + 273.15
        lapse_div_t0 = 0.0065 / t0_k

        def _calc_temp(h):
            return t0_k - 0.0065 * h

        def _calc_pres(h):
            return p0 * (1 - lapse_div_t0 * h) ** 5.2561

        env.set_atmospheric_model(
            type="custom_atmosphere",
            pressure=_calc_pres,
            temperature=_calc_temp,
            wind_u=lambda h: float(u_spline(h)),
            wind_v=lambda h: float(v_spline(h)),
        )

        # ── Build rocket (single call) ─────────────────────────────────────────
        isp_s         = float(params.get("motor_isp",                80.0))
        grain_density = float(params.get("motor_propellant_density", 1700.0))
        motor = build_motor_from_curve(
            thrust_data=thrust_data,
            burn_time=safe_burn_time,
            dry_mass=motor_dry_mass,
            rocket_radius=radius,
            isp_s=isp_s,
            grain_density=grain_density,
        )

        def _pick_drag(curve, scalar):
            if isinstance(curve, list) and len(curve) >= 2:
                return curve
            return scalar

        power_on_drag  = _pick_drag(params.get("cd_curve_power_on"),  power_on_cd)
        power_off_drag = _pick_drag(params.get("cd_curve_power_off"), power_off_cd)

        rk = Rocket(
            radius=radius,
            mass=airframe_mass,
            inertia=(I_xy, I_xy, I_z),
            power_off_drag=power_off_drag,
            power_on_drag=power_on_drag,
            center_of_mass_without_motor=airframe_cg,
            coordinate_system_orientation="nose_to_tail",
        )
        rk.add_motor(motor, position=motor_pos)
        rk.add_nose(length=nose_len, kind=params.get("nose_kind", "vonKarman"), position=0.0)
        # ── Fin position sanity guard (MC path mirrors simulate_once) ─────────
        _fin_pos_raw_mc  = fin_pos
        _fin_pos_safe_mc = _fin_pos_raw_mc

        if _fin_pos_safe_mc <= nose_len or _fin_pos_safe_mc >= airframe_len:
            _fin_pos_safe_mc = max(nose_len + 0.001, airframe_len - fin_root)
            print(
                f"[WARN][MC] FIN POSITION OUT-OF-RANGE: raw={_fin_pos_raw_mc:.4f} m "
                f"(nose_len={nose_len:.4f} m, airframe_len={airframe_len:.4f} m). "
                f"Auto-corrected to {_fin_pos_safe_mc:.4f} m (aft default).",
                flush=True,
            )

        rk.add_trapezoidal_fins(
            n=params.get("fin_count", 4),
            root_chord=fin_root,
            tip_chord=fin_tip,
            span=fin_span,
            position=_fin_pos_safe_mc,
        )

        # ── Single-pass flight with pre-known backfire altitude ───────────────
        safe_backfire_alt = max(float(backfire_alt), 1.0)

        # ── [GUARD 3-MC] Pre-flight diagnostic dump ───────────────────────────
        try:
            _pf_mass_mc = float(rk.total_mass(0))
        except Exception:
            _pf_mass_mc = float('nan')
        print(
            f"[DIAG] PRE-FLIGHT CHECK (MC): "
            f"Mass={_pf_mass_mc:.4f} kg, "
            f"Wind points={len(u_alts)}, "
            f"Parachute Cd*S={para_cd * para_area:.4f} m², "
            f"elev={elev}° azi={azi}° rail={rail} m, "
            f"I_xy={I_xy:.6f} I_z={I_z:.6f} kg·m²",
            flush=True,
        )

        # ── [GUARD 4-MC] Aerodynamic stability check ──────────────────────────
        try:
            _mc_cg = float(rk.center_of_mass(0))
            _mc_cp = float(rk.cp_position(0))
        except Exception as _sm_mc_exc:
            _mc_cg = float('nan')
            _mc_cp = float('nan')
            print(f"[WARN] Could not evaluate CG/CP (MC): {_sm_mc_exc}", flush=True)
        print(
            f"[DIAG] STABILITY (MC trial={trial_idx}): CG={_mc_cg:.4f} m, "
            f"CP={_mc_cp:.4f} m (nose_to_tail; CP>CG → stable)",
            flush=True,
        )
        if math.isfinite(_mc_cg) and math.isfinite(_mc_cp) and (2.0 * radius) > 0:
            _mc_sm = (_mc_cp - _mc_cg) / (2.0 * radius)
            if _mc_sm < 0.0:
                print(
                    f"[FATAL] Rocket is aerodynamically unstable "
                    f"(Static Margin={_mc_sm:.3f} cal, trial={trial_idx}). "
                    "Check CG/CP/Motor position.",
                    flush=True,
                )
                return {"ok": False, "error": "Unstable Rocket"}

        trig = make_backfire_trigger(safe_backfire_alt)
        rk.add_parachute(
            "Main",
            cd_s=para_cd * para_area,
            trigger=trig,
            sampling_rate=105,
            lag=para_lag,
        )

        # Anti-hang ODE limiters: force step floor and loose tolerances.
        fl = Flight(
            rocket=rk, environment=env,
            rail_length=rail, inclination=elev, heading=azi,
            terminate_on_apogee=False,
            max_time=120,
            max_time_step=0.05, rtol=1e-3, atol=1e-3,
        )

        t_vals  = fl.z[:, 0]
        x_vals  = fl.x[:, 1]
        y_vals  = fl.y[:, 1]
        z_vals  = fl.z[:, 1]
        vz_vals = fl.vz[:, 1]

        idx_burnout = int((np.abs(t_vals - safe_burn_time)).argmin())
        descending  = vz_vals < 0
        below_alt   = z_vals <= safe_backfire_alt
        bf_cands    = np.where(descending & below_alt)[0]
        idx_bf      = int(bf_cands[0]) if len(bf_cands) > 0 else int(np.argmax(z_vals))

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
            'backfire_alt':   safe_backfire_alt,
            'burn_time_s':    float(safe_burn_time),
            't_vals':         t_vals,
            'x_vals':         x_vals,
            'y_vals':         y_vals,
            'z_vals':         z_vals,
            'vz_vals':        vz_vals,
            'idx_burnout':    idx_burnout,
            'idx_bf':         idx_bf,
            'idx_para':       idx_para,
            'apogee_idx':     apogee_idx,
            'bf_abs_time':    bf_abs_time,
            'para_open_time': para_open_time,
            'elev':           elev,
            'azi':            azi,
        }

    except Exception as exc:
        import traceback as _tb
        print(f"[simulate_once_mc] trial={trial_idx} {type(exc).__name__}: {exc}", flush=True)
        return {'ok': False, 'error': str(exc)}

