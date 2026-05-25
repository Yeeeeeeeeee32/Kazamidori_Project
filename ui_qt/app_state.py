"""
ui_qt/app_state.py

Single source of truth for all Kazamidori application state.

Every property has a dedicated Signal; any assignment to a property
automatically emits that signal, allowing Views and ViewModels to bind
reactively without polling or manual notification chains.

Categories
----------
- Simulation configuration  (wind/thrust uncertainty, MC settings)
- Launch site               (lat/lon)
- Rocket / flight params    (mass, Cd, area, target radius, mode)
- Simulation results        (landing position, r90, phase1)
- Monte Carlo results       (scatter, ellipse, CEP, KDE contours)
- Phase 2 / live tracking   (p2 ellipse, phase2 active flag)
- Real-time wind            (surface + upper speed/dir, gust)
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional

from PySide6.QtCore import QObject, Signal, Property

# ── Constants ───────────────────────────────────────────────
WIND_HISTORY_MAX_SAMPLES: int = 60

# ── Wind-history altitude constants ───────────────────────────────────────────
# Must match core.wind_model.WIND_SAMPLE_ALTS exactly.
# Defined here to avoid importing from core in the UI state layer.
_WIND_SAMPLE_ALTS: tuple[float, ...] = (3.0, 10.0, 150.0, 300.0, 600.0)
_SURFACE_ALT: float = 3.0   # 自作風速計 (hardware anemometer) altitude (m AGL)


class AppState(QObject):

    # ── Simulation configuration ───────────────────────────────────────────────
    wind_uncertainty_changed   = Signal(float)
    magnetic_declination_changed = Signal(float)
    offline_map_extent_changed = Signal(list)
    thrust_uncertainty_changed = Signal(float)
    landing_prob_changed       = Signal(int)
    mc_n_runs_changed          = Signal(int)

    # ── Launch site ────────────────────────────────────────────────────────────

    # ── Rocket / flight parameters ─────────────────────────────────────────────
    mass_changed           = Signal(float)
    drag_coeff_changed     = Signal(float)
    ref_area_changed       = Signal(float)
    target_radius_changed  = Signal(float)
    launch_lat_changed     = Signal(float)
    launch_lon_changed     = Signal(float)
    operation_mode_changed = Signal(str)

    # ── Aerodynamics & Motor (advanced settings, exposed in Phase B) ──────────
    # power_on_cd / power_off_cd split the previous single body_cd so the
    # coast phase can carry a different drag coefficient from the boost phase.
    # motor_isp / motor_propellant_density are surfaced so the operator can
    # pick the correct propellant chemistry (defaults are Black Powder).
    power_on_cd_changed              = Signal(float)
    power_off_cd_changed             = Signal(float)
    motor_isp_changed                = Signal(float)
    motor_propellant_density_changed = Signal(float)

    # ── Mach-dependent Cd curves (Phase C) ────────────────────────────────────
    # Each carries either ``None`` (fall back to the scalar above) or a
    # ``list[tuple[float, float]]`` of (Mach, Cd) pairs sorted by ascending
    # Mach.  The list payload is object-typed since PySide6's primitive Signal
    # types cannot express ``Optional[list[...]]`` directly.
    cd_curve_power_on_changed  = Signal(object)
    cd_curve_power_off_changed = Signal(object)

    # ── Simulation results ─────────────────────────────────────────────────────
    land_lat_changed       = Signal(float)
    land_lon_changed       = Signal(float)
    r90_radius_changed     = Signal(float)
    has_sim_result_changed = Signal(bool)
    phase1_result_changed  = Signal(object)

    # ── 3D Playback ────────────────────────────────────────────────────────────
    current_playback_index_changed = Signal(int)

    # ── External System Status ─────────────────────────────────────────────────
    koinobori_status_changed    = Signal(str)
    gpv_last_fetch_time_changed = Signal(str)

    # ── Monte Carlo results ────────────────────────────────────────────────────
    mc_scatter_changed   = Signal(object)
    mc_ellipse_changed   = Signal(object)
    mc_cep_changed       = Signal(float)
    kde_contours_changed = Signal(object)
    mc_running_changed   = Signal(bool)

    # ── Phase 2 / live tracking ────────────────────────────────────────────────
    p2_ellipse_changed    = Signal(object)
    phase2_active_changed = Signal(bool)

    # ── Real-time wind ─────────────────────────────────────────────────────────
    wind_profile_changed     = Signal(object)
    wind_profile_data_changed = Signal(object)
    gust_speed_changed       = Signal(float)

    # ── Wind history ───────────────────────────────────────────────────────────
    wind_history_updated    = Signal(object)   # deque snapshot after each append

    # ── CEP probability ────────────────────────────────────────────────────────
    cep_probability_changed = Signal(float)

    # ── View Toggles ───────────────────────────────────────────────────────────
    show_kde_changed     = Signal(bool)
    show_cep_changed     = Signal(bool)
    show_scatter_changed = Signal(bool)
    show_burnout_changed = Signal(bool)
    show_apogee_changed  = Signal(bool)

    # ── Unified simulation result ──────────────────────────────────────────────
    # simulation_result holds the complete payload dict emitted by
    # SimulationWorker.finished.  Any component that needs the full result
    # (trajectory arrays, scatter, ellipse, KDE contours, …) connects to
    # simulation_result_changed rather than observing individual properties.
    simulation_result_changed = Signal(object)

    # needs_redraw is a broadcast notification: "the canvas should repaint".
    # Setting simulation_result automatically emits this signal; it can also
    # be emitted independently (e.g. when only wind params change).
    needs_redraw = Signal()

    # ── Smart redraw signals ───────────────────────────────────────────────────
    # needs_full_redraw: new simulation data arrived — full canvas repaint needed.
    needs_full_redraw    = Signal(dict)
    # needs_partial_redraw: UI-only param changed (e.g. landing_probability) —
    # only recompute overlays from cached_mc_scatter, never re-run simulation.
    needs_partial_redraw = Signal()
    # simulation_started / simulation_finished bracket each worker run so views
    # can transition between idle / busy / results states cleanly.
    simulation_started   = Signal()
    simulation_finished  = Signal(dict)

    # ── Overlay display parameter signals ─────────────────────────────────────
    wind_uncertainty_display_changed  = Signal(float)
    cached_mc_scatter_changed         = Signal(object)

    # ── Wind / Phase 2 operational signals ────────────────────────────────────
    # wind_updated: lightweight ping after every append_wind_reading call.
    # Observers that only need to know "new data arrived" connect here instead
    # of wind_history_updated (which carries the full deque payload).
    wind_updated = Signal()

    # tolerance_exceeded: emitted every wind tick while Phase 2 is active and
    # the live wind vector is outside the locked Phase A k-sigma envelope.
    # Payload: human-readable string with current speed vs. baseline.
    tolerance_exceeded = Signal(str)

    # tolerance_status_changed: emitted only when the status string transitions
    # (e.g. "✓ In bounds" → "⚠ EXCEEDED" or back).  Used to toggle GO/NO-GO.
    tolerance_status_changed = Signal(str)

    # ── Simulation busy flag ───────────────────────────────────────────────────
    # Emitted True the instant a worker thread is launched; False the instant it
    # finishes, errors, or is cancelled.  Views subscribe here to wipe stale
    # simulation artifacts before new results arrive, eliminating data ghosting.
    is_calculating_changed = Signal(bool)

    # ── Two-stage rendering signals ────────────────────────────────────────────
    # nominal_result_changed: carries the full nominal payload dict the instant
    # the single-run completes — before any MC iterations start.  Views that
    # render the 3-D trajectory connect here for the earliest possible draw.
    nominal_result_changed = Signal(object)
    # nominal_needs_redraw: zero-payload companion ping.  Lightweight observers
    # that only need to know "nominal is ready" connect here instead of the
    # heavier object-carrying signal.
    nominal_needs_redraw   = Signal()

    # ── MC run progress ────────────────────────────────────────────────────────
    # progress_changed: 0–100 integer updated after every MC iteration.
    # Setting progress_percentage to 0 signals idle / complete — QProgressBar
    # connections can use this to clear the bar automatically.
    progress_changed = Signal(int)

    # ── Rocket geometry parameters ─────────────────────────────────────────────
    # One signal per physical dimension so views can bind selectively.
    rocket_dry_mass_changed = Signal(float)   # airframe dry mass (kg)
    rocket_cg_changed       = Signal(float)   # airframe CG from nose (m)
    rocket_length_changed   = Signal(float)   # total airframe length (m)
    rocket_diameter_changed = Signal(float)   # body diameter in metres (= 2×radius)
    nose_length_changed     = Signal(float)   # nose cone length (m)
    fin_root_chord_changed  = Signal(float)   # fin root chord (m)
    fin_tip_chord_changed   = Signal(float)   # fin tip chord (m)
    fin_span_changed        = Signal(float)   # fin semi-span (m)
    fin_position_changed    = Signal(float)   # fin leading-edge position from nose (m)
    motor_cg_pos_changed    = Signal(float)   # motor CG from nose (m)
    motor_dry_mass_changed  = Signal(float)   # motor dry mass (kg)
    parachute_cd_changed    = Signal(float)   # parachute drag coefficient (dimensionless)
    parachute_area_changed  = Signal(float)   # parachute reference area (m²)
    parachute_lag_changed   = Signal(float)   # parachute deployment lag (s)
    backfire_delay_changed  = Signal(float)   # ejection charge delay after burnout (s)

    # ── Engine / motor metadata ────────────────────────────────────────────────
    # Emitted when load_engine() is called with a parsed motor CSV.
    # Payload: {"file_path": str, "avg_thrust": float,
    #           "burn_time": float, "curve_data": list}
    engine_loaded = Signal(dict)

    # ── Moment of Inertia (from .rkt parser) ──────────────────────────────────
    # Emitted by set_moi() after an .rkt file is parsed.
    # Carries (Ixx, Iyy, Izz) in kg·m² — roll, pitch, yaw.
    moi_updated = Signal(float, float, float)

    # ── Launch settings ────────────────────────────────────────────────────────
    launch_angle_changed = Signal(float)   # elevation angle (degrees)
    launch_rail_changed  = Signal(float)   # rail length (m)

    # ── Flight mode ────────────────────────────────────────────────────────────
    # Mission profile selected by the operator (e.g. "Altitude", "Precision").
    flight_mode_changed = Signal(str)

    # ── Parameter readiness interlock ─────────────────────────────────────────
    # Emitted True when all critical geometry + recovery parameters are non-None.
    # The RUN button connects here to enable/disable itself without polling.
    sig_ready_state_changed = Signal(bool)

    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self, config: Optional[dict] = None, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        cfg = config or {}

        # Simulation configuration — None until the operator explicitly enters values.
        # _f_int / _f_float: None when the key is absent from cfg, float/int otherwise.
        _fi = lambda k: (int(cfg[k])   if k in cfg else None)
        _ff = lambda k: (float(cfg[k]) if k in cfg else None)
        self._wind_uncertainty   = _ff("wind_uncertainty")    # fractional σ, e.g. 0.20
        self._thrust_uncertainty = _ff("thrust_uncertainty")  # fractional σ, e.g. 0.05
        self._landing_prob       = _fi("landing_prob")        # percentile (e.g. 90)
        self._mc_n_runs          = _fi("mc_n_runs")           # run count (e.g. 200)

        # Launch site — must be confirmed by GPS / manual entry; no hardcoded fallback.
        self._launch_lat = _ff("launch_lat")    # decimal degrees
        self._launch_lon = _ff("launch_lon")    # decimal degrees
        self._magnetic_declination = 0.0
        self._offline_map_extent = [-250.0, 250.0, -250.0, 250.0]

        self._current_playback_index = 0

        # Rocket / flight parameters
        self._mass           = _ff("mass")           # kg   (legacy scalar; geometry detail below)
        self._drag_coeff     = _ff("drag_coeff")     # dimensionless
        self._ref_area       = _ff("ref_area")       # m²
        self._target_radius  = _ff("target_radius")  # m   (landing-zone radius)
        self._operation_mode = str(cfg.get("operation_mode", "Precision Landing"))

        # Simulation results (outputs — None until first run)
        self._land_lat       = None   # not meaningful until a simulation completes
        self._land_lon       = None
        self._r90_radius     = 0.0
        self._has_sim_result = False
        self._phase1_result  = None

        # Monte Carlo results
        self._mc_scatter   = None
        self._mc_ellipse   = None
        self._mc_cep       = 0.0
        self._kde_contours = None
        self._mc_running   = False

        # Phase 2 / live tracking
        self._p2_ellipse    = None
        self._phase2_active = False

        # System statuses
        self._koinobori_status    = "Disconnected"
        self._gpv_last_fetch_time = "N/A"

        # View toggles
        self._show_kde     = True
        self._show_cep     = True
        self._show_scatter = True
        self._show_burnout = True
        self._show_apogee  = True

        # Real-time wind
        self._wind_profile = [
            {"alt_m": 0.0, "speed_ms": 4.0, "dir_deg": 100.0},
            {"alt_m": 500.0, "speed_ms": 8.0, "dir_deg": 90.0}
        ]
        self._gust_speed       = 0.0

        # Unified simulation result (full worker payload dict)
        self._simulation_result = None

        # Wind history — per-altitude rolling 60-sample buffers at ~1 Hz.
        # Structure: dict[alt_m → deque(maxlen=WIND_HISTORY_MAX_SAMPLES)].
        # Each deque entry is a dict:
        #   {"ts": float,       monotonic timestamp (s)
        #    "speed_ms": float, wind speed (m/s)
        #    "dir_deg":  float} meteorological direction (° FROM which wind blows)
        #
        # 3 m (SURFACE_ALT): updated every second from the hardware anemometer.
        # 10 m / 150 m / 300 m / 600 m: updated when a new wind profile arrives
        #   (i.e. after each simulation run via append_wind_nodes).  Between runs
        #   these deques hold the last known API values and are NOT padded.
        self._wind_history: dict[float, deque] = {
            alt: deque(maxlen=WIND_HISTORY_MAX_SAMPLES) for alt in _WIND_SAMPLE_ALTS
        }

        # Zero-Order Hold (ZOH) cache — stores the last VALID sample per altitude.
        # Unlike the rolling deque (which may become empty before data arrives),
        # this dict retains its value indefinitely until overwritten by a new
        # valid reading.  None = no valid reading has ever arrived for this alt.
        # Only updated when speed_ms and dir_deg are present and not NaN.
        self._wind_zoh: dict[float, dict | None] = {
            alt: None for alt in _WIND_SAMPLE_ALTS
        }

        # CEP probability — UI-driven percentile; changing it redraws, not re-simulates
        self._cep_probability: float = 90.0

        # Overlay display parameters — mirrors wind_uncertainty; None until set
        self._wind_uncertainty_display = _ff("wind_uncertainty")
        # Stores the raw MC scatter as a numpy array; None until first simulation.
        self._cached_mc_scatter        = None

        # Moment of Inertia (kg·m²) — populated by set_moi() after .rkt parsing.
        # Zero until an .rkt file is loaded; not required for the RUN interlock.
        self._moi_roll:  float = 0.0   # Ixx — about longitudinal axis
        self._moi_pitch: float = 0.0   # Iyy — about lateral Y axis
        self._moi_yaw:   float = 0.0   # Izz — about lateral Z axis

        # Aerodynamics & motor (advanced settings — see AdvancedSettingsDialog).
        # Defaults assume a Black-Powder Estes-class motor; values are surfaced
        # bidirectionally to the dialog so the operator can override per flight.
        self._power_on_cd:              float = 0.45
        self._power_off_cd:             float = 0.40
        self._motor_isp:                float = 80.0    # s     (Black Powder)
        self._motor_propellant_density: float = 1700.0  # kg/m³ (Black Powder)

        # Mach-dependent drag curves (Phase C).  ``None`` means "use the scalar
        # above"; once an operator loads a (Mach, Cd) CSV via the Advanced
        # Settings dialog, the curve replaces the scalar at simulation time.
        # Stored as ``list[tuple[float, float]]`` sorted by ascending Mach.
        self._cd_curve_power_on:  Optional[list[tuple[float, float]]] = None
        self._cd_curve_power_off: Optional[list[tuple[float, float]]] = None

        # Phase B wind baseline — locked after a successful Phase A run.
        # None until set_wind_lock() is called (CEP <= target_radius).
        self._locked_mu:    tuple[float, float] | None = None   # (u_E, v_N) m/s
        self._locked_sigma: float                      = 0.0    # fractional 1-σ

        # Current human-readable tolerance status string (compared on every
        # tick to avoid emitting tolerance_status_changed on every second).
        self._tolerance_status: str = ""

        # Simulation busy flag — True while a SimulationWorker thread is live.
        self._is_calculating: bool = False

        # Two-stage rendering: nominal result available before MC completes
        self._nominal_result:      dict | None = None
        # MC run progress 0-100; 0 = idle / complete
        self._progress_percentage: int         = 0

        # Rocket geometry parameters — None until explicitly loaded from Rocket.json.
        # No silent numeric defaults: the operator MUST supply all values before
        # the RUN button becomes active (enforced by _check_readiness / interlock).
        _f = lambda k: (float(cfg[k]) if k in cfg else None)
        self._rocket_dry_mass  = _f("rocket_dry_mass")   # kg
        self._rocket_cg        = _f("rocket_cg")         # m from nose
        self._rocket_length    = _f("rocket_length")     # m
        self._rocket_diameter  = _f("rocket_diameter")   # m (= 2 × radius)
        self._nose_length      = _f("nose_length")       # m
        self._fin_root_chord   = _f("fin_root_chord")    # m
        self._fin_tip_chord    = _f("fin_tip_chord")     # m
        self._fin_span         = _f("fin_span")          # m
        self._fin_position     = _f("fin_position")      # m from nose
        self._motor_cg_pos     = None                    # m from nose
        self._motor_dry_mass   = None                    # kg
        self._parachute_cd     = None                    # dimensionless
        self._parachute_area   = None                    # m²
        self._parachute_lag    = None                    # s
        self._backfire_delay   = None                    # s

        # Launch settings — pre-filled with safe operational defaults so the
        # RUN button interlock does NOT wait for these to be entered.
        self._launch_angle: float = float(cfg.get("launch_angle", 85.0))  # degrees
        self._launch_rail:  float = float(cfg.get("launch_rail",   1.0))  # m

        # Flight mode — mission profile selected by the operator
        self._flight_mode: str = str(cfg.get("flight_mode", "Altitude"))

        # Interlock readiness flag — updated by _check_readiness() on every setter
        self._is_ready: bool = False
        self._check_readiness()

    # ── Parameter readiness interlock ─────────────────────────────────────────

    @property
    def is_ready_to_run(self) -> bool:
        """True when all 15 critical geometry + recovery parameters are non-None.

        Motor thrust data (thrust_data, motor_burn_time) lives in
        AppWindow._motor_thrust_data and is checked separately by the controller
        via ``getattr(w, '_motor_thrust_data', None)``.
        """
        return self._is_ready

    def _check_readiness(self) -> None:
        """Re-evaluate and broadcast the parameter readiness state.

        Called by every critical setter and once at the end of __init__.
        Only emits sig_ready_state_changed when the boolean result flips,
        preventing per-keystroke signal floods.

        Required fields (19–20 total):
          - 15 rocket geometry + recovery fields (require Rocket.json load)
          -  2 launch-site coordinates           (require GPS / manual entry)
          -  2 MC uncertainty params             (set by UI spinboxes)
          -  1 landing-zone target radius        (required UNLESS Free Mode)

        Motor thrust data (thrust_data, motor_burn_time) lives in
        AppWindow._motor_thrust_data and is checked separately.
        """
        is_free_mode = "free" in str(self._flight_mode).lower()

        ready = all(v is not None for v in (
            # ── Rocket geometry (Rocket.json) ─────────────────────────────
            self._rocket_dry_mass,
            self._rocket_cg,
            self._rocket_length,
            self._rocket_diameter,
            self._nose_length,
            self._fin_root_chord,
            self._fin_tip_chord,
            self._fin_span,
            self._fin_position,
            self._motor_cg_pos,
            self._motor_dry_mass,
            self._parachute_cd,
            self._parachute_area,
            self._parachute_lag,
            self._backfire_delay,
            # ── Launch site (GPS / spinbox) ───────────────────────────────
            self._launch_lat,
            self._launch_lon,
            # ── Simulation uncertainty params (spinboxes) ─────────────────
            self._wind_uncertainty,
            self._thrust_uncertainty,
        ))
        # Target radius is only required outside Free Mode — the input is
        # disabled in Free Mode so it will naturally be None.
        if not is_free_mode:
            ready = ready and (self._target_radius is not None)

        if ready != self._is_ready:
            self._is_ready = ready
            self.sig_ready_state_changed.emit(ready)

    # ── Moment of Inertia ─────────────────────────────────────────────────────

    def set_moi(self, ixx: float, iyy: float, izz: float) -> None:
        """Store system MoI (kg·m²) and broadcast moi_updated signal.

        Called by the controller after a successful .rkt parse.  The three
        values correspond to roll (Ixx), pitch (Iyy), and yaw (Izz) moments
        about the system CG.
        """
        self._moi_roll  = float(ixx)
        self._moi_pitch = float(iyy)
        self._moi_yaw   = float(izz)
        self.moi_updated.emit(self._moi_roll, self._moi_pitch, self._moi_yaw)

    @property
    def moi_roll(self) -> float:
        """Roll moment of inertia Ixx about the longitudinal axis (kg·m²)."""
        return self._moi_roll

    @property
    def moi_pitch(self) -> float:
        """Pitch moment of inertia Iyy about the lateral Y axis (kg·m²)."""
        return self._moi_pitch

    @property
    def moi_yaw(self) -> float:
        """Yaw moment of inertia Izz about the lateral Z axis (kg·m²)."""
        return self._moi_yaw

    # ── Simulation configuration ───────────────────────────────────────────────

    @Property(float, notify=wind_uncertainty_changed)
    def wind_uncertainty(self) -> float:
        return self._wind_uncertainty

    @wind_uncertainty.setter
    def wind_uncertainty(self, value: float) -> None:
        value = float(value)
        if self._wind_uncertainty != value:
            self._wind_uncertainty = value
            self.wind_uncertainty_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=thrust_uncertainty_changed)
    def thrust_uncertainty(self) -> float:
        return self._thrust_uncertainty

    @thrust_uncertainty.setter
    def thrust_uncertainty(self, value: float) -> None:
        value = float(value)
        if self._thrust_uncertainty != value:
            self._thrust_uncertainty = value
            self.thrust_uncertainty_changed.emit(value)
            self._check_readiness()

    @Property(int, notify=landing_prob_changed)
    def landing_prob(self) -> int:
        return self._landing_prob

    @landing_prob.setter
    def landing_prob(self, value: int) -> None:
        value = int(value)
        if self._landing_prob != value:
            self._landing_prob = value
            self.landing_prob_changed.emit(value)

    @Property(int, notify=mc_n_runs_changed)
    def mc_n_runs(self) -> int:
        return self._mc_n_runs

    @mc_n_runs.setter
    def mc_n_runs(self, value: int) -> None:
        value = int(value)
        if self._mc_n_runs != value:
            self._mc_n_runs = value
            self.mc_n_runs_changed.emit(value)

    # ── Launch site ────────────────────────────────────────────────────────────

    @Property(float, notify=magnetic_declination_changed)
    def magnetic_declination(self) -> float:
        return self._magnetic_declination

    @magnetic_declination.setter
    def magnetic_declination(self, value: float) -> None:
        value = float(value)
        if self._magnetic_declination != value:
            self._magnetic_declination = value
            self.magnetic_declination_changed.emit(value)

    @Property(list, notify=offline_map_extent_changed)
    def offline_map_extent(self) -> list:
        return self._offline_map_extent

    @offline_map_extent.setter
    def offline_map_extent(self, value: list) -> None:
        if self._offline_map_extent != value:
            self._offline_map_extent = value
            self.offline_map_extent_changed.emit(value)

    @Property(float, notify=launch_lat_changed)
    def launch_lat(self) -> float:
        return self._launch_lat

    @launch_lat.setter
    def launch_lat(self, value: float) -> None:
        value = float(value)
        if self._launch_lat != value:
            self._launch_lat = value
            self.launch_lat_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=launch_lon_changed)
    def launch_lon(self) -> float:
        return self._launch_lon

    @launch_lon.setter
    def launch_lon(self, value: float) -> None:
        value = float(value)
        if self._launch_lon != value:
            self._launch_lon = value
            self.launch_lon_changed.emit(value)
            self._check_readiness()

    # ── Rocket / flight parameters ─────────────────────────────────────────────

    @Property(float, notify=mass_changed)
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        value = float(value)
        if self._mass != value:
            self._mass = value
            self.mass_changed.emit(value)

    @Property(float, notify=drag_coeff_changed)
    def drag_coeff(self) -> float:
        return self._drag_coeff

    @drag_coeff.setter
    def drag_coeff(self, value: float) -> None:
        value = float(value)
        if self._drag_coeff != value:
            self._drag_coeff = value
            self.drag_coeff_changed.emit(value)

    # ── Aerodynamics & Motor (advanced settings) ─────────────────────────────

    @Property(float, notify=power_on_cd_changed)
    def power_on_cd(self) -> float:
        """Airframe drag coefficient during powered (boost) phase."""
        return self._power_on_cd

    @power_on_cd.setter
    def power_on_cd(self, value: float) -> None:
        value = float(value)
        if self._power_on_cd != value:
            self._power_on_cd = value
            self.power_on_cd_changed.emit(value)

    @Property(float, notify=power_off_cd_changed)
    def power_off_cd(self) -> float:
        """Airframe drag coefficient during coast (motor off) phase."""
        return self._power_off_cd

    @power_off_cd.setter
    def power_off_cd(self, value: float) -> None:
        value = float(value)
        if self._power_off_cd != value:
            self._power_off_cd = value
            self.power_off_cd_changed.emit(value)

    @Property(float, notify=motor_isp_changed)
    def motor_isp(self) -> float:
        """Motor specific impulse (s).  Default 80 s assumes Black Powder."""
        return self._motor_isp

    @motor_isp.setter
    def motor_isp(self, value: float) -> None:
        value = float(value)
        if self._motor_isp != value:
            self._motor_isp = value
            self.motor_isp_changed.emit(value)

    @Property(float, notify=motor_propellant_density_changed)
    def motor_propellant_density(self) -> float:
        """Motor propellant bulk density (kg/m³).  Default 1700 = Black Powder."""
        return self._motor_propellant_density

    @motor_propellant_density.setter
    def motor_propellant_density(self, value: float) -> None:
        value = float(value)
        if self._motor_propellant_density != value:
            self._motor_propellant_density = value
            self.motor_propellant_density_changed.emit(value)

    # ── Mach-dependent Cd curves (Phase C) ───────────────────────────────────

    @Property(object, notify=cd_curve_power_on_changed)
    def cd_curve_power_on(self) -> Optional[list[tuple[float, float]]]:
        """Optional (Mach, Cd) curve used during the boost phase.

        ``None`` → simulation falls back to the scalar :attr:`power_on_cd`.
        Otherwise a list of ``(Mach, Cd)`` tuples sorted by ascending Mach,
        consumed directly by RocketPy's ``Rocket(power_on_drag=...)``.
        """
        return self._cd_curve_power_on

    @cd_curve_power_on.setter
    def cd_curve_power_on(
        self,
        value: Optional[list[tuple[float, float]]],
    ) -> None:
        # Normalise: treat empty list as "no curve loaded" so downstream
        # consumers only ever see None or a non-empty list.
        if value is not None and len(value) == 0:
            value = None
        if self._cd_curve_power_on != value:
            self._cd_curve_power_on = value
            self.cd_curve_power_on_changed.emit(value)

    @Property(object, notify=cd_curve_power_off_changed)
    def cd_curve_power_off(self) -> Optional[list[tuple[float, float]]]:
        """Optional (Mach, Cd) curve used during the coast phase.

        ``None`` → simulation falls back to the scalar :attr:`power_off_cd`.
        """
        return self._cd_curve_power_off

    @cd_curve_power_off.setter
    def cd_curve_power_off(
        self,
        value: Optional[list[tuple[float, float]]],
    ) -> None:
        if value is not None and len(value) == 0:
            value = None
        if self._cd_curve_power_off != value:
            self._cd_curve_power_off = value
            self.cd_curve_power_off_changed.emit(value)

    @Property(float, notify=ref_area_changed)
    def ref_area(self) -> float:
        return self._ref_area

    @ref_area.setter
    def ref_area(self, value: float) -> None:
        value = float(value)
        if self._ref_area != value:
            self._ref_area = value
            self.ref_area_changed.emit(value)

    @Property(float, notify=target_radius_changed)
    def target_radius(self) -> float:
        return self._target_radius

    @target_radius.setter
    def target_radius(self, value: float) -> None:
        value = float(value)
        if self._target_radius != value:
            self._target_radius = value
            self.target_radius_changed.emit(value)
            self._check_readiness()

    @Property(str, notify=operation_mode_changed)
    def operation_mode(self) -> str:
        return self._operation_mode

    @operation_mode.setter
    def operation_mode(self, value: str) -> None:
        value = str(value)
        if self._operation_mode != value:
            self._operation_mode = value
            self.operation_mode_changed.emit(value)

    # ── Simulation results ─────────────────────────────────────────────────────

    @Property(float, notify=land_lat_changed)
    def land_lat(self) -> float:
        return self._land_lat

    @land_lat.setter
    def land_lat(self, value: float) -> None:
        value = float(value)
        if self._land_lat != value:
            self._land_lat = value
            self.land_lat_changed.emit(value)

    @Property(float, notify=land_lon_changed)
    def land_lon(self) -> float:
        return self._land_lon

    @land_lon.setter
    def land_lon(self, value: float) -> None:
        value = float(value)
        if self._land_lon != value:
            self._land_lon = value
            self.land_lon_changed.emit(value)

    @Property(float, notify=r90_radius_changed)
    def r90_radius(self) -> float:
        return self._r90_radius

    @r90_radius.setter
    def r90_radius(self, value: float) -> None:
        value = float(value)
        if self._r90_radius != value:
            self._r90_radius = value
            self.r90_radius_changed.emit(value)

    @Property(bool, notify=has_sim_result_changed)
    def has_sim_result(self) -> bool:
        return self._has_sim_result

    @has_sim_result.setter
    def has_sim_result(self, value: bool) -> None:
        value = bool(value)
        if self._has_sim_result != value:
            self._has_sim_result = value
            self.has_sim_result_changed.emit(value)

    @Property(int, notify=current_playback_index_changed)
    def current_playback_index(self) -> int:
        return self._current_playback_index

    @current_playback_index.setter
    def current_playback_index(self, value: int) -> None:
        value = int(value)
        if self._current_playback_index != value:
            self._current_playback_index = value
            self.current_playback_index_changed.emit(value)

    @Property(object, notify=phase1_result_changed)
    def phase1_result(self) -> Optional[dict]:
        """Dictionary containing Phase 1 results."""
        return self._phase1_result

    @phase1_result.setter
    def phase1_result(self, value: Optional[dict]) -> None:
        self._phase1_result = value
        self.phase1_result_changed.emit(value)

    # ── Monte Carlo results ────────────────────────────────────────────────────

    @Property(object, notify=mc_scatter_changed)
    def mc_scatter(self) -> Optional[object]:
        """Raw scatter data of Monte Carlo results."""
        return self._mc_scatter

    @mc_scatter.setter
    def mc_scatter(self, value: Optional[object]) -> None:
        self._mc_scatter = value
        self.mc_scatter_changed.emit(value)

    @Property(object, notify=mc_ellipse_changed)
    def mc_ellipse(self) -> Optional[dict]:
        """Dictionary containing parameters of the CEP error ellipse."""
        return self._mc_ellipse

    @mc_ellipse.setter
    def mc_ellipse(self, value: Optional[dict]) -> None:
        self._mc_ellipse = value
        self.mc_ellipse_changed.emit(value)

    @Property(float, notify=mc_cep_changed)
    def mc_cep(self) -> float:
        return self._mc_cep

    @mc_cep.setter
    def mc_cep(self, value: float) -> None:
        value = float(value)
        if self._mc_cep != value:
            self._mc_cep = value
            self.mc_cep_changed.emit(value)

    @Property(object, notify=kde_contours_changed)
    def kde_contours(self) -> Optional[list]:
        """List of KDE contour dictionaries."""
        return self._kde_contours

    @kde_contours.setter
    def kde_contours(self, value: Optional[list]) -> None:
        self._kde_contours = value
        self.kde_contours_changed.emit(value)

    @Property(bool, notify=mc_running_changed)
    def mc_running(self) -> bool:
        return self._mc_running

    @mc_running.setter
    def mc_running(self, value: bool) -> None:
        value = bool(value)
        if self._mc_running != value:
            self._mc_running = value
            self.mc_running_changed.emit(value)

    # ── Phase 2 / live tracking ────────────────────────────────────────────────

    @Property(object, notify=p2_ellipse_changed)
    def p2_ellipse(self) -> Optional[dict]:
        """Dictionary containing parameters of the Phase 2 ellipse."""
        return self._p2_ellipse

    @p2_ellipse.setter
    def p2_ellipse(self, value: Optional[dict]) -> None:
        self._p2_ellipse = value
        self.p2_ellipse_changed.emit(value)

    @Property(bool, notify=phase2_active_changed)
    def phase2_active(self) -> bool:
        return self._phase2_active

    @phase2_active.setter
    def phase2_active(self, value: bool) -> None:
        value = bool(value)
        if self._phase2_active != value:
            self._phase2_active = value
            self.phase2_active_changed.emit(value)

    # ── Real-time wind ─────────────────────────────────────────────────────────





    @Property(object, notify=wind_profile_changed)
    def wind_profile(self) -> list:
        return self._wind_profile

    @wind_profile.setter
    def wind_profile(self, value: list) -> None:
        self._wind_profile = value
        self.wind_profile_changed.emit(value)
        self.wind_profile_data_changed.emit(value)

    @Property(object, notify=wind_profile_data_changed)
    def wind_profile_data(self) -> list:
        return self._wind_profile

    @wind_profile_data.setter
    def wind_profile_data(self, value: list) -> None:
        self.wind_profile = value

    @Property(float, notify=gust_speed_changed)
    def gust_speed(self) -> float:
        return self._gust_speed

    @gust_speed.setter
    def gust_speed(self, value: float) -> None:
        value = float(value)
        if self._gust_speed != value:
            self._gust_speed = value
            self.gust_speed_changed.emit(value)

    # ── Wind history ──────────────────────────────────────────────────────────

    @Property(object, notify=wind_history_updated)
    def wind_history(self) -> dict:
        """Per-altitude rolling wind history.

        Returns a ``dict[float, deque]`` keyed by altitude in metres AGL:
            {3.0: deque, 10.0: deque, 150.0: deque, 300.0: deque, 600.0: deque}

        Each deque contains up to WIND_HISTORY_MAX_SAMPLES samples of the form:
            {"ts": float, "speed_ms": float, "dir_deg": float}
        sorted oldest → newest (deque order).

        The 3 m deque updates every second from the hardware anemometer.
        The upper deques update whenever append_wind_nodes() is called (typically
        once per simulation run).
        """
        return self._wind_history

    def wind_history_for_alt(self, alt_m: float) -> deque:
        """Return the rolling sample deque for a single altitude.

        Returns an empty deque if *alt_m* is not one of the five diagnostic
        altitudes; never raises.
        """
        return self._wind_history.get(float(alt_m), deque())

    def append_wind_reading(self, speed: float, direction: float) -> None:
        """Append one surface anemometer sample (3 m AGL) to the history.

        Writes exclusively to the _SURFACE_ALT (3 m) deque.  Upper-altitude
        deques are updated separately by append_wind_nodes() when a new wind
        profile arrives from the simulation.

        Also updates the ZOH cache for 3 m so get_wind_zoh(3.0) always
        returns the most recent anemometer reading.

        Args:
            speed:     Wind speed in m/s (non-negative).
            direction: Meteorological direction in degrees FROM which wind blows.
        """
        sample = {
            "ts":       time.monotonic(),
            "speed_ms": float(speed),
            "dir_deg":  float(direction),
        }
        self._wind_history[_SURFACE_ALT].append(sample)
        self._wind_zoh[_SURFACE_ALT] = sample     # ZOH: persist last valid reading
        self.wind_history_updated.emit(self._wind_history)
        self.wind_updated.emit()          # lightweight ping for simple observers


    def append_wind_nodes(self, nodes: list) -> None:
        """Append a full 5-altitude wind snapshot to the history.

        Intended to be called once per simulation run with the ``wind_nodes``
        list produced by ``sample_wind_nodes``.  Each node is written to the
        deque for its altitude; altitudes not in _WIND_SAMPLE_ALTS are silently
        ignored so callers do not need to filter.

        Zero-Order Hold (ZOH) behaviour
        ---------------------------------
        If a node's ``speed_ms`` or ``dir_deg`` is ``None`` or ``NaN``, that
        node is silently skipped — the deque and ZOH cache retain their last
        valid value ("hold").  This prevents a transient API gap from injecting
        a zero or garbage reading into the history.

        Args:
            nodes: List of node dicts, each containing at least:
                   ``alt_m`` (float), ``speed_ms`` (float), ``dir_deg`` (float).
                   Typically the direct output of ``core.wind_model.sample_wind_nodes``.
        """
        ts = time.monotonic()
        for node in nodes:
            alt = float(node.get("alt_m", -1.0))
            if alt not in self._wind_history:
                continue
            raw_speed = node.get("speed_ms")
            raw_dir   = node.get("dir_deg")
            # ZOH: skip this tick if data is absent or not a finite number.
            if raw_speed is None or raw_dir is None:
                continue
            try:
                spd = float(raw_speed)
                drc = float(raw_dir)
            except (TypeError, ValueError):
                continue
            if math.isnan(spd) or math.isnan(drc):
                continue
            entry = {"ts": ts, "speed_ms": spd, "dir_deg": drc}
            self._wind_history[alt].append(entry)
            self._wind_zoh[alt] = entry    # update ZOH cache with last valid value
        self.wind_history_updated.emit(self._wind_history)
        self.wind_updated.emit()

    def get_wind_zoh(self, alt_m: float) -> "dict | None":
        """Return the last valid wind reading for *alt_m* using Zero-Order Hold.

        Unlike ``wind_history_for_alt`` (which returns the full rolling deque),
        this method returns a single sample dict — the most recent one for which
        both speed_ms and dir_deg were finite — or ``None`` if no valid reading
        has ever arrived for this altitude.

        The ZOH value persists across deque rollovers: even after WIND_HISTORY_MAX_SAMPLES samples
        have been replaced, the cache still holds the last known-good reading.

        Returns:
            dict with keys ``ts``, ``speed_ms``, ``dir_deg`` — or ``None``.
        """
        return self._wind_zoh.get(float(alt_m))

    # ── Phase B wind baseline + O(1) tolerance monitoring ─────────────────────

    def set_wind_lock(self, mu_u: float, mu_v: float, sigma: float) -> None:
        """Lock the Phase A wind baseline for Phase B GO/NO-GO evaluation.

        Called by SimController when Phase A completes with CEP ≤ target_radius.
        The (mu_u, mu_v) vector is the nominal surface wind used by the MC run,
        expressed as East/North components.  sigma is the fractional 1-σ wind
        uncertainty parameter passed to the worker.

        Resets the tolerance status so the first Phase B tick always emits
        tolerance_status_changed and the indicator updates immediately.
        """
        self._locked_mu     = (float(mu_u), float(mu_v))
        self._locked_sigma  = float(sigma)
        self._tolerance_status = ""

    def check_tolerance(self, speed: float, direction: float) -> None:
        """O(1) Phase B evaluation: compare live wind against the locked baseline.

        Called every wind tick while Phase 2 is active.  Uses
        core.monte_carlo.evaluate_wind_within_bounds which runs in O(1) time
        (three arithmetic ops; no iteration).

        Signals emitted
        ---------------
        tolerance_exceeded       — every tick the live wind is outside bounds.
        tolerance_status_changed — only on GO ↔ NO-GO transitions, preventing
                                   per-tick visual chatter on the indicator.
        """
        if not self._phase2_active or self._locked_mu is None:
            return

        from core.monte_carlo import evaluate_wind_within_bounds

        live_u = speed * math.sin(math.radians(direction))
        live_v = speed * math.cos(math.radians(direction))
        in_bounds = evaluate_wind_within_bounds(
            live_u, live_v,
            self._locked_mu[0], self._locked_mu[1],
            self._locked_sigma,
        )

        if not in_bounds:
            mu_spd = math.hypot(self._locked_mu[0], self._locked_mu[1])
            self.tolerance_exceeded.emit(
                f"Live {speed:.1f} m/s vs locked {mu_spd:.1f} m/s "
                f"(σ={self._locked_sigma:.2f})"
            )
            new_status = "⚠  NO-GO"
        else:
            new_status = "✓  GO"

        if new_status != self._tolerance_status:
            self._tolerance_status = new_status
            self.tolerance_status_changed.emit(new_status)

    # ── CEP probability ────────────────────────────────────────────────────────

    @Property(float, notify=cep_probability_changed)
    def cep_probability(self) -> float:
        return self._cep_probability

    @cep_probability.setter
    def cep_probability(self, value: float) -> None:
        value = float(value)
        if self._cep_probability != value:
            self._cep_probability = value
            self.cep_probability_changed.emit(value)
            self.needs_partial_redraw.emit()   # overlay-only recompute from cached scatter

    # ── Unified simulation result ──────────────────────────────────────────────

    @Property(object, notify=simulation_result_changed)
    def simulation_result(self) -> Optional[dict]:
        """Complete payload dict from the last successful SimulationWorker run.

        Keys: cancelled, has_sim_result, t_vals, x_vals, y_vals, z_vals,
              apogee_m, hang_time, impact_x, impact_y, r_horiz,
              scatter, r_N_radius, cep, ellipse, kde_contours,
              n_runs, landing_prob.

        Setting this property always emits both simulation_result_changed
        (with the new dict) and needs_redraw (no payload), so any connected
        canvas or overlay repaints automatically.
        """
        return self._simulation_result

    @simulation_result.setter
    def simulation_result(self, value: Optional[dict]) -> None:
        self._simulation_result = value
        self.simulation_result_changed.emit(value)
        # Broadcast a unified redraw notification after every new result.
        self.needs_redraw.emit()


    # ── View Toggles ───────────────────────────────────────────────────────────

    @Property(bool, notify=show_kde_changed)
    def show_kde(self) -> bool:
        return self._show_kde

    @show_kde.setter
    def show_kde(self, value: bool) -> None:
        if self._show_kde != value:
            self._show_kde = value
            self.show_kde_changed.emit(value)
            self.needs_redraw.emit()

    @Property(bool, notify=show_cep_changed)
    def show_cep(self) -> bool:
        return self._show_cep

    @show_cep.setter
    def show_cep(self, value: bool) -> None:
        if self._show_cep != value:
            self._show_cep = value
            self.show_cep_changed.emit(value)
            self.needs_redraw.emit()

    @Property(bool, notify=show_scatter_changed)
    def show_scatter(self) -> bool:
        return self._show_scatter

    @show_scatter.setter
    def show_scatter(self, value: bool) -> None:
        if self._show_scatter != value:
            self._show_scatter = value
            self.show_scatter_changed.emit(value)
            self.needs_redraw.emit()

    @Property(bool, notify=show_burnout_changed)
    def show_burnout(self) -> bool:
        return self._show_burnout

    @show_burnout.setter
    def show_burnout(self, value: bool) -> None:
        if self._show_burnout != value:
            self._show_burnout = value
            self.show_burnout_changed.emit(value)
            self.needs_redraw.emit()

    @Property(bool, notify=show_apogee_changed)
    def show_apogee(self) -> bool:
        return self._show_apogee

    @show_apogee.setter
    def show_apogee(self, value: bool) -> None:
        if self._show_apogee != value:
            self._show_apogee = value
            self.show_apogee_changed.emit(value)
            self.needs_redraw.emit()

    # ── Overlay display parameters ─────────────────────────────────────────────

    @Property(float, notify=wind_uncertainty_display_changed)
    def wind_uncertainty_display(self) -> float:
        """Scaling factor for the wind-induced dispersion display overlay."""
        return self._wind_uncertainty_display

    @wind_uncertainty_display.setter
    def wind_uncertainty_display(self, value: float) -> None:
        value = float(value)
        if self._wind_uncertainty_display != value:
            self._wind_uncertainty_display = value
            self.wind_uncertainty_display_changed.emit(value)

    @Property(object, notify=cached_mc_scatter_changed)
    def cached_mc_scatter(self) -> Optional[object]:
        """Raw MC landing scatter as numpy.ndarray of shape (N, 2).

        Columns: [x_east_m, y_north_m] in the ENU metric frame.
        None until the first simulation run completes.  Never cleared by a
        partial redraw — the controller only replaces this on a new full run.
        """
        return self._cached_mc_scatter

    @cached_mc_scatter.setter
    def cached_mc_scatter(self, value: Optional[object]) -> None:
        self._cached_mc_scatter = value
        self.cached_mc_scatter_changed.emit(value)

    # ── Simulation busy flag ───────────────────────────────────────────────────

    @Property(bool, notify=is_calculating_changed)
    def is_calculating(self) -> bool:
        """True while a SimulationWorker thread is live, False otherwise."""
        return self._is_calculating

    @is_calculating.setter
    def is_calculating(self, value: bool) -> None:
        value = bool(value)
        if self._is_calculating != value:
            self._is_calculating = value
            self.is_calculating_changed.emit(value)

    # ── Rocket geometry parameters ─────────────────────────────────────────────

    @Property(float, notify=rocket_dry_mass_changed)
    def rocket_dry_mass(self) -> float:
        """Airframe dry mass in kg (maps to 'airframe_mass' in workers.py)."""
        return self._rocket_dry_mass

    @rocket_dry_mass.setter
    def rocket_dry_mass(self, value: float) -> None:
        value = float(value)
        if self._rocket_dry_mass != value:
            self._rocket_dry_mass = value
            self.rocket_dry_mass_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=rocket_cg_changed)
    def rocket_cg(self) -> float:
        """Airframe centre of gravity from nose in m (maps to 'airframe_cg')."""
        return self._rocket_cg

    @rocket_cg.setter
    def rocket_cg(self, value: float) -> None:
        value = float(value)
        if self._rocket_cg != value:
            self._rocket_cg = value
            self.rocket_cg_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=rocket_length_changed)
    def rocket_length(self) -> float:
        """Total airframe length in m (maps to 'airframe_len')."""
        return self._rocket_length

    @rocket_length.setter
    def rocket_length(self, value: float) -> None:
        value = float(value)
        if self._rocket_length != value:
            self._rocket_length = value
            self.rocket_length_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=rocket_diameter_changed)
    def rocket_diameter(self) -> float:
        """Body diameter in m (= 2 × body radius; maps to 'radius' × 0.5 in workers)."""
        return self._rocket_diameter

    @rocket_diameter.setter
    def rocket_diameter(self, value: float) -> None:
        value = float(value)
        if self._rocket_diameter != value:
            self._rocket_diameter = value
            self.rocket_diameter_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=nose_length_changed)
    def nose_length(self) -> float:
        """Nose cone length in m (maps to 'nose_len')."""
        return self._nose_length

    @nose_length.setter
    def nose_length(self, value: float) -> None:
        value = float(value)
        if self._nose_length != value:
            self._nose_length = value
            self.nose_length_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=fin_root_chord_changed)
    def fin_root_chord(self) -> float:
        """Fin root chord length in m (maps to 'fin_root')."""
        return self._fin_root_chord

    @fin_root_chord.setter
    def fin_root_chord(self, value: float) -> None:
        value = float(value)
        if self._fin_root_chord != value:
            self._fin_root_chord = value
            self.fin_root_chord_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=fin_tip_chord_changed)
    def fin_tip_chord(self) -> float:
        """Fin tip chord length in m (maps to 'fin_tip')."""
        return self._fin_tip_chord

    @fin_tip_chord.setter
    def fin_tip_chord(self, value: float) -> None:
        value = float(value)
        if self._fin_tip_chord != value:
            self._fin_tip_chord = value
            self.fin_tip_chord_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=fin_span_changed)
    def fin_span(self) -> float:
        """Fin semi-span in m (maps to 'fin_span')."""
        return self._fin_span

    @fin_span.setter
    def fin_span(self, value: float) -> None:
        value = float(value)
        if self._fin_span != value:
            self._fin_span = value
            self.fin_span_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=fin_position_changed)
    def fin_position(self) -> float:
        """Fin leading-edge position from nose in m (maps to 'fin_pos')."""
        return self._fin_position

    @fin_position.setter
    def fin_position(self, value: float) -> None:
        value = float(value)
        if self._fin_position != value:
            self._fin_position = value
            self.fin_position_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=motor_cg_pos_changed)
    def motor_cg_pos(self) -> float:
        """Motor centre of gravity from nose in m (maps to 'motor_pos')."""
        return self._motor_cg_pos

    @motor_cg_pos.setter
    def motor_cg_pos(self, value: float) -> None:
        value = float(value)
        if self._motor_cg_pos != value:
            self._motor_cg_pos = value
            self.motor_cg_pos_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=motor_dry_mass_changed)
    def motor_dry_mass(self) -> float:
        """Motor dry (post-burn) mass in kg (maps to 'motor_dry_mass')."""
        return self._motor_dry_mass

    @motor_dry_mass.setter
    def motor_dry_mass(self, value: float) -> None:
        value = float(value)
        if self._motor_dry_mass != value:
            self._motor_dry_mass = value
            self.motor_dry_mass_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=parachute_cd_changed)
    def parachute_cd(self) -> float:
        """Parachute drag coefficient (dimensionless; maps to 'para_cd')."""
        return self._parachute_cd

    @parachute_cd.setter
    def parachute_cd(self, value: float) -> None:
        value = float(value)
        if self._parachute_cd != value:
            self._parachute_cd = value
            self.parachute_cd_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=parachute_area_changed)
    def parachute_area(self) -> float:
        """Parachute reference area in m² (maps to 'para_area')."""
        return self._parachute_area

    @parachute_area.setter
    def parachute_area(self, value: float) -> None:
        value = float(value)
        if self._parachute_area != value:
            self._parachute_area = value
            self.parachute_area_changed.emit(value)
            self._check_readiness()

    @Property(float, notify=parachute_lag_changed)
    def parachute_lag(self) -> float:
        """Parachute deployment lag in seconds (maps to 'para_lag')."""
        return self._parachute_lag

    @parachute_lag.setter
    def parachute_lag(self, value: float) -> None:
        value = float(value)
        if self._parachute_lag != value:
            self._parachute_lag = value
            self.parachute_lag_changed.emit(value)
            self._check_readiness()

    @Property(object, notify=backfire_delay_changed)
    def backfire_delay(self) -> Optional[float]:
        """Ejection charge fires this many seconds after motor burnout (maps to 'backfire_delay')."""
        return self._backfire_delay

    @backfire_delay.setter
    def backfire_delay(self, value: Optional[float]) -> None:
        if value is not None:
            value = float(value)
        if self._backfire_delay != value:
            self._backfire_delay = value
            # Fallback to -9999.0 for signals to match UI uninitialized value
            self.backfire_delay_changed.emit(value if value is not None else -9999.0)
            self._check_readiness()



    # ── External System Status ─────────────────────────────────────────────────

    @Property(str, notify=koinobori_status_changed)
    def koinobori_status(self) -> str:
        return self._koinobori_status

    @koinobori_status.setter
    def koinobori_status(self, value: str) -> None:
        value = str(value)
        if self._koinobori_status != value:
            self._koinobori_status = value
            self.koinobori_status_changed.emit(value)

    @Property(str, notify=gpv_last_fetch_time_changed)
    def gpv_last_fetch_time(self) -> str:
        return self._gpv_last_fetch_time

    @gpv_last_fetch_time.setter
    def gpv_last_fetch_time(self, value: str) -> None:
        value = str(value)
        if self._gpv_last_fetch_time != value:
            self._gpv_last_fetch_time = value
            self.gpv_last_fetch_time_changed.emit(value)

    # ── Engine / motor loader ─────────────────────────────────────────────────

    def load_engine(
        self,
        file_path: str,
        avg_thrust: float,
        burn_time: float,
        curve_data: list,
    ) -> None:
        """Store motor metadata and broadcast it to all interested views.

        Called by whatever UI component parses the motor CSV.  The payload
        dict matches the engine_loaded signal's documented schema so receivers
        do not need to know the caller's internal representation.

        Parameters
        ----------
        file_path  : Absolute path of the source CSV file.
        avg_thrust : Mean thrust over the burn (N).
        burn_time  : Total burn duration (s).
        curve_data : Raw thrust curve as a list of [time_s, thrust_N] pairs.
        """
        payload: dict = {
            "file_path":  str(file_path),
            "avg_thrust": float(avg_thrust),
            "burn_time":  float(burn_time),
            "curve_data": list(curve_data),
        }
        self.engine_loaded.emit(payload)

    # ── Launch settings ────────────────────────────────────────────────────────

    @Property(float, notify=launch_angle_changed)
    def launch_angle(self) -> float:
        """Rail elevation angle in degrees (default 85.0; range 0–90)."""
        return self._launch_angle

    @launch_angle.setter
    def launch_angle(self, value: float) -> None:
        value = float(value)
        if self._launch_angle != value:
            self._launch_angle = value
            self.launch_angle_changed.emit(value)

    @Property(float, notify=launch_rail_changed)
    def launch_rail(self) -> float:
        """Launch rail length in metres (default 1.0)."""
        return self._launch_rail

    @launch_rail.setter
    def launch_rail(self, value: float) -> None:
        value = float(value)
        if self._launch_rail != value:
            self._launch_rail = value
            self.launch_rail_changed.emit(value)

    # ── Flight mode ────────────────────────────────────────────────────────────

    @Property(str, notify=flight_mode_changed)
    def flight_mode(self) -> str:
        """Mission profile (e.g. 'Altitude', 'Precision', 'Winged')."""
        return self._flight_mode

    @flight_mode.setter
    def flight_mode(self, value: str) -> None:
        value = str(value)
        if self._flight_mode != value:
            self._flight_mode = value
            self.flight_mode_changed.emit(value)
            self._check_readiness()

    @property
    def is_free_mode(self) -> bool:
        """True when the current flight_mode is the free/unconstrained profile."""
        fm = str(self._flight_mode)
        return "free" in fm.lower() or "自由" in fm

    # ── Two-stage rendering ────────────────────────────────────────────────────

    @Property(object, notify=nominal_result_changed)
    def nominal_result(self) -> Optional[dict]:
        """Nominal single-run payload; set before MC starts so views render early.

        Setting this property emits both nominal_result_changed (carrying the
        dict) and nominal_needs_redraw (no payload), so any connected canvas
        can either consume the data or just repaint on the lightweight signal.
        """
        return self._nominal_result

    @nominal_result.setter
    def nominal_result(self, value: Optional[dict]) -> None:
        self._nominal_result = value
        self.nominal_result_changed.emit(value)
        self.nominal_needs_redraw.emit()

    # ── MC progress ───────────────────────────────────────────────────────────

    @Property(int, notify=progress_changed)
    def progress_percentage(self) -> int:
        """Integer 0–100 tracking MC run completion.

        0 means idle or finished (not in progress).  Clamped to [0, 100].
        The equality guard suppresses spurious emissions when the value hasn't
        changed (e.g. two consecutive batches that round to the same percent).
        """
        return self._progress_percentage

    @progress_percentage.setter
    def progress_percentage(self, value: int) -> None:
        value = int(max(0, min(100, value)))
        if self._progress_percentage != value:
            self._progress_percentage = value
            self.progress_changed.emit(value)
