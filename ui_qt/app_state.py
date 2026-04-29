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

from typing import Optional

from PySide6.QtCore import QObject, Signal, Property


class AppState(QObject):

    # ── Simulation configuration ───────────────────────────────────────────────
    wind_uncertainty_changed      = Signal(float)
    thrust_uncertainty_changed    = Signal(float)
    allowable_uncertainty_changed = Signal(float)
    landing_prob_changed          = Signal(int)
    mc_n_runs_changed             = Signal(int)

    # ── Launch site ────────────────────────────────────────────────────────────
    launch_lat_changed = Signal(float)
    launch_lon_changed = Signal(float)

    # ── Rocket / flight parameters ─────────────────────────────────────────────
    mass_changed           = Signal(float)
    drag_coeff_changed     = Signal(float)
    ref_area_changed       = Signal(float)
    target_radius_changed  = Signal(float)
    operation_mode_changed = Signal(str)

    # ── Simulation results ─────────────────────────────────────────────────────
    land_lat_changed       = Signal(float)
    land_lon_changed       = Signal(float)
    r90_radius_changed     = Signal(float)
    has_sim_result_changed = Signal(bool)
    phase1_result_changed  = Signal(object)

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
    surf_wind_speed_changed  = Signal(float)
    surf_wind_dir_changed    = Signal(float)
    upper_wind_speed_changed = Signal(float)
    upper_wind_dir_changed   = Signal(float)
    gust_speed_changed       = Signal(float)

    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self, config: Optional[dict] = None, parent=None) -> None:
        super().__init__(parent)
        cfg = config or {}

        # Simulation configuration
        self._wind_uncertainty      = float(cfg.get("wind_uncertainty",      0.20))
        self._thrust_uncertainty    = float(cfg.get("thrust_uncertainty",    0.05))
        self._allowable_uncertainty = float(cfg.get("allowable_uncertainty", 20.0))
        self._landing_prob          = int(cfg.get("landing_prob",            90))
        self._mc_n_runs             = int(cfg.get("mc_n_runs",               200))

        # Launch site
        self._launch_lat = float(cfg.get("launch_lat", 35.6828))
        self._launch_lon = float(cfg.get("launch_lon", 139.7590))

        # Rocket / flight parameters
        self._mass           = float(cfg.get("mass",          0.5))
        self._drag_coeff     = float(cfg.get("drag_coeff",    0.47))
        self._ref_area       = float(cfg.get("ref_area",      0.007854))
        self._target_radius  = float(cfg.get("target_radius", 25.0))
        self._operation_mode = str(cfg.get("operation_mode",  "Precision Landing"))

        # Simulation results
        self._land_lat       = self._launch_lat
        self._land_lon       = self._launch_lon
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

        # Real-time wind
        self._surf_wind_speed  = 0.0
        self._surf_wind_dir    = 0.0
        self._upper_wind_speed = 0.0
        self._upper_wind_dir   = 0.0
        self._gust_speed       = 0.0

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

    @Property(float, notify=thrust_uncertainty_changed)
    def thrust_uncertainty(self) -> float:
        return self._thrust_uncertainty

    @thrust_uncertainty.setter
    def thrust_uncertainty(self, value: float) -> None:
        value = float(value)
        if self._thrust_uncertainty != value:
            self._thrust_uncertainty = value
            self.thrust_uncertainty_changed.emit(value)

    @Property(float, notify=allowable_uncertainty_changed)
    def allowable_uncertainty(self) -> float:
        return self._allowable_uncertainty

    @allowable_uncertainty.setter
    def allowable_uncertainty(self, value: float) -> None:
        value = float(value)
        if self._allowable_uncertainty != value:
            self._allowable_uncertainty = value
            self.allowable_uncertainty_changed.emit(value)

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

    @Property(float, notify=launch_lat_changed)
    def launch_lat(self) -> float:
        return self._launch_lat

    @launch_lat.setter
    def launch_lat(self, value: float) -> None:
        value = float(value)
        if self._launch_lat != value:
            self._launch_lat = value
            self.launch_lat_changed.emit(value)

    @Property(float, notify=launch_lon_changed)
    def launch_lon(self) -> float:
        return self._launch_lon

    @launch_lon.setter
    def launch_lon(self, value: float) -> None:
        value = float(value)
        if self._launch_lon != value:
            self._launch_lon = value
            self.launch_lon_changed.emit(value)

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

    @Property(object, notify=phase1_result_changed)
    def phase1_result(self):
        return self._phase1_result

    @phase1_result.setter
    def phase1_result(self, value) -> None:
        self._phase1_result = value
        self.phase1_result_changed.emit(value)

    # ── Monte Carlo results ────────────────────────────────────────────────────

    @Property(object, notify=mc_scatter_changed)
    def mc_scatter(self):
        return self._mc_scatter

    @mc_scatter.setter
    def mc_scatter(self, value) -> None:
        self._mc_scatter = value
        self.mc_scatter_changed.emit(value)

    @Property(object, notify=mc_ellipse_changed)
    def mc_ellipse(self):
        return self._mc_ellipse

    @mc_ellipse.setter
    def mc_ellipse(self, value) -> None:
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
    def kde_contours(self):
        return self._kde_contours

    @kde_contours.setter
    def kde_contours(self, value) -> None:
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
    def p2_ellipse(self):
        return self._p2_ellipse

    @p2_ellipse.setter
    def p2_ellipse(self, value) -> None:
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

    @Property(float, notify=surf_wind_speed_changed)
    def surf_wind_speed(self) -> float:
        return self._surf_wind_speed

    @surf_wind_speed.setter
    def surf_wind_speed(self, value: float) -> None:
        value = float(value)
        if self._surf_wind_speed != value:
            self._surf_wind_speed = value
            self.surf_wind_speed_changed.emit(value)

    @Property(float, notify=surf_wind_dir_changed)
    def surf_wind_dir(self) -> float:
        return self._surf_wind_dir

    @surf_wind_dir.setter
    def surf_wind_dir(self, value: float) -> None:
        value = float(value)
        if self._surf_wind_dir != value:
            self._surf_wind_dir = value
            self.surf_wind_dir_changed.emit(value)

    @Property(float, notify=upper_wind_speed_changed)
    def upper_wind_speed(self) -> float:
        return self._upper_wind_speed

    @upper_wind_speed.setter
    def upper_wind_speed(self, value: float) -> None:
        value = float(value)
        if self._upper_wind_speed != value:
            self._upper_wind_speed = value
            self.upper_wind_speed_changed.emit(value)

    @Property(float, notify=upper_wind_dir_changed)
    def upper_wind_dir(self) -> float:
        return self._upper_wind_dir

    @upper_wind_dir.setter
    def upper_wind_dir(self, value: float) -> None:
        value = float(value)
        if self._upper_wind_dir != value:
            self._upper_wind_dir = value
            self.upper_wind_dir_changed.emit(value)

    @Property(float, notify=gust_speed_changed)
    def gust_speed(self) -> float:
        return self._gust_speed

    @gust_speed.setter
    def gust_speed(self, value: float) -> None:
        value = float(value)
        if self._gust_speed != value:
            self._gust_speed = value
            self.gust_speed_changed.emit(value)
