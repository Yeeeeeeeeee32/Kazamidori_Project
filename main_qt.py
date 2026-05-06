"""
main_qt.py
PySide6 entry point and top-level controller for the Kazamidori Project.

Responsibilities
----------------
1. Construct QApplication.
2. Build the shared AppState (cross-component data bus for computed results).
3. Show AppWindow (which owns its own reactive plot-state internally).
4. Wire RUN / STOP buttons -> SimulationWorker via SimController.
5. Route worker signals back to AppWindow public API and AppState properties.
6. Start the Qt event loop.

Architecture note
-----------------
AppWindow defines its own lightweight AppState (with a needs_redraw Signal)
to drive the live 3-D plot without polling.  The ui_qt.app_state.AppState
created here is the broader application data bus that holds final simulation
results and will be consumed by future views (map overlay, Phase 2 panel,
etc.).  These are intentionally separate objects; the controller mediates
writes to both.
"""

from __future__ import annotations

import math
import random
import sys
import time as _time

import numpy as np
from PySide6.QtCore import QObject, QTimer, Slot
from PySide6.QtWidgets import QApplication, QPushButton

from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.plot_view import PlotView
from ui_qt.map_view import MapView
from ui_qt.workers import SimulationWorker

DEFAULT_CONFIG: dict = {
    "wind_uncertainty":      0.20,
    "thrust_uncertainty":    0.05,
    "allowable_uncertainty": 20.0,
    "landing_prob":          90,
    "mc_n_runs":             200,
}


class SimController(QObject):
    """
    Thin controller that wires AppWindow buttons to SimulationWorker.

    Does not contain any simulation logic.  Its only job:

        button click  ->  disable UI  ->  build worker  ->  start thread
        worker signal ->  update AppState / AppWindow public API  ->  re-enable UI

    Stop semantics
    --------------
    Clicking STOP sets the worker's stop event (non-blocking); the current
    iteration finishes, the worker emits finished({"cancelled": True}), and
    the finished slot re-enables the UI.  The GUI thread is never blocked.
    """

    def __init__(
        self,
        window: AppWindow,
        state:  AppState,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._window: AppWindow               = window
        self._state:  AppState                = state
        self._worker: SimulationWorker | None = None

        self._rewire_buttons()

        # ── Cross-state signal bridge ──────────────────────────────────────────
        # When a simulation result lands in the shared AppState, automatically
        # drive the AppWindow's internal reactive state so its 3-D canvas
        # repaints without the controller knowing anything about the canvas.
        state.needs_redraw.connect(window.state.needs_redraw)

        # ── Phase 2 tolerance monitoring ───────────────────────────────────────
        # tolerance_exceeded fires every tick the bound is breached; update the
        # status bar with current numbers each time so the operator sees the
        # live drift estimate.
        state.tolerance_exceeded.connect(self._on_tolerance_exceeded)
        # tolerance_status_changed fires only on transitions (OK -> breach or
        # back); use it to flip the GO/NO-GO indicator without visual chatter.
        state.tolerance_status_changed.connect(self._on_tolerance_status_changed)

        # ── Continuous wind monitor ────────────────────────────────────────────
        # Ticks every second from application start — runs before and during
        # Phase 1, and drives Phase 2 tolerance evaluation after Phase 1.
        self._wind_timer = QTimer(self)
        self._wind_timer.setInterval(1000)
        self._wind_timer.timeout.connect(self._on_wind_tick)
        self._wind_timer.start()

        # ── Inject dedicated PlotView and MapView into docks ──────────────────
        # Replace the placeholder matplotlib canvases built by AppWindow with
        # reactive QWidget views that subscribe directly to AppState signals.
        self._plot_view = PlotView(state)
        window.profile_dock.setWidget(self._plot_view)

        self._map_view = MapView(state)
        window.map_dock.setWidget(self._map_view)

        # ── Partial-redraw wiring (no re-simulation) ───────────────────────────
        # cep_prob_input value change → update landing_probability on AppState →
        # needs_partial_redraw → _on_partial_redraw recomputes overlays only.
        state.needs_partial_redraw.connect(self._on_partial_redraw)
        _cep_input = getattr(window, 'cep_prob_input', None)
        if _cep_input is not None:
            _cep_input.valueChanged.connect(self._on_landing_prob_changed)

    # ── Button rewiring ────────────────────────────────────────────────────────

    def _rewire_buttons(self) -> None:
        """
        Redirect every btn_run / btn_stop in the widget tree to controller
        slots.  Disconnects the window-internal stub handlers first so only
        one slot fires per click.

        findChildren searches recursively, picking up both the toolbar and the
        "Simulation Controls" panel buttons in one pass.
        """
        for btn in self._window.findChildren(QPushButton, "btn_run"):
            btn.clicked.disconnect()
            btn.clicked.connect(self._on_run_clicked)

        for btn in self._window.findChildren(QPushButton, "btn_stop"):
            btn.clicked.disconnect()
            btn.clicked.connect(self._on_stop_clicked)

    # ── Run ────────────────────────────────────────────────────────────────────

    @Slot()
    def _on_run_clicked(self) -> None:
        if self._worker and self._worker.isRunning():
            return  # guard against double-click spam

        self._state.mc_running = True
        self._state.simulation_started.emit()
        self._state.is_calculating = True
        self._set_run_buttons_enabled(False)
        self._window.set_status("Simulation running...", "#f9e2af")
        self._window.set_progress(0, "Simulating...")

        self._worker = SimulationWorker(self._collect_params(), parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        # Auto-cleanup the QThread object once the run completes.
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    # ── Stop ───────────────────────────────────────────────────────────────────

    @Slot()
    def _on_stop_clicked(self) -> None:
        """
        Request cancellation; returns immediately without blocking the GUI.
        The worker will emit finished({"cancelled": True}) after its current
        iteration completes, which triggers _on_finished to re-enable the UI.
        """
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._window.set_status(
                "Stop requested — waiting for current run...", "#f38ba8")
            self._window.set_progress(0, "Stopping...")

    # ── Worker signal slots (invoked on the GUI thread via queued connection) ──

    @Slot(int)
    def _on_progress(self, value: int) -> None:
        self._window.set_progress(value, f"Simulating...  {value}%")

    @Slot(dict)
    def _on_finished(self, result: dict) -> None:
        self._state.mc_running = False

        if result.get("cancelled"):
            self._state.is_calculating = False
            self._window.set_status("Simulation cancelled.", "#a6adc8")
            self._window.set_progress(0, "Idle")
            self._worker = None
            self._set_run_buttons_enabled(True)
            return

        # ── Convert metric impact offsets -> geographic coordinates ───────────
        # impact_x = East offset (m), impact_y = North offset (m) from launch.
        lat     = self._window.lat_input.value()
        lon     = self._window.lon_input.value()
        off_e   = result.get("impact_x", 0.0)
        off_n   = result.get("impact_y", 0.0)
        cos_lat = math.cos(math.radians(lat))

        land_lat = lat + off_n / 111_320.0
        land_lon = (lon + off_e / (111_320.0 * cos_lat)
                    if cos_lat > 1e-9 else lon)

        # ── Push scalar summaries into individual AppState properties ─────────
        self._state.land_lat       = land_lat
        self._state.land_lon       = land_lon
        self._state.r90_radius     = result.get("r_N_radius",  0.0)
        self._state.mc_cep         = result.get("cep",         0.0)
        self._state.has_sim_result = True
        self._state.mc_scatter     = result.get("scatter",     [])
        self._state.mc_ellipse     = result.get("ellipse")
        self._state.kde_contours   = result.get("kde_contours", [])

        # ── Refresh AppWindow coordinate labels ────────────────────────────────
        self._window.map_widget.update_landing(land_lat, land_lon)

        # ── Write to global AppState last — emits simulation_result_changed
        #    AND needs_redraw (via signal bridge -> window.state.needs_redraw).
        self._state.simulation_result = result

        # ── Write adapted payload to AppWindow's local state ───────────────────
        # The global AppState signal bridge fires needs_redraw but the window
        # renders from its own state object; this write supplies the data.
        self._window.state.simulation_result = self._adapt_for_window(result)

        # ── Transition to Phase 2 (monitoring mode) ────────────────────────────
        # Store the nominal wind baseline so check_tolerance has a reference,
        # then activate the Phase 2 flag.  The wind timer is already running;
        # from this point every tick evaluates tolerance against these bounds.
        self._state.set_simulation_baseline(
            result.get("nominal_surf_spd", self._window.surf_spd_input.value()),
            result.get("nominal_surf_dir", self._window.surf_dir_input.value()),
        )
        self._state.phase2_active = True
        self._window.set_go_nogo(True)

        r90    = self._state.r90_radius
        cep    = self._state.mc_cep
        apogee = result.get("apogee_m",  0.0)
        tof    = result.get("hang_time", 0.0)
        n      = result.get("n_runs",    0)
        prob   = result.get("landing_prob", int(self._window.cep_prob_input.value()))
        self._window.set_status(
            f"Done  —  R{prob}: {r90:.1f} m   |   CEP50: {cep:.1f} m   |   "
            f"Apogee: {apogee:.0f} m   |   ToF: {tof:.1f} s   ({n} MC runs)",
            "#a6e3a1",
        )
        self._window.set_progress(100, "Done")

        # Cache scatter as numpy so partial redraws (cep_prob change) can
        # recompute the error ellipse without re-running the simulation.
        _scatter_raw = result.get("scatter", [])
        self._state.cached_mc_scatter = (
            np.array(_scatter_raw, dtype=np.float64) if _scatter_raw else None
        )

        # Broadcast smart-redraw lifecycle signals.
        self._state.needs_full_redraw.emit(result)
        self._state.simulation_finished.emit(result)
        self._state.is_calculating = False

        self._worker = None
        self._set_run_buttons_enabled(True)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._state.mc_running = False
        self._state.is_calculating = False
        self._window.set_status(f"Simulation error: {msg}", "#f38ba8")
        self._window.set_progress(0, "Error")
        self._worker = None
        self._set_run_buttons_enabled(True)

    # ── Phase 2 tolerance slots ────────────────────────────────────────────────

    @Slot(str)
    def _on_tolerance_exceeded(self, msg: str) -> None:
        """Fires every tick while the wind-drift bound is breached.

        Continuously refreshes the status bar so the operator sees the live
        drift estimate update each second.  GO/NO-GO is handled by the
        transition slot below to avoid flickering on every tick.
        """
        self._window.set_status(f"WARNING  TOLERANCE EXCEEDED  —  {msg}", "#f38ba8")

    @Slot(str)
    def _on_tolerance_status_changed(self, status: str) -> None:
        """Fires only when tolerance status *transitions* (breach starts or clears).

        Using the transition signal (not the per-tick exceeded signal) to drive
        GO/NO-GO eliminates visual chatter: the indicator flips exactly once
        per transition, not once per second.
        """
        in_bounds = status.startswith("✓")  # "✓"
        self._window.set_go_nogo(in_bounds)
        if in_bounds:
            self._window.set_status(
                f"Phase 2  (monitoring)  —  {status}", "#a6e3a1"
            )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _collect_params(self) -> dict:
        """Read every relevant input widget and return a flat params dict."""
        w = self._window
        return {
            "cep_prob":   w.cep_prob_input.value(),
            "sim_mode":   w.sim_mode_combo.currentText(),
            "launch_lat": w.lat_input.value(),
            "launch_lon": w.lon_input.value(),
            "elev":       w.elev_input.value(),
            "azim":       w.azim_input.value(),
            "surf_spd":   w.surf_spd_input.value(),
            "surf_dir":   w.surf_dir_input.value(),
            "up_spd":     w.up_spd_input.value(),
            "up_dir":     w.up_dir_input.value(),
            "upper_alt":  500.0,
            "mc_runs":    w.mc_runs_input.value(),
            "wind_unc":   w.wind_unc_input.value(),
            "thrust_unc": w.thrust_unc_input.value(),
        }

    @Slot()
    def _on_wind_tick(self) -> None:
        base_spd  = self._window.surf_spd_input.value()
        base_dir  = self._window.surf_dir_input.value()
        speed     = max(0.0, base_spd + random.gauss(0.0, base_spd * 0.05 + 0.1))
        direction = (base_dir + random.gauss(0.0, 3.0)) % 360.0

        # Global AppState: (speed, direction) tuples for future Phase-2 consumers.
        self._state.append_wind_reading(speed, direction)

        # Keep upper-wind state current so PlotView spaghetti/quiver are calibrated.
        self._state.upper_wind_speed = self._window.up_spd_input.value()
        self._state.upper_wind_dir   = self._window.up_dir_input.value()

        # AppWindow local state: (timestamp_s, speed_m_s) tuples for the plot.
        ts = _time.monotonic()
        history = list(self._window.state.wind_history)
        history.append((ts, speed))
        self._window.state.wind_history = history   # setter emits needs_redraw

        # Keep the status-bar readout current.
        self._window.update_wind_readout(
            speed, direction,
            self._window.up_spd_input.value(),
            self._window.up_dir_input.value(),
        )

        # Phase 2 tolerance evaluation — no-op until phase2_active is True.
        self._state.check_tolerance(speed, direction)

    @staticmethod
    def _adapt_for_window(result: dict) -> dict:
        """Remap worker payload keys to the schema AppWindow renderers expect.

        The worker emits generic physics keys (x_vals, scatter, impact_x, ...).
        AppWindow's _draw_real_result / update_map_plot read UI-centric aliases
        (trajectory_x, mc_scatter_x, land_x, cep_ellipses, ...).  All values
        are converted to native Python types so no numpy scalars reach Qt.
        """
        sc   = result.get("scatter", [])
        prob = result.get("landing_prob", 90)

        cep_ellipses: list[dict] = []
        ell = result.get("ellipse")
        if ell and "a" in ell and "b" in ell:
            cep_ellipses.append({
                **ell,
                "color": "#cba6f7",
                "lw":    2.0,
                "label": f"R{prob}",
            })

        adapted = dict(result)   # preserve all original keys for global AppState
        adapted.update({
            "trajectory_x": [float(v) for v in result.get("x_vals", [])],
            "trajectory_y": [float(v) for v in result.get("y_vals", [])],
            "trajectory_z": [float(v) for v in result.get("z_vals", [])],
            "mc_scatter_x": [float(p[0]) for p in sc],
            "mc_scatter_y": [float(p[1]) for p in sc],
            "land_x":       float(result.get("impact_x", 0.0)),
            "land_y":       float(result.get("impact_y", 0.0)),
            "cep_ellipses": cep_ellipses,
        })
        return adapted

    @Slot()
    def _on_partial_redraw(self) -> None:
        """Recompute overlays at the new landing_probability without re-simulating."""
        self._window.update_visual_overlays(self._state)

    @Slot(int)
    def _on_landing_prob_changed(self, value: int) -> None:
        """Propagate cep_prob_input change to AppState.landing_probability."""
        self._state.landing_probability = value

    def _set_run_buttons_enabled(self, enabled: bool) -> None:
        for btn in self._window.findChildren(QPushButton, "btn_run"):
            btn.setEnabled(enabled)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Shared data bus: holds computed simulation results for all future views.
    app_state = AppState(config=DEFAULT_CONFIG)

    # AppWindow manages its own lightweight reactive state for the 3-D plot;
    # we do NOT inject app_state here to avoid a needs_redraw incompatibility.
    window = AppWindow()
    window.show()

    # Controller wires the run/stop buttons to the background worker.
    # Must be assigned to a variable so it is not garbage-collected while
    # the event loop runs.
    controller = SimController(window, app_state)

    sys.exit(app.exec())
