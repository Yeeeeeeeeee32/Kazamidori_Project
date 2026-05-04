"""
main_qt.py
PySide6 entry point and top-level controller for the Kazamidori Project.

Responsibilities
----------------
1. Construct QApplication.
2. Build the shared AppState (cross-component data bus for computed results).
3. Show AppWindow (which owns its own reactive plot-state internally).
4. Wire RUN / STOP buttons → SimulationWorker via SimController.
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

from PySide6.QtCore import QObject, QTimer, Slot
from PySide6.QtWidgets import QApplication, QPushButton

from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
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

        button click  →  disable UI  →  build worker  →  start thread
        worker signal →  update AppState / AppWindow public API  →  re-enable UI

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
        self._window: AppWindow         = window
        self._state:  AppState          = state
        self._worker: SimulationWorker | None = None

        self._rewire_buttons()

        # ── Cross-state signal bridge ──────────────────────────────────────────
        # When a simulation result lands in the shared AppState, automatically
        # drive the AppWindow's internal reactive state so its 3-D canvas
        # repaints without the controller knowing anything about the canvas.
        state.needs_redraw.connect(window.state.needs_redraw)

        # ── Phase 2 wind monitor ───────────────────────────────────────────────
        # Ticks every second; generates a perturbed wind reading and pushes it
        # into AppState so the wind-history graph can update continuously.
        self._wind_timer = QTimer(self)
        self._wind_timer.setInterval(1000)
        self._wind_timer.timeout.connect(self._on_wind_tick)
        self._wind_timer.start()

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
        self._set_run_buttons_enabled(False)
        self._window.set_status("Simulation running…", "#f9e2af")
        self._window.set_progress(0, "Simulating…")

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
            self._window.set_status("Stop requested — waiting for current run…",
                                    "#f38ba8")
            self._window.set_progress(0, "Stopping…")

    # ── Worker signal slots (invoked on the GUI thread via queued connection) ──

    @Slot(int)
    def _on_progress(self, value: int) -> None:
        self._window.set_progress(value, f"Simulating…  {value}%")

    @Slot(dict)
    def _on_finished(self, result: dict) -> None:
        self._state.mc_running = False

        if result.get("cancelled"):
            self._window.set_status("Simulation cancelled.", "#a6adc8")
            self._window.set_progress(0, "Idle")
            self._worker = None
            self._set_run_buttons_enabled(True)
            return

        # ── Convert metric impact offsets → geographic coordinates ─────────────
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
        # These allow fine-grained observation by future views (map circles,
        # Phase 2 overlay, …) without them needing to unpack the full dict.
        self._state.land_lat       = land_lat
        self._state.land_lon       = land_lon
        self._state.r90_radius     = result.get("r_N_radius",  0.0)
        self._state.mc_cep         = result.get("cep",         0.0)
        self._state.has_sim_result = True
        self._state.mc_scatter     = result.get("scatter",      [])
        self._state.mc_ellipse     = result.get("ellipse")
        self._state.kde_contours   = result.get("kde_contours", [])

        # ── Refresh AppWindow's public-API widgets ─────────────────────────────
        self._window.map_widget.update_landing(land_lat, land_lon)

        # ── Write to global AppState last — emits simulation_result_changed
        #    AND needs_redraw (via signal bridge → window.state.needs_redraw).
        self._state.simulation_result = result

        # ── Write adapted payload to AppWindow's local state ───────────────────
        # The global AppState signal bridge fires needs_redraw but the window
        # renders from its own state object; this write supplies the data.
        self._window.state.simulation_result = self._adapt_for_window(result)

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
        self._worker = None
        self._set_run_buttons_enabled(True)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._state.mc_running = False
        self._window.set_status(f"Simulation error: {msg}", "#f38ba8")
        self._window.set_progress(0, "Error")
        self._worker = None
        self._set_run_buttons_enabled(True)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _collect_params(self) -> dict:
        """Read every relevant input widget and return a flat params dict."""
        w = self._window
        return {
            # Simulation setup
            "cep_prob":   w.cep_prob_input.value(),      # percentile (50–99)
            "sim_mode":   w.sim_mode_combo.currentText(),
            # Launch site
            "launch_lat": w.lat_input.value(),
            "launch_lon": w.lon_input.value(),
            # Launch geometry
            "elev":       w.elev_input.value(),           # elevation angle (°)
            "azim":       w.azim_input.value(),           # azimuth (°)
            # Wind observations (used by _build_wind_profiles)
            "surf_spd":   w.surf_spd_input.value(),       # surface speed (m/s)
            "surf_dir":   w.surf_dir_input.value(),       # surface FROM dir (°)
            "up_spd":     w.up_spd_input.value(),         # upper speed (m/s)
            "up_dir":     w.up_dir_input.value(),         # upper FROM dir (°)
            "upper_alt":  500.0,                          # assumed upper obs alt (m AGL)
            # Monte Carlo settings
            "mc_runs":    w.mc_runs_input.value(),
            "wind_unc":   w.wind_unc_input.value(),       # fractional 1-σ
            "thrust_unc": w.thrust_unc_input.value(),     # fractional 1-σ
        }

    @Slot()
    def _on_wind_tick(self) -> None:
        base_spd  = self._window.surf_spd_input.value()
        base_dir  = self._window.surf_dir_input.value()
        speed     = max(0.0, base_spd + random.gauss(0.0, base_spd * 0.05 + 0.1))
        direction = (base_dir + random.gauss(0.0, 3.0)) % 360.0

        # Global AppState: (speed, direction) tuples for future Phase-2 consumers.
        self._state.append_wind_reading(speed, direction)

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

    @staticmethod
    def _adapt_for_window(result: dict) -> dict:
        """Remap worker payload keys to the schema AppWindow renderers expect.

        The worker emits generic physics keys (x_vals, scatter, impact_x, …).
        AppWindow's _draw_real_result / update_map_plot read UI-centric aliases
        (trajectory_x, mc_scatter_x, land_x, cep_ellipses, …).  All values are
        converted to native Python types so no numpy scalars reach the Qt layer.
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
