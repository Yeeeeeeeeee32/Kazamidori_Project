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
import sys

from PySide6.QtCore import QObject, Slot
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
        if result.get("cancelled"):
            self._window.set_status("Simulation cancelled.", "#a6adc8")
            self._window.set_progress(0, "Idle")
            self._set_run_buttons_enabled(True)
            return

        # ── Convert metric offsets → geographic coordinates ────────────────────
        lat      = self._window.lat_input.value()
        lon      = self._window.lon_input.value()
        off_e    = result.get("land_offset_e", 0.0)
        off_n    = result.get("land_offset_n", 0.0)
        cos_lat  = math.cos(math.radians(lat))

        land_lat = lat + off_n / 111_320.0
        land_lon = (lon + off_e / (111_320.0 * cos_lat)
                    if cos_lat > 1e-9 else lon)

        # ── Persist in the shared data-bus AppState ────────────────────────────
        self._state.land_lat       = land_lat
        self._state.land_lon       = land_lon
        self._state.r90_radius     = result.get("r90_radius",    0.0)
        self._state.mc_cep         = result.get("mc_cep",        0.0)
        self._state.has_sim_result = True

        # ── Refresh AppWindow's public-API widgets ─────────────────────────────
        self._window.map_widget.update_landing(land_lat, land_lon)
        # Trigger the window's internal reactive state to repaint the 3-D plot.
        self._window.state.needs_redraw.emit()

        r90 = self._state.r90_radius
        cep = self._state.mc_cep
        self._window.set_status(
            f"Done  —  R90: {r90:.1f} m   |   CEP50: {cep:.1f} m   |   "
            f"Apogee: {result.get('apogee_alt', 0.0):.0f} m   "
            f"({result.get('n_runs', 0)} runs)",
            "#a6e3a1",
        )
        self._window.set_progress(100, "Done")
        self._set_run_buttons_enabled(True)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._window.set_status(f"Simulation error: {msg}", "#f38ba8")
        self._window.set_progress(0, "Error")
        self._set_run_buttons_enabled(True)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _collect_params(self) -> dict:
        """Read every relevant input widget and return a flat params dict."""
        w = self._window
        return {
            "wind_speed":  w.wind_speed_input.value(),
            "wind_dir":    w.wind_dir_input.value(),
            "cep_prob":    w.cep_prob_input.value(),
            "sim_mode":    w.sim_mode_combo.currentText(),
            "launch_lat":  w.lat_input.value(),
            "launch_lon":  w.lon_input.value(),
            "elev":        w.elev_input.value(),
            "azim":        w.azim_input.value(),
            "mc_runs":     w.mc_runs_input.value(),
            "surf_spd":    w.surf_spd_input.value(),
            "surf_dir":    w.surf_dir_input.value(),
            "up_spd":      w.up_spd_input.value(),
            "up_dir":      w.up_dir_input.value(),
            "wind_unc":    w.wind_unc_input.value(),
            "thrust_unc":  w.thrust_unc_input.value(),
        }

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
