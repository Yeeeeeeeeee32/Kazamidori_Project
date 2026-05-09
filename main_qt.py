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
from PySide6.QtCore import QObject, QThread, QTimer, Slot
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QPushButton

from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow, GLOBAL_QSS
from ui_qt.workers import SimulationWorker
from core.monte_carlo  import compute_cep_ellipse
from utils.data_loader import (
    RocketConfigError,
    load_rocket_config,
    parse_parachute_json,
    parse_rkt_file,
)

DEFAULT_CONFIG: dict = {
    "wind_uncertainty":   0.20,
    "thrust_uncertainty": 0.05,
    "landing_prob":        90,
    "mc_n_runs":          200,
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
        # Cache the sig_nominal_done payload so _on_mc_done can forward
        # phases/events (phase-coloured 3-D trajectory) into the final result.
        self._nominal_payload: dict | None    = None

        self._rewire_buttons()

        # ── Parameter readiness interlock ──────────────────────────────────────
        # RUN buttons start disabled; they unlock only after all 15 critical
        # geometry parameters become non-None (i.e. after Rocket.json is loaded).
        # sig_ready_state_changed fires True/False whenever readiness flips.
        self._set_run_buttons_enabled(state.is_ready_to_run)
        state.sig_ready_state_changed.connect(self._set_run_buttons_enabled)

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

        # ── Partial-redraw wiring (no re-simulation) ───────────────────────────
        # cep_prob_input value change → update cep_probability on AppState →
        # needs_partial_redraw → _on_partial_redraw recomputes overlays only.
        state.needs_partial_redraw.connect(self._on_partial_redraw)
        _cep_input = getattr(window, 'cep_prob_input', None)
        if _cep_input is not None:
            _cep_input.valueChanged.connect(self._on_cep_prob_changed)

        # ── Mode ComboBox → AppState.flight_mode ───────────────────────────────
        # UI → AppState binding: sim_mode_combo drives flight_mode so the worker
        # never reads the widget directly.
        _mode_combo = getattr(window, 'mode_combo', None)
        if _mode_combo is not None:
            state.flight_mode = _mode_combo.currentText()   # sync initial value
            _mode_combo.currentTextChanged.connect(self._on_flight_mode_changed)

        # ── rmax_input → AppState.target_radius ────────────────────────────────
        # Bidirectional: spinbox drives target_radius; external writes (JSON load)
        # drive spinbox back via target_radius_changed signal.
        _rmax = getattr(window, 'rmax_input', None)
        if _rmax is not None:
            state.target_radius = float(_rmax.value())
            _rmax.valueChanged.connect(
                lambda v: setattr(state, 'target_radius', float(v))
            )
            state.target_radius_changed.connect(
                lambda v: _rmax.setValue(float(v))
            )

        # ── Gust Input → AppState.gust_speed ───────────────────────────────────
        # Accepts both QDoubleSpinBox (valueChanged→float) and QLineEdit
        # (textChanged→str).  QDoubleSpinBox is preferred; QLineEdit fallback
        # converts the text via float() with a silent no-op on parse failure.
        _gust_input = getattr(window, 'gust_input', None)
        if _gust_input is not None:
            if hasattr(_gust_input, 'valueChanged'):
                state.gust_speed = float(_gust_input.value())
                _gust_input.valueChanged.connect(self._on_gust_speed_changed)
            elif hasattr(_gust_input, 'textChanged'):
                try:
                    state.gust_speed = float(_gust_input.text())
                except ValueError:
                    pass
                _gust_input.textChanged.connect(self._on_gust_speed_text_changed)

        # ── 12 Airframe spinboxes ↔ AppState (CGS display / SI storage) ────────
        # Each spinbox displays in CGS (cm, g, s); AppState stores SI (m, kg, s).
        # Bidirectional binding: spinbox → AppState on user edit; AppState signal
        # → spinbox.setValue on JSON load or external state change.
        self._wire_airframe_spinboxes()

        # ── Load Rocket JSON button → controller ──────────────────────────────
        # AppWindow emits sig_load_rocket_json_clicked instead of opening the
        # dialog itself, keeping file I/O and AppState writes in the controller.
        _json_sig = getattr(window, 'sig_load_rocket_json_clicked', None)
        if _json_sig is not None:
            _json_sig.connect(self._on_load_rocket_json)

        # ── Load .rkt (RockSim) button → controller ────────────────────────────
        _rkt_sig = getattr(window, 'sig_load_rkt_clicked', None)
        if _rkt_sig is not None:
            _rkt_sig.connect(self._on_load_rkt)

        # ── Load Parachute JSON button → controller ────────────────────────────
        _para_sig = getattr(window, 'sig_load_para_json_clicked', None)
        if _para_sig is not None:
            _para_sig.connect(self._on_load_para_json)

        # ── Advanced Settings button → controller-managed dialog ───────────────
        # Reconnect from AppWindow's bare exec() stub to the controller's
        # populate-from-AppState → exec → push-to-AppState flow.
        for _btn in self._window.findChildren(QPushButton, "btn_adv_settings"):
            try:
                _btn.clicked.disconnect()
            except RuntimeError:
                pass
            _btn.clicked.connect(self._on_advanced_settings_clicked)

        # ── Two-stage rendering: AppState → UI ────────────────────────────────
        # nominal_needs_redraw fires (via _on_nominal_done) before MC starts;
        # draw the 3-D profile immediately so the operator sees the trajectory
        # without waiting for the full Monte Carlo batch to finish.
        state.nominal_needs_redraw.connect(window.update_profile_plot)

        # wind_history_updated fires every second (surface) and once per
        # simulation (all 5 altitudes) — drives the rolling wind time-series.
        state.wind_history_updated.connect(window.update_wind_history)
        # After update_wind_history refreshes _wind_hist_buf, update the
        # instantaneous "Current Wind Speed" table that sits beside the plot.
        # Connection order guarantees hist_buf is populated before the table reads it.
        state.wind_history_updated.connect(lambda _: window._update_wind_table())

        # progress_changed fires after every MC iteration (0–100 int) → push the
        # value into the toolbar QProgressBar so the operator sees live progress.
        state.progress_changed.connect(
            lambda pct: window.set_progress(pct, f"MC  {pct}%")
        )

        # simulation_result_changed fires in _on_mc_done (after full MC) → render
        # the 2-D landing map with scatter, CEP contours, and confidence ellipses.
        # The signal carries the result dict as payload; update_map_plot reads from
        # window.state directly so the payload is discarded here.
        state.simulation_result_changed.connect(
            lambda _: window.update_map_plot()
        )
        state.simulation_result_changed.connect(
            lambda _: window.update_profile_plot()
        )

        # ── Flight-mode → map circle switch ───────────────────────────────────
        # update_map_plot reads window.state.sim_mode (already kept in sync by
        # mode_combo → window.state.sim_mode binding in _bind_state), but no
        # redraw is triggered on mode change.  Fire one here.
        state.flight_mode_changed.connect(lambda _: window.update_map_plot())

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

        for btn in self._window.findChildren(QPushButton, "btn_phase1_run"):
            btn.clicked.disconnect()
            btn.clicked.connect(self._on_run_clicked)

    # ── Run ────────────────────────────────────────────────────────────────────

    @Slot()
    def _on_run_clicked(self) -> None:
        if self._worker and self._worker.isRunning():
            return  # guard against double-click spam

        # ── 60-second surface wind buffer check ───────────────────────────────
        surface_hist = list(self._state.wind_history_for_alt(3.0))
        if len(surface_hist) < 5:
            QMessageBox.warning(
                self._window,
                "Wind Buffer Insufficient",
                f"Surface wind monitor has only {len(surface_hist)} sample(s) "
                f"(minimum 5 required).\n\n"
                "Wait a few seconds for the wind monitor to collect data, "
                "then click RUN again.",
            )
            return

        # ── Launch site entry check ────────────────────────────────────────────
        _BLANK = -9999.0
        _unset = []
        if self._window.lat_input.value()  == _BLANK: _unset.append("Latitude")
        if self._window.lon_input.value()  == _BLANK: _unset.append("Longitude")
        if self._window.azim_input.value() == _BLANK: _unset.append("Rail Azimuth")
        if _unset:
            QMessageBox.warning(
                self._window,
                "Launch Settings Incomplete",
                "Please enter a value for:\n  • " + "\n  • ".join(_unset),
            )
            return

        self._state.mc_running = True
        self._state.simulation_started.emit()
        self._state.is_calculating = True
        self._set_run_buttons_enabled(False)
        self._window.set_status("Simulation running...", "#f9e2af")
        self._window.set_progress(0, "Simulating...")

        # Clear stale data from the previous run.
        self._nominal_payload                 = None
        self._window.state.simulation_result = None
        self._window.update_map_plot()

        self._worker = SimulationWorker(self._collect_params(), parent=self)
        self._worker.progress.connect(self._on_progress)
        # ── Two-stage routing ──────────────────────────────────────────────────
        # sig_nominal_done fires before any MC runs start; renders trajectory now.
        self._worker.sig_nominal_done.connect(self._on_nominal_done)
        # sig_progress_updated fires after every single MC iteration (current, total).
        self._worker.sig_progress_updated.connect(self._on_progress_updated)
        # finished covers both cancelled and full-MC-done paths (replaces _on_finished).
        self._worker.finished.connect(self._on_mc_done)
        self._worker.error.connect(self._on_error)
        self._worker.sig_status_text.connect(self._on_worker_status)
        self._worker.sig_early_warning.connect(self._on_early_warning)
        # Auto-cleanup the QThread object once the run completes.
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()
        # Lower priority so the GUI thread wins CPU time at the stage boundary
        # (after sig_nominal_done is emitted) and can drain its event queue
        # before the MC loop starts.  Combined with the msleep(0) in workers.py
        # this ensures the trajectory canvas paints before MC progress begins.
        self._worker.setPriority(QThread.Priority.LowPriority)

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
    def _on_nominal_done(self, payload: dict) -> None:
        """Invoked on the GUI thread when the nominal single run completes.

        Sets AppState.nominal_result, which automatically emits both
        nominal_result_changed (carrying the dict) and nominal_needs_redraw
        (zero-payload), so any subscribed view repaints the 3-D trajectory
        without waiting for the MC loop to finish.

        Also calls append_wind_nodes so all five altitude deques in the wind
        history buffer are populated with the nominal wind snapshot the instant
        the simulation result arrives — before any MC iteration starts.

        Thread safety: sig_nominal_done crosses from the worker thread to the
        main thread via Qt's automatic queued connection (different QThread
        affinity), so this slot always executes on the GUI thread.
        """
        # Populate window state with the rich nominal payload BEFORE firing
        # nominal_result.  nominal_result.setter emits nominal_needs_redraw →
        # update_profile_plot, which reads window.state.simulation_result.
        # _adapt_nominal_for_window preserves all original keys (phases, events,
        # wind_nodes, …) so _draw_real_result gets full phase-colour data.
        self._window.state.simulation_result = (
            self._adapt_nominal_for_window(payload))

        # Setting nominal_result fires nominal_result_changed and
        # nominal_needs_redraw; the latter triggers update_profile_plot.
        self._state.nominal_result = payload
        self._nominal_payload      = payload

        # Early warning: update GO/NO-GO indicator immediately after nominal run,
        # before the MC loop starts, so the operator never waits blindly.
        _off_e    = float(payload.get("impact_x", 0.0))
        _off_n    = float(payload.get("impact_y", 0.0))
        _nom_dist = math.hypot(_off_e, _off_n)
        _target_r = self._state.target_radius
        if _nom_dist > _target_r:
            self._window.update_status_indicator(
                f"⚠️  NO-GO — Nominal {_nom_dist:.0f} m > Target {_target_r:.0f} m  |  MC running…"
            )
        else:
            self._window.update_status_indicator(
                f"⏳  CALCULATING — Nominal {_nom_dist:.0f} m / {_target_r:.0f} m  |  MC running…"
            )

        # Populate all 5 altitude wind-history deques with the nominal snapshot
        # and immediately refresh the Current Wind Speed table with the
        # instantaneous values that were just sampled from the simulation wind
        # profile — one call covers all 5 altitudes at once.
        wind_nodes = payload.get("wind_nodes", [])
        if wind_nodes:
            self._state.append_wind_nodes(wind_nodes)
            self._window._update_wind_table(nodes=wind_nodes)

        apogee = payload.get("apogee_m", 0.0)
        tof    = payload.get("hang_time", 0.0)
        self._window.set_status(
            f"Nominal done — Apogee: {apogee:.0f} m  |  ToF: {tof:.1f} s  |  MC running…",
            "#f9e2af",
        )
        # Force the canvas to commit its pending draw now, before the MC loop
        # (running at LowPriority on the worker thread) monopolises CPU.
        self._window.profile_canvas.draw_idle()
        QApplication.processEvents()

    @Slot(int, int)
    def _on_progress_updated(self, current: int, total: int) -> None:
        """Invoked on the GUI thread after every single MC run.

        Converts the raw (current, total) heartbeat into a 0–100 percentage
        and pushes it to AppState.progress_percentage.  Division-by-zero is
        guarded: if total == 0, percentage stays at 0.

        Thread safety: sig_progress_updated crosses thread boundaries via Qt's
        queued connection, so the AppState write always happens on the GUI thread.
        """
        pct = int(current / total * 100) if total > 0 else 0
        self._state.progress_percentage = pct

    @Slot(dict)
    def _on_mc_done(self, result: dict) -> None:
        self._state.mc_running = False

        if result.get("cancelled"):
            self._state.is_calculating    = False
            self._state.progress_percentage = 0   # clear progress bar
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
        # scatter_points is the canonical key; "scatter" is the backward-compat alias.
        _scatter = result.get("scatter_points", result.get("scatter", []))
        self._state.land_lat       = land_lat
        self._state.land_lon       = land_lon
        self._state.r90_radius     = result.get("r_N_radius",  0.0)
        self._state.mc_cep         = result.get("cep",         0.0)
        self._state.has_sim_result = True
        self._state.mc_scatter     = _scatter
        self._state.mc_ellipse     = result.get("ellipse")
        # Cache scatter as ndarray so _on_partial_redraw can recompute the
        # ellipse instantly when the CEP slider moves — no re-simulation needed.
        if _scatter:
            self._state.cached_mc_scatter = np.array(_scatter, dtype=float)
        self._state.kde_contours   = result.get("kde_contours", [])

        # ── Refresh AppWindow coordinate labels ────────────────────────────────
        self._window.map_widget.update_landing(land_lat, land_lon)

        # ── Populate window state FIRST ────────────────────────────────────────
        # Carry phases/events from the nominal payload into the final result so
        # the phase-coloured 3-D trajectory persists after MC finishes.
        if self._nominal_payload:
            result = dict(result)
            result["phases"] = self._nominal_payload.get("phases")
            result["events"] = self._nominal_payload.get("events")

        # simulation_result_changed (fired on the next line) triggers
        # update_map_plot via the __init__ connection.  update_map_plot reads
        # from window.state, so that write must precede the signal.
        self._window.state.simulation_result = self._adapt_for_window(result)
        # ── Fire simulation_result_changed → update_map_plot + needs_redraw ───
        self._state.simulation_result = result

        # ── Phase A verification: CEP50 ≤ target_radius → SAFE → Phase B ────────
        cep50    = self._state.mc_cep
        target_r = self._state.target_radius
        is_safe  = cep50 <= target_r

        if is_safe:
            # Lock the Phase A wind distribution so Phase B O(1) GO/NO-GO ticks
            # compare live wind against the exact baseline used in the MC run.
            surf_spd = result.get("nominal_surf_spd",
                                  self._window.surf_spd_input.value())
            surf_dir = result.get("nominal_surf_dir",
                                  self._window.surf_dir_input.value())
            wind_unc = float(self._window.wind_unc_input.value())
            mu_u = surf_spd * math.sin(math.radians(surf_dir))
            mu_v = surf_spd * math.cos(math.radians(surf_dir))
            self._state.set_wind_lock(mu_u, mu_v, wind_unc)
            self._state.phase2_active = True
            self._window.update_status_indicator(
                f"🟢  GO  (CEP {cep50:.0f} m ≤ {target_r:.0f} m)"
            )
        else:
            self._window.update_status_indicator(
                f"🔴  NO-GO  (CEP {cep50:.0f} m > {target_r:.0f} m)"
            )

        r90    = self._state.r90_radius
        apogee = result.get("apogee_m",  0.0)
        tof    = result.get("hang_time", 0.0)
        n      = result.get("n_runs",    0)
        prob   = result.get("landing_prob", int(self._window.cep_prob_input.value()))
        verdict = (
            f"SAFE  CEP {cep50:.0f} m <= {target_r:.0f} m"
            if is_safe else
            f"UNSAFE  CEP {cep50:.0f} m > {target_r:.0f} m"
        )
        self._window.set_status(
            f"{verdict}   |   R{prob}: {r90:.1f} m   |   "
            f"Apogee: {apogee:.0f} m   |   ToF: {tof:.1f} s   ({n} runs)",
            "#a6e3a1" if is_safe else "#f9e2af",
        )
        self._window.set_progress(100, "Done")

        # Cache scatter as numpy so partial redraws (cep_prob change) can
        # recompute the error ellipse without re-running the simulation.
        self._state.cached_mc_scatter = (
            np.array(_scatter, dtype=np.float64) if _scatter else None
        )

        # Broadcast smart-redraw lifecycle signals.
        self._state.needs_full_redraw.emit(result)
        self._state.simulation_finished.emit(result)
        # Reset AppState progress flag (fires progress_changed(0) → lambda calls
        # set_progress(0, "MC  0%") synchronously on the GUI thread).
        self._state.progress_percentage = 0
        # Override the "MC  0%" label from the lambda with a clean idle state.
        self._window.set_progress(0, "Idle")
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

    @Slot(str)
    def _on_worker_status(self, msg: str) -> None:
        # Update the main status bar (colour-coded amber for in-progress stages).
        self._window.set_status(msg, "#f9e2af")
        # Mirror the same label onto the progress bar format string so the
        # "Simulating..." / "Monte Carlo..." text appears next to the bar value.
        self._window.set_progress(self._state.progress_percentage, msg)

    # ── Phase 2 tolerance slots ────────────────────────────────────────────────

    @Slot(str)
    def _on_early_warning(self, msg: str) -> None:
        """Fires once if the nominal landing point is outside target_radius.

        Displayed in red so the operator immediately sees the NO-GO state.
        The MC loop continues regardless — the slot does NOT stop the worker.
        """
        self._window.set_status(msg, "#f38ba8")

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
        """Fires only when GO ↔ NO-GO status *transitions*.

        Using the transition signal (not the per-tick exceeded signal) to drive
        the GO/NO-GO indicator eliminates visual chatter: the label flips exactly
        once per transition, not once per second.
        """
        in_bounds = status.startswith("✓")
        self._window.set_go_nogo(in_bounds)
        if in_bounds:
            self._window.set_status(
                f"Phase B  (monitoring)  —  {status}", "#a6e3a1"
            )
        else:
            self._window.set_status(
                f"Phase B  {status}  —  wind outside Phase A envelope", "#f38ba8"
            )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _collect_params(self) -> dict:
        """Build the flat params dict passed to SimulationWorker.

        Unidirectional flow: UI widgets → AppState (via bound slots) → here.
        flight_mode, gust_speed, and all 12 rocket geometry properties are
        read from AppState (never from widgets) to keep the worker thread
        free of any UI coupling.

        Turbulence intensity
        --------------------
        μ and σ are computed from the 60-second surface wind history buffer
        (3 m AGL, updated at 1 Hz by the hardware anemometer).  The absolute
        gust noise passed to the MC perturbation engine is:

            auto_gust_ms = σ   (m/s)

        This replaces the manual gust_speed UI input when σ > 0 so that MC
        scatter reflects the actual observed turbulence at the launch site.

        Unit contract (ALL values in SI — m, kg, s)
        -------------------------------------------
        All spinbox widgets display MKS values directly; AppState stores MKS.
        _collect_params reads only from AppState, so every value in the
        returned dict is already SI.  No conversion is applied in
        workers._build_sim_params.
        """
        w = self._window
        s = self._state

        # ── Turbulence intensity from 60s surface wind history ────────────────
        # Use population std-dev (÷N, not ÷(N-1)) — we have the full population
        # over the 60s window, not a sample from an infinite process.
        _hist = list(s.wind_history_for_alt(3.0))
        if len(_hist) >= 2:
            _speeds     = [e["speed_ms"] for e in _hist]
            _turb_mu    = sum(_speeds) / len(_speeds)
            _turb_sigma = math.sqrt(
                sum((_v - _turb_mu) ** 2 for _v in _speeds) / len(_speeds)
            )
        else:
            # No history yet — fall back to the current anemometer reading
            _turb_mu    = float(s.surf_wind_speed)
            _turb_sigma = 0.0
        # I = σ/μ  (dimensionless turbulence intensity)
        _turb_intensity = _turb_sigma / _turb_mu if _turb_mu > 0.1 else 0.0

        return {
            # ── Simulation / wind config ───────────────────────────────────────
            # cep_prob: read from AppState (authoritative) — updated by both the
            # main-window cep_prob_input widget and the Advanced Settings dialog.
            # Reading the widget directly would silently ignore Advanced Settings.
            "cep_prob":   int(s.landing_prob),
            "launch_lat": w.lat_input.value()  if w.lat_input.value()  != -9999.0 else 35.6828,
            "launch_lon": w.lon_input.value()  if w.lon_input.value()  != -9999.0 else 139.7590,
            "elev":       s.launch_angle,   # degrees — AppState, default 85.0
            "rail":       s.launch_rail,    # m       — AppState, default 1.0
            "azim":       w.azim_input.value() if w.azim_input.value() != -9999.0 else 0.0,
            "surf_spd":   w.surf_spd_input.value(),
            "surf_dir":   w.surf_dir_input.value(),
            "up_spd":     w.up_spd_input.value(),
            "up_dir":     w.up_dir_input.value(),
            "upper_alt":  500.0,
            "mc_runs":    w.mc_runs_input.value(),
            "wind_unc":   w.wind_unc_input.value(),
            "thrust_unc": w.thrust_unc_input.value(),
            # ── Mode and gust — sourced from AppState ──────────────────────────
            "flight_mode":  s.flight_mode,
            "is_free_mode": ("free" in str(s.flight_mode).lower() or "自由" in str(s.flight_mode)),
            "gust_speed":   s.gust_speed,
            # ── Turbulence (auto-computed from 60s wind history) ───────────────
            "turb_mu":        _turb_mu,
            "turb_sigma":     _turb_sigma,
            "turb_intensity": _turb_intensity,
            # auto_gust_ms = σ: absolute gust noise (m/s) for MC perturbation.
            # Overrides the manual gust_speed when σ > 0 (observed turbulence
            # is always more representative than a hand-set UI value).
            "auto_gust_ms":   _turb_sigma,
            # ── 12 rocket geometry params — sourced from AppState (SI) ─────────
            # Key names match _DEFAULT_ROCKET in workers.py.
            # AppState and spinboxes are both SI (m, kg) — no conversion needed.
            "airframe_mass":  s.rocket_dry_mass,       # kg
            "airframe_cg":    s.rocket_cg,             # m from nose
            "airframe_len":   s.rocket_length,         # m
            "radius":         s.rocket_diameter / 2.0, # m  (AppState holds diameter)
            "nose_len":       s.nose_length,           # m
            "fin_root":       s.fin_root_chord,        # m
            "fin_tip":        s.fin_tip_chord,         # m
            "fin_span":       s.fin_span,              # m
            "fin_pos":        s.fin_position,          # m from nose
            "motor_pos":      s.motor_cg,              # m from nose
            "motor_dry_mass": s.motor_dry_mass,        # kg
            "body_cd":        s.drag_coeff or 0.45,          # rocket airframe Cd (None→default)
            "para_cd":        s.parachute_cd,              # parachute Cd (for CdS product)
            "para_area":      s.parachute_area,            # m²
            "para_lag":       s.parachute_lag,         # s
            "backfire_delay": s.backfire_delay,        # s
            # ── Motor thrust curve — persisted by AppWindow._on_load_motor() ──
            "target_radius":  s.target_radius,              # m  (rmax_input spinbox)
            **({
                "thrust_data":     w._motor_thrust_data,
                "motor_burn_time": w._motor_burn_time,
            } if getattr(w, "_motor_thrust_data", None) else {}),
        }

    @Slot(str)
    def _on_flight_mode_changed(self, mode: str) -> None:
        """Propagate Mode ComboBox selection to AppState.flight_mode."""
        self._state.flight_mode = mode

    @Slot(float)
    def _on_gust_speed_changed(self, value: float) -> None:
        """Propagate QDoubleSpinBox gust value to AppState.gust_speed."""
        self._state.gust_speed = value

    @Slot(str)
    def _on_gust_speed_text_changed(self, text: str) -> None:
        """Propagate QLineEdit gust text to AppState.gust_speed (silent on parse fail)."""
        try:
            self._state.gust_speed = float(text)
        except ValueError:
            pass

    # ── Airframe spinbox wiring ────────────────────────────────────────────────

    def _wire_airframe_spinboxes(self) -> None:
        """Bind the 12 SI airframe spinboxes bidirectionally to AppState (SI).

        Spinboxes now display SI values directly (m, kg, s) so both conversion
        lambdas are identity functions, except af_radius_input which shows body
        radius (m) while AppState stores full diameter (m) — factor of 2 only.
        """
        w = self._window
        s = self._state

        def _bind(sb_name: str, prop: str, to_si, from_si) -> None:
            sb = getattr(w, sb_name, None)
            if sb is None:
                return
            # Sentinel -9999.0 means the field is blank; do not push to AppState.
            sb.valueChanged.connect(
                lambda v, _p=prop, _f=to_si:
                    None if v == -9999.0 else setattr(s, _p, _f(v))
            )
            sig = getattr(s, f"{prop}_changed", None)
            if sig is not None:
                sig.connect(lambda v, _sb=sb, _g=from_si: _sb.setValue(_g(v)))

        # af_radius_input shows body radius (m); AppState stores full diameter (m)
        _bind("af_mass_input",     "rocket_dry_mass", lambda v: float(v),      lambda v: float(v))
        _bind("af_cg_input",       "rocket_cg",       lambda v: float(v),      lambda v: float(v))
        _bind("af_len_input",      "rocket_length",   lambda v: float(v),      lambda v: float(v))
        _bind("af_radius_input",   "rocket_diameter", lambda v: v * 2.0,       lambda v: v / 2.0)
        _bind("af_nose_input",     "nose_length",     lambda v: float(v),      lambda v: float(v))
        _bind("af_finroot_input",  "fin_root_chord",  lambda v: float(v),      lambda v: float(v))
        _bind("af_fintip_input",   "fin_tip_chord",   lambda v: float(v),      lambda v: float(v))
        _bind("af_finspan_input",  "fin_span",        lambda v: float(v),      lambda v: float(v))
        _bind("af_finpos_input",   "fin_position",    lambda v: float(v),      lambda v: float(v))
        _bind("af_motorpos_input", "motor_cg",        lambda v: float(v),      lambda v: float(v))
        _bind("af_motormass_input","motor_dry_mass",  lambda v: float(v),      lambda v: float(v))
        _bind("af_backfire_input", "backfire_delay",  lambda v: float(v),      lambda v: float(v))
        _bind("para_cd_input",     "parachute_cd",    lambda v: float(v),      lambda v: float(v))
        _bind("para_area_input",   "parachute_area",  lambda v: float(v),      lambda v: float(v))
        _bind("para_lag_input",    "parachute_lag",   lambda v: float(v),      lambda v: float(v))
        # Launch geometry — pre-filled defaults (85° / 1.0 m); never use -9999.0 sentinel
        _bind("elev_input",        "launch_angle",    lambda v: float(v),      lambda v: float(v))
        _bind("rail_len_input",    "launch_rail",     lambda v: float(v),      lambda v: float(v))

    # ── Airframe JSON loader ───────────────────────────────────────────────────

    @Slot()
    def _on_load_rocket_json(self) -> None:
        """Open a file dialog, parse the Rocket.json, and push all parameters to AppState.

        Data flow (strict unidirectional MVVM):
            File → load_rocket_config (SI pass-through — no conversion)
                 → airframe SI values → AppState rocket geometry properties
                 → parachute SI values → AppState parachute properties
                 → signals fire → spinboxes / bound widgets update automatically

        AppState equality guards prevent re-emission loops when spinbox
        setValue callbacks fire back into the property setters.
        """
        import os as _os
        path, _ = QFileDialog.getOpenFileName(
            self._window,
            "Load Rocket JSON",
            "",
            "Rocket Config (*.json);;All Files (*)",
        )
        if not path:
            return

        try:
            cfg = load_rocket_config(path)
        except RocketConfigError as exc:
            QMessageBox.critical(
                self._window,
                "Rocket Config Error",
                f"Failed to load:\n{path}\n\n{exc}",
            )
            self._window.set_status(f"Rocket JSON load failed: {exc}", "#f38ba8")
            return

        s   = self._state
        af  = cfg["airframe"]
        par = cfg["parachute"]

        # ── Airframe: JSON is SI; push directly into AppState ─────────────────
        # Each setter emits a signal → bound spinbox.setValue updates the UI.
        # JSON and spinboxes are both SI (m, kg); write directly, no conversion.
        s.rocket_dry_mass = af["mass"]                # kg
        s.rocket_cg       = af["cg"]                  # m from nose
        s.rocket_length   = af["length"]              # m
        s.rocket_diameter = af["radius"] * 2.0        # m radius → m diameter
        s.nose_length     = af["nose_length"]         # m
        s.fin_root_chord  = af["fin_root"]            # m
        s.fin_tip_chord   = af["fin_tip"]             # m
        s.fin_span        = af["fin_span"]            # m
        s.fin_position    = af["fin_pos"]             # m from nose
        s.motor_cg        = af["motor_pos"]           # m from nose
        s.motor_dry_mass  = af["motor_dry_mass"]      # kg
        s.backfire_delay  = af["backfire_delay"]      # s

        # ── Parachute: JSON is SI; push directly into AppState ────────────────
        s.parachute_cd   = par["cd"]                  # dimensionless
        s.parachute_area = par["area"]                # m²
        s.parachute_lag  = par["lag"]                 # s

        name = _os.path.basename(path)
        self._window.set_status(
            f"Rocket loaded: {name}  ·  "
            f"Mass {af['mass']:.4f} kg  ·  "
            f"Length {af['length']:.3f} m  ·  "
            f"Chute {par['area']:.4f} m²  ·  "
            f"Cd {par['cd']:.2f}",
            "#a6e3a1",
        )

    # ── RockSim .rkt file loader ──────────────────────────────────────────────

    @Slot()
    def _on_load_rkt(self) -> None:
        """Open a file dialog, parse the RockSim .rkt file, push all
        parameters (airframe geometry, parachute, MoI) into AppState.

        Error handling
        --------------
        RocketConfigError → QMessageBox.critical (bad format / missing element)
        Any other Exception → QMessageBox.critical + console traceback
        Neither case crashes the application.
        """
        import os as _os
        path, _ = QFileDialog.getOpenFileName(
            self._window,
            "Load RockSim File",
            "",
            "RockSim Files (*.rkt);;All Files (*)",
        )
        if not path:
            return

        try:
            cfg = parse_rkt_file(path)
        except RocketConfigError as exc:
            print(f"[_on_load_rkt] RocketConfigError: {exc}")
            QMessageBox.critical(
                self._window, "RKT File Error",
                f"Failed to parse:\n{path}\n\n{exc}",
            )
            self._window.set_status(f".rkt parse failed: {exc}", "#f38ba8")
            return
        except Exception as exc:
            print(f"[_on_load_rkt] Unexpected error: {type(exc).__name__}: {exc}")
            QMessageBox.critical(
                self._window, "RKT File Error",
                f"Unexpected error loading:\n{path}\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            self._window.set_status(f".rkt load error: {exc}", "#f38ba8")
            return

        s   = self._state
        af  = cfg["airframe"]
        par = cfg["parachute"]
        moi = cfg["moi"]

        # ── Airframe geometry → AppState ───────────────────────────────────────
        # Each setter emits its signal → bound spinbox updates automatically.
        s.rocket_dry_mass = af["mass"]
        s.rocket_cg       = af["cg"]
        s.rocket_length   = af["length"]
        s.rocket_diameter = af["radius"] * 2.0
        s.nose_length     = af["nose_length"]
        s.fin_root_chord  = af["fin_root"]
        s.fin_tip_chord   = af["fin_tip"]
        s.fin_span        = af["fin_span"]
        s.fin_position    = af["fin_pos"]
        s.motor_cg        = af["motor_pos"]
        s.motor_dry_mass  = af["motor_dry_mass"]
        s.backfire_delay  = af["backfire_delay"]

        # ── Parachute → AppState ───────────────────────────────────────────────
        s.parachute_cd   = par["cd"]
        s.parachute_area = par["area"]
        s.parachute_lag  = par["lag"]

        # ── MoI → AppState (emits moi_updated signal) ──────────────────────────
        s.set_moi(moi["ixx"], moi["iyy"], moi["izz"])

        name = _os.path.basename(path)
        w = self._window
        w.rkt_label.setText(name)
        w.rkt_label.setStyleSheet(
            "color: #a6e3a1; font-style: normal; font-size: 8pt; padding: 2px 4px;")
        w.set_status(
            f"RKT loaded: {name}  ·  "
            f"Mass {af['mass']:.3f} kg  ·  "
            f"CG {af['cg']:.3f} m  ·  "
            f"Iyy {moi['iyy']:.4f} kg·m²",
            "#a6e3a1",
        )

    # ── Parachute JSON loader ─────────────────────────────────────────────────

    @Slot()
    def _on_load_para_json(self) -> None:
        """Open a file dialog, parse a Parachute JSON file, push cd/area/lag
        into AppState.

        Error handling
        --------------
        RocketConfigError → QMessageBox.critical (bad values)
        Any other Exception → QMessageBox.critical + console print
        """
        import os as _os
        path, _ = QFileDialog.getOpenFileName(
            self._window,
            "Load Parachute JSON",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return

        try:
            par = parse_parachute_json(path)
        except RocketConfigError as exc:
            print(f"[_on_load_para_json] RocketConfigError: {exc}")
            QMessageBox.critical(
                self._window, "Parachute JSON Error",
                f"Failed to parse:\n{path}\n\n{exc}",
            )
            self._window.set_status(f"Parachute JSON failed: {exc}", "#f38ba8")
            return
        except Exception as exc:
            print(f"[_on_load_para_json] Unexpected: {type(exc).__name__}: {exc}")
            QMessageBox.critical(
                self._window, "Parachute JSON Error",
                f"Unexpected error:\n{path}\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            self._window.set_status(f"Parachute JSON error: {exc}", "#f38ba8")
            return

        s = self._state
        s.parachute_cd   = par["cd"]
        s.parachute_area = par["area"]
        s.parachute_lag  = par["lag"]

        name = _os.path.basename(path)
        self._window.set_status(
            f"Parachute loaded: {name}  ·  "
            f"Cd {par['cd']:.2f}  ·  "
            f"Area {par['area']:.4f} m²  ·  "
            f"Lag {par['lag']:.2f} s",
            "#a6e3a1",
        )

    # ── Advanced Settings dialog ───────────────────────────────────────────────

    @Slot()
    def _on_advanced_settings_clicked(self) -> None:
        """Populate the Advanced Settings dialog from AppState, then exec it.

        Before exec(): push current AppState values into all dialog fields so the
        operator always sees up-to-date numbers regardless of how state was last
        changed (e.g. an external wind reading updated surf_wind_speed).

        After OK: pull all field values back into AppState so _collect_params()
        reads the freshly confirmed settings at the next Run click.

        Cancel: AdvancedSettingsDialog._on_cancel() restores its internal snapshot
        (taken at showEvent); AppState is never written, so it remains consistent.
        """
        dlg = getattr(self._window, '_adv_dialog', None)
        if dlg is None:
            return

        s = self._state

        # ── Populate dialog from AppState ──────────────────────────────────────
        dlg.surf_spd_input.setValue(s.surf_wind_speed)
        dlg.surf_dir_input.setValue(s.surf_wind_dir)
        dlg.up_spd_input.setValue(s.upper_wind_speed)
        dlg.up_dir_input.setValue(s.upper_wind_dir)
        dlg.cep_prob_input.setValue(s.landing_prob)
        dlg.mc_runs_input.setValue(s.mc_n_runs)
        dlg.wind_unc_input.setValue(s.wind_uncertainty)
        dlg.thrust_unc_input.setValue(s.thrust_uncertainty)

        # ── Execute dialog ─────────────────────────────────────────────────────
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return   # Cancel — snapshot already restored by dialog itself

        self._on_settings_confirmed(dlg)

    def _on_settings_confirmed(self, dlg) -> None:
        """Push accepted dialog values to AppState and trigger smart redraw.

        Called after the Advanced Settings dialog is accepted.  No worker is
        started — `_on_partial_redraw` only recomputes the CEP ellipse from the
        already-cached MC scatter and repaints the overlay artists.
        """
        s = self._state
        s.surf_wind_speed    = float(dlg.surf_spd_input.value())
        s.surf_wind_dir      = float(dlg.surf_dir_input.value())
        s.upper_wind_speed   = float(dlg.up_spd_input.value())
        s.upper_wind_dir     = float(dlg.up_dir_input.value())
        s.landing_prob       = int(dlg.cep_prob_input.value())
        s.mc_n_runs          = int(dlg.mc_runs_input.value())
        s.wind_uncertainty   = float(dlg.wind_unc_input.value())
        s.thrust_uncertainty = float(dlg.thrust_unc_input.value())
        s.cep_probability    = float(dlg.cep_prob_input.value())  # drives smart redraw

        self._window.set_status(
            f"Settings updated  ·  "
            f"Surf {s.surf_wind_speed:.1f} m/s  ·  "
            f"MC {s.mc_n_runs} runs  ·  "
            f"Wind unc {s.wind_uncertainty:.2f}",
            "#a6adc8",
        )

        self._on_partial_redraw()  # recompute ellipse + repaint, no simulation

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
        sc   = result.get("scatter_points", result.get("scatter", []))
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

    @staticmethod
    def _adapt_nominal_for_window(payload: dict) -> dict:
        """Adapt a sig_nominal_done payload so AppWindow._draw_real_result can render it.

        Translates ``x_vals / y_vals / z_vals`` → ``trajectory_x/y/z`` and
        fills in empty MC arrays.  All original keys (phases, events,
        wind_nodes, …) are preserved so _draw_real_result gets full phase data.
        """
        adapted = dict(payload)
        adapted.update({
            "trajectory_x": [float(v) for v in payload.get("x_vals", [])],
            "trajectory_y": [float(v) for v in payload.get("y_vals", [])],
            "trajectory_z": [float(v) for v in payload.get("z_vals", [])],
            "mc_scatter_x": [],
            "mc_scatter_y": [],
            "land_x":       float(payload.get("impact_x", 0.0)),
            "land_y":       float(payload.get("impact_y", 0.0)),
            "cep_ellipses": [],
        })
        return adapted

    @Slot()
    def _on_partial_redraw(self) -> None:
        """Recompute the error ellipse from cached scatter at the new probability.

        Called whenever cep_probability changes (slider move).  Runs entirely
        from cached_mc_scatter — no simulation, no Monte Carlo loop.

        Only the ellipse is recalculated (numpy eigendecomposition, ~1 ms on
        the GUI thread — safe).  KDE contours are expensive and intentionally
        left unchanged.  The SimulationWorker is NOT started.
        """
        pts = self._state.cached_mc_scatter
        ellipse_data: dict | None = None
        if pts is not None and len(pts) >= 4:
            prob         = self._state.cep_probability / 100.0   # 90.0 → 0.90
            ellipse_data = compute_cep_ellipse(pts, prob)
            self._state.mc_ellipse = ellipse_data
            # Overwrite the ellipse in BOTH cached result dicts.  Without this,
            # any full repaint (wind-tick → update_map_plot, or refresh_visuals)
            # reads the stale worker-written ellipse and reverts the overlay.
            sr = self._state.simulation_result
            if isinstance(sr, dict):
                sr["ellipse"] = ellipse_data
            wsr = self._window.state.simulation_result
            if isinstance(wsr, dict):
                wsr["ellipse"] = ellipse_data
        self._window.update_ellipse_layer(ellipse_data)

    @Slot(float)
    def _on_cep_prob_changed(self, value: float) -> None:
        """Propagate cep_prob_input change → both AppState probability fields.

        cep_probability drives partial visual redraws (float, emits needs_partial_redraw).
        landing_prob is the authoritative integer used by _collect_params → worker.
        Both must stay in sync so the map overlay and the MC computation always
        use the same percentile, regardless of which widget last changed the value.
        """
        self._state.cep_probability = float(value)
        self._state.landing_prob    = int(round(value))

    def _set_run_buttons_enabled(self, enabled: bool) -> None:
        for btn in self._window.findChildren(QPushButton, "btn_run"):
            btn.setEnabled(enabled)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Task 4: enforce high-contrast dark theme globally before any widget is shown.
    app.setStyleSheet(GLOBAL_QSS)

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
