"""
ui_qt/sim_controller.py
Application-level controller that mediates between AppWindow, AppState, and workers.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Slot, Signal
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QPushButton, QSystemTrayIcon, QStyle

from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.workers import SimulationWorker, OptimizationWorker, MapDownloadWorker
from core.monte_carlo  import compute_cep_ellipse
from utils.data_loader import (
    RocketConfigError,
    load_rocket_config,
    parse_parachute_json,
    parse_rkt_file,
)

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
        print(f"=== SimController.__init__ === Received AppState: id={id(self._state)}")
        self._worker: SimulationWorker | None = None
        # Cache the sig_nominal_done payload so _on_mc_done can forward
        # phases/events (phase-coloured 3-D trajectory) into the final result.
        self._nominal_payload: dict | None    = None

        # ── Setup System Tray Icon ─────────────────────────────────────────────
        self._tray_icon = QSystemTrayIcon(self)
        self._tray_icon.setIcon(self._window.style().standardIcon(QStyle.SP_ComputerIcon))
        self._tray_icon.show()

        self._rewire_buttons()

        # Bind global state signals to UI labels
        self._state.koinobori_status_changed.connect(
            lambda v: self._window.lbl_koinobori_status.setText(f"Koinobori: {v}"))
        self._state.gpv_last_fetch_time_changed.connect(
            lambda v: self._window.lbl_gpv_status.setText(f"GPV Updated: {v}"))

        # Initialize labels with current values
        self._window.lbl_koinobori_status.setText(f"Koinobori: {self._state.koinobori_status}")
        self._window.lbl_gpv_status.setText(f"GPV Updated: {self._state.gpv_last_fetch_time}")

        # ── Parameter readiness interlock ──────────────────────────────────────
        # RUN buttons start enabled; validation happens inside the slots to
        # avoid silent disablement and provide user feedback.
        self._set_run_buttons_enabled(True)

        # NOTE: needs_redraw → update_profile_plot / update_map_plot / update_wind_plot
        # connections are established inside AppWindow.bind_app_state() (called before
        # SimController.__init__).  Connecting again here would cause every emit to
        # trigger each canvas slot TWICE, producing a redraw storm on the GUI thread
        # and causing the "Not Responding" freeze on Windows.

        # ── Bi-directional Map Sync ────────────────────────────────────────────
        if hasattr(self._window, 'map_widget') and hasattr(self._window.map_widget, 'coordinates_picked'):
            self._window.map_widget.coordinates_picked.connect(self._on_map_coordinates_picked)

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

        # ── Launch Coordinates → AppState.launch_lat / launch_lon ──────────────
        _lat_input = getattr(window, 'lat_input', None)
        if _lat_input is not None:
            state.launch_lat = float(_lat_input.value())
            _lat_input.valueChanged.connect(
                lambda v: setattr(state, 'launch_lat', float(v))
            )

        _lon_input = getattr(window, 'lon_input', None)
        if _lon_input is not None:
            state.launch_lon = float(_lon_input.value())
            _lon_input.valueChanged.connect(
                lambda v: setattr(state, 'launch_lon', float(v))
            )

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

        # NOTE: simulation_result_changed → update_map_plot and update_profile_plot
        # are NOT connected here.  The simulation_result.setter already emits
        # needs_redraw which is wired (in AppWindow.bind_app_state) to all three
        # canvas update slots.  Adding simulation_result_changed connections on
        # top would cause each plot slot to execute twice per MC completion,
        # doubling the GUI-thread work and causing "Not Responding" freezes.


        # ── Flight-mode → map circle switch ───────────────────────────────────
        # update_map_plot reads window.state.sim_mode (already kept in sync by
        # mode_combo → window.state.sim_mode binding in _bind_state), but no
        # redraw is triggered on mode change.  Fire one here.
        state.flight_mode_changed.connect(lambda _: window.update_map_plot())

    # ── Button rewiring ────────────────────────────────────────────────────────

    def _rewire_buttons(self) -> None:
        """
        Redirect every btn_run / btn_stop in the widget tree to controller
        slots.
        """
        for btn in self._window.findChildren(QPushButton, "btn_run"):
            btn.clicked.connect(self._on_run_clicked)

        for btn in self._window.findChildren(QPushButton, "btn_stop"):
            btn.clicked.connect(self._on_stop_clicked)

        for btn in self._window.findChildren(QPushButton, "btn_phase1_run"):
            btn.clicked.connect(self._on_phase1_clicked)

        if hasattr(self._window, "btn_download_map"):
            self._window.btn_download_map.clicked.connect(self._on_download_map_clicked)
        for btn in self._window.findChildren(QPushButton, "btn_download_map"):
            btn.clicked.connect(self._on_download_map_clicked)

    # ── Run ────────────────────────────────────────────────────────────────────

    def _validate_run_prerequisites(self) -> bool:
        print("[DIAG] _validate_run_prerequisites CALLED! Stack trace follows:", flush=True)
        import traceback
        traceback.print_stack()
        
        missing = []
        s = self._state

        # Rocket Geometry
        if any(v is None for v in (
            s._rocket_dry_mass, s._rocket_cg, s._rocket_length, s._rocket_diameter,
            s._nose_length, s._fin_root_chord, s._fin_tip_chord, s._fin_span,
            s._fin_position
        )):
            missing.append("Rocket Geometry (All fields in Airframe tab must be filled manually or loaded via JSON)")

        if s.motor_cg_pos is None or s.motor_dry_mass is None:
            missing.append("Motor Physical Parameters (Motor CG Pos and Motor Dry Mass must be filled manually)")

        # Parachute Parameters
        if any(v is None for v in (s._parachute_cd, s._parachute_area, s._parachute_lag)):
            missing.append("Parachute Parameters (All fields must be filled manually or loaded via JSON)")

        # Backfire Delay
        if s._backfire_delay is None:
            missing.append("Backfire Delay (Must be filled manually)")

        print("=== COORDINATE FORENSICS ===")
        print(f"1. UI SpinBox LAT: {self._window.lat_input.value()} (Type: {type(self._window.lat_input.value())})")
        print(f"2. UI SpinBox LON: {self._window.lon_input.value()} (Type: {type(self._window.lon_input.value())})")
        print(f"3. AppState Getter LAT: {self._state.launch_lat} (Type: {type(self._state.launch_lat)})")
        print(f"4. AppState Getter LON: {self._state.launch_lon} (Type: {type(self._state.launch_lon)})")
        try:
            print(f"5. AppState Raw _launch_lat: {self._state._launch_lat}")
        except AttributeError:
            print("5. AppState Raw _launch_lat: DOES NOT EXIST")
        print("============================")

        # Launch Coordinates
        if any(v is None for v in (s._launch_lat, s._launch_lon)) or \
           self._window.lat_input.value() == -9999.0 or \
           self._window.lon_input.value() == -9999.0:
            missing.append("Launch Coordinates (Latitude/Longitude)")

        # Rail Azimuth
        if self._window.azim_input.value() == -9999.0:
            missing.append("Rail Azimuth")

        # Simulation Uncertainty Params
        if any(v is None for v in (s._wind_uncertainty, s._thrust_uncertainty)):
            missing.append("Simulation Uncertainty Parameters")

        # Target Radius
        is_free_mode = "free" in str(s._flight_mode).lower() or "自由" in str(s._flight_mode).lower()
        if not is_free_mode and s._target_radius is None:
            missing.append("Target Radius")

        # Motor Thrust Data
        if not getattr(self._window, '_motor_thrust_data', None):
            missing.append("Motor Thrust Curve (.csv)")

        if missing:
            print("[DIAG] _validate_run_prerequisites FAILED: Missing Parameters: " + ", ".join(missing), flush=True)
            self._window.set_status("Missing Parameters: " + ", ".join(missing), "#f38ba8")
            return False

        # ── 60-second surface wind buffer check ───────────────────────────────
        surface_hist = list(self._state.wind_history_for_alt(3.0))
        print(f"[DIAG] wind_history_for_alt(3.0) sample count: {len(surface_hist)}")
        if len(surface_hist) < 5:
            print(f"[DIAG] BLOCKED: Wind buffer insufficient ({len(surface_hist)} samples < 5). Returning False.")
            self._window.set_status(
                f"Wind Buffer Insufficient: Surface wind monitor has only {len(surface_hist)} samples "
                "(minimum 5 required). Please wait for the monitor to collect data.",
                "#f38ba8"
            )
            return False

        print("[DIAG] _validate_run_prerequisites: PASSED — returning True")
        return True

    @Slot()
    def _on_download_map_clicked(self) -> None:
        print("[DIAG] DOWNLOAD BUTTON CLICKED - Slot triggered.", flush=True)
        if self._worker is not None and self._worker.isRunning():
            return

        lat_text = self._window.lat_input.cleanText()
        lon_text = self._window.lon_input.cleanText()

        if not lat_text or not lon_text:
            self._window.set_status("Invalid Coordinates: Coordinates are not entered. Please enter valid launch pad coordinates first.", "#f38ba8")
            return

        try:
            lat = float(lat_text)
            lon = float(lon_text)
        except ValueError:
            self._window.set_status("Invalid Coordinates: Coordinates are not entered or invalid.", "#f38ba8")
            return

        self._state.launch_lat = lat
        self._state.launch_lon = lon

        self._state.is_calculating = True
        self._state.status_text = "Downloading Map..."

        # Instantiate and start the worker
        self._worker = MapDownloadWorker(lat, lon, parent=self)
        self._worker.sig_progress.connect(self._on_progress_updated)
        self._worker.sig_finished.connect(self._on_download_map_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    @Slot(dict)
    def _on_download_map_finished(self, meta: dict) -> None:
        self._state.is_calculating = False
        self._state.status_text = "Map Download Complete"

        declination = meta.get("magnetic_declination", 0.0)
        self._state.magnetic_declination = declination

        # Force map view to reload the new background tiles and redraw
        self._state.simulation_result_changed.emit(None)
        self._state.needs_redraw.emit()

        self._window.set_status(
            f"Download Complete: Offline map tiles downloaded. "
            f"Magnetic Declination: {declination:.4f}°",
            "#a6e3a1",
        )

    @Slot()
    def _on_run_clicked(self) -> None:
        print("[DIAG] RUN SIMULATION BUTTON CLICKED - Slot triggered.", flush=True)
        print(f"\n{'='*60}")
        print(f"[DIAG] _on_run_clicked ENTERED")
        print(f"[DIAG]   AppState id        : {id(self._state)}")
        try:
            _is_running = self._worker.isRunning() if self._worker else False
        except RuntimeError:
            # C++ object already deleted by deleteLater — treat as idle
            self._worker = None
            _is_running = False
        print(f"[DIAG]   _worker running?   : {_is_running}")
        if _is_running:
            print("[DIAG] EARLY RETURN: worker already running")
            return  # guard against double-click spam

        print(f"[DIAG]   motor_cg_pos       : {self._state.motor_cg_pos}")
        print(f"[DIAG]   motor_dry_mass     : {self._state.motor_dry_mass}")
        if self._state.motor_cg_pos is None or self._state.motor_dry_mass is None:
            print("[DIAG] EARLY RETURN: motor params missing")
            self._window.set_status("Validation Error: Motor Physical Parameters must be set before running the simulation.", "#f38ba8")
            return

        print("[DIAG] Calling _validate_run_prerequisites...")
        if not self._validate_run_prerequisites():
            print("[DIAG] EARLY RETURN: _validate_run_prerequisites returned False")
            return

        print("[DIAG] Prerequisites PASSED — stopping wind timer")
        # Pause the 1 Hz wind monitor during the simulation run.
        self._wind_timer.stop()

        self._state.mc_running = True
        self._state.simulation_started.emit()
        self._state.is_calculating = True
        self._set_run_buttons_enabled(False)
        self._window.set_status("Simulation running...", "#f9e2af")
        self._window.set_progress(0, "Simulating...")

        # Clear stale data from the previous run.
        # IMPORTANT: bypass the public setter to avoid emitting needs_redraw
        # which would trigger synchronous Matplotlib redraws (~1 s) on the GUI
        # thread and block progress-bar updates from the worker.
        self._nominal_payload                 = None
        self._state._simulation_result = None
        self._state.current_playback_index = 0
        self._window.update_map_plot()

        print("[DIAG] Creating SimulationWorker...")
        try:
            collected = self._collect_params()
            print(f"[DIAG] _collect_params() succeeded — keys: {list(collected.keys())}")
        except Exception as _cp_exc:
            import traceback as _tb
            print(f"[DIAG] _collect_params() RAISED: {_cp_exc}")
            print(_tb.format_exc())
            self._window.set_status(f"Parameter collection failed: {_cp_exc}", "#f38ba8")
            self._wind_timer.start()
            self._state.is_calculating = False
            self._state.mc_running = False
            self._set_run_buttons_enabled(True)
            return

        self._worker = SimulationWorker(collected, parent=self)
        print(f"[DIAG] SimulationWorker created: {self._worker}")
        
        from PySide6.QtCore import Qt
        
        self._worker.progress.connect(self._on_progress, Qt.QueuedConnection)
        # ── Two-stage routing ──────────────────────────────────────────────────
        # sig_nominal_done fires before any MC runs start; renders trajectory now.
        self._worker.sig_nominal_done.connect(self._on_nominal_done, Qt.QueuedConnection)
        # sig_progress fires after every single MC iteration (current, total, msg).
        self._worker.sig_progress.connect(self._on_progress_updated, Qt.QueuedConnection)
        # sig_finished covers both cancelled and full-MC-done paths (replaces _on_finished).
        self._worker.sig_finished.connect(self._on_mc_done, Qt.QueuedConnection)
        self._worker.error.connect(self._on_error, Qt.QueuedConnection)
        self._worker.sig_status_text.connect(self._on_worker_status, Qt.QueuedConnection)
        self._worker.sig_early_warning.connect(self._on_early_warning, Qt.QueuedConnection)
        # Auto-cleanup: schedule C++ deletion, then null the Python wrapper so
        # subsequent isRunning() guards don't hit a dangling shiboken pointer.
        self._worker.finished.connect(self._worker.deleteLater, Qt.QueuedConnection)
        self._worker.finished.connect(lambda: setattr(self, '_worker', None), Qt.QueuedConnection)
        print("[DIAG] Calling self._worker.start()...")
        self._worker.start()
        print(f"[DIAG] worker.start() called — isRunning={self._worker.isRunning()}")
        print(f"{'='*60}\n")
        # Note: LowPriority was removed — on Windows, LowPriority causes the
        # OS scheduler to starve the worker thread (practically 0 CPU time),
        # making the MC loop appear to never finish. Default (InheritPriority)
        # ensures fair scheduling between the GUI thread and the worker.


    @Slot()
    def _on_phase1_clicked(self) -> None:
        print("[DIAG] RUN SIMULATION BUTTON CLICKED - Slot triggered.", flush=True)
        try:
            if self._worker and self._worker.isRunning():
                return  # guard against double-click spam
        except RuntimeError:
            # C++ object already deleted by deleteLater — treat as idle
            self._worker = None

        if self._state.motor_cg_pos is None or self._state.motor_dry_mass is None:
            print("[DIAG] EARLY RETURN: motor params missing", flush=True)
            self._window.set_status("Validation Error: Motor Physical Parameters must be set before running the simulation.", "#f38ba8")
            return

        if not self._validate_run_prerequisites():
            return

        # Pause the 1 Hz wind monitor during optimisation + MC run.
        self._wind_timer.stop()

        self._state.mc_running = True
        self._state.simulation_started.emit()
        self._state.is_calculating = True
        self._set_run_buttons_enabled(False)
        self._window.set_status("Optimisation running...", "#fab387")
        self._window.set_progress(0, "Optimising...")

        # Clear stale data from the previous run.
        # IMPORTANT: bypass the public setter to avoid emitting needs_redraw
        # which would trigger synchronous Matplotlib redraws (~1 s) on the GUI
        # thread and block progress-bar updates from the worker.
        self._nominal_payload                 = None
        self._state._simulation_result = None
        self._state.current_playback_index = 0
        self._window.update_map_plot()

        try:
            collected = self._collect_params()
        except Exception as _cp_exc:
            import traceback as _tb
            print(f"[DIAG] _on_phase1_clicked: _collect_params() RAISED: {_cp_exc}")
            print(_tb.format_exc())
            self._window.set_status(f"Parameter collection failed: {_cp_exc}", "#f38ba8")
            self._wind_timer.start()
            self._state.is_calculating = False
            self._state.mc_running = False
            self._set_run_buttons_enabled(True)
            return

        from PySide6.QtCore import Qt
        if self._state.is_free_mode:
            self._worker = SimulationWorker(collected, parent=self)
            self._worker.progress.connect(self._on_progress, Qt.QueuedConnection)
            self._worker.sig_nominal_done.connect(self._on_nominal_done, Qt.QueuedConnection)
            self._worker.sig_progress.connect(self._on_progress_updated, Qt.QueuedConnection)
            self._worker.sig_finished.connect(self._on_mc_done, Qt.QueuedConnection)
            self._worker.error.connect(self._on_error, Qt.QueuedConnection)
            self._worker.sig_status_text.connect(self._on_worker_status, Qt.QueuedConnection)
            self._worker.sig_early_warning.connect(self._on_early_warning, Qt.QueuedConnection)
            self._worker.finished.connect(self._worker.deleteLater, Qt.QueuedConnection)
            self._worker.finished.connect(lambda: setattr(self, '_worker', None), Qt.QueuedConnection)
            self._worker.start()
        else:
            self._worker = OptimizationWorker(collected, parent=self)
            self._worker.progress.connect(self._on_progress, Qt.QueuedConnection)
            self._worker.sig_nominal_done.connect(self._on_nominal_done, Qt.QueuedConnection)
            self._worker.sig_progress.connect(self._on_progress_updated, Qt.QueuedConnection)
            self._worker.sig_finished.connect(self._on_mc_done, Qt.QueuedConnection)
            self._worker.error.connect(self._on_error, Qt.QueuedConnection)
            self._worker.sig_status_text.connect(self._on_worker_status, Qt.QueuedConnection)
            self._worker.sig_early_warning.connect(self._on_early_warning, Qt.QueuedConnection)
            self._worker.sig_optimization_done.connect(self._on_optimization_done, Qt.QueuedConnection)
            self._worker.finished.connect(self._worker.deleteLater, Qt.QueuedConnection)
            self._worker.finished.connect(lambda: setattr(self, '_worker', None), Qt.QueuedConnection)
            self._worker.start()
        # Note: LowPriority was removed — see _on_run_clicked for rationale.

    @Slot(float, float)
    def _on_optimization_done(self, elev: float, azi: float) -> None:
        """Update the UI dynamically when the optimal angle is found"""
        self._state.launch_angle = elev
        self._window.azim_input.setValue(azi)
        self._window.elev_input.setValue(elev)


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
        self._window.set_progress(value)

    def populate_results(self, nominal_data: dict, mc_data: dict) -> None:
        """Populates the Simulation Results panel in AppWindow based on mode."""
        w = self._window
        w._results_grp.setVisible(True)

        mode = self._state.flight_mode
        is_free = "free" in mode.lower() or "自由" in mode

        def set_val(lbl, val, fmt, suffix=""):
            if val is not None and val != "N/A":
                lbl.setText(f"{val:{fmt}}{suffix}")
            else:
                lbl.setText("N/A" if val == "N/A" else "-")

        if is_free:
            set_val(w.lbl_res_apogee, nominal_data.get("apogee_m"), ".1f", " m")
            set_val(w.lbl_res_hdist, nominal_data.get("horizontal_distance_m"), ".1f", " m")
            set_val(w.lbl_res_hang, nominal_data.get("hang_time"), ".1f", " s")

            set_val(w.lbl_res_mc_avg_apo, mc_data.get("mc_avg_apogee", "N/A"), ".1f", " m")
            set_val(w.lbl_res_mc_min_apo, mc_data.get("mc_min_apogee", "N/A"), ".1f", " m")
            set_val(w.lbl_res_mc_avg_hdist, mc_data.get("mc_avg_hdist", "N/A"), ".1f", " m")
            set_val(w.lbl_res_mc_max_hdist, mc_data.get("mc_max_hdist", "N/A"), ".1f", " m")

        elif mode == "定点滞空":
            set_val(w.lbl_res_angle, nominal_data.get("elev", mc_data.get("elev")), ".1f", " °")
            set_val(w.lbl_res_azimuth, nominal_data.get("azi", mc_data.get("azi")), ".1f", " °")
            set_val(w.lbl_res_score, nominal_data.get("score", mc_data.get("score")), ".2f")
            set_val(w.lbl_res_hdist, nominal_data.get("horizontal_distance_m"), ".1f", " m")
            set_val(w.lbl_res_hang, nominal_data.get("hang_time"), ".1f", " s")

            set_val(w.lbl_res_mc_avg_score, mc_data.get("mc_avg_score", "N/A"), ".2f")
            set_val(w.lbl_res_mc_min_score, mc_data.get("mc_min_score", "N/A"), ".2f")

        elif mode == "高度":
            set_val(w.lbl_res_angle, nominal_data.get("elev", mc_data.get("elev")), ".1f", " °")
            set_val(w.lbl_res_azimuth, nominal_data.get("azi", mc_data.get("azi")), ".1f", " °")
            set_val(w.lbl_res_apogee, nominal_data.get("apogee_m"), ".1f", " m")
            set_val(w.lbl_res_hdist, nominal_data.get("horizontal_distance_m"), ".1f", " m")

            set_val(w.lbl_res_mc_avg_apo, mc_data.get("mc_avg_apogee", "N/A"), ".1f", " m")
            set_val(w.lbl_res_mc_min_apo, mc_data.get("mc_min_apogee", "N/A"), ".1f", " m")
            set_val(w.lbl_res_mc_max_hdist, mc_data.get("mc_max_hdist", "N/A"), ".1f", " m")

        elif mode == "有翼":
            set_val(w.lbl_res_angle, nominal_data.get("elev", mc_data.get("elev")), ".1f", " °")
            set_val(w.lbl_res_azimuth, nominal_data.get("azi", mc_data.get("azi")), ".1f", " °")
            set_val(w.lbl_res_hang, nominal_data.get("hang_time"), ".1f", " s")
            set_val(w.lbl_res_hdist, nominal_data.get("horizontal_distance_m"), ".1f", " m")

            set_val(w.lbl_res_mc_avg_hang, mc_data.get("mc_avg_hang_time", mc_data.get("mc_avg_hang", "N/A")), ".1f", " s")
            set_val(w.lbl_res_mc_min_hang, mc_data.get("mc_min_hang_time", mc_data.get("mc_min_hang", "N/A")), ".1f", " s")
            set_val(w.lbl_res_mc_max_hdist, mc_data.get("mc_max_hdist", "N/A"), ".1f", " m")

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
        # Pre-populate _simulation_result without emitting needs_redraw so that
        # update_profile_plot (triggered below by nominal_needs_redraw) can read
        # the data.  Using the private field avoids the extra needs_redraw blast
        # that the public setter would emit — _on_nominal_done must touch the
        # profile plot exactly ONCE to keep the GUI thread responsive.
        print("[DIAG] _on_nominal_done ENTERED on GUI thread", flush=True)

        # simulation_result_changed is still emitted so any other subscriber
        # (e.g. map_view) gets notified without triggering a full redraw cycle.
        adapted_nominal = self._adapt_nominal_for_window(payload)
        # Write the adapted nominal result into AppState WITHOUT emitting
        # needs_redraw.  The public setter fires needs_redraw → update_profile_plot,
        # and then nominal_result (set below) fires nominal_needs_redraw →
        # update_profile_plot AGAIN — two consecutive 3D redraws on the GUI thread
        # that block progress-bar updates for ~1 second.  By using the private
        # field + manual simulation_result_changed, MapView and other subscribers
        # still get notified, but update_profile_plot fires only ONCE via
        # nominal_needs_redraw below.
        self._state._simulation_result = adapted_nominal
        self._state.simulation_result_changed.emit(adapted_nominal)

        # Setting nominal_result fires nominal_result_changed and
        # nominal_needs_redraw; the latter triggers update_profile_plot (once).
        self._state.nominal_result = payload
        self._nominal_payload      = payload

        # Update the UI with nominal data, leaving MC data as None (displays "-")
        self.populate_results(payload, {})

        # Early warning: update GO/NO-GO indicator immediately after nominal run,
        # before the MC loop starts, so the operator never waits blindly.
        # FREE MODE: no target radius concept — skip the boundary check entirely.
        _mode_str = str(self._state.flight_mode)
        _is_free_mode = "free" in _mode_str.lower() or "自由" in _mode_str
        _off_e    = float(payload.get("impact_x", 0.0))
        _off_n    = float(payload.get("impact_y", 0.0))
        _tgt_x    = float(getattr(self._state, "target_x", 0.0) or 0.0)
        _tgt_y    = float(getattr(self._state, "target_y", 0.0) or 0.0)
        _nom_dist = math.hypot(_off_e - _tgt_x, _off_n - _tgt_y)
        _target_r = self._state.target_radius
        print(
            f"[DIAG] _on_nominal_done GO/NOGO check: "
            f"mode={_mode_str!r} is_free={_is_free_mode} "
            f"nom_dist={_nom_dist:.1f}m target_r={_target_r}",
            flush=True,
        )
        if _is_free_mode:
            # Free Mode: distance from launch pad is informational only — never NOGO.
            self._window.update_status_indicator(
                f"⏳  FREE MODE — Nominal dist {_nom_dist:.0f} m  |  MC running…"
            )
        elif _target_r is None:
            print("[DIAG] NOGO guard: target_r is None in non-Free mode — skipping indicator", flush=True)
            self._window.update_status_indicator("⏳  CALCULATING — MC running…")
        elif _nom_dist > _target_r:
            print(
                f"[DIAG] NOGO: _on_nominal_done — nominal {_nom_dist:.1f}m > target_r {_target_r:.1f}m",
                flush=True,
            )
            self._window.update_status_indicator(
                f"⚠️  NO-GO — Nominal {_nom_dist:.0f} m > Target {_target_r:.0f} m  |  MC running…"
            )
        else:
            self._window.update_status_indicator(
                f"⏳  CALCULATING — Nominal {_nom_dist:.0f} m / {_target_r:.0f} m  |  MC running…"
            )

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
        # (running on the worker thread) monopolises CPU.
        self._window.profile_canvas.draw_idle()


    @Slot(int, int, str)
    def _on_progress_updated(self, current: int, total: int, msg: str) -> None:
        """Invoked on the GUI thread after every single MC run.

        Converts the raw (current, total) heartbeat into a 0–100 percentage
        and pushes it to AppState.progress_percentage.  Division-by-zero is
        guarded: if total == 0, percentage stays at 0.

        Thread safety: sig_progress crosses thread boundaries via Qt's
        queued connection, so the AppState write always happens on the GUI thread.
        """
        pct = int((current / total) * 100) if total > 0 else 0
        self._state.progress_percentage = pct
        self._window._progress.setValue(pct)
        self._window._status_label.setText(msg)

    @Slot(dict)
    def _on_mc_done(self, result: dict) -> None:
        print(f"[DIAG] _on_mc_done ENTERED — cancelled={result.get('cancelled')}", flush=True)
        self._state.mc_running = False

        if result.get("cancelled"):
            self._state.is_calculating    = False
            self._state.progress_percentage = 0   # clear progress bar
            self._window.set_status("Simulation cancelled.", "#a6adc8")
            self._window.set_progress(0, "Idle")
            self._set_run_buttons_enabled(True)
            # Resume 1 Hz wind monitor now that the run has ended.
            self._wind_timer.start()
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

        # Update Results Panel using the new dynamic method
        self.populate_results(self._nominal_payload or {}, result)

        # ── Refresh AppWindow coordinate labels ────────────────────────────────
        self._window.map_widget.update_landing(land_lat, land_lon)

        # ── Populate window state FIRST ────────────────────────────────────────
        # Carry phases/events from the nominal payload into the final result so
        # the phase-coloured 3-D trajectory persists after MC finishes.
        if self._nominal_payload:
            result = dict(result)
            result["phases"] = self._nominal_payload.get("phases")
            result["events"] = self._nominal_payload.get("events")

        # Write the adapted result exactly ONCE through the global AppState.
        # AppWindow.bind_app_state() already replaced self._window.state with
        # self._state, so both references are the same object.  Setting the
        # property fires simulation_result_changed and needs_redraw exactly once
        # each — no duplicate redraws, no GUI thread flooding.
        self._state.simulation_result = self._adapt_for_window(result)

        # ── Phase A verification: CEP50 ≤ target_radius → SAFE → Phase B ────────
        # FREE MODE: no target radius / landing zone concept — skip entirely.
        cep50    = self._state.mc_cep
        target_r = self._state.target_radius
        _mode_str_mc = str(self._state.flight_mode)
        _is_free_mode_mc = "free" in _mode_str_mc.lower() or "自由" in _mode_str_mc
        print(
            f"[DIAG] _on_mc_done Phase A check: "
            f"mode={_mode_str_mc!r} is_free={_is_free_mode_mc} "
            f"cep50={cep50:.1f}m target_r={target_r}",
            flush=True,
        )

        r90    = self._state.r90_radius
        apogee = result.get("apogee_m",  0.0)
        tof    = result.get("hang_time", 0.0)
        n      = result.get("n_runs",    0)
        prob   = result.get("landing_prob", int(self._window.cep_prob_input.value()))

        if _is_free_mode_mc:
            # Free Mode: no SAFE/UNSAFE concept — display informational summary only.
            self._window.update_status_indicator(
                f"🟢  FREE MODE — R{prob}: {r90:.0f} m  |  Apogee: {apogee:.0f} m"
            )
            self._window.set_status(
                f"Free Mode  |  R{prob}: {r90:.1f} m  |  "
                f"Apogee: {apogee:.0f} m  |  ToF: {tof:.1f} s  ({n} runs)",
                "#89dceb",
            )
        elif target_r is None:
            # Non-Free mode but no target radius set — show a neutral warning.
            print("[DIAG] NOGO guard: target_r is None in non-Free mode — cannot evaluate Phase A", flush=True)
            self._window.update_status_indicator("⚠️  target_radius not set — cannot evaluate GO/NO-GO")
            self._window.set_status(
                f"target_radius not set  |  R{prob}: {r90:.1f} m  |  "
                f"Apogee: {apogee:.0f} m  |  ToF: {tof:.1f} s  ({n} runs)",
                "#f9e2af",
            )
        else:
            is_safe = cep50 <= target_r
            print(
                f"[DIAG] Phase A result: cep50={cep50:.1f}m {'<=' if is_safe else '>'} "
                f"target_r={target_r:.1f}m  ->  {'SAFE' if is_safe else 'NOGO'}",
                flush=True,
            )
            if is_safe:
                # Lock the Phase A wind distribution so Phase B O(1) GO/NO-GO ticks
                # compare live wind against the exact baseline used in the MC run.
                lowest_wind = sorted(self._state.wind_profile_data, key=lambda n: n["alt_m"])[0] if self._state.wind_profile_data else {"speed_ms": 0.0, "dir_deg": 0.0}
                surf_spd = result.get("nominal_surf_spd", lowest_wind["speed_ms"])
                surf_dir = result.get("nominal_surf_dir", lowest_wind["dir_deg"])
                wind_unc = float(self._window.wind_unc_input.value())
                mu_u = surf_spd * math.sin(math.radians(surf_dir))
                mu_v = surf_spd * math.cos(math.radians(surf_dir))
                self._state.set_wind_lock(mu_u, mu_v, wind_unc)
                self._state.phase2_active = True
                self._window.update_status_indicator(
                    f"🟢  GO  (CEP {cep50:.0f} m ≤ {target_r:.0f} m)"
                )
            else:
                print(
                    f"[DIAG] NOGO: _on_mc_done — cep50={cep50:.1f}m > target_r={target_r:.1f}m",
                    flush=True,
                )
                self._window.update_status_indicator(
                    f"🔴  NO-GO  (CEP {cep50:.0f} m > {target_r:.0f} m)"
                )
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

        self._set_run_buttons_enabled(True)

        # Resume 1 Hz wind monitor now that the simulation run has ended.
        self._wind_timer.start()

        # Trigger completion notification
        self._tray_icon.showMessage(
            "Kazamidori Simulation",
            "Simulation Complete!",
            QSystemTrayIcon.Information,
            3000
        )

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        # Surface the worker traceback on stderr so silent-window failures
        # produced by background QThread exceptions become diagnosable.
        import sys as _sys
        print("[DIAG] _on_error ENTERED:", flush=True)
        print("--- SIMULATION WORKER ERROR ---", file=_sys.stderr, flush=True)
        print(msg, file=_sys.stderr, flush=True)
        print("-------------------------------", file=_sys.stderr, flush=True)

        self._state.mc_running = False
        self._state.is_calculating = False
        self._window.set_status(f"Simulation error: {msg}", "#f38ba8")
        self._window.set_progress(0, "Error")
        self._set_run_buttons_enabled(True)

        # Resume 1 Hz wind monitor even on error so telemetry is not lost.
        self._wind_timer.start()

        self._tray_icon.showMessage(
            "Simulation Error",
            msg,
            QSystemTrayIcon.Warning,
            5000
        )

    @Slot(str)
    def _on_worker_status(self, msg: str) -> None:
        print(f"[DIAG] _on_worker_status: {msg!r}", flush=True)
        # SAFE-ONLY UPDATE: touch ONLY plain QLabel widgets via setText().
        # Do NOT call set_status() or set_progress() here — those methods call
        # QProgressBar.setValue() / setFormat() which enqueue a Qt repaint event.
        # If this slot fires while Matplotlib's Qt backend (backend_qt.py) is
        # initialising its canvas, the repaint re-enters the Qt event loop from
        # within an active paint handler, producing the observed GUI deadlock.
        # Text labels are synchronous property writes that bypass the render queue.
        self._window._status_label.setText(msg)
        self._window._status_label.setStyleSheet("color: #f9e2af; padding-left: 8px;")
        _phase_lbl = getattr(self._window, '_phase_label', None)
        if _phase_lbl is not None:
            _phase_lbl.setText(msg)

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
            lowest_wind = sorted(s.wind_profile_data, key=lambda n: n["alt_m"])[0] if s.wind_profile_data else {"speed_ms": 0.0, "dir_deg": 0.0}
            _turb_mu    = lowest_wind["speed_ms"]
            _turb_sigma = 0.0
        # I = σ/μ  (dimensionless turbulence intensity)
        _turb_intensity = _turb_sigma / _turb_mu if _turb_mu > 0.1 else 0.0

        try:
            safe_motor_pos      = float(s.motor_cg_pos)
            safe_motor_dry_mass = float(s.motor_dry_mass)
        except (TypeError, ValueError):
            safe_motor_pos      = 0.0
            safe_motor_dry_mass = 0.0

        return {
            # ── Simulation / wind config ───────────────────────────────────────
            # cep_prob: read from AppState (authoritative) — updated by both the
            # main-window cep_prob_input widget and the Advanced Settings dialog.
            # Reading the widget directly would silently ignore Advanced Settings.
            "cep_prob":   int(s.landing_prob) if s.landing_prob is not None else 90,
            "launch_lat": w.lat_input.value()  if w.lat_input.value()  != -9999.0 else 35.6828,
            "launch_lon": w.lon_input.value()  if w.lon_input.value()  != -9999.0 else 139.7590,
            "elev":       s.launch_angle,   # degrees — AppState, default 85.0
            "rail":       s.launch_rail,    # m       — AppState, default 1.0
            "azim":       w.azim_input.value() if w.azim_input.value() != -9999.0 else 0.0,
            "wind_profile_data": s.wind_profile_data,
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
            "fin_count":      s.fin_count,             # int
            "nose_kind":      s.nose_shape,            # str
            "motor_pos":      safe_motor_pos,          # m from nose
            "motor_dry_mass": safe_motor_dry_mass,     # kg
            "body_cd":        s.drag_coeff or 0.45,          # rocket airframe Cd (None→default)
            "para_cd":        s.parachute_cd,              # parachute Cd (for CdS product)
            "para_area":      s.parachute_area,            # m²
            "para_lag":       s.parachute_lag,         # s
            "backfire_delay": None if s.backfire_delay == -9999.0 else s.backfire_delay,        # s
            # ── Motor thrust curve — persisted by AppWindow._on_load_motor() ──
            "target_radius":  s.target_radius,              # m  (rmax_input spinbox)
            # Target ENU offset (metres from launch pad origin).
            # Defaults to 0.0/0.0 (target IS at the launch pad) until a UI
            # widget is wired to provide a non-zero offset.
            "target_x":       float(getattr(s, "target_x", 0.0) or 0.0),  # East  (m)
            "target_y":       float(getattr(s, "target_y", 0.0) or 0.0),  # North (m)
            **({
                "thrust_data":     w._motor_thrust_data,
                "motor_burn_time": w._motor_burn_time,
            } if getattr(w, "_motor_thrust_data", None) else {}),
            # ── Environmental factors ──────────────────────────────────────────
            "hellmann_alpha": s.hellmann_alpha,
            "env_pressure":   s.env_pressure,
            "env_temp":       s.env_temp,
            "env_humidity":   s.env_humidity,
            
            # ── Advanced config passed directly into params to decouple UI ─────
            "I_z":            s.moi_roll if s.moi_roll > 0.0 else None,
            "I_xy":           s.moi_pitch if s.moi_pitch > 0.0 else None,
            "power_on_cd":    s.power_on_cd,
            "power_off_cd":   s.power_off_cd,
            "motor_isp":      s.motor_isp,
            "motor_propellant_density": s.motor_propellant_density,
            "cd_curve_power_on":  s.cd_curve_power_on,
            "cd_curve_power_off": s.cd_curve_power_off,
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

            from PySide6.QtWidgets import QLineEdit
            if isinstance(sb, QLineEdit):
                # Text input handling
                def validate_and_set(text: str, _sb=sb, _p=prop, _f=to_si):
                    val_str = text.strip()
                    if val_str == "":
                        _sb.setStyleSheet("border: 1px solid red;")
                        setattr(s, _p, None)
                    else:
                        try:
                            val_float = float(val_str)
                            _sb.setStyleSheet("")
                            setattr(s, _p, _f(val_float))
                        except ValueError:
                            _sb.setStyleSheet("border: 1px solid red;")
                            setattr(s, _p, None)

                sb.editingFinished.connect(lambda _sb=sb, _p=prop, _f=to_si: validate_and_set(_sb.text(), _sb, _p, _f))
                
                sig = getattr(s, f"{prop}_changed", None)
                if sig is not None:
                    def update_sb(v, _sb=sb, _g=from_si):
                        if v is None:
                            if _sb.text() != "":
                                _sb.setText("")
                            _sb.setStyleSheet("border: 1px solid red;")
                        else:
                            val_str = str(round(_g(v), 5))
                            if _sb.text() != val_str:
                                _sb.setText(val_str)
                            _sb.setStyleSheet("")
                    sig.connect(update_sb)
                
                # Check initial state
                init_val = getattr(s, prop, None)
                if init_val is None:
                    sb.setStyleSheet("border: 1px solid red;")
                else:
                    sb.setStyleSheet("")
            else:
                # Sentinel -9999.0 means the field is blank; do not push to AppState.
                sb.valueChanged.connect(
                    lambda v, _p=prop, _f=to_si:
                        None if v == -9999.0 else setattr(s, _p, _f(v))
                )
                sig = getattr(s, f"{prop}_changed", None)
                if sig is not None:
                    def update_sb(v, _sb=sb, _g=from_si):
                        if v is None:
                            _sb.clear()
                        else:
                            _sb.setValue(_g(v))
                    sig.connect(update_sb)

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
        _bind("motor_cg_input",    "motor_cg_pos",    lambda v: float(v),      lambda v: float(v))
        _bind("motor_dry_mass_input","motor_dry_mass",  lambda v: float(v),      lambda v: float(v))
        _bind("af_backfire_input", "backfire_delay",  lambda v: float(v),      lambda v: float(v))
        _bind("para_cd_input",     "parachute_cd",    lambda v: float(v),      lambda v: float(v))
        _bind("para_area_input",   "parachute_area",  lambda v: float(v),      lambda v: float(v))
        _bind("para_lag_input",    "parachute_lag",   lambda v: float(v),      lambda v: float(v))
        # Launch geometry — pre-filled defaults (85° / 1.0 m); never use -9999.0 sentinel
        _bind("elev_input",        "launch_angle",    lambda v: float(v),      lambda v: float(v))
        _bind("rail_len_input",    "launch_rail",     lambda v: float(v),      lambda v: float(v))

        # ── Custom bindings for Fin Count and Nose Shape ──────────────────────
        if hasattr(w, "af_fincount_input"):
            w.af_fincount_input.valueChanged.connect(lambda v: setattr(s, "fin_count", int(v)))
            s.fin_count_changed.connect(lambda v: w.af_fincount_input.setValue(int(v)) if v is not None else None)
        
        if hasattr(w, "af_noseshape_input"):
            w.af_noseshape_input.currentTextChanged.connect(lambda v: setattr(s, "nose_shape", str(v)))
            s.nose_shape_changed.connect(lambda v: w.af_noseshape_input.setCurrentText(str(v)) if v is not None else None)
        
        if hasattr(w, "hellmann_alpha_input"):
            w.hellmann_alpha_input.valueChanged.connect(lambda v: setattr(s, "hellmann_alpha", float(v)))
            s.hellmann_alpha_changed.connect(lambda v: w.hellmann_alpha_input.setValue(float(v)) if v is not None else None)

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
        s.rocket_diameter = af["radius"] * 2.0

        # Task 1.3: Information dynamic label
        import os as _os
        name = _os.path.basename(path)
        lbl_text = f"Name: {name}\nCG: {af['cg']:.2f} m  |  L: {af['length']:.2f} m"
        self._window.rkt_label.setText(lbl_text)
        self._window.rkt_label.setStyleSheet("color: #a6e3a1; font-weight: bold; font-size: 8pt; padding: 2px 4px;")        # m radius → m diameter
        s.nose_length     = af["nose_length"]         # m
        s.fin_root_chord  = af["fin_root"]            # m
        s.fin_tip_chord   = af["fin_tip"]             # m
        s.fin_span        = af["fin_span"]            # m
        s.fin_position    = af["fin_pos"]             # m from nose
        # s.motor_cg_pos    = af["motor_pos"]           # Operational input only
        # s.motor_dry_mass  = af["motor_dry_mass"]      # Operational input only
        # s.backfire_delay  = af["backfire_delay"]      # s (Operational input only, do not overwrite on load)

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
            self._window.set_status(f".rkt parse failed: {exc}", "#f38ba8")
            return
        except Exception as exc:
            print(f"[_on_load_rkt] Unexpected error: {type(exc).__name__}: {exc}")
            self._window.set_status(f".rkt load error: {exc}", "#f38ba8")
            return

        s   = self._state
        af  = cfg["airframe"]
        moi = cfg["moi"]

        # ── Airframe geometry → AppState ───────────────────────────────────────
        # Each setter emits its signal → bound spinbox updates automatically.
        s.rocket_dry_mass = af["mass"]
        s.rocket_cg       = af["cg"]
        s.rocket_length   = af["length"]
        s.rocket_diameter = af["radius"] * 2.0

        # Task 1.3: Information dynamic label
        import os as _os
        name = _os.path.basename(path)
        lbl_text = f"Name: {name}\nCG: {af['cg']:.2f} m  |  L: {af['length']:.2f} m"
        self._window.rkt_label.setText(lbl_text)
        self._window.rkt_label.setStyleSheet("color: #a6e3a1; font-weight: bold; font-size: 8pt; padding: 2px 4px;")
        s.nose_length     = af["nose_length"]
        s.fin_root_chord  = af["fin_root"]
        s.fin_tip_chord   = af["fin_tip"]
        s.fin_span        = af["fin_span"]
        s.fin_position    = af["fin_pos"]
        if "fin_count" in af:
            s.fin_count = af["fin_count"]
        if "nose_shape" in af:
            s.nose_shape = af["nose_shape"]
        # s.backfire_delay  = af["backfire_delay"]  # Operational input only

        missing_info = cfg.get("missing_info", {})
        failed_fields = missing_info.get("failed_fields", [])
        manual_fields = missing_info.get("manual_fields", [])

        # Add manual fields that might have been detected globally but missed by specific tags
        if ((cfg.get("parachute", {}).get("cd") or 0) > 0 and
            "パラシュート関連パラメータ (Cd, Area, Lag)" not in manual_fields):
            manual_fields.append("パラシュート関連パラメータ (Cd, Area, Lag)")
        if (((af.get("motor_dry_mass") or 0) > 0 or (af.get("motor_pos") or 0) != 0) and
            "モーターパラメータ" not in manual_fields):
            manual_fields.append("モーターパラメータ")
        if ((af.get("backfire_delay") or 0) > 0 and
            "バックファイア遅延 (Backfire Delay)" not in manual_fields):
            manual_fields.append("バックファイア遅延 (Backfire Delay)")

        if failed_fields or manual_fields:
            self._window.set_status("RKT load warning: check Manual Config for missing parameters.", "#f9e2af")

        # ── MoI → AppState (emits moi_updated signal) ──────────────────────────
        s.set_moi(moi["ixx"], moi["iyy"], moi["izz"])

        s.original_rocket_config = dict(af)

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
            self._window.set_status(f"Parachute JSON failed: {exc}", "#f38ba8")
            return
        except Exception as exc:
            print(f"[_on_load_para_json] Unexpected: {type(exc).__name__}: {exc}")
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

    @Slot()
    def _on_wind_tick(self) -> None:
        # Surface and upper wind baselines now live in wind_profile_data
        wp = self._state.wind_profile_data or []
        lowest_wind = sorted(wp, key=lambda n: n["alt_m"])[0] if wp else {"speed_ms": 0.0, "dir_deg": 0.0}
        highest_wind = sorted(wp, key=lambda n: n["alt_m"])[-1] if wp else {"speed_ms": 0.0, "dir_deg": 0.0}
        
        base_spd  = float(lowest_wind.get("speed_ms", 0.0))
        base_dir  = float(lowest_wind.get("dir_deg", 0.0))
        up_spd    = float(highest_wind.get("speed_ms", 0.0))
        up_dir    = float(highest_wind.get("dir_deg", 0.0))
        speed     = max(0.0, base_spd + random.gauss(0.0, base_spd * 0.05 + 0.1))
        direction = (base_dir + random.gauss(0.0, 3.0)) % 360.0

        # Global AppState: (speed, direction) tuples for future Phase-2 consumers.
        self._state.append_wind_reading(speed, direction)

        # Mock Koinobori environmental data (Pressure, Temp, Humidity)
        # Assuming sea-level launch baseline (101325 Pa, 15°C, 50% Hum) with small noise.
        # This will be replaced by actual serial/UDP parsing from the Koinobori hardware.
        mock_p = 101325.0 + random.gauss(0.0, 50.0)
        mock_t = 15.0 + random.gauss(0.0, 0.5)
        mock_h = 50.0 + random.gauss(0.0, 2.0)
        self._state.env_pressure = mock_p
        self._state.env_temp = mock_t
        self._state.env_humidity = mock_h

        # Keep the status-bar readout current.
        self._window.update_wind_readout(speed, direction, up_spd, up_dir)

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

    @Slot(float, float)
    def _on_map_coordinates_picked(self, lat: float, lon: float) -> None:
        """Handle 'Shift + Drag' launch site relocation from the Map View."""
        import math

        def get_distance(lat1, lon1, lat2, lon2):
            R = 6371.0 # km
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c

        old_lat = self._state.launch_lat
        old_lon = self._state.launch_lon

        # Guard against None values
        if old_lat is None: old_lat = 0.0
        if old_lon is None: old_lon = 0.0

        dist = get_distance(old_lat, old_lon, lat, lon)

        # Update AppState correctly. AppState emits signal that UI input should react to,
        # but to ensure strict sync, we update the inputs which inherently triggers AppState sync via their bindings
        self._state.launch_lat = lat
        self._state.launch_lon = lon

        if dist > 5.0:
            self._on_run_clicked()
        else:
            self._state.needs_redraw.emit()

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
        for btn in self._window.findChildren(QPushButton, "btn_phase1_run"):
            btn.setEnabled(enabled)
        for btn in self._window.findChildren(QPushButton, "btn_stop"):
            btn.setEnabled(self._state.is_calculating)
