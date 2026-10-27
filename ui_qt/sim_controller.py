"""
ui_qt/sim_controller.py
Application-level controller that mediates between AppWindow, AppState, and workers.
"""

from __future__ import annotations

import math
import random
import time
import uuid
from copy import deepcopy
from importlib.metadata import version

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Slot, Signal, QSignalBlocker, Qt
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QPushButton, QSystemTrayIcon, QStyle, QMessageBox

from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.workers import SimulationWorker, OptimizationWorker, MapDownloadWorker
from ui_qt.location_service import LocationService
from ui_qt.airframe_load_dialog import select_airframe_files
from core.monte_carlo  import compute_cep_ellipse
from core.wind_model import ANEMOMETER_ALT
from utils.data_loader import (
    RocketConfigError,
    AirframeConfigError,
    load_airframe_config,
    validate_airframe_config,
    save_airframe_config,
    parse_parachute_json,
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
        self._input_revision = 0
        self._run_snapshot = None
        self._applying_result = False
        self._closing_requested = False
        self._forecast_nodes = {}
        window._shutdown_guard = self.request_shutdown
        self._manual_editing = False
        self._manual_pending_airframe = None
        self._manual_pending_path = None
        # Cache the sig_nominal_done payload so _on_mc_done can forward
        # phases/events (phase-coloured 3-D trajectory) into the final result.
        self._nominal_payload: dict | None    = None

        # ── Setup System Tray Icon ─────────────────────────────────────────────
        self._tray_icon = QSystemTrayIcon(self)
        self._tray_icon.setIcon(self._window.style().standardIcon(QStyle.SP_ComputerIcon))
        self._tray_icon.show()

        self._rewire_buttons()
        window.action_run.triggered.connect(self._on_run_requested)
        window.action_stop.triggered.connect(self._on_stop_clicked)
        window.sig_import_results_clicked.connect(self._on_import_results)
        self._location_service = LocationService(self)
        window.sig_get_location_clicked.connect(self._on_get_current_location)
        self._location_service.position_ready.connect(self._on_current_location_ready)
        self._location_service.error.connect(self._on_current_location_error)
        self._location_service.busy_changed.connect(lambda busy: window.btn_get_location.setEnabled(not busy))

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
        self._window.map_view.coordinates_picked.connect(self._on_map_coordinates_picked)

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
                lambda v: _rmax.setValue(float(v)) if v is not None else None
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
        for key, spin in window.montblanc_inputs.items():
            spin.setValue(state.montblanc_options[key])
            spin.valueChanged.connect(lambda value, key=key: setattr(state, 'montblanc_options', {key:value}))
        def sync_montblanc(values):
            for key, spin in window.montblanc_inputs.items():
                was_blocked = spin.blockSignals(True)
                try:
                    spin.setValue(values[key])
                finally:
                    spin.blockSignals(was_blocked)
        state.montblanc_options_changed.connect(sync_montblanc)

        # ── Load Rocket JSON button → controller ──────────────────────────────
        # AppWindow emits sig_load_airframe_json_clicked instead of opening the
        # dialog itself, keeping file I/O and AppState writes in the controller.
        _json_sig = getattr(window, 'sig_load_airframe_json_clicked', None)
        if _json_sig is not None:
            _json_sig.connect(self._on_load_airframe_json)
        window.sig_manual_config_clicked.connect(self._on_manual_config_clicked)
        window._manual_dialog.sig_apply.connect(self._on_apply_manual_config)
        window.sig_save_rocket_json_clicked.connect(self._on_save_airframe_json)

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

        # Progress is the whole operation's percentage; labels arrive separately.
        state.progress_changed.connect(
            lambda pct: window.set_progress(pct)
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
        names = ('rocket_dry_mass','rocket_cg','rocket_cp','rocket_length','rocket_diameter',
                 'nose_length','fin_root_chord','fin_tip_chord','fin_span','fin_position',
                 'motor_cg_pos','motor_dry_mass','parachute_cd','parachute_area','parachute_lag',
                 'backfire_delay','launch_angle','launch_rail','flight_mode','target_radius',
                 'launch_lat','launch_lon','wind_uncertainty','thrust_uncertainty','gust_speed',
                 'landing_prob','mc_n_runs','target_x','target_y','montblanc_options',
                 'power_on_cd','power_off_cd','motor_isp','motor_propellant_density',
                 'fin_count','nose_shape','hellmann_alpha','phase4_options','fin_options',
                 'cd_curve_power_on','cd_curve_power_off','wind_profile_data')
        for name in names:
            getattr(state,name+'_changed').connect(self._invalidate_inputs)
        state.parachute_wind_drift_changed.connect(self._invalidate_inputs)
        state.moi_updated.connect(self._invalidate_inputs)
        window.sig_thrust_curve_changed.connect(self._invalidate_inputs)
        state.koinobori_status = 'Practice (simulated)'
        self._forecast_key = self._forecast_source_key()
        for signal in (state.phase4_options_changed,state.launch_lat_changed,state.launch_lon_changed):
            signal.connect(self._reset_forecast)

    def _forecast_source_key(self):
        p = self._state.phase4_options
        return (p['wind_csv'],p['wind_time_utc'],self._state.launch_lat,self._state.launch_lon)

    def _reset_forecast(self,*_args):
        key = self._forecast_source_key()
        if key != self._forecast_key:
            self._forecast_key = key
            self._forecast_nodes = {}
            self._window._upper_wind_means = {}
            self._state.reset_upper_wind()
            self._state.gpv_last_fetch_time = ('Pending — run to load selected CSV' if key[0] else 'Manual practice wind')

    def _invalidate_inputs(self, *_args):
        if self._applying_result:
            return
        self._input_revision += 1
        self._state.result_inputs_stale = True
        was_active = self._state.phase2_active
        self._state.phase2_active = False
        if was_active or self._state.has_sim_result:
            self._window.update_status_indicator('INPUT CHANGED — RECALCULATE')

    def request_shutdown(self):
        workers = [w for w in [self._worker] + self._window.findChildren(QThread) if w is not None]
        active = []
        for worker in workers:
            try:
                if worker.isRunning():
                    active.append(worker)
                    worker.stop() if hasattr(worker,'stop') else worker.requestInterruption()
            except RuntimeError:
                pass
        if not active:
            self._tray_icon.hide()
            return True
        self._closing_requested = True
        self._wind_timer.stop()
        self._window.set_status('Waiting for background operations to stop…')
        QTimer.singleShot(100,self._window.close)
        return False

    # ── Button rewiring ────────────────────────────────────────────────────────

    def _rewire_buttons(self) -> None:
        """
        Redirect every btn_run / btn_stop in the widget tree to controller
        slots.
        """
        for btn in self._window.findChildren(QPushButton, "btn_run"):
            btn.clicked.connect(self._on_run_requested)

        for btn in self._window.findChildren(QPushButton, "btn_stop"):
            btn.clicked.connect(self._on_stop_clicked)

        if hasattr(self._window, "btn_download_map"):
            self._window.btn_download_map.clicked.connect(self._on_download_map_clicked)

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
            missing.append("Rocket Geometry (Load an OpenRocket .ork design, or fill the missing fields in Manual Config)")

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
            missing.append("Maximum Radius")

        # Motor Thrust Data
        if not getattr(self._window, '_motor_thrust_data', None):
            missing.append("Motor Thrust Curve (.csv)")

        if missing:
            print("[DIAG] _validate_run_prerequisites FAILED: Missing Parameters: " + ", ".join(missing), flush=True)
            self._window.set_status("Missing Parameters: " + ", ".join(missing), "#f38ba8")
            return False

        # ── 60-second surface wind buffer check ───────────────────────────────
        now = time.monotonic()
        surface_hist = [e for e in self._state.wind_history_for_alt(ANEMOMETER_ALT) if now-e['ts'] <= 60]
        if surface_hist and now-surface_hist[-1]['ts'] > 5:
            self._window.set_status('Surface wind data is stale. Wait for new readings.', '#f38ba8')
            return False
        print(f"[DIAG] wind_history_for_alt(ANEMOMETER_ALT) sample count: {len(surface_hist)}")
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
    def _on_get_current_location(self) -> None:
        self._window.location_label.setText("Acquiring current location from the system...")
        self._window.set_status("Acquiring current location...", "#f9e2af")
        self._location_service.request()

    @Slot(float, float, float)
    def _on_current_location_ready(self, lat, lon, accuracy) -> None:
        self._state.launch_lat = lat
        self._state.launch_lon = lon
        self._window.map_widget.update_launch(lat, lon)
        precision = f"Accuracy: ±{accuracy:.0f} m" if accuracy >= 0 else "Accuracy unavailable"
        self._window.location_label.setText(f"Current location: {lat:.6f}, {lon:.6f}\n{precision}")
        self._on_download_map_clicked()

    @Slot(str)
    def _on_current_location_error(self, message) -> None:
        self._window.location_label.setText(f"Current location acquisition failed. {message}")
        self._window.set_status(message, "#f38ba8")

    @Slot()
    def _on_download_map_clicked(self) -> None:
        from utils import map_downloader as offline_maps

        try:
            if self._worker is not None and self._worker.isRunning():
                self._window.set_status("Wait for the current operation to finish.", "#f9e2af")
                return
        except RuntimeError:
            self._worker = None

        # Commit pending spinbox edits before reading the launch coordinates.
        self._window.lat_input.interpretText()
        self._window.lon_input.interpretText()
        try:
            lat, lon = offline_maps.validate_map_coordinates(
                self._window.lat_input.value(), self._window.lon_input.value())
        except ValueError as exc:
            self._window.set_status(str(exc), "#f38ba8")
            return
        self._state.launch_lat = lat
        self._state.launch_lon = lon

        # Validate the image as well as its metadata before accepting a hit.
        try:
            meta, _image = offline_maps.read_offline_map()
            if (abs(lat - float(meta["center_lat"])) < 1e-5
                    and abs(lon - float(meta["center_lon"])) < 1e-5):
                self._window.map_view.request_render(True)
                self._window.set_status("Saved offline map loaded.", "#a6e3a1")
                return
        except (OSError, ValueError, TypeError, KeyError):
            pass

        self._window.btn_download_map.setEnabled(False)
        self._state.progress_percentage = 0
        self._state.is_calculating = True
        self._set_run_buttons_enabled(False)
        # Map downloads do not implement simulation cancellation.
        self._window.action_stop.setEnabled(False)
        for btn in self._window.findChildren(QPushButton, "btn_stop"):
            btn.setEnabled(False)
        self._window.set_status("Downloading offline map...", "#f9e2af")

        self._worker = MapDownloadWorker(lat, lon, parent=self)
        self._worker.sig_progress.connect(self._on_progress_updated)
        self._worker.sig_finished.connect(self._on_download_map_finished)
        self._worker.error.connect(self._on_map_download_error)
        self._worker.finished.connect(self._on_map_worker_finished)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    @Slot(dict)
    def _on_download_map_finished(self, meta: dict) -> None:
        self._window.map_view.request_render(True)
        self._window.set_progress(100, "Map downloaded")
        self._window.set_status("Offline map saved and refreshed.", "#a6e3a1")

    @Slot(str)
    def _on_map_download_error(self, msg: str) -> None:
        print(f"Offline map download failed: {msg}")
        self._window.map_view.request_render(True)
        self._window.set_progress(0, "Map download failed")
        self._window.set_status(
            "Map download failed. Check your connection and retry. Any previously saved map is unchanged.",
            "#f38ba8")

    @Slot()
    def _on_map_worker_finished(self) -> None:
        if self.sender() is self._worker:
            self._worker = None
        self._state.is_calculating = False
        self._window.btn_download_map.setEnabled(True)
        self._set_run_buttons_enabled(True)

    @Slot()
    def _on_run_requested(self) -> None:
        if self._state.is_free_mode:
            self._on_run_clicked()
        else:
            self._on_phase1_clicked()

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

        print("[DIAG] Prerequisites PASSED — wind monitor remains active")

        self._state.mc_running = True
        self._state.simulation_started.emit()
        self._state.is_calculating = True
        self._set_run_buttons_enabled(False)
        self._window.set_status("Simulation running...", "#f9e2af")
        self._window.set_progress(0, "Simulating...")
        self._state.progress_percentage = 0
        self._progress_message = 'Simulating...'

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
            self._on_error(f"Parameter collection failed: {_cp_exc}")
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

        # Continue displaying live wind while the worker uses its input snapshot.

        self._state.mc_running = True
        self._state.simulation_started.emit()
        self._state.is_calculating = True
        self._set_run_buttons_enabled(False)
        self._window.set_status("Optimisation running...", "#fab387")
        self._window.set_progress(0, "Optimising...")
        self._state.progress_percentage = 0
        self._progress_message = 'Optimising...'

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
            self._on_error(f"Parameter collection failed: {_cp_exc}")
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
        if self._run_snapshot and self._run_snapshot['input_revision'] != self._input_revision:
            return
        self._applying_result = True
        self._state.launch_angle = elev
        from utils.magnetic import input_bearing
        reference,declination = getattr(self,'_run_bearing',
            (self._state.phase4_options['azimuth_reference'],self._state.effective_declination))
        self._window.azim_input.setValue(input_bearing(azi, reference, declination))
        self._window.elev_input.setValue(elev)
        self._applying_result = False


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
        value = max(self._state.progress_percentage, value) if self._state.is_calculating else value
        self._state.progress_percentage = value
        self._window.set_progress(value, getattr(self, '_progress_message', 'Preparing'))

    def populate_results(self, nominal_data: dict, mc_data: dict) -> None:
        """Populates the Simulation Results panel in AppWindow based on mode."""
        w = self._window
        w._results_grp.setVisible(True)

        mode = self._state.flight_mode
        mode = mc_data.get('input_snapshot',{}).get('flight_mode',mode)
        n = mc_data.get('n_runs',0)
        attempted = mc_data.get('n_attempted',n)
        w.mc_quality_label.setText(f"MC: {n}/{attempted} completed; "
            f"{mc_data.get('mc_disqualified_count',0)} disqualified; {attempted-n} failed. Scores: valid trials only.")
        is_free = "free" in mode.lower() or "自由" in mode
        from core.montblanc import is_montblanc
        if is_montblanc(mode):
            values = dict(mc_data)
            values.update(nominal_data)
            def metric(key):
                value = values.get(key)
                return '-' if value is None else f'{value:.2f}'
            w.montblanc_result_label.setText(
                f"着地水平速度: {metric('landing_horizontal_speed_ms')} m/s\n"
                f"直前区間の最大速度: {metric('landing_window_max_speed_ms')} m/s\n"
                f"速度変動幅: {metric('landing_speed_variation_ms')} m/s\n"
                f"評価時間の確保: {'OK' if values.get('landing_window_covered') else '不足 / 不明'}\n"
                f"公称条件: {'適合' if values.get('montblanc_pass') else '不適合 / 不明'}\n"
                f"MC全条件適合: {mc_data.get('mc_montblanc_joint_count',0)}/{attempted}")
            for label, key, unit in ((w.lbl_res_angle,'elev',' °'), (w.lbl_res_azimuth,'azi',' °'),
                                     (w.lbl_res_hdist,'r_horiz',' m'), (w.lbl_res_hang,'hang_time',' s')):
                value = values.get(key)
                label.setText('-' if value is None else f'{value:.2f}{unit}')

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
        snapshot = payload.get('input_snapshot') or {}
        if snapshot and snapshot.get('input_revision') != self._input_revision:
            return
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
                f"FREE MODE — Nominal dist {_nom_dist:.0f} m  |  MC running…"
            )
        elif _target_r is None:
            print("[DIAG] NOGO guard: target_r is None in non-Free mode — skipping indicator", flush=True)
            self._window.update_status_indicator("CALCULATING — MC running…")
        elif _nom_dist > _target_r:
            print(
                f"[DIAG] NOGO: _on_nominal_done — nominal {_nom_dist:.1f}m > target_r {_target_r:.1f}m",
                flush=True,
            )
            self._window.update_status_indicator(
                f"NO-GO — Nominal {_nom_dist:.0f} m > Target {_target_r:.0f} m  |  MC running…"
            )
        elif _mode_str == 'モンブラン' and not payload.get('montblanc_pass', False):
            self._window.update_status_indicator('NO-GO — 水平速度・安定条件が不適合 / 評価時間不足 | MC running…')
        else:
            self._window.update_status_indicator(
                f"CALCULATING — Nominal {_nom_dist:.0f} m / {_target_r:.0f} m  |  MC running…"
            )

        # profile — one call covers all 5 altitudes at once.
        wind_nodes = payload.get("wind_nodes", [])
        if wind_nodes:
            if self._state.phase4_options['wind_csv']:
                self._window._upper_wind_means = {}
                self._state.reset_upper_wind()
                self._forecast_nodes = {float(n['alt_m']):dict(n) for n in wind_nodes
                                        if float(n['alt_m']) > ANEMOMETER_ALT}
                from utils.time_display import format_jst
                self._state.gpv_last_fetch_time = 'Forecast target: ' + format_jst(self._state.phase4_options['wind_time_utc'])
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
        self._on_worker_status(msg)
        self._on_progress(pct)

    @Slot(dict)
    def _on_mc_done(self, result: dict) -> None:
        print(f"[DIAG] _on_mc_done ENTERED — cancelled={result.get('cancelled')}", flush=True)
        self._state.mc_running = False
        snapshot = result.get('input_snapshot') or {}
        if snapshot and snapshot.get('input_revision') != self._input_revision:
            self._state.phase2_active = False
            self._state.is_calculating = False
            self._state.simulation_result = None
            self._state.has_sim_result = False
            self._set_run_buttons_enabled(True)
            self._window.update_status_indicator('INPUT CHANGED — RECALCULATE')
            self._window.set_status('Inputs changed during calculation; stale result discarded.', '#f9e2af')
            return
        self._state.result_inputs_stale = False

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
        lat     = snapshot.get('launch_lat',self._window.lat_input.value())
        lon     = snapshot.get('launch_lon',self._window.lon_input.value())
        off_e   = result.get("impact_x", 0.0)
        off_n   = result.get("impact_y", 0.0)

        from utils.geo_math import offset_to_latlon
        land_lat,land_lon = offset_to_latlon(lat,lon,off_e,off_n)

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

        # Phase A verification: target containment at the selected probability.
        # FREE MODE: no target radius / landing zone concept — skip entirely.
        target_r = snapshot.get('target_radius',self._state.target_radius)
        _mode_str_mc = str(snapshot.get('flight_mode',self._state.flight_mode))
        _is_free_mode_mc = "free" in _mode_str_mc.lower() or "自由" in _mode_str_mc
        print(
            f"[DIAG] _on_mc_done Phase A check: "
            f"mode={_mode_str_mc!r} is_free={_is_free_mode_mc} "
            f"target_r={target_r}",
            flush=True,
        )

        r90    = self._state.r90_radius
        apogee = result.get("apogee_m",  0.0)
        tof    = result.get("hang_time", 0.0)
        n      = result.get("n_runs",    0)
        prob   = result.get("landing_prob", int(self._window.cep_prob_input.value()))

        self._state.phase2_active = False
        if _is_free_mode_mc:
            # Free Mode: no SAFE/UNSAFE concept — display informational summary only.
            self._window.update_status_indicator(
                f"FREE MODE — R{prob}: {r90:.0f} m  |  Apogee: {apogee:.0f} m"
            )
            self._window.set_status(
                f"Free Mode  |  R{prob}: {r90:.1f} m  |  "
                f"Apogee: {apogee:.0f} m  |  ToF: {tof:.1f} s  ({n} runs)",
                "#89dceb",
            )
        elif target_r is None:
            # Non-Free mode but no target radius set — show a neutral warning.
            print("[DIAG] NOGO guard: target_r is None in non-Free mode — cannot evaluate Phase A", flush=True)
            self._window.update_status_indicator("target_radius not set — cannot evaluate GO/NO-GO")
            self._window.set_status(
                f"target_radius not set  |  R{prob}: {r90:.1f} m  |  "
                f"Apogee: {apogee:.0f} m  |  ToF: {tof:.1f} s  ({n} runs)",
                "#f9e2af",
            )
        elif _mode_str_mc == 'モンブラン':
            attempted = int(result.get('n_attempted', len(_scatter)))
            joint = int(result.get('mc_montblanc_joint_count', 0))
            fraction = joint / attempted if attempted > 0 else 0.
            nominal_ok = bool(result.get('montblanc_pass', False))
            is_safe = nominal_ok and len(_scatter) >= 4 and fraction >= prob / 100.
            verdict = 'GO' if is_safe else 'NO-GO'
            self._window.update_status_indicator(
                f'{verdict} (Montblanc 全条件 {fraction:.1%}; 必要 {prob}%)')
            self._window.set_status(
                f'{verdict} | 水平速度・安定・着地範囲: {joint}/{attempted} | '
                '実行時条件の判定。速度条件のライブ再評価には再計算が必要。',
                '#a6e3a1' if is_safe else '#f9e2af')
            # Surface-wind tolerance alone cannot revalidate a speed window.
            self._state.phase2_active = False
        else:
            from core.monte_carlo import landing_region_fraction
            fraction = landing_region_fraction(
                _scatter, float(target_r),
                (getattr(self._state,'target_x',0.) or 0., getattr(self._state,'target_y',0.) or 0.),
                attempted=result.get('n_attempted',len(_scatter)))
            is_safe = len(_scatter) >= 4 and fraction >= prob / 100.0
            print(
                f"[DIAG] Phase A result: containment={fraction:.1%}, required={prob}% "
                f"target_r={target_r:.1f}m  ->  {'SAFE' if is_safe else 'NOGO'}",
                flush=True,
            )
            if is_safe:
                # Lock the Phase A wind distribution so Phase B O(1) GO/NO-GO ticks
                # compare live wind against the exact baseline used in the MC run.
                lowest_wind = sorted(self._state.wind_profile_data, key=lambda n: n["alt_m"])[0] if self._state.wind_profile_data else {"speed_ms": 0.0, "dir_deg": 0.0}
                surf_spd = result.get("nominal_surf_spd", lowest_wind["speed_ms"])
                surf_dir = result.get("nominal_surf_dir", lowest_wind["dir_deg"])
                wind_unc = float(snapshot.get('wind_unc',self._window.wind_unc_input.value()))
                from core.wind_model import speed_dir_to_uv
                mu_u, mu_v = speed_dir_to_uv(surf_spd, surf_dir)
                self._state.set_wind_lock(mu_u, mu_v, wind_unc)
                self._state.phase2_active = True
                self._window.update_status_indicator(
                    f"GO  (Target containment {fraction:.1%} ≥ {prob}%)"
                )
            else:
                print(
                    f"[DIAG] NOGO: containment={fraction:.1%}, required={prob}%, samples={len(_scatter)}",
                    flush=True,
                )
                self._window.update_status_indicator(
                    f"NO-GO  (Target containment {fraction:.1%}; required {prob}%, ≥4 samples)"
                )
            verdict = (
                f"GO  Target containment {fraction:.1%} >= {prob}%"
                if is_safe else
                f"NO-GO  Target containment {fraction:.1%}; required {prob}%"
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
        # Keep 100% visible until the next operation starts.
        self._state.progress_percentage = 100
        self._window.set_progress(100, 'Completed')
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
        self._state.phase2_active = False
        lines = msg.strip().splitlines() or ["Unknown simulation error"]
        is_traceback = msg.lstrip().startswith("Traceback")
        is_unstable = "Unstable Rocket" in msg
        if is_unstable:
            title = "機体・モーターの不適合"
            summary = "機体・モーターが不適合です（機体が不安定）"
            explanation = (
                "この機体設計とモーターの組み合わせは不適合です。\n"
                "機体が空力的に不安定と判定されたため、計算を中止しました。\n"
                "機体に適合するモーターを選択してください。"
            )
            if getattr(self._window, "_motor_thrust_data", None):
                explanation += f"\n選択したモーター：{self._window.motor_label.text()}"
            self._window.update_status_indicator("NO-GO — 不適合・不安定")
        else:
            title = "シミュレーションエラー"
            summary = "シミュレーションを実行できませんでした"
            explanation = lines[-1] if is_traceback else "\n".join(lines)
            self._window.update_status_indicator("ERROR — シミュレーション失敗")
        self._window.set_status(f"{title}：{summary}", "#f38ba8")
        self._window.set_progress(0, "エラー")
        self._set_run_buttons_enabled(True)

        if hasattr(self._window, "btn_download_map"):
            self._window.btn_download_map.setEnabled(True)

        # Resume 1 Hz wind monitor even on error so telemetry is not lost.
        self._wind_timer.start()

        # A multi-line cause is unreadable in the one-line status bar or a
        # short-lived tray balloon. Keep it available until dismissed, while
        # allowing the Qt event loop (including telemetry) to keep running.
        previous = getattr(self, "_error_dialog", None)
        if previous is not None:
            previous.close()
            previous.deleteLater()
        self._error_dialog = QMessageBox(self._window)
        self._error_dialog.setWindowTitle(title)
        self._error_dialog.setIcon(QMessageBox.Icon.Warning)
        self._error_dialog.setTextFormat(Qt.TextFormat.PlainText)
        self._error_dialog.setText(summary)
        self._error_dialog.setInformativeText(explanation)
        self._error_dialog.setDetailedText(msg)
        self._error_dialog.open()

        self._tray_icon.showMessage(
            title,
            summary,
            QSystemTrayIcon.Warning,
            5000
        )

    @Slot(str)
    def _on_worker_status(self, msg: str) -> None:
        print(f"[DIAG] _on_worker_status: {msg!r}", flush=True)
        # All labels and the progress bar are updated on the GUI thread.
        self._window._status_label.setText(msg)
        self._progress_message = msg
        self._window.set_progress(self._state.progress_percentage, msg)
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
        in_bounds = status == 'GO'
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
        μ and σ describe the retained surface samples (up to 60 samples,
        1.5 m AGL; currently generated by the practice timer). The absolute
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

        from utils.magnetic import declination_for_site, true_bearing
        options = s.phase4_options
        if options['declination_mode'] == 'auto':
            s.magnetic_declination = declination_for_site(s.launch_lat, s.launch_lon, options['launch_alt_m'])
        self._run_bearing = (options['azimuth_reference'], s.effective_declination)
        azimuth = true_bearing(w.azim_input.value(), *self._run_bearing)
        profile = [dict(node) for node in s.wind_profile_data]
        surface = s.get_wind_zoh(ANEMOMETER_ALT)
        if surface:
            profile = [dict(alt_m=ANEMOMETER_ALT, speed_ms=surface['speed_ms'], dir_deg=surface['dir_deg'])] + [
                node for node in profile if float(node['alt_m']) != ANEMOMETER_ALT]

        # ── Turbulence intensity from 60s surface wind history ────────────────
        # Descriptive standard deviation (ddof=0), not an estimate with a
        # guaranteed confidence level for the true atmospheric process.
        _hist = list(s.wind_history_for_alt(ANEMOMETER_ALT))
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

        params = {
            # ── Simulation / wind config ───────────────────────────────────────
            # cep_prob: read from AppState (authoritative) — updated by both the
            # main-window cep_prob_input widget and the Advanced Settings dialog.
            # Reading the widget directly would silently ignore Advanced Settings.
            "cep_prob":   int(s.landing_prob) if s.landing_prob is not None else 90,
            "launch_lat": w.lat_input.value()  if w.lat_input.value()  != -9999.0 else 35.6828,
            "launch_lon": w.lon_input.value()  if w.lon_input.value()  != -9999.0 else 139.7590,
            "elev":       s.launch_angle,   # degrees — AppState, default 85.0
            "rail":       s.launch_rail,    # m       — AppState, default 1.0
            **options,
            "azim":       azimuth,
            "wind_profile_data": profile,
            "surf_spd": float(surface['speed_ms']) if surface else 0.,
            "surf_dir": float(surface['dir_deg']) if surface else 0.,
            "mc_runs":    w.mc_runs_input.value(),
            "wind_unc":   w.wind_unc_input.value(),
            "thrust_unc": w.thrust_unc_input.value(),
            # ── Mode and gust — sourced from AppState ──────────────────────────
            "flight_mode":  s.flight_mode,
            "montblanc_options": s.montblanc_options,
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
            "cp_override":    s.rocket_cp,             # optional Mach-0 CP, m from nose
            **s.fin_options,
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
            "para_wind_drift_ratio": s.parachute_wind_drift_pct / 100.,
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
        if s.airframe_provenance:
            from core.airframe_parameters import resolve_airframe
            resolved = resolve_airframe({'provenance': s.airframe_provenance, 'cp':s.rocket_cp})
            mapping = dict(mass='airframe_mass', cg='airframe_cg', length='airframe_len',
                           nose_length='nose_len', nose_shape='nose_kind')
            for key,value in resolved.items():
                if key not in ('provenance','cp'):
                    params[mapping.get(key,key)] = value
            # Displayed calculated CP is a preview. Keep the live geometry model
            # for MC perturbations rather than forcing a stale CP calibration.
            params['cp_override'] = (resolved.get('cp')
                if resolved['provenance']['sources'].get('cp') in ('ork','json','manual') else None)
            for key,value in s.airframe_provenance.get('parachute', {}).items():
                params['para_'+key] = value
        params.update(run_id=str(uuid.uuid4()),input_revision=self._input_revision,
                      model_version='time-events-v2',
                      dependencies={k:version(k) for k in ('rocketpy','numpy','scipy','PySide6')})
        self._run_snapshot = deepcopy(params)
        return params

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
        self._airframe_input_refreshers = []

        def _bind(sb_name: str, prop: str, to_si, from_si) -> None:
            sb = getattr(w, sb_name, None)
            if sb is None:
                return

            from PySide6.QtWidgets import QLineEdit
            if isinstance(sb, QLineEdit):
                # Text input handling
                def validate_and_set(text: str, _sb=sb, _p=prop, _f=to_si):
                    if self._manual_editing and _sb.parent() is not None and w._manual_dialog.isAncestorOf(_sb):
                        return
                    val_str = text.strip()
                    if val_str == "":
                        _sb.setStyleSheet("" if _p == 'rocket_cp' else "border: 1px solid red;")
                        setattr(s, _p, None)
                    else:
                        try:
                            val_float = float(val_str)
                            _sb.setStyleSheet("")
                            setattr(s, _p, _f(val_float))
                        except ValueError:
                            _sb.setStyleSheet("border: 1px solid red;")
                            setattr(s, _p, None)

                sb.textChanged.connect(validate_and_set)
                
                def update_sb(v, _sb=sb, _g=from_si, _p=prop):
                    if self._manual_editing and w._manual_dialog.isAncestorOf(_sb):
                        return
                    if v is not None:
                        try:
                            if float(_sb.text()) == _g(v):
                                _sb.setStyleSheet("")
                                return
                        except ValueError:
                            pass
                    # Display rounding must not feed back into the loaded data.
                    with QSignalBlocker(_sb):
                        if v is None:
                            if _sb.text() != "":
                                _sb.setText("")
                            _sb.setStyleSheet("" if _p == 'rocket_cp' else "border: 1px solid red;")
                        else:
                            val_str = str(round(_g(v), 5))
                            if _sb.text() != val_str:
                                _sb.setText(val_str)
                            _sb.setStyleSheet("")
                sig = getattr(s, f"{prop}_changed", None)
                if sig is not None:
                    sig.connect(update_sb)
                self._airframe_input_refreshers.append((prop, update_sb))
                update_sb(getattr(s, prop, None))
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
        _bind("af_cp_input",       "rocket_cp",       lambda v: float(v),      lambda v: float(v))
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
            w.af_fincount_input.valueChanged.connect(lambda v: None if self._manual_editing else setattr(s, "fin_count", int(v)))
            s.fin_count_changed.connect(lambda v: w.af_fincount_input.setValue(int(v)) if v is not None else None)
        
        if hasattr(w, "af_noseshape_input"):
            w.af_noseshape_input.currentTextChanged.connect(lambda v: None if self._manual_editing else setattr(s, "nose_shape", str(v)))
            s.nose_shape_changed.connect(lambda v: w.af_noseshape_input.setCurrentText(str(v)) if v is not None else None)
        
        if hasattr(w, "hellmann_alpha_input"):
            w.hellmann_alpha_input.valueChanged.connect(lambda v: setattr(s, "hellmann_alpha", float(v)))
            s.hellmann_alpha_changed.connect(lambda v: w.hellmann_alpha_input.setValue(float(v)) if v is not None else None)

    def _refresh_airframe_inputs(self) -> None:
        """Also refresh unchanged values when the same file is loaded again."""
        for prop, update in self._airframe_input_refreshers:
            update(getattr(self._state, prop, None))

    def _manual_geometry_fields(self) -> dict:
        w = self._window
        return {
            "mass": (w.af_mass_input, "rocket_dry_mass", 1.0),
            "cg": (w.af_cg_input, "rocket_cg", 1.0),
            "length": (w.af_len_input, "rocket_length", 1.0),
            "radius": (w.af_radius_input, "rocket_diameter", 0.5),
            "nose_length": (w.af_nose_input, "nose_length", 1.0),
            "fin_root": (w.af_finroot_input, "fin_root_chord", 1.0),
            "fin_tip": (w.af_fintip_input, "fin_tip_chord", 1.0),
            "fin_span": (w.af_finspan_input, "fin_span", 1.0),
            "fin_pos": (w.af_finpos_input, "fin_position", 1.0),
        }

    def _manual_airframe_values(self) -> dict:
        values = dict(self._manual_pending_airframe or {})
        if self._manual_pending_airframe is None and self._state.airframe_provenance:
            values['provenance'] = self._state.airframe_provenance
        values.update({key: widget.text().strip()
                       for key, (widget, _, _) in self._manual_geometry_fields().items()})
        for key, prop in (("motor_pos", "motor_cg_pos"), ("motor_dry_mass", "motor_dry_mass"),
                          ("backfire_delay", "backfire_delay")):
            values.setdefault(key, getattr(self._state, prop))
        values["fin_count"] = self._window.af_fincount_input.value()
        values["nose_shape"] = self._window.af_noseshape_input.currentText()
        values['cp'] = self._window.af_cp_input.text().strip() or None
        values.update(self._window._manual_dialog.fin_options())
        if values.get('provenance'):
            from core.airframe_parameters import resolve_airframe
            values = resolve_airframe(values)
        return values

    def _refresh_airframe_sources(self, data=None):
        """Explain and lock higher-priority values; unfilled values remain editable."""
        w = self._window
        d = w._manual_dialog
        if data is None:
            data = self._manual_airframe_values()
        sources = data.get('provenance', {}).get('sources', {})
        fields = {key:widget for key,(widget,_,_) in self._manual_geometry_fields().items()}
        fields.update(cp=w.af_cp_input, fin_count=w.af_fincount_input,
                      nose_shape=w.af_noseshape_input, fin_shape=d.fin_shape_input,
                      fin_sweep=d.fin_sweep_input, fin_cant=d.fin_cant_input)
        if not self._manual_editing:
            fields.update(motor_pos=w.motor_cg_input, motor_dry_mass=w.motor_dry_mass_input,
                          backfire_delay=w.af_backfire_input)
        for key,widget in fields.items():
            source = sources.get(key)
            locked = source in ('ork','calculated')
            if hasattr(widget,'setReadOnly'):
                widget.setReadOnly(locked)
            else:
                widget.setEnabled(not locked)
            if widget.property('original_tooltip') is None:
                widget.setProperty('original_tooltip',widget.toolTip())
            widget.setToolTip(('出典: .ork' if source=='ork' else '出典: Kazamidori自動計算')
                              + '（手入力より優先）' if locked else widget.property('original_tooltip'))
            if source == 'json':
                widget.setToolTip('出典: 補完JSON（未取得項目の補完・編集可能）')
        locked_fin = sources.get('fin_shape') in ('ork','calculated')
        d.edit_fin_btn.setEnabled(not locked_fin)
        d.load_ork_fins_btn.setEnabled(not locked_fin)
        if not self._manual_editing:
            para = data.get('provenance',{}).get('parachute',{})
            for key,widget in (('cd',w.para_cd_input),('area',w.para_area_input)):
                widget.setReadOnly(key in para)
                widget.setToolTip('出典: OpenRocket（面積は直径から換算）' if key in para else '')
        d.show_airframe_sources(data)

    @Slot()
    def _on_manual_config_clicked(self) -> None:
        w = self._window
        self._refresh_airframe_inputs()
        w.af_fincount_input.setValue(self._state.fin_count)
        w.af_noseshape_input.setCurrentText(self._state.nose_shape)
        w._manual_dialog.set_fin_options(self._state.fin_options)
        self._manual_editing = True
        w._manual_draft_active = True
        self._manual_pending_airframe = None
        self._manual_pending_path = None
        w._manual_dialog.error_label.setText("")
        self._refresh_airframe_sources()
        try:
            w._manual_dialog.exec()
        finally:
            self._manual_editing = False
            w._manual_draft_active = False
            self._manual_pending_airframe = None
            self._manual_pending_path = None
            self._refresh_airframe_inputs()
            w.af_fincount_input.setValue(self._state.fin_count)
            w.af_noseshape_input.setCurrentText(self._state.nose_shape)
            w._manual_dialog.set_fin_options(self._state.fin_options)
            self._refresh_airframe_sources()
            w._evaluate_config_deltas()

    @Slot()
    def _on_apply_manual_config(self) -> None:
        values = self._manual_airframe_values()
        try:
            # Geometry can be edited before the motor's operational values exist.
            geometry = validate_airframe_config(values, self._manual_geometry_fields())
        except AirframeConfigError as exc:
            self._window._manual_dialog.error_label.setText(str(exc))
            return
        values.update(geometry)
        self._manual_editing = False
        self._window._manual_draft_active = False
        self._apply_airframe_config(values, self._manual_pending_path)
        self._window._manual_dialog.accept()

    @Slot()
    def _on_save_airframe_json(self) -> None:
        from pathlib import Path
        values = self._manual_airframe_values()
        try:
            values = validate_airframe_config(values)
        except AirframeConfigError as exc:
            self._window._manual_dialog.error_label.setText(str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self._window._manual_dialog, "Save airframe",
            str(Path(__file__).resolve().parent.parent / "Airframe.json"),
            "Airframe JSON (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            save_airframe_config(path, values)
        except AirframeConfigError as exc:
            self._window._manual_dialog.error_label.setText(str(exc))
            return
        self._window._manual_dialog.error_label.setText("")
        if self._manual_editing:
            self._manual_pending_path = path
        self._window.set_status(f"Airframe saved: {Path(path).name}")

    # ── Airframe JSON loader ───────────────────────────────────────────────────

    @Slot()
    def _on_load_airframe_json(self) -> None:
        """Select ORK with staged Apply/Cancel; retain legacy internal tuples."""
        from pathlib import Path

        selection = select_airframe_files(self._window._manual_dialog if self._manual_editing else self._window)
        if selection is None:
            return
        ork_path, json_path = selection
        path = ork_path or json_path
        try:
            af = load_airframe_config(path, json_path if ork_path and json_path else None)
        except AirframeConfigError as exc:
            self._window.set_status(f"Airframe load failed: {exc}", "#f38ba8")
            if self._manual_editing:
                self._window._manual_dialog.error_label.setText(str(exc))
            return

        if self._manual_editing:
            self._manual_pending_airframe = af
            self._manual_pending_path = path
            for key, (widget, _, _) in self._manual_geometry_fields().items():
                with QSignalBlocker(widget):
                    widget.setText(format(af[key], ".12g") if af.get(key) is not None else '')
            with QSignalBlocker(self._window.af_cp_input):
                self._window.af_cp_input.setText(format(af['cp'], '.12g') if af.get('cp') is not None else '')
            self._window._manual_dialog.set_fin_options(af)
            for widget, value in ((self._window.af_fincount_input, af.get("fin_count", self._state.fin_count)),):
                with QSignalBlocker(widget):
                    widget.setValue(value)
            with QSignalBlocker(self._window.af_noseshape_input):
                self._window.af_noseshape_input.setCurrentText(af.get("nose_shape", self._state.nose_shape))
            self._window._manual_dialog.error_label.setText("")
            self._refresh_airframe_sources(af)
            return

        self._apply_airframe_config(af, path)

    def _apply_airframe_config(self, af: dict, path=None) -> None:
        from pathlib import Path

        s = self._state
        fields = {
            "mass": "rocket_dry_mass", "cg": "rocket_cg",
            "length": "rocket_length", "nose_length": "nose_length",
            "fin_root": "fin_root_chord", "fin_tip": "fin_tip_chord",
            "fin_span": "fin_span", "fin_pos": "fin_position",
            "motor_pos": "motor_cg_pos", "motor_dry_mass": "motor_dry_mass",
            "backfire_delay": "backfire_delay",
        }
        if af.get('provenance'):
            from core.airframe_parameters import resolve_airframe
            af = resolve_airframe(af)
        from copy import deepcopy
        s.airframe_provenance = deepcopy(af.get('provenance', {}))
        if path:
            s.original_rocket_config = dict(af)
        s.rocket_cp = af.get('cp')  # Legacy files restore automatic CP.
        s.fin_options = {k: af.get(k, default) for k, default in dict(
            fin_shape='trapezoid', fin_points=None, fin_sweep=None, fin_cant=0.).items()}
        for key, prop in fields.items():
            if af.get(key) is not None or (s.airframe_provenance and key in af):
                setattr(s, prop, af[key])
        if "fin_count" in af:
            s.fin_count = af["fin_count"]
        if "nose_shape" in af:
            s.nose_shape = af["nose_shape"]
        s.rocket_diameter = af["radius"] * 2.0
        # Airframe.json has no inertia tensor; discard any previous model's MoI.
        s.set_moi(0.0, 0.0, 0.0)
        self._refresh_airframe_inputs()
        self._window._manual_dialog.set_fin_options(s.fin_options)
        if s.airframe_provenance:
            para = s.airframe_provenance.get('parachute', {})
            if path:
                # A new design must not inherit the previous design's recovery
                # system or motor curve without the user choosing them.
                s.parachute_cd = para.get('cd')
                s.parachute_area = para.get('area')
                s.parachute_lag = None
                self._window.parachute_label.setText(
                    f"{s.airframe_provenance.get('file','OpenRocket')} (Cd / Area)" if para else 'No parachute loaded')
                self._window._motor_thrust_data = None
                self._window._motor_burn_time = None
                self._window.motor_label.setText(
                    f".ork: {s.airframe_provenance.get('motor') or '未指定'} — Load Thrust Curve")
                for label in (self._window.lbl_avg_thrust, self._window.lbl_max_thrust,
                              self._window.lbl_burn_time, self._window.lbl_total_impulse):
                    label.setText('—')
            else:
                if 'cd' in para:
                    s.parachute_cd = para['cd']
                if 'area' in para:
                    s.parachute_area = para['area']
        self._refresh_airframe_sources(af)

        w = self._window
        name = Path(path).name if path else w.airframe_label.text().split(" (Modified)")[0]
        if path and s.airframe_provenance.get('source_path'):
            name = Path(s.airframe_provenance['source_path']).name
            supplement = (path if str(path).lower().endswith('.json')
                          else s.airframe_provenance.get('supplement_path'))
            if supplement:
                name += f" + {Path(supplement).name}"
        if not path:
            if name.startswith("No airframe"):
                name = "Manual configuration"
            name += " (Modified)"
        w.airframe_label.setText(
            name
        )
        w.airframe_label.setToolTip(
            f"Mass: {af.get('mass')} kg | CG: {af.get('cg')} m | L: {af['length']:.3f} m\n"
            + w._manual_dialog.airframe_sources_note.text()
        )
        w.airframe_label.setStyleSheet(
            "color: #a6e3a1; font-size: 8pt; padding: 2px 4px;"
        )
        w._evaluate_config_deltas()
        w.set_status(
            f"Airframe loaded: {name}  |  CG {af.get('cg')} m  |  "
            "Manual Configで出典・未取得項目を確認できます",
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
        imported = s.airframe_provenance.get('parachute', {})
        s.parachute_cd   = imported.get('cd', par["cd"])
        s.parachute_area = imported.get('area', par["area"])
        s.parachute_lag  = par["lag"]
        s.parachute_wind_drift_pct = par.get('wind_drift_pct', 100.)

        name = _os.path.basename(path)
        self._window.parachute_label.setText(name + (' (Lag; Cd/Area: .ork)' if imported else ''))
        self._window.parachute_label.setToolTip(path)
        self._window.set_status(
            f"Parachute loaded: {name}  ·  "
            f"Cd {s.parachute_cd:.2f}  ·  "
            f"Area {s.parachute_area:.4f} m²  ·  "
            f"Lag {par['lag']:.2f} s",
            "#a6e3a1",
        )

    @Slot()
    def _on_wind_tick(self) -> None:
        # Surface and upper wind baselines now live in wind_profile_data
        wp = self._state.wind_profile_data or []
        forecast = self._forecast_nodes if self._state.phase4_options['wind_csv'] else {}
        if forecast:
            wp = [n for n in wp if float(n['alt_m']) == ANEMOMETER_ALT] + list(forecast.values())
        lowest_wind = sorted(wp, key=lambda n: n["alt_m"])[0] if wp else {"speed_ms": 0.0, "dir_deg": 0.0}
        highest_wind = sorted(wp, key=lambda n: n["alt_m"])[-1] if wp else {"speed_ms": 0.0, "dir_deg": 0.0}
        
        base_spd  = float(lowest_wind.get("speed_ms", 0.0))
        base_dir  = float(lowest_wind.get("dir_deg", 0.0))
        up_spd    = float(highest_wind.get("speed_ms", 0.0))
        up_dir    = float(highest_wind.get("dir_deg", 0.0))
        speed     = max(0.0, base_spd + random.gauss(0.0, base_spd * 0.05 + 0.1))
        direction = (base_dir + random.gauss(0.0, 3.0)) % 360.0

        # Global AppState: (speed, direction) tuples for future Phase-2 consumers.
        nodes = [dict(n) if float(n['alt_m']) in forecast else dict(alt_m=float(n['alt_m']),
                      speed_ms=max(0., float(n['speed_ms']) + random.gauss(0., float(n['speed_ms'])*.05+.1)),
                      dir_deg=(float(n['dir_deg'])+random.gauss(0.,3.))%360)
                 for n in wp]
        if nodes:
            nodes.sort(key=lambda n: n['alt_m'])
            speed, direction = nodes[0]['speed_ms'], nodes[0]['dir_deg']
            self._state.append_wind_nodes(nodes)
        else:
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
        if not self._state.is_calculating:
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
        """Apply Ctrl-click coordinates without starting another operation."""
        from utils.map_downloader import validate_map_coordinates

        try:
            lat, lon = validate_map_coordinates(lat, lon)
        except ValueError as exc:
            self._window.set_status(str(exc), "#f38ba8")
            return
        self._state.launch_lat = lat
        self._state.launch_lon = lon
        self._window.map_widget.update_launch(lat, lon)
        self._window.set_status(
            f"Launch coordinates selected: {lat:.6f}, {lon:.6f}", "#a6e3a1")
        self._window.location_label.setText("Coordinates selected on the map. Current location has not been acquired.")

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
        # Render from the shared result so View toggles also apply after a
        # percentile edit. The former update_ellipse_layer API no longer exists.
        sr = self._state.simulation_result
        if isinstance(sr, dict) and pts is not None and len(pts):
            sr["landing_prob"] = int(round(self._state.cep_probability))
            # Use the same empirical radius convention as the worker.
            radii = np.sort(np.linalg.norm(pts - pts.mean(axis=0), axis=1))
            sr["r_N_radius"] = sr["cep_radius"] = float(np.quantile(radii, self._state.cep_probability / 100.0))
            self._state.r90_radius = sr["r_N_radius"]
            self._state.simulation_result = sr
            if self._state.phase2_active:
                self._state.phase2_active = False
                self._window.update_status_indicator('確率を変更しました。GO/NO-GO判定は再計算してください。')

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

    @Slot()
    def _on_import_results(self) -> None:
        from pathlib import Path
        from PySide6.QtCore import QSignalBlocker
        from utils.result_io import load_archive
        from utils.geo_math import offset_to_latlon

        s, w = self._state, self._window
        if s.is_calculating:
            w.set_status("Wait for the current operation before importing results.", "#f9e2af")
            return
        filepath, _ = QFileDialog.getOpenFileName(
            w, "Import Results", "", "Kazamidori result archives (*.json)")
        if not filepath:
            return
        try:
            archive = load_archive(filepath)
            result, context = archive["result"], archive["context"]
            nominal = archive.get("nominal") or result
            scatter = list(zip(result.get("mc_scatter_x", []), result.get("mc_scatter_y", [])))
            lat, lon = context.get("launch_lat"), context.get("launch_lon")
            landing = (offset_to_latlon(lat, lon, result.get("land_x", 0), result.get("land_y", 0))
                       if lat is not None else (None, None))
        except (OSError, ValueError, TypeError) as exc:
            w.set_status(f"Import failed: {exc}", "#f38ba8")
            return

        # All data is validated before updating shared state. Historical results
        # do not start a simulation, download a map, or activate wind monitoring.
        s.phase2_active = False
        s.cached_mc_scatter = None  # Do not recompute the old result on probability change.
        s.launch_lat, s.launch_lon = lat, lon
        s.target_x,s.target_y = context.get('target_x',0.),context.get('target_y',0.)
        w.mode_combo.setCurrentText(context["flight_mode"])
        s.flight_mode = context["flight_mode"]
        if 'montblanc_options' in context:
            s.montblanc_options = context['montblanc_options']
        # Mode changes install convenient defaults; restore the recorded
        # radius afterwards so importing cannot silently change the target.
        s.target_radius = context.get("target_radius")
        probability = result.get("landing_prob", 90)
        s.landing_prob = int(probability)
        s.cep_probability = float(probability)
        s.mc_scatter = scatter
        s.cached_mc_scatter = np.asarray(scatter, dtype=float).reshape(-1, 2)
        s.mc_ellipse = result.get("ellipse")
        s.kde_contours = result.get("kde_contours", [])
        s.r90_radius = result.get("r_N_radius", 0.0)
        s.mc_cep = result.get("cep", 0.0)
        if lat is not None:
            s.land_lat, s.land_lon = landing
        s.has_sim_result = True
        self._nominal_payload = nominal
        # Only the final result assignment should trigger the profile redraw.
        with QSignalBlocker(s):
            s.nominal_result = nominal
        s.simulation_result = result
        self.populate_results(nominal, result)
        if lat is not None:
            w.map_widget.update_launch(lat, lon)
            w.map_widget.update_landing(*landing)
        else:
            w._map_launch_lbl.setText("Launch: —")
            w._map_landing_lbl.setText("Landing: —")
        w.location_label.setText("Launch coordinates restored from result archive; current location has not been acquired.")
        w.update_status_indicator("IMPORTED RESULT — wind tolerance monitoring inactive")
        w.set_progress(0, "Imported")
        w.set_status(f"Results imported: {Path(filepath).name}", "#a6e3a1")

    def _set_run_buttons_enabled(self, enabled: bool) -> None:
        from PySide6.QtWidgets import QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox
        if not enabled and not getattr(self,'_disabled_controls',None):
            controls = []
            for kind in (QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QPushButton):
                controls.extend(self._window.findChildren(kind))
            self._disabled_controls = [c for c in controls if c.isEnabled() and c.objectName() != 'btn_stop']
            for control in self._disabled_controls:
                control.setEnabled(False)
        elif enabled:
            for control in getattr(self,'_disabled_controls',[]):
                control.setEnabled(True)
            self._disabled_controls = []
        self._window.action_run.setEnabled(enabled)
        self._window.action_stop.setEnabled(self._state.is_calculating)
        for btn in self._window.findChildren(QPushButton, "btn_run"):
            btn.setEnabled(enabled)
        for btn in self._window.findChildren(QPushButton, "btn_stop"):
            btn.setEnabled(self._state.is_calculating)
