"""
ui/app_window.py
Main application window — parameter panel, simulation control, Phase 1/2
orchestration.

Delegates all rendering to PlotView (centre column) and MapView (right column).
Heavy computation runs in daemon threads; results are delivered via queue and
consumed by .after() polls on the Tk main thread.
"""

from __future__ import annotations

import json
import math
import os
import queue
import random
import sys
import threading
import time
import webbrowser
from collections import deque
from tkinter import messagebox, filedialog, ttk
from typing import Optional
import tkinter as tk

import numpy as np
import requests

from core.optimization import build_wind_profile, run_phase1
from core.simulation import simulate_once
from core.monte_carlo import (
    run_mc_scatter,
    compute_error_ellipse,
    compute_error_ellipse_polygon,
    compute_cep,
    compute_cep_polygon,
    compute_kde_contours,
)
from monitor.wind_tracker import WindTracker
from monitor.status_manager import evaluate as p2_evaluate
from utils.geo_math import offset_to_latlon

from ui.plot_view import PlotView
from ui.map_view import MapView


class AppWindow(tk.Tk):

    OPERATION_MODES = (
        "Altitude Competition",
        "Precision Landing",
        "Winged Hover",
        "Free",
    )
    _MODE_DEFAULT_RMAX = {
        "Free": None,
        "Precision Landing":    50.0,
        "Altitude Competition": 250.0,
        "Winged Hover":         250.0,
    }
    _CHI2_2DOF = {50: 1.386, 68: 2.296, 80: 3.219, 85: 3.794,
                  90: 4.605, 95: 5.991, 99: 9.210}

    def __init__(self, config: dict = None) -> None:
        super().__init__()
        _cfg = config or {}
        self.title("Kazamidori_Project - Trajectory & Landing Point Simulator")
        self.geometry("1250x880")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.columnconfigure(0, weight=1, minsize=350)
        self.columnconfigure(1, weight=2)
        self.columnconfigure(2, weight=3, minsize=500)
        self.rowconfigure(0, weight=1)

        # ── Launch / landing state ─────────────────────────────────────────────
        self.launch_lat = 35.6828
        self.launch_lon = 139.7590
        self.land_lat   = self.launch_lat
        self.land_lon   = self.launch_lon
        self.r90_radius = 10.0

        # ── Motor / thrust state ───────────────────────────────────────────────
        self.selected_motor_file = None
        self.selected_motor_name = "(none selected)"
        self.motor_burn_time     = 0.0
        self.motor_avg_thrust    = None
        self.motor_max_thrust    = None
        self.thrust_data         = None

        # ── Rolling wind history ───────────────────────────────────────────────
        self.surf_wind_history      = deque(maxlen=300)
        self.surf_wind_time_history = deque(maxlen=300)
        self._wind_start_time       = time.time()
        self._sim_base_wind         = 4.0

        # ── Wind tracker for Phase 2 (GO/NO-GO) monitoring ────────────────────
        self.wind_tracker = WindTracker(maxlen=300)

        # ── Uncertainty / probability settings (overridable via config) ───────
        self.wind_uncertainty      = float(_cfg.get("wind_uncertainty",      0.20))
        self.thrust_uncertainty    = float(_cfg.get("thrust_uncertainty",    0.05))
        self.allowable_uncertainty = float(_cfg.get("allowable_uncertainty", 20.0))
        self.landing_prob          = int(_cfg.get("landing_prob",            90))

        # ── Lock & Monitor state ───────────────────────────────────────────────
        self._baseline_wind    = None
        self._monitor_after_id = None
        self._settings_win     = None
        self._last_sim_data    = None

        # ── Simulation result bookkeeping ─────────────────────────────────────
        self._has_sim_result = False

        # ── Wind direction StringVars ──────────────────────────────────────────
        self.surf_dir_var = tk.StringVar(value="100")
        self.up_spd_var   = tk.StringVar(value="8.0")
        self.up_dir_var   = tk.StringVar(value="90")

        # ── Operation mode state ───────────────────────────────────────────────
        self.operation_mode_var = tk.StringVar(value="Free")
        self.r_max_var          = tk.StringVar(value="50.0")
        self._last_optimization_info = None

        # ── Phase 1 state ──────────────────────────────────────────────────────
        self._p1_queue     = queue.Queue()
        self._p1_stop_flag = threading.Event()
        self._p1_thread    = None
        self._p1_running   = False
        self._phase1_result = None
        self._p1_win        = None
        self._p1_win_bar    = None
        self._p1_win_msg    = None

        # ── Phase 2 state ──────────────────────────────────────────────────────
        self._p2_after_id = None
        self._p2_ellipse  = None

        # ── MC visualization state ─────────────────────────────────────────────
        self._mc_scatter         = None
        self._mc_ellipse         = None
        self._mc_ellipse_polygon = None   # lat/lon polygon for map display
        self._mc_cep             = None
        self._mc_cep_polygon     = None   # dict with cx_m, cy_m, radius_m, latlons
        self._mc_running         = False
        self._mc_queue           = queue.Queue()
        self._mc_n_runs          = int(_cfg.get("mc_n_runs", 200))
        self._mc_wind_profiles   = None
        self._kde_contours       = None

        # ── Build UI ───────────────────────────────────────────────────────────
        self.create_data_section()
        self.plot_view = PlotView(self)
        self.map_view  = MapView(self, self.launch_lat, self.launch_lon)
        self.map_view.set_fit_command(self.fit_map_bounds)

        self.after(500,  lambda: self.get_current_location(manual=False))
        self.after(1000, self.simulate_realtime_wind)
        self.update_plots()
        self.after(1000, self.fit_map_bounds)
        self.after(0,    self._on_mode_change)

    # ── Application lifecycle ─────────────────────────────────────────────────

    def on_closing(self) -> None:
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self._p1_stop_flag.set()
            if self._p2_after_id is not None:
                try:
                    self.after_cancel(self._p2_after_id)
                except Exception:
                    pass
            import matplotlib.pyplot as plt
            plt.close('all')
            self.quit()
            self.destroy()
            sys.exit()

    # ── Location ──────────────────────────────────────────────────────────────

    def get_current_location(self, manual: bool = False) -> None:
        try:
            response = requests.get('https://ipinfo.io/json', timeout=3)
            loc = response.json().get('loc', '')
            if loc:
                lat, lon = loc.split(',')
                self.launch_lat, self.launch_lon = float(lat), float(lon)
                self.lat_entry.delete(0, tk.END)
                self.lat_entry.insert(0, lat)
                self.lon_entry.delete(0, tk.END)
                self.lon_entry.insert(0, lon)
                self.map_view.set_position(self.launch_lat, self.launch_lon)
                self._clear_previous_landing()
                self.update_plots()
                self.fit_map_bounds()
                if manual:
                    messagebox.showinfo("Location Retrieved",
                                        f"Location acquired:\nLat: {lat}\nLon: {lon}")
        except Exception as e:
            if manual:
                messagebox.showerror("Fetch Error", f"Failed to get location.\n{e}")

    # ── Realtime wind simulation ──────────────────────────────────────────────

    def simulate_realtime_wind(self) -> None:
        base_wind = (sum(self.surf_wind_history) / len(self.surf_wind_history)
                     if self.surf_wind_history else self._sim_base_wind)
        current_wind = max(0.0, random.gauss(base_wind, base_wind * 0.15))
        if random.random() < 0.05:
            current_wind *= 1.5

        t_now = time.time() - self._wind_start_time
        self.surf_wind_history.append(current_wind)
        self.surf_wind_time_history.append((t_now, current_wind))

        # Push to WindTracker for Phase 2 monitoring
        try:
            surf_dir = float(self.surf_dir_var.get())
            up_spd   = float(self.up_spd_var.get())
            up_dir   = float(self.up_dir_var.get())
            self.wind_tracker.push(t_now, current_wind, surf_dir, up_spd, up_dir)
        except Exception:
            self.wind_tracker.push(t_now, current_wind)

        try:
            self._update_realtime_wind_label()
        except Exception:
            pass
        try:
            self._update_wind_subplots()
        except Exception:
            pass
        self.after(1000, self.simulate_realtime_wind)

    # ── Wind helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _wind_components(spd: float, dir_deg: float):
        rad = math.radians(dir_deg)
        return -spd * math.sin(rad), -spd * math.cos(rad)

    def _read_current_wind(self):
        def _get(var_name, default):
            v = getattr(self, var_name, None)
            if v is None:
                return default
            try:
                return float(v.get())
            except Exception:
                return default
        hist     = getattr(self, 'surf_wind_history', None) or []
        surf_spd = float(hist[-1]) if hist else self._sim_base_wind
        surf_dir = _get('surf_dir_var', 0.0)
        up_spd   = _get('up_spd_var',   0.0)
        up_dir   = _get('up_dir_var',   0.0)
        return surf_spd, surf_dir, up_spd, up_dir

    def _wind_avg_recent(self, window_sec: float = 10.0) -> float:
        history = list(getattr(self, 'surf_wind_time_history', []) or [])
        if not history:
            return self._sim_base_wind
        t_latest = history[-1][0]
        recent   = [w for (t, w) in history if t >= t_latest - window_sec]
        if not recent:
            recent = [history[-1][1]]
        return sum(recent) / len(recent)

    def _capture_wind_baseline(self) -> None:
        try:
            self._baseline_wind = {
                "surf_spd": self._wind_avg_recent(window_sec=10.0),
                "surf_dir": float(self.surf_dir_var.get()),
                "up_spd":   float(self.up_spd_var.get()),
                "up_dir":   float(self.up_dir_var.get()),
            }
        except (ValueError, AttributeError):
            self._baseline_wind = None

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return abs(((a - b) + 180.0) % 360.0 - 180.0)

    # ── Simulation parameter gathering ───────────────────────────────────────

    def _gather_sim_params(self) -> Optional[dict]:
        try:
            launch_lat       = float(self.lat_entry.get())
            launch_lon       = float(self.lon_entry.get())
            elev             = float(self.elev_spin.get())
            azi              = float(self.azi_spin.get())
            rail             = float(self.rail_entry.get())
            airframe_mass    = float(self.mass_entry.get())
            airframe_cg      = float(self.cg_entry.get())
            airframe_len     = float(self.len_entry.get())
            radius           = float(self.radius_entry.get())
            nose_len         = float(self.nose_len_entry.get())
            fin_root         = float(self.fin_root_entry.get())
            fin_tip          = float(self.fin_tip_entry.get())
            fin_span         = float(self.fin_span_entry.get())
            fin_pos          = float(self.fin_pos_entry.get())
            motor_pos        = float(self.motor_pos_entry.get())
            motor_dry_mass   = float(self.motor_dry_mass_entry.get())
            backfire_delay   = float(self.backfire_delay_entry.get())
            para_cd          = float(self.cd_entry.get())
            para_area        = float(self.area_entry.get())
            para_lag         = float(self.lag_entry.get())
        except ValueError:
            messagebox.showerror(
                "Input Error",
                "Some parameters are missing.\nPlease fill in all fields with valid numbers.")
            return None

        self.launch_lat = launch_lat
        self.launch_lon = launch_lon

        if self.surf_wind_history:
            _ws    = list(self.surf_wind_history)
            _mu    = sum(_ws) / len(_ws)
            _sigma = (float(np.std(_ws)) if len(_ws) > 1 else _mu * 0.10)
            surf_spd = max(0.0, random.gauss(_mu, _sigma))
        else:
            surf_spd = self._sim_base_wind
        surf_dir = float(self.surf_dir_var.get())
        up_spd   = float(self.up_spd_var.get())
        up_dir   = float(self.up_dir_var.get())

        surf_u, surf_v = self._wind_components(surf_spd, surf_dir)
        up_u,   up_v   = self._wind_components(up_spd,   up_dir)
        wind_u_prof = [(0, 0), (3, surf_u), (100, up_u), (5000, up_u)]
        wind_v_prof = [(0, 0), (3, surf_v), (100, up_v), (5000, up_v)]

        if not self.thrust_data:
            messagebox.showerror(
                "No Engine Selected",
                "No engine has been selected.\nLoad a thrust CSV using [Load Local CSV].")
            return None

        return {
            'launch_lat': launch_lat, 'launch_lon': launch_lon,
            'elev': elev, 'azi': azi, 'rail': rail,
            'airframe_mass': airframe_mass, 'airframe_cg': airframe_cg,
            'airframe_len': airframe_len, 'radius': radius,
            'nose_len': nose_len, 'fin_root': fin_root, 'fin_tip': fin_tip,
            'fin_span': fin_span, 'fin_pos': fin_pos,
            'motor_pos': motor_pos, 'motor_dry_mass': motor_dry_mass,
            'backfire_delay': backfire_delay,
            'para_cd': para_cd, 'para_area': para_area, 'para_lag': para_lag,
            'surf_spd': surf_spd, 'surf_dir': surf_dir,
            'up_spd': up_spd, 'up_dir': up_dir,
            'wind_u_prof': wind_u_prof, 'wind_v_prof': wind_v_prof,
            'thrust_data': [list(p) for p in self.thrust_data],
            'motor_burn_time': self.motor_burn_time,
        }

    # ── Single simulation ─────────────────────────────────────────────────────

    def _simulate_once(self, elev: float, azi: float, params: dict) -> dict:
        return simulate_once(elev, azi, params)

    def _apply_sim_result_to_ui(self, res: dict, params: dict,
                                 override_r90=None) -> dict:
        x_vals  = res['x_vals'];  y_vals  = res['y_vals']
        z_vals  = res['z_vals'];  vz_vals = res['vz_vals']
        t_vals  = res['t_vals']
        elev    = res['elev'];    azi     = res['azi']

        azi_rad   = math.radians(azi)
        downrange = x_vals * math.sin(azi_rad) + y_vals * math.cos(azi_rad)

        idx_bf   = res['idx_bf']
        idx_para = res['idx_para']
        bf_z_val = float(z_vals[idx_bf])   if 0 <= idx_bf   < len(z_vals)   else 0.0
        bf_dr    = float(downrange[idx_bf]) if 0 <= idx_bf   < len(downrange) else 0.0

        self.land_lat, self.land_lon = offset_to_latlon(
            params['launch_lat'], params['launch_lon'],
            res['impact_x'], res['impact_y'])

        apogee_idx     = res['apogee_idx']
        fall_time      = t_vals[-1] - t_vals[apogee_idx]
        horiz_dist     = res['r_horiz']
        surf_spd       = params['surf_spd']
        wind_sigma_m   = surf_spd * self.wind_uncertainty * max(fall_time, 0.0)
        thrust_sigma_m = self.thrust_uncertainty * horiz_dist
        combined_sigma = math.hypot(wind_sigma_m, thrust_sigma_m)
        z_score        = self._prob_to_z(self.landing_prob)
        self.r90_radius = z_score * combined_sigma
        if (override_r90 is not None
                and isinstance(override_r90, (int, float))
                and math.isfinite(override_r90)
                and override_r90 > 0):
            self.r90_radius = float(override_r90)

        self.apogee_label.config(text=f"Apogee: {res['apogee_m']:.1f} m")
        self.velocity_label.config(text=f"Impact Vel: {abs(float(vz_vals[-1])):.1f} m/s")

        self._baseline_wind = {
            "surf_spd": params['surf_spd'], "surf_dir": params['surf_dir'],
            "up_spd":   params['up_spd'],   "up_dir":   params['up_dir'],
        }

        sim_data = {
            'x': x_vals, 'y': y_vals, 'z': z_vals,
            'downrange': downrange,
            'impact_dr': downrange[-1], 'r90': self.r90_radius,
            'wind_u_prof': params['wind_u_prof'],
            'wind_v_prof': params['wind_v_prof'],
            'azi': azi,
            'bf_z': bf_z_val, 'bf_dr': bf_dr,
            'bf_time': res['bf_abs_time'],
            'bf_x': float(x_vals[idx_bf]) if 0 <= idx_bf < len(x_vals) else 0.0,
            'bf_y': float(y_vals[idx_bf]) if 0 <= idx_bf < len(y_vals) else 0.0,
            'para_time': res['para_open_time'],
            'idx_para': idx_para, 'idx_bf': idx_bf,
            'impact_x': res['impact_x'], 'impact_y': res['impact_y'],
            'apogee_m': res['apogee_m'],
            'r_horiz': res['r_horiz'],
            'hang_time': res['hang_time'],
            'fall_time': fall_time,
            'surf_spd': params['surf_spd'],
        }
        self._has_sim_result = True
        self._last_sim_data  = sim_data
        self.update_plots(sim_data)
        self.fit_map_bounds()
        return sim_data

    def run_simulation(self) -> None:
        self._last_optimization_info = None
        self._render_current_params()

    def _render_current_params(self, override_r90=None) -> bool:
        params = self._gather_sim_params()
        if params is None:
            return False
        res = simulate_once(params['elev'], params['azi'], params)
        if not res['ok']:
            if "ZeroDivisionError" in res.get('error', ''):
                messagebox.showerror(
                    "Launch Failure / Unstable Attitude",
                    "Simulation diverged and could not complete.\n\n"
                    "Likely causes:\n"
                    "1. Motor thrust too weak to leave the rail\n"
                    "2. Aerodynamic parameters (CG, fins) make the rocket highly unstable\n\n"
                    f"Selected engine: {self.selected_motor_name}\n")
            else:
                messagebox.showerror(
                    "Sim Error",
                    f"RocketPy execution error:\n{res.get('error', 'unknown')}")
            return False
        self._apply_sim_result_to_ui(res, params, override_r90=override_r90)
        self._mc_scatter         = None
        self._mc_ellipse         = None
        self._mc_ellipse_polygon = None
        self._mc_cep             = None
        self._mc_cep_polygon     = None
        self._kde_contours       = None
        self._start_mc_visualization(params)
        return True

    # ── MC visualization ──────────────────────────────────────────────────────

    def _start_mc_visualization(self, params: dict) -> None:
        if self._mc_running:
            return
        self._mc_running = True
        t = threading.Thread(
            target=self._mc_viz_worker,
            args=(params, self._mc_n_runs, self._mc_queue),
            daemon=True)
        t.start()
        self.after(400, self._poll_mc_viz_queue)

    def _mc_viz_worker(self, params: dict, n_runs: int,
                       result_queue: queue.Queue) -> None:
        scatter, wind_profiles = run_mc_scatter(
            params, n_runs, self.wind_uncertainty, self.thrust_uncertainty)
        result_queue.put((scatter, wind_profiles))

    def _poll_mc_viz_queue(self) -> None:
        try:
            result = self._mc_queue.get_nowait()
        except queue.Empty:
            self.after(400, self._poll_mc_viz_queue)
            return
        self._mc_running = False
        scatter, wind_profiles = result
        self._apply_mc_viz_results(scatter, wind_profiles)

    def _apply_mc_viz_results(self, scatter, wind_profiles=None) -> None:
        if len(scatter) < 4:
            return
        self._mc_scatter       = scatter
        self._mc_wind_profiles = wind_profiles

        # Metric ellipse for the 3-D matplotlib plot (cx/cy/a/b in metres)
        ellipse = compute_error_ellipse(scatter, prob_pct=self.landing_prob)
        if ellipse is not None:
            self._mc_ellipse = ellipse

        if self.launch_lat is not None:
            # Lat/lon polygon for the map — all stats in metres, converted at end
            self._mc_ellipse_polygon = compute_error_ellipse_polygon(
                scatter, self.launch_lat, self.launch_lon,
                prob_pct=self.landing_prob)

            # CEP circle: centroid + 50th-percentile radius, in both frames
            cep_data = compute_cep_polygon(scatter, self.launch_lat, self.launch_lon)
            self._mc_cep_polygon = cep_data
            # Keep scalar for the 3D plot banner (radius in metres)
            self._mc_cep = cep_data['radius_m'] if cep_data is not None else compute_cep(scatter)

            self._kde_contours = compute_kde_contours(
                scatter, self.launch_lat, self.launch_lon,
                conf_pct=self.landing_prob)
        else:
            self._mc_cep = compute_cep(scatter)

        if self._last_sim_data is not None:
            self.update_plots(self._last_sim_data)
        self.draw_map_elements()

    # ── Rendering delegators ──────────────────────────────────────────────────

    def _get_r_max_val(self) -> Optional[float]:
        try:
            return float(self.r_max_var.get())
        except Exception:
            return None

    def update_plots(self, data=None) -> None:
        self.plot_view.update_3d(
            data,
            mc_scatter     = self._mc_scatter,
            mc_ellipse     = self._mc_ellipse,
            mc_cep         = self._mc_cep,
            mc_running     = self._mc_running,
            r90_radius     = self.r90_radius,
            landing_prob   = self.landing_prob,
            phase1_result  = self._phase1_result,
            last_opt_info  = self._last_optimization_info,
            operation_mode = self.operation_mode_var.get(),
            r_max_val      = self._get_r_max_val(),
        )
        try:
            self._update_wind_subplots()
        except Exception:
            pass
        self.draw_map_elements()
        try:
            self.fit_map_bounds()
        except Exception:
            pass

    def _update_wind_subplots(self) -> None:
        surf_spd, surf_dir, up_spd, up_dir = self._read_current_wind()
        avg_10s = self._wind_avg_recent(window_sec=10.0)
        self.plot_view.update_wind(
            list(self.surf_wind_time_history),
            surf_dir, up_spd, up_dir,
            mc_wind_profiles = self._mc_wind_profiles,
            wind_avg_recent  = avg_10s,
        )
        try:
            self._update_realtime_wind_label()
        except Exception:
            pass

    def _update_realtime_wind_label(self) -> None:
        try:
            surf_spd, surf_dir, up_spd, up_dir = self._read_current_wind()
            hist = getattr(self, 'surf_wind_history', None) or []
            gust = max(hist) if hist else surf_spd
            self.plot_view.update_realtime_wind_label(
                surf_spd, surf_dir, up_spd, up_dir, gust)
        except Exception:
            pass

    def draw_map_elements(self) -> None:
        try:
            mode     = self.operation_mode_var.get()
            r_target = float(self.r_max_var.get()) if mode != 'Free' else 50.0
        except Exception:
            r_target = 50.0

        self.map_view.draw_elements(
            launch_lat         = self.launch_lat,
            launch_lon         = self.launch_lon,
            land_lat           = self.land_lat,
            land_lon           = self.land_lon,
            r_target           = r_target,
            r90                = self.r90_radius,
            has_sim_result     = self._has_sim_result,
            p2_ellipse         = self._p2_ellipse,
            kde_contours       = self._kde_contours,
            mc_ellipse_polygon = self._mc_ellipse_polygon,
            mc_cep_polygon     = self._mc_cep_polygon,
        )

    def fit_map_bounds(self) -> None:
        self.map_view.fit_bounds(
            launch_lat     = self.launch_lat,
            launch_lon     = self.launch_lon,
            land_lat       = self.land_lat,
            land_lon       = self.land_lon,
            r90            = self.r90_radius,
            has_sim_result = self._has_sim_result,
        )

    # ── Operation mode ────────────────────────────────────────────────────────

    def _on_mode_change(self, event=None) -> None:
        mode = self.operation_mode_var.get()
        self._apply_mode_default_rmax(mode)
        if mode == "Free":
            try:
                self.rmax_label.grid_remove()
                self.rmax_entry.grid_remove()
            except Exception:
                pass
        else:
            try:
                self.rmax_label.grid()
                self.rmax_entry.grid()
            except Exception:
                pass
            self._release_lock_if_active(reason_label="⭘ Unlocked")
        try:
            self.lock_monitor_check.state(["!disabled"])
        except Exception:
            pass
        self._update_main_action_btn()

    def _apply_mode_default_rmax(self, mode: str) -> None:
        default = self._MODE_DEFAULT_RMAX.get(mode)
        if default is None:
            return
        try:
            self.r_max_var.set(f"{default:.1f}")
        except Exception:
            pass

    def _update_main_action_btn(self) -> None:
        btn = getattr(self, 'main_action_btn', None)
        if btn is None:
            return
        mode = self.operation_mode_var.get()
        if mode == 'Free':
            btn.config(text='🚀 Run Single Simulation',
                       command=self._render_current_params)
        else:
            btn.config(text='▶ Phase 1: Launch Angle Optimization',
                       command=self._start_phase1)

    def _p1_objective_score(self, res: dict, mode: str) -> float:
        if mode == 'Altitude Competition':
            return res['apogee_m']
        elif mode == 'Precision Landing':
            return -res['r_horiz']
        elif mode == 'Winged Hover':
            return res['hang_time']
        return res['apogee_m']

    # ── Uncertainty helpers ───────────────────────────────────────────────────

    def _prob_to_z(self, pct: int) -> float:
        _probs = [50, 68, 80, 85, 90, 95, 99]
        _zs    = [0.674, 1.000, 1.282, 1.440, 1.645, 1.960, 2.576]
        return float(np.interp(int(pct), _probs, _zs))

    def _chi2_scale(self, prob_pct: int) -> float:
        _probs = [50, 68, 80, 85, 90, 95, 99]
        _chi2s = [1.386, 2.296, 3.219, 3.794, 4.605, 5.991, 9.210]
        return math.sqrt(float(np.interp(int(prob_pct), _probs, _chi2s)))

    def _recompute_r90_from_cache(self) -> None:
        """Recompute r90_radius from cached sim data after uncertainty settings change."""
        if self._last_sim_data is None:
            return
        d            = self._last_sim_data
        fall_time    = d.get('fall_time', 0.0)
        surf_spd     = d.get('surf_spd',  0.0)
        horiz_dist   = d.get('r_horiz',   0.0)
        wind_sigma   = surf_spd * self.wind_uncertainty * max(fall_time, 0.0)
        thrust_sigma = self.thrust_uncertainty * horiz_dist
        combined     = math.hypot(wind_sigma, thrust_sigma)
        z_score      = self._prob_to_z(self.landing_prob)
        new_r90 = z_score * combined
        if new_r90 > 0:
            self.r90_radius             = new_r90
            self._last_sim_data['r90']  = new_r90

    # ── Settings window ───────────────────────────────────────────────────────

    def _open_settings_window(self) -> None:
        if self._settings_win is not None:
            try:
                if self._settings_win.winfo_exists():
                    self._settings_win.lift()
                    return
            except tk.TclError:
                pass

        win = tk.Toplevel(self)
        self._settings_win = win
        win.title("Uncertainty Settings")
        win.geometry("400x400")
        win.resizable(False, False)
        win.transient(self)

        frm = ttk.Frame(win, padding=14)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=0)

        ttk.Label(frm, text="Monte-Carlo / Uncertainty Parameters",
                  font=("Arial", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(frm, text="Wind speed uncertainty (%):").grid(
            row=1, column=0, sticky="w", pady=4)
        wind_var = tk.StringVar(value=f"{self.wind_uncertainty * 100:.1f}")
        ttk.Entry(frm, textvariable=wind_var, width=10).grid(
            row=1, column=1, sticky="e", pady=4)

        ttk.Label(frm, text="Motor thrust uncertainty (±%):").grid(
            row=2, column=0, sticky="w", pady=4)
        thrust_var = tk.StringVar(value=f"{self.thrust_uncertainty * 100:.1f}")
        ttk.Entry(frm, textvariable=thrust_var, width=10).grid(
            row=2, column=1, sticky="e", pady=4)

        ttk.Label(frm, text="Allowable Uncertainty (%):").grid(
            row=3, column=0, sticky="w", pady=4)
        allow_var = tk.StringVar(value=f"{self.allowable_uncertainty:.1f}")
        ttk.Entry(frm, textvariable=allow_var, width=10).grid(
            row=3, column=1, sticky="e", pady=4)

        # ── Confidence level: Spinbox (editable) + linked Scale slider ────────
        ttk.Label(frm, text="Landing circle probability (%):").grid(
            row=4, column=0, sticky="w", pady=(4, 0))
        prob_var = tk.StringVar(value=str(self.landing_prob))
        prob_spin = ttk.Spinbox(frm, textvariable=prob_var, from_=50, to=99,
                                increment=1, width=5)
        prob_spin.grid(row=4, column=1, sticky="e", pady=(4, 0))

        _sync = [False]

        def _slider_to_spin(val):
            if _sync[0]:
                return
            _sync[0] = True
            try:
                prob_var.set(str(int(float(val))))
            finally:
                _sync[0] = False

        def _spin_to_slider(*_):
            if _sync[0]:
                return
            _sync[0] = True
            try:
                v = max(50, min(99, int(float(prob_var.get()))))
                if prob_slider_var.get() != v:
                    prob_slider_var.set(v)
            except Exception:
                pass
            finally:
                _sync[0] = False

        prob_slider_var = tk.IntVar(value=self.landing_prob)
        ttk.Scale(frm, from_=50, to=99, orient="horizontal",
                  variable=prob_slider_var,
                  command=_slider_to_spin).grid(
            row=5, column=0, columnspan=2, sticky="ew", padx=2, pady=(2, 8))
        prob_var.trace_add("write", _spin_to_slider)

        ttk.Label(frm,
                  text="Uncertainty changes apply immediately from cached scatter.\n"
                       "Re-run simulation to refresh the point cloud.",
                  font=("Arial", 8), foreground="gray").grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(0, 10))

        btn_f = ttk.Frame(frm)
        btn_f.grid(row=7, column=0, columnspan=2, sticky="ew")
        btn_f.columnconfigure(0, weight=1)
        btn_f.columnconfigure(1, weight=1)

        def apply_and_close():
            try:
                w   = float(wind_var.get())   / 100.0
                th  = float(thrust_var.get()) / 100.0
                aw  = float(allow_var.get())
                p   = int(float(prob_var.get()))
                if w < 0 or th < 0:
                    raise ValueError("Uncertainty must be ≥ 0.")
                if aw < 0:
                    raise ValueError("Allowable Uncertainty must be ≥ 0.")
                if not 50 <= p <= 99:
                    raise ValueError("Probability must be between 50 and 99.")

                uncertainty_changed = (
                    w  != self.wind_uncertainty or
                    th != self.thrust_uncertainty or
                    aw != self.allowable_uncertainty)
                prob_changed = (p != self.landing_prob)

                self.wind_uncertainty      = w
                self.thrust_uncertainty    = th
                self.allowable_uncertainty = aw
                self.landing_prob          = p

                if prob_changed or uncertainty_changed:
                    self._release_lock_if_active(reason_label="⭘ Unlocked")

                # Smart Redraw: recompute from cached scatter, skip re-running MC
                if (prob_changed or uncertainty_changed) and self._mc_scatter is not None:
                    if uncertainty_changed:
                        self._recompute_r90_from_cache()
                    self._apply_mc_viz_results(self._mc_scatter, self._mc_wind_profiles)

                messagebox.showinfo(
                    "Settings Applied",
                    f"Wind uncertainty      : ±{w*100:.1f}%\n"
                    f"Thrust uncertainty    : ±{th*100:.1f}%\n"
                    f"Allowable Uncertainty : {aw:.1f}%\n"
                    f"Landing confidence   : {p}%",
                    parent=win)
                win.destroy()
                self._settings_win = None
            except ValueError as e:
                messagebox.showerror("Invalid input",
                                     f"Could not parse settings:\n{e}", parent=win)

        def cancel():
            win.destroy()
            self._settings_win = None

        ttk.Button(btn_f, text="Apply & Close",
                   command=apply_and_close).grid(
            row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(btn_f, text="Cancel",
                   command=cancel).grid(
            row=0, column=1, sticky="ew", padx=(3, 0))
        win.protocol("WM_DELETE_WINDOW", cancel)

    # ── Lock & Monitor ────────────────────────────────────────────────────────

    def _release_lock_if_active(self, reason_label: Optional[str] = None) -> None:
        if not hasattr(self, 'lock_monitor_var'):
            return
        was_locked = False
        try:
            was_locked = bool(self.lock_monitor_var.get())
        except Exception:
            pass
        if was_locked:
            try:
                self.lock_monitor_var.set(False)
                self._toggle_lock_monitor()
            except Exception:
                pass
        if reason_label is not None:
            try:
                self.monitor_status_label.config(
                    text=reason_label, foreground="gray", background="")
            except Exception:
                pass

    def _toggle_lock_monitor(self) -> None:
        if self.lock_monitor_var.get():
            self._capture_wind_baseline()
            try:
                self.main_action_btn.state(["disabled"])
            except Exception:
                pass
            self.monitor_status_label.config(
                text="🔒 LOCKED — monitoring wind", foreground="green")
            self._schedule_monitor_tick()
        else:
            try:
                self.main_action_btn.state(["!disabled"])
            except Exception:
                pass
            self.monitor_status_label.config(text="⭘ Unlocked", foreground="gray")
            if self._monitor_after_id is not None:
                try:
                    self.after_cancel(self._monitor_after_id)
                except Exception:
                    pass
                self._monitor_after_id = None

    def _auto_enable_monitor_mode(self) -> None:
        try:
            if not self.lock_monitor_var.get():
                self.lock_monitor_var.set(True)
                self._toggle_lock_monitor()
            else:
                self._capture_wind_baseline()
        except Exception:
            pass

    def _schedule_monitor_tick(self) -> None:
        self._monitor_after_id = self.after(2000, self._monitor_wind_tick)

    def _monitor_wind_tick(self) -> None:
        self._monitor_after_id = None
        if not self.lock_monitor_var.get():
            return
        if self._baseline_wind is None:
            self._schedule_monitor_tick()
            return
        try:
            cur_surf     = self._wind_avg_recent(window_sec=10.0)
            cur_surf_dir = float(self.surf_dir_var.get())
            cur_up       = float(self.up_spd_var.get())
            cur_up_dir   = float(self.up_dir_var.get())
        except (ValueError, AttributeError):
            self._schedule_monitor_tick()
            return

        b = self._baseline_wind
        SPD_TOL, DIR_TOL = 2.0, 15.0
        surf_spd_diff = abs(cur_surf     - b["surf_spd"])
        up_spd_diff   = abs(cur_up       - b["up_spd"])
        surf_dir_diff = self._angle_diff(cur_surf_dir, b["surf_dir"])
        up_dir_diff   = self._angle_diff(cur_up_dir,   b["up_dir"])

        exceeded = (
            surf_spd_diff > SPD_TOL or
            up_spd_diff   > SPD_TOL or
            surf_dir_diff > DIR_TOL or
            up_dir_diff   > DIR_TOL
        )
        if exceeded:
            messagebox.showwarning(
                "Wind Tolerance Exceeded",
                "Wind conditions have exceeded the tolerance threshold.\n"
                "Re-simulation will run automatically.\n\n"
                f"Surface speed:  baseline {b['surf_spd']:.1f} m/s → current {cur_surf:.1f} m/s "
                f"(Δ={surf_spd_diff:.2f}, tol ±{SPD_TOL:.1f})\n"
                f"Surface dir:    baseline {b['surf_dir']:.0f}° → current {cur_surf_dir:.0f}° "
                f"(Δ={surf_dir_diff:.0f}°, tol ±{DIR_TOL:.0f}°)\n"
                f"Upper speed:    baseline {b['up_spd']:.1f} m/s → current {cur_up:.1f} m/s "
                f"(Δ={up_spd_diff:.2f}, tol ±{SPD_TOL:.1f})\n"
                f"Upper dir:      baseline {b['up_dir']:.0f}° → current {cur_up_dir:.0f}° "
                f"(Δ={up_dir_diff:.0f}°, tol ±{DIR_TOL:.0f}°)"
            )
            try:
                self.main_action_btn.state(["!disabled"])
            except Exception:
                pass
            try:
                self.run_simulation()
            except Exception as e:
                messagebox.showerror("Re-simulation Error",
                                     f"Automatic re-simulation failed:\n{e}")
            try:
                self.main_action_btn.state(["disabled"])
            except Exception:
                pass
            self._capture_wind_baseline()

        self._schedule_monitor_tick()

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    def _start_phase1(self) -> None:
        if self._p1_running:
            messagebox.showinfo('Phase 1 Running', 'Phase 1 is already running.')
            return
        params = self._gather_sim_params()
        if params is None:
            return
        try:
            target_r = float(self.r_max_var.get())
            if target_r <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror('Input Error',
                                 'Please enter a positive number for Target radius.')
            return

        u_prof, v_prof = build_wind_profile(
            params['surf_spd'], params['surf_dir'], 3.0,
            params['up_spd'],   params['up_dir'],   100.0)
        params['wind_u_prof'] = u_prof
        params['wind_v_prof'] = v_prof

        mode = self.operation_mode_var.get()

        self._p1_running = True
        self._p1_stop_flag.clear()
        while not self._p1_queue.empty():
            try:
                self._p1_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self.main_action_btn.state(['disabled'])
        except Exception:
            pass

        self._show_p1_win(mode, target_r)

        self._p1_thread = threading.Thread(
            target=self._p1_worker,
            args=(params, target_r, mode),
            daemon=True)
        self._p1_thread.start()
        self.after(200, self._poll_p1_queue)

    def _p1_worker(self, base_params: dict, target_r: float, mode: str) -> None:
        q = self._p1_queue

        def progress_cb(msg: str, frac: float) -> None:
            q.put(('p1_prog', msg, frac))

        try:
            result = run_phase1(base_params, target_r, mode,
                                self._p1_stop_flag, progress_cb)
            q.put(('p1_done', result))
        except Exception as exc:
            q.put(('p1_error', f'Phase 1 worker error:\n{exc}'))

    def _stop_phase1(self) -> None:
        self._p1_stop_flag.set()
        try:
            if self._p1_win_msg is not None:
                self._p1_win_msg.set('Cancelling…')
        except Exception:
            pass

    def _poll_p1_queue(self) -> None:
        if not self._p1_running:
            return
        try:
            while True:
                msg  = self._p1_queue.get_nowait()
                kind = msg[0]
                if kind == 'p1_prog':
                    _, text, frac = msg
                    try:
                        if self._p1_win_msg is not None:
                            self._p1_win_msg.set(text)
                        if self._p1_win_bar is not None:
                            self._p1_win_bar['value'] = frac * 100
                    except Exception:
                        pass
                elif kind == 'p1_done':
                    self._finish_phase1(msg[1]); return
                elif kind == 'p1_cancel':
                    self._finish_phase1(None, cancelled=True); return
                elif kind == 'p1_error':
                    self._finish_phase1(None, error=msg[1]); return
        except queue.Empty:
            pass
        self.after(200, self._poll_p1_queue)

    def _finish_phase1(self, result, cancelled=False, error=None) -> None:
        self._p1_running = False
        self._close_p1_win()
        try:
            self.main_action_btn.state(['!disabled'])
        except Exception:
            pass

        if cancelled:
            return
        if error:
            messagebox.showerror('Phase 1 Error', error)
            return

        self._phase1_result = result

        try:
            self.elev_spin.delete(0, tk.END)
            self.elev_spin.insert(0, f'{result.best_elev:.1f}')
            self.azi_spin.delete(0, tk.END)
            self.azi_spin.insert(0, f'{result.best_azi:.1f}')
        except Exception:
            pass

        try:
            self._render_current_params()
        except Exception:
            pass

        try:
            self.p1_result_label.config(
                text=(f'Apogee {result.apogee_m:.0f} m  '
                      f'μ_max {result.mu_max:.1f} m/s  '
                      f'σ_max {result.sigma_max:.1f} m/s'),
                foreground='#004400')
        except Exception:
            pass

        self._start_phase2()
        self._show_phase1_complete_dialog(result)

    def _show_p1_win(self, mode: str, target_r: float) -> None:
        if self._p1_win is not None:
            try:
                self._p1_win.destroy()
            except Exception:
                pass
        win = tk.Toplevel(self)
        self._p1_win = win
        win.title(f"Phase 1 — {mode}")
        win.geometry("440x165")
        win.resizable(False, False)
        win.grab_set()
        win.protocol("WM_DELETE_WINDOW", self._stop_phase1)

        frm = ttk.Frame(win, padding=12)
        frm.pack(fill='both', expand=True)

        ttk.Label(frm,
                  text=f"Mode: {mode}  |  Target radius: {target_r:.1f} m",
                  font=("Arial", 9)).pack(anchor='w')

        self._p1_win_msg = tk.StringVar(value="Preparing…")
        ttk.Label(frm, textvariable=self._p1_win_msg,
                  font=("Arial", 9), wraplength=400).pack(anchor='w', pady=(6, 3))

        self._p1_win_bar = ttk.Progressbar(frm, mode='determinate', maximum=100)
        self._p1_win_bar.pack(fill='x', pady=(0, 8))

        ttk.Button(frm, text="Cancel",
                   command=self._stop_phase1).pack(anchor='e')

    def _close_p1_win(self) -> None:
        try:
            if self._p1_win is not None:
                self._p1_win.destroy()
        except Exception:
            pass
        self._p1_win     = None
        self._p1_win_bar = None
        self._p1_win_msg = None

    def _show_phase1_complete_dialog(self, result) -> None:
        dlg = tk.Toplevel(self)
        dlg.title('Phase 1 Complete')
        dlg.resizable(False, False)
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=16)
        frm.pack(fill='both', expand=True)

        mode  = getattr(result, 'mode',       '')
        score = getattr(result, 'best_score', None)

        # ── Mode-specific emphasis banner ──────────────────────────────────────
        if mode == 'Altitude Competition':
            emph_label = 'Apogee'
            emph_value = f'{result.apogee_m:.0f} m'
            banner_bg  = '#1a3a8f'
            fg_sub     = '#aabbff'
        elif mode == 'Precision Landing':
            r_land    = abs(score) if score is not None else 0.0
            r_max     = getattr(result, 'target_radius_m', 0.0)
            score_val = r_max - r_land
            emph_label = f'Score  (r_max − r)  [{r_max:.0f} − {r_land:.1f}]'
            emph_value = f'{score_val:+.1f} m'
            banner_bg  = '#7a3000'
            fg_sub     = '#ffddaa'
        elif mode == 'Winged Hover':
            emph_label = 'Hang Time'
            emph_value = f'{score:.2f} s' if score is not None else '-- s'
            banner_bg  = '#004d00'
            fg_sub     = '#aaffaa'
        else:
            emph_label = 'Apogee'
            emph_value = f'{result.apogee_m:.0f} m'
            banner_bg  = '#444444'
            fg_sub     = '#cccccc'

        banner_f = tk.Frame(frm, bg=banner_bg, padx=12, pady=8)
        banner_f.pack(fill='x', pady=(0, 10))
        tk.Label(banner_f, text=emph_label,
                 font=('Arial', 10), fg=fg_sub, bg=banner_bg).pack(anchor='w')
        tk.Label(banner_f, text=emph_value,
                 font=('Arial', 28, 'bold'), fg='white', bg=banner_bg).pack(anchor='w')

        details = (
            f'Optimal launch angle:  elev = {result.best_elev:.1f}°'
            f'   azi = {result.best_azi:.1f}°\n'
            f'Apogee: {result.apogee_m:.0f} m\n\n'
            f'── GO/NO-GO Limits ─────────────────\n'
            f'  μ_max  (max mean wind speed):   {result.mu_max:.2f} m/s\n'
            f'  σ_max  (max wind std dev):       {result.sigma_max:.2f} m/s\n\n'
            f'Error ellipse ({self.landing_prob}%):  '
            f'a = {result.ellipse_a:.1f} m   b = {result.ellipse_b:.1f} m\n\n'
            f'Phase 2 monitoring has started.'
        )
        tk.Label(frm, text=details, justify='left',
                 font=('Consolas', 9)).pack(anchor='w')

        ttk.Button(frm, text='OK', command=dlg.destroy).pack(anchor='e', pady=(10, 0))
        dlg.bind('<Return>', lambda _: dlg.destroy())
        self.wait_window(dlg)

    # ── Phase 2 GO/NO-GO ──────────────────────────────────────────────────────

    def _start_phase2(self) -> None:
        if self._p2_after_id is not None:
            try:
                self.after_cancel(self._p2_after_id)
            except Exception:
                pass
        self._p2_after_id = self.after(1000, self._phase2_tick)

    def _phase2_tick(self) -> None:
        self._p2_after_id = None
        ph1 = self._phase1_result
        if ph1 is None:
            return

        status = p2_evaluate(ph1, self.wind_tracker)
        if status is None:
            self._p2_after_id = self.after(1000, self._phase2_tick)
            return

        self._update_go_nogo_ui(
            status.go, status.mu_cur, status.sigma_cur,
            status.cond_a, status.cond_b, status.cond_c, ph1)

        self._p2_ellipse = {
            'cx': status.ellipse['cx'], 'cy': status.ellipse['cy'],
            'a':  status.ellipse['a'],  'b':  status.ellipse['b'],
            'angle_rad': status.ellipse['angle_rad'],
            'go': status.go,
        }
        try:
            self.draw_map_elements()
        except Exception:
            pass

        self._p2_after_id = self.after(1000, self._phase2_tick)

    def _update_go_nogo_ui(self, go, mu_cur, sigma_cur,
                           cond_a, cond_b, cond_c, ph1) -> None:
        color   = '#007700' if go else '#cc0000'
        verdict = '●  GO  — Ready for Launch' if go else '●  NO-GO  — Hold'
        try:
            self.go_nogo_label.config(
                text=verdict, background=color, foreground='white')
        except Exception:
            pass

        def _m(ok): return '✓' if ok else '✗'
        detail = (
            f'{_m(cond_a)} A  μ = {mu_cur:.2f}  /  μ_max = {ph1.mu_max:.2f} m/s\n'
            f'{_m(cond_b)} B  σ = {sigma_cur:.2f}  /  σ_max = {ph1.sigma_max:.2f} m/s\n'
            f'{_m(cond_c)} C  Ellipse ⊂ circle  (r = {ph1.target_radius_m:.0f} m)'
        )
        try:
            self.go_nogo_detail_label.config(text=detail)
        except Exception:
            pass

    # ── Config load / save ────────────────────────────────────────────────────

    def _collect_airframe_dict(self) -> dict:
        return {
            "mass":           float(self.mass_entry.get()),
            "cg":             float(self.cg_entry.get()),
            "length":         float(self.len_entry.get()),
            "radius":         float(self.radius_entry.get()),
            "nose_length":    float(self.nose_len_entry.get()),
            "fin_root":       float(self.fin_root_entry.get()),
            "fin_tip":        float(self.fin_tip_entry.get()),
            "fin_span":       float(self.fin_span_entry.get()),
            "fin_pos":        float(self.fin_pos_entry.get()),
            "motor_pos":      float(self.motor_pos_entry.get()),
            "motor_dry_mass": float(self.motor_dry_mass_entry.get()),
            "backfire_delay": float(self.backfire_delay_entry.get()),
        }

    def _collect_parachute_dict(self) -> dict:
        return {
            "cd":   float(self.cd_entry.get()),
            "area": float(self.area_entry.get()),
            "lag":  float(self.lag_entry.get()),
        }

    def _apply_airframe_dict(self, af: dict) -> None:
        def _set(entry, val):
            entry.delete(0, tk.END)
            if val is not None:
                entry.insert(0, str(val))
        _set(self.mass_entry,           af.get("mass",           "0.0872"))
        _set(self.cg_entry,              af.get("cg",             "0.21"))
        _set(self.len_entry,             af.get("length",         "0.383"))
        _set(self.radius_entry,          af.get("radius",         "0.015"))
        _set(self.nose_len_entry,        af.get("nose_length",    "0.08"))
        _set(self.fin_root_entry,        af.get("fin_root",       "0.04"))
        _set(self.fin_tip_entry,         af.get("fin_tip",        "0.02"))
        _set(self.fin_span_entry,        af.get("fin_span",       "0.03"))
        _set(self.fin_pos_entry,         af.get("fin_pos",        "0.35"))
        _set(self.motor_pos_entry,       af.get("motor_pos",      "0.38"))
        _set(self.motor_dry_mass_entry,  af.get("motor_dry_mass", "0.015"))
        if "backfire_delay" in af:
            _set(self.backfire_delay_entry, af["backfire_delay"])

    def _apply_parachute_dict(self, pa: dict) -> None:
        self.cd_entry.delete(0,   tk.END); self.cd_entry.insert(0,   str(pa.get("cd",   "")))
        self.area_entry.delete(0, tk.END); self.area_entry.insert(0, str(pa.get("area", "")))
        self.lag_entry.delete(0,  tk.END); self.lag_entry.insert(0,  str(pa.get("lag",  "")))

    def save_config(self) -> None:
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Save Rocket Config")
        if not filepath:
            return
        try:
            data = {
                "version":   2,
                "airframe":  self._collect_airframe_dict(),
                "parachute": self._collect_parachute_dict(),
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            base = os.path.basename(filepath)
            self.af_name_label.config(text=f"Airframe: {base}")
            self.para_name_label.config(text=f"Parachute: {base}")
            try:
                self.config_file_label.config(
                    text=f"Config file: {base}", foreground="#555555")
            except Exception:
                pass
            messagebox.showinfo("Saved", f"Airframe + parachute config saved.\n{base}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{e}")

    def load_config(self) -> None:
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json")],
            title="Load Rocket Config")
        if not filepath:
            return
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            base = os.path.basename(filepath)
            af   = data.get("airframe")
            pa   = data.get("parachute")
            applied = []
            if af is None and pa is None:
                if "mass" in data or "fin_root" in data:
                    self._apply_airframe_dict(data)
                    applied.append("Airframe")
                    self.af_name_label.config(text=f"Airframe: {base}")
                if "cd" in data or "area" in data or "lag" in data:
                    self._apply_parachute_dict(data)
                    applied.append("Parachute")
                    self.para_name_label.config(text=f"Parachute: {base}")
            else:
                if af is not None:
                    self._apply_airframe_dict(af)
                    applied.append("Airframe")
                    self.af_name_label.config(text=f"Airframe: {base}")
                if pa is not None:
                    self._apply_parachute_dict(pa)
                    applied.append("Parachute")
                    self.para_name_label.config(text=f"Parachute: {base}")
            if not applied:
                raise ValueError("JSON contains neither airframe nor parachute data.")
            try:
                self.config_file_label.config(
                    text=f"Config file: {base}", foreground="#555555")
            except Exception:
                pass
            messagebox.showinfo("Loaded", f"{' + '.join(applied)} config loaded.\n{base}")
        except Exception as e:
            messagebox.showerror("Error", f"Load failed:\n{e}")

    def save_af_settings(self):   self.save_config()
    def load_af_settings(self):   self.load_config()
    def save_para_settings(self): self.save_config()
    def load_para_settings(self): self.load_config()

    # ── Motor loading ─────────────────────────────────────────────────────────

    def open_thrustcurve_web(self) -> None:
        try:
            webbrowser.open("https://www.thrustcurve.org/motors/search.html")
            messagebox.showinfo(
                "Browser Opened",
                "Opened ThrustCurve.org search page.\n"
                "Download a motor CSV (RockSim format) and load it with [Load CSV].")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open browser.\n{e}")

    def load_local_motor(self) -> None:
        filepath = filedialog.askopenfilename(
            title="Select Motor CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not filepath:
            return
        try:
            motor_name        = os.path.basename(filepath).replace('.csv', '')
            time_thrust_points = []
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().replace('"', '').split(',')
                    if len(parts) >= 2:
                        try:
                            t = float(parts[0])
                            T = float(parts[1])
                            time_thrust_points.append([t, T])
                        except ValueError:
                            if parts[0].strip().lower() in ["motor:", "motor"]:
                                motor_name = parts[1].strip()

            if not time_thrust_points:
                raise ValueError(
                    "No valid numeric data found. Please verify it is a RockSim-format CSV.")
            if time_thrust_points[0][0] != 0.0:
                time_thrust_points.insert(0, [0.0, time_thrust_points[0][1]])

            burn_time     = time_thrust_points[-1][0]
            thrusts       = [p[1] for p in time_thrust_points]
            max_thrust    = max(thrusts) if thrusts else 0.0
            total_impulse = sum((T0 + T1) * 0.5 * (t1 - t0)
                                for (t0, T0), (t1, T1)
                                in zip(time_thrust_points, time_thrust_points[1:]))
            avg_thrust = (total_impulse / burn_time) if burn_time > 0 else 0.0

            self.selected_motor_file = filepath
            self.thrust_data         = time_thrust_points
            self.selected_motor_name = motor_name
            self.motor_burn_time     = burn_time
            self.motor_avg_thrust    = avg_thrust
            self.motor_max_thrust    = max_thrust
            self.motor_ui_label.config(text=f"Engine: {motor_name}")
            if hasattr(self, 'motor_specs_label'):
                self.motor_specs_label.config(
                    text=(f"Burn: {burn_time:.2f} s   "
                          f"Avg: {avg_thrust:.1f} N   "
                          f"Max: {max_thrust:.1f} N"))

            messagebox.showinfo(
                "Motor Loaded",
                "Motor CSV loaded successfully.\n"
                f"  Name:       {motor_name}\n"
                f"  Burn time:  {burn_time:.3f} s\n"
                f"  Avg thrust: {avg_thrust:.1f} N\n"
                f"  Max thrust: {max_thrust:.1f} N")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load motor file:\n{e}")

    # ── Parameter edit callbacks ──────────────────────────────────────────────

    def on_parameter_edit_af(self, event=None) -> None:
        if self.af_name_label.cget("text") != "Airframe: (none selected)":
            self.af_name_label.config(text="Airframe: (none selected)")
            try:
                self.config_file_label.config(
                    text="Config file: (modified)", foreground="#aa6600")
            except Exception:
                pass
        self._release_lock_if_active(reason_label="⭘ Unlocked (param changed)")

    def on_parameter_edit_para(self, event=None) -> None:
        if self.para_name_label.cget("text") != "Parachute: (none selected)":
            self.para_name_label.config(text="Parachute: (none selected)")
            try:
                self.config_file_label.config(
                    text="Config file: (modified)", foreground="#aa6600")
            except Exception:
                pass
        self._release_lock_if_active(reason_label="⭘ Unlocked (param changed)")

    # ── Scrollable params canvas helpers ──────────────────────────────────────

    def _on_params_wheel(self, event, delta_override=None) -> str:
        canvas = getattr(self, '_params_canvas', None)
        if canvas is None:
            return "break"
        d = delta_override if delta_override is not None else getattr(event, 'delta', 0)
        try:
            if d == 0:
                return "break"
            # Divide by 60 so a standard Windows notch (±120) → ±2 units of 20 px each.
            # Clamp to ±1 for very small trackpad nudges so the scroll still registers.
            step = int(-d / 60)
            if step == 0:
                step = 1 if d < 0 else -1
            canvas.yview_scroll(step, "units")
        except Exception:
            pass
        return "break"

    def _bind_params_wheel_recursive(self, widget) -> None:
        try:
            widget.bind("<MouseWheel>", self._on_params_wheel, add="+")
            widget.bind("<Button-4>",
                        lambda e: self._on_params_wheel(e, delta_override=+120), add="+")
            widget.bind("<Button-5>",
                        lambda e: self._on_params_wheel(e, delta_override=-120), add="+")
        except Exception:
            pass
        try:
            for child in widget.winfo_children():
                self._bind_params_wheel_recursive(child)
        except Exception:
            pass

    # ── Map centre ────────────────────────────────────────────────────────────

    def update_map_center(self) -> None:
        try:
            self.launch_lat = float(self.lat_entry.get())
            self.launch_lon = float(self.lon_entry.get())
            self.map_view.set_position(self.launch_lat, self.launch_lon)
            self._clear_previous_landing()
            self.update_plots()
            self.fit_map_bounds()
        except ValueError:
            pass

    def _clear_previous_landing(self) -> None:
        self.land_lat       = self.launch_lat
        self.land_lon       = self.launch_lon
        self.r90_radius     = 0.0
        self._has_sim_result = False
        self._last_sim_data  = None

    # ── Left panel (data section) ─────────────────────────────────────────────

    def create_data_section(self) -> None:
        outer = ttk.Frame(self)
        outer.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        self._params_canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0,
                                        yscrollincrement=20)
        self._params_canvas.grid(row=0, column=0, sticky="nsew")
        vbar = ttk.Scrollbar(outer, orient="vertical",
                             command=self._params_canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")
        self._params_canvas.configure(yscrollcommand=vbar.set)

        frame = ttk.Frame(self._params_canvas, padding=(4, 4))
        self._params_inner  = frame
        self._params_window = self._params_canvas.create_window(
            (0, 0), window=frame, anchor="nw")
        frame.columnconfigure(0, weight=1)

        def _on_inner_configure(event):
            bbox = self._params_canvas.bbox("all")
            self._params_canvas.configure(
                scrollregion=(0, 0, bbox[2], bbox[3]) if bbox else (0, 0, 0, 0))

        def _on_canvas_configure(event):
            self._params_canvas.itemconfigure(
                self._params_window, width=event.width)

        frame.bind("<Configure>", _on_inner_configure)
        self._params_canvas.bind("<Configure>", _on_canvas_configure)
        self._params_canvas.bind("<MouseWheel>", self._on_params_wheel)
        self._params_canvas.bind(
            "<Button-4>", lambda e: self._on_params_wheel(e, delta_override=+120))
        self._params_canvas.bind(
            "<Button-5>", lambda e: self._on_params_wheel(e, delta_override=-120))
        frame.bind("<MouseWheel>", self._on_params_wheel)
        frame.bind(
            "<Button-4>", lambda e: self._on_params_wheel(e, delta_override=+120))
        frame.bind(
            "<Button-5>", lambda e: self._on_params_wheel(e, delta_override=-120))
        self._params_canvas.bind("<Enter>", lambda e: self._params_canvas.focus_set())

        def param_row(parent, label_text, row, default=""):
            ttk.Label(parent, text=label_text, font=("Arial", 8)).grid(
                row=row, column=0, sticky="w", padx=(4, 2), pady=1)
            e = ttk.Entry(parent, width=7, font=("Arial", 8))
            e.grid(row=row, column=1, sticky="e", padx=(2, 4), pady=1)
            if default:
                e.insert(0, default)
            return e

        def _unlock_on_edit(event=None):
            self._release_lock_if_active(reason_label="⭘ Unlocked (param changed)")

        # ── Engine ────────────────────────────────────────────────────────────
        engine_info_f = ttk.Frame(frame)
        engine_info_f.grid(row=0, column=0, sticky="ew", pady=(0, 1))
        engine_info_f.columnconfigure(0, weight=1)
        self.motor_ui_label = ttk.Label(
            engine_info_f, text=f"Engine: {self.selected_motor_name}",
            font=("Arial", 9, "bold"), foreground="#B22222")
        self.motor_ui_label.grid(row=0, column=0, sticky="w")
        self.motor_specs_label = ttk.Label(
            engine_info_f, text="Burn: — s   Avg: — N   Max: — N",
            font=("Arial", 8), foreground="#555555")
        self.motor_specs_label.grid(row=1, column=0, sticky="w")

        mbf = ttk.Frame(frame)
        mbf.grid(row=1, column=0, sticky="ew", pady=(0, 3))
        mbf.columnconfigure(0, weight=1); mbf.columnconfigure(1, weight=1)
        ttk.Button(mbf, text="[ThrustCurve Web]",
                   command=self.open_thrustcurve_web).grid(
            row=0, column=0, sticky="ew", padx=(0, 1))
        ttk.Button(mbf, text="[Load CSV]",
                   command=self.load_local_motor).grid(
            row=0, column=1, sticky="ew", padx=(1, 0))
        ttk.Label(mbf, text="Backfire Delay (s):", font=("Arial", 8)).grid(
            row=1, column=0, sticky="w", padx=(4, 2), pady=(3, 1))
        self.backfire_delay_entry = ttk.Entry(mbf, width=7, font=("Arial", 8))
        self.backfire_delay_entry.insert(0, "0.0")
        self.backfire_delay_entry.grid(
            row=1, column=1, sticky="e", padx=(2, 4), pady=(3, 1))
        self.backfire_delay_entry.bind("<KeyRelease>", self.on_parameter_edit_af)

        ttk.Separator(frame, orient="horizontal").grid(
            row=2, column=0, sticky="ew", pady=3)

        # ── Airframe ──────────────────────────────────────────────────────────
        self.af_name_label = ttk.Label(frame, text="Airframe: (none selected)",
                                       font=("Arial", 8, "bold"))
        self.af_name_label.grid(row=3, column=0, sticky="w")
        self.af_name_label.grid_remove()

        af_lf = ttk.LabelFrame(frame, text="Airframe", padding=(2, 2))
        af_lf.grid(row=4, column=0, sticky="ew", pady=(1, 2))
        af_lf.columnconfigure(0, weight=1); af_lf.columnconfigure(1, weight=0)

        self.mass_entry   = param_row(af_lf, "Dry Mass (kg)",    0)
        self.cg_entry     = param_row(af_lf, "CG from Nose (m)", 1)
        self.len_entry    = param_row(af_lf, "Length (m)",        2)
        self.radius_entry = param_row(af_lf, "Radius (m)",        3)

        aero_lf = ttk.LabelFrame(frame, text="Aero & Motor", padding=(2, 2))
        aero_lf.grid(row=5, column=0, sticky="ew", pady=(1, 2))
        aero_lf.columnconfigure(0, weight=1); aero_lf.columnconfigure(1, weight=0)

        self.nose_len_entry       = param_row(aero_lf, "Nose Length (m)",        0)
        self.fin_root_entry       = param_row(aero_lf, "Fin Root (m)",           1)
        self.fin_tip_entry        = param_row(aero_lf, "Fin Tip (m)",            2)
        self.fin_span_entry       = param_row(aero_lf, "Fin Span (m)",           3)
        self.fin_pos_entry        = param_row(aero_lf, "Fin Pos fr Nose (m)",    4)
        self.motor_pos_entry      = param_row(aero_lf, "Motor Pos fr Nose (m)", 5)
        self.motor_dry_mass_entry = param_row(aero_lf, "Motor Dry Mass (kg)",   6)

        af_entries = [self.mass_entry, self.cg_entry, self.len_entry, self.radius_entry,
                      self.nose_len_entry, self.fin_root_entry, self.fin_tip_entry,
                      self.fin_span_entry, self.fin_pos_entry, self.motor_pos_entry,
                      self.motor_dry_mass_entry]
        for e in af_entries:
            e.bind("<KeyRelease>", self.on_parameter_edit_af)

        ttk.Separator(frame, orient="horizontal").grid(
            row=6, column=0, sticky="ew", pady=3)

        # ── Parachute ─────────────────────────────────────────────────────────
        self.para_name_label = ttk.Label(frame, text="Parachute: (none selected)",
                                         font=("Arial", 8, "bold"))
        self.para_name_label.grid(row=7, column=0, sticky="w")
        self.para_name_label.grid_remove()

        para_lf = ttk.LabelFrame(frame, text="Parachute", padding=(2, 2))
        para_lf.grid(row=8, column=0, sticky="ew", pady=(1, 2))
        para_lf.columnconfigure(0, weight=1); para_lf.columnconfigure(1, weight=0)

        self.cd_entry   = param_row(para_lf, "Cd",        0)
        self.area_entry = param_row(para_lf, "Area (m²)", 1)
        self.lag_entry  = param_row(para_lf, "Lag (s)",   2)

        self.cd_entry.bind("<KeyRelease>",   self.on_parameter_edit_para)
        self.area_entry.bind("<KeyRelease>", self.on_parameter_edit_para)
        self.lag_entry.bind("<KeyRelease>",  self.on_parameter_edit_para)

        # ── Config file label + Load/Save buttons ─────────────────────────────
        config_area = ttk.Frame(frame)
        config_area.grid(row=9, column=0, sticky="ew", pady=(0, 2))
        config_area.columnconfigure(0, weight=1)

        self.config_file_label = ttk.Label(
            config_area, text="Config file: (none loaded)",
            font=("Arial", 8), foreground="#666666")
        self.config_file_label.grid(row=0, column=0, sticky="w", padx=2, pady=(0, 2))

        para_btn_f = ttk.Frame(config_area)
        para_btn_f.grid(row=1, column=0, sticky="ew")
        para_btn_f.columnconfigure(0, weight=1); para_btn_f.columnconfigure(1, weight=1)
        ttk.Button(para_btn_f, text="Load Rocket Config",
                   command=self.load_config).grid(
            row=0, column=0, sticky="ew", padx=(0, 1))
        ttk.Button(para_btn_f, text="Save Rocket Config",
                   command=self.save_config).grid(
            row=0, column=1, sticky="ew", padx=(1, 0))

        ttk.Separator(frame, orient="horizontal").grid(
            row=10, column=0, sticky="ew", pady=3)

        # ── Launcher ──────────────────────────────────────────────────────────
        ttk.Label(frame, text="Launcher", font=("Arial", 9, "bold")).grid(
            row=11, column=0, sticky="w", pady=(0, 1))

        launch_lf = ttk.LabelFrame(frame, text="Position & Rail", padding=(2, 2))
        launch_lf.grid(row=12, column=0, sticky="ew", pady=(1, 2))
        launch_lf.columnconfigure(0, weight=1); launch_lf.columnconfigure(1, weight=0)

        ttk.Label(launch_lf, text="Lat:", font=("Arial", 8)).grid(
            row=0, column=0, sticky="w", padx=(4, 2), pady=1)
        self.lat_entry = ttk.Entry(launch_lf, width=11, font=("Arial", 8))
        self.lat_entry.insert(0, str(self.launch_lat))
        self.lat_entry.grid(row=0, column=1, sticky="e", padx=(2, 4), pady=1)

        ttk.Label(launch_lf, text="Lon:", font=("Arial", 8)).grid(
            row=1, column=0, sticky="w", padx=(4, 2), pady=1)
        self.lon_entry = ttk.Entry(launch_lf, width=11, font=("Arial", 8))
        self.lon_entry.insert(0, str(self.launch_lon))
        self.lon_entry.grid(row=1, column=1, sticky="e", padx=(2, 4), pady=1)

        ttk.Label(launch_lf, text="Rail (m):", font=("Arial", 8)).grid(
            row=2, column=0, sticky="w", padx=(4, 2), pady=1)
        self.rail_entry = ttk.Entry(launch_lf, width=7, font=("Arial", 8))
        self.rail_entry.insert(0, "1.0")
        self.rail_entry.grid(row=2, column=1, sticky="e", padx=(2, 4), pady=1)

        ttk.Label(launch_lf, text="Elevation:", font=("Arial", 8)).grid(
            row=3, column=0, sticky="w", padx=(4, 2), pady=1)
        self.elev_spin = ttk.Spinbox(launch_lf, from_=0, to=90,
                                     width=6, font=("Arial", 8))
        self.elev_spin.set("85")
        self.elev_spin.grid(row=3, column=1, sticky="e", padx=(2, 4), pady=1)

        ttk.Label(launch_lf, text="Azimuth:", font=("Arial", 8)).grid(
            row=4, column=0, sticky="w", padx=(4, 2), pady=1)
        self.azi_spin = ttk.Spinbox(launch_lf, from_=0, to=360,
                                    width=6, font=("Arial", 8))
        self.azi_spin.set("0")
        self.azi_spin.grid(row=4, column=1, sticky="e", padx=(2, 4), pady=1)

        for _w in (self.lat_entry, self.lon_entry, self.rail_entry,
                   self.elev_spin, self.azi_spin):
            _w.bind("<KeyRelease>", _unlock_on_edit)

        loc_btn_f = ttk.Frame(frame)
        loc_btn_f.grid(row=13, column=0, sticky="ew", pady=(0, 2))
        loc_btn_f.columnconfigure(0, weight=1); loc_btn_f.columnconfigure(1, weight=1)
        ttk.Button(loc_btn_f, text="Get Location (IP)",
                   command=lambda: self.get_current_location(manual=True)).grid(
            row=0, column=0, sticky="ew", padx=(0, 1))
        ttk.Button(loc_btn_f, text="Update Map",
                   command=self.update_map_center).grid(
            row=0, column=1, sticky="ew", padx=(1, 0))

        ttk.Separator(frame, orient="horizontal").grid(
            row=14, column=0, sticky="ew", pady=3)

        # ── Operation Mode ────────────────────────────────────────────────────
        mode_lf = ttk.LabelFrame(frame, text="Operation Mode", padding=(4, 4))
        mode_lf.grid(row=15, column=0, sticky="ew", pady=(1, 2))
        mode_lf.columnconfigure(0, weight=1); mode_lf.columnconfigure(1, weight=0)

        self.mode_combo = ttk.Combobox(
            mode_lf, textvariable=self.operation_mode_var,
            values=list(self.OPERATION_MODES), state="readonly", font=("Arial", 9))
        self.mode_combo.grid(row=0, column=0, columnspan=2,
                             sticky="ew", padx=2, pady=(0, 4))
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        self.rmax_label = ttk.Label(mode_lf, text="Target radius r_max (m):",
                                    font=("Arial", 8))
        self.rmax_label.grid(row=1, column=0, sticky="w", padx=(4, 2), pady=1)
        self.rmax_entry = ttk.Entry(mode_lf, textvariable=self.r_max_var,
                                    width=8, font=("Arial", 8))
        self.rmax_entry.grid(row=1, column=1, sticky="e", padx=(2, 4), pady=1)
        self.rmax_label.grid_remove()
        self.rmax_entry.grid_remove()

        # ── Main Action Button ─────────────────────────────────────────────────
        self.main_action_btn = ttk.Button(
            frame, text='🚀 Run Single Simulation',
            command=self._render_current_params)
        self.main_action_btn.grid(row=16, column=0, sticky="ew", ipady=4, pady=(2, 4))

        res_f = ttk.Frame(frame)
        res_f.grid(row=17, column=0, sticky="ew")
        res_f.columnconfigure(0, weight=1); res_f.columnconfigure(1, weight=1)
        self.apogee_label   = ttk.Label(res_f, text="Apogee: -- m",
                                        font=("Arial", 9, "bold"))
        self.apogee_label.grid(row=0, column=0, sticky="w", padx=4)
        self.velocity_label = ttk.Label(res_f, text="Impact: -- m/s",
                                        font=("Arial", 9, "bold"))
        self.velocity_label.grid(row=0, column=1, sticky="e", padx=4)

        ttk.Separator(frame, orient="horizontal").grid(
            row=18, column=0, sticky="ew", pady=3)

        # ── Lock & Monitor + Settings row ──────────────────────────────────────
        lock_f = ttk.Frame(frame)
        lock_f.grid(row=19, column=0, sticky="ew", pady=(0, 2))
        lock_f.columnconfigure(0, weight=1); lock_f.columnconfigure(1, weight=0)

        self.lock_monitor_var = tk.BooleanVar(value=False)
        self.lock_monitor_check = ttk.Checkbutton(
            lock_f, text="🔒 Lock & Monitor",
            variable=self.lock_monitor_var,
            command=self._toggle_lock_monitor)
        self.lock_monitor_check.grid(row=0, column=0, sticky="w")
        ttk.Button(lock_f, text="⚙ Settings",
                   command=self._open_settings_window).grid(
            row=0, column=1, sticky="e")

        self.monitor_status_label = tk.Label(
            lock_f, text="⭘ Unlocked",
            foreground="gray", font=("Arial", 8), anchor="w")
        self.monitor_status_label.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        ttk.Separator(frame, orient="horizontal").grid(
            row=20, column=0, sticky="ew", pady=3)

        # ── Phase 1 Pre-Calculation ────────────────────────────────────────────
        p1_lf = ttk.LabelFrame(frame, text="Phase 1: Pre-Calculation", padding=(4, 4))
        p1_lf.grid(row=23, column=0, sticky="ew", pady=(1, 2))
        p1_lf.columnconfigure(0, weight=1)

        p1_btn_f = ttk.Frame(p1_lf)
        p1_btn_f.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        p1_btn_f.columnconfigure(0, weight=1)
        ttk.Button(p1_btn_f, text="■ Stop",
                   command=self._stop_phase1).grid(row=0, column=0, sticky="e")

        self.p1_result_label = ttk.Label(
            p1_lf, text="", foreground="#004400", font=("Arial", 8))
        self.p1_result_label.grid(row=1, column=0, sticky="w", padx=4, pady=(0, 2))

        # ── Phase 2 GO/NO-GO ───────────────────────────────────────────────────
        go_lf = ttk.LabelFrame(frame, text="Phase 2: GO / NO-GO", padding=(4, 4))
        go_lf.grid(row=24, column=0, sticky="ew", pady=(1, 4))
        go_lf.columnconfigure(0, weight=1)

        self.go_nogo_label = tk.Label(
            go_lf, text="●  STANDBY",
            font=("Arial", 13, "bold"),
            foreground="white", background="#777777",
            anchor="center", pady=5)
        self.go_nogo_label.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self.go_nogo_detail_label = ttk.Label(
            go_lf,
            text="Run Phase 1 to start GO/NO-GO monitoring",
            font=("Arial", 8), foreground="gray", justify="left")
        self.go_nogo_detail_label.grid(row=1, column=0, sticky="w")

        self._bind_params_wheel_recursive(self._params_inner)
