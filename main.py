import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkintermapview
import requests
import random
from collections import deque
import json
import webbrowser
import threading
import queue
from rocketpy import Environment, SolidMotor, Rocket, Flight

# ─── Wind Profile Helper ────────────────────────────────────────────────────
class WindProfileBuilder:
    """Builds a vertically resolved wind profile via the power-law (Hellmann) model.

    Two measurement points (surface anemometer + upper-level reading) are
    fitted to  v(z) = v_ref × (z / z_ref)^α  where α is derived from the
    two known (speed, height) pairs.
    """

    @staticmethod
    def hellmann_alpha(v_lo, z_lo, v_hi, z_hi):
        try:
            if v_lo < 1e-6 or z_lo <= 0 or z_hi <= z_lo:
                return 0.14
            return math.log(max(v_hi, 1e-9) / v_lo) / math.log(z_hi / z_lo)
        except (ValueError, ZeroDivisionError):
            return 0.14

    @staticmethod
    def build(v_surf, dir_surf_deg, z_surf, v_upper, dir_upper_deg, z_upper):
        """Return (u_prof, v_prof) tuples for RocketPy custom_atmosphere.

        Altitude 0 is forced to zero wind (below the anemometer).
        """
        alpha = WindProfileBuilder.hellmann_alpha(v_surf, z_surf, v_upper, z_upper)

        def _speed(z):
            if z <= 0:
                return 0.0
            if z <= z_surf:
                return v_surf * (z / z_surf) ** alpha
            if z >= z_upper:
                return v_upper
            return v_surf * (z / z_surf) ** alpha

        def _dir(z):
            if z <= z_surf:
                return dir_surf_deg
            if z >= z_upper:
                return dir_upper_deg
            frac = (z - z_surf) / (z_upper - z_surf)
            diff = ((dir_upper_deg - dir_surf_deg + 180.0) % 360.0) - 180.0
            return dir_surf_deg + frac * diff

        alts = sorted({0, 3, z_surf, 30, 100, 300, z_upper, 1000, 5000})
        u_prof, v_prof = [(0, 0.0)], [(0, 0.0)]
        for z in alts:
            if z == 0:
                continue
            spd = _speed(z)
            rad = math.radians(_dir(z))
            u_prof.append((z, -spd * math.sin(rad)))
            v_prof.append((z, -spd * math.cos(rad)))
        return u_prof, v_prof


# ─── Phase 1 Result Container ────────────────────────────────────────────────

class Phase1Result:
    """Immutable container for Phase 1 pre-calculation outputs."""

    __slots__ = (
        'best_elev', 'best_azi', 'apogee_m',
        'nominal_cx', 'nominal_cy',
        'mu_nominal', 'mu_max', 'sigma_max',
        'ellipse_a', 'ellipse_b', 'ellipse_angle_rad',
        'ellipse_scale_per_sigma',
        'dcx_dmu', 'dcy_dmu',
        'target_radius_m',
        'best_score', 'mode',
    )

    def __init__(self, best_elev, best_azi, apogee_m,
                 nominal_cx, nominal_cy,
                 mu_nominal, mu_max, sigma_max,
                 ellipse_a, ellipse_b, ellipse_angle_rad,
                 ellipse_scale_per_sigma,
                 dcx_dmu=0.0, dcy_dmu=0.0,
                 target_radius_m=50.0,
                 best_score=0.0, mode='Free'):
        self.best_elev               = best_elev
        self.best_azi                = best_azi
        self.apogee_m                = apogee_m
        self.nominal_cx              = nominal_cx
        self.nominal_cy              = nominal_cy
        self.mu_nominal              = mu_nominal
        self.mu_max                  = mu_max
        self.sigma_max               = sigma_max
        self.ellipse_a               = ellipse_a
        self.ellipse_b               = ellipse_b
        self.ellipse_angle_rad       = ellipse_angle_rad
        self.ellipse_scale_per_sigma = ellipse_scale_per_sigma
        self.dcx_dmu                 = dcx_dmu
        self.dcy_dmu                 = dcy_dmu
        self.target_radius_m         = target_radius_m
        self.best_score              = best_score
        self.mode                    = mode


class KazamidoriUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kazamidori_Project - Trajectory & Landing Point Simulator")
        self.geometry("1250x880")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.columnconfigure(0, weight=1, minsize=350)
        self.columnconfigure(1, weight=2)
        self.columnconfigure(2, weight=3, minsize=500)
        self.rowconfigure(0, weight=1)

        self.launch_lat = 35.6828
        self.launch_lon = 139.7590
        self.land_lat = self.launch_lat
        self.land_lon = self.launch_lon
        self.r90_radius = 10.0

        self.selected_motor_file = None
        self.selected_motor_name = "(none selected)"
        # Feat 6: keep burn/avg/max thrust as first-class state so the UI
        # label can be refreshed any time (not just right after a CSV load).
        self.motor_burn_time     = None
        self.motor_avg_thrust    = None
        self.motor_max_thrust    = None
        self.motor_burn_time = 0.0
        self.thrust_data = None

        self.surf_wind_history = deque(maxlen=300)
        # Feat 1: time-series storage with elapsed-second timestamps so the
        # wind speed graph can scroll horizontally as new samples arrive.
        # Each element is the tuple (t_sec_since_start, wind_speed_mps).
        self.surf_wind_time_history = deque(maxlen=300)
        self._wind_start_time = time.time()

        # ── 3-D view state (single source of truth) ───────────────────────────────
        self._fixed_elev     = 25       # locked elevation (deg)
        self._fixed_azim     = 45.0     # Feat 4: azimuth clamped to [0, 90]°
        self._azim_updating  = False    # re-entry guard for slider/drag sync
        self._rot_start_x    = None
        self._rot_start_azim = None

        # ── Uncertainty / dispersion settings (editable via Settings window) ──
        self.wind_uncertainty   = 0.20   # σ_wind / wind (dimensionless)
        self.thrust_uncertainty = 0.05   # σ_thrust / thrust
        self.landing_prob       = 90     # integer % for landing-circle confidence

        # ── Lock & Monitor state ───────────────────────────────────────────────
        self._baseline_wind     = None   # dict set at each successful sim
        self._monitor_after_id  = None   # tk after() id for cancellation
        self._settings_win      = None   # Toplevel ref (None or destroyed-ok)
        self._last_sim_data     = None   # cached last sim_data for auto-rerun

        # ── Result + compass bookkeeping (added with the v2 feature pass) ─────
        self._has_sim_result    = False
        self._compass_ax        = None

        # ── Operation mode state (v0.4 — four operation modes) ────────────────
        self.OPERATION_MODES = (
            "Altitude Competition",
            "Precision Landing",
            "Winged Hover",
            "Free",
        )
        self.operation_mode_var = tk.StringVar(value="Free")
        self.r_max_var          = tk.StringVar(value="50.0")
        self._optimizing        = False
        self._opt_queue         = queue.Queue()
        self._opt_stop_flag     = threading.Event()
        self._opt_thread        = None
        self._opt_progress_win  = None
        self._opt_progress_msg  = None
        self._opt_progress_bar  = None
        self._last_optimization_info = None

        # ── Wind state (direction fixed; speed comes from real-time deque) ─────
        # These StringVars stay as internal state; UI sliders/entries removed.
        # Directions will be replaced by API data in the future.
        self.surf_dir_var = tk.StringVar(value="100")
        self.up_spd_var   = tk.StringVar(value="8.0")
        self.up_dir_var   = tk.StringVar(value="90")
        self._sim_base_wind = 4.0   # m/s — seed for fake realtime generation

        # ── Two-Phase Architecture state ──────────────────────────────────────
        self._p1_queue      = queue.Queue()
        self._p1_stop_flag  = threading.Event()
        self._p1_thread     = None
        self._p1_running    = False
        self._phase1_result = None   # Phase1Result once Phase 1 completes
        self._p2_after_id   = None   # after() id for 1 Hz Phase 2 tick
        self._p2_ellipse    = None   # current ellipse state dict for drawing
        self._p1_win        = None   # Phase 1 progress popup window
        self._p1_win_bar    = None
        self._p1_win_msg    = None

        # ── MC Visualization state (Error Ellipse, CEP, KDE) ─────────────────
        self._mc_scatter   = None   # list of (impact_x, impact_y)
        self._mc_ellipse   = None   # dict: cx, cy, a, b, angle_rad
        self._mc_cep       = None   # float: CEP radius (50th-pctile distance)
        self._mc_running   = False
        self._mc_queue     = queue.Queue()
        self._kde_contours = None   # list of (latlons, color, border_width)
        self._mc_n_runs        = 200
        self._mc_wind_profiles = None   # list of [(alt, spd), …] for spaghetti

        self.create_data_section()
        self.create_profile_section()
        self.create_map_section()

        self.after(500, lambda: self.get_current_location(manual=False))
        self.after(1000, self.simulate_realtime_wind)

        self.update_plots()
        self.after(1000, self.fit_map_bounds)

        self.after(0, self._on_mode_change)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self._p1_stop_flag.set()   # signal Phase 1 thread to stop
            if self._p2_after_id is not None:
                try:
                    self.after_cancel(self._p2_after_id)
                except Exception:
                    pass
            plt.close('all')
            self.quit()
            self.destroy()
            sys.exit()

    def get_current_location(self, manual=False):
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
                self.map_widget.set_position(self.launch_lat, self.launch_lon)
                self._clear_previous_landing()
                self.update_plots()
                self.fit_map_bounds()
                if manual:
                    messagebox.showinfo("Location Retrieved", f"Location acquired:\nLat: {lat}\nLon: {lon}")
        except Exception as e:
            if manual:
                messagebox.showerror("Fetch Error", f"Failed to get location.\n{e}")

    def simulate_realtime_wind(self):
        # Base from rolling mean if available; otherwise use seed value
        base_wind = (sum(self.surf_wind_history) / len(self.surf_wind_history)
                     if self.surf_wind_history else self._sim_base_wind)
        current_wind = max(0.0, random.gauss(base_wind, base_wind * 0.15))
        if random.random() < 0.05:
            current_wind *= 1.5

        t_now = time.time() - self._wind_start_time
        self.surf_wind_history.append(current_wind)
        self.surf_wind_time_history.append((t_now, current_wind))

        # Fix 7 + Feat 1/2/3: keep the realtime label AND the bottom wind
        # sub-figure (time-series + 10-s average + compass) in sync at 1 Hz.
        try:
            self._update_realtime_wind_label()
        except Exception:
            pass
        try:
            self._update_wind_subplots()
        except Exception:
            pass
        self.after(1000, self.simulate_realtime_wind)

    # ── Wind helper (refactored out of run_simulation for readability) ────────
    @staticmethod
    def _wind_components(spd, dir_deg):
        rad = math.radians(dir_deg)
        return -spd * math.sin(rad), -spd * math.cos(rad)

    @staticmethod
    def _meters_per_degree(lat_deg):
        phi = math.radians(lat_deg)
        m_per_deg_lat = (111132.92
                         - 559.82 * math.cos(2 * phi)
                         + 1.175  * math.cos(4 * phi)
                         - 0.0023 * math.cos(6 * phi))
        m_per_deg_lon = (111412.84 * math.cos(phi)
                         - 93.5    * math.cos(3 * phi)
                         + 0.118   * math.cos(5 * phi))
        return m_per_deg_lat, m_per_deg_lon

    def _offset_to_latlon(self, lat0, lon0, dx_east, dy_north):
        m_lat, m_lon = self._meters_per_degree(lat0)
        return (lat0 + dy_north / m_lat,
                lon0 + dx_east  / m_lon)

    def _make_backfire_trigger(self, backfire_alt):
        triggered = [False]

        def trigger(p, h, y):
            if triggered[0]:
                return True
            if y[5] < 0 and h <= backfire_alt:
                triggered[0] = True
                return True
            return False

        return trigger

    def _gather_sim_params(self):
        try:
            launch_lat   = float(self.lat_entry.get())
            launch_lon   = float(self.lon_entry.get())
            elev         = float(self.elev_spin.get())
            azi          = float(self.azi_spin.get())
            rail         = float(self.rail_entry.get())
            airframe_mass = float(self.mass_entry.get())
            airframe_cg   = float(self.cg_entry.get())
            airframe_len  = float(self.len_entry.get())
            radius        = float(self.radius_entry.get())
            nose_len  = float(self.nose_len_entry.get())
            fin_root  = float(self.fin_root_entry.get())
            fin_tip   = float(self.fin_tip_entry.get())
            fin_span  = float(self.fin_span_entry.get())
            fin_pos   = float(self.fin_pos_entry.get())
            motor_pos      = float(self.motor_pos_entry.get())
            motor_dry_mass = float(self.motor_dry_mass_entry.get())
            backfire_delay = float(self.backfire_delay_entry.get())
            para_cd   = float(self.cd_entry.get())
            para_area = float(self.area_entry.get())
            para_lag  = float(self.lag_entry.get())
        except ValueError:
            messagebox.showerror("Input Error",
                "Some parameters are missing.\nPlease fill in all fields with valid numbers.")
            return None

        self.launch_lat = launch_lat
        self.launch_lon = launch_lon

        # Randomize surface wind from rolling deque statistics (μ ± σ)
        if self.surf_wind_history:
            _ws = list(self.surf_wind_history)
            _mu = sum(_ws) / len(_ws)
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
            messagebox.showerror("No Engine Selected",
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

    def _simulate_once(self, elev, azi, params):
        try:
            airframe_mass = max(0.01, params['airframe_mass'])
            airframe_len  = max(0.01, params['airframe_len'])
            radius        = max(0.001, params['radius'])
            airframe_cg   = params['airframe_cg']
            nose_len      = params['nose_len']
            fin_root      = params['fin_root']
            fin_tip       = params['fin_tip']
            fin_span      = params['fin_span']
            fin_pos       = params['fin_pos']
            motor_pos     = params['motor_pos']
            motor_dry_mass = params['motor_dry_mass']
            backfire_delay = params['backfire_delay']
            para_cd    = params['para_cd']
            para_area  = params['para_area']
            para_lag   = params['para_lag']
            rail       = params['rail']
            launch_lat = params['launch_lat']
            launch_lon = params['launch_lon']
            wind_u_prof = params['wind_u_prof']
            wind_v_prof = params['wind_v_prof']
            thrust_data = params['thrust_data']
            motor_burn_time = params['motor_burn_time']

            if not thrust_data:
                return {'ok': False, 'error': 'No thrust data'}

            safe_burn_time = max(0.1, motor_burn_time)
            backfire_time  = safe_burn_time + backfire_delay

            I_z  = 0.5  * airframe_mass * (radius ** 2)
            I_xy = (1/12) * airframe_mass * (3 * (radius ** 2) + airframe_len ** 2)

            env = Environment(latitude=launch_lat, longitude=launch_lon, elevation=0)
            env.set_atmospheric_model(
                type="custom_atmosphere", pressure=None, temperature=300,
                wind_u=wind_u_prof, wind_v=wind_v_prof,
            )

            def _build_rocket():
                motor = SolidMotor(
                    thrust_source=thrust_data,
                    burn_time=safe_burn_time,
                    grain_number=1, grain_density=1815,
                    grain_outer_radius=radius * 0.8,
                    grain_initial_inner_radius=0.005,
                    grain_initial_height=0.1,
                    nozzle_radius=radius * 0.8, throat_radius=0.005,
                    interpolation_method="linear",
                    nozzle_position=0,
                    coordinate_system_orientation="nozzle_to_combustion_chamber",
                    dry_mass=motor_dry_mass,
                    dry_inertia=(1e-5, 1e-5, 1e-6),
                    grain_separation=0.0,
                    grains_center_of_mass_position=0.0,
                    center_of_dry_mass_position=0.0,
                )
                rk = Rocket(
                    radius=radius, mass=airframe_mass, inertia=(I_xy, I_xy, I_z),
                    power_off_drag=para_cd, power_on_drag=para_cd,
                    center_of_mass_without_motor=-airframe_cg,
                )
                rk.add_motor(motor, position=-motor_pos)
                rk.add_nose(length=nose_len, kind="vonKarman", position=0.0)
                rk.add_trapezoidal_fins(
                    n=4, root_chord=fin_root, tip_chord=fin_tip,
                    span=fin_span, position=-fin_pos,
                )
                return rk

            # Pass 1: no-chute flight to find apogee + backfire altitude
            rk1 = _build_rocket()
            fl1 = Flight(
                rocket=rk1, environment=env,
                rail_length=rail, inclination=elev, heading=azi,
                terminate_on_apogee=True,
            )
            t1_arr = fl1.z[:, 0]
            z1_arr = fl1.z[:, 1]
            if backfire_time >= t1_arr[-1]:
                backfire_alt = float(z1_arr[-1])
            else:
                idx_bf_p1 = int((np.abs(t1_arr - backfire_time)).argmin())
                backfire_alt = float(z1_arr[idx_bf_p1])
            backfire_alt = max(backfire_alt, 1.0)

            # Pass 2: full flight with altitude-based parachute trigger
            rk2 = _build_rocket()
            trig = self._make_backfire_trigger(backfire_alt)
            rk2.add_parachute(
                "Main",
                cd_s=para_cd * para_area,
                trigger=trig,
                sampling_rate=105,
                lag=para_lag,
            )
            fl2 = Flight(
                rocket=rk2, environment=env,
                rail_length=rail, inclination=elev, heading=azi,
                terminate_on_apogee=False,
            )

            t_vals  = fl2.z[:, 0]
            x_vals  = fl2.x[:, 1]
            y_vals  = fl2.y[:, 1]
            z_vals  = fl2.z[:, 1]
            vz_vals = fl2.vz[:, 1]

            descending = vz_vals < 0
            below_alt  = z_vals <= backfire_alt
            bf_cands   = np.where(descending & below_alt)[0]
            if len(bf_cands) > 0:
                idx_bf = int(bf_cands[0])
            else:
                idx_bf = int(np.argmax(z_vals))

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
                'ok': True,
                'apogee_m': apogee_m, 'hang_time': hang_time,
                'impact_x': impact_x, 'impact_y': impact_y,
                'r_horiz': r_horiz,
                't_vals': t_vals, 'x_vals': x_vals, 'y_vals': y_vals,
                'z_vals': z_vals, 'vz_vals': vz_vals,
                'idx_bf': idx_bf, 'idx_para': idx_para,
                'bf_abs_time': bf_abs_time, 'para_open_time': para_open_time,
                'backfire_alt': backfire_alt, 'apogee_idx': apogee_idx,
                'elev': elev, 'azi': azi,
            }
        except ZeroDivisionError:
            return {'ok': False,
                    'error': 'ZeroDivisionError (launch failure or unstable attitude)'}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def _apply_sim_result_to_ui(self, res, params, override_r90=None):
        x_vals  = res['x_vals'];  y_vals  = res['y_vals']
        z_vals  = res['z_vals'];  vz_vals = res['vz_vals']
        t_vals  = res['t_vals']
        elev    = res['elev'];    azi     = res['azi']

        azi_rad   = math.radians(azi)
        downrange = x_vals * math.sin(azi_rad) + y_vals * math.cos(azi_rad)

        idx_bf   = res['idx_bf']
        idx_para = res['idx_para']
        bf_z_val = float(z_vals[idx_bf]) if 0 <= idx_bf < len(z_vals) else 0.0
        bf_dr    = float(downrange[idx_bf]) if 0 <= idx_bf < len(downrange) else 0.0

        self.land_lat, self.land_lon = self._offset_to_latlon(
            params['launch_lat'], params['launch_lon'],
            res['impact_x'], res['impact_y'])

        apogee_idx = res['apogee_idx']
        fall_time  = t_vals[-1] - t_vals[apogee_idx]
        horiz_dist = res['r_horiz']
        surf_spd   = params['surf_spd']
        wind_sigma_m   = surf_spd * self.wind_uncertainty * max(fall_time, 0.0)
        thrust_sigma_m = self.thrust_uncertainty * horiz_dist
        combined_sigma = math.hypot(wind_sigma_m, thrust_sigma_m)
        z_score = self._prob_to_z(self.landing_prob)
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
        }
        self._has_sim_result = True
        self._last_sim_data  = sim_data
        self.update_plots(sim_data)
        self.fit_map_bounds()
        return sim_data

    def run_simulation(self):
        self._last_optimization_info = None
        self._render_current_params()

    def _render_current_params(self, override_r90=None):
        params = self._gather_sim_params()
        if params is None:
            return False
        res = self._simulate_once(params['elev'], params['azi'], params)
        if not res['ok']:
            if "ZeroDivisionError" in res.get('error', ''):
                messagebox.showerror(
                    "Launch Failure / Unstable Attitude",
                    "Simulation diverged and could not complete.\n\n"
                    "Likely causes:\n"
                    "1. Motor thrust too weak to leave the rail\n"
                    "2. Aerodynamic parameters (CG, fins) make the rocket highly unstable\n\n"
                    f"Selected engine: {self.selected_motor_name}\n"
                )
            else:
                messagebox.showerror("Sim Error",
                    f"RocketPy execution error:\n{res.get('error', 'unknown')}")
            return False
        self._apply_sim_result_to_ui(res, params, override_r90=override_r90)
        # Clear stale MC results and start a fresh visualization run
        self._mc_scatter   = None
        self._mc_ellipse   = None
        self._mc_cep       = None
        self._kde_contours = None
        self._start_mc_visualization(params)
        return True

    def _monte_carlo_r90(self, elev, azi, base_params,
                         n_trials=8, stop_flag=None):
        distances = []
        succeeded = 0
        rng = random.Random()
        wu = max(self.wind_uncertainty, 0.0)
        tu = max(self.thrust_uncertainty, 0.0)
        raw_thrust = base_params['thrust_data']

        for _ in range(n_trials):
            if stop_flag is not None and stop_flag.is_set():
                break
            u_prof, v_prof, _, _, _ = self._build_perturbed_wind_prof(
                base_params, rng, wu)
            thrust_scale     = max(0.1, 1.0 + rng.gauss(0.0, tu))
            perturbed_thrust = [[t, T * thrust_scale] for (t, T) in raw_thrust]

            p = dict(base_params)
            p['wind_u_prof'] = u_prof
            p['wind_v_prof'] = v_prof
            p['thrust_data'] = perturbed_thrust

            r = self._simulate_once(elev, azi, p)
            if r['ok']:
                distances.append(math.hypot(r['impact_x'], r['impact_y']))
                succeeded += 1

        if not distances:
            return float('inf'), 0.0
        distances.sort()
        p_idx = max(0, min(len(distances) - 1,
                           int(round((self.landing_prob / 100.0)
                                     * len(distances))) - 1))
        return distances[p_idx], succeeded / n_trials

    # ── MC Visualization: Error Ellipse / CEP / KDE ───────────────────────────

    def _build_perturbed_wind_prof(self, params, rng, wu):
        """Full-profile wind perturbation for MC simulations.

        Applies global speed/direction uncertainty to both surface and
        upper-level anchors, builds a multi-layer profile via
        WindProfileBuilder, then adds independent per-layer noise to
        model upper-level turbulence.  Returns (u_prof, v_prof,
        surf_spd, up_spd, spd_profile) where spd_profile is a list of
        (alt_m, speed_m_s) tuples for the spaghetti visualisation.
        """
        base_surf   = max(params['surf_spd'], 0.1)
        base_up     = max(params['up_spd'],   0.1)
        dir_sigma   = wu * 60.0

        surf_spd = max(0.0, rng.gauss(params['surf_spd'], wu * base_surf))
        up_spd   = max(0.0, rng.gauss(params['up_spd'],   wu * base_up))
        surf_dir = params['surf_dir'] + rng.gauss(0.0, dir_sigma)
        up_dir   = params['up_dir']   + rng.gauss(0.0, dir_sigma)

        u_prof, v_prof = WindProfileBuilder.build(
            surf_spd, surf_dir, 3.0, up_spd, up_dir, 100.0)

        # Independent per-layer perturbation — models upper-level turbulence
        layer_sigma = wu * base_surf * 0.35
        if layer_sigma > 1e-6:
            u_prof = [(z, u + rng.gauss(0.0, layer_sigma)) for z, u in u_prof]
            v_prof = [(z, v + rng.gauss(0.0, layer_sigma)) for z, v in v_prof]

        # Speed magnitude per altitude point for spaghetti plot
        spd_prof = [(z_u, math.sqrt(u ** 2 + v ** 2))
                    for (z_u, u), (_, v) in zip(u_prof, v_prof)]

        return u_prof, v_prof, surf_spd, up_spd, spd_prof

    def _start_mc_visualization(self, params):
        if self._mc_running:
            return
        self._mc_running = True
        t = threading.Thread(
            target=self._mc_viz_worker,
            args=(params, self._mc_n_runs, self._mc_queue),
            daemon=True)
        t.start()
        self.after(400, self._poll_mc_viz_queue)

    def _mc_viz_worker(self, params, n_runs, result_queue):
        scatter       = []
        wind_profiles = []
        rng = random.Random()
        wu  = max(self.wind_uncertainty,  0.0)
        tu  = max(self.thrust_uncertainty, 0.0)
        raw_thrust = params['thrust_data']
        elev = params['elev']
        azi  = params['azi']
        for _ in range(n_runs):
            u_prof, v_prof, _, _, spd_prof = self._build_perturbed_wind_prof(
                params, rng, wu)
            thrust_scale = max(0.1, 1.0 + rng.gauss(0.0, tu))
            perturbed    = [[t, T * thrust_scale] for (t, T) in raw_thrust]
            p = dict(params)
            p['wind_u_prof'] = u_prof
            p['wind_v_prof'] = v_prof
            p['thrust_data'] = perturbed
            r = self._simulate_once(elev, azi, p)
            if r['ok']:
                scatter.append((r['impact_x'], r['impact_y']))
            wind_profiles.append(spd_prof)
        result_queue.put((scatter, wind_profiles))

    def _poll_mc_viz_queue(self):
        try:
            result = self._mc_queue.get_nowait()
        except queue.Empty:
            self.after(400, self._poll_mc_viz_queue)
            return
        self._mc_running = False
        scatter, wind_profiles = result
        self._apply_mc_viz_results(scatter, wind_profiles)

    def _apply_mc_viz_results(self, scatter, wind_profiles=None):
        if len(scatter) < 4:
            return
        self._mc_scatter       = scatter
        self._mc_wind_profiles = wind_profiles
        arr = np.array(scatter, dtype=float)
        cx, cy = float(arr[:, 0].mean()), float(arr[:, 1].mean())
        cov = np.cov(arr[:, 0], arr[:, 1])
        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigh returns ascending order; largest eigenvalue last
        lam2, lam1 = float(eigvals[0]), float(eigvals[1])
        major_vec = eigvecs[:, 1]
        angle_rad = float(math.atan2(float(major_vec[1]), float(major_vec[0])))
        k = self._chi2_scale(self.landing_prob)
        a = k * math.sqrt(max(lam1, 0.0))
        b = k * math.sqrt(max(lam2, 0.0))
        # floor: prevent degenerate line when scatter is nearly collinear
        b = max(b, max(0.5, a * 0.05))
        self._mc_ellipse = {'cx': cx, 'cy': cy, 'a': a, 'b': b,
                            'angle_rad': angle_rad}
        # CEP: 50th-percentile distance from origin (not from centroid)
        dists = sorted(math.hypot(x, y) for x, y in scatter)
        mid = (len(dists) - 1) / 2.0
        lo, hi = int(mid), min(int(mid) + 1, len(dists) - 1)
        self._mc_cep = dists[lo] + (mid - lo) * (dists[hi] - dists[lo])
        # KDE contours (use configured confidence level)
        launch_lat = getattr(self, 'launch_lat', None)
        launch_lon = getattr(self, 'launch_lon', None)
        if launch_lat is not None:
            self._kde_contours = self._compute_kde_contours(
                scatter, launch_lat, launch_lon,
                conf_pct=self.landing_prob)
        # Refresh plots and map
        if getattr(self, '_last_sim_data', None) is not None:
            self.update_plots(self._last_sim_data)
        self.draw_map_elements()

    def _compute_kde_contours(self, scatter, launch_lat, launch_lon, conf_pct=90):
        try:
            from scipy.stats import gaussian_kde
            import matplotlib.pyplot as _plt
            import numpy as _np
        except ImportError:
            return []
        if len(scatter) < 5:
            return []
        xs = _np.array([p[0] for p in scatter])
        ys = _np.array([p[1] for p in scatter])
        try:
            kde = gaussian_kde(_np.vstack([xs, ys]))
        except Exception:
            return []
        pad = max(xs.ptp(), ys.ptp(), 1.0) * 0.5
        gx = _np.linspace(xs.min() - pad, xs.max() + pad, 120)
        gy = _np.linspace(ys.min() - pad, ys.max() + pad, 120)
        GX, GY = _np.meshgrid(gx, gy)
        Z = kde(_np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        # compute probability-mass thresholds: always 50%, 70%, + configured conf_pct
        z_flat = Z.ravel()
        z_sorted = _np.sort(z_flat)[::-1]
        cumsum = _np.cumsum(z_sorted)
        cumsum /= cumsum[-1]
        outer_frac = max(min(conf_pct / 100.0, 0.999), 0.501)
        levels_pm = sorted({0.50, 0.70, outer_frac})
        level_vals = []
        for pm in levels_pm:
            idx = _np.searchsorted(cumsum, pm)
            idx = min(idx, len(z_sorted) - 1)
            level_vals.append(float(z_sorted[idx]))
        # deduplicate while preserving order
        seen, unique_vals = set(), []
        for v in level_vals:
            key = round(v, 12)
            if key not in seen:
                seen.add(key); unique_vals.append(v)
        if len(unique_vals) < 2:
            return []
        # extract contour paths via a temporary figure
        fig_tmp, ax_tmp = _plt.subplots()
        cs = ax_tmp.contour(GX, GY, Z, levels=sorted(unique_vals))
        _plt.close(fig_tmp)
        sorted_lv = sorted(unique_vals)
        # innermost (50%) = brightest orange; outermost (conf_pct%) = yellow
        palette = ['#ff6600', '#ff9900', '#ffcc00', '#ffe066']
        width   = [3, 2, 1, 1]
        lv_style = {lv: (palette[min(i, len(palette)-1)],
                         width[min(i, len(width)-1)])
                    for i, lv in enumerate(sorted_lv[::-1])}
        contours = []
        for collection, lv in zip(cs.collections, sorted_lv):
            col, bw = lv_style.get(lv, ('#ffcc00', 1))
            for path in collection.get_paths():
                verts = path.vertices
                if len(verts) < 3:
                    continue
                latlons = [self._offset_to_latlon(launch_lat, launch_lon,
                                                   float(v[0]), float(v[1]))
                           for v in verts]
                contours.append((latlons, col, bw))
        return contours

    def _optimize_worker(self, mode, base_params, r_max):
        try:
            if mode == "Precision Landing":
                elev_grid = [60, 66, 72, 78, 84, 90]
                azi_grid  = [0, 30, 60, 90, 120, 150,
                             180, 210, 240, 270, 300, 330]
                use_mc = True

                def objective(res, mc_r=None):
                    if not res['ok']:
                        return float('-inf')
                    r = res['r_horiz']
                    if mc_r is None:
                        if r > r_max:
                            return float('-inf')
                        return (r_max - r) + res['hang_time']
                    if r + mc_r > r_max:
                        return float('-inf')
                    return (r_max - r) + res['hang_time']

            elif mode == "Altitude Competition":
                elev_grid = [60, 66, 72, 78, 84, 90]
                azi_grid  = [0, 45, 90, 135, 180, 225, 270, 315]
                use_mc = True

                def objective(res, mc_r=None):
                    if not res['ok']:
                        return float('-inf')
                    r = res['r_horiz']
                    if mc_r is None:
                        if r > r_max:
                            return float('-inf')
                        return res['apogee_m']
                    if r + mc_r > r_max:
                        return float('-inf')
                    return res['apogee_m']

            elif mode == "Winged Hover":
                elev_grid = [60, 66, 72, 78, 84, 90]
                azi_grid  = [0, 45, 90, 135, 180, 225, 270, 315]
                use_mc = True

                def objective(res, mc_r=None):
                    if not res['ok']:
                        return float('-inf')
                    r = res['r_horiz']
                    if mc_r is None:
                        if r > r_max:
                            return float('-inf')
                        return res['hang_time']
                    if r + mc_r > r_max:
                        return float('-inf')
                    return res['hang_time']
            else:
                self._opt_queue.put(('error', f'Unknown mode: {mode}'))
                return

            candidates = []
            total = len(elev_grid) * len(azi_grid)
            done = 0
            phase1_weight = 0.6 if use_mc else 1.0

            self._opt_queue.put(
                ('progress', f"Phase 1: Coarse search (0/{total})", 0.0))

            for e_ in elev_grid:
                for a_ in azi_grid:
                    if self._opt_stop_flag.is_set():
                        self._opt_queue.put(('cancelled', None))
                        return
                    res = self._simulate_once(e_, a_, base_params)
                    done += 1
                    if res['ok']:
                        score = objective(res, mc_r=None)
                        candidates.append((score, e_, a_, res))
                    frac = (done / total) * phase1_weight
                    self._opt_queue.put((
                        'progress',
                        f"Phase 1: Coarse search ({done}/{total}) "
                        f"elev={e_:.0f}° azi={a_:.0f}°",
                        frac))

            if not candidates:
                self._opt_queue.put((
                    'error',
                    'Simulation failed for all candidates.\n'
                    'Please check your parameters.'))
                return

            candidates.sort(key=lambda x: -x[0] if math.isfinite(x[0])
                                              else float('inf'))

            if not use_mc:
                finite = [c for c in candidates if math.isfinite(c[0])]
                if not finite:
                    self._opt_queue.put((
                        'error',
                        f'No candidate satisfies constraint (r ≤ {r_max:.1f} m).\n'
                        'Try increasing r_max or adjusting airframe / wind settings.'))
                    return
                score, best_e, best_a, best_res = finite[0]
                best_mc_r = None
            else:
                top_n = min(5, len(candidates))
                mc_trials = 8
                best = None

                for i in range(top_n):
                    if self._opt_stop_flag.is_set():
                        self._opt_queue.put(('cancelled', None))
                        return
                    _, e_, a_, res = candidates[i]
                    mc_r, succ = self._monte_carlo_r90(
                        e_, a_, base_params,
                        n_trials=mc_trials,
                        stop_flag=self._opt_stop_flag)
                    score = objective(res, mc_r=mc_r)
                    phase2_span = (1 - phase1_weight) * 0.75
                    prog_frac = phase1_weight + (i + 1) / top_n * phase2_span
                    self._opt_queue.put((
                        'progress',
                        f"Phase 2: MC verification ({i+1}/{top_n}) "
                        f"elev={e_:.0f}° azi={a_:.0f}°  "
                        f"MC r={mc_r:.1f}m (≤{r_max:.1f}m?)",
                        prog_frac))
                    if math.isfinite(score):
                        if best is None or score > best[0]:
                            best = (score, e_, a_, res, mc_r)

                if best is None:
                    self._opt_queue.put((
                        'error',
                        f'No candidate satisfies constraint (r + MC {self.landing_prob}% circle ≤ {r_max:.1f} m).\n'
                        'Try increasing r_max or adjusting wind / airframe settings.'))
                    return

                score, best_e, best_a, best_res, best_mc_r = best

            if self._opt_stop_flag.is_set():
                self._opt_queue.put(('cancelled', None))
                return
            self._opt_queue.put((
                'progress',
                f"Phase 3: Final MC analysis (elev={best_e:.1f}° azi={best_a:.1f}°)",
                0.9))
            final_mc_trials = 16
            final_mc_r, final_mc_succ = self._monte_carlo_r90(
                best_e, best_a, base_params,
                n_trials=final_mc_trials,
                stop_flag=self._opt_stop_flag)
            if self._opt_stop_flag.is_set():
                self._opt_queue.put(('cancelled', None))
                return
            if math.isfinite(final_mc_r):
                reported_mc_r = final_mc_r
            else:
                reported_mc_r = best_mc_r

            self._opt_queue.put(('progress', 'Phase 3: Complete', 1.0))
            self._opt_queue.put(('done', {
                'mode': mode, 'r_max': r_max,
                'elev': best_e, 'azi': best_a, 'score': score,
                'result': best_res, 'mc_r': reported_mc_r,
                'mc_success': final_mc_succ,
                'mc_trials': final_mc_trials,
            }))
        except Exception as e:
            self._opt_queue.put(('error', f'Error during optimization: {e}'))

    def _run_optimization_threaded(self, mode, params, r_max):
        self._optimizing = True
        self._opt_stop_flag.clear()
        try:
            while True:
                self._opt_queue.get_nowait()
        except queue.Empty:
            pass

        win = tk.Toplevel(self)
        self._opt_progress_win = win
        win.title(f"Optimizing — {mode}")
        win.geometry("460x170")
        win.resizable(False, False)
        win.transient(self)

        frm = ttk.Frame(win, padding=12)
        frm.pack(fill='both', expand=True)
        ttk.Label(frm, text=f"Mode: {mode}",
                  font=("Arial", 10, "bold")).pack(anchor='w')
        ttk.Label(frm,
                  text=f"Target radius r_max = {r_max:.1f} m").pack(anchor='w')
        self._opt_progress_msg = tk.StringVar(value="Preparing...")
        ttk.Label(frm, textvariable=self._opt_progress_msg,
                  font=("Arial", 9)).pack(anchor='w', pady=(6, 3))
        self._opt_progress_bar = ttk.Progressbar(
            frm, mode='determinate', maximum=100)
        self._opt_progress_bar.pack(fill='x', pady=(0, 8))
        ttk.Button(frm, text="Cancel",
                   command=self._cancel_optimization).pack(anchor='e')
        win.protocol("WM_DELETE_WINDOW", self._cancel_optimization)

        try:
            self.main_action_btn.state(["disabled"])
        except Exception:
            pass

        self._opt_thread = threading.Thread(
            target=self._optimize_worker,
            args=(mode, params, r_max),
            daemon=True)
        self._opt_thread.start()
        self.after(150, self._poll_optimization)

    def _cancel_optimization(self):
        self._opt_stop_flag.set()
        if self._opt_progress_msg is not None:
            try:
                self._opt_progress_msg.set("Cancelling...")
            except Exception:
                pass

    def _poll_optimization(self):
        if not self._optimizing:
            return
        drained = False
        try:
            while True:
                msg = self._opt_queue.get_nowait()
                drained = True
                kind = msg[0]
                if kind == 'progress':
                    _, text, frac = msg
                    if self._opt_progress_msg is not None:
                        try:
                            self._opt_progress_msg.set(text)
                        except Exception:
                            pass
                    if self._opt_progress_bar is not None and frac is not None:
                        try:
                            self._opt_progress_bar['value'] = max(
                                0.0, min(100.0, float(frac) * 100.0))
                        except Exception:
                            pass
                elif kind == 'done':
                    self._finish_optimization(msg[1])
                    return
                elif kind == 'cancelled':
                    self._finish_optimization(None, cancelled=True)
                    return
                elif kind == 'error':
                    self._finish_optimization(None, error=msg[1])
                    return
        except queue.Empty:
            pass
        self.after(150, self._poll_optimization)

    def _finish_optimization(self, payload, cancelled=False, error=None):
        self._optimizing = False
        try:
            if self._opt_progress_win is not None:
                self._opt_progress_win.destroy()
        except Exception:
            pass
        self._opt_progress_win  = None
        self._opt_progress_msg  = None
        self._opt_progress_bar  = None
        try:
            self.main_action_btn.state(["!disabled"])
        except Exception:
            pass

        if cancelled:
            messagebox.showinfo("Optimization Cancelled",
                                "Optimization was stopped.\n"
                                "No optimal values have been applied.")
            return
        if error:
            messagebox.showerror("Optimization Error", error)
            return
        if payload is None:
            return

        best_elev = float(payload['elev'])
        best_azi  = float(payload['azi'])
        res       = payload['result']
        mc_r      = payload.get('mc_r')
        mode      = payload['mode']
        r_max     = payload['r_max']

        try:
            self.elev_spin.delete(0, tk.END)
            self.elev_spin.insert(0, f"{best_elev:.1f}")
        except Exception:
            pass
        try:
            self.azi_spin.delete(0, tk.END)
            self.azi_spin.insert(0, f"{best_azi:.1f}")
        except Exception:
            pass

        try:
            self.update_idletasks()
        except Exception:
            pass

        mode_short_map = {
            "Precision Landing": "Hover",
            "Altitude Competition": "Altitude",
            "Winged Hover": "Winged",
        }
        hover_score = None
        if mode == "Precision Landing":
            try:
                hover_score = (r_max - float(res['r_horiz'])) + float(res['hang_time'])
            except Exception:
                hover_score = None

        self._last_optimization_info = {
            'elev':        best_elev,
            'azi':         best_azi,
            'mode':        mode,
            'mode_short':  mode_short_map.get(mode, mode),
            'mc_r':        mc_r,
            'r_max':       r_max,
            'hover_score': hover_score,
        }

        final_ok = self._render_current_params(override_r90=mc_r)
        if not final_ok:
            messagebox.showwarning(
                "Final Simulation Failed",
                "Could not reproduce the optimal trajectory.\n"
                "Optimal angles have been applied to the UI — press Run to retry.")

        lines = [
            f"[{mode} — Optimization Complete]",
            "",
            f"★ Best Elevation: {best_elev:.1f}°",
            f"★ Best Azimuth:   {best_azi:.1f}°",
            "",
            f"Apogee:      {res['apogee_m']:.1f} m",
            f"Hang time:   {res['hang_time']:.2f} s",
            f"Horiz dist r: {res['r_horiz']:.1f} m  /  r_max: {r_max:.1f} m",
        ]
        if mc_r is not None and math.isfinite(mc_r):
            trials = payload.get('mc_trials', 0)
            lines.append(
                f"MC {self.landing_prob}% circle: {mc_r:.1f} m  "
                f"(≤ r_max={r_max:.1f} m,  trials={trials})")
        if hover_score is not None:
            lines.append("")
            lines.append(
                f"★ Hover score (r_max - r + t): {hover_score:.2f}")
        lines.append("")
        if final_ok:
            lines.append("Optimal angles applied to Elevation/Azimuth inputs.")
            lines.append("Optimal trajectory drawn on 3D profile and map.")
            lines.append("")
            lines.append("⚠ Wind monitor (Lock & Monitor) enabled automatically.")
            lines.append("  An alert and re-simulation will trigger if wind drifts beyond tolerance.")
        messagebox.showinfo("Optimization Complete",
                            "\n".join(lines))

        if final_ok:
            self._auto_enable_monitor_mode()

    _MODE_DEFAULT_RMAX = {
        "Free": None,
        "Precision Landing":    50.0,
        "Altitude Competition":    250.0,
        "Winged Hover":    250.0,
    }

    def _apply_mode_default_rmax(self, mode):
        default = self._MODE_DEFAULT_RMAX.get(mode)
        if default is None:
            return
        try:
            self.r_max_var.set(f"{default:.1f}")
        except Exception:
            pass

    def _on_mode_change(self, event=None):
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

    def _update_main_action_btn(self):
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

    def _p1_objective_score(self, res, mode):
        if mode == 'Altitude Competition':
            return res['apogee_m']
        elif mode == 'Precision Landing':
            return -res['r_horiz']
        elif mode == 'Winged Hover':
            return res['hang_time']
        else:
            return res['apogee_m']

    # ── Azimuth helpers ───────────────────────────────────────────────────────
    def _set_azim(self, azim, source="code"):
        if self._azim_updating:
            return
        self._azim_updating = True
        try:
            a = float(azim)
            a = ((a + 180.0) % 360.0) - 180.0
            if a < 0:
                a = 0.0
            elif a > 90:
                a = 90.0
            self._fixed_azim = a

            if hasattr(self, 'azim_label'):
                self.azim_label.config(text=f"{a:+.0f}°")

            if source != "slider" and hasattr(self, 'azim_var'):
                try:
                    if abs(self.azim_var.get() - a) > 0.5:
                        self.azim_var.set(a)
                except tk.TclError:
                    pass

            if hasattr(self, 'ax'):
                try:
                    self.ax.view_init(elev=self._fixed_elev, azim=a)
                    if getattr(self, '_compass_ax', None) is not None:
                        self._draw_compass()
                    self.canvas.draw_idle()
                except Exception:
                    pass
        finally:
            self._azim_updating = False

    def _on_azim_slider(self, value):
        self._set_azim(value, source="slider")

    def _reset_azim(self):
        self._set_azim(45.0, source="code")

    def _on_wheel_rotate_azim(self, event, delta_override=None):
        d = delta_override if delta_override is not None else getattr(event, 'delta', 0)
        if d == 0:
            return
        step = 5.0 if d > 0 else -5.0
        new_azim = self._fixed_azim + step
        self._set_azim(new_azim, source="code")
        return "break"

    def _on_canvas_press(self, event):
        if event.inaxes is self.ax and event.button == 1:
            self._rot_start_x    = event.x
            self._rot_start_azim = self._fixed_azim

    def _on_canvas_motion(self, event):
        if self._rot_start_x is None:
            return
        if event.button != 1:
            return
        dx = event.x - self._rot_start_x
        self._set_azim(self._rot_start_azim - dx * 0.4, source="drag")

    def _on_canvas_release(self, event):
        self._rot_start_x = None

    def _on_view_changed(self, event=None):
        if not hasattr(self, 'ax'):
            return
        try:
            ax_azim = float(self.ax.azim)
            ax_elev = float(self.ax.elev)
        except Exception:
            return

        azim_drift = abs(((ax_azim - self._fixed_azim) + 180.0) % 360.0 - 180.0)
        elev_drift = abs(ax_elev - self._fixed_elev)
        if azim_drift < 0.5 and elev_drift < 0.5:
            return

        self._fixed_azim = ((ax_azim + 180.0) % 360.0) - 180.0
        self._fixed_elev = ax_elev

        if hasattr(self, 'azim_var'):
            self._azim_updating = True
            try:
                try:
                    self.azim_var.set(self._fixed_azim)
                except tk.TclError:
                    pass
                if hasattr(self, 'azim_label'):
                    self.azim_label.config(text=f"{self._fixed_azim:+.0f}°")
            finally:
                self._azim_updating = False

        self._draw_compass()
        self.canvas.draw_idle()

    # ── Uncertainty helpers ──────────────────────────────────────────────────
    def _prob_to_z(self, pct):
        table = {50: 0.674, 68: 1.000, 80: 1.282, 85: 1.440,
                 90: 1.645, 95: 1.960, 99: 2.576}
        return table.get(int(pct), 1.645)

    # Chi-squared 2-DOF quantiles: chi2.ppf(p, df=2)
    _CHI2_2DOF = {50: 1.386, 68: 2.296, 80: 3.219, 85: 3.794,
                  90: 4.605, 95: 5.991, 99: 9.210}

    def _chi2_scale(self, prob_pct):
        """Return sqrt(chi²(2, prob_pct/100)) for error-ellipse scaling."""
        val = self._CHI2_2DOF.get(int(prob_pct), 4.605)
        return math.sqrt(val)

    def _open_settings_window(self):
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
        win.geometry("380x260")
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

        ttk.Label(frm, text="Landing circle probability (%):").grid(
            row=3, column=0, sticky="w", pady=4)
        prob_var = tk.StringVar(value=str(self.landing_prob))
        prob_combo = ttk.Combobox(frm, textvariable=prob_var, width=8,
                                  values=[50, 68, 80, 85, 90, 95, 99],
                                  state="normal")
        prob_combo.grid(row=3, column=1, sticky="e", pady=4)

        ttk.Label(frm,
                  text="(Re-run the simulation to apply the new settings.)",
                  font=("Arial", 8), foreground="gray").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(6, 10))

        btn_f = ttk.Frame(frm)
        btn_f.grid(row=5, column=0, columnspan=2, sticky="ew")
        btn_f.columnconfigure(0, weight=1)
        btn_f.columnconfigure(1, weight=1)

        def apply_and_close():
            try:
                w  = float(wind_var.get())   / 100.0
                th = float(thrust_var.get()) / 100.0
                p  = int(float(prob_var.get()))
                if w < 0 or th < 0:
                    raise ValueError("Uncertainty must be ≥ 0.")
                if not 1 <= p <= 99:
                    raise ValueError("Probability must be between 1 and 99.")
                prob_changed = (p != self.landing_prob)
                self.wind_uncertainty   = w
                self.thrust_uncertainty = th
                self.landing_prob       = p
                if prob_changed:
                    self._release_lock_if_active(reason_label="⭘ Unlocked")
                messagebox.showinfo(
                    "Settings Applied",
                    f"Wind uncertainty   : ±{w*100:.1f}%\n"
                    f"Thrust uncertainty: ±{th*100:.1f}%\n"
                    f"Landing confidence: {p}%",
                    parent=win)
                win.destroy()
                self._settings_win = None
            except ValueError as e:
                messagebox.showerror("Invalid input",
                                     f"Could not parse settings:\n{e}",
                                     parent=win)

        def cancel():
            win.destroy()
            self._settings_win = None

        ttk.Button(btn_f, text="Apply & Close",
                   command=apply_and_close).grid(row=0, column=0,
                                                 sticky="ew", padx=(0, 3))
        ttk.Button(btn_f, text="Cancel",
                   command=cancel).grid(row=0, column=1,
                                        sticky="ew", padx=(3, 0))
        win.protocol("WM_DELETE_WINDOW", cancel)

    # ── Lock & Monitor helpers ───────────────────────────────────────────────
    def _release_lock_if_active(self, reason_label=None):
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

    def _toggle_lock_monitor(self):
        if self.lock_monitor_var.get():
            self._capture_wind_baseline()
            try:
                self.main_action_btn.state(["disabled"])
            except Exception:
                pass
            self.monitor_status_label.config(
                text="🔒 LOCKED — monitoring wind",
                foreground="green")
            self._schedule_monitor_tick()
        else:
            try:
                self.main_action_btn.state(["!disabled"])
            except Exception:
                pass
            self.monitor_status_label.config(
                text="⭘ Unlocked",
                foreground="gray")
            if self._monitor_after_id is not None:
                try:
                    self.after_cancel(self._monitor_after_id)
                except Exception:
                    pass
                self._monitor_after_id = None

    def _auto_enable_monitor_mode(self):
        try:
            if not self.lock_monitor_var.get():
                self.lock_monitor_var.set(True)
                self._toggle_lock_monitor()
            else:
                self._capture_wind_baseline()
        except Exception:
            pass

    def _wind_avg_recent(self, window_sec=10.0):
        """Feat 3: rolling average of surface wind over the past ``window_sec``.

        Used both as the live reference line on the wind-speed time series
        and as the comparison value inside ``_monitor_wind_tick`` so the
        monitor reacts to *sustained* drift rather than single-tick noise.
        """
        history = list(getattr(self, 'surf_wind_time_history', []) or [])
        if not history:
            return self._sim_base_wind
        t_latest = history[-1][0]
        recent = [w for (t, w) in history if t >= t_latest - window_sec]
        if not recent:
            recent = [history[-1][1]]
        return sum(recent) / len(recent)

    def _capture_wind_baseline(self):
        """Snapshot current wind values to compare against during monitoring.

        Feat 3 / Fix 6: the surface baseline is the past-10-second moving
        average rather than the long-running average over the full history,
        so a fresh lock reflects the *current* atmospheric state. Using the
        same window on both sides of the comparison (baseline AND
        ``_monitor_wind_tick``'s ``cur_surf``) ensures the monitor reacts to
        real drift instead of being smothered by long-window smoothing.
        """
        try:
            surf_spd = self._wind_avg_recent(window_sec=10.0)
            self._baseline_wind = {
                "surf_spd": surf_spd,
                "surf_dir": float(self.surf_dir_var.get()),
                "up_spd":   float(self.up_spd_var.get()),
                "up_dir":   float(self.up_dir_var.get()),
            }
        except (ValueError, AttributeError):
            self._baseline_wind = None

    def _schedule_monitor_tick(self):
        self._monitor_after_id = self.after(2000, self._monitor_wind_tick)

    @staticmethod
    def _angle_diff(a, b):
        return abs(((a - b) + 180.0) % 360.0 - 180.0)

    def _monitor_wind_tick(self):
        """Periodic wind check.

        Fix 6 — Tolerance logic:
          The previous version compared the *long-running* average of
          ``surf_wind_history`` (300-sample deque ≈ 5 minutes) against
          itself — both baseline and current ended up nearly equal,
          so the threshold was effectively never crossed. We now use
          the past-10-second rolling average (matches the red horizontal
          line on the time-series graph) on BOTH sides of the comparison,
          so a real change in conditions actually crosses the threshold.

        Fix 7 — ``_flash_alert`` removed:
          The old call to ``self._flash_alert()`` raised
          ``AttributeError: '_tkinter.tkapp' object has no attribute '_flash_alert'``
          because that helper never existed.  We now show a blocking
          ``messagebox.showwarning`` pop-up and then automatically execute
          ``run_simulation()`` to recompute the trajectory under the new wind.
        """
        self._monitor_after_id = None
        if not self.lock_monitor_var.get():
            return
        if self._baseline_wind is None:
            self._schedule_monitor_tick()
            return

        try:
            # Fix 6: 10-second rolling avg, matching how the baseline was captured.
            cur_surf     = self._wind_avg_recent(window_sec=10.0)
            cur_surf_dir = float(self.surf_dir_var.get())
            cur_up       = float(self.up_spd_var.get())
            cur_up_dir   = float(self.up_dir_var.get())
        except (ValueError, AttributeError):
            self._schedule_monitor_tick()
            return

        b = self._baseline_wind
        SPD_TOL, DIR_TOL = 2.0, 15.0

        # Fix 6: explicit absolute differences — easier to reason about and
        # easier to surface in the warning popup.
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
            # Fix 7: messagebox.showwarning replaces the missing _flash_alert.
            # showwarning is modal — it returns only after the user dismisses
            # it, so the ensuing run_simulation() call is sequential and safe.
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
            # Temporarily allow the internal RUN call, then re-lock.
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
            # Refresh baseline so the next tick compares against the NEW state,
            # not the just-superseded one (prevents repeated immediate retriggers).
            self._capture_wind_baseline()

        self._schedule_monitor_tick()

    # ── Realtime wind display ────────────────────────────────────────────────
    def _update_realtime_wind_label(self):
        lbl = getattr(self, 'realtime_wind_label', None)
        if lbl is None:
            return
        try:
            surf_spd, surf_dir, up_spd, up_dir = self._read_current_wind()
            hist = getattr(self, 'surf_wind_history', None) or []
            gust = max(hist) if hist else surf_spd
            # Feat 4: show only numerical values — no "Now:" prefix needed.
            lbl.config(
                text=(f"Surface: {surf_spd:.1f} m/s  @ {surf_dir:.0f}°"
                      f"   (Gust {gust:.1f})"
                      f"   |   Upper: {up_spd:.1f} m/s @ {up_dir:.0f}°"),
            )
        except Exception:
            pass

    # ── Params scroll helpers ────────────────────────────────────────────────
    def _on_params_wheel(self, event, delta_override=None):
        canvas = getattr(self, '_params_canvas', None)
        if canvas is None:
            return "break"
        d = delta_override if delta_override is not None else getattr(event, 'delta', 0)
        try:
            step = -1 if d > 0 else 1
            canvas.yview_scroll(step, "units")
        except Exception:
            pass
        return "break"

    def _bind_params_wheel_recursive(self, widget):
        try:
            widget.bind("<MouseWheel>", self._on_params_wheel, add="+")
            widget.bind(
                "<Button-4>",
                lambda e: self._on_params_wheel(e, delta_override=+120),
                add="+",
            )
            widget.bind(
                "<Button-5>",
                lambda e: self._on_params_wheel(e, delta_override=-120),
                add="+",
            )
        except Exception:
            pass
        try:
            for child in widget.winfo_children():
                self._bind_params_wheel_recursive(child)
        except Exception:
            pass

    # ── Layout helper ────────────────────────────────────────────────────────
    def _apply_safe_layout(self):
        import warnings
        try:
            if getattr(self, 'ax', None) is not None:
                self.ax.set_position(list(self._PLOT_RECT))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                self.fig.subplots_adjust(
                    left=self._PLOT_RECT[0],
                    right=self._PLOT_RECT[0] + self._PLOT_RECT[2],
                    bottom=self._PLOT_RECT[1],
                    top=self._PLOT_RECT[1] + self._PLOT_RECT[3],
                )
        except Exception:
            pass

    # ── Compass + fixed-label drawing helpers ────────────────────────────────
    def _draw_compass(self):
        if getattr(self, '_compass_ax', None) is not None:
            try:
                self._compass_ax.remove()
            except Exception:
                pass
            self._compass_ax = None

        cax = self.fig.add_axes([0.83, 0.04, 0.14, 0.14], facecolor='none',
                                zorder=20)
        self._compass_ax = cax
        cax.set_xlim(-1.4, 1.4); cax.set_ylim(-1.4, 1.4)
        cax.set_aspect('equal')
        cax.set_xticks([]); cax.set_yticks([])
        for sp in cax.spines.values():
            sp.set_visible(False)

        cax.add_patch(plt.Circle((0, 0), 1.15, fill=True,
                                 facecolor='white', edgecolor='gray',
                                 lw=0.8, alpha=0.85))

        a = math.radians(self._fixed_azim)
        e = math.radians(self._fixed_elev)
        def proj(vx, vy):
            sx = vx * (-math.sin(a)) + vy * math.cos(a)
            sy = (vx * (-math.sin(e) * math.cos(a))
                  + vy * (-math.sin(e) * math.sin(a)))
            return sx, sy

        def norm(x, y):
            r = math.hypot(x, y) or 1.0
            return x / r, y / r

        nx, ny = norm(*proj(0, 1))
        ex, ey = norm(*proj(1, 0))
        sx, sy = -nx, -ny
        wx, wy = -ex, -ey

        R = 0.78
        cax.annotate('', xy=(nx * R, ny * R), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='-|>', color='red', lw=1.8))
        for dx, dy in ((ex, ey), (sx, sy), (wx, wy)):
            cax.annotate('', xy=(dx * R, dy * R), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='-|>',
                                         color='dimgray', lw=1.0))

        cax.text(nx * 1.10, ny * 1.10, 'N',
                 color='red', fontsize=9, fontweight='bold',
                 ha='center', va='center')
        cax.text(ex * 1.10, ey * 1.10, 'E',
                 color='dimgray', fontsize=8, ha='center', va='center')
        cax.text(sx * 1.10, sy * 1.10, 'S',
                 color='dimgray', fontsize=8, ha='center', va='center')
        cax.text(wx * 1.10, wy * 1.10, 'W',
                 color='dimgray', fontsize=8, ha='center', va='center')

    def _apply_fixed_axis_labels(self):
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        self.ax.set_zlabel('')
        self.ax.text2D(0.02, 0.02, 'Altitude (Up)',
                       transform=self.ax.transAxes,
                       ha='left', va='bottom',
                       fontsize=8, fontweight='bold', color='#333333')

    def _clear_previous_landing(self):
        self.land_lat = self.launch_lat
        self.land_lon = self.launch_lon
        self.r90_radius = 0.0
        self._has_sim_result = False
        self._last_sim_data = None

    _PLOT_RECT      = (0.02, 0.00, 0.96, 0.92)
    _TOP_STRIP_FRAC = 0.92

    def update_plots(self, data=None):
        self.fig.clear()
        self._compass_ax = None
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_position(list(self._PLOT_RECT))

        if not data:
            self.ax.scatter([0], [0], [0], marker='^', color='blue',
                            s=60, zorder=6, label='Launch')
            self.ax.set_xlim(-60, 60); self.ax.set_ylim(-60, 60)
            self.ax.set_zlim(0, 60)
            self._apply_fixed_axis_labels()
            self.ax.view_init(elev=self._fixed_elev, azim=self._fixed_azim)
            # Feat 5: two-column legend.
            self.ax.legend(loc='upper right',
                           bbox_to_anchor=(0.98, 0.985),
                           bbox_transform=self.fig.transFigure,
                           ncol=2, fontsize=10, framealpha=0.85)
            self._apply_safe_layout()
            self._draw_compass()
            self.canvas.draw()
            self.draw_map_elements()
            return

        x_vals   = data['x']
        y_vals   = data['y']
        z_vals   = data['z']
        r90      = data['r90']
        impact_x = data['impact_x']
        impact_y = data['impact_y']
        bf_time  = data.get('bf_time')
        bf_x     = data.get('bf_x')
        bf_y     = data.get('bf_y')
        bf_z     = data.get('bf_z')
        para_time = data.get('para_time')
        idx_para  = data.get('idx_para', -1)
        idx_bf    = data.get('idx_bf',   -1)
        wind_u_prof = data['wind_u_prof']
        wind_v_prof = data['wind_v_prof']

        alt_max  = float(np.max(z_vals)) if len(z_vals) > 0 else 100.0
        has_bf   = idx_bf   != -1 and idx_bf   < len(x_vals)
        has_para = idx_para != -1 and idx_para < len(x_vals)

        lw = 2.0
        if has_bf and has_para:
            self.ax.plot(x_vals[:idx_bf+1], y_vals[:idx_bf+1], z_vals[:idx_bf+1],
                         color='royalblue', lw=lw, label='Powered / Coast')
            self.ax.plot(x_vals[idx_bf:idx_para+1], y_vals[idx_bf:idx_para+1], z_vals[idx_bf:idx_para+1],
                         color='darkorange', lw=lw, label='Freefall (post-backfire)')
            self.ax.plot(x_vals[idx_para:], y_vals[idx_para:], z_vals[idx_para:],
                         color='deepskyblue', lw=lw, linestyle='--', label='Under Canopy')
            self.ax.scatter([x_vals[idx_para]], [y_vals[idx_para]], [z_vals[idx_para]],
                            marker='v', color='limegreen', s=60, zorder=5,
                            label='Fully Open')
        elif has_bf:
            self.ax.plot(x_vals[:idx_bf+1], y_vals[:idx_bf+1], z_vals[:idx_bf+1],
                         color='royalblue', lw=lw, label='Powered / Coast')
            self.ax.plot(x_vals[idx_bf:], y_vals[idx_bf:], z_vals[idx_bf:],
                         color='darkorange', lw=lw, label='Freefall (no chute)')
        elif has_para:
            self.ax.plot(x_vals[:idx_para+1], y_vals[:idx_para+1], z_vals[:idx_para+1],
                         color='royalblue', lw=lw, label='Freefall')
            self.ax.plot(x_vals[idx_para:], y_vals[idx_para:], z_vals[idx_para:],
                         color='deepskyblue', lw=lw, linestyle='--', label='Under Canopy')
        else:
            self.ax.plot(x_vals, y_vals, z_vals, color='royalblue', lw=lw, label='Trajectory')

        if len(z_vals) > 0:
            ap_idx = int(np.argmax(z_vals))
            ax_, ay_, az_ = x_vals[ap_idx], y_vals[ap_idx], z_vals[ap_idx]
            self.ax.plot([ax_, ax_], [ay_, ay_], [0, az_],
                         color='gray', linestyle=':', lw=1.2)
            self.ax.scatter([ax_], [ay_], [az_],
                            marker='*', color='gold', s=120, zorder=6,
                            label='Apogee')

        if bf_x is not None and bf_z is not None:
            self.ax.scatter([bf_x], [bf_y], [bf_z],
                            marker='X', color='magenta', s=80, zorder=6,
                            label='Backfire')
            self.ax.plot([bf_x, bf_x], [bf_y, bf_y], [0, bf_z],
                         color='magenta', linestyle=':', lw=1.0, alpha=0.6)

        self.ax.scatter([0], [0], [0], marker='^', color='blue', s=60, zorder=6,
                        label='Launch')

        self.ax.scatter([impact_x], [impact_y], [0],
                        marker='o', color='red', s=60, zorder=6, label='Impact')

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        _mc_scatter  = getattr(self, '_mc_scatter',  None)
        _mc_ellipse  = getattr(self, '_mc_ellipse',  None)

        # MC scatter dots (up to 100 semi-transparent orange points)
        if _mc_scatter is not None and len(_mc_scatter) > 0:
            pts  = _mc_scatter[:100]
            sc_x = [p[0] for p in pts]
            sc_y = [p[1] for p in pts]
            self.ax.scatter(sc_x, sc_y, np.zeros(len(pts)),
                            s=6, c='orange', alpha=0.4, zorder=3)

        if _mc_ellipse is not None:
            # 90% error ellipse derived from MC covariance
            _theta = np.linspace(0, 2 * math.pi, 72)
            _ca = math.cos(_mc_ellipse['angle_rad'])
            _sa = math.sin(_mc_ellipse['angle_rad'])
            _a, _b = _mc_ellipse['a'], _mc_ellipse['b']
            _cx, _cy = _mc_ellipse['cx'], _mc_ellipse['cy']
            ex = _a * np.cos(_theta) * _ca - _b * np.sin(_theta) * _sa + _cx
            ey = _a * np.cos(_theta) * _sa + _b * np.sin(_theta) * _ca + _cy
            ez = np.zeros_like(_theta)
            self.ax.plot(ex, ey, ez, color='darkorange', lw=2.0, alpha=0.85,
                         label=f'{self.landing_prob}% Error Ellipse')
            verts = [list(zip(ex, ey, ez))]
            poly  = Poly3DCollection(verts, alpha=0.10,
                                     facecolor='orange', edgecolor='none')
            self.ax.add_collection3d(poly)
        else:
            # Fallback: analytic r90 circle while MC has not completed yet
            theta = np.linspace(0, 2 * math.pi, 72)
            cx_r  = impact_x + r90 * np.cos(theta)
            cy_r  = impact_y + r90 * np.sin(theta)
            cz_r  = np.zeros_like(theta)
            self.ax.plot(cx_r, cy_r, cz_r, color='red', lw=1.5, alpha=0.6,
                         label=f'Landing Area ({self.landing_prob}%)')
            n_pts      = 60
            disc_theta = np.linspace(0, 2 * math.pi, n_pts)
            disc_x = impact_x + r90 * np.cos(disc_theta)
            disc_y = impact_y + r90 * np.sin(disc_theta)
            disc_z = np.zeros(n_pts)
            verts  = [list(zip(disc_x, disc_y, disc_z))]
            poly   = Poly3DCollection(verts, alpha=0.12,
                                      facecolor='red', edgecolor='none')
            self.ax.add_collection3d(poly)

        self.ax.plot(x_vals, y_vals, np.zeros_like(z_vals),
                     color='gray', lw=0.8, alpha=0.35, linestyle='--')

        z_keys   = [p[0] for p in wind_u_prof]
        u_vals_w = [p[1] for p in wind_u_prof]
        v_vals_w = [p[1] for p in wind_v_prof]
        arrow_len = max(r90 * 0.4, alt_max * 0.12, 3.0)
        for alt in np.linspace(0, alt_max, 6):
            u_a = np.interp(alt, z_keys, u_vals_w)
            v_a = np.interp(alt, z_keys, v_vals_w)
            spd = math.sqrt(u_a**2 + v_a**2)
            if spd < 1e-6:
                continue
            scale = arrow_len * (spd / max(
                math.sqrt(u_vals_w[-1]**2 + v_vals_w[-1]**2), 0.1))
            self.ax.quiver(0, 0, alt,
                           u_a * scale / (spd + 1e-9),
                           v_a * scale / (spd + 1e-9),
                           0,
                           color='limegreen', lw=1.2, arrow_length_ratio=0.3)
            self.ax.text(u_a * scale / (spd + 1e-9),
                         v_a * scale / (spd + 1e-9),
                         alt + alt_max * 0.02,
                         f'{spd:.1f}m/s', color='green', fontsize=7)

        self.ax.view_init(elev=self._fixed_elev, azim=self._fixed_azim)

        self._apply_fixed_axis_labels()
        self.ax.tick_params(labelsize=7)

        downrange_m = math.hypot(impact_x, impact_y)
        apogee_m    = data.get('apogee_m',
                               float(np.max(z_vals)) if len(z_vals) > 0 else 0.0)
        para_str    = f'{para_time:.2f} s' if para_time is not None else '— s'
        bf_str      = f'{bf_time:.2f} s' if bf_time is not None else '— s'
        _mc_cep     = getattr(self, '_mc_cep',     None)
        _mc_running = getattr(self, '_mc_running', False)
        if _mc_running:
            cep_str = 'computing…'
        elif _mc_cep is not None:
            cep_str = f'{_mc_cep:.1f} m'
        else:
            cep_str = '—'

        # ── Prominent landing radius banner ───────────────────────────────────
        cur_mode    = getattr(self.operation_mode_var, 'get', lambda: 'Free')()
        r_horiz     = data.get('r_horiz', downrange_m)
        hang_time   = data.get('hang_time', None)
        try:
            r_max_val = float(self.r_max_var.get())
        except Exception:
            r_max_val = None

        radius_banner = f'Pred. Landing Radius:  {self.r90_radius:.1f} m  ({self.landing_prob}%)'
        radius_color  = '#cc0000'
        self.fig.text(
            0.50, 0.985, radius_banner,
            ha='center', va='top',
            fontsize=13, fontweight='bold', color=radius_color,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.30',
                      facecolor='#fff0f0', edgecolor='#cc0000', alpha=0.92))

        # ── Score formula for competition modes ───────────────────────────────
        ph1 = getattr(self, '_phase1_result', None)
        ph1_mode  = getattr(ph1, 'mode',       None) if ph1 is not None else None
        ph1_score = getattr(ph1, 'best_score', None) if ph1 is not None else None
        score_lines = []
        if cur_mode in ('Precision Landing', 'Winged Hover') and r_max_val is not None:
            if hang_time is not None:
                score_formula = (f'r_max={r_max_val:.0f}  r={r_horiz:.1f}  '
                                 f't={hang_time:.2f}\n'
                                 f'Score = r_max - r + t = '
                                 f'{r_max_val - r_horiz + hang_time:.2f}')
                score_lines.append(score_formula)
        if ph1_score is not None and ph1_mode in ('Precision Landing', 'Winged Hover'):
            if ph1_mode == 'Winged Hover':
                score_lines.append(f'★ Phase 1 best hover time: {ph1_score:.2f} s')
            else:
                score_lines.append(f'★ Phase 1 best score:      {ph1_score:.2f}')
        if score_lines:
            self.fig.text(
                0.50, 0.935, '\n'.join(score_lines),
                ha='center', va='top',
                fontsize=9, fontweight='bold', color='#7700aa',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.28',
                          facecolor='#f5eeff', edgecolor='#9933cc', alpha=0.88))

        stats_text = (
            f'Apogee:           {apogee_m:.1f} m\n'
            f'Backfire:         {bf_str}\n'
            f'Parachute Open:   {para_str}\n'
            f'Downrange:        {downrange_m:.1f} m\n'
            f'CEP (50%):        {cep_str}'
        )
        opt_info = getattr(self, '_last_optimization_info', None)
        if opt_info:
            stats_text += (
                f'\nBest Elevation:   {opt_info["elev"]:.1f}°\n'
                f'Best Azimuth:     {opt_info["azi"]:.1f}°'
            )
        self.fig.text(
            0.02, 0.985, stats_text,
            ha='left', va='top',
            fontsize=10, fontweight='bold', color='#222222',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.35',
                      facecolor='white', edgecolor='gray', alpha=0.9))

        _me = getattr(self, '_mc_ellipse', None)
        if _me is not None:
            ext = max(_me['a'], _me['b']) + math.hypot(_me['cx'], _me['cy'])
            all_horiz = np.concatenate([x_vals, y_vals,
                                        [_me['cx'] + ext, _me['cx'] - ext,
                                         _me['cy'] + ext, _me['cy'] - ext]])
        else:
            all_horiz = np.concatenate([x_vals, y_vals,
                                        [impact_x + r90, impact_x - r90,
                                         impact_y + r90, impact_y - r90]])
        h_range = max(float(np.max(all_horiz) - np.min(all_horiz)), 1.0)
        mid_x   = float((np.max(x_vals) + np.min(x_vals)) / 2)
        mid_y   = float((np.max(y_vals) + np.min(y_vals)) / 2)
        self.ax.set_xlim(mid_x - h_range * 0.6, mid_x + h_range * 0.6)
        self.ax.set_ylim(mid_y - h_range * 0.6, mid_y + h_range * 0.6)
        self.ax.set_zlim(0, alt_max * 1.15)

        # Feat 5: 3-D profile legend in two columns per spec.
        self.ax.legend(loc='upper right',
                       bbox_to_anchor=(0.98, 0.985),
                       bbox_transform=self.fig.transFigure,
                       ncol=2, fontsize=10, framealpha=0.85)
        self._apply_safe_layout()
        self._draw_compass()
        self.canvas.draw()
        try:
            self._update_wind_subplots()
        except Exception:
            pass
        self.draw_map_elements()
        try:
            self.fit_map_bounds()
        except Exception:
            pass

    def draw_map_elements(self):
        self.map_widget.delete_all_polygon()
        self.map_widget.set_polygon(self.get_circle_coords(self.launch_lat, self.launch_lon, 2.5), fill_color="blue")
        try:
            _mode = self.operation_mode_var.get()
            _target_r_map = float(self.r_max_var.get()) if _mode != 'Free' else 50.0
        except Exception:
            _target_r_map = 50.0
        self.map_widget.set_polygon(self.get_circle_coords(self.launch_lat, self.launch_lon, _target_r_map), outline_color="blue")
        if getattr(self, '_has_sim_result', False) and self.r90_radius > 0:
            self.map_widget.set_polygon(self.get_circle_coords(self.land_lat, self.land_lon, 2.5), fill_color="red")
            self.map_widget.set_polygon(self.get_circle_coords(self.land_lat, self.land_lon, self.r90_radius), outline_color="red", border_width=2)

        # Phase 2: live 90% error ellipse
        e2 = getattr(self, '_p2_ellipse', None)
        if e2 is not None:
            color = '#00bb00' if e2['go'] else '#dd0000'
            ellipse_coords = self._get_ellipse_polygon(
                self.land_lat, self.land_lon,
                e2['cx'], e2['cy'], e2['a'], e2['b'], e2['angle_rad'])
            self.map_widget.set_polygon(
                ellipse_coords, outline_color=color, border_width=2)

        # MC KDE probability-mass contours (50% / 70% / 90%)
        kde_contours = getattr(self, '_kde_contours', None)
        if kde_contours:
            for latlons, col, bw in kde_contours:
                if len(latlons) >= 3:
                    self.map_widget.set_polygon(
                        latlons, outline_color=col, border_width=bw)

    def get_circle_coords(self, lat, lon, radius_m):
        coords = []
        for i in range(36):
            angle = math.pi * 2 * i / 36
            dx, dy = radius_m * math.cos(angle), radius_m * math.sin(angle)
            d_lat = (dy / 6378137.0) * (180 / math.pi)
            d_lon = (dx / (6378137.0 * math.cos(math.pi * lat / 180))) * (180 / math.pi)
            coords.append((lat + d_lat, lon + d_lon))
        return coords

    def fit_map_bounds(self):
        try:
            launch_ring = 50.0
            land_ring   = max(getattr(self, 'r90_radius', 0.0) or 0.0, 2.5)

            m_lat, m_lon = self._meters_per_degree(self.launch_lat)

            def _ring_extents(lat, lon, r_m):
                dlat = r_m / m_lat
                dlon = r_m / m_lon
                return (lat - dlat, lat + dlat, lon - dlon, lon + dlon)

            have_landing = (getattr(self, '_has_sim_result', False)
                            and land_ring > 0
                            and hasattr(self, 'land_lat'))

            lat_mins, lat_maxs, lon_mins, lon_maxs = [], [], [], []
            for la, lo, r in ([(self.launch_lat, self.launch_lon, launch_ring)]
                               + ([(self.land_lat, self.land_lon, land_ring)]
                                  if have_landing else [])):
                lamin, lamax, lomin, lomax = _ring_extents(la, lo, r)
                lat_mins.append(lamin); lat_maxs.append(lamax)
                lon_mins.append(lomin); lon_maxs.append(lomax)

            min_lat, max_lat = min(lat_mins), max(lat_maxs)
            min_lon, max_lon = min(lon_mins), max(lon_maxs)

            pad_lat = (max_lat - min_lat) * 0.10
            pad_lon = (max_lon - min_lon) * 0.10
            pad_lat = max(pad_lat, 5.0 / m_lat)
            pad_lon = max(pad_lon, 5.0 / m_lon)

            self.map_widget.fit_bounding_box(
                (max_lat + pad_lat, min_lon - pad_lon),
                (min_lat - pad_lat, max_lon + pad_lon),
            )
        except Exception:
            pass

    def on_parameter_edit_af(self, event=None):
        if self.af_name_label.cget("text") != "Airframe: (none selected)":
            self.af_name_label.config(text="Airframe: (none selected)")
        self._release_lock_if_active(reason_label="⭘ Unlocked (param changed)")

    def on_parameter_edit_para(self, event=None):
        if self.para_name_label.cget("text") != "Parachute: (none selected)":
            self.para_name_label.config(text="Parachute: (none selected)")
        self._release_lock_if_active(reason_label="⭘ Unlocked (param changed)")

    def _collect_airframe_dict(self):
        return {
            "mass":            float(self.mass_entry.get()),
            "cg":              float(self.cg_entry.get()),
            "length":          float(self.len_entry.get()),
            "radius":          float(self.radius_entry.get()),
            "nose_length":     float(self.nose_len_entry.get()),
            "fin_root":        float(self.fin_root_entry.get()),
            "fin_tip":         float(self.fin_tip_entry.get()),
            "fin_span":        float(self.fin_span_entry.get()),
            "fin_pos":         float(self.fin_pos_entry.get()),
            "motor_pos":       float(self.motor_pos_entry.get()),
            "motor_dry_mass":  float(self.motor_dry_mass_entry.get()),
            "backfire_delay":  float(self.backfire_delay_entry.get()),
        }

    def _collect_parachute_dict(self):
        return {
            "cd":   float(self.cd_entry.get()),
            "area": float(self.area_entry.get()),
            "lag":  float(self.lag_entry.get()),
        }

    def _apply_airframe_dict(self, af):
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

    def _apply_parachute_dict(self, pa):
        self.cd_entry.delete(0,  tk.END); self.cd_entry.insert(0,  str(pa.get("cd",   "")))
        self.area_entry.delete(0, tk.END); self.area_entry.insert(0, str(pa.get("area", "")))
        self.lag_entry.delete(0,  tk.END); self.lag_entry.insert(0,  str(pa.get("lag",  "")))

    def save_config(self):
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
            messagebox.showinfo("Saved", f"Airframe + parachute config saved.\n{base}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{e}")

    def load_config(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json")],
            title="Load Rocket Config")
        if not filepath:
            return
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            base = os.path.basename(filepath)

            af = data.get("airframe")
            pa = data.get("parachute")

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
                raise ValueError(
                    "JSON contains neither airframe nor parachute data.")
            messagebox.showinfo(
                "Loaded",
                f"{' + '.join(applied)} config loaded.\n{base}")
        except Exception as e:
            messagebox.showerror("Error", f"Load failed:\n{e}")

    def save_af_settings(self):   self.save_config()
    def load_af_settings(self):   self.load_config()
    def save_para_settings(self): self.save_config()
    def load_para_settings(self): self.load_config()

    def open_thrustcurve_web(self):
        try:
            webbrowser.open("https://www.thrustcurve.org/motors/search.html")
            messagebox.showinfo("Browser Opened", "Opened ThrustCurve.org search page.\nDownload a motor CSV (RockSim format) and load it with [Load CSV].")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open browser.\n{e}")

    def load_local_motor(self):
        filepath = filedialog.askopenfilename(
            title="Select Motor CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not filepath:
            return

        try:
            motor_name = os.path.basename(filepath).replace('.csv', '')
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
                            pass

            if not time_thrust_points:
                raise ValueError("No valid numeric data found in the file. Please verify it is a RockSim-format CSV.")

            if time_thrust_points[0][0] != 0.0:
                time_thrust_points.insert(0, [0.0, time_thrust_points[0][1]])

            burn_time = time_thrust_points[-1][0]

            thrusts = [p[1] for p in time_thrust_points]
            max_thrust = max(thrusts) if thrusts else 0.0
            total_impulse = 0.0
            for i in range(1, len(time_thrust_points)):
                t0, T0 = time_thrust_points[i - 1]
                t1, T1 = time_thrust_points[i]
                total_impulse += (T0 + T1) * 0.5 * (t1 - t0)
            avg_thrust = (total_impulse / burn_time) if burn_time > 0 else 0.0

            self.selected_motor_file = filepath
            self.thrust_data = time_thrust_points
            self.selected_motor_name = motor_name
            self.motor_burn_time = burn_time
            self.motor_avg_thrust = avg_thrust
            self.motor_max_thrust = max_thrust
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

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Pre-Optimization + Limit Margin Search
    # ══════════════════════════════════════════════════════════════════════════

    def _p1_params_at_wind(self, base_params, mu_surf):
        """Return a copy of base_params with wind speed scaled to mu_surf."""
        ratio    = mu_surf / max(base_params['surf_spd'], 1e-6)
        mu_upper = base_params['up_spd'] * ratio
        u_prof, v_prof = WindProfileBuilder.build(
            mu_surf, base_params['surf_dir'], 3.0,
            mu_upper, base_params['up_dir'], 100.0,
        )
        p = dict(base_params)
        p['wind_u_prof'] = u_prof
        p['wind_v_prof'] = v_prof
        p['surf_spd']    = mu_surf
        p['up_spd']      = mu_upper
        return p

    def _p1_mc_points(self, elev, azi, base_params, mu, sigma, n, stop_flag=None):
        """Run n Monte Carlo sims, returning list of (impact_x, impact_y) pairs."""
        rng        = random.Random()
        mu_nominal = max(base_params['surf_spd'], 1e-6)
        points     = []
        for _ in range(n):
            if stop_flag is not None and stop_flag.is_set():
                break
            surf_spd = max(0.0, rng.gauss(mu, sigma))
            ratio    = surf_spd / mu_nominal
            up_spd   = max(0.0, rng.gauss(base_params['up_spd'] * ratio, sigma * 0.5))
            u_prof, v_prof = WindProfileBuilder.build(
                surf_spd, base_params['surf_dir'], 3.0,
                up_spd,   base_params['up_dir'],   100.0,
            )
            p = dict(base_params)
            p['wind_u_prof'] = u_prof
            p['wind_v_prof'] = v_prof
            p['surf_spd']    = surf_spd
            r = self._simulate_once(elev, azi, p)
            if r['ok']:
                points.append((r['impact_x'], r['impact_y']))
        return points

    @staticmethod
    def _p1_ellipse_params(points):
        """Return (cx, cy, eigvals, eigvecs) for the 90%-confidence ellipse."""
        pts = np.array(points)
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        return cx, cy, eigvals, eigvecs

    @staticmethod
    def _p1_ellipse_breaches_circle(cx, cy, eigvals, eigvecs, R, n_pts=180):
        """True if the 90% error ellipse extends beyond the circle of radius R."""
        k = math.sqrt(4.605)          # chi²(2, 90 %)
        a = k * math.sqrt(max(float(eigvals[1]), 0.0))
        b = k * math.sqrt(max(float(eigvals[0]), 0.0))
        ang = math.atan2(float(eigvecs[1, 1]), float(eigvecs[0, 1]))
        for i in range(n_pts):
            t  = 2.0 * math.pi * i / n_pts
            xe = a * math.cos(t) * math.cos(ang) - b * math.sin(t) * math.sin(ang)
            ye = a * math.cos(t) * math.sin(ang) + b * math.sin(t) * math.cos(ang)
            if math.hypot(cx + xe, cy + ye) > R:
                return True
        return False

    # ── Phase 1 thread worker ─────────────────────────────────────────────────

    def _p1_worker(self, base_params, target_r, mode):
        q = self._p1_queue

        def prog(msg, frac):
            q.put(('p1_prog', msg, frac))

        try:
            mu_nom = base_params['surf_spd']

            # Step 1 — Grid search with mode-dependent objective + optional r_horiz filter
            elev_grid   = list(range(60, 91, 6))   # 60,66,72,78,84,90
            azi_grid    = list(range(0, 360, 15))   # 24 azimuths
            use_r_filter = (mode != 'Precision Landing')     # Precision Landing scores by -r_horiz directly
            total = len(elev_grid) * len(azi_grid)
            done, cands = 0, []
            prog(f'Step 1/5  Grid search (0/{total})', 0.0)

            for e in elev_grid:
                for a in azi_grid:
                    if self._p1_stop_flag.is_set():
                        q.put(('p1_cancel', None)); return
                    p   = self._p1_params_at_wind(base_params, mu_nom)
                    res = self._simulate_once(e, a, p)
                    done += 1
                    if res['ok']:
                        if not use_r_filter or res['r_horiz'] <= target_r:
                            score = self._p1_objective_score(res, mode)
                            cands.append((score, e, a, res))
                    prog(f'Step 1/5  Grid ({done}/{total})  e={e}° a={a}°',
                         done / total * 0.25)

            if not cands:
                q.put(('p1_error',
                       f'No trajectory satisfies r_horiz ≤ {target_r:.0f} m.\n'
                       'Check parameters (r_max, wind speed, airframe specs).'))
                return

            cands.sort(key=lambda x: -x[0])
            _, best_e, best_a, best_res = cands[0]
            best_apogee = best_res['apogee_m']
            prog(f'Step 1/5  done  best elev={best_e}° azi={best_a}°'
                 f'  apogee={best_apogee:.1f} m', 0.26)

            # Step 2 — Nominal MC: 90% error ellipse at nominal wind
            N_NOM     = 40
            sigma_nom = max(mu_nom * 0.08, 0.3)
            prog(f'Step 2/5  Nominal MC  ({N_NOM} runs, σ={sigma_nom:.2f} m/s)…', 0.28)

            pts_nom = self._p1_mc_points(
                best_e, best_a, base_params, mu_nom, sigma_nom,
                n=N_NOM, stop_flag=self._p1_stop_flag)
            if self._p1_stop_flag.is_set():
                q.put(('p1_cancel', None)); return
            if len(pts_nom) < 6:
                q.put(('p1_error', 'Nominal MC: insufficient samples (< 6). Check parameters.'))
                return

            cx_nom, cy_nom, eig_v, eig_vc = self._p1_ellipse_params(pts_nom)
            K  = math.sqrt(4.605)
            a_nom     = K * math.sqrt(max(float(eig_v[1]), 0.0))
            b_nom     = K * math.sqrt(max(float(eig_v[0]), 0.0))
            angle_rad = math.atan2(float(eig_vc[1, 1]), float(eig_vc[0, 1]))
            scale_per_sigma = (a_nom / sigma_nom) if sigma_nom > 0 else 10.0
            prog('Step 2/5  Nominal MC done', 0.42)

            # Step 3 — Landing sensitivity  d(cx,cy)/dμ  (central difference)
            prog('Step 3/5  Wind sensitivity…', 0.44)
            dmu   = max(mu_nom * 0.15, 0.5)
            p_hi  = self._p1_params_at_wind(base_params, mu_nom + dmu)
            p_lo  = self._p1_params_at_wind(base_params, max(mu_nom - dmu, 0.1))
            r_hi  = self._simulate_once(best_e, best_a, p_hi)
            r_lo  = self._simulate_once(best_e, best_a, p_lo)
            if r_hi['ok'] and r_lo['ok']:
                dcx_dmu = (r_hi['impact_x'] - r_lo['impact_x']) / (2 * dmu)
                dcy_dmu = (r_hi['impact_y'] - r_lo['impact_y']) / (2 * dmu)
            else:
                dcx_dmu = dcy_dmu = 0.0
            prog('Step 3/5  Sensitivity done', 0.50)

            # Step 4 — Binary search μ_max (deterministic, σ = 0)
            prog('Step 4/5  μ_max search…', 0.52)
            mu_lo_s, mu_hi_s = mu_nom, mu_nom * 8.0
            for _ in range(22):
                if self._p1_stop_flag.is_set():
                    q.put(('p1_cancel', None)); return
                if mu_hi_s - mu_lo_s < 0.05:
                    break
                mu_mid = (mu_lo_s + mu_hi_s) / 2.0
                p_m    = self._p1_params_at_wind(base_params, mu_mid)
                r_m    = self._simulate_once(best_e, best_a, p_m)
                if r_m['ok'] and r_m['r_horiz'] <= target_r:
                    mu_lo_s = mu_mid
                else:
                    mu_hi_s = mu_mid
            mu_max = mu_lo_s
            prog(f'Step 4/5  μ_max = {mu_max:.2f} m/s', 0.70)

            # Step 5 — Binary search σ_max (MC-based ellipse containment)
            prog('Step 5/5  σ_max search (MC)…', 0.72)
            N_SIG = 20
            sig_lo, sig_hi = 0.0, max(mu_nom * 3.0, 5.0)

            def _sigma_ok(sig):
                if self._p1_stop_flag.is_set():
                    return False
                pts = self._p1_mc_points(
                    best_e, best_a, base_params, mu_nom, sig,
                    n=N_SIG, stop_flag=self._p1_stop_flag)
                if len(pts) < 6:
                    return False
                cx_m = float(np.mean([p[0] for p in pts]))
                cy_m = float(np.mean([p[1] for p in pts]))
                _, _, ev, evc = self._p1_ellipse_params(pts)
                return not self._p1_ellipse_breaches_circle(cx_m, cy_m, ev, evc, target_r)

            if _sigma_ok(sig_hi):
                sigma_max = sig_hi
            else:
                for _ in range(15):
                    if self._p1_stop_flag.is_set():
                        q.put(('p1_cancel', None)); return
                    if sig_hi - sig_lo < 0.05:
                        break
                    sig_mid = (sig_lo + sig_hi) / 2.0
                    if _sigma_ok(sig_mid):
                        sig_lo = sig_mid
                    else:
                        sig_hi = sig_mid
                sigma_max = sig_lo

            prog(f'Step 5/5  σ_max = {sigma_max:.2f} m/s', 0.99)

            # Compute displayable best score for the completion dialog
            if mode == 'Precision Landing':
                _display_score = best_res['r_horiz']     # lower = better
            elif mode == 'Winged Hover':
                _display_score = best_res['hang_time']   # seconds
            else:
                _display_score = best_res['apogee_m']    # altitude

            result = Phase1Result(
                best_elev=float(best_e),    best_azi=float(best_a),
                apogee_m=float(best_apogee),
                nominal_cx=float(cx_nom),   nominal_cy=float(cy_nom),
                mu_nominal=float(mu_nom),
                mu_max=float(mu_max),        sigma_max=float(sigma_max),
                ellipse_a=float(a_nom),      ellipse_b=float(b_nom),
                ellipse_angle_rad=float(angle_rad),
                ellipse_scale_per_sigma=float(scale_per_sigma),
                dcx_dmu=float(dcx_dmu),     dcy_dmu=float(dcy_dmu),
                target_radius_m=float(target_r),
                best_score=float(_display_score),
                mode=mode,
            )
            prog('Phase 1 complete ✓', 1.0)
            q.put(('p1_done', result))

        except Exception as exc:
            q.put(('p1_error', f'Phase 1 worker error:\n{exc}'))

    # ── Phase 1: progress popup helpers ──────────────────────────────────────

    def _show_p1_win(self, mode, target_r):
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
                  font=("Arial", 9), wraplength=400).pack(
            anchor='w', pady=(6, 3))

        self._p1_win_bar = ttk.Progressbar(frm, mode='determinate', maximum=100)
        self._p1_win_bar.pack(fill='x', pady=(0, 8))

        ttk.Button(frm, text="Cancel",
                   command=self._stop_phase1).pack(anchor='e')

    def _close_p1_win(self):
        try:
            if self._p1_win is not None:
                self._p1_win.destroy()
        except Exception:
            pass
        self._p1_win     = None
        self._p1_win_bar = None
        self._p1_win_msg = None

    # ── Phase 1: start / poll / finish ───────────────────────────────────────

    def _start_phase1(self):
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
            messagebox.showerror('Input Error', 'Please enter a positive number for Target radius.')
            return

        # Build power-law wind profile into params before handing to thread
        u_prof, v_prof = WindProfileBuilder.build(
            params['surf_spd'], params['surf_dir'], 3.0,
            params['up_spd'],   params['up_dir'],   100.0,
        )
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
            target=self._p1_worker, args=(params, target_r, mode), daemon=True)
        self._p1_thread.start()
        self.after(200, self._poll_p1_queue)

    def _stop_phase1(self):
        self._p1_stop_flag.set()
        try:
            if self._p1_win_msg is not None:
                self._p1_win_msg.set('Cancelling…')
        except Exception:
            pass

    def _poll_p1_queue(self):
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

    def _finish_phase1(self, result, cancelled=False, error=None):
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

        # Apply best angles to launcher spinboxes
        try:
            self.elev_spin.delete(0, tk.END)
            self.elev_spin.insert(0, f'{result.best_elev:.1f}')
            self.azi_spin.delete(0, tk.END)
            self.azi_spin.insert(0, f'{result.best_azi:.1f}')
        except Exception:
            pass

        # Render the optimal trajectory immediately so the user sees results now
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

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Real-Time GO/NO-GO (1 Hz, no heavy RocketPy calls)
    # ══════════════════════════════════════════════════════════════════════════

    def _show_phase1_complete_dialog(self, result):
        """Custom completion dialog that highlights the mode-specific score."""
        dlg = tk.Toplevel(self)
        dlg.title('Phase 1 Complete')
        dlg.resizable(False, False)
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=16)
        frm.pack(fill='both', expand=True)

        # ── Score banner ─────────────────────────────────────────────────────
        mode = getattr(result, 'mode', '')
        score = getattr(result, 'best_score', None)
        if score is not None and mode:
            if mode == 'Winged Hover':
                score_text = f'Hover Time:  {score:.2f} s'
            elif mode == 'Precision Landing':
                score_text = f'Landing Radius:  {score:.1f} m'
            else:
                score_text = f'Apogee:  {score:.0f} m'
            tk.Label(frm, text=score_text,
                     font=('Arial', 18, 'bold'),
                     foreground='#b22222',
                     relief='groove', padx=10, pady=6).pack(
                fill='x', pady=(0, 10))

        # ── Details ───────────────────────────────────────────────────────────
        details = (
            f'Optimal launch angle:  elev = {result.best_elev:.1f}°'
            f'   azi = {result.best_azi:.1f}°\n'
            f'Apogee: {result.apogee_m:.0f} m\n\n'
            f'── GO/NO-GO Limits ─────────────────\n'
            f'  μ_max  (max mean wind speed):   {result.mu_max:.2f} m/s\n'
            f'  σ_max  (max wind std dev):       {result.sigma_max:.2f} m/s\n\n'
            f'Error ellipse (90%):  '
            f'a = {result.ellipse_a:.1f} m   b = {result.ellipse_b:.1f} m\n\n'
            f'Phase 2 monitoring has started.'
        )
        tk.Label(frm, text=details, justify='left',
                 font=('Consolas', 9)).pack(anchor='w')

        ttk.Button(frm, text='OK', command=dlg.destroy).pack(
            anchor='e', pady=(10, 0))
        dlg.bind('<Return>', lambda _: dlg.destroy())
        self.wait_window(dlg)

    def _start_phase2(self):
        if self._p2_after_id is not None:
            try:
                self.after_cancel(self._p2_after_id)
            except Exception:
                pass
        self._p2_after_id = self.after(1000, self._phase2_tick)

    def _phase2_tick(self):
        self._p2_after_id = None
        ph1 = self._phase1_result
        if ph1 is None:
            return

        # Current statistics from rolling 300-sample deque
        history = list(self.surf_wind_time_history)
        if not history:
            self._p2_after_id = self.after(1000, self._phase2_tick)
            return

        ws       = [h[1] for h in history]
        mu_cur   = float(np.mean(ws))
        sigma_cur = float(np.std(ws, ddof=1)) if len(ws) > 1 else 0.0

        # Estimated landing centre — linear sensitivity model
        dmu    = mu_cur - ph1.mu_nominal
        cx_cur = ph1.nominal_cx + ph1.dcx_dmu * dmu
        cy_cur = ph1.nominal_cy + ph1.dcy_dmu * dmu

        # Scale ellipse axes proportionally with σ_current
        scale  = ph1.ellipse_scale_per_sigma
        a_cur  = max(ph1.ellipse_a, scale * sigma_cur)
        ratio  = ph1.ellipse_b / max(ph1.ellipse_a, 1e-6)
        b_cur  = max(ph1.ellipse_b, scale * sigma_cur * ratio)
        ang    = ph1.ellipse_angle_rad

        # Reconstruct synthetic eigenvalues for the breach check
        K        = math.sqrt(4.605)
        ev_cur   = np.array([(b_cur / K) ** 2, (a_cur / K) ** 2])
        c, s     = math.cos(ang), math.sin(ang)
        evc_cur  = np.array([[c, -s], [s, c]])

        # Evaluate three GO/NO-GO conditions
        cond_a = mu_cur    <= ph1.mu_max
        cond_b = sigma_cur <= ph1.sigma_max
        cond_c = not self._p1_ellipse_breaches_circle(
            cx_cur, cy_cur, ev_cur, evc_cur, ph1.target_radius_m)
        go = cond_a and cond_b and cond_c

        self._update_go_nogo_ui(go, mu_cur, sigma_cur, cond_a, cond_b, cond_c, ph1)

        # Store ellipse state and refresh map overlay
        self._p2_ellipse = dict(
            cx=cx_cur, cy=cy_cur, a=a_cur, b=b_cur,
            angle_rad=ang, go=go)
        try:
            self.draw_map_elements()
        except Exception:
            pass

        self._p2_after_id = self.after(1000, self._phase2_tick)

    def _update_go_nogo_ui(self, go, mu_cur, sigma_cur,
                           cond_a, cond_b, cond_c, ph1):
        color   = '#007700' if go else '#cc0000'
        verdict = '●  GO  — Ready for Launch' if go else '●  NO-GO  — Hold'
        try:
            self.go_nogo_label.config(
                text=verdict, background=color, foreground='white')
        except Exception:
            pass

        def _m(ok):
            return '✓' if ok else '✗'

        detail = (
            f'{_m(cond_a)} A  μ = {mu_cur:.2f}  /  μ_max = {ph1.mu_max:.2f} m/s\n'
            f'{_m(cond_b)} B  σ = {sigma_cur:.2f}  /  σ_max = {ph1.sigma_max:.2f} m/s\n'
            f'{_m(cond_c)} C  Ellipse ⊂ circle  (r = {ph1.target_radius_m:.0f} m)'
        )
        try:
            self.go_nogo_detail_label.config(text=detail)
        except Exception:
            pass

    # ── Ellipse polygon coords (shared by map drawing) ────────────────────────

    def _get_ellipse_polygon(self, ref_lat, ref_lon, cx, cy, a, b, angle_rad, n=60):
        """Return list of (lat, lon) approximating the rotated ellipse."""
        coords = []
        for i in range(n):
            t  = 2.0 * math.pi * i / n
            xe = (a * math.cos(t) * math.cos(angle_rad)
                  - b * math.sin(t) * math.sin(angle_rad))
            ye = (a * math.cos(t) * math.sin(angle_rad)
                  + b * math.sin(t) * math.cos(angle_rad))
            lat, lon = self._offset_to_latlon(ref_lat, ref_lon,
                                              cx + xe, cy + ye)
            coords.append((lat, lon))
        return coords

    def create_data_section(self):
        outer = ttk.Frame(self)
        outer.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        self._params_canvas = tk.Canvas(
            outer, borderwidth=0, highlightthickness=0)
        self._params_canvas.grid(row=0, column=0, sticky="nsew")
        vbar = ttk.Scrollbar(outer, orient="vertical",
                             command=self._params_canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")
        self._params_canvas.configure(yscrollcommand=vbar.set)

        frame = ttk.Frame(self._params_canvas, padding=(4, 4))
        self._params_inner = frame
        self._params_window = self._params_canvas.create_window(
            (0, 0), window=frame, anchor="nw")
        frame.columnconfigure(0, weight=1)

        def _on_inner_configure(event):
            bbox = self._params_canvas.bbox("all")
            if bbox:
                self._params_canvas.configure(
                    scrollregion=(0, 0, bbox[2], bbox[3]))
            else:
                self._params_canvas.configure(scrollregion=(0, 0, 0, 0))
        frame.bind("<Configure>", _on_inner_configure)

        def _on_canvas_configure(event):
            self._params_canvas.itemconfigure(
                self._params_window, width=event.width)
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
        self._params_canvas.bind(
            "<Enter>", lambda e: self._params_canvas.focus_set())

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

        mbf = ttk.Frame(frame); mbf.grid(row=1, column=0, sticky="ew", pady=(0, 3))
        mbf.columnconfigure(0, weight=1); mbf.columnconfigure(1, weight=1)
        ttk.Button(mbf, text="[ThrustCurve Web]", command=self.open_thrustcurve_web).grid(
            row=0, column=0, sticky="ew", padx=(0, 1))
        ttk.Button(mbf, text="[Load CSV]", command=self.load_local_motor).grid(
            row=0, column=1, sticky="ew", padx=(1, 0))
        ttk.Label(mbf, text="Backfire Delay (s):", font=("Arial", 8)).grid(
            row=1, column=0, sticky="w", padx=(4, 2), pady=(3, 1))
        self.backfire_delay_entry = ttk.Entry(mbf, width=7, font=("Arial", 8))
        self.backfire_delay_entry.insert(0, "0.0")
        self.backfire_delay_entry.grid(row=1, column=1, sticky="e", padx=(2, 4), pady=(3, 1))
        self.backfire_delay_entry.bind("<KeyRelease>", self.on_parameter_edit_af)

        ttk.Separator(frame, orient="horizontal").grid(row=2, column=0, sticky="ew", pady=3)

        # ── Airframe ──────────────────────────────────────────────────────────
        self.af_name_label = ttk.Label(frame, text="Airframe: (none selected)", font=("Arial", 8, "bold"))
        self.af_name_label.grid(row=3, column=0, sticky="w")
        self.af_name_label.grid_remove()

        af_lf = ttk.LabelFrame(frame, text="Airframe", padding=(2, 2))
        af_lf.grid(row=4, column=0, sticky="ew", pady=(1, 2))
        af_lf.columnconfigure(0, weight=1); af_lf.columnconfigure(1, weight=0)

        self.mass_entry          = param_row(af_lf, "Dry Mass (kg)",       0)
        self.cg_entry            = param_row(af_lf, "CG from Nose (m)",    1)
        self.len_entry           = param_row(af_lf, "Length (m)",          2)
        self.radius_entry        = param_row(af_lf, "Radius (m)",          3)

        aero_lf = ttk.LabelFrame(frame, text="Aero & Motor", padding=(2, 2))
        aero_lf.grid(row=5, column=0, sticky="ew", pady=(1, 2))
        aero_lf.columnconfigure(0, weight=1); aero_lf.columnconfigure(1, weight=0)

        self.nose_len_entry      = param_row(aero_lf, "Nose Length (m)",      0)
        self.fin_root_entry      = param_row(aero_lf, "Fin Root (m)",         1)
        self.fin_tip_entry       = param_row(aero_lf, "Fin Tip (m)",          2)
        self.fin_span_entry      = param_row(aero_lf, "Fin Span (m)",         3)
        self.fin_pos_entry       = param_row(aero_lf, "Fin Pos fr Nose (m)",  4)
        self.motor_pos_entry     = param_row(aero_lf, "Motor Pos fr Nose (m)",5)
        self.motor_dry_mass_entry= param_row(aero_lf, "Motor Dry Mass (kg)",  6)

        af_entries = [self.mass_entry, self.cg_entry, self.len_entry, self.radius_entry,
                      self.nose_len_entry, self.fin_root_entry, self.fin_tip_entry,
                      self.fin_span_entry, self.fin_pos_entry, self.motor_pos_entry,
                      self.motor_dry_mass_entry]
        for e in af_entries:
            e.bind("<KeyRelease>", self.on_parameter_edit_af)

        ttk.Separator(frame, orient="horizontal").grid(row=6, column=0, sticky="ew", pady=3)

        # ── Parachute ─────────────────────────────────────────────────────────
        self.para_name_label = ttk.Label(frame, text="Parachute: (none selected)", font=("Arial", 8, "bold"))
        self.para_name_label.grid(row=7, column=0, sticky="w")
        self.para_name_label.grid_remove()

        para_lf = ttk.LabelFrame(frame, text="Parachute", padding=(2, 2))
        para_lf.grid(row=8, column=0, sticky="ew", pady=(1, 2))
        para_lf.columnconfigure(0, weight=1); para_lf.columnconfigure(1, weight=0)

        self.cd_entry   = param_row(para_lf, "Cd",       0)
        self.area_entry = param_row(para_lf, "Area (m²)",1)
        self.lag_entry  = param_row(para_lf, "Lag (s)",  2)

        self.cd_entry.bind("<KeyRelease>",   self.on_parameter_edit_para)
        self.area_entry.bind("<KeyRelease>", self.on_parameter_edit_para)
        self.lag_entry.bind("<KeyRelease>",  self.on_parameter_edit_para)

        para_btn_f = ttk.Frame(frame); para_btn_f.grid(row=9, column=0, sticky="ew", pady=(0, 2))
        para_btn_f.columnconfigure(0, weight=1); para_btn_f.columnconfigure(1, weight=1)
        ttk.Button(para_btn_f, text="Load Rocket Config",
                   command=self.load_config).grid(
            row=0, column=0, sticky="ew", padx=(0, 1))
        ttk.Button(para_btn_f, text="Save Rocket Config",
                   command=self.save_config).grid(
            row=0, column=1, sticky="ew", padx=(1, 0))

        ttk.Separator(frame, orient="horizontal").grid(row=10, column=0, sticky="ew", pady=3)

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
        self.elev_spin = ttk.Spinbox(launch_lf, from_=0, to=90, width=6, font=("Arial", 8))
        self.elev_spin.set("85")
        self.elev_spin.grid(row=3, column=1, sticky="e", padx=(2, 4), pady=1)

        ttk.Label(launch_lf, text="Azimuth:", font=("Arial", 8)).grid(
            row=4, column=0, sticky="w", padx=(4, 2), pady=1)
        self.azi_spin = ttk.Spinbox(launch_lf, from_=0, to=360, width=6, font=("Arial", 8))
        self.azi_spin.set("0")
        self.azi_spin.grid(row=4, column=1, sticky="e", padx=(2, 4), pady=1)

        for _w in (self.lat_entry, self.lon_entry, self.rail_entry,
                   self.elev_spin, self.azi_spin):
            _w.bind("<KeyRelease>", _unlock_on_edit)

        loc_btn_f = ttk.Frame(frame); loc_btn_f.grid(row=13, column=0, sticky="ew", pady=(0, 2))
        loc_btn_f.columnconfigure(0, weight=1); loc_btn_f.columnconfigure(1, weight=1)
        ttk.Button(loc_btn_f, text="Get Location (IP)",
                   command=lambda: self.get_current_location(manual=True)).grid(
            row=0, column=0, sticky="ew", padx=(0, 1))
        ttk.Button(loc_btn_f, text="Update Map",
                   command=self.update_map_center).grid(
            row=0, column=1, sticky="ew", padx=(1, 0))

        ttk.Separator(frame, orient="horizontal").grid(row=14, column=0, sticky="ew", pady=3)

        # ── Operation Mode ────────────────────────────────────────────────────
        mode_lf = ttk.LabelFrame(frame, text="Operation Mode", padding=(4, 4))
        mode_lf.grid(row=15, column=0, sticky="ew", pady=(1, 2))
        mode_lf.columnconfigure(0, weight=1)
        mode_lf.columnconfigure(1, weight=0)

        self.mode_combo = ttk.Combobox(
            mode_lf, textvariable=self.operation_mode_var,
            values=list(self.OPERATION_MODES),
            state="readonly", font=("Arial", 9))
        self.mode_combo.grid(row=0, column=0, columnspan=2,
                             sticky="ew", padx=2, pady=(0, 4))
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        self.rmax_label = ttk.Label(mode_lf, text="Target radius r_max (m):",
                                    font=("Arial", 8))
        self.rmax_label.grid(row=1, column=0, sticky="w",
                             padx=(4, 2), pady=1)
        self.rmax_entry = ttk.Entry(mode_lf, textvariable=self.r_max_var,
                                    width=8, font=("Arial", 8))
        self.rmax_entry.grid(row=1, column=1, sticky="e",
                             padx=(2, 4), pady=1)
        self.rmax_label.grid_remove()
        self.rmax_entry.grid_remove()

        # ── Main Action Button (state machine: Free→single sim, others→Phase 1) ─
        self.main_action_btn = ttk.Button(
            frame, text='🚀 Run Single Simulation',
            command=self._render_current_params)
        self.main_action_btn.grid(row=16, column=0, sticky="ew",
                                  ipady=4, pady=(2, 4))
        res_f = ttk.Frame(frame); res_f.grid(row=17, column=0, sticky="ew")
        res_f.columnconfigure(0, weight=1); res_f.columnconfigure(1, weight=1)
        self.apogee_label   = ttk.Label(res_f, text="Apogee: -- m",
                                        font=("Arial", 9, "bold"))
        self.apogee_label.grid(row=0, column=0, sticky="w", padx=4)
        self.velocity_label = ttk.Label(res_f, text="Impact: -- m/s",
                                        font=("Arial", 9, "bold"))
        self.velocity_label.grid(row=0, column=1, sticky="e", padx=4)

        ttk.Separator(frame, orient="horizontal").grid(row=18, column=0, sticky="ew", pady=3)

        # ── Lock & Monitor + Settings row ─────────────────────────────────────
        lock_f = ttk.Frame(frame)
        lock_f.grid(row=19, column=0, sticky="ew", pady=(0, 2))
        lock_f.columnconfigure(0, weight=1)
        lock_f.columnconfigure(1, weight=0)

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

        ttk.Separator(frame, orient="horizontal").grid(row=20, column=0, sticky="ew", pady=3)

        # ── Phase 1: Pre-Calculation ───────────────────────────────────────────
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
        self.p1_result_label.grid(
            row=1, column=0, sticky="w", padx=4, pady=(0, 2))

        # ── Phase 2: GO/NO-GO display ─────────────────────────────────────────
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

    def create_profile_section(self):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        frame = ttk.Frame(self, padding=10, relief="solid", borderwidth=1)
        frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        frame.rowconfigure(0, weight=3)
        frame.rowconfigure(1, weight=0)
        frame.rowconfigure(2, weight=0)
        frame.rowconfigure(3, weight=1)
        frame.columnconfigure(0, weight=1)

        self.fig = plt.figure(figsize=(6.4, 5.2), dpi=100)
        self.ax  = self.fig.add_subplot(111, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        rot_bar = ttk.Frame(frame)
        rot_bar.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        rot_bar.columnconfigure(1, weight=1)

        ttk.Label(rot_bar, text="↻ Rotate:", font=("Arial", 8)).grid(
            row=0, column=0, sticky="w", padx=(2, 4))

        init_azim_ui = max(0.0, min(90.0, float(self._fixed_azim)))
        self._fixed_azim = init_azim_ui
        self.azim_var = tk.DoubleVar(value=init_azim_ui)
        self.azim_slider = ttk.Scale(
            rot_bar, from_=0, to=90, orient="horizontal",
            variable=self.azim_var, command=self._on_azim_slider
        )
        self.azim_slider.grid(row=0, column=1, sticky="ew", padx=2)

        self.azim_label = ttk.Label(
            rot_bar, text=f"{self._fixed_azim:+.0f}°",
            font=("Arial", 8), width=6, anchor="e"
        )
        self.azim_label.grid(row=0, column=2, sticky="e", padx=(4, 2))

        ttk.Button(rot_bar, text="Reset", width=6,
                   command=self._reset_azim).grid(row=0, column=3, padx=(4, 2))

        try:
            self.canvas.get_tk_widget().bind(
                "<MouseWheel>", self._on_wheel_rotate_azim)
            self.canvas.get_tk_widget().bind(
                "<Button-4>",
                lambda e: self._on_wheel_rotate_azim(e, delta_override=+120))
            self.canvas.get_tk_widget().bind(
                "<Button-5>",
                lambda e: self._on_wheel_rotate_azim(e, delta_override=-120))
        except Exception:
            pass

        self.canvas.mpl_connect('button_press_event',   self._on_canvas_press)
        self.canvas.mpl_connect('motion_notify_event',  self._on_canvas_motion)
        self.canvas.mpl_connect('button_release_event', self._on_canvas_release)
        self.canvas.mpl_connect('draw_event',           self._on_view_changed)

        # ── Bottom wind figure: time-series + compass (Feat 1+2+3) ────────────
        self.wind_fig = plt.figure(figsize=(6.4, 2.0), dpi=100)
        gs = self.wind_fig.add_gridspec(
            1, 3, width_ratios=[2.4, 1.8, 1.0], wspace=0.48)
        self.wind_ax_spd     = self.wind_fig.add_subplot(gs[0, 0])
        self.wind_ax_profile = self.wind_fig.add_subplot(gs[0, 1])
        self.wind_ax_compass = self.wind_fig.add_subplot(gs[0, 2], projection='polar')
        self.wind_fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.22)

        # Realtime wind label (Feat 4: numerical-only display).
        rt_bar = ttk.Frame(frame)
        rt_bar.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        rt_bar.columnconfigure(0, weight=1)
        self.realtime_wind_label = ttk.Label(
            rt_bar,
            text="Surface: -- m/s  @ --°   |   Upper: -- m/s @ --°",
            font=("Arial", 10, "bold"),
            foreground="#1a237e",
            background="#fffde7",
            anchor="center",
            padding=(6, 2),
            relief="groove",
        )
        self.realtime_wind_label.grid(row=0, column=0, sticky="ew")

        self.wind_canvas = FigureCanvasTkAgg(self.wind_fig, master=frame)
        self.wind_canvas.get_tk_widget().grid(row=3, column=0, sticky="nsew",
                                              pady=(4, 0))

        self._update_wind_subplots()
        self._update_realtime_wind_label()

    def _read_current_wind(self):
        def _get(var_name, default):
            v = getattr(self, var_name, None)
            if v is None:
                return default
            try:
                return float(v.get())
            except Exception:
                return default
        hist = getattr(self, 'surf_wind_history', None) or []
        if hist:
            surf_spd = float(hist[-1])
        else:
            surf_spd = self._sim_base_wind
        surf_dir = _get('surf_dir_var', 0.0)
        up_spd   = _get('up_spd_var',   0.0)
        up_dir   = _get('up_dir_var',   0.0)
        return surf_spd, surf_dir, up_spd, up_dir

    def _update_wind_subplots(self):
        """Feat 1+2+3+4: time-series wind speed graph (with 10-s moving
        average horizontal line) on the left, real-time wind-direction
        compass on the right.

        • Horizontal axis = elapsed seconds since program start
        • Vertical axis   = wind speed (m/s)
        • Red dashed ``axhline`` overlays the past-10-second moving average
        • Compass arrows reflect the *current* surface + upper directions
        • The redundant "Now:" prefix has been removed from the overlay tag
        """
        try:
            import numpy as np
        except Exception:
            return

        spd_ax = getattr(self, 'wind_ax_spd', None)
        cmp_ax = getattr(self, 'wind_ax_compass', None)
        if spd_ax is None or cmp_ax is None:
            return

        surf_spd, surf_dir, up_spd, up_dir = self._read_current_wind()

        # ── Left: Time-series wind speed (Feat 1 + 2 + 3) ────────────────────
        spd_ax.clear()
        spd_ax.set_title('Wind Speed (Time Series)', fontsize=10)
        spd_ax.set_xlabel('Time (s ago)', fontsize=9)
        spd_ax.set_ylabel('Wind Speed (m/s)', fontsize=9)
        spd_ax.tick_params(labelsize=8)
        spd_ax.grid(True, alpha=0.3)

        history = list(getattr(self, 'surf_wind_time_history', []) or [])
        if history:
            t_latest = history[-1][0]
            # Convert to "seconds ago": 0 = now, negative = past
            ts_rel = [h[0] - t_latest for h in history]
            ws = [h[1] for h in history]
            spd_ax.plot(ts_rel, ws,
                        linestyle='-', linewidth=1.6,
                        marker='.', markersize=3,
                        color='#1f77b4', label='Surface')

            # 10-second moving average as a horizontal red line.
            avg_10s = self._wind_avg_recent(window_sec=10.0)
            spd_ax.axhline(y=avg_10s, color='red', linestyle='--',
                           linewidth=1.4,
                           label=f'10s Avg: {avg_10s:.1f} m/s')

            # Upper wind speed as a horizontal green line.
            spd_ax.axhline(y=up_spd, color='green', linestyle=':',
                           linewidth=1.4,
                           label=f'Upper: {up_spd:.1f} m/s')

            # x-axis: -60 (past) on the left, 0 (now) on the right.
            spd_ax.set_xlim(-60.0, 1.0)

            y_max = max(max(ws), avg_10s, up_spd, 1.0) * 1.25
            spd_ax.set_ylim(0, y_max)
            spd_ax.legend(fontsize=8, loc='upper left')
        else:
            spd_ax.text(0.5, 0.5, 'Waiting for wind data...',
                        transform=spd_ax.transAxes,
                        ha='center', va='center', color='gray', fontsize=9)
            spd_ax.set_xlim(-60.0, 1.0)
            spd_ax.set_ylim(0, 10)

        # ── Middle: vertical wind-profile spaghetti ──────────────────────────
        prof_ax = getattr(self, 'wind_ax_profile', None)
        if prof_ax is not None:
            prof_ax.clear()
            prof_ax.set_title('Wind Profile', fontsize=9)
            prof_ax.set_xlabel('Speed (m/s)', fontsize=8)
            prof_ax.set_ylabel('Alt (m)', fontsize=8)
            prof_ax.tick_params(labelsize=7)
            prof_ax.grid(True, alpha=0.3)
            ALT_CAP = 500  # show up to 500 m
            mc_profs = getattr(self, '_mc_wind_profiles', None)
            if mc_profs:
                # thin spaghetti lines (cap at 80 for performance)
                for sp in mc_profs[:80]:
                    az = [z for z, _ in sp if z <= ALT_CAP]
                    as_ = [s for z, s in sp if z <= ALT_CAP]
                    if az:
                        prof_ax.plot(as_, az, color='#888888',
                                     lw=0.4, alpha=0.25)
                # bold mean profile
                from collections import defaultdict
                agg = defaultdict(list)
                for sp in mc_profs:
                    for z, s in sp:
                        if z <= ALT_CAP:
                            agg[z].append(s)
                sorted_z = sorted(agg.keys())
                mean_s   = [float(np.mean(agg[z])) for z in sorted_z]
                prof_ax.plot(mean_s, sorted_z,
                             color='#1f77b4', lw=2.2, label='Mean')
                prof_ax.legend(fontsize=7, loc='upper right')
            else:
                # nominal profile from current wind settings
                _u, _v = WindProfileBuilder.build(
                    surf_spd, surf_dir, 3.0, up_spd, up_dir, 100.0)
                _az  = [z for z, _ in _u if z <= ALT_CAP]
                _as  = [math.sqrt(u**2 + v**2)
                        for (z, u), (_, v) in zip(_u, _v) if z <= ALT_CAP]
                if _az:
                    prof_ax.plot(_as, _az, color='#1f77b4', lw=2.0)
            spd_vals = [s for sp in (mc_profs or []) for z, s in sp if z <= ALT_CAP]
            x_max = max(max(spd_vals, default=up_spd), up_spd, surf_spd, 1.0) * 1.2
            prof_ax.set_xlim(0, x_max)
            prof_ax.set_ylim(0, ALT_CAP)

        # ── Right: real-time wind-direction compass (Feat 2) ─────────────────
        cmp_ax.clear()
        cmp_ax.set_title('Wind From', fontsize=10, pad=8)
        cmp_ax.set_theta_zero_location('N')
        cmp_ax.set_theta_direction(-1)
        cmp_ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        cmp_ax.set_xticklabels(['N', 'E', 'S', 'W'], fontsize=9)
        cmp_ax.set_yticklabels([])
        cmp_ax.set_ylim(0, 1.0)
        cmp_ax.grid(True, alpha=0.3)

        def _draw_arrow(deg_from, color, label):
            try:
                theta = np.deg2rad(float(deg_from))
            except Exception:
                return
            cmp_ax.annotate(
                '', xy=(theta, 0.95), xytext=(theta, 0.0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0),
            )
            cmp_ax.text(theta, 1.12, label, color=color,
                        fontsize=8, ha='center', va='center')

        _draw_arrow(up_dir,   '#1f77b4', 'Upper')
        _draw_arrow(surf_dir, '#ff7f0e', 'Surface')

        try:
            self.wind_canvas.draw_idle()
        except Exception:
            pass
        try:
            self._update_realtime_wind_label()
        except Exception:
            pass

    def create_map_section(self):
        frame = ttk.Frame(self, padding=10, relief="solid", borderwidth=1)
        frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        ctrl_frame = ttk.Frame(frame)
        ctrl_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(ctrl_frame, text="Map View", font=("Arial", 10, "bold")).pack(side="left")
        ttk.Button(ctrl_frame, text="[Center Map]", command=self.fit_map_bounds).pack(side="right")

        self.map_widget = tkintermapview.TkinterMapView(frame)
        self.map_widget.pack(fill="both", expand=True)
        self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga")
        self.map_widget.set_position(self.launch_lat, self.launch_lon)

    def update_map_center(self):
        try:
            self.launch_lat = float(self.lat_entry.get())
            self.launch_lon = float(self.lon_entry.get())
            try:
                self.map_widget.set_position(self.launch_lat, self.launch_lon)
            except Exception:
                pass
            self._clear_previous_landing()
            self.update_plots()
            self.fit_map_bounds()
        except ValueError:
            pass

if __name__ == "__main__":
    app = KazamidoriUI()
    app.mainloop()