"""
ui/plot_view.py
Matplotlib rendering controller — 3-D trajectory plot and wind sub-plots.

Owns all figure/canvas objects and azimuth-rotation state.
Receives pre-computed data from AppWindow; no simulation or optimisation here.

Public API
----------
PlotView(parent_frame)
    Build the centre-column panel (3-D plot, rotation bar, wind strip).

PlotView.update_3d(data, *, mc_scatter, mc_ellipse, ...) -> None
    Re-render the 3-D trajectory and all overlays.

PlotView.update_wind(surf_wind_time_history, surf_dir, up_spd, up_dir,
                     mc_wind_profiles, wind_avg_recent) -> None
    Refresh the three wind sub-plots.

PlotView.update_realtime_wind_label(surf_spd, surf_dir, up_spd, up_dir, gust)
    Update the yellow wind info banner.

PlotView.set_azim(azim, source) / PlotView.reset_azim()
    Programmatically set or reset the 3-D view azimuth.

PlotView.azim / PlotView.elev  (read-only properties)
    Current 3-D view angles.

PlotView.canvas / PlotView.wind_canvas
    FigureCanvasTkAgg references (for additional event binding if needed).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

from core.optimization import build_wind_profile


class PlotView:
    """Manages all matplotlib rendering for the centre column."""

    _PLOT_RECT = (0.02, 0.00, 0.96, 0.92)
    _CHI2_2DOF = {50: 1.386, 68: 2.296, 80: 3.219, 85: 3.794,
                  90: 4.605, 95: 5.991, 99: 9.210}

    def __init__(self, parent: tk.Widget) -> None:
        self._fixed_azim     = 45.0
        self._fixed_elev     = 25
        self._azim_updating  = False
        self._rot_start_x   : Optional[float] = None
        self._rot_start_azim: Optional[float] = None
        self._compass_ax     = None
        self._build(parent)

    # ── Construction ──────────────────────────────────────────────────────────

    def _build(self, parent: tk.Widget) -> None:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        frame = ttk.Frame(parent, padding=10, relief="solid", borderwidth=1)
        frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        frame.rowconfigure(0, weight=3)
        frame.rowconfigure(1, weight=0)
        frame.rowconfigure(2, weight=0)
        frame.rowconfigure(3, weight=1)
        frame.columnconfigure(0, weight=1)
        self._frame = frame

        # 3-D trajectory figure
        self.fig = plt.figure(figsize=(6.4, 5.2), dpi=100)
        self.ax  = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Rotation bar
        rot_bar = ttk.Frame(frame)
        rot_bar.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        rot_bar.columnconfigure(1, weight=1)

        ttk.Label(rot_bar, text="↻ Rotate:", font=("Arial", 8)).grid(
            row=0, column=0, sticky="w", padx=(2, 4))

        init_azim = max(0.0, min(90.0, self._fixed_azim))
        self._fixed_azim = init_azim
        self.azim_var = tk.DoubleVar(value=init_azim)
        self.azim_slider = ttk.Scale(
            rot_bar, from_=0, to=90, orient="horizontal",
            variable=self.azim_var, command=self._on_azim_slider)
        self.azim_slider.grid(row=0, column=1, sticky="ew", padx=2)

        self.azim_label = ttk.Label(
            rot_bar, text=f"{init_azim:+.0f}°",
            font=("Arial", 8), width=6, anchor="e")
        self.azim_label.grid(row=0, column=2, sticky="e", padx=(4, 2))

        ttk.Button(rot_bar, text="Reset", width=6,
                   command=self.reset_azim).grid(row=0, column=3, padx=(4, 2))

        # Mouse-wheel + drag-to-rotate
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

        # Wind figure: time-series + profile + compass
        self.wind_fig = plt.figure(figsize=(6.4, 2.0), dpi=100)
        gs = self.wind_fig.add_gridspec(
            1, 3, width_ratios=[2.4, 1.8, 1.0], wspace=0.48)
        self.wind_ax_spd     = self.wind_fig.add_subplot(gs[0, 0])
        self.wind_ax_profile = self.wind_fig.add_subplot(gs[0, 1])
        self.wind_ax_compass = self.wind_fig.add_subplot(gs[0, 2], projection='polar')
        self.wind_fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.22)

        # Realtime wind banner
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
        self.wind_canvas.get_tk_widget().grid(
            row=3, column=0, sticky="nsew", pady=(4, 0))

    # ── Azimuth properties ────────────────────────────────────────────────────

    @property
    def azim(self) -> float:
        return self._fixed_azim

    @property
    def elev(self) -> float:
        return self._fixed_elev

    # ── Azimuth public API ────────────────────────────────────────────────────

    def set_azim(self, azim: float, source: str = "code") -> None:
        if self._azim_updating:
            return
        self._azim_updating = True
        try:
            a = float(azim)
            a = ((a + 180.0) % 360.0) - 180.0
            a = max(0.0, min(90.0, a))
            self._fixed_azim = a

            try:
                self.azim_label.config(text=f"{a:+.0f}°")
            except Exception:
                pass

            if source != "slider":
                try:
                    if abs(self.azim_var.get() - a) > 0.5:
                        self.azim_var.set(a)
                except tk.TclError:
                    pass

            try:
                self.ax.view_init(elev=self._fixed_elev, azim=a)
                if self._compass_ax is not None:
                    self.draw_compass()
                self.canvas.draw_idle()
            except Exception:
                pass
        finally:
            self._azim_updating = False

    def reset_azim(self) -> None:
        self.set_azim(45.0, source="code")

    # ── Internal azimuth event handlers ──────────────────────────────────────

    def _on_azim_slider(self, value: Any) -> None:
        self.set_azim(value, source="slider")

    def _on_wheel_rotate_azim(self, event: Any,
                               delta_override: Optional[int] = None) -> str:
        d = delta_override if delta_override is not None else getattr(event, 'delta', 0)
        if d == 0:
            return "break"
        step = 5.0 if d > 0 else -5.0
        self.set_azim(self._fixed_azim + step, source="code")
        return "break"

    def _on_canvas_press(self, event: Any) -> None:
        if event.inaxes is self.ax and event.button == 1:
            self._rot_start_x    = event.x
            self._rot_start_azim = self._fixed_azim

    def _on_canvas_motion(self, event: Any) -> None:
        if self._rot_start_x is None or event.button != 1:
            return
        dx = event.x - self._rot_start_x
        self.set_azim(self._rot_start_azim - dx * 0.4, source="drag")

    def _on_canvas_release(self, event: Any) -> None:
        self._rot_start_x = None

    def _on_view_changed(self, event: Any = None) -> None:
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

        self._azim_updating = True
        try:
            try:
                self.azim_var.set(self._fixed_azim)
            except tk.TclError:
                pass
            try:
                self.azim_label.config(text=f"{self._fixed_azim:+.0f}°")
            except Exception:
                pass
        finally:
            self._azim_updating = False

        self.draw_compass()
        self.canvas.draw_idle()

    # ── 3-D trajectory rendering ──────────────────────────────────────────────

    def _chi2_scale(self, prob_pct: int) -> float:
        val = self._CHI2_2DOF.get(int(prob_pct), 4.605)
        return math.sqrt(val)

    def update_3d(
        self,
        data: Optional[dict],
        *,
        mc_scatter=None,
        mc_ellipse=None,
        mc_cep=None,
        mc_running: bool = False,
        r90_radius: float = 0.0,
        landing_prob: int = 90,
        phase1_result=None,
        last_opt_info=None,
        operation_mode: str = "Free",
        r_max_val: Optional[float] = None,
    ) -> None:
        from mpl_toolkits.mplot3d import Axes3D       # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.fig.clear()
        self._compass_ax = None
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_position(list(self._PLOT_RECT))

        # ── Empty state ───────────────────────────────────────────────────────
        if not data:
            self.ax.scatter([0], [0], [0], marker='^', color='blue',
                            s=60, zorder=6, label='Launch')
            self.ax.set_xlim(-60, 60)
            self.ax.set_ylim(-60, 60)
            self.ax.set_zlim(0, 60)
            self._apply_fixed_axis_labels()
            self.ax.view_init(elev=self._fixed_elev, azim=self._fixed_azim)
            self.ax.legend(loc='upper right',
                           bbox_to_anchor=(0.98, 0.985),
                           bbox_transform=self.fig.transFigure,
                           ncol=2, fontsize=10, framealpha=0.85)
            self._apply_safe_layout()
            self.draw_compass()
            self.canvas.draw()
            return

        # ── Unpack data ───────────────────────────────────────────────────────
        x_vals      = data['x']
        y_vals      = data['y']
        z_vals      = data['z']
        r90         = data['r90']
        impact_x    = data['impact_x']
        impact_y    = data['impact_y']
        bf_time     = data.get('bf_time')
        bf_x        = data.get('bf_x')
        bf_y        = data.get('bf_y')
        bf_z        = data.get('bf_z')
        para_time   = data.get('para_time')
        idx_para    = data.get('idx_para', -1)
        idx_bf      = data.get('idx_bf',   -1)
        wind_u_prof = data['wind_u_prof']
        wind_v_prof = data['wind_v_prof']

        alt_max  = float(np.max(z_vals)) if len(z_vals) > 0 else 100.0
        has_bf   = idx_bf   != -1 and idx_bf   < len(x_vals)
        has_para = idx_para != -1 and idx_para < len(x_vals)

        # ── Trajectory segments ───────────────────────────────────────────────
        lw = 2.0
        if has_bf and has_para:
            self.ax.plot(x_vals[:idx_bf+1], y_vals[:idx_bf+1], z_vals[:idx_bf+1],
                         color='royalblue', lw=lw, label='Powered / Coast')
            self.ax.plot(x_vals[idx_bf:idx_para+1],
                         y_vals[idx_bf:idx_para+1],
                         z_vals[idx_bf:idx_para+1],
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
            self.ax.plot(x_vals, y_vals, z_vals,
                         color='royalblue', lw=lw, label='Trajectory')

        # Apogee
        if len(z_vals) > 0:
            ap_idx = int(np.argmax(z_vals))
            ax_, ay_, az_ = x_vals[ap_idx], y_vals[ap_idx], z_vals[ap_idx]
            self.ax.plot([ax_, ax_], [ay_, ay_], [0, az_],
                         color='gray', linestyle=':', lw=1.2)
            self.ax.scatter([ax_], [ay_], [az_],
                            marker='*', color='gold', s=120, zorder=6, label='Apogee')

        # Backfire
        if bf_x is not None and bf_z is not None:
            self.ax.scatter([bf_x], [bf_y], [bf_z],
                            marker='X', color='magenta', s=80, zorder=6,
                            label='Backfire')
            self.ax.plot([bf_x, bf_x], [bf_y, bf_y], [0, bf_z],
                         color='magenta', linestyle=':', lw=1.0, alpha=0.6)

        self.ax.scatter([0], [0], [0],
                        marker='^', color='blue', s=60, zorder=6, label='Launch')
        self.ax.scatter([impact_x], [impact_y], [0],
                        marker='o', color='red', s=60, zorder=6, label='Impact')

        # ── MC scatter + error ellipse (or analytic circle) ───────────────────
        if mc_scatter is not None and len(mc_scatter) > 0:
            pts  = mc_scatter[:100]
            sc_x = [p[0] for p in pts]
            sc_y = [p[1] for p in pts]
            self.ax.scatter(sc_x, sc_y, np.zeros(len(pts)),
                            s=6, c='orange', alpha=0.4, zorder=3)

        if mc_ellipse is not None:
            _theta = np.linspace(0, 2 * math.pi, 72)
            _ca    = math.cos(mc_ellipse['angle_rad'])
            _sa    = math.sin(mc_ellipse['angle_rad'])
            _a, _b = mc_ellipse['a'], mc_ellipse['b']
            _cx, _cy = mc_ellipse['cx'], mc_ellipse['cy']
            ex = _a * np.cos(_theta) * _ca - _b * np.sin(_theta) * _sa + _cx
            ey = _a * np.cos(_theta) * _sa + _b * np.sin(_theta) * _ca + _cy
            ez = np.zeros_like(_theta)
            self.ax.plot(ex, ey, ez, color='darkorange', lw=2.0, alpha=0.85,
                         label=f'{landing_prob}% Error Ellipse')
            poly = Poly3DCollection([list(zip(ex, ey, ez))],
                                    alpha=0.10, facecolor='orange', edgecolor='none')
            self.ax.add_collection3d(poly)
        else:
            theta  = np.linspace(0, 2 * math.pi, 72)
            cx_r   = impact_x + r90 * np.cos(theta)
            cy_r   = impact_y + r90 * np.sin(theta)
            cz_r   = np.zeros_like(theta)
            self.ax.plot(cx_r, cy_r, cz_r, color='red', lw=1.5, alpha=0.6,
                         label=f'Landing Area ({landing_prob}%)')
            n_pts      = 60
            disc_theta = np.linspace(0, 2 * math.pi, n_pts)
            disc_x = impact_x + r90 * np.cos(disc_theta)
            disc_y = impact_y + r90 * np.sin(disc_theta)
            disc_z = np.zeros(n_pts)
            poly   = Poly3DCollection([list(zip(disc_x, disc_y, disc_z))],
                                      alpha=0.12, facecolor='red', edgecolor='none')
            self.ax.add_collection3d(poly)

        # Ground projection
        self.ax.plot(x_vals, y_vals, np.zeros_like(z_vals),
                     color='gray', lw=0.8, alpha=0.35, linestyle='--')

        # ── Wind arrows ───────────────────────────────────────────────────────
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
                           v_a * scale / (spd + 1e-9), 0,
                           color='limegreen', lw=1.2, arrow_length_ratio=0.3)
            self.ax.text(u_a * scale / (spd + 1e-9),
                         v_a * scale / (spd + 1e-9),
                         alt + alt_max * 0.02,
                         f'{spd:.1f}m/s', color='green', fontsize=7)

        self.ax.view_init(elev=self._fixed_elev, azim=self._fixed_azim)
        self._apply_fixed_axis_labels()
        self.ax.tick_params(labelsize=7)

        # ── Stats & overlay text ──────────────────────────────────────────────
        downrange_m = math.hypot(impact_x, impact_y)
        apogee_m    = data.get('apogee_m',
                               float(np.max(z_vals)) if len(z_vals) > 0 else 0.0)
        para_str    = f'{para_time:.2f} s' if para_time is not None else '— s'
        bf_str      = f'{bf_time:.2f} s'  if bf_time  is not None else '— s'
        if mc_running:
            cep_str = 'computing…'
        elif mc_cep is not None:
            cep_str = f'{mc_cep:.1f} m'
        else:
            cep_str = '—'

        # Prominent landing-radius banner
        self.fig.text(
            0.50, 0.985,
            f'Pred. Landing Radius:  {r90_radius:.1f} m  ({landing_prob}%)',
            ha='center', va='top',
            fontsize=13, fontweight='bold', color='#cc0000',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.30',
                      facecolor='#fff0f0', edgecolor='#cc0000', alpha=0.92))

        # Score formula for competition modes
        r_horiz   = data.get('r_horiz', downrange_m)
        hang_time = data.get('hang_time', None)
        ph1_mode  = getattr(phase1_result, 'mode',       None) if phase1_result else None
        ph1_score = getattr(phase1_result, 'best_score', None) if phase1_result else None
        score_lines = []
        if operation_mode in ('Precision Landing', 'Winged Hover') and r_max_val is not None:
            if hang_time is not None:
                score_lines.append(
                    f'r_max={r_max_val:.0f}  r={r_horiz:.1f}  t={hang_time:.2f}\n'
                    f'Score = r_max - r + t = {r_max_val - r_horiz + hang_time:.2f}')
        if ph1_score is not None and ph1_mode in ('Precision Landing', 'Winged Hover'):
            label = ('★ Phase 1 best hover time: ' if ph1_mode == 'Winged Hover'
                     else '★ Phase 1 best score:      ')
            score_lines.append(f'{label}{ph1_score:.2f}' +
                                (' s' if ph1_mode == 'Winged Hover' else ''))
        if score_lines:
            self.fig.text(
                0.50, 0.935, '\n'.join(score_lines),
                ha='center', va='top',
                fontsize=9, fontweight='bold', color='#7700aa',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.28',
                          facecolor='#f5eeff', edgecolor='#9933cc', alpha=0.88))

        # Stats box (top-left)
        stats_text = (
            f'Apogee:           {apogee_m:.1f} m\n'
            f'Backfire:         {bf_str}\n'
            f'Parachute Open:   {para_str}\n'
            f'Downrange:        {downrange_m:.1f} m\n'
            f'CEP (50%):        {cep_str}'
        )
        if last_opt_info:
            stats_text += (f'\nBest Elevation:   {last_opt_info["elev"]:.1f}°\n'
                           f'Best Azimuth:     {last_opt_info["azi"]:.1f}°')
        self.fig.text(
            0.02, 0.985, stats_text,
            ha='left', va='top',
            fontsize=10, fontweight='bold', color='#222222',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.35',
                      facecolor='white', edgecolor='gray', alpha=0.9))

        # ── Axis limits ───────────────────────────────────────────────────────
        _me = mc_ellipse
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

        self.ax.legend(loc='upper right',
                       bbox_to_anchor=(0.98, 0.985),
                       bbox_transform=self.fig.transFigure,
                       ncol=2, fontsize=10, framealpha=0.85)
        self._apply_safe_layout()
        self.draw_compass()
        self.canvas.draw()

    # ── Wind sub-plots ────────────────────────────────────────────────────────

    def update_wind(
        self,
        surf_wind_time_history,
        surf_dir: float,
        up_spd: float,
        up_dir: float,
        mc_wind_profiles=None,
        wind_avg_recent: float = 0.0,
    ) -> None:
        spd_ax = getattr(self, 'wind_ax_spd',     None)
        cmp_ax = getattr(self, 'wind_ax_compass',  None)
        if spd_ax is None or cmp_ax is None:
            return

        history  = list(surf_wind_time_history)
        surf_spd = history[-1][1] if history else 0.0

        # Left: time-series
        spd_ax.clear()
        spd_ax.set_title('Wind Speed (Time Series)', fontsize=10)
        spd_ax.set_xlabel('Time (s ago)', fontsize=9)
        spd_ax.set_ylabel('Wind Speed (m/s)', fontsize=9)
        spd_ax.tick_params(labelsize=8)
        spd_ax.grid(True, alpha=0.3)

        if history:
            t_latest = history[-1][0]
            ts_rel   = [h[0] - t_latest for h in history]
            ws       = [h[1] for h in history]
            spd_ax.plot(ts_rel, ws,
                        linestyle='-', linewidth=1.6,
                        marker='.', markersize=3, color='#1f77b4', label='Surface')
            spd_ax.axhline(y=wind_avg_recent, color='red', linestyle='--',
                           linewidth=1.4,
                           label=f'10s Avg: {wind_avg_recent:.1f} m/s')
            spd_ax.axhline(y=up_spd, color='green', linestyle=':',
                           linewidth=1.4, label=f'Upper: {up_spd:.1f} m/s')
            spd_ax.set_xlim(-60.0, 1.0)
            y_max = max(max(ws), wind_avg_recent, up_spd, 1.0) * 1.25
            spd_ax.set_ylim(0, y_max)
            spd_ax.legend(fontsize=8, loc='upper left')
        else:
            spd_ax.text(0.5, 0.5, 'Waiting for wind data...',
                        transform=spd_ax.transAxes,
                        ha='center', va='center', color='gray', fontsize=9)
            spd_ax.set_xlim(-60.0, 1.0)
            spd_ax.set_ylim(0, 10)

        # Middle: wind-profile spaghetti
        prof_ax = getattr(self, 'wind_ax_profile', None)
        if prof_ax is not None:
            ALT_CAP = 500
            prof_ax.clear()
            prof_ax.set_title('Wind Profile', fontsize=9)
            prof_ax.set_xlabel('Speed (m/s)', fontsize=8)
            prof_ax.set_ylabel('Alt (m)', fontsize=8)
            prof_ax.tick_params(labelsize=7)
            prof_ax.grid(True, alpha=0.3)

            if mc_wind_profiles:
                for sp in mc_wind_profiles[:80]:
                    az  = [z for z, _ in sp if z <= ALT_CAP]
                    as_ = [s for z, s in sp if z <= ALT_CAP]
                    if az:
                        prof_ax.plot(as_, az, color='#888888', lw=0.4, alpha=0.25)
                agg = defaultdict(list)
                for sp in mc_wind_profiles:
                    for z, s in sp:
                        if z <= ALT_CAP:
                            agg[z].append(s)
                sorted_z = sorted(agg.keys())
                mean_s   = [float(np.mean(agg[z])) for z in sorted_z]
                prof_ax.plot(mean_s, sorted_z, color='#1f77b4', lw=2.2, label='Mean')
                prof_ax.legend(fontsize=7, loc='upper right')
            else:
                _u, _v = build_wind_profile(surf_spd, surf_dir, 3.0,
                                            up_spd, up_dir, 100.0)
                _az = [z for z, _ in _u if z <= ALT_CAP]
                _as = [math.sqrt(u**2 + v**2)
                       for (z, u), (_, v) in zip(_u, _v) if z <= ALT_CAP]
                if _az:
                    prof_ax.plot(_as, _az, color='#1f77b4', lw=2.0)

            spd_vals = [s for sp in (mc_wind_profiles or [])
                        for z, s in sp if z <= ALT_CAP]
            x_max = max(max(spd_vals, default=up_spd), up_spd, surf_spd, 1.0) * 1.2
            prof_ax.set_xlim(0, x_max)
            prof_ax.set_ylim(0, ALT_CAP)

        # Right: compass
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
            cmp_ax.annotate('', xy=(theta, 0.95), xytext=(theta, 0.0),
                            arrowprops=dict(arrowstyle='->', color=color, lw=2.0))
            cmp_ax.text(theta, 1.12, label, color=color,
                        fontsize=8, ha='center', va='center')

        _draw_arrow(up_dir,   '#1f77b4', 'Upper')
        _draw_arrow(surf_dir, '#ff7f0e', 'Surface')

        try:
            self.wind_canvas.draw_idle()
        except Exception:
            pass

    def update_realtime_wind_label(
        self,
        surf_spd: float,
        surf_dir: float,
        up_spd:   float,
        up_dir:   float,
        gust:     float,
    ) -> None:
        try:
            self.realtime_wind_label.config(
                text=(f"Surface: {surf_spd:.1f} m/s  @ {surf_dir:.0f}°"
                      f"   (Gust {gust:.1f})"
                      f"   |   Upper: {up_spd:.1f} m/s @ {up_dir:.0f}°"))
        except Exception:
            pass

    # ── 3-D compass overlay ───────────────────────────────────────────────────

    def draw_compass(self) -> None:
        if self._compass_ax is not None:
            try:
                self._compass_ax.remove()
            except Exception:
                pass
            self._compass_ax = None

        cax = self.fig.add_axes([0.83, 0.04, 0.14, 0.14], facecolor='none', zorder=20)
        self._compass_ax = cax
        cax.set_xlim(-1.4, 1.4)
        cax.set_ylim(-1.4, 1.4)
        cax.set_aspect('equal')
        cax.set_xticks([])
        cax.set_yticks([])
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
                         arrowprops=dict(arrowstyle='-|>', color='dimgray', lw=1.0))

        cax.text(nx * 1.10, ny * 1.10, 'N',
                 color='red', fontsize=9, fontweight='bold',
                 ha='center', va='center')
        cax.text(ex * 1.10, ey * 1.10, 'E',
                 color='dimgray', fontsize=8, ha='center', va='center')
        cax.text(sx * 1.10, sy * 1.10, 'S',
                 color='dimgray', fontsize=8, ha='center', va='center')
        cax.text(wx * 1.10, wy * 1.10, 'W',
                 color='dimgray', fontsize=8, ha='center', va='center')

    # ── Layout helpers ────────────────────────────────────────────────────────

    def _apply_safe_layout(self) -> None:
        import warnings
        try:
            self.ax.set_position(list(self._PLOT_RECT))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                self.fig.subplots_adjust(
                    left   = self._PLOT_RECT[0],
                    right  = self._PLOT_RECT[0] + self._PLOT_RECT[2],
                    bottom = self._PLOT_RECT[1],
                    top    = self._PLOT_RECT[1] + self._PLOT_RECT[3],
                )
        except Exception:
            pass

    def _apply_fixed_axis_labels(self) -> None:
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        self.ax.set_zlabel('')
        self.ax.text2D(0.02, 0.02, 'Altitude (Up)',
                       transform=self.ax.transAxes,
                       ha='left', va='bottom',
                       fontsize=8, fontweight='bold', color='#333333')
