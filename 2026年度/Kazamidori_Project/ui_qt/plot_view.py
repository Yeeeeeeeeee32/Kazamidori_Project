"""
ui_qt/plot_view.py

PySide6-hosted Matplotlib canvas: 3-D trajectory (top) + polar wind compass
(bottom), separated by a thin QSplitter handle.

Ghosting prevention
-------------------
Subscribes to AppState.is_calculating_changed.
  True  → sweep all tracked simulation artists and call draw_idle().
  False → no-op; simulation_result_changed delivers the fresh payload.

Wind compass
------------
Draws exactly 5 altitude nodes (3 m, 10 m, 150 m, 300 m, 600 m) as annotate
arrows on a polar compass rose.  Each arrow points in the direction the wind is
TRAVELLING (meteorological FROM direction + 180°); length ∝ speed.
Colour scheme: warm (red/orange) at low altitude → cool (blue) at high altitude.

When a simulation result supplies wind_nodes, those values are used directly.
Between simulations the wind at each altitude is interpolated linearly between
the live surface reading and the upper-wind AppState setting.

Quiver compass
--------------
After each simulation result, wind-direction arrows are plotted on the 3-D
trajectory at ~6 equidistant altitude levels.  Colour transitions warm → cool.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)
from PySide6.QtCore import Qt

from ui_qt.app_state import AppState


# ── Altitude nodes for the compass ───────────────────────────────────────────
_ALT_NODES  = [3.0, 10.0, 150.0, 300.0, 600.0]
_ALT_LABELS = ["3 m", "10 m", "150 m", "300 m", "600 m"]
# Warm (low alt) → cool (high alt) — matches app_window.py _NODE_COLORS
_NODE_COLORS = ["#f38ba8", "#fab387", "#f9e2af", "#a6e3a1", "#89b4fa"]

_UPPER_ALT = 500.0   # altitude assumed for the upper wind reading


class PlotView(QWidget):
    """
    3-D trajectory canvas (top) + polar wind compass (bottom).

    Both canvases are embedded in a vertical QSplitter so the user can
    resize the split without restarting the application.
    """

    _ELEV = 25.0
    _AZIM = 45.0

    def __init__(self, app_state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = app_state

        # Tracked artists for the *current* simulation result (trajectory canvas).
        self._sim_artists: list[Artist] = []

        self._build_ui()

        app_state.is_calculating_changed.connect(self._on_calculating_changed)
        app_state.simulation_result_changed.connect(self._on_simulation_result)
        app_state.wind_history_updated.connect(self._on_wind_tick)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setHandleWidth(2)

        # ── Top: 3-D trajectory ───────────────────────────────────────────────
        self.traj_fig    = Figure(figsize=(6.4, 5.2), dpi=100, facecolor="#1e1e2e")
        self.traj_ax     = self.traj_fig.add_subplot(111, projection="3d")
        self.traj_canvas = FigureCanvasQTAgg(self.traj_fig)
        self.traj_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        top = QWidget(splitter)
        tl  = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)
        tl.addWidget(self.traj_canvas)

        # ── Bottom: dual wind panel — speed profile (left) + compass (right) ──
        self.wind_fig        = Figure(figsize=(9.0, 2.8), dpi=100, facecolor="#1e1e2e")
        self.wind_profile_ax = self.wind_fig.add_subplot(121)
        self.wind_ax         = self.wind_fig.add_subplot(122, projection="polar")
        self.wind_canvas     = FigureCanvasQTAgg(self.wind_fig)
        self.wind_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        bot = QWidget(splitter)
        bl  = QVBoxLayout(bot)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        bl.addWidget(self.wind_canvas)

        splitter.addWidget(top)
        splitter.addWidget(bot)
        splitter.setSizes([600, 280])

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(splitter)

        self._draw_empty_trajectory()
        self._redraw_compass(nodes=None)

    # ── Empty-state drawing ───────────────────────────────────────────────────

    def _draw_empty_trajectory(self) -> None:
        ax = self.traj_ax
        ax.scatter([0], [0], [0], marker="^", color="#4488ff",
                   s=60, zorder=6, label="Launch")
        ax.set_xlim(-60, 60); ax.set_ylim(-60, 60); ax.set_zlim(0, 60)
        ax.set_xlabel("East (m)");  ax.set_ylabel("North (m)")
        ax.set_zlabel("Alt (m)")
        ax.view_init(elev=self._ELEV, azim=self._AZIM)
        _style_3d_axes(ax, self.traj_fig)
        self.traj_canvas.draw_idle()

    # ── Ghosting prevention ───────────────────────────────────────────────────

    @Slot(bool)
    def _on_calculating_changed(self, calculating: bool) -> None:
        if not calculating:
            return
        for artist in list(self._sim_artists):
            try:
                artist.remove()
            except Exception:
                pass
        self._sim_artists.clear()

        for coll in list(self.traj_ax.collections):
            try: coll.remove()
            except Exception: pass
        for line in list(self.traj_ax.lines):
            try: line.remove()
            except Exception: pass
        for text in list(self.traj_ax.texts):
            try: text.remove()
            except Exception: pass
        for text in list(self.traj_fig.texts):
            try: text.remove()
            except Exception: pass

        self.traj_canvas.draw_idle()

    # ── Simulation result rendering ───────────────────────────────────────────

    @Slot(object)
    def _on_simulation_result(self, result: Optional[dict]) -> None:
        if result is None or result.get("cancelled"):
            self.traj_ax.cla()
            self._sim_artists.clear()
            self._draw_empty_trajectory()
            return
        self._render(result)
        # Update compass with wind_nodes from result if present
        wind_nodes = result.get("wind_nodes") if result else None
        self._redraw_compass(nodes=wind_nodes)

    def _render(self, result: dict) -> None:
        # Defensive clear in case is_calculating_changed(True) was missed.
        for artist in list(self._sim_artists):
            try: artist.remove()
            except Exception: pass
        self._sim_artists.clear()
        for text in list(self.traj_fig.texts):
            try: text.remove()
            except Exception: pass

        x_vals   = result.get("x_vals",   [])
        y_vals   = result.get("y_vals",   [])
        z_vals   = result.get("z_vals",   [])
        impact_x = float(result.get("impact_x",  0.0))
        impact_y = float(result.get("impact_y",  0.0))
        scatter  = result.get("scatter",  [])
        ellipse  = result.get("ellipse")
        cep      = float(result.get("cep",        0.0))
        r90      = float(result.get("r_N_radius", 0.0))
        prob     = int(result.get("landing_prob", 90))
        apogee   = float(result.get("apogee_m",   0.0))
        tof      = float(result.get("hang_time",  0.0))

        ax = self.traj_ax

        # ── Trajectory ────────────────────────────────────────────────────────
        if x_vals:
            (line,) = ax.plot(x_vals, y_vals, z_vals,
                              color="royalblue", lw=2.0, label="Trajectory")
            (proj,) = ax.plot(x_vals, y_vals, [0.0] * len(z_vals),
                              color="gray", lw=0.8, alpha=0.35, linestyle="--")
            self._sim_artists.extend([line, proj])

            ap_idx = int(np.argmax(z_vals))
            ax_, ay_, az_ = x_vals[ap_idx], y_vals[ap_idx], float(z_vals[ap_idx])
            (drop,) = ax.plot([ax_, ax_], [ay_, ay_], [0.0, az_],
                              color="gray", linestyle=":", lw=1.2)
            apex = ax.scatter([ax_], [ay_], [az_], marker="*",
                              color="gold", s=120, zorder=6, label="Apogee")
            self._sim_artists.extend([drop, apex])

        # ── Wind quiver at multiple altitude levels ────────────────────────────
        self._add_wind_quiver(ax, x_vals, y_vals, z_vals)

        # ── Launch / impact ───────────────────────────────────────────────────
        launch = ax.scatter([0], [0], [0], marker="^",
                            color="#4488ff", s=60, zorder=6, label="Launch")
        impact = ax.scatter([impact_x], [impact_y], [0], marker="o",
                            color="red", s=60, zorder=6, label="Impact")
        self._sim_artists.extend([launch, impact])

        # ── MC scatter (first 100 pts) ────────────────────────────────────────
        if scatter:
            pts = scatter[:100]
            mc_sc = ax.scatter(
                [p[0] for p in pts], [p[1] for p in pts], [0.0] * len(pts),
                s=6, c="orange", alpha=0.4, zorder=3)
            self._sim_artists.append(mc_sc)

        # ── Error ellipse ─────────────────────────────────────────────────────
        if ellipse:
            theta  = np.linspace(0, 2 * math.pi, 72)
            ca, sa = math.cos(ellipse["angle_rad"]), math.sin(ellipse["angle_rad"])
            a, b   = ellipse["a"], ellipse["b"]
            cx, cy = ellipse["cx"], ellipse["cy"]
            ex = a * np.cos(theta) * ca - b * np.sin(theta) * sa + cx
            ey = a * np.cos(theta) * sa + b * np.sin(theta) * ca + cy
            (ell_line,) = ax.plot(ex, ey, [0.0] * 72,
                                  color="darkorange", lw=2.0, alpha=0.85,
                                  label=f"R{prob} Ellipse")
            self._sim_artists.append(ell_line)
        elif r90 > 0:
            theta = np.linspace(0, 2 * math.pi, 72)
            (circ,) = ax.plot(
                impact_x + r90 * np.cos(theta),
                impact_y + r90 * np.sin(theta),
                [0.0] * 72,
                color="red", lw=1.5, alpha=0.6, label=f"R{prob}")
            self._sim_artists.append(circ)

        # ── CEP 50 % ─────────────────────────────────────────────────────────
        if cep > 0:
            theta = np.linspace(0, 2 * math.pi, 60)
            (cep_circ,) = ax.plot(
                impact_x + cep * np.cos(theta),
                impact_y + cep * np.sin(theta),
                [0.0] * 60,
                color="#9933cc", lw=1.8, linestyle=":", alpha=0.85, label="CEP 50%")
            self._sim_artists.append(cep_circ)

        # ── Axis limits ───────────────────────────────────────────────────────
        if z_vals:
            margin = max(abs(impact_x), abs(impact_y), r90, cep, 30.0) * 1.45
            ax.set_xlim(-margin, margin)
            ax.set_ylim(-margin, margin)
            ax.set_zlim(0, float(max(z_vals)) * 1.15)

        ax.set_xlabel("East (m)");  ax.set_ylabel("North (m)")
        ax.set_zlabel("Alt (m)")
        ax.view_init(elev=self._ELEV, azim=self._AZIM)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.82,
                  facecolor="#1e1e2e", edgecolor="#45475a", labelcolor="#cdd6f4")
        _style_3d_axes(ax, self.traj_fig)

        banner = self.traj_fig.text(
            0.50, 0.99,
            f"R{prob}: {r90:.1f} m   |   CEP50: {cep:.1f} m   |   "
            f"Apogee: {apogee:.0f} m   |   ToF: {tof:.1f} s",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color="#cc0000", family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff0f0",
                      edgecolor="#cc0000", linewidth=2, alpha=0.95))
        self._sim_artists.append(banner)
        self.traj_canvas.draw_idle()

    # ── Wind-direction quiver on 3-D trajectory ───────────────────────────────

    def _add_wind_quiver(
        self,
        ax,
        x_vals: list,
        y_vals: list,
        z_vals: list,
    ) -> None:
        """Draw colour-coded wind arrows at ~6 altitude levels on the trajectory.

        Arrow direction = where the wind is blowing TO.
        Colour: warm (low) → cool (high), matching _NODE_COLORS convention.
        """
        if not x_vals or not z_vals:
            return

        surf_spd = float(getattr(self._state, "surf_wind_speed", 4.0))
        surf_dir = float(getattr(self._state, "surf_wind_dir",   0.0))
        up_spd   = float(getattr(self._state, "upper_wind_speed", 8.0))
        up_dir   = float(getattr(self._state, "upper_wind_dir",   0.0))

        if surf_spd < 0.01 and up_spd < 0.01:
            return

        def _wind_at(t: float) -> tuple[float, float]:
            u_s = -surf_spd * math.sin(math.radians(surf_dir))
            v_s = -surf_spd * math.cos(math.radians(surf_dir))
            u_u = -up_spd  * math.sin(math.radians(up_dir))
            v_u = -up_spd  * math.cos(math.radians(up_dir))
            return u_s * (1 - t) + u_u * t, v_s * (1 - t) + v_u * t

        z_arr = np.asarray(z_vals, dtype=float)
        z_max = float(z_arr.max()) if len(z_arr) > 0 else 1.0
        n_lev = 6
        alt_levels = np.linspace(0, z_max * 0.9, n_lev)

        from matplotlib import cm as _cm
        colours = _cm.cool(np.linspace(0.1, 0.9, n_lev))

        scale = max(z_max * 0.12, 20.0)

        for alt, col in zip(alt_levels, colours):
            t_blend = min(alt / max(_UPPER_ALT, 1.0), 1.0)
            u, v = _wind_at(t_blend)
            speed = math.hypot(u, v)
            if speed < 0.01:
                continue

            idx = int(np.argmin(np.abs(z_arr - alt)))
            xp  = float(x_vals[idx])
            yp  = float(y_vals[idx])
            zp  = float(z_vals[idx])

            arrow_len = scale * 0.7
            uf = u / speed * arrow_len
            vf = v / speed * arrow_len

            qv = ax.quiver(xp, yp, zp, uf, vf, 0.0,
                           color=col, arrow_length_ratio=0.35,
                           linewidth=1.2, alpha=0.80)
            self._sim_artists.append(qv)

    # ── Live wind compass (replaces spaghetti) ────────────────────────────────

    @Slot(object)
    def _on_wind_tick(self, history) -> None:
        """Refresh the compass whenever a new wind reading arrives."""
        if not history:
            return
        self._redraw_compass(nodes=None)

    def _redraw_compass(self, nodes: Optional[list]) -> None:
        """
        Draw both wind subplots: speed profile (left) and polar compass (right).

        Parameters
        ----------
        nodes : list[dict] | None
            Each dict must have keys alt_m, speed_ms, dir_deg.
            When None, values are synthesised by interpolating between the
            current surface and upper-wind AppState settings.
        """
        ax_p = self.wind_profile_ax   # left — Cartesian speed profile
        ax_c = self.wind_ax           # right — polar compass
        fig  = self.wind_fig
        ax_p.cla()
        ax_c.cla()
        fig.patch.set_facecolor("#1e1e2e")

        # ── Build 5-node data list ─────────────────────────────────────────────
        if nodes:
            alt_data = nodes[:5]
        else:
            surf_spd = float(getattr(self._state, "surf_wind_speed", 4.0))
            surf_dir = float(getattr(self._state, "surf_wind_dir",   0.0))
            up_spd   = float(getattr(self._state, "upper_wind_speed", 8.0))
            up_dir   = float(getattr(self._state, "upper_wind_dir",   0.0))

            def _interp_speed(t: float) -> float:
                return surf_spd * (1.0 - t) + up_spd * t

            def _interp_dir(t: float) -> float:
                u_s = -surf_spd * math.sin(math.radians(surf_dir))
                v_s = -surf_spd * math.cos(math.radians(surf_dir))
                u_u = -up_spd  * math.sin(math.radians(up_dir))
                v_u = -up_spd  * math.cos(math.radians(up_dir))
                u   = u_s * (1.0 - t) + u_u * t
                v   = v_s * (1.0 - t) + v_u * t
                spd = math.hypot(u, v)
                if spd < 1e-6:
                    return surf_dir
                return (math.degrees(math.atan2(-u, -v)) % 360.0)

            alt_data = [
                {
                    "alt_m":    alt,
                    "speed_ms": _interp_speed(min(alt / _UPPER_ALT, 1.0)),
                    "dir_deg":  _interp_dir(min(alt / _UPPER_ALT, 1.0)),
                }
                for alt in _ALT_NODES
            ]

        speeds  = [float(n.get("speed_ms", 0.0)) for n in alt_data]
        alts    = [float(n.get("alt_m",    0.0)) for n in alt_data]
        dirs    = [float(n.get("dir_deg",  0.0)) for n in alt_data]
        colors  = [_NODE_COLORS[i] if i < len(_NODE_COLORS) else "#cdd6f4"
                   for i in range(len(alt_data))]
        lbls    = [_ALT_LABELS[i] if i < len(_ALT_LABELS)
                   else f"{n.get('alt_m', '?'):.0f} m"
                   for i, n in enumerate(alt_data)]
        max_spd = max(speeds, default=1.0)
        max_spd = max(max_spd, 1.0)

        # ════════════════════════════════════════════════════════════════════
        # LEFT SUBPLOT: Wind Speed Profile (altitude on Y-axis)
        # ════════════════════════════════════════════════════════════════════
        ax_p.set_facecolor("#1a1a2e")
        ax_p.tick_params(colors="#a6adc8", labelsize=6)
        for spine in ax_p.spines.values():
            spine.set_edgecolor("#45475a")
        ax_p.grid(True, color="#333355", linewidth=0.5, alpha=0.7)

        if len(speeds) > 1:
            ax_p.plot(speeds, alts,
                      color="#44445a", lw=1.2, alpha=0.55, zorder=1,
                      linestyle="--")

        for spd, alt, col in zip(speeds, alts, colors):
            marker = "D" if alt == 3.0 else "o"   # diamond = hardware anemometer
            ax_p.scatter([spd], [alt], color=col, s=52, zorder=5,
                         marker=marker, edgecolors="#1a1a2e", linewidths=0.8)
            ax_p.text(spd + max_spd * 0.05, alt, f"{spd:.1f}",
                      color=col, fontsize=6, va="center")

        ax_p.set_xlabel("Speed  (m/s)", color="#6c7086", fontsize=7, labelpad=3)
        ax_p.set_ylabel("Altitude  (m)", color="#6c7086", fontsize=7, labelpad=3)
        ax_p.set_title("Wind Speed Profile", color="#aaaaaa", fontsize=8, pad=6)
        ax_p.set_xlim(0.0, max_spd * 1.40)
        ax_p.set_ylim(-30.0, max(alts, default=600.0) * 1.18 + 10.0)

        if alts and alts[0] == 3.0:
            ax_p.annotate(
                "⬡ anemometer",
                xy=(speeds[0], alts[0]),
                xytext=(max_spd * 0.10, 60.0),
                fontsize=5, color="#aaaaaa",
                arrowprops=dict(arrowstyle="->", color="#555555",
                                lw=0.8, shrinkA=4, shrinkB=4),
            )

        # ════════════════════════════════════════════════════════════════════
        # RIGHT SUBPLOT: Polar Wind Compass
        # ════════════════════════════════════════════════════════════════════
        ax_c.set_facecolor("#1a1a2e")
        ax_c.tick_params(colors="#555555", labelsize=6)
        ax_c.grid(True, color="#333355", linewidth=0.6, alpha=0.7)
        ax_c.set_theta_zero_location("N")
        ax_c.set_theta_direction(-1)
        ax_c.set_rlabel_position(135)

        ax_c.set_rmax(1.05)
        ax_c.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax_c.set_yticklabels(
            [f"{max_spd * r:.1f}" for r in (0.25, 0.5, 0.75, 1.0)],
            color="#666666", fontsize=5,
        )

        for spd, d_from, col, lbl in zip(speeds, dirs, colors, lbls):
            r_norm = spd / max_spd
            theta  = np.radians((d_from + 180.0) % 360.0)

            ax_c.annotate(
                "",
                xy=(theta, r_norm),
                xytext=(theta, 0.0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=col,
                    lw=2.0,
                    mutation_scale=16,
                ),
            )

            if r_norm > 0.05:
                ax_c.text(
                    theta, min(r_norm + 0.09, 1.02),
                    f"{spd:.1f}",
                    fontsize=6, color=col,
                    ha="center", va="center", fontweight="bold",
                )

            ax_c.plot([], [], color=col, lw=2.5,
                      label=f"{lbl}  {spd:.1f} m/s @ {d_from:.0f}°")

        ax_c.set_title("Wind Compass", color="#aaaaaa", fontsize=8, pad=12)
        ax_c.legend(
            loc="upper right",
            fontsize=6,
            facecolor="#2b2b2b", edgecolor="#555555",
            labelcolor="#ffffff", framealpha=0.88,
            ncol=1,
        )

        fig.tight_layout(pad=0.5)
        self.wind_canvas.draw_idle()


# ── Axis styling helpers ──────────────────────────────────────────────────────

def _style_3d_axes(ax, fig: Optional[Figure] = None) -> None:
    ax.set_facecolor("#313244")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#45475a")
    ax.tick_params(colors="#a6adc8", labelsize=7)
    if fig is not None:
        fig.patch.set_facecolor("#1e1e2e")
