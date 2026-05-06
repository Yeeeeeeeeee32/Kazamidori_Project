"""
ui_qt/plot_view.py

PySide6-hosted Matplotlib canvas: 3-D trajectory (top) + unified wind
spaghetti panel (bottom), separated by a thin QSplitter handle.

Ghosting prevention
-------------------
Subscribes to AppState.is_calculating_changed.
  True  → sweep all tracked simulation artists and call draw_idle().
  False → no-op; simulation_result_changed delivers the fresh payload.

Wind panel
----------
Subscribes to AppState.wind_history_updated (fires every 1-second tick).
Maintains a 60-sample rolling buffer per "altitude level":
  • Surface (~0 m)   — from the live wind_history deque (actual readings)
  • Mid (~250 m)     — blend of surface and upper: 50/50 mix + slight offset
  • Upper (~500 m)   — from AppState.upper_wind_speed (changes on sim run)
Each level is drawn as a separate line (spaghetti), legend fixed upper-right.

Quiver compass
--------------
After each simulation result, wind-direction arrows are plotted on the 3-D
trajectory at ~6 equidistant altitude levels.  Arrow colour encodes altitude:
  orange → lower,  skyblue → upper.
Components use the meteorological convention from core/wind_model:
  u_east  = −speed · sin(dir_rad)
  v_north = −speed · cos(dir_rad)
"""

from __future__ import annotations

import math
from collections import deque
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


# ── Altitude levels for the spaghetti plot ────────────────────────────────────
# Each entry: (label, colour, blend_t)
# blend_t = 0.0 → pure surface,  1.0 → pure upper wind speed
_WIND_LEVELS = [
    ("Surface (~0 m)",  "orange",   0.0),
    ("Mid (~250 m)",    "#89b4fa",  0.5),
    ("Upper (~500 m)",  "skyblue",  1.0),
]

_WIND_BUF_LEN = 60   # one sample per second, 60-second window


class PlotView(QWidget):
    """
    3-D trajectory canvas (top) + unified wind spaghetti canvas (bottom).

    Both canvases are embedded in a vertical QSplitter so the user can
    resize the split without restarting the application.
    """

    _ELEV = 25.0
    _AZIM = 45.0

    # Altitude at which the upper wind reading is assumed valid (metres AGL).
    _UPPER_ALT = 500.0

    def __init__(self, app_state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = app_state

        # Tracked artists for the *current* simulation result (trajectory canvas).
        self._sim_artists: list[Artist] = []

        # Rolling 60-sample wind history per altitude level.
        # Index matches _WIND_LEVELS: [surface_deque, mid_deque, upper_deque]
        self._wind_bufs: list[deque] = [
            deque(maxlen=_WIND_BUF_LEN) for _ in _WIND_LEVELS
        ]

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

        # ── Bottom: unified wind spaghetti ────────────────────────────────────
        self.wind_fig    = Figure(figsize=(6.4, 2.8), dpi=100, facecolor="#1e1e2e")
        self.wind_ax     = self.wind_fig.add_subplot(111)
        self.wind_canvas = FigureCanvasQTAgg(self.wind_fig)
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
        self._draw_empty_wind()

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

    def _draw_empty_wind(self) -> None:
        ax = self.wind_ax
        ax.cla()
        _style_2d_axes(ax, self.wind_fig)
        ax.set_xlabel("Sample (1 s / div)", color="#6c7086", fontsize=7)
        ax.set_ylabel("Speed (m/s)",        color="#6c7086", fontsize=7)
        ax.set_title("Wind Speed — 60-s Spaghetti (awaiting data)",
                     color="#a6adc8", fontsize=8)
        ax.text(0.5, 0.5, "Wind monitor running…",
                transform=ax.transAxes, ha="center", va="center",
                color="#45475a", fontsize=9)
        self.wind_fig.tight_layout(pad=0.4)
        self.wind_canvas.draw_idle()

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

        Arrow direction = where the wind is blowing TO (meteorological FROM
        direction inverted).  Colour transitions orange (low) → skyblue (high).
        """
        if not x_vals or not z_vals:
            return

        surf_spd = float(self._state.surf_wind_speed)
        surf_dir = float(self._state.surf_wind_dir)
        up_spd   = float(self._state.upper_wind_speed)
        up_dir   = float(self._state.upper_wind_dir)

        if surf_spd < 0.01 and up_spd < 0.01:
            return

        # u, v components (wind blowing TO, not FROM)
        def _wind_at(t: float) -> tuple[float, float]:
            """Interpolate wind at blend factor t ∈ [0, 1]."""
            spd = surf_spd * (1 - t) + up_spd * t
            # Interpolate in U/V space to avoid 0°/360° wrap issues
            u_s = -surf_spd * math.sin(math.radians(surf_dir))
            v_s = -surf_spd * math.cos(math.radians(surf_dir))
            u_u = -up_spd * math.sin(math.radians(up_dir))
            v_u = -up_spd * math.cos(math.radians(up_dir))
            u = u_s * (1 - t) + u_u * t
            v = v_s * (1 - t) + v_u * t
            return u, v

        z_arr = np.asarray(z_vals, dtype=float)
        z_max = float(z_arr.max()) if len(z_arr) > 0 else 1.0
        n_lev = 6
        alt_levels = np.linspace(0, z_max * 0.9, n_lev)

        # Colour map: orange at ground, skyblue at top
        from matplotlib import cm as _cm
        colours = _cm.cool(np.linspace(0.1, 0.9, n_lev))

        scale = max(z_max * 0.12, 20.0)  # quiver arrow length in scene units

        for alt, col in zip(alt_levels, colours):
            t_blend = min(alt / max(self._UPPER_ALT, 1.0), 1.0)
            u, v = _wind_at(t_blend)
            speed = math.hypot(u, v)
            if speed < 0.01:
                continue

            # Find trajectory point closest to this altitude
            idx = int(np.argmin(np.abs(z_arr - alt)))
            xp, yp, zp = float(x_vals[idx]), float(y_vals[idx]), float(z_vals[idx])

            # Normalise to fixed visual length
            arrow_len = scale * 0.7
            uf = u / speed * arrow_len
            vf = v / speed * arrow_len

            qv = ax.quiver(xp, yp, zp, uf, vf, 0.0,
                           color=col, arrow_length_ratio=0.35,
                           linewidth=1.2, alpha=0.80)
            self._sim_artists.append(qv)

    # ── Live wind spaghetti ───────────────────────────────────────────────────

    @Slot(object)
    def _on_wind_tick(self, history) -> None:
        """Update internal buffers and redraw the unified wind canvas.

        *history* is the full deque of (speed_m_s, dir_deg) pairs from
        AppState.wind_history_updated.
        """
        if not history:
            return

        surf_speed = float(list(history)[-1][0])
        up_speed   = float(self._state.upper_wind_speed)

        # Buffer each level
        self._wind_bufs[0].append(surf_speed)
        self._wind_bufs[1].append(surf_speed * 0.5 + up_speed * 0.5)
        self._wind_bufs[2].append(up_speed)

        self._redraw_wind()

    def _redraw_wind(self) -> None:
        ax = self.wind_ax
        ax.cla()
        _style_2d_axes(ax, self.wind_fig)

        has_data = False
        for (label, colour, _), buf in zip(_WIND_LEVELS, self._wind_bufs):
            if not buf:
                continue
            has_data = True
            xs  = list(range(len(buf)))
            ys  = list(buf)
            ax.plot(xs, ys, color=colour, lw=1.4, alpha=0.85, label=label)

            # Moving average (window = 10 s)
            if len(ys) >= 10:
                ma = np.convolve(ys, np.ones(10) / 10, mode="valid")
                ax.plot(range(9, 9 + len(ma)), ma,
                        color=colour, lw=2.0, alpha=0.55,
                        linestyle="--")

        if has_data:
            ax.set_xlim(0, _WIND_BUF_LEN)
            ax.set_xlabel("Sample  (1 s / div)", color="#6c7086", fontsize=7)
            ax.set_ylabel("Speed (m/s)",          color="#6c7086", fontsize=7)
            ax.set_title("Wind Speed — 60-s Spaghetti",
                         color="#a6adc8", fontsize=8)
            # Legend fixed at upper right, outside simulation artists
            ax.legend(loc="upper right", fontsize=7, framealpha=0.85,
                      facecolor="#1e1e2e", edgecolor="#45475a",
                      labelcolor="#cdd6f4")

        self.wind_fig.tight_layout(pad=0.3)
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


def _style_2d_axes(ax, fig: Optional[Figure] = None) -> None:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#a6adc8", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#45475a")
    ax.grid(True, color="#2a2a42", linewidth=0.6, alpha=0.8)
    if fig is not None:
        fig.patch.set_facecolor("#1e1e2e")
