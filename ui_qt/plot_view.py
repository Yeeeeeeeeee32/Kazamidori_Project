"""
ui_qt/plot_view.py

PySide6-hosted Matplotlib 3-D trajectory canvas.

Ghosting prevention
-------------------
Subscribes to AppState.is_calculating_changed.
  True  → immediately sweep ax.collections / ax.lines / ax.texts / fig.texts
           and call canvas.draw_idle().  ALL Matplotlib objects are removed via
           their own .remove() method before the Python list reference is dropped,
           so the C++ side is never accessed through a dangling pointer.
  False → no-op; the imminent simulation_result_changed signal will deliver
           the fresh payload.

Thread safety
-------------
is_calculating_changed is always emitted from the main thread (see SimController).
The Qt direct-connection default therefore keeps every canvas operation on the
main thread — Matplotlib is never touched from the worker thread.
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
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from ui_qt.app_state import AppState


class PlotView(QWidget):
    """3-D trajectory Matplotlib canvas embedded in a PySide6 QWidget."""

    _ELEV = 25.0
    _AZIM = 45.0

    def __init__(self, app_state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = app_state
        # Tracks every Matplotlib artist created for the *current* simulation
        # result.  Cleared (via artist.remove()) when a new run starts.
        self._sim_artists: list[Artist] = []

        self._build_canvas()

        app_state.is_calculating_changed.connect(self._on_calculating_changed)
        app_state.simulation_result_changed.connect(self._on_simulation_result)

    # ── Canvas construction ───────────────────────────────────────────────────

    def _build_canvas(self) -> None:
        self.fig    = Figure(figsize=(6.4, 5.2), dpi=100)
        self.ax     = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._draw_empty_state()

    def _draw_empty_state(self) -> None:
        """Minimal canvas: just the launch-site marker, ready for a new run."""
        self.ax.scatter([0], [0], [0], marker='^', color='#4488ff',
                        s=60, zorder=6, label='Launch')
        self.ax.set_xlim(-60, 60)
        self.ax.set_ylim(-60, 60)
        self.ax.set_zlim(0, 60)
        self.ax.set_xlabel('East (m)')
        self.ax.set_ylabel('North (m)')
        self.ax.set_zlabel('Alt (m)')
        self.ax.view_init(elev=self._ELEV, azim=self._AZIM)
        self.canvas.draw_idle()

    # ── Ghosting prevention ───────────────────────────────────────────────────

    @Slot(bool)
    def _on_calculating_changed(self, calculating: bool) -> None:
        """
        Called synchronously on the main thread via Qt direct connection.

        True  → remove every stale simulation artist so the canvas shows a
                clean empty state while the worker runs.
        False → no-op; _on_simulation_result handles the incoming payload.
        """
        if not calculating:
            return

        # ── Step 1: remove explicitly tracked artists ─────────────────────────
        # Iterate a copy; the original list is cleared immediately after so
        # no dangling Python reference survives past this frame.
        for artist in list(self._sim_artists):
            try:
                artist.remove()
            except Exception:
                pass
        self._sim_artists.clear()

        # ── Step 2: belt-and-suspenders sweep of axes and figure ──────────────
        # Catches any artist that was added outside _render() (e.g. by a future
        # partial-redraw path) and would otherwise ghost across runs.
        for coll in list(self.ax.collections):
            try:
                coll.remove()
            except Exception:
                pass
        for line in list(self.ax.lines):
            try:
                line.remove()
            except Exception:
                pass
        for text in list(self.ax.texts):
            try:
                text.remove()
            except Exception:
                pass
        for text in list(self.fig.texts):
            try:
                text.remove()
            except Exception:
                pass

        # ── Step 3: force immediate visual update ─────────────────────────────
        self.canvas.draw_idle()

    # ── Simulation result rendering ───────────────────────────────────────────

    @Slot(object)
    def _on_simulation_result(self, result: Optional[dict]) -> None:
        if result is None or result.get('cancelled'):
            # Cancelled: restore empty state with the launch marker.
            self.ax.cla()
            self._sim_artists.clear()
            self._draw_empty_state()
            return
        self._render(result)

    def _render(self, result: dict) -> None:
        """
        Populate the axes with one complete simulation result.

        All created artists are appended to self._sim_artists so the next call
        to _on_calculating_changed(True) can remove them precisely — no artist
        leaks across simulation runs.
        """
        # Defensive clear in case is_calculating_changed(True) was not received.
        for artist in list(self._sim_artists):
            try:
                artist.remove()
            except Exception:
                pass
        self._sim_artists.clear()
        for text in list(self.fig.texts):
            try:
                text.remove()
            except Exception:
                pass

        x_vals   = result.get('x_vals',    [])
        y_vals   = result.get('y_vals',    [])
        z_vals   = result.get('z_vals',    [])
        impact_x = float(result.get('impact_x',   0.0))
        impact_y = float(result.get('impact_y',   0.0))
        scatter  = result.get('scatter',   [])
        ellipse  = result.get('ellipse')
        cep      = float(result.get('cep',         0.0))
        r90      = float(result.get('r_N_radius',  0.0))
        prob     = int(result.get('landing_prob',  90))
        apogee   = float(result.get('apogee_m',    0.0))
        tof      = float(result.get('hang_time',   0.0))

        # ── Trajectory ────────────────────────────────────────────────────────
        if x_vals:
            (line,) = self.ax.plot(x_vals, y_vals, z_vals,
                                   color='royalblue', lw=2.0, label='Trajectory')
            (proj,) = self.ax.plot(x_vals, y_vals, [0.0] * len(z_vals),
                                   color='gray', lw=0.8, alpha=0.35, linestyle='--')
            self._sim_artists.extend([line, proj])

            # Apogee
            ap_idx = int(np.argmax(z_vals))
            ax_, ay_, az_ = x_vals[ap_idx], y_vals[ap_idx], z_vals[ap_idx]
            (drop,) = self.ax.plot([ax_, ax_], [ay_, ay_], [0.0, az_],
                                   color='gray', linestyle=':', lw=1.2)
            apex = self.ax.scatter([ax_], [ay_], [az_],
                                   marker='*', color='gold', s=120, zorder=6,
                                   label='Apogee')
            self._sim_artists.extend([drop, apex])

        # ── Launch / impact markers ────────────────────────────────────────────
        launch = self.ax.scatter([0], [0], [0], marker='^',
                                 color='#4488ff', s=60, zorder=6, label='Launch')
        impact = self.ax.scatter([impact_x], [impact_y], [0], marker='o',
                                 color='red', s=60, zorder=6, label='Impact')
        self._sim_artists.extend([launch, impact])

        # ── MC scatter (first 100 points) ─────────────────────────────────────
        if scatter:
            pts = scatter[:100]
            mc_sc = self.ax.scatter(
                [p[0] for p in pts], [p[1] for p in pts], [0.0] * len(pts),
                s=6, c='orange', alpha=0.4, zorder=3)
            self._sim_artists.append(mc_sc)

        # ── Error ellipse ──────────────────────────────────────────────────────
        if ellipse:
            theta  = np.linspace(0, 2 * math.pi, 72)
            ca, sa = math.cos(ellipse['angle_rad']), math.sin(ellipse['angle_rad'])
            a, b   = ellipse['a'], ellipse['b']
            cx, cy = ellipse['cx'], ellipse['cy']
            ex = a * np.cos(theta) * ca - b * np.sin(theta) * sa + cx
            ey = a * np.cos(theta) * sa + b * np.sin(theta) * ca + cy
            (ell_line,) = self.ax.plot(ex, ey, [0.0] * 72,
                                       color='darkorange', lw=2.0, alpha=0.85,
                                       label=f'R{prob} Ellipse')
            self._sim_artists.append(ell_line)
        elif r90 > 0:
            theta = np.linspace(0, 2 * math.pi, 72)
            (circ,) = self.ax.plot(
                impact_x + r90 * np.cos(theta),
                impact_y + r90 * np.sin(theta),
                [0.0] * 72,
                color='red', lw=1.5, alpha=0.6, label=f'R{prob}')
            self._sim_artists.append(circ)

        # ── CEP 50% circle ─────────────────────────────────────────────────────
        if cep > 0:
            theta = np.linspace(0, 2 * math.pi, 60)
            (cep_circ,) = self.ax.plot(
                impact_x + cep * np.cos(theta),
                impact_y + cep * np.sin(theta),
                [0.0] * 60,
                color='#9933cc', lw=1.8, linestyle=':', alpha=0.85, label='CEP 50%')
            self._sim_artists.append(cep_circ)

        # ── Axis limits ────────────────────────────────────────────────────────
        if z_vals:
            margin = max(abs(impact_x), abs(impact_y), r90, cep, 30.0) * 1.45
            self.ax.set_xlim(-margin, margin)
            self.ax.set_ylim(-margin, margin)
            self.ax.set_zlim(0, float(max(z_vals)) * 1.15)

        self.ax.set_xlabel('East (m)')
        self.ax.set_ylabel('North (m)')
        self.ax.set_zlabel('Alt (m)')
        self.ax.view_init(elev=self._ELEV, azim=self._AZIM)
        self.ax.legend(loc='upper right', fontsize=9, framealpha=0.85)

        # Stats banner — stored as a fig.text so it is caught by the fig.texts
        # sweep inside _on_calculating_changed(True).
        banner = self.fig.text(
            0.50, 0.99,
            f'R{prob}: {r90:.1f} m   |   CEP50: {cep:.1f} m   |   '
            f'Apogee: {apogee:.0f} m   |   ToF: {tof:.1f} s',
            ha='center', va='top', fontsize=10, fontweight='bold',
            color='#cc0000', family='monospace',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#fff0f0',
                      edgecolor='#cc0000', linewidth=2, alpha=0.95))
        self._sim_artists.append(banner)

        self.canvas.draw_idle()
