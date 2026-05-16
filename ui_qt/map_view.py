import math
import os

os.environ["QT_API"] = "pyside6"
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.patches as patches

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QStackedLayout, QHBoxLayout
from PySide6.QtCore import Qt, Slot, Signal, QObject
from ui_qt.app_state import AppState

class MapView(QWidget):
    # This signal is kept for API compatibility if something connects to it,
    # though it might not be used directly in pure Matplotlib display without custom event handlers.
    coordinates_picked = Signal(float, float)

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self._state = app_state
        self._build_ui()
        self._draw_static_items(getattr(app_state, 'target_radius', 0.0))

        if hasattr(app_state, 'is_calculating_changed'):
            app_state.is_calculating_changed.connect(self._on_calculating_changed)
        if hasattr(app_state, 'simulation_result_changed'):
            app_state.simulation_result_changed.connect(self._on_simulation_result)

        # We don't necessarily re-render everything just on lat/lon change for a metric map,
        # but we keep the connections to avoid breaking the expected reactivity.
        if hasattr(app_state, 'launch_lat_changed'):
            app_state.launch_lat_changed.connect(lambda _: self._render_result(getattr(self._state, 'simulation_result', {}) or {}))
        if hasattr(app_state, 'launch_lon_changed'):
            app_state.launch_lon_changed.connect(lambda _: self._render_result(getattr(self._state, 'simulation_result', {}) or {}))

        # Connect visibility toggle signals from AppState to redraw map view
        if hasattr(app_state, 'show_kde_changed'):
            app_state.show_kde_changed.connect(lambda _: self._render_result(getattr(self._state, 'simulation_result', {}) or {}))
        if hasattr(app_state, 'show_cep_changed'):
            app_state.show_cep_changed.connect(lambda _: self._render_result(getattr(self._state, 'simulation_result', {}) or {}))
        if hasattr(app_state, 'show_scatter_changed'):
            app_state.show_scatter_changed.connect(lambda _: self._render_result(getattr(self._state, 'simulation_result', {}) or {}))

    def _build_ui(self):
        layout = QStackedLayout(self)
        layout.setStackingMode(QStackedLayout.StackAll)

        # Bottom Layer: Matplotlib Canvas
        self.figure = plt.figure(facecolor='#1e1e2e')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e2e')
        self.ax.tick_params(colors='#cdd6f4')
        for spine in self.ax.spines.values():
            spine.set_color('#45475a')

        # Top Layer: Overlays
        top_widget = QWidget(self)
        top_widget.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(10, 10, 10, 10)

        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Reset View", top_widget)
        self.btn_reset.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.btn_reset.clicked.connect(self._on_reset_view)
        self.btn_reset.setStyleSheet("background: #1e1e2e; color: #cdd6f4; border: 1px solid #45475a; padding: 4px; font-size: 10px; font-weight: bold; border-radius: 3px;")
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch()

        self._info = QLabel("No simulation result.")
        self._info.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self._info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info.setStyleSheet("QLabel { font-size: 11px; font-weight: bold; padding: 4px; background: rgba(30, 30, 46, 0.85); color: #cdd6f4; border-radius: 3px; }")

        top_layout.addLayout(btn_layout)
        top_layout.addStretch()
        top_layout.addWidget(self._info)

        layout.addWidget(self.canvas)
        layout.addWidget(top_widget)

    def _draw_static_items(self, target_radius):
        self._render_result({})

    def update_landing(self, lat, lon):
        # We don't use lat/lon directly in metric map anymore, but keep API
        pass

    @Slot(bool)
    def _on_calculating_changed(self, calculating: bool):
        if calculating:
            self._info.setText("Calculating...")

    @Slot(object)
    def _on_simulation_result(self, result):
        if not result or result.get('cancelled'):
            self._info.setText("Simulation cancelled or no result.")
            return
        self._render_result(result)

    def _render_result(self, result):
        self.ax.clear()

        # Setup Axes
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.grid(True, linestyle='--', alpha=0.7, color='#45475a')
        self.ax.set_xlabel("East (m)", color='#cdd6f4')
        self.ax.set_ylabel("North (m)", color='#cdd6f4')
        self.ax.tick_params(colors='#cdd6f4')

        # Launch Site
        self.ax.scatter(0, 0, marker='*', s=150, color='#4488ff', label='Launch Site', zorder=10)

        target_radius = getattr(self._state, 'target_radius', 0.0) or 0.0
        if target_radius > 0:
            target_circle = patches.Circle((0, 0), radius=target_radius, edgecolor='#0055ff', facecolor='none', linestyle='--', zorder=5)
            self.ax.add_patch(target_circle)

        if not result:
            self.canvas.draw()
            return

        impact_x = float(result.get('land_x', 0.0))
        impact_y = float(result.get('land_y', 0.0))
        r90 = float(result.get('r_N_radius', 0.0))
        cep = float(result.get('cep', 0.0))
        scatter_x = result.get('mc_scatter_x', [])
        scatter_y = result.get('mc_scatter_y', [])
        ellipse = result.get('cep_ellipse') or result.get('ellipse') # fallback for 'ellipse'
        contours = result.get('kde_contours', [])
        prob = int(result.get('landing_prob', 90))
        apogee = float(result.get('apogee_m', 0.0))
        tof = float(result.get('hang_time', 0.0))

        self._info.setText(
            f"R{prob}: {r90:.1f} m  |  CEP50: {cep:.1f} m  |  "
            f"Apogee: {apogee:.0f} m  |  ToF: {tof:.1f} s"
        )

        # Impact Scatter
        if len(scatter_x) > 0 and len(scatter_y) > 0 and getattr(self._state, 'show_scatter', True):
            self.ax.scatter(scatter_x[:500], scatter_y[:500], c='#ff6633', s=10, alpha=0.5, label='MC Impacts', zorder=2)

        # Nominal Landing point
        self.ax.scatter(impact_x, impact_y, marker='o', s=40, color='#ff4444', edgecolor='#cc0000', label='Nominal Impact', zorder=6)

        # R90 Circle
        if r90 > 0:
            r90_circle = patches.Circle((impact_x, impact_y), radius=r90, edgecolor='#cc0000', facecolor='none', linewidth=2, zorder=5)
            self.ax.add_patch(r90_circle)

        # CEP Circle
        if cep > 0 and getattr(self._state, 'show_cep', True):
            cep_circle = patches.Circle((impact_x, impact_y), radius=cep, edgecolor='#9933cc', facecolor='none', linewidth=1.8, linestyle='--', zorder=5)
            self.ax.add_patch(cep_circle)

        # 90% CEP Ellipse
        if ellipse and getattr(self._state, 'show_cep', True):
            cx = ellipse.get('cx', impact_x)
            cy = ellipse.get('cy', impact_y)
            width = ellipse['a'] * 2
            height = ellipse['b'] * 2
            angle_deg = math.degrees(ellipse['angle_rad'])

            ellipse_patch = patches.Ellipse((cx, cy), width, height, angle=angle_deg,
                                            edgecolor='#00bb00', facecolor='#00bb00', alpha=0.3, linewidth=2, label='90% CEP', zorder=3)
            self.ax.add_patch(ellipse_patch)

        # KDE Contours
        if contours and getattr(self._state, 'show_kde', True):
            for i, contour in enumerate(contours):
                points = contour['points_m']
                poly = patches.Polygon(points, closed=True, edgecolor='#cc5500', facecolor='none', linewidth=1.5, zorder=4, label='KDE Contours' if i == 0 else "")
                self.ax.add_patch(poly)

        # Legend
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            # Filter out duplicate labels
            by_label = dict(zip(labels, handles))
            legend = self.ax.legend(by_label.values(), by_label.keys(), loc='upper right', facecolor='#1e1e2e', edgecolor='#45475a', labelcolor='#cdd6f4')
            legend.set_zorder(20)

        self.figure.tight_layout()
        self.canvas.draw()

    def _on_reset_view(self):
        # Reset the view to autoscale based on all data
        self.ax.autoscale()
        self.canvas.draw()
