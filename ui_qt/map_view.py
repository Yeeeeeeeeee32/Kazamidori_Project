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
from utils.geo_math import offset_to_latlon

def _safe_float(val, default=0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def _safe_int(val, default=0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

class MapView(QWidget):
    # This signal is kept for API compatibility if something connects to it,
    # though it might not be used directly in pure Matplotlib display without custom event handlers.
    coordinates_picked = Signal(float, float)

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self._state = app_state
        print(f"=== MapView.__init__ === Received State: id={id(self._state)}")
        self._current_all_x = []
        self._current_all_y = []
        self._build_ui()

        # Map interaction state
        self._drag_start = None
        self._is_dragging = False
        self._ghost_marker = None

        self._is_panning = False
        self._pan_start = None
        self._pan_xlim = None
        self._pan_ylim = None

        # Connect Matplotlib events
        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion_notify)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.canvas.mpl_connect('scroll_event', self._on_mouse_scroll)

        if hasattr(app_state, 'is_calculating_changed'):
            app_state.is_calculating_changed.connect(self._on_calculating_changed)

        self.bind_app_state(app_state)

        # We don't necessarily re-render everything just on lat/lon change for a metric map,
        # but we keep the connections to avoid breaking the expected reactivity.
        if hasattr(app_state, 'launch_lat_changed'):
            app_state.launch_lat_changed.connect(lambda _: self._render_current_state())
        if hasattr(app_state, 'launch_lon_changed'):
            app_state.launch_lon_changed.connect(lambda _: self._render_current_state())

        # Connect visibility toggle signals from AppState to redraw map view
        if hasattr(app_state, 'show_kde_changed'):
            app_state.show_kde_changed.connect(lambda _: self._render_current_state())
        if hasattr(app_state, 'show_cep_changed'):
            app_state.show_cep_changed.connect(lambda _: self._render_current_state())
        if hasattr(app_state, 'show_scatter_changed'):
            app_state.show_scatter_changed.connect(lambda _: self._render_current_state())


    def _render_current_state(self):
        result = getattr(self._state, 'simulation_result', {}) or {}
        self._render_result(result)

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
        self.btn_reset = QPushButton("🔄 Reset View", top_widget)
        self.btn_reset.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.btn_reset.setToolTip("Reset map view to default bounds (Home)")
        self.btn_reset.setShortcut("Home")
        self.btn_reset.clicked.connect(self._on_reset_view)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background: #1e1e2e; color: #cdd6f4; border: 1px solid #45475a;
                padding: 4px 8px; font-size: 10px; font-weight: bold; border-radius: 3px;
            }
            QPushButton:hover {
                background: #313244; border-color: #89b4fa;
            }
            QPushButton:pressed {
                background: #45475a; color: #ffffff;
            }
        """)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch()

        self._info = QLabel("No simulation result. Configure parameters and click 'Run' (F5).")
        self._info.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self._info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info.setStyleSheet("QLabel { font-size: 11px; font-weight: bold; padding: 4px; background: rgba(30, 30, 46, 0.85); color: #cdd6f4; border-radius: 3px; }")

        top_layout.addLayout(btn_layout)
        top_layout.addStretch()
        top_layout.addWidget(self._info)

        layout.addWidget(self.canvas)
        layout.addWidget(top_widget)

    def update_landing(self, lat, lon):
        # We don't use lat/lon directly in metric map anymore, but keep API
        pass

    def bind_app_state(self, state):
        self._state = state
        print(f"=== MapView.bind_app_state === Received State: id={id(self._state)}")

        # Connect Simulation Results
        if hasattr(self._state, 'simulation_result_changed'):
            try:
                self._state.simulation_result_changed.connect(self._render_result)
                print("=== MapView.bind_app_state === SUCCESSFULLY connected 'simulation_result_changed'")
            except Exception as e:
                print(f"=== MapView.bind_app_state === ERROR connecting 'simulation_result_changed': {e}")
        else:
            print("=== MapView.bind_app_state === WARNING: 'simulation_result_changed' not found on state object")

        if hasattr(self._state, 'target_radius_changed'):
            self._state.target_radius_changed.connect(lambda _: self._render_result())

        # Connect Coordinate Changes
        if hasattr(self._state, 'launch_lat_changed'):
            self._state.launch_lat_changed.connect(lambda _: self._render_result())
        if hasattr(self._state, 'launch_lon_changed'):
            self._state.launch_lon_changed.connect(lambda _: self._render_result())

        # Trigger initial draw
        self._render_result()

    @Slot(bool)
    def _on_calculating_changed(self, calculating: bool):
        if calculating:
            self._info.setText("Calculating...")

    def _render_result(self, result=None):
        import traceback
        print(f"=== MAP RENDER TRIGGERED === Payload present: {result is not None}")
        try:
            if result is None:
                result = getattr(self, '_last_result', {})
            self._last_result = result

            self.ax.clear()

            # Setup Axes
            self.ax.set_aspect('equal', adjustable='datalim')
            self.ax.grid(True, linestyle='--', alpha=0.7, color='#45475a')
            self.ax.set_xlabel("East (m)", color='#cdd6f4')
            self.ax.set_ylabel("North (m)", color='#cdd6f4')
            self.ax.tick_params(colors='#cdd6f4')

            try:
                cur_lat = float(getattr(self._state, 'launch_lat', 0.0))
                cur_lon = float(getattr(self._state, 'launch_lon', 0.0))
                self.ax.set_title(f"Launch Site: {cur_lat:.5f}, {cur_lon:.5f}", color='#cdd6f4')
            except Exception:
                pass

            # Phase 4.1: Offline Map Raster Tile Engine
            self._render_map_tiles()

            # To keep track of all points for manual bounds
            all_x = [0.0]
            all_y = [0.0]

            # Launch Site
            self.ax.scatter(0, 0, marker='*', s=150, color='#4488ff', label='Launch Site', zorder=10)

            target_radius = _safe_float(getattr(self._state, 'target_radius', 0.0))
            if target_radius > 0:
                target_circle = patches.Circle((0, 0), radius=target_radius, edgecolor='#0055ff', facecolor='none', linestyle='--', zorder=5)
                self.ax.add_patch(target_circle)
                all_x.extend([-target_radius, target_radius])
                all_y.extend([-target_radius, target_radius])

            if not result:
                self.canvas.draw()
                print("=== MAP RENDER SUCCESSFULLY COMPLETED ===")
                return

            impact_x = _safe_float(result.get('impact_x', result.get('land_x', 0.0)))
            impact_y = _safe_float(result.get('impact_y', result.get('land_y', 0.0)))
            r90 = _safe_float(result.get('r_N_radius', 0.0))
            cep = _safe_float(result.get('cep', 0.0))

            scatter_x = result.get('mc_scatter_x', [])
            scatter_y = result.get('mc_scatter_y', [])
            scatter_points = result.get('scatter_points', [])
            if not scatter_x and scatter_points:
                scatter_x = [p[0] for p in scatter_points]
                scatter_y = [p[1] for p in scatter_points]

            ellipse = result.get('cep_ellipse') or result.get('ellipse') # fallback for 'ellipse'
            contours = result.get('kde_contours', [])
            prob = _safe_int(result.get('landing_prob', 90), 90)
            apogee = _safe_float(result.get('apogee_m', 0.0))
            tof = _safe_float(result.get('hang_time', 0.0))

            self._info.setText(
                f"R{prob}: {r90:.1f} m  |  CEP50: {cep:.1f} m  |  "
                f"Apogee: {apogee:.0f} m  |  ToF: {tof:.1f} s"
            )

            # Nominal Landing point
            self.ax.scatter(impact_x, impact_y, marker='o', s=40, color='#ff4444', edgecolor='#cc0000', label='Impact Site', zorder=6)
            all_x.append(impact_x)
            all_y.append(impact_y)



            # R90 Circle
            if r90 > 0:
                r90_circle = patches.Circle((impact_x, impact_y), radius=r90, edgecolor='#cc0000', facecolor='none', linewidth=2, zorder=5)
                self.ax.add_patch(r90_circle)
                all_x.extend([impact_x - r90, impact_x + r90])
                all_y.extend([impact_y - r90, impact_y + r90])

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
                                                edgecolor='#00bb00', facecolor='none', linewidth=2, label='90% CEP', zorder=3)
                self.ax.add_patch(ellipse_patch)

                # Approximate ellipse bounds for autoscaling
                a = ellipse['a']
                b = ellipse['b']
                # Max possible extent in x and y is roughly cx +- max(a,b)
                r_max = max(a, b)
                all_x.extend([cx - r_max, cx + r_max])
                all_y.extend([cy - r_max, cy + r_max])

            # KDE Contours
            if contours and getattr(self._state, 'show_kde', True):
                for i, contour in enumerate(contours):
                    points = contour['points_m'] if 'points_m' in contour else contour
                    if points:
                        poly = patches.Polygon(points, closed=True, edgecolor='#cc5500', facecolor='none', linewidth=1.5, zorder=4, label='KDE Contours' if i == 0 else "")
                        self.ax.add_patch(poly)
                        all_x.extend([p[0] for p in points])
                        all_y.extend([p[1] for p in points])

            # Impact Scatter (Filtered by KDE contours)
            if len(scatter_x) > 0 and len(scatter_y) > 0 and getattr(self._state, 'show_scatter', True):
                import numpy as np
                from matplotlib.path import Path

                sx_filtered = scatter_x
                sy_filtered = scatter_y

                # Check if we have KDE contours to filter by
                if contours and getattr(self._state, 'show_kde', True):
                    outer_contour = contours[0]
                    if isinstance(outer_contour, dict) and 'points_m' in outer_contour:
                        outer_points = outer_contour['points_m']
                    else:
                        outer_points = outer_contour

                    if outer_points and len(outer_points) > 2:
                        path = Path(outer_points)
                        pts = np.column_stack((scatter_x, scatter_y))
                        mask = path.contains_points(pts)
                        # We want points OUTSIDE the outermost contour
                        outside_mask = ~mask

                        sx_filtered = np.array(scatter_x)[outside_mask].tolist()
                        sy_filtered = np.array(scatter_y)[outside_mask].tolist()

                sx = sx_filtered[:500]
                sy = sy_filtered[:500]

                if len(sx) > 0:
                    self.ax.scatter(sx, sy, c='#ff6633', s=10, alpha=0.5, label='MC Scatter', zorder=2)
                    all_x.extend(sx)
                    all_y.extend(sy)

            # Legend
            handles, labels = self.ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                legend = self.ax.legend(by_label.values(), by_label.keys(), loc='upper right', facecolor='#1e1e2e', edgecolor='#45475a', labelcolor='#cdd6f4')
                legend.set_zorder(20)

            # Explicit Manual Auto-Scaling
            self._current_all_x = all_x
            self._current_all_y = all_y
            self._calculate_and_apply_bounds(self._current_all_x, self._current_all_y)

            self.figure.tight_layout()
            self.canvas.draw()
            print("=== MAP RENDER SUCCESSFULLY COMPLETED ===")
        except Exception as e:
            import traceback
            print(f"=== MAP RENDER ERROR ===\n{traceback.format_exc()}")


    def _calculate_and_apply_bounds(self, all_x, all_y):
        if not all_x or not all_y:
            return

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Avoid singular bounds
        if max_x == min_x:
            max_x += 100
            min_x -= 100
        if max_y == min_y:
            max_y += 100
            min_y -= 100

        dx = max_x - min_x
        dy = max_y - min_y

        # 10% margin
        margin_x = dx * 0.10
        margin_y = dy * 0.10

        self.ax.set_xlim(min_x - margin_x, max_x + margin_x)
        self.ax.set_ylim(min_y - margin_y, max_y + margin_y)

    def _on_button_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 1:
            if event.key == 'shift':
                if event.xdata is not None and event.ydata is not None:
                    self._drag_start = (event.xdata, event.ydata)
                    self._is_dragging = True
                    self._ghost_marker, = self.ax.plot([event.xdata], [event.ydata], marker='*', markersize=15,
                                                      color='white', alpha=0.5, zorder=20)
                    self.canvas.draw_idle()
            elif event.key is None:
                self._is_panning = True
                self._pan_start = (event.x, event.y)
                self._pan_xlim = self.ax.get_xlim()
                self._pan_ylim = self.ax.get_ylim()

    def _on_motion_notify(self, event):
        if self._is_dragging and event.inaxes == self.ax:
            if event.xdata is not None and event.ydata is not None:
                if self._ghost_marker:
                    self._ghost_marker.set_data([event.xdata], [event.ydata])
                    self.canvas.draw_idle()
        elif self._is_panning:
            if self._pan_start is None or self._pan_xlim is None or self._pan_ylim is None:
                return

            x0, y0 = self.ax.transData.inverted().transform(self._pan_start)
            x1, y1 = self.ax.transData.inverted().transform((event.x, event.y))

            dx_data = x1 - x0
            dy_data = y1 - y0

            new_xlim = (self._pan_xlim[0] - dx_data, self._pan_xlim[1] - dx_data)
            new_ylim = (self._pan_ylim[0] - dy_data, self._pan_ylim[1] - dy_data)

            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.canvas.draw_idle()

    def _on_button_release(self, event):
        if self._is_dragging:
            self._is_dragging = False
            if self._ghost_marker:
                self._ghost_marker.remove()
                self._ghost_marker = None

            if event.xdata is not None and event.ydata is not None:
                try:
                    current_lat = float(self._state.launch_lat)
                    current_lon = float(self._state.launch_lon)

                    new_lat, new_lon = offset_to_latlon(current_lat, current_lon, event.xdata, event.ydata)

                    # Update AppState directly
                    self._state.launch_lat = new_lat
                    self._state.launch_lon = new_lon

                    # Also emit signal just in case (SimController listens to it)
                    self.coordinates_picked.emit(new_lat, new_lon)
                except Exception as e:
                    import traceback
                    print(f"Error updating coordinates: {e}\n{traceback.format_exc()}")
            self.canvas.draw_idle()

        elif self._is_panning:
            self._is_panning = False

    def _on_mouse_scroll(self, event):
        if event.inaxes != self.ax: return
        if event.xdata is None or event.ydata is None: return

        base_scale = 1.2
        if event.step > 0:
            # zoom in
            scale_factor = 1.0 / base_scale
        elif event.step < 0:
            # zoom out
            scale_factor = base_scale
        else:
            scale_factor = 1.0

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        new_xlim = [xdata + (x - xdata) * scale_factor for x in xlim]
        new_ylim = [ydata + (y - ydata) * scale_factor for y in ylim]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def _render_map_tiles(self) -> None:
        """Render offline map tiles behind the coordinate canvas using the pre-stitched background.png."""
        import os
        import json
        from PIL import Image
        import numpy as np

        meta_path = "assets/offline_map/map_meta.json"
        img_path = "assets/offline_map/background.png"

        if not os.path.exists(meta_path) or not os.path.exists(img_path):
            return

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            declination = meta.get("magnetic_declination", 0.0)
            if getattr(self._state, 'magnetic_declination', None) != declination:
                self._state.magnetic_declination = declination

            tile_bounds = meta.get("tile_bounds", {})
            x_min = tile_bounds.get("x_min")
            x_max = tile_bounds.get("x_max")
            y_min = tile_bounds.get("y_min")
            y_max = tile_bounds.get("y_max")

            if None in (x_min, x_max, y_min, y_max):
                return

            # Render tiles
            # Need to calculate each tile's ENU coordinates relative to the center_lat/center_lon
            # so they form a continuous map aligned with the 0,0 center.

            # Load the single stitched background image
            background_path = "assets/offline_map/background.png"
            if not os.path.exists(background_path):
                return

            img = Image.open(background_path)

            # We strictly render it relative to the origin within the 500x500 bounds
            extent = meta.get("extent_meters", [-250.0, 250.0, -250.0, 250.0])

            self.ax.imshow(np.array(img), extent=extent, origin='upper', zorder=0, alpha=0.6)
            # We assume it was correctly generated as 500x500m ENU extent [-250, 250, -250, 250]
            # based on the map_downloader logic.
            img = Image.open(img_path)
            self.ax.imshow(np.array(img), extent=[-250, 250, -250, 250], zorder=-10, alpha=0.7)

        except Exception as e:
            import traceback
            print(f"=== MAP RENDER ERROR ===\n{traceback.format_exc()}")
            print(f"Error rendering offline map tiles: {e}")

    def _on_reset_view(self):
        # Reset the view to the dynamic bounds
        if getattr(self, '_current_all_x', None) and getattr(self, '_current_all_y', None):
            self._calculate_and_apply_bounds(self._current_all_x, self._current_all_y)
        else:
            self.ax.autoscale()
        self.canvas.draw()
