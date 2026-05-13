import math
import io
import folium
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QStackedLayout, QHBoxLayout
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import Qt, Slot, Signal, QObject
from PySide6.QtWebChannel import QWebChannel
from ui_qt.app_state import AppState
from utils.geo_math import offset_to_latlon, ellipse_polygon, circle_polygon

class MapBridge(QObject):
    coordinates_picked = Signal(float, float)

    @Slot(float, float)
    def receive_coordinates(self, lat: float, lon: float):
        self.coordinates_picked.emit(lat, lon)


class MapView(QWidget):
    coordinates_picked = Signal(float, float)

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self._state = app_state
        self._land_lat = 0.0
        self._land_lon = 0.0

        self._build_ui()
        self._draw_static_items(getattr(app_state, 'target_radius', 0.0))

        if hasattr(app_state, 'is_calculating_changed'): app_state.is_calculating_changed.connect(self._on_calculating_changed)
        if hasattr(app_state, 'simulation_result_changed'): app_state.simulation_result_changed.connect(self._on_simulation_result)

    def _build_ui(self):
        layout = QStackedLayout(self)
        layout.setStackingMode(QStackedLayout.StackAll)

        # Bottom Layer: Web View
        self.web_view = QWebEngineView(self)

        self.channel = QWebChannel(self.web_view.page())
        self.bridge = MapBridge(self)
        self.channel.registerObject("bridge", self.bridge)
        self.web_view.page().setWebChannel(self.channel)

        self.bridge.coordinates_picked.connect(self.coordinates_picked)

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

        layout.addWidget(self.web_view)
        layout.addWidget(top_widget)

    def _draw_static_items(self, target_radius):
        self._render_result({})

    def update_landing(self, lat, lon):
        self._land_lat = float(lat)
        self._land_lon = float(lon)

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
        lat0 = getattr(self._state, 'launch_lat', 0.0)
        lon0 = getattr(self._state, 'launch_lon', 0.0)

        impact_x = float(result.get('land_x', 0.0))
        impact_y = float(result.get('land_y', 0.0))
        r90 = float(result.get('r_N_radius', 0.0))
        cep = float(result.get('cep', 0.0))
        scatter_x = result.get('mc_scatter_x', [])
        scatter_y = result.get('mc_scatter_y', [])
        ellipse = result.get('ellipse')
        contours = result.get('kde_contours', [])
        prob = int(result.get('landing_prob', 90))
        apogee = float(result.get('apogee_m', 0.0))
        tof = float(result.get('hang_time', 0.0))
        target_radius = getattr(self._state, 'target_radius', 0.0)

        self._land_lat, self._land_lon = offset_to_latlon(lat0, lon0, impact_x, impact_y)

        m = folium.Map(location=[lat0, lon0], zoom_start=15, control_scale=True, scrollWheelZoom=True, dragging=True)

        # Launch dot
        folium.CircleMarker(
            location=[lat0, lon0],
            radius=5, color='#4488ff', fill=True, fill_opacity=1
        ).add_to(m)

        if target_radius > 0:
            folium.Circle(
                location=[lat0, lon0],
                radius=target_radius,
                color='#0055ff',
                dash_array='5, 5',
                fill=False,
            ).add_to(m)

        if result:
            self._info.setText(
                f"R{prob}: {r90:.1f} m  |  CEP50: {cep:.1f} m  |  "
                f"Apogee: {apogee:.0f} m  |  ToF: {tof:.1f} s"
            )

            # Landing dot
            folium.CircleMarker(
                location=[self._land_lat, self._land_lon],
                radius=4, color='#cc0000', fill=True, fill_color='#ff4444', fill_opacity=1
            ).add_to(m)

            if r90 > 0:
                folium.Circle(
                    location=[self._land_lat, self._land_lon],
                    radius=r90,
                    color='#cc0000',
                    weight=2,
                    fill=False,
                ).add_to(m)

            if cep > 0 and show_cep:
                folium.Circle(
                    location=[self._land_lat, self._land_lon],
                    radius=cep,
                    color='#9933cc',
                    weight=1.8,
                    dash_array='5, 5',
                    fill=False,
                ).add_to(m)

            if ellipse and show_cep:
                poly = ellipse_polygon(lat0, lon0, impact_x, impact_y, ellipse['a'], ellipse['b'], ellipse['angle_rad'])
                folium.Polygon(
                    locations=poly,
                    color='#00bb00',
                    weight=2,
                    fill=True,
                    fill_color='#00bb00',
                    fill_opacity=0.3
                ).add_to(m)

            if contours and show_kde:
                for contour in contours:
                    poly = [offset_to_latlon(lat0, lon0, px, py) for px, py in contour['points_m']]
                    folium.Polygon(
                        locations=poly,
                        color='#cc5500',
                        weight=1.5,
                        fill=False,
                    ).add_to(m)

            if len(scatter_x) > 0 and len(scatter_y) > 0:
                for px, py in zip(scatter_x[:500], scatter_y[:500]):
                    plat, plon = offset_to_latlon(lat0, lon0, px, py)
                    folium.CircleMarker(
                        location=[plat, plon],
                        radius=2, color='none', fill=True, fill_color='#ff6633', fill_opacity=0.45
                    ).add_to(m)

        data = io.BytesIO()
        m.save(data, close_file=False)
        html = data.getvalue().decode()

        # Inject custom ID so we can access map object in JS

        html = html.replace('L.map(', 'window.mapObj = L.map(', 1)
        html = html.replace('<head>', '<head>\n    <script src="qrc:///qtwebchannel/qwebchannel.js"></script>', 1)

        js_injection = """
        <script>
        new QWebChannel(qt.webChannelTransport, function(channel) {
            window.bridge = channel.objects.bridge;
            window.mapObj.on('click', function(e) {
                if (e.originalEvent.ctrlKey) {
                    window.bridge.receive_coordinates(e.latlng.lat, e.latlng.lng);
                }
            });
        });
        </script>
        """
        html = html.replace('</body>', f'{js_injection}\n</body>', 1)

        self.web_view.setHtml(html)

    def _on_reset_view(self):
        js = f"if (window.mapObj) {{ window.mapObj.setView([{getattr(self._state, 'launch_lat', 0.0)}, {getattr(self._state, 'launch_lon', 0.0)}], 15); }}"
        self.web_view.page().runJavaScript(js)

