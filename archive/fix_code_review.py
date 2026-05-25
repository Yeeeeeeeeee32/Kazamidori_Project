import re

with open('ui_qt/app_window.py', 'r') as f:
    content = f.read()

# 1. Move state updates outside the threshold
search_coord_changed = """    def _on_manual_coord_changed(self, _v=None) -> None:
        lat = self.lat_input.value()
        lon = self.lon_input.value()

        if lat == -9999.0 or lon == -9999.0:
            return

        old_lat = getattr(self.state, 'launch_lat', 0.0)
        old_lon = getattr(self.state, 'launch_lon', 0.0)

        # Check if coordinates moved significantly
        if abs(lat - old_lat) > 0.0001 or abs(lon - old_lon) > 0.0001:
            self.state.launch_lat = lat
            self.state.launch_lon = lon
            self.map_widget.update_launch(lat, lon)
            self.state.needs_redraw.emit()

            # Emit download map signal if we have the button
            if hasattr(self, 'btn_download_map') and self.btn_download_map:
                self.btn_download_map.clicked.emit()"""

replace_coord_changed = """    def _on_manual_coord_changed(self, _v=None) -> None:
        lat = self.lat_input.value()
        lon = self.lon_input.value()

        if lat == -9999.0 or lon == -9999.0:
            return

        old_lat = getattr(self.state, 'launch_lat', 0.0)
        old_lon = getattr(self.state, 'launch_lon', 0.0)

        self.state.launch_lat = lat
        self.state.launch_lon = lon
        self.map_widget.update_launch(lat, lon)
        self.state.needs_redraw.emit()

        # Check if coordinates moved significantly
        if abs(lat - old_lat) > 0.0001 or abs(lon - old_lon) > 0.0001:
            # Emit download map signal if we have the button
            if hasattr(self, 'btn_download_map') and self.btn_download_map:
                self.btn_download_map.clicked.emit()"""

content = content.replace(search_coord_changed, replace_coord_changed)

# 2. Fix the toolbar button variable clash
search_toolbar_btn = """        btn_download_map = QPushButton("Download Map", tb)
        btn_download_map.setObjectName("btn_download_map")
        btn_download_map.setToolTip("Download offline map tiles for current location")
        tb.addWidget(btn_download_map)"""

replace_toolbar_btn = """        self.btn_download_map = QPushButton("Download Map", tb)
        self.btn_download_map.setObjectName("btn_download_map")
        self.btn_download_map.setToolTip("Download offline map tiles for current location")
        tb.addWidget(self.btn_download_map)"""
content = content.replace(search_toolbar_btn, replace_toolbar_btn)


search_settings_btn = """        self.btn_download_map = QPushButton("🗺️  Download Offline Map", w)
        self.btn_download_map.setObjectName("btn_download_map")
        self.btn_download_map.setToolTip("Download OSM tiles for the current coordinates to use offline")
        btn_dl_map = self.btn_download_map"""

replace_settings_btn = """        self.btn_offline_map = QPushButton("🗺️  Download Offline Map", w)
        self.btn_offline_map.setObjectName("btn_download_map")
        self.btn_offline_map.setToolTip("Download OSM tiles for the current coordinates to use offline")
        btn_dl_map = self.btn_offline_map"""
content = content.replace(search_settings_btn, replace_settings_btn)

# 3. Add disconnect logic to bind_app_state
search_bind_state = """        if hasattr(state, 'launch_lat_changed'):
            state.launch_lat_changed.connect(self._on_state_lat_changed)
        if hasattr(state, 'launch_lon_changed'):
            state.launch_lon_changed.connect(self._on_state_lon_changed)"""

replace_bind_state = """        if hasattr(state, 'launch_lat_changed'):
            try: state.launch_lat_changed.disconnect()
            except Exception: pass
            state.launch_lat_changed.connect(self._on_state_lat_changed)
        if hasattr(state, 'launch_lon_changed'):
            try: state.launch_lon_changed.disconnect()
            except Exception: pass
            state.launch_lon_changed.connect(self._on_state_lon_changed)"""
content = content.replace(search_bind_state, replace_bind_state)


with open('ui_qt/app_window.py', 'w') as f:
    f.write(content)
