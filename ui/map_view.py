"""
ui/map_view.py
tkintermapview wrapper: draws launch/landing circles, error ellipses,
and KDE probability-mass contours on the map.

No simulation logic — receives pre-computed geometry from AppWindow.

Public API
----------
MapView(parent, launch_lat, launch_lon)
    Build the right-column map panel.

MapView.draw_elements(...)
    Re-draw all map overlays from the given state.

MapView.fit_bounds(...)
    Auto-zoom the map to show launch + landing rings.

MapView.set_position(lat, lon)
    Recentre the map.

MapView.set_fit_command(cmd)
    Bind the "Center Map" button to *cmd*.
"""

from __future__ import annotations

import math
from typing import Optional

import tkinter as tk
from tkinter import ttk
import tkintermapview


class MapView:
    """Manages the right-column map panel."""

    def __init__(
        self,
        parent: tk.Widget,
        launch_lat: float = 35.6828,
        launch_lon: float = 139.7590,
    ) -> None:
        self._build(parent, launch_lat, launch_lon)

    # ── Construction ──────────────────────────────────────────────────────────

    def _build(self, parent: tk.Widget, lat: float, lon: float) -> None:
        frame = ttk.Frame(parent, padding=10, relief="solid", borderwidth=1)
        frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self._frame = frame

        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=(0, 5))
        ttk.Label(ctrl, text="Map View",
                  font=("Arial", 10, "bold")).pack(side="left")
        self._fit_btn = ttk.Button(ctrl, text="[Center Map]")
        self._fit_btn.pack(side="right")

        self.map_widget = tkintermapview.TkinterMapView(frame)
        self.map_widget.pack(fill="both", expand=True)
        self.map_widget.set_tile_server(
            "https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga")
        self.map_widget.set_position(lat, lon)

    # ── Public helpers ────────────────────────────────────────────────────────

    def set_fit_command(self, cmd) -> None:
        self._fit_btn.config(command=cmd)

    def set_position(self, lat: float, lon: float) -> None:
        try:
            self.map_widget.set_position(lat, lon)
        except Exception:
            pass

    def fit_bounds(
        self,
        *,
        launch_lat: float,
        launch_lon: float,
        land_lat: float,
        land_lon: float,
        r90: float,
        has_sim_result: bool,
    ) -> None:
        try:
            m_lat, m_lon = self._meters_per_degree(launch_lat)
            launch_ring = 50.0
            land_ring   = max(r90 or 0.0, 2.5)

            def _ring_extents(la, lo, r_m):
                dlat = r_m / m_lat
                dlon = r_m / m_lon
                return la - dlat, la + dlat, lo - dlon, lo + dlon

            points = [(launch_lat, launch_lon, launch_ring)]
            if has_sim_result and land_ring > 0:
                points.append((land_lat, land_lon, land_ring))

            lat_mins, lat_maxs, lon_mins, lon_maxs = [], [], [], []
            for la, lo, r in points:
                lamin, lamax, lomin, lomax = _ring_extents(la, lo, r)
                lat_mins.append(lamin);  lat_maxs.append(lamax)
                lon_mins.append(lomin);  lon_maxs.append(lomax)

            min_lat, max_lat = min(lat_mins), max(lat_maxs)
            min_lon, max_lon = min(lon_mins), max(lon_maxs)
            pad_lat = max((max_lat - min_lat) * 0.10, 5.0 / m_lat)
            pad_lon = max((max_lon - min_lon) * 0.10, 5.0 / m_lon)

            self.map_widget.fit_bounding_box(
                (max_lat + pad_lat, min_lon - pad_lon),
                (min_lat - pad_lat, max_lon + pad_lon),
            )
        except Exception:
            pass

    def draw_elements(
        self,
        *,
        launch_lat:    float,
        launch_lon:    float,
        land_lat:      float,
        land_lon:      float,
        r_target:      float,
        r90:           float,
        has_sim_result: bool,
        p2_ellipse:    Optional[dict] = None,
        kde_contours=None,
    ) -> None:
        self.map_widget.delete_all_polygon()

        # Launch site dot + target radius ring
        self.map_widget.set_polygon(
            self._circle_coords(launch_lat, launch_lon, 2.5),
            fill_color="blue")
        self.map_widget.set_polygon(
            self._circle_coords(launch_lat, launch_lon, r_target),
            outline_color="blue")

        # Landing dot + predicted landing radius ring
        if has_sim_result and r90 > 0:
            self.map_widget.set_polygon(
                self._circle_coords(land_lat, land_lon, 2.5),
                fill_color="red")
            self.map_widget.set_polygon(
                self._circle_coords(land_lat, land_lon, r90),
                outline_color="red", border_width=2)

        # Phase 2 live error ellipse
        if p2_ellipse is not None:
            color = '#00bb00' if p2_ellipse['go'] else '#dd0000'
            coords = self._ellipse_polygon(
                land_lat, land_lon,
                p2_ellipse['cx'], p2_ellipse['cy'],
                p2_ellipse['a'],  p2_ellipse['b'],
                p2_ellipse['angle_rad'])
            self.map_widget.set_polygon(coords, outline_color=color, border_width=2)

        # MC KDE probability-mass contours
        if kde_contours:
            for latlons, col, bw in kde_contours:
                if len(latlons) >= 3:
                    self.map_widget.set_polygon(
                        latlons, outline_color=col, border_width=bw)

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _meters_per_degree(lat_deg: float):
        phi   = math.radians(lat_deg)
        m_lat = (111132.92
                 - 559.82 * math.cos(2 * phi)
                 + 1.175  * math.cos(4 * phi)
                 - 0.0023 * math.cos(6 * phi))
        m_lon = (111412.84 * math.cos(phi)
                 - 93.5    * math.cos(3 * phi)
                 + 0.118   * math.cos(5 * phi))
        return m_lat, m_lon

    @staticmethod
    def _circle_coords(lat: float, lon: float, radius_m: float, n: int = 36):
        coords = []
        for i in range(n):
            angle = math.pi * 2 * i / n
            dx    = radius_m * math.cos(angle)
            dy    = radius_m * math.sin(angle)
            d_lat = (dy / 6378137.0) * (180 / math.pi)
            d_lon = (dx / (6378137.0 * math.cos(math.pi * lat / 180))) * (180 / math.pi)
            coords.append((lat + d_lat, lon + d_lon))
        return coords

    @staticmethod
    def _ellipse_polygon(
        ref_lat: float, ref_lon: float,
        cx: float, cy: float,
        a: float, b: float,
        angle_rad: float,
        n: int = 60,
    ):
        m_lat, m_lon = MapView._meters_per_degree(ref_lat)
        coords = []
        for i in range(n):
            t  = 2.0 * math.pi * i / n
            xe = (a * math.cos(t) * math.cos(angle_rad)
                  - b * math.sin(t) * math.sin(angle_rad))
            ye = (a * math.cos(t) * math.sin(angle_rad)
                  + b * math.sin(t) * math.cos(angle_rad))
            lat = ref_lat + (cy + ye) / m_lat
            lon = ref_lon + (cx + xe) / m_lon
            coords.append((lat, lon))
        return coords
