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
        self._label_markers: list = []
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

        # max_zoom=22 allows street-level detail
        self.map_widget = tkintermapview.TkinterMapView(frame, max_zoom=22)
        self.map_widget.pack(fill="both", expand=True)
        self.map_widget.set_tile_server(
            "https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga")
        self.map_widget.set_position(lat, lon)

        # Legend overlay (bottom-left corner, above map content)
        self._legend_canvas = tk.Canvas(
            frame, bg='white', highlightthickness=1,
            highlightbackground='#aaaaaa', cursor='arrow')
        self._draw_map_legend()
        self._legend_canvas.place(relx=0.01, rely=0.99, anchor='sw')

    # ── Public helpers ────────────────────────────────────────────────────────

    def set_fit_command(self, cmd) -> None:
        self._fit_btn.config(command=cmd)

    def set_position(self, lat: float, lon: float) -> None:
        try:
            self.map_widget.set_position(lat, lon)
        except Exception:
            pass

    def _draw_map_legend(self) -> None:
        c = self._legend_canvas
        c.delete("all")
        items = [
            ('Launch site',    'blue',    'sq'),
            ('Target radius',  '#0055ff', 'circ'),
            ('Landing center', 'red',     'sq'),
            ('Landing radius', '#cc0000', 'circ'),
            ('CEP (50%)',      '#9933cc', 'circ'),
            ('Error ellipse',  '#00bb00', 'oval'),
            ('KDE contours',   '#ff8800', 'band'),
        ]
        lh, pad, iw, tw = 17, 5, 14, 108
        w = pad * 2 + iw + 6 + tw
        h = len(items) * lh + pad * 2
        c.config(width=w, height=h)
        for i, (label, color, style) in enumerate(items):
            ym = pad + i * lh + lh // 2
            x0, x1 = pad, pad + iw
            if style == 'sq':
                c.create_rectangle(x0 + 2, ym - 4, x1 - 2, ym + 4,
                                   fill=color, outline=color)
            elif style == 'circ':
                c.create_oval(x0, ym - 5, x1, ym + 5,
                              fill='', outline=color, width=2)
            elif style == 'oval':
                c.create_oval(x0, ym - 4, x1, ym + 4,
                              fill='', outline=color, width=2)
            elif style == 'band':
                light = self._alpha_hex(color, 0.55)
                c.create_rectangle(x0, ym - 4, x1, ym + 4,
                                   fill=light, outline=color, width=1)
            c.create_text(x1 + 6, ym, text=label, anchor='w',
                          font=('Arial', 8), fill='#111111')

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
        launch_lat:         float,
        launch_lon:         float,
        land_lat:           float,
        land_lon:           float,
        r_target:           float,
        r90:                float,
        has_sim_result:     bool,
        p2_ellipse:         Optional[dict] = None,
        kde_contours=None,
        mc_cep:             Optional[float] = None,
        mc_ellipse_polygon: Optional[list]  = None,
        mc_cep_polygon:     Optional[dict]  = None,
        auto_fit:           bool = True,
    ) -> None:
        # Clear previous KDE label markers and all polygons
        for m in self._label_markers:
            try:
                m.delete()
            except Exception:
                pass
        self._label_markers = []
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

        # CEP (50%) circle — prefer pre-computed polygon (centroid-centred),
        # fall back to scalar radius + nominal landing position.
        if has_sim_result:
            cep_latlons = (mc_cep_polygon.get('latlons')
                           if mc_cep_polygon is not None else None)
            if cep_latlons and len(cep_latlons) >= 3:
                cep_fill = self._alpha_hex('#9933cc', 0.85)
                self.map_widget.set_polygon(
                    cep_latlons, outline_color='#9933cc',
                    fill_color=cep_fill, border_width=2)
            elif mc_cep is not None and mc_cep > 0:
                cep_fill = self._alpha_hex('#9933cc', 0.85)
                self.map_widget.set_polygon(
                    self._circle_coords(land_lat, land_lon, mc_cep),
                    outline_color='#9933cc', fill_color=cep_fill, border_width=2)

        # MC 90 % error ellipse — drawn as a pre-computed (lat, lon) polygon.
        # All statistics were performed in the metric East-North frame inside
        # compute_error_ellipse_polygon; no degree-scale covariance is involved.
        if has_sim_result and mc_ellipse_polygon and len(mc_ellipse_polygon) >= 3:
            ell_fill = self._alpha_hex('#00bb00', 0.55)
            self.map_widget.set_polygon(
                mc_ellipse_polygon,
                outline_color='#00bb00', fill_color=ell_fill, border_width=2)

        # Phase 2 live error ellipse (GO/NO-GO, only active after Phase 1).
        # cx/cy are metre offsets from the LAUNCH point — use launch_lat/lon
        # as the reference, not land_lat/lon, to avoid doubling the offset.
        if p2_ellipse is not None:
            color = '#00bb00' if p2_ellipse['go'] else '#dd0000'
            fill  = self._alpha_hex(color, 0.62)
            coords = self._ellipse_polygon(
                launch_lat, launch_lon,
                p2_ellipse['cx'], p2_ellipse['cy'],
                p2_ellipse['a'],  p2_ellipse['b'],
                p2_ellipse['angle_rad'])
            self.map_widget.set_polygon(
                coords, outline_color=color, fill_color=fill, border_width=2)

        # MC KDE probability-mass contours (heatmap-style: outer → inner)
        # Each item: (latlons, color, border_width[, pct_label])
        if kde_contours:
            n_cont = len(kde_contours)
            for i, item in enumerate(kde_contours):
                latlons   = item[0]
                col       = item[1]
                bw        = item[2]
                pct_label = item[3] if len(item) > 3 else None
                if len(latlons) >= 3:
                    lightness = 0.72 - 0.38 * (i / max(n_cont - 1, 1))
                    fill_col  = self._alpha_hex(col, lightness)
                    self.map_widget.set_polygon(
                        latlons, outline_color=col, fill_color=fill_col, border_width=bw)
                    if pct_label:
                        # Place label marker at northernmost point of contour
                        label_pt = max(latlons, key=lambda p: p[0])
                        try:
                            m = self.map_widget.set_marker(
                                label_pt[0], label_pt[1], text=pct_label)
                            self._label_markers.append(m)
                        except Exception:
                            pass

        # Auto-fit map view to contain all drawn elements
        if auto_fit and has_sim_result:
            self.fit_bounds(
                launch_lat=launch_lat, launch_lon=launch_lon,
                land_lat=land_lat,     land_lon=land_lon,
                r90=r90,               has_sim_result=has_sim_result,
            )

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

    @staticmethod
    def _alpha_hex(hex_color: str, lightness: float = 0.55) -> str:
        """Blend hex_color toward white; lightness 0 = original, 1 = white."""
        h = hex_color.lstrip('#')
        if len(h) == 3:
            h = h[0] * 2 + h[1] * 2 + h[2] * 2
        if len(h) != 6:
            return hex_color
        try:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        except ValueError:
            return hex_color
        return (f'#{min(255, int(r + (255 - r) * lightness)):02x}'
                f'{min(255, int(g + (255 - g) * lightness)):02x}'
                f'{min(255, int(b + (255 - b) * lightness)):02x}')
