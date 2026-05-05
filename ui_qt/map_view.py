"""
ui_qt/map_view.py

PySide6 QGraphicsView-based simulation overlay map.

Coordinate system
-----------------
Scene origin (0, 0) = launch site.
X increases East (metres), scene-Y increases South (Qt default), so all
metric North values are stored as −north_m in the scene.  The helper
_scene_pt() handles this negation uniformly.

Ghosting prevention
-------------------
Subscribes to AppState.is_calculating_changed.
  True  → call scene.removeItem() on every item in _sim_items, then clear the
           list.  Because scene.removeItem() keeps the C++ object alive until
           the Python wrapper's refcount reaches zero, clearing the list
           immediately after removeItem() ensures deterministic destruction
           without dangling-pointer segfaults.
  False → no-op; the imminent simulation_result_changed signal supplies the
           fresh overlay payload.

Thread safety
-------------
is_calculating_changed is always emitted from the main thread (SimController).
Qt direct connections therefore keep all scene operations on the main thread.
"""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import Qt, QPointF, QRectF, Slot
from PySide6.QtGui import (
    QBrush, QColor, QPainter, QPen, QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui_qt.app_state import AppState


# ── Zoomable QGraphicsView subclass ──────────────────────────────────────────

class _ZoomView(QGraphicsView):
    """QGraphicsView with mouse-wheel zoom centred on the cursor."""

    _ZOOM_FACTOR = 1.18

    def wheelEvent(self, event) -> None:
        factor = (self._ZOOM_FACTOR
                  if event.angleDelta().y() > 0
                  else 1.0 / self._ZOOM_FACTOR)
        self.scale(factor, factor)


# ── MapView ───────────────────────────────────────────────────────────────────

class MapView(QWidget):
    """
    Metric-coordinate overlay map.

    Static items (launch cross-hair, target-radius ring) are added once and
    never removed.  All simulation-specific items (landing dot, error circles,
    ellipse, scatter) are tracked in self._sim_items and removed atomically
    when is_calculating_changed(True) fires.
    """

    # Cosmetic sizes (scene units = metres)
    _LAUNCH_CROSS = 10.0   # half-length of launch cross-hair arms
    _SCATTER_R    = 2.0    # radius of each scatter dot
    _LANDING_R    = 4.0    # radius of landing centre dot

    def __init__(self, app_state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state    = app_state
        # Geographic coordinates of the last known landing position.
        # Populated by update_landing(); used for logging/future tile-map support.
        self._land_lat: float = app_state.launch_lat
        self._land_lon: float = app_state.launch_lon

        self._scene    = QGraphicsScene(self)
        self._view     = _ZoomView(self._scene, self)

        # Items that persist across runs (launch marker, target ring).
        self._static_items: list[QGraphicsItem] = []
        # Items that represent the *current* simulation result.
        # Cleared atomically when is_calculating_changed(True) fires.
        self._sim_items: list[QGraphicsItem] = []

        self._build_ui()
        self._draw_static_items(app_state.target_radius)

        app_state.is_calculating_changed.connect(self._on_calculating_changed)
        app_state.simulation_result_changed.connect(self._on_simulation_result)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._view.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setBackgroundBrush(QBrush(QColor('#1e1e2e')))
        self._view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Initial scene rect (±200 m around launch site)
        self._view.setSceneRect(-200, -200, 400, 400)

        self._info = QLabel("No simulation result.")
        self._info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info.setStyleSheet(
            "QLabel { font-size: 11px; font-weight: bold; padding: 4px; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view, 1)
        layout.addWidget(self._info)

    def _draw_static_items(self, target_radius: float) -> None:
        """Add the launch cross-hair and target-radius ring (never cleared)."""
        h   = self._LAUNCH_CROSS
        pen = QPen(QColor('#4488ff'), 2.0)
        pen.setCosmetic(True)   # line width stays constant across zoom levels
        cross_h = self._scene.addLine(-h, 0, h, 0, pen)
        cross_v = self._scene.addLine(0, -h, 0, h, pen)
        self._static_items.extend([cross_h, cross_v])

        # Launch dot
        dot = QGraphicsEllipseItem(-4, -4, 8, 8)
        dot_pen = QPen(QColor('#4488ff'), 1.5)
        dot_pen.setCosmetic(True)
        dot.setPen(dot_pen)
        dot.setBrush(QBrush(QColor('#2255cc')))
        self._scene.addItem(dot)
        self._static_items.append(dot)

        # Target-radius ring (dashed blue)
        if target_radius > 0:
            r = float(target_radius)
            ring = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            ring_pen = QPen(QColor('#0055ff'), 1.5, Qt.PenStyle.DashLine)
            ring.setPen(ring_pen)
            ring.setBrush(Qt.BrushStyle.NoBrush)
            self._scene.addItem(ring)
            self._static_items.append(ring)

    # ── Public API called by SimController ────────────────────────────────────

    def update_landing(self, lat: float, lon: float) -> None:
        """Store the geographic landing position (called before full render)."""
        self._land_lat = float(lat)
        self._land_lon = float(lon)

    # ── Ghosting prevention ───────────────────────────────────────────────────

    @Slot(bool)
    def _on_calculating_changed(self, calculating: bool) -> None:
        """
        Called synchronously on the main thread via Qt direct connection.

        True  → remove every item in _sim_items from the scene, then drop all
                Python references.  The C++ QGraphicsItem objects are destroyed
                when their refcount reaches zero (immediately after list.clear()).
        False → no-op; _on_simulation_result handles the incoming payload.
        """
        if not calculating:
            return

        for item in self._sim_items:
            self._scene.removeItem(item)
        # Drop all Python references → C++ objects destroyed deterministically.
        self._sim_items.clear()

        self._info.setText("Calculating…")

    # ── Simulation result rendering ───────────────────────────────────────────

    @Slot(object)
    def _on_simulation_result(self, result: Optional[dict]) -> None:
        if result is None or result.get('cancelled'):
            self._info.setText("Simulation cancelled or no result.")
            return
        self._render_result(result)

    def _render_result(self, result: dict) -> None:
        """
        Draw one complete simulation result onto the scene.

        All created items are appended to _sim_items so they are cleared
        atomically on the next _on_calculating_changed(True) call.
        """
        # Defensive clear (handles the rare case where is_calculating_changed
        # was not received before this signal).
        for item in self._sim_items:
            self._scene.removeItem(item)
        self._sim_items.clear()

        impact_x = float(result.get('impact_x',  0.0))
        impact_y = float(result.get('impact_y',  0.0))
        r90      = float(result.get('r_N_radius', 0.0))
        cep      = float(result.get('cep',        0.0))
        scatter  = result.get('scatter',  [])
        ellipse  = result.get('ellipse')
        prob     = int(result.get('landing_prob', 90))
        apogee   = float(result.get('apogee_m',   0.0))
        tof      = float(result.get('hang_time',  0.0))

        # Landing dot
        self._add_circle(impact_x, impact_y, self._LANDING_R,
                         fill='#ff4444', outline='#cc0000', lw=1.5)

        # R-N% landing radius circle
        if r90 > 0:
            self._add_circle(impact_x, impact_y, r90,
                             fill=None, outline='#cc0000', lw=2.0)

        # CEP 50% circle (dashed purple)
        if cep > 0:
            self._add_circle(impact_x, impact_y, cep,
                             fill=None, outline='#9933cc', lw=1.8,
                             style=Qt.PenStyle.DashLine)

        # Error ellipse (filled green polygon)
        if ellipse:
            pts = _ellipse_points(
                impact_x, impact_y,
                ellipse['a'], ellipse['b'], ellipse['angle_rad'])
            self._add_polygon(pts, fill='#00bb0030', outline='#00bb00', lw=2.0)

        # MC scatter dots (first 100)
        for px, py in scatter[:100]:
            self._add_circle(float(px), float(py), self._SCATTER_R,
                             fill='#ff880060', outline=None, lw=0)

        # Info label
        self._info.setText(
            f"R{prob}: {r90:.1f} m  |  CEP50: {cep:.1f} m  |  "
            f"Apogee: {apogee:.0f} m  |  ToF: {tof:.1f} s")

        # Auto-fit view to show the landing area with generous padding
        pad  = max(r90, cep, 30.0) * 0.35
        half = max(r90, cep, 30.0) + pad
        rect = QRectF(impact_x - half, -impact_y - half, 2 * half, 2 * half)
        self._view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    # ── Scene helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _scene_pt(east_m: float, north_m: float) -> QPointF:
        """Convert ENU metric offset to scene coordinates (Y-negated for North-up)."""
        return QPointF(east_m, -north_m)

    def _add_circle(
        self,
        east_m: float, north_m: float, r: float,
        fill: Optional[str],
        outline: Optional[str],
        lw: float,
        style: Qt.PenStyle = Qt.PenStyle.SolidLine,
    ) -> None:
        """Add a circle centred at (east_m, north_m) with radius r metres."""
        # Bounding-rect top-left in scene coords (Y-negated)
        item = QGraphicsEllipseItem(east_m - r, -north_m - r, 2 * r, 2 * r)
        item.setBrush(QBrush(QColor(fill)) if fill else Qt.BrushStyle.NoBrush)
        if outline:
            pen = QPen(QColor(outline), lw)
            pen.setStyle(style)
            item.setPen(pen)
        else:
            item.setPen(Qt.PenStyle.NoPen)
        self._scene.addItem(item)
        self._sim_items.append(item)

    def _add_polygon(
        self,
        points: list[tuple[float, float]],
        fill: Optional[str],
        outline: Optional[str],
        lw: float,
    ) -> None:
        """Add a closed polygon from a list of (east_m, north_m) metric pairs."""
        poly = QPolygonF([QPointF(px, -py) for px, py in points])
        item = QGraphicsPolygonItem(poly)
        item.setBrush(QBrush(QColor(fill)) if fill else Qt.BrushStyle.NoBrush)
        item.setPen(QPen(QColor(outline), lw) if outline else Qt.PenStyle.NoPen)
        self._scene.addItem(item)
        self._sim_items.append(item)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _ellipse_points(
    cx: float, cy: float,
    a: float, b: float,
    angle_rad: float,
    n: int = 64,
) -> list[tuple[float, float]]:
    """Sample n points on an ellipse in the ENU metric plane."""
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    pts = []
    for i in range(n):
        t  = 2.0 * math.pi * i / n
        xe = a * math.cos(t) * ca - b * math.sin(t) * sa + cx
        ye = a * math.cos(t) * sa + b * math.sin(t) * ca + cy
        pts.append((xe, ye))
    return pts
