"""
ui_qt/app_window.py
PySide6 / Qt6 main window for the Kazamidori Project.

Migration target from ui/app_window.py (Tkinter).
The existing ui/ directory is intentionally left untouched.

Layout
------
MainWindow (QMainWindow)
  ├── MenuBar       — File / Simulation / View / Help
  ├── MainToolBar   — Run · MC · Phase 1 · Stop · Center Map · progress bar
  ├── StatusBar     — left: status text  |  right: live wind readout
  ├── Central       — _MapWidget  (grid placeholder → future QWebEngineView / Folium)
  └── DockWidgets   (floatable · draggable · re-arrangeable):
      ├── ParametersDock  LEFT  — launch site, vehicle, wind, MC, controls
      └── ProfileDock     RIGHT — 3-D trajectory profile (FigureCanvasQTAgg)

Standalone preview:
    python -m ui_qt.app_window
"""

from __future__ import annotations

import sys
from typing import Optional

# ── Matplotlib Qt backend — must be set before any pyplot import ──────────────
import matplotlib
matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QSize, QPointF
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QScrollArea,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox,
    QComboBox, QPushButton, QToolBar, QStatusBar,
    QSizePolicy, QProgressBar, QFrame, QFileDialog,
    QMessageBox,
)
from PySide6.QtGui import (
    QAction, QPainter, QPen, QColor, QLinearGradient,
)


# ── Catppuccin Mocha dark palette ─────────────────────────────────────────────
_QSS = """
/* ── Global ─────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "SF Pro Text", Arial, sans-serif;
    font-size: 11px;
}

/* ── Dock widgets ────────────────────────────────────────── */
QDockWidget {
    color: #cdd6f4;
    font-weight: bold;
}
QDockWidget::title {
    background: #313244;
    padding: 5px 10px;
    border-bottom: 2px solid #89b4fa;
    text-align: left;
}
QDockWidget::close-button, QDockWidget::float-button {
    border: none;
    background: transparent;
    padding: 2px;
}

/* ── Group boxes ─────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
    font-weight: bold;
    font-size: 10px;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    background-color: #1e1e2e;
}

/* ── Input widgets ───────────────────────────────────────── */
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 3px 6px;
    color: #cdd6f4;
    min-width: 80px;
}
QLineEdit:focus, QDoubleSpinBox:focus,
QSpinBox:focus,  QComboBox:focus {
    border-color: #89b4fa;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button,       QSpinBox::down-button {
    background: #45475a;
    border: none;
    width: 16px;
    border-radius: 2px;
}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
QSpinBox::up-button:hover,       QSpinBox::down-button:hover {
    background: #585b70;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #313244;
    border: 1px solid #45475a;
    selection-background-color: #45475a;
    color: #cdd6f4;
    outline: none;
}

/* ── Push buttons ────────────────────────────────────────── */
QPushButton {
    background: #313244;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 5px 14px;
    color: #cdd6f4;
    font-weight: bold;
}
QPushButton:hover   { background: #45475a; border-color: #89b4fa; }
QPushButton:pressed { background: #89b4fa; color: #1e1e2e; }

QPushButton#btn_run  { background: #a6e3a1; color: #1e1e2e; border-color: #a6e3a1; }
QPushButton#btn_run:hover  { background: #94e2d5; }
QPushButton#btn_mc   { background: #89b4fa; color: #1e1e2e; border-color: #89b4fa; }
QPushButton#btn_mc:hover   { background: #b4befe; }
QPushButton#btn_stop { background: #f38ba8; color: #1e1e2e; border-color: #f38ba8; }
QPushButton#btn_stop:hover { background: #eba0ac; }

/* ── Tool bar ────────────────────────────────────────────── */
QToolBar {
    background: #181825;
    border: none;
    border-bottom: 1px solid #313244;
    padding: 3px 6px;
    spacing: 4px;
}
QToolBar QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 3px 8px;
    color: #cdd6f4;
}
QToolBar QToolButton:hover   { background: #313244; border-color: #45475a; }
QToolBar QToolButton:pressed { background: #45475a; }

/* ── Menu bar / menus ────────────────────────────────────── */
QMenuBar {
    background: #181825;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
}
QMenuBar::item { padding: 5px 12px; background: transparent; }
QMenuBar::item:selected { background: #313244; border-radius: 3px; }
QMenu {
    background: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px;
}
QMenu::item { padding: 5px 20px 5px 12px; border-radius: 3px; }
QMenu::item:selected { background: #313244; color: #89b4fa; }
QMenu::separator { height: 1px; background: #45475a; margin: 3px 8px; }

/* ── Status bar ──────────────────────────────────────────── */
QStatusBar {
    background: #181825;
    color: #a6adc8;
    border-top: 1px solid #313244;
    font-size: 10px;
}
QStatusBar::item { border: none; }

/* ── Scroll bars ─────────────────────────────────────────── */
QScrollBar:vertical {
    background: #1e1e2e;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #585b70; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #1e1e2e;
    height: 8px;
}
QScrollBar::handle:horizontal {
    background: #45475a;
    border-radius: 4px;
    min-width: 24px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ── Progress bar ────────────────────────────────────────── */
QProgressBar {
    background: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    color: #cdd6f4;
    font-size: 10px;
    max-height: 18px;
}
QProgressBar::chunk { background: #89b4fa; border-radius: 3px; }

/* ── Scroll area ─────────────────────────────────────────── */
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: #1e1e2e; }

/* ── Form labels ─────────────────────────────────────────── */
QFormLayout QLabel { color: #a6adc8; }
"""


# ── Matplotlib canvas wrapper ─────────────────────────────────────────────────

class _MplCanvas(FigureCanvasQTAgg):
    """Thin FigureCanvasQTAgg with a sensible expanding size policy."""

    def __init__(self, fig: Figure,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.updateGeometry()


# ── 3-D axes style helper ─────────────────────────────────────────────────────

def _style_3d(ax, fig: Optional[Figure] = None) -> None:
    """Apply Catppuccin Mocha dark styling to a 3-D Axes3D."""
    ax.set_facecolor("#313244")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#45475a")
    ax.tick_params(colors="#a6adc8", labelsize=7)
    if fig is not None:
        fig.patch.set_facecolor("#1e1e2e")


# ── Map placeholder HTML (used when QWebEngineView is available) ──────────────

_MAP_PLACEHOLDER_HTML: str = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body {
    background: #0d0d1a;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh;
    font-family: 'Segoe UI', Arial, sans-serif;
    overflow: hidden;
}
.grid {
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(#1c1c2e 1px, transparent 1px),
        linear-gradient(90deg, #1c1c2e 1px, transparent 1px),
        linear-gradient(#161622 1px, transparent 1px),
        linear-gradient(90deg, #161622 1px, transparent 1px);
    background-size: 100px 100px, 100px 100px, 20px 20px, 20px 20px;
}
.content { position: relative; text-align: center; padding: 32px; }
.icon    { font-size: 64px; margin-bottom: 16px; }
h2       { color: #4a4c6a; font-size: 17px; margin-bottom: 12px;
           font-weight: 600; letter-spacing: 0.5px; }
p        { color: #353550; font-size: 12px; line-height: 2.0; }
.accent  { color: #89b4fa; }
.muted   { color: #2a2a40; }
</style>
</head>
<body>
<div class="grid"></div>
<div class="content">
    <div class="icon">🗺</div>
    <h2>Map View</h2>
    <p>
        Integration target:<br>
        <span class="accent">QWebEngineView + Leaflet.js / Folium</span>
    </p>
    <p style="margin-top: 14px;" class="muted">
        Launch site · Target radius · Predicted landing<br>
        CEP circles · KDE probability contours · Error ellipses
    </p>
</div>
</body>
</html>"""


# ── Map placeholder fallback widget (QPainter grid) ──────────────────────────

class _MapPlaceholderGrid(QWidget):
    """
    Fallback map placeholder rendered with QPainter.

    Used when ``PySide6.QtWebEngineWidgets`` is not installed.
    A dark gradient grid fills the widget; a centred, transparent QLabel
    carries the HTML info overlay so the grid shows through.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(400, 300)

        self._overlay = QLabel(self)
        self._overlay.setTextFormat(Qt.TextFormat.RichText)
        self._overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay.setStyleSheet("background: transparent;")
        self._overlay.setText(
            "<div style='text-align:center; line-height:2.2;'>"
            "<p style='font-size:52px;'>🗺</p>"
            "<p style='font-size:15px; color:#4a4c6a; font-weight:600;"
            "   margin-top:10px;'>Map View</p>"
            "<p style='font-size:10px; color:#353550; margin-top:6px;'>"
            "Integration target: QWebEngineView + Leaflet.js / Folium</p>"
            "<p style='font-size:9px; color:#2c2c42; margin-top:4px;'>"
            "Launch site · Target radius · Predicted landing<br>"
            "CEP circles · KDE contours · Error ellipses</p>"
            "</div>"
        )

    # Keep the overlay label filling the widget on resize.
    def resizeEvent(self, event) -> None:
        self._overlay.setGeometry(0, 0, self.width(), self.height())
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        w, h = self.width(), self.height()

        # Dark gradient background
        grad = QLinearGradient(QPointF(0, 0), QPointF(0, h))
        grad.setColorAt(0.0, QColor("#0d0d1a"))
        grad.setColorAt(1.0, QColor("#13131f"))
        p.fillRect(self.rect(), grad)

        # Minor grid (20 px pitch)
        p.setPen(QPen(QColor("#161622"), 1))
        for x in range(0, w, 20):
            p.drawLine(x, 0, x, h)
        for y in range(0, h, 20):
            p.drawLine(0, y, w, y)

        # Major grid (100 px pitch)
        p.setPen(QPen(QColor("#1c1c2e"), 1))
        for x in range(0, w, 100):
            p.drawLine(x, 0, x, h)
        for y in range(0, h, 100):
            p.drawLine(0, y, w, y)

        # Dashed crosshair at widget centre
        cx, cy = w // 2, h // 2
        p.setPen(QPen(QColor("#252540"), 1, Qt.PenStyle.DashLine))
        p.drawLine(cx, 0, cx, h)
        p.drawLine(0, cy, w, cy)

        # Symbolic launch-site indicator
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor("#3a3c5a"), 1, Qt.PenStyle.DotLine))
        p.setBrush(QColor(0, 0, 0, 0))
        p.drawEllipse(QPointF(cx, cy), 13.0, 13.0)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#3a3c5a"))
        p.drawEllipse(QPointF(cx, cy), 3.5, 3.5)


# ── Central map widget ────────────────────────────────────────────────────────

class _MapWidget(QWidget):
    """
    Central map area that fills the QMainWindow's central region.

    ┌──────────────────────────────────────────────────────────┐
    │  info bar (32 px, fixed)                                 │
    │  Launch: …°N, …°E              |    Landing: …          │
    ├──────────────────────────────────────────────────────────┤
    │  map body  (stretches to fill remaining space)           │
    │  → QWebEngineView   if PySide6.QtWebEngineWidgets found  │
    │  → _MapPlaceholderGrid          otherwise                │
    └──────────────────────────────────────────────────────────┘

    Public API
    ----------
    update_launch(lat, lon)   Refresh the launch coordinate in the info bar.
    update_landing(lat, lon)  Refresh the predicted landing coordinate.
    clear_landing()           Reset landing readout to "—".
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_info_bar())
        root.addWidget(self._build_map_body(), stretch=1)

    def _build_info_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("MapInfoBar")
        bar.setFixedHeight(32)
        bar.setStyleSheet(
            "QFrame#MapInfoBar {"
            "  background: #181825;"
            "  border-bottom: 1px solid #313244;"
            "}"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(6)

        self.launch_label = QLabel("Launch:  35.682800°N, 139.759000°E")
        self.launch_label.setStyleSheet(
            "color: #89b4fa; font-size: 11px; background: transparent;")

        _sep = QLabel("|")
        _sep.setStyleSheet("color: #45475a; background: transparent;")

        self.landing_label = QLabel("Landing:  —")
        self.landing_label.setStyleSheet(
            "color: #f38ba8; font-size: 11px; background: transparent;")

        lay.addWidget(self.launch_label)
        lay.addStretch()
        lay.addWidget(_sep)
        lay.addStretch()
        lay.addWidget(self.landing_label)
        return bar

    def _build_map_body(self) -> QWidget:
        """Return a QWebEngineView (if available) or the QPainter fallback."""
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore
            view = QWebEngineView()
            view.setHtml(_MAP_PLACEHOLDER_HTML)
            self._web_view = view
            return view
        except (ImportError, RuntimeError):
            return _MapPlaceholderGrid()

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_launch(self, lat: float, lon: float) -> None:
        self.launch_label.setText(f"Launch:  {lat:.6f}°N, {lon:.6f}°E")

    def update_landing(self, lat: float, lon: float) -> None:
        self.landing_label.setText(f"Landing:  {lat:.6f}°N, {lon:.6f}°E")

    def clear_landing(self) -> None:
        self.landing_label.setText("Landing:  —")


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """
    Top-level PySide6 window for the Kazamidori Project.

    All parameter input widgets are public instance attributes so a future
    AppController can read / write them without touching the widget hierarchy.

    Public widget attributes
    ------------------------
    lat_input, lon_input            : QDoubleSpinBox
    elev_input, azim_input          : QDoubleSpinBox
    motor_label                     : QLabel   (loaded filename)
    mode_combo                      : QComboBox
    rmax_input                      : QDoubleSpinBox
    surf_spd_input, surf_dir_input  : QDoubleSpinBox
    up_spd_input,   up_dir_input    : QDoubleSpinBox
    mc_runs_input                   : QSpinBox
    landing_prob_combo              : QComboBox
    wind_unc_input                  : QDoubleSpinBox
    thrust_unc_input                : QDoubleSpinBox
    allow_unc_input                 : QDoubleSpinBox
    map_widget                      : _MapWidget

    Figure / canvas attributes
    --------------------------
    profile_fig     : Figure         (3-D axes owner)
    profile_ax      : Axes3D         (3-D subplot)
    profile_canvas  : _MplCanvas     (Qt canvas)
    """

    OPERATION_MODES = (
        "Altitude Competition",
        "Precision Landing",
        "Winged Hover",
        "Free",
    )
    LANDING_PROBS = (50, 68, 80, 85, 90, 95, 99)

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        _cfg = config or {}

        self.setWindowTitle(
            "Kazamidori  —  Trajectory & Landing Simulator  [Qt6 / PySide6]"
        )
        self.resize(1440, 920)
        self.setMinimumSize(920, 620)

        self._apply_theme()
        self._build_figures()
        self._build_menu_bar()
        self._build_tool_bar()
        self._build_status_bar()
        self._build_central_widget()   # ← must precede _build_docks (map_widget ref)
        self._build_docks()
        self._populate_profile_placeholder()
        self._set_dock_sizes()
        self._dock_params.raise_()

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        self.setStyleSheet(_QSS)

    # ── Figure construction ───────────────────────────────────────────────────

    def _build_figures(self) -> None:
        self.profile_fig    = Figure(figsize=(6, 8), facecolor="#1e1e2e")
        self.profile_ax     = self.profile_fig.add_subplot(111, projection="3d")
        self.profile_canvas = _MplCanvas(self.profile_fig)

    # ── Menu bar ──────────────────────────────────────────────────────────────

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        # File
        fm = mb.addMenu("&File")
        fm.addAction(QAction("Load Motor File…", self,
                             triggered=self._on_load_motor))
        fm.addAction(QAction("Export Results…",  self))
        fm.addSeparator()
        fm.addAction(QAction("Quit", self, triggered=self.close))

        # Simulation
        sm = mb.addMenu("&Simulation")
        sm.addAction(QAction("▶  Run Simulation",   self, triggered=self._on_run))
        sm.addAction(QAction("🎲  Monte Carlo",     self, triggered=self._on_mc))
        sm.addAction(QAction("🔍  Phase 1 Optimize",self, triggered=self._on_phase1))
        sm.addAction(QAction("⏹  Stop",             self, triggered=self._on_stop))
        sm.addSeparator()
        sm.addAction(QAction("🗺  Center Map",       self, triggered=self._on_center_map))

        # View  (dock toggles injected in _build_docks)
        self._view_menu = mb.addMenu("&View")

        # Help
        hm = mb.addMenu("&Help")
        hm.addAction(QAction("About Kazamidori", self, triggered=self._on_about))

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_tool_bar(self) -> None:
        tb = QToolBar("Main Toolbar", self)
        tb.setObjectName("MainToolBar")
        tb.setMovable(False)
        tb.setFloatable(False)

        def _vline() -> None:
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setFrameShadow(QFrame.Shadow.Sunken)
            sep.setStyleSheet("color: #45475a;")
            tb.addWidget(sep)

        btn_run = QPushButton("▶  Run")
        btn_run.setObjectName("btn_run")
        btn_run.setFixedWidth(90)
        btn_run.clicked.connect(self._on_run)

        btn_mc = QPushButton("🎲  MC")
        btn_mc.setObjectName("btn_mc")
        btn_mc.setFixedWidth(78)
        btn_mc.clicked.connect(self._on_mc)

        btn_ph1 = QPushButton("🔍  Phase 1")
        btn_ph1.setFixedWidth(94)
        btn_ph1.clicked.connect(self._on_phase1)

        btn_stop = QPushButton("⏹  Stop")
        btn_stop.setObjectName("btn_stop")
        btn_stop.setFixedWidth(74)
        btn_stop.clicked.connect(self._on_stop)

        btn_map = QPushButton("🗺  Center Map")
        btn_map.setFixedWidth(112)
        btn_map.clicked.connect(self._on_center_map)

        for w in (btn_run, btn_mc, btn_ph1):
            tb.addWidget(w)
        _vline()
        tb.addWidget(btn_stop)
        _vline()
        tb.addWidget(btn_map)

        spacer = QWidget()
        spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(172)
        self._progress.setValue(0)
        self._progress.setFormat("Idle")
        self._progress.setTextVisible(True)
        tb.addWidget(self._progress)

        self.addToolBar(tb)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        sb = QStatusBar(self)
        self.setStatusBar(sb)

        self._status_label = QLabel("Ready")
        self._status_label.setContentsMargins(8, 0, 8, 0)

        self._wind_status = QLabel(
            "Surface: -- m/s @ --°   |   Upper: -- m/s @ --°")
        self._wind_status.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._wind_status.setContentsMargins(8, 0, 8, 0)
        self._wind_status.setStyleSheet("color: #89b4fa;")

        sb.addWidget(self._status_label, stretch=1)
        sb.addPermanentWidget(self._wind_status)

    # ── Central widget — Map ──────────────────────────────────────────────────

    def _build_central_widget(self) -> None:
        self.map_widget = _MapWidget()
        self.setCentralWidget(self.map_widget)

    # ── Dock widgets ──────────────────────────────────────────────────────────

    def _build_docks(self) -> None:
        _ALL = Qt.DockWidgetArea.AllDockWidgetAreas

        # ── LEFT: Parameters ─────────────────────────────────────────────────
        self._dock_params = QDockWidget("Parameters", self)
        self._dock_params.setObjectName("ParametersDock")
        self._dock_params.setAllowedAreas(_ALL)
        self._dock_params.setWidget(self._build_parameters_panel())
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self._dock_params)

        # ── RIGHT: 3-D Flight Profile ────────────────────────────────────────
        self._dock_profile = QDockWidget("Flight Profile  (3D)", self)
        self._dock_profile.setObjectName("ProfileDock")
        self._dock_profile.setAllowedAreas(_ALL)

        pframe = QWidget()
        pl = QVBoxLayout(pframe)
        pl.setContentsMargins(2, 2, 2, 2)
        pl.setSpacing(0)
        nav = NavigationToolbar2QT(self.profile_canvas, pframe)
        nav.setIconSize(QSize(16, 16))
        pl.addWidget(nav)
        pl.addWidget(self.profile_canvas)

        self._dock_profile.setWidget(pframe)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self._dock_profile)

        # Populate View menu with dock toggle actions
        for dock in (self._dock_params, self._dock_profile):
            self._view_menu.addAction(dock.toggleViewAction())

    # ── Parameters panel ──────────────────────────────────────────────────────

    def _build_parameters_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Launch Site ───────────────────────────────────────────────────────
        grp_site = QGroupBox("Launch Site")
        frm = QFormLayout(grp_site)
        frm.setSpacing(5)

        self.lat_input = QDoubleSpinBox()
        self.lat_input.setRange(-90, 90)
        self.lat_input.setDecimals(6)
        self.lat_input.setValue(35.682800)
        self.lat_input.setSuffix("°")
        self.lat_input.setToolTip("Launch site latitude")
        self.lat_input.valueChanged.connect(
            lambda v: self.map_widget.update_launch(
                v, self.lon_input.value()))

        self.lon_input = QDoubleSpinBox()
        self.lon_input.setRange(-180, 180)
        self.lon_input.setDecimals(6)
        self.lon_input.setValue(139.759000)
        self.lon_input.setSuffix("°")
        self.lon_input.setToolTip("Launch site longitude")
        self.lon_input.valueChanged.connect(
            lambda v: self.map_widget.update_launch(
                self.lat_input.value(), v))

        btn_loc = QPushButton("📍  Get Current Location")
        btn_loc.clicked.connect(self._on_get_location)

        frm.addRow("Latitude:",  self.lat_input)
        frm.addRow("Longitude:", self.lon_input)
        frm.addRow("",           btn_loc)
        layout.addWidget(grp_site)

        # ── Vehicle & Launch Angle ────────────────────────────────────────────
        grp_veh = QGroupBox("Vehicle & Launch Angle")
        frm2 = QFormLayout(grp_veh)
        frm2.setSpacing(5)

        self.elev_input = QDoubleSpinBox()
        self.elev_input.setRange(0, 90)
        self.elev_input.setDecimals(1)
        self.elev_input.setValue(85.0)
        self.elev_input.setSuffix("°")
        self.elev_input.setToolTip("Launch rail elevation angle")

        self.azim_input = QDoubleSpinBox()
        self.azim_input.setRange(0, 360)
        self.azim_input.setDecimals(1)
        self.azim_input.setValue(0.0)
        self.azim_input.setSuffix("°")
        self.azim_input.setWrapping(True)
        self.azim_input.setToolTip("Launch rail azimuth (0° = North, CW)")

        self.motor_label = QLabel("(none selected)")
        self.motor_label.setStyleSheet("color: #fab387; font-style: italic;")

        btn_motor = QPushButton("📂  Load Motor File")
        btn_motor.clicked.connect(self._on_load_motor)

        frm2.addRow("Elevation:", self.elev_input)
        frm2.addRow("Azimuth:",   self.azim_input)
        frm2.addRow("Motor:",     self.motor_label)
        frm2.addRow("",           btn_motor)
        layout.addWidget(grp_veh)

        # ── Wind Parameters ───────────────────────────────────────────────────
        grp_wind = QGroupBox("Wind Parameters")
        frm3 = QFormLayout(grp_wind)
        frm3.setSpacing(5)

        self.surf_spd_input = QDoubleSpinBox()
        self.surf_spd_input.setRange(0, 50)
        self.surf_spd_input.setDecimals(1)
        self.surf_spd_input.setValue(4.0)
        self.surf_spd_input.setSuffix(" m/s")

        self.surf_dir_input = QDoubleSpinBox()
        self.surf_dir_input.setRange(0, 360)
        self.surf_dir_input.setDecimals(1)
        self.surf_dir_input.setValue(100.0)
        self.surf_dir_input.setSuffix("°")
        self.surf_dir_input.setWrapping(True)

        self.up_spd_input = QDoubleSpinBox()
        self.up_spd_input.setRange(0, 100)
        self.up_spd_input.setDecimals(1)
        self.up_spd_input.setValue(8.0)
        self.up_spd_input.setSuffix(" m/s")

        self.up_dir_input = QDoubleSpinBox()
        self.up_dir_input.setRange(0, 360)
        self.up_dir_input.setDecimals(1)
        self.up_dir_input.setValue(90.0)
        self.up_dir_input.setSuffix("°")
        self.up_dir_input.setWrapping(True)

        frm3.addRow("Surface Speed:", self.surf_spd_input)
        frm3.addRow("Surface From:",  self.surf_dir_input)
        frm3.addRow("Upper Speed:",   self.up_spd_input)
        frm3.addRow("Upper From:",    self.up_dir_input)
        layout.addWidget(grp_wind)

        # ── Operation Mode ────────────────────────────────────────────────────
        grp_mode = QGroupBox("Operation Mode")
        frm4 = QFormLayout(grp_mode)
        frm4.setSpacing(5)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.OPERATION_MODES)
        self.mode_combo.setCurrentText("Free")

        self._rmax_label = QLabel("R_max:")
        self.rmax_input  = QDoubleSpinBox()
        self.rmax_input.setRange(0, 9999)
        self.rmax_input.setDecimals(1)
        self.rmax_input.setValue(50.0)
        self.rmax_input.setSuffix(" m")

        frm4.addRow("Mode:",          self.mode_combo)
        frm4.addRow(self._rmax_label, self.rmax_input)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._on_mode_changed("Free")
        layout.addWidget(grp_mode)

        # ── Monte Carlo & Uncertainty ─────────────────────────────────────────
        grp_mc = QGroupBox("Monte Carlo & Uncertainty")
        frm5 = QFormLayout(grp_mc)
        frm5.setSpacing(5)

        self.mc_runs_input = QSpinBox()
        self.mc_runs_input.setRange(10, 5000)
        self.mc_runs_input.setValue(200)
        self.mc_runs_input.setSingleStep(50)

        self.landing_prob_combo = QComboBox()
        for p in self.LANDING_PROBS:
            self.landing_prob_combo.addItem(f"{p} %", p)
        self.landing_prob_combo.setCurrentIndex(4)   # default 90 %

        self.wind_unc_input = QDoubleSpinBox()
        self.wind_unc_input.setRange(0, 1)
        self.wind_unc_input.setDecimals(2)
        self.wind_unc_input.setValue(0.20)
        self.wind_unc_input.setSingleStep(0.01)
        self.wind_unc_input.setSuffix("  (±ratio)")

        self.thrust_unc_input = QDoubleSpinBox()
        self.thrust_unc_input.setRange(0, 1)
        self.thrust_unc_input.setDecimals(2)
        self.thrust_unc_input.setValue(0.05)
        self.thrust_unc_input.setSingleStep(0.01)
        self.thrust_unc_input.setSuffix("  (±ratio)")

        self.allow_unc_input = QDoubleSpinBox()
        self.allow_unc_input.setRange(0, 9999)
        self.allow_unc_input.setDecimals(1)
        self.allow_unc_input.setValue(20.0)
        self.allow_unc_input.setSuffix(" m")

        frm5.addRow("MC Runs:",            self.mc_runs_input)
        frm5.addRow("Landing Prob:",        self.landing_prob_combo)
        frm5.addRow("Wind Uncertainty:",    self.wind_unc_input)
        frm5.addRow("Thrust Uncertainty:",  self.thrust_unc_input)
        frm5.addRow("Allowable Radius:",    self.allow_unc_input)
        layout.addWidget(grp_mc)

        # ── Phase 2 GO / NO-GO indicator ─────────────────────────────────────
        grp_p2 = QGroupBox("Phase 2  GO / NO-GO")
        p2_vbox = QVBoxLayout(grp_p2)

        self._go_nogo_label = QLabel("●  STANDBY")
        self._go_nogo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._go_nogo_label.setStyleSheet(
            "font-size: 18px; font-weight: bold;"
            "color: #6c7086; padding: 8px;")
        p2_vbox.addWidget(self._go_nogo_label)
        layout.addWidget(grp_p2)

        # ── Simulation controls ───────────────────────────────────────────────
        grp_ctrl = QGroupBox("Simulation Controls")
        ctrl_vbox = QVBoxLayout(grp_ctrl)
        ctrl_vbox.setSpacing(6)

        for obj_name, label, slot in (
            ("btn_run",  "▶  Run Simulation",    self._on_run),
            ("btn_mc",   "🎲  Run Monte Carlo",  self._on_mc),
            ("",         "🔍  Phase 1 Optimize", self._on_phase1),
            ("btn_stop", "⏹  Stop",              self._on_stop),
        ):
            btn = QPushButton(label)
            if obj_name:
                btn.setObjectName(obj_name)
            btn.setMinimumHeight(30)
            btn.clicked.connect(slot)
            ctrl_vbox.addWidget(btn)

        layout.addWidget(grp_ctrl)
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    # ── 3-D profile placeholder content ──────────────────────────────────────

    def _populate_profile_placeholder(self) -> None:
        ax = self.profile_ax
        _style_3d(ax, self.profile_fig)

        # Launch-site marker at origin
        ax.scatter([0], [0], [0], c="#89b4fa", s=120, marker="^", zorder=5)

        # Faint directional reference arrows
        span = 60
        arrows = (
            ([0, span], [0, 0],    [0, 0],    "#f38ba8", "E"),
            ([0, 0],    [0, span], [0, 0],    "#a6e3a1", "N"),
            ([0, 0],    [0, 0],    [0, span], "#89b4fa", "Up"),
        )
        for xs, ys, zs, col, lbl in arrows:
            ax.plot(xs, ys, zs, color=col, lw=1.2, alpha=0.45,
                    linestyle="--")
            ax.text(xs[-1] * 1.06, ys[-1] * 1.06, zs[-1] * 1.06,
                    lbl, color=col, fontsize=7)

        ax.text2D(
            0.5, 0.44,
            "Run a simulation\nto display the 3D trajectory",
            transform=ax.transAxes, ha="center", va="center",
            color="#6c7086", fontsize=11, linespacing=1.8,
        )

        ax.set_xlim(-80, 80)
        ax.set_ylim(-80, 80)
        ax.set_zlim(0, 120)
        ax.set_xlabel("East  (m)",  color="#6c7086", fontsize=8, labelpad=4)
        ax.set_ylabel("North  (m)", color="#6c7086", fontsize=8, labelpad=4)
        ax.set_zlabel("Alt  (m)",   color="#6c7086", fontsize=8, labelpad=4)
        ax.view_init(elev=22, azim=45)
        self.profile_fig.tight_layout(pad=0.6)
        self.profile_canvas.draw()

    # ── Initial dock sizing ───────────────────────────────────────────────────

    def _set_dock_sizes(self) -> None:
        self.resizeDocks(
            [self._dock_params],  [300], Qt.Orientation.Horizontal)
        self._dock_params.setMinimumWidth(250)
        self._dock_params.setMaximumWidth(460)

        self.resizeDocks(
            [self._dock_profile], [420], Qt.Orientation.Horizontal)
        self._dock_profile.setMinimumWidth(300)

    # ── Action handlers (stubs — wire to AppController later) ────────────────

    def _on_run(self) -> None:
        self.set_status("Simulation running…", "#f9e2af")
        self._progress.setFormat("Simulating…")
        self._progress.setValue(30)

    def _on_stop(self) -> None:
        self.set_status("Stopped.", "#f38ba8")
        self._progress.setFormat("Idle")
        self._progress.setValue(0)

    def _on_mc(self) -> None:
        self.set_status("Monte Carlo running…", "#89b4fa")
        self._progress.setFormat("Monte Carlo…")
        self._progress.setValue(10)

    def _on_phase1(self) -> None:
        self.set_status("Phase 1 optimisation running…", "#fab387")
        self._progress.setFormat("Phase 1 Opt…")
        self._progress.setValue(50)

    def _on_center_map(self) -> None:
        self.set_status("Map centred on predicted landing point.")

    def _on_get_location(self) -> None:
        self.set_status(
            "Requesting current GPS / network location…", "#f9e2af")

    def _on_load_motor(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Motor File", "",
            "Thrust CSV (*.csv);;All Files (*)")
        if path:
            import os
            name = os.path.basename(path)
            self.motor_label.setText(name)
            self.motor_label.setStyleSheet(
                "color: #a6e3a1; font-style: normal;")
            self.set_status(f"Motor loaded: {name}")

    def _on_mode_changed(self, mode: str) -> None:
        visible = mode in ("Precision Landing", "Winged Hover",
                           "Altitude Competition")
        self._rmax_label.setVisible(visible)
        self.rmax_input.setVisible(visible)

    def _on_about(self) -> None:
        QMessageBox.information(
            self, "About Kazamidori",
            "Kazamidori  —  Trajectory & Landing Point Simulator\n\n"
            "Qt6 / PySide6 migration shell  (ui_qt/)\n"
            "Legacy Tkinter UI preserved in  ui/\n\n"
            "Both UIs share the same core/ simulation engine.",
        )

    # ── Public update API (called by future AppController) ───────────────────

    def set_status(self, msg: str,
                   color: Optional[str] = None) -> None:
        """Update the left-side status bar text."""
        self._status_label.setText(msg)
        c = color or "#a6adc8"
        self._status_label.setStyleSheet(
            f"color: {c}; padding-left: 8px;")

    def update_wind_readout(
        self,
        surf_spd: float, surf_dir: float,
        up_spd:   float, up_dir:   float,
        gust:     float = 0.0,
    ) -> None:
        """Refresh the permanent wind readout in the status bar."""
        self._wind_status.setText(
            f"Surface: {surf_spd:.1f} m/s @ {surf_dir:.0f}°"
            f"   (Gust {gust:.1f})"
            f"   |   Upper: {up_spd:.1f} m/s @ {up_dir:.0f}°"
        )

    def set_go_nogo(self, go: bool) -> None:
        """Update the Phase 2 GO / NO-GO indicator label."""
        if go:
            self._go_nogo_label.setText("✔   GO")
            self._go_nogo_label.setStyleSheet(
                "font-size: 18px; font-weight: bold;"
                "color: #a6e3a1; padding: 8px;")
        else:
            self._go_nogo_label.setText("✘   NO-GO")
            self._go_nogo_label.setStyleSheet(
                "font-size: 18px; font-weight: bold;"
                "color: #f38ba8; padding: 8px;")

    def set_progress(self, value: int, label: str = "") -> None:
        """Set the toolbar progress bar value (0–100)."""
        self._progress.setValue(max(0, min(100, value)))
        if label:
            self._progress.setFormat(label)


# ── Standalone preview entry point ────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
