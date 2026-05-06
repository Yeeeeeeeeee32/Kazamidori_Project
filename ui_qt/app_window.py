"""
ui_qt/app_window.py
Main application window — Kazamidori Project.

3-pane docking layout
---------------------
  settings_dock (Left)  |  profile_dock (Centre)  |  map_dock (Right)
  Parameters / GO-NOGO     3-D trajectory + wind       2-D landing map

All heavy computation lives in core/ — this module is view-only.

Standalone preview:
    python -m ui_qt.app_window
"""

from __future__ import annotations

import sys
from typing import Optional

import matplotlib
matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QSize, QObject, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QScrollArea,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox,
    QComboBox, QPushButton, QToolBar, QStatusBar,
    QSizePolicy, QProgressBar, QFrame, QFileDialog,
    QMessageBox, QToolBox, QSplitter,
)
from PySide6.QtGui import QAction


# ── Window-local reactive state ───────────────────────────────────────────────

class AppState(QObject):
    """
    Lightweight reactive state driving AppWindow's plot canvases.

    Every property setter emits ``needs_redraw`` on change so all three
    canvases stay in sync without polling.
    """

    needs_redraw: Signal = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._wind_speed:        float          = 4.0
        self._wind_dir:          float          = 100.0
        self._cep_prob:          int            = 90
        self._sim_mode:          str            = "Point-Return"
        self._simulation_result: Optional[dict] = None
        self._wind_profile:      list           = []
        self._wind_history:      list           = []

    @property
    def wind_speed(self) -> float: return self._wind_speed

    @wind_speed.setter
    def wind_speed(self, v: float) -> None:
        if self._wind_speed != v:
            self._wind_speed = float(v)
            self.needs_redraw.emit()

    @property
    def wind_dir(self) -> float: return self._wind_dir

    @wind_dir.setter
    def wind_dir(self, v: float) -> None:
        if self._wind_dir != v:
            self._wind_dir = float(v)
            self.needs_redraw.emit()

    @property
    def cep_prob(self) -> int: return self._cep_prob

    @cep_prob.setter
    def cep_prob(self, v: int) -> None:
        if self._cep_prob != v:
            self._cep_prob = int(v)
            self.needs_redraw.emit()

    @property
    def sim_mode(self) -> str: return self._sim_mode

    @sim_mode.setter
    def sim_mode(self, v: str) -> None:
        if self._sim_mode != v:
            self._sim_mode = str(v)
            self.needs_redraw.emit()

    @property
    def simulation_result(self) -> Optional[dict]:
        return self._simulation_result

    @simulation_result.setter
    def simulation_result(self, v: Optional[dict]) -> None:
        self._simulation_result = v
        self.needs_redraw.emit()

    @property
    def wind_profile(self) -> list: return self._wind_profile

    @wind_profile.setter
    def wind_profile(self, v: list) -> None:
        self._wind_profile = list(v) if v is not None else []
        self.needs_redraw.emit()

    @property
    def wind_history(self) -> list: return self._wind_history

    @wind_history.setter
    def wind_history(self, v: list) -> None:
        self._wind_history = list(v) if v is not None else []
        self.needs_redraw.emit()


# ── Catppuccin Mocha dark palette ─────────────────────────────────────────────
_QSS = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "SF Pro Text", Arial, sans-serif;
    font-size: 9pt;
}
QDockWidget { color: #cdd6f4; font-weight: bold; }
QDockWidget::title {
    background: #313244; padding: 5px 10px;
    border-bottom: 2px solid #89b4fa; text-align: left;
}
QDockWidget::close-button, QDockWidget::float-button {
    border: none; background: transparent; padding: 2px;
}
QToolBox::tab {
    background: #313244; color: #89b4fa; font-weight: bold;
    font-size: 8pt; padding: 6px 10px;
    border: 1px solid #45475a; border-radius: 4px; margin-bottom: 2px;
}
QToolBox::tab:selected { background: #45475a; color: #cba6f7; border-color: #cba6f7; }
QToolBox::tab:hover    { background: #3d3f5a; border-color: #89b4fa; }
QSplitter::handle       { background: #45475a; }
QSplitter::handle:hover { background: #89b4fa; }
QGroupBox {
    border: 1px solid #45475a; border-radius: 6px; margin-top: 10px;
    padding: 8px 6px 6px 6px; font-weight: bold; font-size: 8pt; color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 4px;
    background-color: #1e1e2e;
}
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background: #313244; border: 1px solid #45475a; border-radius: 4px;
    padding: 3px 6px; color: #cdd6f4; min-width: 80px;
}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
    border-color: #89b4fa;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button,       QSpinBox::down-button {
    background: #45475a; border: none; width: 16px; border-radius: 2px;
}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
QSpinBox::up-button:hover,       QSpinBox::down-button:hover {
    background: #585b70;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #313244; border: 1px solid #45475a;
    selection-background-color: #45475a; color: #cdd6f4; outline: none;
}
QPushButton {
    background: #313244; border: 1px solid #45475a; border-radius: 5px;
    padding: 5px 14px; color: #cdd6f4; font-weight: bold;
}
QPushButton:hover   { background: #45475a; border-color: #89b4fa; }
QPushButton:pressed { background: #89b4fa; color: #1e1e2e; }
QPushButton#btn_run  { background: #a6e3a1; color: #1e1e2e; border-color: #a6e3a1; }
QPushButton#btn_run:hover  { background: #94e2d5; }
QPushButton#btn_mc   { background: #89b4fa; color: #1e1e2e; border-color: #89b4fa; }
QPushButton#btn_mc:hover   { background: #b4befe; }
QPushButton#btn_stop { background: #f38ba8; color: #1e1e2e; border-color: #f38ba8; }
QPushButton#btn_stop:hover { background: #eba0ac; }
QPushButton#btn_phase1_run {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #cba6f7, stop:1 #89b4fa);
    color: #1e1e2e; border: none; border-radius: 6px;
    font-size: 10pt; font-weight: bold; padding: 10px 16px;
}
QPushButton#btn_phase1_run:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #d4b5ff, stop:1 #99c0ff);
}
QPushButton#btn_phase1_run:pressed { background: #89b4fa; color: #181825; }
QToolBar {
    background: #181825; border: none;
    border-bottom: 1px solid #313244; padding: 3px 6px; spacing: 4px;
}
QToolBar QToolButton {
    background: transparent; border: 1px solid transparent;
    border-radius: 4px; padding: 3px 8px; color: #cdd6f4;
}
QToolBar QToolButton:hover   { background: #313244; border-color: #45475a; }
QToolBar QToolButton:pressed { background: #45475a; }
QMenuBar { background: #181825; color: #cdd6f4; border-bottom: 1px solid #313244; }
QMenuBar::item { padding: 5px 12px; background: transparent; }
QMenuBar::item:selected { background: #313244; border-radius: 3px; }
QMenu {
    background: #1e1e2e; border: 1px solid #45475a;
    border-radius: 4px; padding: 4px;
}
QMenu::item { padding: 5px 20px 5px 12px; border-radius: 3px; }
QMenu::item:selected { background: #313244; color: #89b4fa; }
QMenu::separator { height: 1px; background: #45475a; margin: 3px 8px; }
QStatusBar {
    background: #181825; color: #a6adc8;
    border-top: 1px solid #313244; font-size: 8pt;
}
QStatusBar::item { border: none; }
QScrollBar:vertical   { background: #1e1e2e; width: 8px;  margin: 0; }
QScrollBar:horizontal { background: #1e1e2e; height: 8px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #45475a; border-radius: 4px;
    min-height: 24px; min-width: 24px;
}
QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover { background: #585b70; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { height: 0; width: 0; }
QProgressBar {
    background: #313244; border: 1px solid #45475a; border-radius: 4px;
    text-align: center; color: #cdd6f4; font-size: 8pt; max-height: 18px;
}
QProgressBar::chunk { background: #89b4fa; border-radius: 3px; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: #1e1e2e; }
QFormLayout QLabel { color: #a6adc8; }
QMainWindow::separator { width: 2px; height: 2px; background: #313244; }
QMainWindow::separator:hover { background: #45475a; }
"""


# ── Matplotlib canvas wrapper ─────────────────────────────────────────────────

class _MplCanvas(FigureCanvasQTAgg):
    def __init__(self, fig: Figure, parent: Optional[QWidget] = None) -> None:
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()


# ── Axis styling ──────────────────────────────────────────────────────────────

def _style_3d(ax, fig: Optional[Figure] = None) -> None:
    ax.set_facecolor("#313244")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#45475a")
    ax.tick_params(colors="#a6adc8", labelsize=7)
    if fig is not None:
        fig.patch.set_facecolor("#1e1e2e")


def _style_2d(ax, fig: Optional[Figure] = None, bg: str = "#0d0d1a") -> None:
    ax.set_facecolor(bg)
    ax.tick_params(colors="#a6adc8", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#45475a")
    ax.grid(True, color="#1c1c2e", linewidth=0.7, alpha=0.8)
    if fig is not None:
        fig.patch.set_facecolor(bg)


# ── 3-D rendering helpers ─────────────────────────────────────────────────────

def _equalise_3d_axes(ax) -> None:
    limits  = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    max_r   = max((limits[:, 1] - limits[:, 0]).max() / 2.0, 1.0)
    ax.set_xlim3d(centers[0] - max_r, centers[0] + max_r)
    ax.set_ylim3d(centers[1] - max_r, centers[1] + max_r)
    ax.set_zlim3d(max(0.0, centers[2] - max_r), centers[2] + max_r)


def _make_altitude_lc(x, y, z):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import matplotlib.cm as _cm
    pts  = np.column_stack([x, y, z])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    z_mid = (z[:-1] + z[1:]) / 2.0
    norm  = (z_mid - z.min()) / max(z.max() - z.min(), 1e-6)
    return Line3DCollection(segs, colors=_cm.cool(norm), linewidth=2.0, alpha=0.92)


def _draw_ellipse_3d(ax, *, cx, cy, a, b, angle_rad=0.0,
                     color="#cba6f7", lw=1.6, label="") -> None:
    t  = np.linspace(0.0, 2.0 * np.pi, 120)
    xe = a * np.cos(t) * np.cos(angle_rad) - b * np.sin(t) * np.sin(angle_rad)
    ye = a * np.cos(t) * np.sin(angle_rad) + b * np.sin(t) * np.cos(angle_rad)
    ax.plot(cx + xe, cy + ye, np.zeros(120),
            color=color, lw=lw, linestyle="--", alpha=0.90,
            label=label if label else "_nolegend_")


def _draw_ellipse_2d(ax, *, cx, cy, a, b, angle_rad=0.0,
                     color="#cba6f7", lw=1.6, alpha=0.90, label=""):
    t  = np.linspace(0.0, 2.0 * np.pi, 120)
    xe = a * np.cos(t) * np.cos(angle_rad) - b * np.sin(t) * np.sin(angle_rad)
    ye = a * np.cos(t) * np.sin(angle_rad) + b * np.sin(t) * np.cos(angle_rad)
    (line,) = ax.plot(cx + xe, cy + ye,
                      color=color, lw=lw, linestyle="--", alpha=alpha,
                      label=label if label else "_nolegend_")
    return line


# ── Map coordinate proxy ──────────────────────────────────────────────────────

class _MapCoordProxy:
    def __init__(self, launch_label: QLabel, landing_label: QLabel) -> None:
        self._launch_lbl  = launch_label
        self._landing_lbl = landing_label

    def update_launch(self, lat: float, lon: float) -> None:
        self._launch_lbl.setText(f"Launch:  {lat:.6f}°N, {lon:.6f}°E")

    def update_landing(self, lat: float, lon: float) -> None:
        self._landing_lbl.setText(f"Landing:  {lat:.6f}°N, {lon:.6f}°E")

    def clear_landing(self) -> None:
        self._landing_lbl.setText("Landing:  —")


# ── Main window ───────────────────────────────────────────────────────────────

class AppWindow(QMainWindow):
    """
    Top-level PySide6 window — 3-pane docking layout.

    Public widget attributes (consumed by SimController._collect_params)
    -------------------------------------------------------------------
    wind_speed_input, wind_dir_input  : QDoubleSpinBox
    cep_prob_input                    : QSpinBox
    sim_mode_combo                    : QComboBox
    lat_input, lon_input              : QDoubleSpinBox
    elev_input, azim_input            : QDoubleSpinBox
    mc_runs_input                     : QSpinBox
    surf_spd_input, surf_dir_input    : QDoubleSpinBox
    up_spd_input,   up_dir_input      : QDoubleSpinBox
    wind_unc_input, thrust_unc_input  : QDoubleSpinBox
    allow_unc_input                   : QDoubleSpinBox
    landing_prob_combo                : QComboBox
    motor_label                       : QLabel
    mode_combo                        : QComboBox
    rmax_input                        : QDoubleSpinBox
    map_widget                        : _MapCoordProxy

    Window-internal reactive state
    ------------------------------
    state : AppState  — drives profile / map / wind canvases via needs_redraw
    """

    OPERATION_MODES = ("Altitude Competition", "Precision Landing",
                       "Winged Hover", "Free")
    LANDING_PROBS   = (50, 68, 80, 85, 90, 95, 99)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = AppState()

        self.setWindowTitle("Kazamidori Project")
        self.resize(1600, 900)
        self.setMinimumSize(960, 640)
        self.setDockNestingEnabled(True)

        # Ignored-policy placeholder keeps docks filling the full client area.
        _ph = QWidget(self)
        _ph.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setCentralWidget(_ph)

        self._apply_theme()
        self._build_figures()
        self._build_menu_bar()
        self._build_tool_bar()
        self._build_status_bar()
        self._setup_docks()
        self._setup_layout()
        self._bind_state()

    # ── Theme ──────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        self.setStyleSheet(_QSS)

    # ── Figures ────────────────────────────────────────────────────────────────

    def _build_figures(self) -> None:
        self.profile_fig    = Figure(figsize=(5, 5), facecolor="#1e1e2e")
        self.profile_ax     = self.profile_fig.add_subplot(111, projection="3d")
        self.profile_canvas = _MplCanvas(self.profile_fig)

        self.map_fig    = Figure(figsize=(6, 6), facecolor="#0d0d1a")
        self.map_ax     = self.map_fig.add_subplot(111)
        self.map_canvas = _MplCanvas(self.map_fig)

        self.wind_fig     = Figure(figsize=(5, 3), facecolor="#1e1e2e")
        self.wind_ax_prof = self.wind_fig.add_subplot(121)
        self.wind_ax_ts   = self.wind_fig.add_subplot(122)
        self.wind_canvas  = _MplCanvas(self.wind_fig)

        # Overlay artist tracking — populated by update_map_plot() and
        # _render_overlays() so partial redraws can remove exactly these
        # artists without touching the base scatter or trajectory layers.
        self._overlay_artists: list = []

    # ── Menu bar ───────────────────────────────────────────────────────────────

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        fm = mb.addMenu("&File")
        fm.addAction(QAction("Load Motor File…", self, triggered=self._on_load_motor))
        fm.addAction(QAction("Export Results…",  self))
        fm.addSeparator()
        fm.addAction(QAction("Quit", self, triggered=self.close))

        sm = mb.addMenu("&Simulation")
        sm.addAction(QAction("▶  Run Simulation",    self, triggered=self._on_run))
        sm.addAction(QAction("🎲  Monte Carlo",      self, triggered=self._on_mc))
        sm.addAction(QAction("🔍  Phase 1 Optimize", self, triggered=self._on_phase1))
        sm.addAction(QAction("⏹  Stop",              self, triggered=self._on_stop))

        self._view_menu = mb.addMenu("&View")

        hm = mb.addMenu("&Help")
        hm.addAction(QAction("About Kazamidori", self, triggered=self._on_about))

    # ── Toolbar ────────────────────────────────────────────────────────────────

    def _build_tool_bar(self) -> None:
        tb = QToolBar("Main Toolbar", self)
        tb.setObjectName("MainToolBar")
        tb.setMovable(False)
        tb.setFloatable(False)

        def _vline():
            sep = QFrame(tb)
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setFrameShadow(QFrame.Shadow.Sunken)
            sep.setStyleSheet("color: #45475a;")
            tb.addWidget(sep)

        btn_run  = QPushButton("▶  Run",     tb); btn_run.setObjectName("btn_run")
        btn_mc   = QPushButton("🎲  MC",      tb); btn_mc.setObjectName("btn_mc")
        btn_ph1  = QPushButton("🔍  Phase 1", tb)
        btn_stop = QPushButton("⏹  Stop",    tb); btn_stop.setObjectName("btn_stop")

        btn_run.setFixedWidth(90);   btn_run.clicked.connect(self._on_run)
        btn_mc.setFixedWidth(78);    btn_mc.clicked.connect(self._on_mc)
        btn_ph1.setFixedWidth(94);   btn_ph1.clicked.connect(self._on_phase1)
        btn_stop.setFixedWidth(74);  btn_stop.clicked.connect(self._on_stop)

        for w in (btn_run, btn_mc, btn_ph1):
            tb.addWidget(w)
        _vline()
        tb.addWidget(btn_stop)

        spacer = QWidget(tb)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        self._progress = QProgressBar(tb)
        self._progress.setFixedWidth(172)
        self._progress.setValue(0)
        self._progress.setFormat("Idle")
        self._progress.setTextVisible(True)
        tb.addWidget(self._progress)

        self.addToolBar(tb)

    # ── Status bar ─────────────────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        sb = QStatusBar(self)
        self.setStatusBar(sb)

        self._status_label = QLabel("Ready", sb)
        self._status_label.setContentsMargins(8, 0, 8, 0)

        self._wind_status = QLabel(
            "Surface: -- m/s @ --°   |   Upper: -- m/s @ --°", sb)
        self._wind_status.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._wind_status.setContentsMargins(8, 0, 8, 0)
        self._wind_status.setStyleSheet("color: #89b4fa;")

        sb.addWidget(self._status_label, stretch=1)
        sb.addPermanentWidget(self._wind_status)

    # ── Task 1: dock creation ─────────────────────────────────────────────────

    def _setup_docks(self) -> None:
        _features = (
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        # map_dock must be created FIRST so self.map_widget exists before
        # _build_parameters_panel() wires the lat/lon lambda closures.
        self.map_dock = QDockWidget("Map View", self)
        self.map_dock.setObjectName("MapDock")
        self.map_dock.setFeatures(_features)
        self.map_dock.setWidget(self._build_map_dock_widget())

        self.settings_dock = QDockWidget("Settings", self)
        self.settings_dock.setObjectName("SettingsDock")
        self.settings_dock.setFeatures(_features)
        self.settings_dock.setWidget(self._build_parameters_panel())

        self.profile_dock = QDockWidget("Flight Profile", self)
        self.profile_dock.setObjectName("ProfileDock")
        self.profile_dock.setFeatures(_features)
        self.profile_dock.setWidget(self._build_profile_dock_widget())

        for dock in (self.settings_dock, self.profile_dock, self.map_dock):
            self._view_menu.addAction(dock.toggleViewAction())

    # ── Task 2: 3-column layout ───────────────────────────────────────────────

    def _setup_layout(self) -> None:
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,  self.settings_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.profile_dock)
        self.splitDockWidget(
            self.profile_dock, self.map_dock, Qt.Orientation.Horizontal)

    # ── Aspect-ratio enforcement ──────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Defer until after the OS has committed the window geometry.
        QTimer.singleShot(0, self._apply_initial_sizes)

    def _apply_initial_sizes(self) -> None:
        # settings: 300 px  |  profile: 650 px  |  map: 650 px
        self.resizeDocks(
            [self.settings_dock, self.profile_dock, self.map_dock],
            [300, 650, 650],
            Qt.Orientation.Horizontal,
        )

    # ── Profile dock content (3-D trajectory + wind) ──────────────────────────

    def _build_profile_dock_widget(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(2)

        top = QWidget(splitter)
        tl  = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)
        nav3d = NavigationToolbar2QT(self.profile_canvas, top)
        nav3d.setIconSize(QSize(14, 14))
        tl.addWidget(nav3d)
        tl.addWidget(self.profile_canvas)

        bot = QWidget(splitter)
        bl  = QVBoxLayout(bot)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        hdr = QLabel("  Wind Profile  ·  60-s Spaghetti", bot)
        hdr.setStyleSheet("color: #6c7086; font-size: 7pt; padding: 1px 4px;")
        nav_w = NavigationToolbar2QT(self.wind_canvas, bot)
        nav_w.setIconSize(QSize(14, 14))
        bl.addWidget(hdr)
        bl.addWidget(nav_w)
        bl.addWidget(self.wind_canvas)

        splitter.addWidget(top)
        splitter.addWidget(bot)
        splitter.setSizes([600, 300])
        return splitter

    # ── Parameters panel (settings dock) ──────────────────────────────────────

    def _build_parameters_panel(self) -> QWidget:
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 4)
        lay.setSpacing(2)

        tb = QToolBox(container)
        tb.addItem(self._build_settings_page(),     "⚙   Settings")
        tb.addItem(self._build_engine_page(),       "🔧  Engine (Motor)")
        tb.addItem(self._build_airframe_page(),     "🚀  Airframe")
        tb.addItem(self._build_launch_point_page(), "📍  Launch Point")
        tb.addItem(self._build_launch_mode_page(),  "🎯  Launch Mode")
        lay.addWidget(tb, stretch=1)

        self._go_nogo_label = QLabel("●  STANDBY", container)
        self._go_nogo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._go_nogo_label.setStyleSheet(
            "font-size: 12pt; font-weight: bold; color: #6c7086; padding: 6px;")
        lay.addWidget(self._go_nogo_label)

        btn = QPushButton("🚀   RUN PHASE 1 SIMULATION", container)
        btn.setObjectName("btn_phase1_run")
        btn.setMinimumHeight(48)
        btn.clicked.connect(self._on_phase1)
        lay.addWidget(btn)

        return container

    # ── Settings page ──────────────────────────────────────────────────────────

    def _build_settings_page(self) -> QScrollArea:
        w   = QWidget()
        frm = QFormLayout(w)
        frm.setSpacing(6)
        frm.setContentsMargins(8, 8, 8, 8)

        self.wind_speed_input = QDoubleSpinBox(w)
        self.wind_speed_input.setRange(0.0, 50.0); self.wind_speed_input.setDecimals(1)
        self.wind_speed_input.setValue(4.0);        self.wind_speed_input.setSuffix(" m/s")

        self.wind_dir_input = QDoubleSpinBox(w)
        self.wind_dir_input.setRange(0.0, 360.0); self.wind_dir_input.setDecimals(1)
        self.wind_dir_input.setValue(100.0);       self.wind_dir_input.setSuffix("°")
        self.wind_dir_input.setWrapping(True)

        self.surf_spd_input = QDoubleSpinBox(w)
        self.surf_spd_input.setRange(0, 50); self.surf_spd_input.setDecimals(1)
        self.surf_spd_input.setValue(4.0);   self.surf_spd_input.setSuffix(" m/s")

        self.surf_dir_input = QDoubleSpinBox(w)
        self.surf_dir_input.setRange(0, 360); self.surf_dir_input.setDecimals(1)
        self.surf_dir_input.setValue(100.0);  self.surf_dir_input.setSuffix("°")
        self.surf_dir_input.setWrapping(True)

        self.up_spd_input = QDoubleSpinBox(w)
        self.up_spd_input.setRange(0, 100); self.up_spd_input.setDecimals(1)
        self.up_spd_input.setValue(8.0);    self.up_spd_input.setSuffix(" m/s")

        self.up_dir_input = QDoubleSpinBox(w)
        self.up_dir_input.setRange(0, 360); self.up_dir_input.setDecimals(1)
        self.up_dir_input.setValue(90.0);   self.up_dir_input.setSuffix("°")
        self.up_dir_input.setWrapping(True)

        self.cep_prob_input = QSpinBox(w)
        self.cep_prob_input.setRange(50, 99); self.cep_prob_input.setValue(90)
        self.cep_prob_input.setSuffix(" %")

        self.mc_runs_input = QSpinBox(w)
        self.mc_runs_input.setRange(10, 5000); self.mc_runs_input.setValue(200)
        self.mc_runs_input.setSingleStep(50)

        self.landing_prob_combo = QComboBox(w)
        for p in self.LANDING_PROBS:
            self.landing_prob_combo.addItem(f"{p} %", p)
        self.landing_prob_combo.setCurrentIndex(4)

        self.wind_unc_input = QDoubleSpinBox(w)
        self.wind_unc_input.setRange(0, 1); self.wind_unc_input.setDecimals(2)
        self.wind_unc_input.setValue(0.20); self.wind_unc_input.setSingleStep(0.01)
        self.wind_unc_input.setSuffix("  (±ratio)")

        self.thrust_unc_input = QDoubleSpinBox(w)
        self.thrust_unc_input.setRange(0, 1); self.thrust_unc_input.setDecimals(2)
        self.thrust_unc_input.setValue(0.05); self.thrust_unc_input.setSingleStep(0.01)
        self.thrust_unc_input.setSuffix("  (±ratio)")

        self.allow_unc_input = QDoubleSpinBox(w)
        self.allow_unc_input.setRange(0, 9999); self.allow_unc_input.setDecimals(1)
        self.allow_unc_input.setValue(20.0);    self.allow_unc_input.setSuffix(" m")

        frm.addRow("Wind Speed:",        self.wind_speed_input)
        frm.addRow("Wind From:",          self.wind_dir_input)
        frm.addRow("Surf. Speed:",        self.surf_spd_input)
        frm.addRow("Surf. From:",         self.surf_dir_input)
        frm.addRow("Upper Speed:",        self.up_spd_input)
        frm.addRow("Upper From:",         self.up_dir_input)
        frm.addRow(QLabel(""))
        frm.addRow("CEP Prob:",           self.cep_prob_input)
        frm.addRow("MC Runs:",            self.mc_runs_input)
        frm.addRow("Landing Prob:",       self.landing_prob_combo)
        frm.addRow("Wind Uncertainty:",   self.wind_unc_input)
        frm.addRow("Thrust Uncertainty:", self.thrust_unc_input)
        frm.addRow("Allowable Radius:",   self.allow_unc_input)

        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        sa.setWidget(w)
        return sa

    # ── Engine page ────────────────────────────────────────────────────────────

    def _build_engine_page(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        btn_motor = QPushButton("📂  Load Thrust Curve (.csv)", w)
        btn_motor.clicked.connect(self._on_load_motor)

        self.motor_label = QLabel("(none selected)", w)
        self.motor_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.motor_label.setStyleSheet(
            "color: #fab387; font-style: italic; font-size: 8pt; padding: 2px 4px;")
        self.motor_label.setWordWrap(True)

        # ── Read-only motor spec summary ──────────────────────────────────────
        grp     = QGroupBox("Motor Specifications", w)
        grp_lay = QFormLayout(grp)
        grp_lay.setSpacing(6)
        grp_lay.setContentsMargins(10, 10, 10, 8)

        _tag = (
            "QLabel { color: #cdd6f4; background: #181825; font-weight: bold; "
            "font-family: 'Consolas', monospace; padding: 3px 8px; "
            "border-radius: 4px; border: 1px solid #45475a; }"
        )

        self.lbl_avg_thrust    = QLabel("—", grp)
        self.lbl_max_thrust    = QLabel("—", grp)
        self.lbl_burn_time     = QLabel("—", grp)
        self.lbl_total_impulse = QLabel("—", grp)
        for _lbl in (self.lbl_avg_thrust, self.lbl_max_thrust,
                     self.lbl_burn_time, self.lbl_total_impulse):
            _lbl.setStyleSheet(_tag)

        grp_lay.addRow("Avg Thrust:",    self.lbl_avg_thrust)
        grp_lay.addRow("Max Thrust:",    self.lbl_max_thrust)
        grp_lay.addRow("Burn Time:",     self.lbl_burn_time)
        grp_lay.addRow("Total Impulse:", self.lbl_total_impulse)

        lay.addWidget(btn_motor)
        lay.addWidget(self.motor_label)
        lay.addWidget(grp)
        lay.addStretch()
        return w

    # ── Airframe page ──────────────────────────────────────────────────────────
    # Units: CGMS — lengths in cm (from nose tip), mass in g, delay in s.

    def _build_airframe_page(self) -> QScrollArea:
        w   = QWidget()
        frm = QFormLayout(w)
        frm.setSpacing(5)
        frm.setContentsMargins(8, 8, 8, 8)

        def _dsb(lo, hi, val, dec, suffix, parent=w):
            sb = QDoubleSpinBox(parent)
            sb.setRange(lo, hi); sb.setDecimals(dec)
            sb.setValue(val);    sb.setSuffix(suffix)
            return sb

        # ── Mass / balance ────────────────────────────────────────────────────
        self.af_mass_input   = _dsb(1,   50_000, 1000.0, 1, " g")
        self.af_cg_input     = _dsb(0,     500,    50.0, 1, " cm")

        # ── Body geometry ─────────────────────────────────────────────────────
        self.af_len_input    = _dsb(1,     500,   110.0, 1, " cm")
        self.af_radius_input = _dsb(0.5,    30,     3.5, 2, " cm")
        self.af_nose_input   = _dsb(1,     200,    20.0, 1, " cm")

        # ── Fins (measured from nose tip per CGS spec) ────────────────────────
        self.af_finroot_input = _dsb(0.5,  100,  12.0, 1, " cm")
        self.af_fintip_input  = _dsb(0.5,  100,   6.0, 1, " cm")
        self.af_finspan_input = _dsb(0.5,  100,   8.0, 1, " cm")
        self.af_finpos_input  = _dsb(0,    500,  95.0, 1, " cm")

        # ── Motor mount ───────────────────────────────────────────────────────
        self.af_motorpos_input  = _dsb(0,   500,  100.0, 1, " cm")
        self.af_motormass_input = _dsb(0, 5_000,  100.0, 1, " g")
        self.af_backfire_input  = _dsb(0,    10,    0.5, 2, " s")

        frm.addRow("Mass:",               self.af_mass_input)
        frm.addRow("CG (from nose):",     self.af_cg_input)
        frm.addRow(QLabel(""))
        frm.addRow("Airframe Length:",    self.af_len_input)
        frm.addRow("Airframe Radius:",    self.af_radius_input)
        frm.addRow("Nose Length:",        self.af_nose_input)
        frm.addRow(QLabel(""))
        frm.addRow("Fin Root Chord:",     self.af_finroot_input)
        frm.addRow("Fin Tip Chord:",      self.af_fintip_input)
        frm.addRow("Fin Semi-Span:",      self.af_finspan_input)
        frm.addRow("Fin LE Position:",    self.af_finpos_input)
        frm.addRow(QLabel(""))
        frm.addRow("Motor CG Position:",  self.af_motorpos_input)
        frm.addRow("Motor Dry Mass:",     self.af_motormass_input)
        frm.addRow("Backfire Delay:",     self.af_backfire_input)

        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        sa.setWidget(w)
        return sa

    # ── Launch Point page ──────────────────────────────────────────────────────

    def _build_launch_point_page(self) -> QWidget:
        w   = QWidget()
        frm = QFormLayout(w)
        frm.setSpacing(6)
        frm.setContentsMargins(8, 8, 8, 8)

        self.lat_input = QDoubleSpinBox(w)
        self.lat_input.setRange(-90, 90); self.lat_input.setDecimals(6)
        self.lat_input.setValue(35.682800); self.lat_input.setSuffix("°")
        self.lat_input.valueChanged.connect(
            lambda v: self.map_widget.update_launch(v, self.lon_input.value()))

        self.lon_input = QDoubleSpinBox(w)
        self.lon_input.setRange(-180, 180); self.lon_input.setDecimals(6)
        self.lon_input.setValue(139.759000); self.lon_input.setSuffix("°")
        self.lon_input.valueChanged.connect(
            lambda v: self.map_widget.update_launch(self.lat_input.value(), v))

        self.elev_input = QDoubleSpinBox(w)
        self.elev_input.setRange(0, 90); self.elev_input.setDecimals(1)
        self.elev_input.setValue(85.0);   self.elev_input.setSuffix("°")

        self.azim_input = QDoubleSpinBox(w)
        self.azim_input.setRange(0, 360); self.azim_input.setDecimals(1)
        self.azim_input.setValue(0.0);    self.azim_input.setSuffix("°")
        self.azim_input.setWrapping(True)

        btn_gps = QPushButton("📍  Get Current Location", w)
        btn_gps.clicked.connect(self._on_get_location)

        frm.addRow("Latitude:",       self.lat_input)
        frm.addRow("Longitude:",      self.lon_input)
        frm.addRow("",                btn_gps)
        frm.addRow(QLabel(""))
        frm.addRow("Rail Elevation:", self.elev_input)
        frm.addRow("Rail Azimuth:",   self.azim_input)
        return w

    # ── Launch Mode page ───────────────────────────────────────────────────────

    def _build_launch_mode_page(self) -> QWidget:
        w   = QWidget()
        frm = QFormLayout(w)
        frm.setSpacing(6)
        frm.setContentsMargins(8, 8, 8, 8)

        self.sim_mode_combo = QComboBox(w)
        self.sim_mode_combo.addItems(["Point-Return", "Altitude", "Glider"])
        self.sim_mode_combo.setCurrentText("Point-Return")

        self.mode_combo = QComboBox(w)
        self.mode_combo.addItems(self.OPERATION_MODES)
        self.mode_combo.setCurrentText("Free")

        self._rmax_label = QLabel("R_max:", w)
        self.rmax_input  = QDoubleSpinBox(w)
        self.rmax_input.setRange(0, 9999); self.rmax_input.setDecimals(1)
        self.rmax_input.setValue(50.0);    self.rmax_input.setSuffix(" m")

        frm.addRow("Sim Mode:",       self.sim_mode_combo)
        frm.addRow("Operation Mode:", self.mode_combo)
        frm.addRow(self._rmax_label,  self.rmax_input)

        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._on_mode_changed("Free")
        return w

    # ── Map dock content ───────────────────────────────────────────────────────

    def _build_map_dock_widget(self) -> QWidget:
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        info = QFrame(container)
        info.setObjectName("MapInfoBar")
        info.setFixedHeight(32)
        info.setStyleSheet(
            "QFrame#MapInfoBar {"
            "  background: #181825; border-bottom: 1px solid #313244;"
            "}")
        ilay = QHBoxLayout(info)
        ilay.setContentsMargins(14, 0, 14, 0)
        ilay.setSpacing(6)

        self._map_launch_lbl = QLabel("Launch:  35.682800°N, 139.759000°E", info)
        self._map_launch_lbl.setStyleSheet(
            "color: #89b4fa; font-size: 9pt; background: transparent;")

        _sep = QLabel("|", info)
        _sep.setStyleSheet("color: #45475a; background: transparent;")

        self._map_landing_lbl = QLabel("Landing:  —", info)
        self._map_landing_lbl.setStyleSheet(
            "color: #f38ba8; font-size: 9pt; background: transparent;")

        ilay.addWidget(self._map_launch_lbl)
        ilay.addStretch()
        ilay.addWidget(_sep)
        ilay.addStretch()
        ilay.addWidget(self._map_landing_lbl)

        nav = NavigationToolbar2QT(self.map_canvas, container)
        nav.setIconSize(QSize(14, 14))

        lay.addWidget(info)
        lay.addWidget(nav)
        lay.addWidget(self.map_canvas, stretch=1)

        self.map_widget = _MapCoordProxy(self._map_launch_lbl, self._map_landing_lbl)
        return container

    # ── Reactive binding ───────────────────────────────────────────────────────

    def _bind_state(self) -> None:
        s = self.state
        self.wind_speed_input.valueChanged.connect(lambda v: setattr(s, "wind_speed", v))
        self.wind_dir_input.valueChanged.connect(  lambda v: setattr(s, "wind_dir",   v))
        self.cep_prob_input.valueChanged.connect(  lambda v: setattr(s, "cep_prob",   v))
        self.sim_mode_combo.currentTextChanged.connect(
            lambda v: setattr(s, "sim_mode", v))

        s.needs_redraw.connect(self.update_profile_plot)
        s.needs_redraw.connect(self.update_map_plot)
        s.needs_redraw.connect(self.update_wind_plot)

        self.update_profile_plot()
        self.update_map_plot()
        self.update_wind_plot()

    # ══ Plot: 3-D Flight Profile ══════════════════════════════════════════════

    def update_profile_plot(self) -> None:
        ax = self.profile_ax
        ax.cla()
        _style_3d(ax, self.profile_fig)

        s   = self.state
        res = s.simulation_result

        if res is not None:
            self._draw_real_result(ax, res, s)
        else:
            self._draw_empty_profile(ax)

        ax.set_xlabel("East  (m)",  color="#6c7086", fontsize=8, labelpad=4)
        ax.set_ylabel("North  (m)", color="#6c7086", fontsize=8, labelpad=4)
        ax.set_zlabel("Alt  (m)",   color="#6c7086", fontsize=8, labelpad=4)
        ax.view_init(elev=22, azim=45)
        if res is not None:
            _equalise_3d_axes(ax)
        self.profile_fig.tight_layout(pad=0.5)
        self.profile_canvas.draw()

    def _draw_empty_profile(self, ax) -> None:
        ax.set_xlim3d(-80, 80); ax.set_ylim3d(-80, 80); ax.set_zlim3d(0, 200)
        span, alpha = 60, 0.35
        for xs, ys, zs, c, lbl in (
            ([0, span], [0, 0],    [0, 0],    "#f38ba8", "E"),
            ([0, 0],    [0, span], [0, 0],    "#a6e3a1", "N"),
            ([0, 0],    [0, 0],    [0, span], "#89b4fa", "Up"),
        ):
            ax.plot(xs, ys, zs, color=c, lw=1.0, alpha=alpha, linestyle="--")
            ax.text(xs[-1]*1.07, ys[-1]*1.07, zs[-1]*1.07,
                    lbl, color=c, fontsize=7, alpha=alpha)
        ax.scatter([0], [0], [0], c="#a6e3a1", s=100, marker="^", zorder=5,
                   label="Launch (0, 0, 0)")
        ax.text2D(0.5, 0.40, "Run a simulation\nto display the 3D trajectory",
                  transform=ax.transAxes, ha="center", va="center",
                  color="#45475a", fontsize=10, linespacing=1.8)
        ax.legend(loc="upper left", fontsize=7,
                  facecolor="#1e1e2e", edgecolor="#45475a",
                  labelcolor="#cdd6f4", framealpha=0.85)
        ax.set_title("3D Flight Profile", color="#a6adc8", fontsize=9, pad=6)

    def _draw_real_result(self, ax, res: dict, s: AppState) -> None:
        tx = np.asarray(res.get("trajectory_x", [0.0]), dtype=float)
        ty = np.asarray(res.get("trajectory_y", [0.0]), dtype=float)
        tz = np.clip(np.asarray(res.get("trajectory_z", [0.0]), dtype=float), 0.0, None)
        mc_x     = np.asarray(res.get("mc_scatter_x", []), dtype=float)
        mc_y     = np.asarray(res.get("mc_scatter_y", []), dtype=float)
        ellipses = res.get("cep_ellipses", [])
        land_x   = float(res.get("land_x", tx[-1] if len(tx) else 0.0))
        land_y   = float(res.get("land_y", ty[-1] if len(ty) else 0.0))

        ax.plot(tx, ty, np.zeros_like(tz),
                color="#45475a", lw=0.8, linestyle=":", alpha=0.45)
        if len(tx) > 1:
            ax.add_collection3d(_make_altitude_lc(tx, ty, tz))
        ax.plot([], [], [], color="#89b4fa", lw=2.0, label="Trajectory  (cool = alt)")

        profile = s.wind_profile
        if profile and len(tx) > 1:
            n_q   = min(6, len(profile))
            q_alts = np.linspace(float(tz.min()), float(tz.max()) * 0.92, n_q)
            scale  = max(float(tz.max()), 50.0) * 0.10
            for q_alt in q_alts:
                closest = min(profile, key=lambda p: abs(p.get("alt", 0) - q_alt))
                qspd = float(closest.get("speed", 0.0))
                qdir = float(closest.get("dir_deg", 0.0))
                idx  = int(np.argmin(np.abs(tz - q_alt)))
                arrow_len = max(qspd, 0.5) / 10.0 * scale
                w_e = np.sin(np.radians(qdir)) * arrow_len
                w_n = np.cos(np.radians(qdir)) * arrow_len
                ax.quiver(float(tx[idx]), float(ty[idx]), float(tz[idx]),
                          w_e, w_n, 0.0,
                          color="#f9e2af", alpha=0.65,
                          arrow_length_ratio=0.35, linewidth=1.0)

        apex_i = int(np.argmax(tz))
        apex_z = float(tz[apex_i])
        ax.scatter([tx[apex_i]], [ty[apex_i]], [apex_z],
                   c="#f9e2af", s=90, marker="*", zorder=6,
                   label=f"Apogee  {apex_z:.0f} m")
        ax.text(tx[apex_i], ty[apex_i], apex_z * 1.04,
                f"  {apex_z:.0f} m", color="#f9e2af", fontsize=7)

        n_mc = min(len(mc_x), len(mc_y))
        if n_mc > 0:
            ax.scatter(mc_x[:n_mc], mc_y[:n_mc], np.zeros(n_mc),
                       c="#fab387", s=6, alpha=0.35, marker=".",
                       label=f"MC landings  (n = {n_mc})")

        for ell in ellipses:
            if "a" not in ell or "b" not in ell:
                continue
            _draw_ellipse_3d(
                ax,
                cx=float(ell.get("cx", land_x)), cy=float(ell.get("cy", land_y)),
                a=float(ell["a"]), b=float(ell["b"]),
                angle_rad=float(ell.get("angle_rad", 0.0)),
                color=str(ell.get("color", "#cba6f7")),
                lw=float(ell.get("lw", 1.6)),
                label=str(ell.get("label", "")),
            )

        ax.scatter([land_x], [land_y], [0.0], c="#f38ba8", s=130,
                   marker="v", zorder=7, label="Nominal landing")
        ax.scatter([0.0], [0.0], [0.0], c="#a6e3a1", s=130,
                   marker="^", zorder=8, label="Launch  (0, 0, 0)")

        h_dist = float(np.hypot(land_x, land_y))
        ax.text2D(0.98, 0.98,
                  f"Apogee:  {apex_z:.0f} m\nH-dist:  {h_dist:.0f} m\n"
                  f"n MC:    {n_mc if n_mc > 0 else '—'}",
                  transform=ax.transAxes, ha="right", va="top",
                  color="#cdd6f4", fontsize=7.5,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#313244",
                            edgecolor="#45475a", alpha=0.88))
        ax.legend(loc="upper left", fontsize=7,
                  facecolor="#1e1e2e", edgecolor="#45475a",
                  labelcolor="#cdd6f4", framealpha=0.88, borderpad=0.6)

        all_x = np.concatenate([tx, mc_x[:n_mc]]) if n_mc > 0 else tx
        all_y = np.concatenate([ty, mc_y[:n_mc]]) if n_mc > 0 else ty
        pad   = max(abs(all_x).max() * 0.12, abs(all_y).max() * 0.12, 10.0)
        ax.set_xlim3d(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim3d(all_y.min() - pad, all_y.max() + pad)
        ax.set_zlim3d(0.0, max(tz.max() * 1.12, 10.0))
        ax.set_title(
            f"Mode: {s.sim_mode}   ·   "
            f"Wind: {s.wind_speed:.1f} m/s @ {s.wind_dir:.0f}°   ·   "
            f"CEP: {s.cep_prob} %",
            color="#a6adc8", fontsize=9, pad=8,
        )

    # ══ Plot: 2-D Landing Map ═════════════════════════════════════════════════

    def update_map_plot(self) -> None:
        ax  = self.map_ax
        fig = self.map_fig
        ax.cla()
        # cla() removes every artist from the axis; reset overlay tracking so
        # the next partial redraw doesn't try to remove already-gone artists.
        self._overlay_artists.clear()
        _style_2d(ax, fig, bg="#0d0d1a")

        theta = np.linspace(0.0, 2.0 * np.pi, 200)
        ax.plot(50  * np.cos(theta), 50  * np.sin(theta),
                color="#f38ba8", lw=1.2, linestyle="--", alpha=0.60,
                label="Target r = 50 m")
        ax.plot(250 * np.cos(theta), 250 * np.sin(theta),
                color="#45475a", lw=1.0, linestyle="--", alpha=0.45,
                label="Target r = 250 m")
        ax.scatter([0], [0], c="#a6e3a1", s=130, marker="^", zorder=5,
                   label="Launch (0, 0)")

        res  = self.state.simulation_result
        xlim = ylim = 300.0

        if res is not None:
            lx = float(res.get("land_x", 0.0))
            ly = float(res.get("land_y", 0.0))

            mc_x = np.asarray(res.get("mc_scatter_x", []), dtype=float)
            mc_y = np.asarray(res.get("mc_scatter_y", []), dtype=float)
            n = min(len(mc_x), len(mc_y))
            if n > 0:
                ax.scatter(mc_x[:n], mc_y[:n], c="#fab387", s=4,
                           alpha=0.30, marker=".", zorder=3,
                           label=f"MC landings  (n = {n})")

            _kde_pal = ["#89b4fa", "#cba6f7", "#f38ba8", "#fab387", "#f9e2af"]
            for i, contour in enumerate(res.get("kde_contours", [])):
                pts = contour.get("points_m", [])
                if len(pts) < 2:
                    continue
                cx_pts = [float(p[0]) for p in pts]
                cy_pts = [float(p[1]) for p in pts]
                col = _kde_pal[i % len(_kde_pal)]
                lbl = contour.get("label",
                                  f"KDE {int(contour.get('prob_frac', 0)*100)} %")
                (line,) = ax.plot(cx_pts + [cx_pts[0]], cy_pts + [cy_pts[0]],
                                  color=col, lw=1.0, alpha=0.55, zorder=4, label=lbl)
                self._overlay_artists.append(line)

            target_prob = self.state.cep_prob
            for ell in res.get("cep_ellipses", []):
                if "a" not in ell or "b" not in ell:
                    continue
                lbl    = str(ell.get("label", ""))
                is_tgt = str(target_prob) in lbl
                col    = str(ell.get("color", "#cba6f7" if is_tgt else "#585b70"))
                lw_val = float(ell.get("lw", 2.0 if is_tgt else 0.9))
                line = _draw_ellipse_2d(
                    ax,
                    cx=float(ell.get("cx", lx)), cy=float(ell.get("cy", ly)),
                    a=float(ell["a"]), b=float(ell["b"]),
                    angle_rad=float(ell.get("angle_rad", 0.0)),
                    color=col, lw=lw_val,
                    alpha=0.95 if is_tgt else 0.35,
                    label=lbl if lbl else "_nolegend_",
                )
                self._overlay_artists.append(line)

            ax.scatter([lx], [ly], c="#f38ba8", s=130, marker="v",
                       zorder=6, label="Nominal landing")

            all_x = np.concatenate([[0, lx], mc_x[:n] if n > 0 else []])
            all_y = np.concatenate([[0, ly], mc_y[:n] if n > 0 else []])
            pad   = max(abs(all_x).max(), abs(all_y).max()) * 0.25 + 60.0
            xlim  = ylim = pad

        ax.set_xlim(-xlim, xlim); ax.set_ylim(-ylim, ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East  (m)",  color="#6c7086", fontsize=8, labelpad=4)
        ax.set_ylabel("North  (m)", color="#6c7086", fontsize=8, labelpad=4)
        ax.set_title("Landing Zone Map  (ENU frame, launch at origin)",
                     color="#a6adc8", fontsize=9, pad=6)
        ax.legend(loc="upper right", fontsize=7,
                  facecolor="#1e1e2e", edgecolor="#45475a",
                  labelcolor="#cdd6f4", framealpha=0.88)
        fig.tight_layout(pad=0.6)
        self.map_canvas.draw()

    # ══ Plot: Wind Profile + Time-Series ═════════════════════════════════════

    def update_wind_plot(self) -> None:
        fig     = self.wind_fig
        ax_prof = self.wind_ax_prof
        ax_ts   = self.wind_ax_ts
        profile = self.state.wind_profile
        history = self.state.wind_history

        for ax in (ax_prof, ax_ts):
            ax.cla()
            _style_2d(ax, bg="#1e1e2e")
        fig.patch.set_facecolor("#1e1e2e")

        if profile:
            alts  = np.array([p.get("alt",     0.0) for p in profile])
            spds  = np.array([p.get("speed",   0.0) for p in profile])
            dirs  = np.array([p.get("dir_deg", 0.0) for p in profile])
            bar_h = max(float(alts.max()) / max(len(alts), 1) * 0.7, 5.0)
            ax_prof.barh(alts, spds, height=bar_h,
                         color="#89b4fa", alpha=0.70, label="Speed")
            for alt, spd, d in zip(alts, spds, dirs):
                u = np.sin(np.radians(d)); v = np.cos(np.radians(d))
                ax_prof.annotate(
                    "", xy=(spd + u * 0.8, alt + v * bar_h * 0.3),
                    xytext=(spd, alt),
                    arrowprops=dict(arrowstyle="->", color="#f9e2af",
                                   lw=1.0, alpha=0.75))
            ax_prof.set_xlabel("Speed (m/s)", color="#6c7086", fontsize=7)
            ax_prof.set_ylabel("Altitude (m)", color="#6c7086", fontsize=7)
            ax_prof.legend(fontsize=6, facecolor="#1e1e2e",
                           edgecolor="#45475a", labelcolor="#cdd6f4")
        else:
            ax_prof.text(0.5, 0.5, "No wind profile\navailable",
                         transform=ax_prof.transAxes, ha="center", va="center",
                         color="#45475a", fontsize=9, linespacing=1.8)
        ax_prof.set_title("Wind Profile", color="#a6adc8", fontsize=8)

        if history:
            times = np.array([h[0] for h in history], dtype=float)
            spds  = np.array([h[1] for h in history], dtype=float)
            t_max = times[-1]
            mask  = times >= (t_max - 60.0)
            tw    = times[mask] - times[mask][0]
            sw    = spds[mask]
            ax_ts.plot(tw, sw, color="#89b4fa", lw=1.2, alpha=0.80,
                       label="Wind speed")
            n_ma = max(3, len(sw) // 10)
            if len(sw) >= n_ma:
                ma = np.convolve(sw, np.ones(n_ma) / n_ma, mode="valid")
                ax_ts.plot(tw[n_ma - 1:], ma, color="#f9e2af", lw=1.8,
                           label=f"MA ({n_ma})")
            ax_ts.set_xlabel("Time (s)",    color="#6c7086", fontsize=7)
            ax_ts.set_ylabel("Speed (m/s)", color="#6c7086", fontsize=7)
            ax_ts.legend(fontsize=6, facecolor="#1e1e2e",
                         edgecolor="#45475a", labelcolor="#cdd6f4")
        else:
            ax_ts.text(0.5, 0.5, "No wind history\navailable",
                       transform=ax_ts.transAxes, ha="center", va="center",
                       color="#45475a", fontsize=9, linespacing=1.8)
        ax_ts.set_title("Wind Time-Series  (last 60 s)", color="#a6adc8", fontsize=8)

        fig.tight_layout(pad=0.4)
        self.wind_canvas.draw()

    # ── Smart partial redraw ──────────────────────────────────────────────────

    def update_visual_overlays(self, state) -> None:
        """Recompute and repaint only the error-ellipse and KDE overlays.

        Retrieves cached_mc_scatter from *state*, recomputes error ellipse and
        KDE contours at the current landing_probability, and replaces only
        those artists on the map canvas.  The base scatter, target circles,
        and landing/launch markers are untouched — no ax.cla() is called.

        Exits silently if cached_mc_scatter is None or too small to fit a
        covariance ellipse (fewer than 4 points).
        """
        from core.monte_carlo import compute_error_ellipse, compute_kde_contours

        scatter = state.cached_mc_scatter
        if scatter is None:
            return

        if isinstance(scatter, np.ndarray):
            if scatter.ndim != 2 or scatter.shape[1] < 2:
                return
            pts = [(float(r[0]), float(r[1])) for r in scatter]
        else:
            pts = [(float(p[0]), float(p[1])) for p in scatter]

        if len(pts) < 4:
            return

        prob         = state.landing_probability
        ellipse      = compute_error_ellipse(pts, prob_pct=prob)
        kde_contours = compute_kde_contours(pts, conf_pct=prob)

        self._render_overlays(ellipse, kde_contours, prob)

    def _render_overlays(
        self,
        ellipse,
        kde_contours: list,
        prob: int,
    ) -> None:
        """Remove stale overlay artists and draw fresh ones at *prob* level.

        Called by both update_visual_overlays (partial redraw path) and
        update_map_plot (full redraw path) so self._overlay_artists always
        reflects exactly the currently visible overlay artists and subsequent
        partial redraws can replace them cleanly.
        """
        for artist in self._overlay_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._overlay_artists.clear()

        ax      = self.map_ax
        pal     = ["#89b4fa", "#cba6f7", "#f38ba8", "#fab387", "#f9e2af"]

        for i, contour in enumerate(kde_contours):
            raw_pts = contour.get("points_m", [])
            if len(raw_pts) < 2:
                continue
            cx_pts = [float(p[0]) for p in raw_pts] + [float(raw_pts[0][0])]
            cy_pts = [float(p[1]) for p in raw_pts] + [float(raw_pts[0][1])]
            lbl    = contour.get("label",
                                 f"KDE {int(contour.get('prob_frac', 0) * 100)} %")
            (line,) = ax.plot(cx_pts, cy_pts,
                              color=pal[i % len(pal)], lw=1.0, alpha=0.55,
                              zorder=4, label=lbl if lbl else "_nolegend_")
            self._overlay_artists.append(line)

        if ellipse and "a" in ellipse and "b" in ellipse:
            t   = np.linspace(0.0, 2.0 * np.pi, 120)
            ang = float(ellipse.get("angle_rad", 0.0))
            xe  = (float(ellipse["a"]) * np.cos(t) * np.cos(ang)
                   - float(ellipse["b"]) * np.sin(t) * np.sin(ang))
            ye  = (float(ellipse["a"]) * np.cos(t) * np.sin(ang)
                   + float(ellipse["b"]) * np.sin(t) * np.cos(ang))
            (line,) = ax.plot(
                float(ellipse["cx"]) + xe,
                float(ellipse["cy"]) + ye,
                color="#cba6f7", lw=2.0, linestyle="--", alpha=0.90,
                zorder=5, label=f"R{prob}",
            )
            self._overlay_artists.append(line)

        self.map_canvas.draw_idle()

    # ── Action handlers ────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        self.set_status("Simulation running…", "#f9e2af")
        self._progress.setFormat("Simulating…"); self._progress.setValue(30)

    def _on_stop(self) -> None:
        self.set_status("Stopped.", "#f38ba8")
        self._progress.setFormat("Idle"); self._progress.setValue(0)

    def _on_mc(self) -> None:
        self.set_status("Monte Carlo running…", "#89b4fa")
        self._progress.setFormat("Monte Carlo…"); self._progress.setValue(10)

    def _on_phase1(self) -> None:
        self.set_status("Phase 1 optimisation running…", "#fab387")
        self._progress.setFormat("Phase 1 Opt…"); self._progress.setValue(50)

    def _on_get_location(self) -> None:
        self.set_status("Requesting current GPS / network location…", "#f9e2af")

    def _on_load_motor(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Thrust Curve", "",
            "Thrust CSV (*.csv);;All Files (*)")
        if not path:
            return

        import os, csv as _csv

        name = os.path.basename(path)
        try:
            thrust_data: list[tuple[float, float]] = []
            with open(path, newline='', encoding='utf-8-sig') as f:
                for row in _csv.reader(f):
                    if not row or row[0].strip().startswith(('#', ';', '!')):
                        continue
                    try:
                        thrust_data.append((float(row[0]), float(row[1])))
                    except (ValueError, IndexError):
                        continue

            if len(thrust_data) >= 2:
                burn_time = thrust_data[-1][0] - thrust_data[0][0]
                max_thrust = max(F for _, F in thrust_data)
                # Trapezoidal total impulse
                total_impulse = sum(
                    (thrust_data[i + 1][1] + thrust_data[i][1]) * 0.5
                    * (thrust_data[i + 1][0] - thrust_data[i][0])
                    for i in range(len(thrust_data) - 1)
                )
                avg_thrust = (total_impulse / burn_time) if burn_time > 0 else 0.0

                self.lbl_avg_thrust.setText(f"{avg_thrust:.1f} N")
                self.lbl_max_thrust.setText(f"{max_thrust:.1f} N")
                self.lbl_burn_time.setText(f"{burn_time:.3f} s")
                self.lbl_total_impulse.setText(f"{total_impulse:.1f} Ns")
                self.motor_label.setText(name)
                self.motor_label.setStyleSheet(
                    "color: #a6e3a1; font-style: normal; font-size: 8pt; padding: 2px 4px;")
                self.set_status(
                    f"Motor: {name}  ·  Avg {avg_thrust:.1f} N  ·  "
                    f"Max {max_thrust:.1f} N  ·  Burn {burn_time:.3f} s",
                    "#a6e3a1")
            else:
                self.motor_label.setText(f"{name}  (no data)")
                self.set_status(f"No valid thrust rows found in {name}", "#f38ba8")

        except Exception as exc:
            self.motor_label.setText(f"{name}  (error)")
            self.set_status(f"Motor load error: {exc}", "#f38ba8")

    def _on_mode_changed(self, mode: str) -> None:
        visible = mode in ("Precision Landing", "Winged Hover", "Altitude Competition")
        self._rmax_label.setVisible(visible)
        self.rmax_input.setVisible(visible)

    def _on_about(self) -> None:
        QMessageBox.information(
            self, "About Kazamidori",
            "Kazamidori  —  Trajectory & Landing Point Simulator\n\n"
            "Qt6 / PySide6  (ui_qt/)   |   Tkinter legacy (ui/)\n\n"
            "Both UIs share the same core/ simulation engine.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_status(self, msg: str, color: Optional[str] = None) -> None:
        self._status_label.setText(msg)
        c = color or "#a6adc8"
        self._status_label.setStyleSheet(f"color: {c}; padding-left: 8px;")

    def update_wind_readout(
        self,
        surf_spd: float, surf_dir: float,
        up_spd:   float, up_dir:   float,
        gust:     float = 0.0,
    ) -> None:
        self._wind_status.setText(
            f"Surface: {surf_spd:.1f} m/s @ {surf_dir:.0f}°"
            f"   (Gust {gust:.1f})"
            f"   |   Upper: {up_spd:.1f} m/s @ {up_dir:.0f}°"
        )

    def set_go_nogo(self, go: bool) -> None:
        if go:
            self._go_nogo_label.setText("✔   GO")
            self._go_nogo_label.setStyleSheet(
                "font-size: 12pt; font-weight: bold; color: #a6e3a1; padding: 6px;")
        else:
            self._go_nogo_label.setText("✘   NO-GO")
            self._go_nogo_label.setStyleSheet(
                "font-size: 12pt; font-weight: bold; color: #f38ba8; padding: 6px;")

    def set_progress(self, value: int, label: str = "") -> None:
        self._progress.setValue(max(0, min(100, value)))
        if label:
            self._progress.setFormat(label)


# ── Standalone entry point ────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = AppWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
