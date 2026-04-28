"""
ui_qt/app_window.py
PySide6 / Qt6 main window for the Kazamidori Project.

Migration target from ui/app_window.py (Tkinter).
The existing ui/ directory is intentionally left untouched.

Architecture
------------
MainWindow (QMainWindow)
  ├── MenuBar      — File / Simulation / View / Help
  ├── MainToolBar  — Run · MC · Phase 1 · Stop · Center Map · progress bar
  ├── StatusBar    — left: status text  |  right: live wind readout
  ├── Central      — 3-D Trajectory figure (FigureCanvasQTAgg)
  └── DockWidgets  (all floatable, draggable, re-arrangeable):
      ├── LaunchParamsDock   LEFT   — coords, vehicle, mode, controls
      ├── WindInputDock      LEFT   — wind speeds/dirs, MC uncertainty
      ├── WindAnalysisDock   BOTTOM — time-series · profile · compass
      ├── LandingAnalysisDock RIGHT — 2-D scatter + CEP/ellipse rings
      └── MapDock            RIGHT  — placeholder for Qt map widget

Run standalone for UI preview:
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

from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QScrollArea,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox,
    QComboBox, QPushButton, QToolBar, QStatusBar,
    QSizePolicy, QProgressBar, QFrame, QFileDialog,
    QMessageBox,
)
from PySide6.QtGui import QAction


# ── Catppuccin Mocha dark theme ───────────────────────────────────────────────
_QSS = """
/* ── Global ─────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "SF Pro Text", Arial, sans-serif;
    font-size: 11px;
}

/* ── Dock widgets ────────────────────────────────────────── */
QDockWidget { color: #cdd6f4; font-weight: bold; }
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
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
    border-color: #89b4fa;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {
    background: #45475a;
    border: none;
    width: 16px;
    border-radius: 2px;
}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
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
QPushButton:hover  { background: #45475a; border-color: #89b4fa; }
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
    padding: 3px 6px;
    color: #cdd6f4;
}
QToolBar QToolButton:hover  { background: #313244; border-color: #45475a; }
QToolBar QToolButton:pressed { background: #45475a; }

/* ── Menu bar / menus ────────────────────────────────────── */
QMenuBar { background: #181825; color: #cdd6f4; border-bottom: 1px solid #313244; }
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
QScrollBar:vertical { background: #1e1e2e; width: 8px; margin: 0; }
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #585b70; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #1e1e2e; height: 8px; }
QScrollBar::handle:horizontal { background: #45475a; border-radius: 4px; min-width: 24px; }
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
    """Thin FigureCanvasQTAgg with sensible size policy."""
    def __init__(self, fig: Figure, parent: Optional[QWidget] = None) -> None:
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.updateGeometry()


# ── Figure styling helpers ────────────────────────────────────────────────────

def _style_3d(ax, fig: Optional[Figure] = None) -> None:
    """Apply Catppuccin Mocha dark style to a 3-D Axes."""
    ax.set_facecolor("#313244")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#45475a")
    ax.tick_params(colors="#a6adc8", labelsize=7)
    if fig is not None:
        fig.patch.set_facecolor("#1e1e2e")


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply dark style to a standard 2-D Axes."""
    ax.set_facecolor("#313244")
    ax.tick_params(colors="#a6adc8", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#45475a")
    ax.grid(True, alpha=0.18, color="#45475a")
    if title:
        ax.set_title(title, color="#cdd6f4", fontsize=9, pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, color="#a6adc8", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color="#a6adc8", fontsize=8)
    if ax.figure is not None:
        ax.figure.patch.set_facecolor("#1e1e2e")


def _style_polar(ax, title: str = "") -> None:
    """Apply dark style to a polar Axes."""
    ax.set_facecolor("#313244")
    ax.tick_params(colors="#a6adc8", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#45475a")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.18, color="#45475a")
    if title:
        ax.set_title(title, color="#cdd6f4", fontsize=9, pad=10)
    if ax.figure is not None:
        ax.figure.patch.set_facecolor("#1e1e2e")


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """
    Top-level PySide6 window for the Kazamidori Project.

    All parameter input widgets are exposed as public instance attributes so
    future application logic can read and write them without reaching into the
    widget hierarchy.

    Public widget attributes
    ------------------------
    lat_input, lon_input            : QDoubleSpinBox
    elev_input, azim_input          : QDoubleSpinBox
    motor_label                     : QLabel  (shows loaded filename)
    mode_combo                      : QComboBox
    rmax_input                      : QDoubleSpinBox
    surf_spd_input, surf_dir_input  : QDoubleSpinBox
    up_spd_input, up_dir_input      : QDoubleSpinBox
    mc_runs_input                   : QSpinBox
    landing_prob_combo              : QComboBox
    wind_unc_input, thrust_unc_input, allow_unc_input : QDoubleSpinBox

    Figure / canvas attributes
    --------------------------
    traj_fig, traj_ax, traj_canvas
    wind_fig, wind_ax_ts, wind_ax_profile, wind_ax_compass, wind_canvas
    landing_fig, landing_ax, landing_canvas
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
        self.resize(1420, 920)
        self.setMinimumSize(900, 600)

        self._apply_theme()
        self._build_figures()
        self._build_menu_bar()
        self._build_tool_bar()
        self._build_status_bar()
        self._build_central_widget()
        self._build_docks()
        self._populate_figure_placeholders()
        self._set_initial_dock_sizes()

        # Raise the first tab in each tabbed group
        self._dock_params.raise_()
        self._dock_landing.raise_()

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        self.setStyleSheet(_QSS)

    # ── Figure construction ───────────────────────────────────────────────────

    def _build_figures(self) -> None:
        # 3-D Trajectory (central widget)
        self.traj_fig = Figure(figsize=(8, 6), facecolor="#1e1e2e")
        self.traj_ax  = self.traj_fig.add_subplot(111, projection="3d")
        self.traj_canvas = _MplCanvas(self.traj_fig)

        # Wind Analysis (bottom dock): time-series | wind-profile | compass
        self.wind_fig = Figure(figsize=(9, 2.4), facecolor="#1e1e2e")
        _gs = self.wind_fig.add_gridspec(
            1, 3, width_ratios=[2.4, 1.8, 1.0], wspace=0.48
        )
        self.wind_ax_ts      = self.wind_fig.add_subplot(_gs[0, 0])
        self.wind_ax_profile = self.wind_fig.add_subplot(_gs[0, 1])
        self.wind_ax_compass = self.wind_fig.add_subplot(_gs[0, 2], projection="polar")
        self.wind_fig.subplots_adjust(left=0.08, right=0.96, top=0.85, bottom=0.22)
        self.wind_canvas = _MplCanvas(self.wind_fig)

        # Landing Analysis (right dock): 2-D top-view scatter
        self.landing_fig = Figure(figsize=(5, 5), facecolor="#1e1e2e")
        self.landing_ax  = self.landing_fig.add_subplot(111, aspect="equal")
        self.landing_canvas = _MplCanvas(self.landing_fig)

    # ── Menu bar ──────────────────────────────────────────────────────────────

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("&File")
        file_menu.addAction(QAction(
            "Load Motor File…", self, triggered=self._on_load_motor))
        file_menu.addAction(QAction(
            "Export Results…", self))
        file_menu.addSeparator()
        file_menu.addAction(QAction(
            "Quit", self, triggered=self.close))

        # Simulation
        sim_menu = mb.addMenu("&Simulation")
        sim_menu.addAction(QAction(
            "▶  Run Simulation", self, triggered=self._on_run))
        sim_menu.addAction(QAction(
            "🎲  Monte Carlo", self, triggered=self._on_mc))
        sim_menu.addAction(QAction(
            "🔍  Phase 1 Optimize", self, triggered=self._on_phase1))
        sim_menu.addAction(QAction(
            "⏹  Stop", self, triggered=self._on_stop))
        sim_menu.addSeparator()
        sim_menu.addAction(QAction(
            "🗺  Center Map", self, triggered=self._on_center_map))

        # View (dock toggle actions added after docks are created)
        self._view_menu = mb.addMenu("&View")

        # Help
        help_menu = mb.addMenu("&Help")
        help_menu.addAction(QAction(
            "About Kazamidori", self, triggered=self._on_about))

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

        # Primary action buttons
        btn_run = QPushButton("▶  Run")
        btn_run.setObjectName("btn_run")
        btn_run.setFixedWidth(96)
        btn_run.clicked.connect(self._on_run)

        btn_mc = QPushButton("🎲  MC")
        btn_mc.setObjectName("btn_mc")
        btn_mc.setFixedWidth(80)
        btn_mc.clicked.connect(self._on_mc)

        btn_ph1 = QPushButton("🔍  Phase 1")
        btn_ph1.setFixedWidth(96)
        btn_ph1.clicked.connect(self._on_phase1)

        _vline()

        btn_stop = QPushButton("⏹  Stop")
        btn_stop.setObjectName("btn_stop")
        btn_stop.setFixedWidth(76)
        btn_stop.clicked.connect(self._on_stop)

        _vline()

        btn_map = QPushButton("🗺  Center Map")
        btn_map.setFixedWidth(110)
        btn_map.clicked.connect(self._on_center_map)

        for w in (btn_run, btn_mc, btn_ph1):
            tb.addWidget(w)
        _vline()
        tb.addWidget(btn_stop)
        _vline()
        tb.addWidget(btn_map)

        # Flexible spacer
        spacer = QWidget()
        spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        # Progress bar (right side of toolbar)
        self._progress = QProgressBar()
        self._progress.setFixedWidth(170)
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
            "Surface: -- m/s @ --°   |   Upper: -- m/s @ --°"
        )
        self._wind_status.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._wind_status.setContentsMargins(8, 0, 8, 0)
        self._wind_status.setStyleSheet("color: #89b4fa;")

        sb.addWidget(self._status_label, stretch=1)
        sb.addPermanentWidget(self._wind_status)

    # ── Central widget (3-D trajectory) ──────────────────────────────────────

    def _build_central_widget(self) -> None:
        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        nav = NavigationToolbar2QT(self.traj_canvas, frame)
        nav.setIconSize(QSize(16, 16))
        layout.addWidget(nav)
        layout.addWidget(self.traj_canvas)

        self.setCentralWidget(frame)

    # ── Dock widgets ──────────────────────────────────────────────────────────

    def _build_docks(self) -> None:
        _ALL = Qt.DockWidgetArea.AllDockWidgetAreas

        # ── LEFT: Launch Parameters ──────────────────────────────────────────
        self._dock_params = QDockWidget("Launch Parameters", self)
        self._dock_params.setObjectName("LaunchParamsDock")
        self._dock_params.setAllowedAreas(_ALL)
        self._dock_params.setWidget(self._build_params_panel())
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dock_params)

        # ── LEFT: Wind & Uncertainty (tabbed with params) ────────────────────
        self._dock_wind_in = QDockWidget("Wind & Uncertainty", self)
        self._dock_wind_in.setObjectName("WindInputDock")
        self._dock_wind_in.setAllowedAreas(_ALL)
        self._dock_wind_in.setWidget(self._build_wind_panel())
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dock_wind_in)
        self.tabifyDockWidget(self._dock_params, self._dock_wind_in)

        # ── BOTTOM: Wind Analysis ────────────────────────────────────────────
        self._dock_wind_plot = QDockWidget("Wind Analysis", self)
        self._dock_wind_plot.setObjectName("WindAnalysisDock")
        self._dock_wind_plot.setAllowedAreas(_ALL)
        wind_frame = QWidget()
        wl = QVBoxLayout(wind_frame)
        wl.setContentsMargins(2, 2, 2, 2)
        wl.setSpacing(0)
        wl.addWidget(NavigationToolbar2QT(self.wind_canvas, wind_frame))
        wl.addWidget(self.wind_canvas)
        self._dock_wind_plot.setWidget(wind_frame)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._dock_wind_plot)

        # ── RIGHT: Landing Analysis ──────────────────────────────────────────
        self._dock_landing = QDockWidget("Landing Analysis", self)
        self._dock_landing.setObjectName("LandingDock")
        self._dock_landing.setAllowedAreas(_ALL)
        land_frame = QWidget()
        ll = QVBoxLayout(land_frame)
        ll.setContentsMargins(2, 2, 2, 2)
        ll.setSpacing(0)
        ll.addWidget(NavigationToolbar2QT(self.landing_canvas, land_frame))
        ll.addWidget(self.landing_canvas)
        self._dock_landing.setWidget(land_frame)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_landing)

        # ── RIGHT: Map (tabbed with Landing Analysis) ────────────────────────
        self._dock_map = QDockWidget("Map View", self)
        self._dock_map.setObjectName("MapDock")
        self._dock_map.setAllowedAreas(_ALL)
        self._dock_map.setWidget(self._build_map_placeholder())
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_map)
        self.tabifyDockWidget(self._dock_landing, self._dock_map)

        # Populate View menu with dock toggle actions
        for dock in (self._dock_params, self._dock_wind_in,
                     self._dock_wind_plot, self._dock_landing, self._dock_map):
            self._view_menu.addAction(dock.toggleViewAction())

    # ── Parameter panel (left dock) ───────────────────────────────────────────

    def _build_params_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Launch Site ──────────────────────────────────────────────────────
        grp_site = QGroupBox("Launch Site")
        frm = QFormLayout(grp_site)
        frm.setSpacing(5)

        self.lat_input = QDoubleSpinBox()
        self.lat_input.setRange(-90, 90)
        self.lat_input.setDecimals(6)
        self.lat_input.setValue(35.682800)
        self.lat_input.setSuffix("°")
        self.lat_input.setToolTip("Launch site latitude")

        self.lon_input = QDoubleSpinBox()
        self.lon_input.setRange(-180, 180)
        self.lon_input.setDecimals(6)
        self.lon_input.setValue(139.759000)
        self.lon_input.setSuffix("°")
        self.lon_input.setToolTip("Launch site longitude")

        btn_loc = QPushButton("📍  Get Current Location")
        btn_loc.clicked.connect(self._on_get_location)

        frm.addRow("Latitude:", self.lat_input)
        frm.addRow("Longitude:", self.lon_input)
        frm.addRow("", btn_loc)
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

        self.azim_input = QDoubleSpinBox()
        self.azim_input.setRange(0, 360)
        self.azim_input.setDecimals(1)
        self.azim_input.setValue(0.0)
        self.azim_input.setSuffix("°")
        self.azim_input.setWrapping(True)

        self.motor_label = QLabel("(none selected)")
        self.motor_label.setStyleSheet("color: #fab387; font-style: italic;")

        btn_motor = QPushButton("📂  Load Motor File")
        btn_motor.clicked.connect(self._on_load_motor)

        frm2.addRow("Elevation:", self.elev_input)
        frm2.addRow("Azimuth:", self.azim_input)
        frm2.addRow("Motor:", self.motor_label)
        frm2.addRow("", btn_motor)
        layout.addWidget(grp_veh)

        # ── Operation Mode ────────────────────────────────────────────────────
        grp_mode = QGroupBox("Operation Mode")
        frm3 = QFormLayout(grp_mode)
        frm3.setSpacing(5)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.OPERATION_MODES)
        self.mode_combo.setCurrentText("Free")

        self._rmax_label = QLabel("R_max:")
        self.rmax_input  = QDoubleSpinBox()
        self.rmax_input.setRange(0, 9999)
        self.rmax_input.setDecimals(1)
        self.rmax_input.setValue(50.0)
        self.rmax_input.setSuffix(" m")

        frm3.addRow("Mode:", self.mode_combo)
        frm3.addRow(self._rmax_label, self.rmax_input)

        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._on_mode_changed("Free")
        layout.addWidget(grp_mode)

        # ── Simulation Controls ───────────────────────────────────────────────
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

    # ── Wind & Uncertainty panel (left dock, tabbed) ──────────────────────────

    def _build_wind_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        def _wind_group(title, spd_attr, dir_attr,
                        spd_def, dir_def, spd_max=50):
            grp = QGroupBox(title)
            frm = QFormLayout(grp)
            frm.setSpacing(5)

            spd = QDoubleSpinBox()
            spd.setRange(0, spd_max)
            spd.setDecimals(1)
            spd.setValue(spd_def)
            spd.setSuffix(" m/s")
            setattr(self, spd_attr, spd)

            d = QDoubleSpinBox()
            d.setRange(0, 360)
            d.setDecimals(1)
            d.setValue(dir_def)
            d.setSuffix("°")
            d.setWrapping(True)
            setattr(self, dir_attr, d)

            frm.addRow("Speed:", spd)
            frm.addRow("From (°):", d)
            return grp

        layout.addWidget(_wind_group(
            "Surface Wind", "surf_spd_input", "surf_dir_input", 4.0, 100.0))
        layout.addWidget(_wind_group(
            "Upper Wind", "up_spd_input", "up_dir_input", 8.0, 90.0, spd_max=100))

        # ── Monte Carlo & Uncertainty ─────────────────────────────────────────
        grp_mc = QGroupBox("Monte Carlo & Uncertainty")
        frm_mc = QFormLayout(grp_mc)
        frm_mc.setSpacing(5)

        self.mc_runs_input = QSpinBox()
        self.mc_runs_input.setRange(10, 5000)
        self.mc_runs_input.setValue(200)
        self.mc_runs_input.setSingleStep(50)

        self.landing_prob_combo = QComboBox()
        for p in self.LANDING_PROBS:
            self.landing_prob_combo.addItem(f"{p} %", p)
        self.landing_prob_combo.setCurrentIndex(4)  # default 90 %

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

        frm_mc.addRow("MC Runs:",           self.mc_runs_input)
        frm_mc.addRow("Landing Prob:",       self.landing_prob_combo)
        frm_mc.addRow("Wind Uncertainty:",   self.wind_unc_input)
        frm_mc.addRow("Thrust Uncertainty:", self.thrust_unc_input)
        frm_mc.addRow("Allowable Radius:",   self.allow_unc_input)
        layout.addWidget(grp_mc)

        # ── Phase 2 GO / NO-GO status ─────────────────────────────────────────
        grp_p2 = QGroupBox("Phase 2  GO / NO-GO")
        p2_vbox = QVBoxLayout(grp_p2)

        self._go_nogo_label = QLabel("●  STANDBY")
        self._go_nogo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._go_nogo_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #6c7086; padding: 8px;")
        p2_vbox.addWidget(self._go_nogo_label)
        layout.addWidget(grp_p2)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    # ── Map placeholder (right dock, tabbed) ──────────────────────────────────

    def _build_map_placeholder(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background-color: #313244;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)

        lbl = QLabel(
            "🗺   Map View\n\n"
            "Migration placeholder.\n\n"
            "Recommended implementation:\n"
            "  QWebEngineView  +  Leaflet.js  +  OpenStreetMap\n\n"
            "Will render:\n"
            "  ▪ Launch site (blue dot)\n"
            "  ▪ Target radius ring (blue)\n"
            "  ▪ Predicted landing (red dot)\n"
            "  ▪ Landing radius ring (red)\n"
            "  ▪ CEP circle (purple)\n"
            "  ▪ KDE probability contours\n"
            "  ▪ Error ellipse (green / red)"
        )
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            "color: #6c7086;"
            "font-size: 12px;"
            "line-height: 1.9;"
            "background: transparent;"
        )
        layout.addWidget(lbl)
        return container

    # ── Placeholder figure content ────────────────────────────────────────────

    def _populate_figure_placeholders(self) -> None:
        # ── 3-D Trajectory ───────────────────────────────────────────────────
        ax = self.traj_ax
        _style_3d(ax, self.traj_fig)
        ax.scatter([0], [0], [0], c="#89b4fa", s=100, marker="^", zorder=5)

        # Faint coordinate cross at origin
        span = 50
        for xs, ys, zs, c in (
            ([0, span], [0, 0], [0, 0], "#f38ba8"),
            ([0, 0], [0, span], [0, 0], "#a6e3a1"),
            ([0, 0], [0, 0], [0, span], "#89b4fa"),
        ):
            ax.plot(xs, ys, zs, color=c, lw=1.0, alpha=0.4, linestyle="--")

        ax.text2D(
            0.5, 0.44,
            "Run a simulation to display\nthe 3-D trajectory",
            transform=ax.transAxes, ha="center", va="center",
            color="#6c7086", fontsize=12, linespacing=1.8,
        )
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_zlim(0, 100)
        ax.set_xlabel("East  (m)",  color="#6c7086", fontsize=8, labelpad=4)
        ax.set_ylabel("North  (m)", color="#6c7086", fontsize=8, labelpad=4)
        ax.set_zlabel("Alt  (m)",   color="#6c7086", fontsize=8, labelpad=4)
        ax.view_init(elev=25, azim=45)
        self.traj_canvas.draw()

        # ── Wind Analysis ─────────────────────────────────────────────────────
        for ax_, title in (
            (self.wind_ax_ts,      "Wind Speed  (Time Series)"),
            (self.wind_ax_profile, "Wind Profile"),
        ):
            _style_ax(ax_, title)
            ax_.text(
                0.5, 0.5, "—",
                transform=ax_.transAxes, ha="center", va="center",
                color="#6c7086", fontsize=20,
            )
        _style_polar(self.wind_ax_compass, "Wind From")
        self.wind_canvas.draw()

        # ── Landing Analysis (2-D top view) ───────────────────────────────────
        ax = self.landing_ax
        _style_ax(ax, "2-D Landing Scatter  (Top View)",
                  xlabel="East  (m)", ylabel="North  (m)")

        # Target ring placeholder
        theta = np.linspace(0, 2 * np.pi, 72)
        for r, col, ls, lbl in (
            (50,  "#89b4fa", "--", "Target radius"),
            (20,  "#f38ba8", ":",  "Landing area"),
        ):
            ax.plot(r * np.cos(theta), r * np.sin(theta),
                    color=col, lw=1.4, linestyle=ls, alpha=0.55, label=lbl)

        ax.scatter([0], [0], c="#89b4fa", s=80, marker="^",
                   zorder=5, label="Launch site")
        ax.text(3, 3, "Simulate + MC\nto populate",
                ha="left", va="bottom", color="#6c7086", fontsize=9)
        ax.set_xlim(-120, 120)
        ax.set_ylim(-120, 120)
        ax.legend(fontsize=8, loc="upper right",
                  facecolor="#313244", edgecolor="#45475a",
                  labelcolor="#a6adc8")
        ax.set_aspect("equal")
        self.landing_fig.tight_layout(pad=1.2)
        self.landing_canvas.draw()

    # ── Dock sizing ───────────────────────────────────────────────────────────

    def _set_initial_dock_sizes(self) -> None:
        self.resizeDocks(
            [self._dock_params],     [320], Qt.Orientation.Horizontal)
        self._dock_params.setMinimumWidth(260)
        self._dock_params.setMaximumWidth(480)

        self.resizeDocks(
            [self._dock_landing],    [380], Qt.Orientation.Horizontal)
        self._dock_landing.setMinimumWidth(280)

        self.resizeDocks(
            [self._dock_wind_plot],  [230], Qt.Orientation.Vertical)
        self._dock_wind_plot.setMinimumHeight(160)

    # ── Action handlers (stubs — connect to core logic in AppController) ──────

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
        self.set_status("Requesting current GPS / network location…", "#f9e2af")

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
        show = mode in ("Precision Landing", "Winged Hover",
                        "Altitude Competition")
        self._rmax_label.setVisible(show)
        self.rmax_input.setVisible(show)

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
        up_spd: float,   up_dir: float,
        gust: float = 0.0,
    ) -> None:
        """Refresh the permanent wind readout in the status bar."""
        self._wind_status.setText(
            f"Surface: {surf_spd:.1f} m/s @ {surf_dir:.0f}°"
            f"   (Gust {gust:.1f})"
            f"   |   Upper: {up_spd:.1f} m/s @ {up_dir:.0f}°"
        )

    def set_go_nogo(self, go: bool) -> None:
        """Update the Phase 2 GO / NO-GO indicator."""
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
        """Set the toolbar progress bar (0–100)."""
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
