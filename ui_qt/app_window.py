"""
ui_qt/app_window.py  —  Phase 3.5: UI Restoration & Plotting
PySide6 / Qt6 main window for the Kazamidori Project.

Layout
------
MainWindow (QMainWindow)
  ├── MenuBar       — File / Simulation / View / Help
  ├── MainToolBar   — Run · MC · Phase 1 · Stop · progress bar
  ├── StatusBar     — left: status text  |  right: live wind readout
  ├── Central       — Matplotlib 2-D Map View  (ENU top-down)
  └── DockWidgets:
      ├── ParametersDock  LEFT  — QToolBox (5 pages) + GO/NO-GO + RUN button
      └── ProfileDock     RIGHT — QSplitter (vertical):
                                    top:    3-D trajectory + quiver arrows
                                    bottom: wind profile + 60-s time-series

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

from PySide6.QtCore import Qt, QSize, QObject, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QScrollArea,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox,
    QComboBox, QPushButton, QToolBar, QStatusBar,
    QSizePolicy, QProgressBar, QFrame, QFileDialog,
    QMessageBox, QToolBox, QSplitter,
)
from PySide6.QtGui import QAction


# ── Reactive state model ──────────────────────────────────────────────────────

class AppState(QObject):
    """
    Lightweight reactive state driving AppWindow's plot canvases.

    Every property setter emits ``needs_redraw`` on change so all three
    canvases (profile, map, wind) stay in sync without polling.

    Inject an instance:
        state = AppState()
        win   = AppWindow(state=state)
    or let AppWindow create a private default.
    """

    needs_redraw: Signal = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._wind_speed:        float         = 4.0
        self._wind_dir:          float         = 100.0
        self._cep_prob:          int           = 90
        self._sim_mode:          str           = "Point-Return"
        self._simulation_result: Optional[dict] = None
        # wind_profile: list of {"alt": float, "speed": float, "dir_deg": float}
        self._wind_profile:      list          = []
        # wind_history: list of (timestamp_s: float, speed_m_s: float)
        self._wind_history:      list          = []

    # ── wind_speed ─────────────────────────────────────────────────────────────
    @property
    def wind_speed(self) -> float: return self._wind_speed

    @wind_speed.setter
    def wind_speed(self, v: float) -> None:
        if self._wind_speed != v:
            self._wind_speed = float(v)
            self.needs_redraw.emit()

    # ── wind_dir ───────────────────────────────────────────────────────────────
    @property
    def wind_dir(self) -> float: return self._wind_dir

    @wind_dir.setter
    def wind_dir(self, v: float) -> None:
        if self._wind_dir != v:
            self._wind_dir = float(v)
            self.needs_redraw.emit()

    # ── cep_prob ───────────────────────────────────────────────────────────────
    @property
    def cep_prob(self) -> int: return self._cep_prob

    @cep_prob.setter
    def cep_prob(self, v: int) -> None:
        if self._cep_prob != v:
            self._cep_prob = int(v)
            self.needs_redraw.emit()

    # ── sim_mode ───────────────────────────────────────────────────────────────
    @property
    def sim_mode(self) -> str: return self._sim_mode

    @sim_mode.setter
    def sim_mode(self, v: str) -> None:
        if self._sim_mode != v:
            self._sim_mode = str(v)
            self.needs_redraw.emit()

    # ── simulation_result ──────────────────────────────────────────────────────
    @property
    def simulation_result(self) -> Optional[dict]:
        """
        Dict from the worker thread after a simulation run.  Setting to
        ``None`` resets all canvases to their empty-grid state.

        Expected keys
        -------------
        trajectory_x/y/z  : array-like (m, ENU)
        mc_scatter_x/y    : array-like (m, ground plane)
        cep_ellipses      : list[dict] — each must have a, b; optional:
                            angle_rad, cx, cy, label, color, lw
        land_x, land_y    : float  nominal landing offsets (m)
        """
        return self._simulation_result

    @simulation_result.setter
    def simulation_result(self, v: Optional[dict]) -> None:
        self._simulation_result = v
        self.needs_redraw.emit()

    # ── wind_profile ───────────────────────────────────────────────────────────
    @property
    def wind_profile(self) -> list: return self._wind_profile

    @wind_profile.setter
    def wind_profile(self, v: list) -> None:
        self._wind_profile = list(v) if v is not None else []
        self.needs_redraw.emit()

    # ── wind_history ───────────────────────────────────────────────────────────
    @property
    def wind_history(self) -> list: return self._wind_history

    @wind_history.setter
    def wind_history(self, v: list) -> None:
        self._wind_history = list(v) if v is not None else []
        self.needs_redraw.emit()


# ── Catppuccin Mocha dark palette ─────────────────────────────────────────────
_QSS = """
/* ── Global ──────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "SF Pro Text", Arial, sans-serif;
    font-size: 9pt;
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

/* ── QToolBox ────────────────────────────────────────────── */
QToolBox::tab {
    background: #313244;
    color: #89b4fa;
    font-weight: bold;
    font-size: 8pt;
    padding: 6px 10px;
    border: 1px solid #45475a;
    border-radius: 4px;
    margin-bottom: 2px;
}
QToolBox::tab:selected {
    background: #45475a;
    color: #cba6f7;
    border-color: #cba6f7;
}
QToolBox::tab:hover {
    background: #3d3f5a;
    border-color: #89b4fa;
}

/* ── QSplitter ───────────────────────────────────────────── */
QSplitter::handle {
    background: #45475a;
}
QSplitter::handle:hover {
    background: #89b4fa;
}

/* ── Group boxes ─────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
    font-weight: bold;
    font-size: 8pt;
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

QPushButton#btn_phase1_run {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #cba6f7, stop:1 #89b4fa);
    color: #1e1e2e;
    border: none;
    border-radius: 6px;
    font-size: 10pt;
    font-weight: bold;
    letter-spacing: 0.4px;
    padding: 10px 16px;
}
QPushButton#btn_phase1_run:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #d4b5ff, stop:1 #99c0ff);
}
QPushButton#btn_phase1_run:pressed {
    background: #89b4fa;
    color: #181825;
}

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
    font-size: 8pt;
}
QStatusBar::item { border: none; }

/* ── Scroll bars ─────────────────────────────────────────── */
QScrollBar:vertical {
    background: #1e1e2e; width: 8px; margin: 0;
}
QScrollBar::handle:vertical {
    background: #45475a; border-radius: 4px; min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #585b70; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #1e1e2e; height: 8px;
}
QScrollBar::handle:horizontal {
    background: #45475a; border-radius: 4px; min-width: 24px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ── Progress bar ────────────────────────────────────────── */
QProgressBar {
    background: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    color: #cdd6f4;
    font-size: 8pt;
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
    """Thin FigureCanvasQTAgg with an expanding size policy."""

    def __init__(self, fig: Figure, parent: Optional[QWidget] = None) -> None:
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.updateGeometry()


# ── Axes styling helpers ──────────────────────────────────────────────────────

def _style_3d(ax, fig: Optional[Figure] = None) -> None:
    """Catppuccin Mocha dark styling for Axes3D."""
    ax.set_facecolor("#313244")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#45475a")
    ax.tick_params(colors="#a6adc8", labelsize=7)
    if fig is not None:
        fig.patch.set_facecolor("#1e1e2e")


def _style_2d(ax, fig: Optional[Figure] = None, bg: str = "#0d0d1a") -> None:
    """Catppuccin Mocha dark styling for 2-D axes."""
    ax.set_facecolor(bg)
    ax.tick_params(colors="#a6adc8", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#45475a")
    ax.grid(True, color="#1c1c2e", linewidth=0.7, alpha=0.8)
    if fig is not None:
        fig.patch.set_facecolor(bg)


# ── 3-D rendering helpers ─────────────────────────────────────────────────────

def _equalise_3d_axes(ax) -> None:
    """Force equal aspect ratio on Axes3D (matplotlib ignores set_aspect)."""
    limits  = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    max_r   = max((limits[:, 1] - limits[:, 0]).max() / 2.0, 1.0)
    ax.set_xlim3d(centers[0] - max_r, centers[0] + max_r)
    ax.set_ylim3d(centers[1] - max_r, centers[1] + max_r)
    ax.set_zlim3d(max(0.0, centers[2] - max_r), centers[2] + max_r)


def _make_altitude_lc(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
) -> "Line3DCollection":
    """Line3DCollection with per-segment altitude colour (cool colormap)."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import matplotlib.cm as _cm
    pts  = np.column_stack([x, y, z])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    z_mid = (z[:-1] + z[1:]) / 2.0
    norm  = (z_mid - z.min()) / max(z.max() - z.min(), 1e-6)
    return Line3DCollection(segs, colors=_cm.cool(norm),
                            linewidth=2.0, alpha=0.92)


def _draw_ellipse_3d(
    ax, *, cx: float, cy: float, a: float, b: float,
    angle_rad: float = 0.0, color: str = "#cba6f7",
    lw: float = 1.6, label: str = "",
) -> None:
    """Parametric error ellipse at z = 0 inside Axes3D."""
    t  = np.linspace(0.0, 2.0 * np.pi, 120)
    xe = a * np.cos(t) * np.cos(angle_rad) - b * np.sin(t) * np.sin(angle_rad)
    ye = a * np.cos(t) * np.sin(angle_rad) + b * np.sin(t) * np.cos(angle_rad)
    ax.plot(cx + xe, cy + ye, np.zeros(120),
            color=color, lw=lw, linestyle="--", alpha=0.90,
            label=label if label else "_nolegend_")


def _draw_ellipse_2d(
    ax, *, cx: float, cy: float, a: float, b: float,
    angle_rad: float = 0.0, color: str = "#cba6f7",
    lw: float = 1.6, alpha: float = 0.90, label: str = "",
) -> None:
    """Parametric error ellipse on a 2-D axes."""
    t  = np.linspace(0.0, 2.0 * np.pi, 120)
    xe = a * np.cos(t) * np.cos(angle_rad) - b * np.sin(t) * np.sin(angle_rad)
    ye = a * np.cos(t) * np.sin(angle_rad) + b * np.sin(t) * np.cos(angle_rad)
    ax.plot(cx + xe, cy + ye,
            color=color, lw=lw, linestyle="--", alpha=alpha,
            label=label if label else "_nolegend_")


# ── Map coordinate proxy (compatibility shim for main_qt.py) ─────────────────

class _MapCoordProxy:
    """
    Thin proxy keeping the ``map_widget.update_landing()`` API expected by
    ``SimController`` in ``main_qt.py``.  Updates the info-bar labels only;
    the map plot itself is driven by ``state.simulation_result``.
    """

    def __init__(
        self,
        launch_label: QLabel,
        landing_label: QLabel,
    ) -> None:
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
    Top-level PySide6 window for the Kazamidori Project.

    Public widget attributes (used by SimController._collect_params)
    ----------------------------------------------------------------
    wind_speed_input, wind_dir_input  : QDoubleSpinBox
    cep_prob_input                    : QSpinBox
    sim_mode_combo                    : QComboBox
    lat_input, lon_input              : QDoubleSpinBox
    elev_input, azim_input            : QDoubleSpinBox
    mc_runs_input                     : QSpinBox
    surf_spd_input, surf_dir_input    : QDoubleSpinBox
    up_spd_input, up_dir_input        : QDoubleSpinBox
    wind_unc_input, thrust_unc_input  : QDoubleSpinBox
    allow_unc_input                   : QDoubleSpinBox
    landing_prob_combo                : QComboBox
    motor_label                       : QLabel
    mode_combo                        : QComboBox
    rmax_input                        : QDoubleSpinBox
    map_widget                        : _MapCoordProxy

    Figure / canvas attributes
    --------------------------
    profile_fig, profile_ax, profile_canvas  — 3-D flight profile
    map_fig,     map_ax,     map_canvas       — 2-D landing map
    wind_fig,    wind_ax_prof, wind_ax_ts,
    wind_canvas                               — wind profile + time-series
    """

    OPERATION_MODES = (
        "Altitude Competition",
        "Precision Landing",
        "Winged Hover",
        "Free",
    )
    LANDING_PROBS = (50, 68, 80, 85, 90, 95, 99)

    def __init__(self, state: Optional[AppState] = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.state: AppState = state if state is not None else AppState()

        self.setWindowTitle(
            "Kazamidori  —  Trajectory & Landing Simulator  [Qt6 / PySide6]"
        )
        self.resize(1520, 960)
        self.setMinimumSize(960, 640)

        self._apply_theme()
        self._build_figures()
        self._build_menu_bar()
        self._build_tool_bar()
        self._build_status_bar()
        self._build_central_widget()   # 3-D profile + wind splitter
        self._build_docks()            # _build_map_dock_widget sets self.map_widget
        self._set_dock_sizes()
        self._bind_state()
        self._dock_params.raise_()

    # ── Theme ──────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        self.setStyleSheet(_QSS)

    # ── Figures ────────────────────────────────────────────────────────────────

    def _build_figures(self) -> None:
        # 3-D flight profile
        self.profile_fig    = Figure(figsize=(5, 5), facecolor="#1e1e2e")
        self.profile_ax     = self.profile_fig.add_subplot(111, projection="3d")
        self.profile_canvas = _MplCanvas(self.profile_fig)

        # 2-D landing map (central widget)
        self.map_fig    = Figure(figsize=(8, 6), facecolor="#0d0d1a")
        self.map_ax     = self.map_fig.add_subplot(111)
        self.map_canvas = _MplCanvas(self.map_fig)

        # Wind profile + time-series (two side-by-side subplots)
        self.wind_fig     = Figure(figsize=(5, 3), facecolor="#1e1e2e")
        self.wind_ax_prof = self.wind_fig.add_subplot(121)
        self.wind_ax_ts   = self.wind_fig.add_subplot(122)
        self.wind_canvas  = _MplCanvas(self.wind_fig)

    # ── Menu bar ───────────────────────────────────────────────────────────────

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        fm = mb.addMenu("&File")
        fm.addAction(QAction("Load Motor File…", self,
                             triggered=self._on_load_motor))
        fm.addAction(QAction("Export Results…",  self))
        fm.addSeparator()
        fm.addAction(QAction("Quit", self, triggered=self.close))

        sm = mb.addMenu("&Simulation")
        sm.addAction(QAction("▶  Run Simulation",    self, triggered=self._on_run))
        sm.addAction(QAction("🎲  Monte Carlo",      self, triggered=self._on_mc))
        sm.addAction(QAction("🔍  Phase 1 Optimize", self, triggered=self._on_phase1))
        sm.addAction(QAction("⏹  Stop",              self, triggered=self._on_stop))
        sm.addSeparator()
        sm.addAction(QAction("🗺  Center Map",        self,
                             triggered=self._on_center_map))

        self._view_menu = mb.addMenu("&View")

        hm = mb.addMenu("&Help")
        hm.addAction(QAction("About Kazamidori", self, triggered=self._on_about))

    # ── Toolbar ────────────────────────────────────────────────────────────────

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

        btn_run = QPushButton("▶  Run");  btn_run.setObjectName("btn_run")
        btn_run.setFixedWidth(90);        btn_run.clicked.connect(self._on_run)

        btn_mc = QPushButton("🎲  MC");   btn_mc.setObjectName("btn_mc")
        btn_mc.setFixedWidth(78);         btn_mc.clicked.connect(self._on_mc)

        btn_ph1 = QPushButton("🔍  Phase 1")
        btn_ph1.setFixedWidth(94);        btn_ph1.clicked.connect(self._on_phase1)

        btn_stop = QPushButton("⏹  Stop"); btn_stop.setObjectName("btn_stop")
        btn_stop.setFixedWidth(74);        btn_stop.clicked.connect(self._on_stop)

        btn_map = QPushButton("🗺  Center Map")
        btn_map.setFixedWidth(112);        btn_map.clicked.connect(self._on_center_map)

        for w in (btn_run, btn_mc, btn_ph1):
            tb.addWidget(w)
        _vline()
        tb.addWidget(btn_stop)
        _vline()
        tb.addWidget(btn_map)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        self._progress = QProgressBar()
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

        self._status_label = QLabel("Ready")
        self._status_label.setContentsMargins(8, 0, 8, 0)

        self._wind_status = QLabel(
            "Surface: -- m/s @ --°   |   Upper: -- m/s @ --°")
        self._wind_status.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._wind_status.setContentsMargins(8, 0, 8, 0)
        self._wind_status.setStyleSheet("color: #89b4fa;")

        sb.addWidget(self._status_label, stretch=1)
        sb.addPermanentWidget(self._wind_status)

    # ── Central widget — vertical splitter (3-D profile / wind graph) ─────────

    def _build_central_widget(self) -> None:
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(5)

        # Top pane: 3-D flight profile
        top = QWidget()
        tl  = QVBoxLayout(top)
        tl.setContentsMargins(2, 2, 2, 2)
        tl.setSpacing(0)
        nav3d = NavigationToolbar2QT(self.profile_canvas, top)
        nav3d.setIconSize(QSize(14, 14))
        tl.addWidget(nav3d)
        tl.addWidget(self.profile_canvas)

        # Bottom pane: wind profile + 60-s time-series
        bot = QWidget()
        bl  = QVBoxLayout(bot)
        bl.setContentsMargins(2, 2, 2, 2)
        bl.setSpacing(0)
        hdr = QLabel("  Wind Profile  ·  Time-Series (60 s)")
        hdr.setStyleSheet("color: #6c7086; font-size: 7pt; padding: 2px 4px;")
        nav_w = NavigationToolbar2QT(self.wind_canvas, bot)
        nav_w.setIconSize(QSize(14, 14))
        bl.addWidget(hdr)
        bl.addWidget(nav_w)
        bl.addWidget(self.wind_canvas)

        splitter.addWidget(top)
        splitter.addWidget(bot)
        splitter.setSizes([580, 320])
        self.setCentralWidget(splitter)

    # ── Dock widgets ───────────────────────────────────────────────────────────

    def _build_docks(self) -> None:
        _ALL = Qt.DockWidgetArea.AllDockWidgetAreas

        # RIGHT — Map view.  Built FIRST so self.map_widget exists before
        # _build_parameters_panel() wires the lat/lon lambda closures.
        self._dock_map = QDockWidget("Map View", self)
        self._dock_map.setObjectName("MapDock")
        self._dock_map.setAllowedAreas(_ALL)
        self._dock_map.setWidget(self._build_map_dock_widget())
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self._dock_map)

        # LEFT — Parameters (QToolBox + RUN button)
        self._dock_params = QDockWidget("Parameters", self)
        self._dock_params.setObjectName("ParametersDock")
        self._dock_params.setAllowedAreas(_ALL)
        self._dock_params.setWidget(self._build_parameters_panel())
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self._dock_params)

        for dock in (self._dock_params, self._dock_map):
            self._view_menu.addAction(dock.toggleViewAction())

    # ── Parameters panel (left dock) ───────────────────────────────────────────

    def _build_parameters_panel(self) -> QWidget:
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        tb = QToolBox()
        tb.addItem(self._build_settings_page(),     "⚙   Settings")
        tb.addItem(self._build_engine_page(),       "🔧  Engine (Motor)")
        tb.addItem(self._build_airframe_page(),     "🚀  Airframe")
        tb.addItem(self._build_launch_point_page(), "📍  Launch Point")
        tb.addItem(self._build_launch_mode_page(),  "🎯  Launch Mode")
        lay.addWidget(tb, stretch=1)

        # Phase 2 GO / NO-GO indicator
        self._go_nogo_label = QLabel("●  STANDBY")
        self._go_nogo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._go_nogo_label.setStyleSheet(
            "font-size: 12pt; font-weight: bold; color: #6c7086; padding: 6px;")
        lay.addWidget(self._go_nogo_label)

        # Primary action button
        btn = QPushButton("🚀   RUN PHASE 1 SIMULATION")
        btn.setObjectName("btn_phase1_run")
        btn.setMinimumHeight(48)
        btn.setToolTip("Execute Phase 1 trajectory optimisation")
        btn.clicked.connect(self._on_phase1)
        lay.addWidget(btn)

        return container

    # ── Settings page ──────────────────────────────────────────────────────────

    def _build_settings_page(self) -> QScrollArea:
        w = QWidget()
        frm = QFormLayout(w)
        frm.setSpacing(6)
        frm.setContentsMargins(8, 8, 8, 8)

        # ── Reactive wind (bound to AppState) ─────────────────────────────────
        self.wind_speed_input = QDoubleSpinBox()
        self.wind_speed_input.setRange(0.0, 50.0)
        self.wind_speed_input.setDecimals(1)
        self.wind_speed_input.setValue(4.0)
        self.wind_speed_input.setSuffix(" m/s")
        self.wind_speed_input.setToolTip("Surface wind speed (reactive)")

        self.wind_dir_input = QDoubleSpinBox()
        self.wind_dir_input.setRange(0.0, 360.0)
        self.wind_dir_input.setDecimals(1)
        self.wind_dir_input.setValue(100.0)
        self.wind_dir_input.setSuffix("°")
        self.wind_dir_input.setWrapping(True)
        self.wind_dir_input.setToolTip("Wind from direction (0=N, CW)")

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

        # ── CEP & MC ───────────────────────────────────────────────────────────
        self.cep_prob_input = QSpinBox()
        self.cep_prob_input.setRange(50, 99)
        self.cep_prob_input.setValue(90)
        self.cep_prob_input.setSuffix(" %")
        self.cep_prob_input.setToolTip(
            "CEP confidence level — instantly redraws ellipses on both plots")

        self.mc_runs_input = QSpinBox()
        self.mc_runs_input.setRange(10, 5000)
        self.mc_runs_input.setValue(200)
        self.mc_runs_input.setSingleStep(50)

        self.landing_prob_combo = QComboBox()
        for p in self.LANDING_PROBS:
            self.landing_prob_combo.addItem(f"{p} %", p)
        self.landing_prob_combo.setCurrentIndex(4)

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

        frm.addRow("Wind Speed:",         self.wind_speed_input)
        frm.addRow("Wind From:",           self.wind_dir_input)
        frm.addRow("Surf. Speed:",         self.surf_spd_input)
        frm.addRow("Surf. From:",          self.surf_dir_input)
        frm.addRow("Upper Speed:",         self.up_spd_input)
        frm.addRow("Upper From:",          self.up_dir_input)
        frm.addRow(QLabel(""))
        frm.addRow("CEP Prob:",            self.cep_prob_input)
        frm.addRow("MC Runs:",             self.mc_runs_input)
        frm.addRow("Landing Prob:",        self.landing_prob_combo)
        frm.addRow("Wind Uncertainty:",    self.wind_unc_input)
        frm.addRow("Thrust Uncertainty:",  self.thrust_unc_input)
        frm.addRow("Allowable Radius:",    self.allow_unc_input)

        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        sa.setWidget(w)
        return sa

    # ── Engine (Motor) page ────────────────────────────────────────────────────

    def _build_engine_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self.motor_label = QLabel("(none selected)")
        self.motor_label.setStyleSheet(
            "color: #fab387; font-style: italic; padding: 4px;")
        self.motor_label.setWordWrap(True)

        btn_motor = QPushButton("📂  Load Motor File")
        btn_motor.clicked.connect(self._on_load_motor)

        grp = QGroupBox("Thrust Curve")
        grp_lay = QFormLayout(grp)
        grp_lay.setSpacing(5)

        # Placeholder read-only labels (populated after motor load)
        self._motor_isp_lbl   = QLabel("—")
        self._motor_thrust_lbl = QLabel("—")
        self._motor_burn_lbl   = QLabel("—")
        grp_lay.addRow("Motor:",        self.motor_label)
        grp_lay.addRow("Avg Thrust:",   self._motor_thrust_lbl)
        grp_lay.addRow("Burn Time:",    self._motor_burn_lbl)
        grp_lay.addRow("Total Isp:",    self._motor_isp_lbl)

        lay.addWidget(grp)
        lay.addWidget(btn_motor)
        lay.addStretch()
        return w

    # ── Airframe page ──────────────────────────────────────────────────────────

    def _build_airframe_page(self) -> QWidget:
        w = QWidget()
        frm = QFormLayout(w)
        frm.setSpacing(6)
        frm.setContentsMargins(8, 8, 8, 8)

        self._mass_input = QDoubleSpinBox()
        self._mass_input.setRange(0.1, 50.0)
        self._mass_input.setDecimals(3)
        self._mass_input.setValue(0.500)
        self._mass_input.setSuffix(" kg")

        self._cd_input = QDoubleSpinBox()
        self._cd_input.setRange(0.01, 5.0)
        self._cd_input.setDecimals(3)
        self._cd_input.setValue(0.470)

        self._area_input = QDoubleSpinBox()
        self._area_input.setRange(0.0001, 0.5)
        self._area_input.setDecimals(6)
        self._area_input.setValue(0.007854)
        self._area_input.setSuffix(" m²")

        frm.addRow("Dry Mass:",   self._mass_input)
        frm.addRow("Drag Coeff:", self._cd_input)
        frm.addRow("Ref. Area:",  self._area_input)
        frm.addRow(QLabel(""))

        note = QLabel(
            "These values are not yet wired to the\n"
            "simulation engine in this release.")
        note.setStyleSheet("color: #45475a; font-size: 7pt;")
        frm.addRow(note)
        return w

    # ── Launch Point page ──────────────────────────────────────────────────────

    def _build_launch_point_page(self) -> QWidget:
        w = QWidget()
        frm = QFormLayout(w)
        frm.setSpacing(6)
        frm.setContentsMargins(8, 8, 8, 8)

        self.lat_input = QDoubleSpinBox()
        self.lat_input.setRange(-90, 90)
        self.lat_input.setDecimals(6)
        self.lat_input.setValue(35.682800)
        self.lat_input.setSuffix("°")
        self.lat_input.valueChanged.connect(
            lambda v: self.map_widget.update_launch(v, self.lon_input.value()))

        self.lon_input = QDoubleSpinBox()
        self.lon_input.setRange(-180, 180)
        self.lon_input.setDecimals(6)
        self.lon_input.setValue(139.759000)
        self.lon_input.setSuffix("°")
        self.lon_input.valueChanged.connect(
            lambda v: self.map_widget.update_launch(self.lat_input.value(), v))

        self.elev_input = QDoubleSpinBox()
        self.elev_input.setRange(0, 90)
        self.elev_input.setDecimals(1)
        self.elev_input.setValue(85.0)
        self.elev_input.setSuffix("°")
        self.elev_input.setToolTip("Launch rail elevation from horizontal")

        self.azim_input = QDoubleSpinBox()
        self.azim_input.setRange(0, 360)
        self.azim_input.setDecimals(1)
        self.azim_input.setValue(0.0)
        self.azim_input.setSuffix("°")
        self.azim_input.setWrapping(True)
        self.azim_input.setToolTip("Launch rail azimuth (0=N, CW)")

        btn_gps = QPushButton("📍  Get Current Location")
        btn_gps.clicked.connect(self._on_get_location)

        frm.addRow("Latitude:",   self.lat_input)
        frm.addRow("Longitude:",  self.lon_input)
        frm.addRow("",            btn_gps)
        frm.addRow(QLabel(""))
        frm.addRow("Rail Elevation:", self.elev_input)
        frm.addRow("Rail Azimuth:",   self.azim_input)
        return w

    # ── Launch Mode page ───────────────────────────────────────────────────────

    def _build_launch_mode_page(self) -> QWidget:
        w = QWidget()
        frm = QFormLayout(w)
        frm.setSpacing(6)
        frm.setContentsMargins(8, 8, 8, 8)

        self.sim_mode_combo = QComboBox()
        self.sim_mode_combo.addItems(["Point-Return", "Altitude", "Glider"])
        self.sim_mode_combo.setCurrentText("Point-Return")
        self.sim_mode_combo.setToolTip("Trajectory model for Phase 1 optimisation")

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.OPERATION_MODES)
        self.mode_combo.setCurrentText("Free")

        self._rmax_label = QLabel("R_max:")
        self.rmax_input  = QDoubleSpinBox()
        self.rmax_input.setRange(0, 9999)
        self.rmax_input.setDecimals(1)
        self.rmax_input.setValue(50.0)
        self.rmax_input.setSuffix(" m")

        frm.addRow("Sim Mode:",      self.sim_mode_combo)
        frm.addRow("Operation Mode:", self.mode_combo)
        frm.addRow(self._rmax_label,  self.rmax_input)

        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._on_mode_changed("Free")
        return w

    # ── Map dock widget (right dock) ───────────────────────────────────────────

    def _build_map_dock_widget(self) -> QWidget:
        """2-D map canvas with coordinate info bar — right dock content."""
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Info bar: live launch / landing coordinate readout
        info = QFrame()
        info.setObjectName("MapInfoBar")
        info.setFixedHeight(32)
        info.setStyleSheet(
            "QFrame#MapInfoBar {"
            "  background: #181825;"
            "  border-bottom: 1px solid #313244;"
            "}"
        )
        ilay = QHBoxLayout(info)
        ilay.setContentsMargins(14, 0, 14, 0)
        ilay.setSpacing(6)

        self._map_launch_lbl = QLabel("Launch:  35.682800°N, 139.759000°E")
        self._map_launch_lbl.setStyleSheet(
            "color: #89b4fa; font-size: 9pt; background: transparent;")

        _sep = QLabel("|")
        _sep.setStyleSheet("color: #45475a; background: transparent;")

        self._map_landing_lbl = QLabel("Landing:  —")
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

        self.map_widget = _MapCoordProxy(
            self._map_launch_lbl, self._map_landing_lbl)
        return container

    # ── Dock sizing ────────────────────────────────────────────────────────────

    def _set_dock_sizes(self) -> None:
        self.resizeDocks([self._dock_params], [300], Qt.Orientation.Horizontal)
        self._dock_params.setMinimumWidth(240)
        self._dock_params.setMaximumWidth(460)
        self.resizeDocks([self._dock_map],   [480], Qt.Orientation.Horizontal)
        self._dock_map.setMinimumWidth(320)

    # ── Reactive binding ───────────────────────────────────────────────────────

    def _bind_state(self) -> None:
        """
        Wire every Simulation Setup widget → self.state property.
        All three canvases are connected to needs_redraw.
        """
        s = self.state

        self.wind_speed_input.valueChanged.connect(
            lambda v: setattr(s, "wind_speed", v))
        self.wind_dir_input.valueChanged.connect(
            lambda v: setattr(s, "wind_dir", v))
        self.cep_prob_input.valueChanged.connect(
            lambda v: setattr(s, "cep_prob", v))
        self.sim_mode_combo.currentTextChanged.connect(
            lambda v: setattr(s, "sim_mode", v))

        # All three plots redraw on any state change
        s.needs_redraw.connect(self.update_profile_plot)
        s.needs_redraw.connect(self.update_map_plot)
        s.needs_redraw.connect(self.update_wind_plot)

        # Initial render (clean empty grids)
        self.update_profile_plot()
        self.update_map_plot()
        self.update_wind_plot()

    # ══ Plot: 3-D Flight Profile ══════════════════════════════════════════════

    def update_profile_plot(self) -> None:
        """
        Dual-path renderer for the 3-D profile canvas.

        simulation_result is not None  →  _draw_real_result  (full data)
        simulation_result is None      →  _draw_empty_profile (clean grid)
        """
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
        """Clean 3-D grid — shown on startup and after result is cleared."""
        ax.set_xlim3d(-80, 80)
        ax.set_ylim3d(-80, 80)
        ax.set_zlim3d(0, 200)
        # Faint axis arrows
        span, alpha = 60, 0.35
        for xs, ys, zs, c, lbl in (
            ([0, span], [0, 0],    [0, 0],    "#f38ba8", "E"),
            ([0, 0],    [0, span], [0, 0],    "#a6e3a1", "N"),
            ([0, 0],    [0, 0],    [0, span], "#89b4fa", "Up"),
        ):
            ax.plot(xs, ys, zs, color=c, lw=1.0, alpha=alpha, linestyle="--")
            ax.text(xs[-1]*1.07, ys[-1]*1.07, zs[-1]*1.07,
                    lbl, color=c, fontsize=7, alpha=alpha)
        # Launch marker
        ax.scatter([0], [0], [0], c="#a6e3a1", s=100, marker="^", zorder=5,
                   label="Launch (0, 0, 0)")
        ax.text2D(0.5, 0.40,
                  "Run a simulation\nto display the 3D trajectory",
                  transform=ax.transAxes, ha="center", va="center",
                  color="#45475a", fontsize=10, linespacing=1.8)
        ax.legend(loc="upper left", fontsize=7,
                  facecolor="#1e1e2e", edgecolor="#45475a",
                  labelcolor="#cdd6f4", framealpha=0.85)
        ax.set_title("3D Flight Profile", color="#a6adc8", fontsize=9, pad=6)

    def _draw_real_result(self, ax, res: dict, s: "AppState") -> None:
        """Full post-simulation 3-D renderer."""
        tx = np.asarray(res.get("trajectory_x", [0.0]), dtype=float)
        ty = np.asarray(res.get("trajectory_y", [0.0]), dtype=float)
        tz = np.clip(np.asarray(res.get("trajectory_z", [0.0]), dtype=float),
                     0.0, None)
        mc_x     = np.asarray(res.get("mc_scatter_x", []), dtype=float)
        mc_y     = np.asarray(res.get("mc_scatter_y", []), dtype=float)
        ellipses = res.get("cep_ellipses", [])
        land_x   = float(res.get("land_x", tx[-1] if len(tx) else 0.0))
        land_y   = float(res.get("land_y", ty[-1] if len(ty) else 0.0))

        # Ground shadow
        ax.plot(tx, ty, np.zeros_like(tz),
                color="#45475a", lw=0.8, linestyle=":", alpha=0.45,
                label="_nolegend_")

        # Altitude-coloured trajectory
        if len(tx) > 1:
            ax.add_collection3d(_make_altitude_lc(tx, ty, tz))
        ax.plot([], [], [], color="#89b4fa", lw=2.0,
                label="Trajectory  (cool = alt)")

        # Wind quiver arrows at sampled altitudes
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
                qx, qy, qz = float(tx[idx]), float(ty[idx]), float(tz[idx])
                # Arrow length proportional to speed (10 m/s ≈ full scale)
                arrow_len = max(qspd, 0.5) / 10.0 * scale
                w_e = np.sin(np.radians(qdir)) * arrow_len
                w_n = np.cos(np.radians(qdir)) * arrow_len
                ax.quiver(qx, qy, qz, w_e, w_n, 0.0,
                          color="#f9e2af", alpha=0.65,
                          arrow_length_ratio=0.35, linewidth=1.0)

        # Apogee marker
        apex_i = int(np.argmax(tz))
        apex_z = float(tz[apex_i])
        ax.scatter([tx[apex_i]], [ty[apex_i]], [apex_z],
                   c="#f9e2af", s=90, marker="*", zorder=6,
                   label=f"Apogee  {apex_z:.0f} m")
        ax.text(tx[apex_i], ty[apex_i], apex_z * 1.04,
                f"  {apex_z:.0f} m", color="#f9e2af", fontsize=7)

        # MC scatter cloud
        n_mc = min(len(mc_x), len(mc_y))
        if n_mc > 0:
            ax.scatter(mc_x[:n_mc], mc_y[:n_mc], np.zeros(n_mc),
                       c="#fab387", s=6, alpha=0.35, marker=".",
                       label=f"MC landings  (n = {n_mc})")

        # CEP ellipses at z = 0
        for ell in ellipses:
            if "a" not in ell or "b" not in ell:
                continue
            _draw_ellipse_3d(
                ax,
                cx=float(ell.get("cx", land_x)),
                cy=float(ell.get("cy", land_y)),
                a=float(ell["a"]), b=float(ell["b"]),
                angle_rad=float(ell.get("angle_rad", 0.0)),
                color=str(ell.get("color", "#cba6f7")),
                lw=float(ell.get("lw", 1.6)),
                label=str(ell.get("label", "")),
            )

        # Nominal landing ▼ and Launch ▲
        ax.scatter([land_x], [land_y], [0.0],
                   c="#f38ba8", s=130, marker="v", zorder=7,
                   label="Nominal landing")
        ax.scatter([0.0], [0.0], [0.0],
                   c="#a6e3a1", s=130, marker="^", zorder=8,
                   label="Launch  (0, 0, 0)")

        # Stats text box
        h_dist = float(np.hypot(land_x, land_y))
        stats  = (
            f"Apogee:  {apex_z:.0f} m\n"
            f"H-dist:  {h_dist:.0f} m\n"
            f"n MC:    {n_mc if n_mc > 0 else '—'}"
        )
        ax.text2D(0.98, 0.98, stats,
                  transform=ax.transAxes, ha="right", va="top",
                  color="#cdd6f4", fontsize=7.5,
                  bbox=dict(boxstyle="round,pad=0.4",
                            facecolor="#313244", edgecolor="#45475a",
                            alpha=0.88))

        ax.legend(loc="upper left", fontsize=7,
                  facecolor="#1e1e2e", edgecolor="#45475a",
                  labelcolor="#cdd6f4", framealpha=0.88, borderpad=0.6)

        # Explicit limits for _equalise_3d_axes
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
        """
        2-D ENU top-down map.

        Always shown
        ------------
        • Launch point ▲ at (0, 0)
        • Target boundary circles (50 m and 250 m)

        Shown when simulation_result is set
        ------------------------------------
        • Nominal landing ▼
        • MC scatter cloud
        • CEP ellipses (target cep_prob highlighted)
        """
        ax  = self.map_ax
        fig = self.map_fig
        ax.cla()
        _style_2d(ax, fig, bg="#0d0d1a")

        theta = np.linspace(0.0, 2.0 * np.pi, 200)

        # ── Target boundary circles ────────────────────────────────────────────
        ax.plot(50  * np.cos(theta), 50  * np.sin(theta),
                color="#f38ba8", lw=1.2, linestyle="--", alpha=0.60,
                label="Target r = 50 m")
        ax.plot(250 * np.cos(theta), 250 * np.sin(theta),
                color="#45475a", lw=1.0, linestyle="--", alpha=0.45,
                label="Target r = 250 m")

        # ── Launch point ───────────────────────────────────────────────────────
        ax.scatter([0], [0], c="#a6e3a1", s=130, marker="^", zorder=5,
                   label="Launch (0, 0)")

        res = self.state.simulation_result
        xlim = ylim = 300.0   # default view radius (m)

        if res is not None:
            lx = float(res.get("land_x", 0.0))
            ly = float(res.get("land_y", 0.0))

            # MC scatter
            mc_x = np.asarray(res.get("mc_scatter_x", []), dtype=float)
            mc_y = np.asarray(res.get("mc_scatter_y", []), dtype=float)
            n = min(len(mc_x), len(mc_y))
            if n > 0:
                ax.scatter(mc_x[:n], mc_y[:n],
                           c="#fab387", s=4, alpha=0.30, marker=".", zorder=3,
                           label=f"MC landings  (n = {n})")

            # KDE probability-mass contours (outermost → innermost)
            _kde_palette = ["#89b4fa", "#cba6f7", "#f38ba8", "#fab387", "#f9e2af"]
            for i, contour in enumerate(res.get("kde_contours", [])):
                pts = contour.get("points_m", [])
                if len(pts) < 2:
                    continue
                cx_pts = [float(p[0]) for p in pts]
                cy_pts = [float(p[1]) for p in pts]
                col = _kde_palette[i % len(_kde_palette)]
                lbl = contour.get("label",
                                  f"KDE {int(contour.get('prob_frac', 0) * 100)} %")
                # Close the contour path
                ax.plot(cx_pts + [cx_pts[0]], cy_pts + [cy_pts[0]],
                        color=col, lw=1.0, linestyle="-", alpha=0.55,
                        zorder=4, label=lbl)

            # CEP ellipses — highlight the one matching cep_prob
            target_prob = self.state.cep_prob
            for ell in res.get("cep_ellipses", []):
                if "a" not in ell or "b" not in ell:
                    continue
                lbl      = str(ell.get("label", ""))
                is_tgt   = str(target_prob) in lbl
                col      = str(ell.get("color", "#cba6f7" if is_tgt else "#585b70"))
                lw_val   = float(ell.get("lw", 2.0 if is_tgt else 0.9))
                alpha    = 0.95 if is_tgt else 0.35
                _draw_ellipse_2d(
                    ax,
                    cx=float(ell.get("cx", lx)),
                    cy=float(ell.get("cy", ly)),
                    a=float(ell["a"]), b=float(ell["b"]),
                    angle_rad=float(ell.get("angle_rad", 0.0)),
                    color=col, lw=lw_val, alpha=alpha,
                    label=lbl if lbl else "_nolegend_",
                )

            # Nominal landing ▼
            ax.scatter([lx], [ly], c="#f38ba8", s=130, marker="v", zorder=6,
                       label="Nominal landing")

            # Auto-scale to data
            all_x = np.concatenate([[0, lx], mc_x[:n] if n > 0 else []])
            all_y = np.concatenate([[0, ly], mc_y[:n] if n > 0 else []])
            pad   = max(abs(all_x).max(), abs(all_y).max()) * 0.25 + 60.0
            xlim = ylim = pad

        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
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

    # ══ Plot: Wind Profile + Time-Series ════════════════════════════════════

    def update_wind_plot(self) -> None:
        """
        Left subplot  — altitude vs wind speed (barh) + direction arrows.
        Right subplot — last-60-s wind speed time-series + moving average.
        Both show placeholder text when no data is available.
        """
        fig       = self.wind_fig
        ax_prof   = self.wind_ax_prof
        ax_ts     = self.wind_ax_ts
        profile   = self.state.wind_profile
        history   = self.state.wind_history

        for ax in (ax_prof, ax_ts):
            ax.cla()
            _style_2d(ax, bg="#1e1e2e")

        fig.patch.set_facecolor("#1e1e2e")

        # ── Left: altitude wind profile ────────────────────────────────────────
        if profile:
            alts  = np.array([p.get("alt",     0.0) for p in profile])
            spds  = np.array([p.get("speed",   0.0) for p in profile])
            dirs  = np.array([p.get("dir_deg", 0.0) for p in profile])
            bar_h = max(float(alts.max()) / max(len(alts), 1) * 0.7, 5.0)

            ax_prof.barh(alts, spds, height=bar_h,
                         color="#89b4fa", alpha=0.70, label="Speed")

            # Direction arrows overlaid at bar tips
            for alt, spd, d in zip(alts, spds, dirs):
                u = np.sin(np.radians(d))
                v = np.cos(np.radians(d))
                ax_prof.annotate(
                    "", xy=(spd + u * 0.8, alt + v * bar_h * 0.3),
                    xytext=(spd, alt),
                    arrowprops=dict(arrowstyle="->", color="#f9e2af",
                                   lw=1.0, alpha=0.75),
                )

            ax_prof.set_xlabel("Speed (m/s)", color="#6c7086", fontsize=7)
            ax_prof.set_ylabel("Altitude (m)", color="#6c7086", fontsize=7)
            ax_prof.legend(fontsize=6, facecolor="#1e1e2e",
                           edgecolor="#45475a", labelcolor="#cdd6f4")
        else:
            ax_prof.text(0.5, 0.5,
                         "No wind profile\navailable",
                         transform=ax_prof.transAxes,
                         ha="center", va="center",
                         color="#45475a", fontsize=9, linespacing=1.8)

        ax_prof.set_title("Wind Profile", color="#a6adc8", fontsize=8)

        # ── Right: 60-s time-series ────────────────────────────────────────────
        if history:
            times = np.array([h[0] for h in history], dtype=float)
            spds  = np.array([h[1] for h in history], dtype=float)

            # Clip to last 60 s
            t_max   = times[-1]
            mask    = times >= (t_max - 60.0)
            times_w = times[mask] - times[mask][0]
            spds_w  = spds[mask]

            ax_ts.plot(times_w, spds_w,
                       color="#89b4fa", lw=1.2, alpha=0.80, label="Wind speed")

            # Moving average (window ≈ 10 % of samples, min 3)
            n_ma = max(3, len(spds_w) // 10)
            if len(spds_w) >= n_ma:
                ma = np.convolve(spds_w, np.ones(n_ma) / n_ma, mode="valid")
                ax_ts.plot(times_w[n_ma - 1:], ma,
                           color="#f9e2af", lw=1.8,
                           label=f"MA ({n_ma})")

            ax_ts.set_xlabel("Time (s)",    color="#6c7086", fontsize=7)
            ax_ts.set_ylabel("Speed (m/s)", color="#6c7086", fontsize=7)
            ax_ts.legend(fontsize=6, facecolor="#1e1e2e",
                         edgecolor="#45475a", labelcolor="#cdd6f4")
        else:
            ax_ts.text(0.5, 0.5,
                       "No wind history\navailable",
                       transform=ax_ts.transAxes,
                       ha="center", va="center",
                       color="#45475a", fontsize=9, linespacing=1.8)

        ax_ts.set_title("Wind Time-Series  (last 60 s)",
                        color="#a6adc8", fontsize=8)

        fig.tight_layout(pad=0.4)
        self.wind_canvas.draw()

    # ── Action handlers ────────────────────────────────────────────────────────

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

    # ── Public API (called by AppController / SimController) ──────────────────

    def set_status(self, msg: str, color: Optional[str] = None) -> None:
        """Update left-side status bar text."""
        self._status_label.setText(msg)
        c = color or "#a6adc8"
        self._status_label.setStyleSheet(f"color: {c}; padding-left: 8px;")

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
        """Update the Phase 2 GO / NO-GO indicator."""
        if go:
            self._go_nogo_label.setText("✔   GO")
            self._go_nogo_label.setStyleSheet(
                "font-size: 12pt; font-weight: bold;"
                "color: #a6e3a1; padding: 6px;")
        else:
            self._go_nogo_label.setText("✘   NO-GO")
            self._go_nogo_label.setStyleSheet(
                "font-size: 12pt; font-weight: bold;"
                "color: #f38ba8; padding: 6px;")

    def set_progress(self, value: int, label: str = "") -> None:
        """Set toolbar progress bar (0–100)."""
        self._progress.setValue(max(0, min(100, value)))
        if label:
            self._progress.setFormat(label)


# ── Standalone entry point ────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    state = AppState()
    win   = AppWindow(state=state)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
