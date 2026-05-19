"""
ui_qt/app_window.py
Main application window — Kazamidori Project.

3-pane docking layout
---------------------
  parameters_dock (Left) |  profile_dock (Centre)  |  map_dock (Right)
  Airframe / Launch /        3-D trajectory + wind       2-D landing map
  Launch Mode / Run btn

All heavy computation lives in core/ — this module is view-only.

Standalone preview:
    python -m ui_qt.app_window
"""

from __future__ import annotations

import sys
from typing import Optional

import matplotlib
matplotlib.use("QtAgg")
# Windows 日本語フォントを優先し、CJK グリフ欠損警告を抑止する。
# DejaVu Sans はフォールバックとして末尾に残す。
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'DejaVu Sans']

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QSize, QObject, Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QScrollArea,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox,
    QComboBox, QPushButton, QToolBar, QStatusBar,
    QSizePolicy, QProgressBar, QFrame, QFileDialog,
    QMessageBox, QToolBox, QSplitter, QSlider,
    QDialog, QDialogButtonBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractButton,
)
from PySide6.QtGui import QAction, QColor
from ui_qt.map_view import MapView


# ── Constants ───────────────────────────────────────────────────────────────
DEFAULT_AZIMUTH: int = -90
AZIMUTH_STEP: int = 3
MARKER_SIZE: int = 50
SCROLL_STEP: int = 5
MAX_SCATTER_POINTS: int = 500
WIND_HISTORY_SAMPLES: int = 60

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
        self._show_kde = True
        self._show_cep = True
        self._show_scatter = True
        self._show_burnout = True
        self._show_apogee = True


    # ── View Toggles ───────────────────────────────────────────────────────────
    @property
    def show_kde(self) -> bool: return self._show_kde
    @show_kde.setter
    def show_kde(self, v: bool) -> None:
        if self._show_kde != v:
            self._show_kde = v
            self.needs_redraw.emit()

    @property
    def show_cep(self) -> bool: return self._show_cep
    @show_cep.setter
    def show_cep(self, v: bool) -> None:
        if self._show_cep != v:
            self._show_cep = v
            self.needs_redraw.emit()

    @property
    def show_scatter(self) -> bool: return self._show_scatter
    @show_scatter.setter
    def show_scatter(self, v: bool) -> None:
        if self._show_scatter != v:
            self._show_scatter = v
            self.needs_redraw.emit()

    @property
    def show_burnout(self) -> bool: return self._show_burnout
    @show_burnout.setter
    def show_burnout(self, v: bool) -> None:
        if self._show_burnout != v:
            self._show_burnout = v
            self.needs_redraw.emit()

    @property
    def show_apogee(self) -> bool: return self._show_apogee
    @show_apogee.setter
    def show_apogee(self, v: bool) -> None:
        if self._show_apogee != v:
            self._show_apogee = v
            self.needs_redraw.emit()

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


# ── Global dark theme ─────────────────────────────────────────────────────────
# Task 4 colour contract:
#   Background  : #2b2b2b   Text       : #ffffff
#   Input bg    : #3c3c3c   Deep bg    : #1e1e1e
#   Border      : #555555   Accent blue: #7eb3ff
_QSS = """
QMainWindow, QWidget {
    background-color: #2b2b2b;
    color: #ffffff;
    font-family: "Segoe UI", "SF Pro Text", Arial, sans-serif;
    font-size: 9pt;
}
QDialog { background-color: #2b2b2b; color: #ffffff; }
QToolBox::tab {
    background: #3c3c3c; color: #7eb3ff; font-weight: bold;
    font-size: 9pt; padding: 6px 10px;
    border: 1px solid #555555; border-radius: 4px; margin-bottom: 2px;
}
QToolBox::tab:selected { background: #4a4a4a; color: #c5a5f7; border-color: #c5a5f7; }
QToolBox::tab:hover    { background: #444444; border-color: #7eb3ff; }
QSplitter::handle         { background: #555555; width: 4px; height: 4px; }
QSplitter::handle:hover   { background: #7eb3ff; }
QGroupBox {
    border: 1px solid #555555; border-radius: 6px; margin-top: 12px;
    padding: 8px 6px 6px 6px; font-weight: bold; font-size: 8pt; color: #7eb3ff;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 4px;
    background-color: #2b2b2b;
}
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background: #3c3c3c; border: 1px solid #555555; border-radius: 4px;
    padding: 3px 6px; color: #ffffff; min-width: 80px;
}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
    border-color: #7eb3ff; background: #444444;
}
QLineEdit:disabled, QDoubleSpinBox:disabled,
QSpinBox:disabled,  QComboBox:disabled {
    background: #333333; color: #777777; border-color: #444444;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button,       QSpinBox::down-button {
    background: #555555; border: none; width: 16px; border-radius: 2px;
}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
QSpinBox::up-button:hover,       QSpinBox::down-button:hover { background: #666666; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #3c3c3c; border: 1px solid #555555;
    selection-background-color: #555555; color: #ffffff; outline: none;
}
QPushButton {
    background: #3c3c3c; border: 1px solid #555555; border-radius: 5px;
    padding: 5px 14px; color: #ffffff; font-weight: bold;
}
QPushButton:hover    { background: #4a4a4a; border-color: #7eb3ff; }
QPushButton:pressed  { background: #7eb3ff; color: #1e1e1e; }
QPushButton:disabled { background: #333333; color: #666666; border-color: #444444; }
QPushButton#btn_run  { background: #a8e6a1; color: #1e1e1e; border-color: #a8e6a1; }
QPushButton#btn_run:hover  { background: #8ed9a8; }
QPushButton#btn_stop { background: #f38ba8; color: #1e1e1e; border-color: #f38ba8; }
QPushButton#btn_stop:hover { background: #eba0ac; }
QPushButton#btn_phase1_run {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #c5a5f7, stop:1 #7eb3ff);
    color: #1e1e1e; border: none; border-radius: 6px;
    font-size: 10pt; font-weight: bold; padding: 10px 16px;
}
QPushButton#btn_phase1_run:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #d4b5ff, stop:1 #9dc5ff);
}
QPushButton#btn_phase1_run:pressed { background: #7eb3ff; color: #1e1e1e; }
QPushButton#btn_adv_settings {
    background: transparent; border: 1px solid #555555; border-radius: 4px;
    padding: 4px 10px; color: #aaaaaa; font-size: 8pt;
}
QPushButton#btn_adv_settings:hover { border-color: #7eb3ff; color: #ffffff; }
QToolBar {
    background: #1e1e1e; border: none;
    border-bottom: 1px solid #3c3c3c; padding: 3px 6px; spacing: 4px;
}
QToolBar QToolButton {
    background: transparent; border: 1px solid transparent;
    border-radius: 4px; padding: 3px 8px; color: #ffffff;
}
QToolBar QToolButton:hover   { background: #3c3c3c; border-color: #555555; }
QToolBar QToolButton:pressed { background: #555555; }
QMenuBar { background: #1e1e1e; color: #ffffff; border-bottom: 1px solid #3c3c3c; }
QMenuBar::item { padding: 5px 12px; background: transparent; }
QMenuBar::item:selected { background: #3c3c3c; border-radius: 3px; }
QMenu {
    background: #2b2b2b; border: 1px solid #555555;
    border-radius: 4px; padding: 4px;
}
QMenu::item { padding: 5px 20px 5px 12px; border-radius: 3px; }
QMenu::item:selected { background: #3c3c3c; color: #7eb3ff; }
QMenu::separator { height: 1px; background: #555555; margin: 3px 8px; }
QStatusBar {
    background: #1e1e1e; color: #aaaaaa;
    border-top: 1px solid #3c3c3c; font-size: 8pt;
}
QStatusBar::item { border: none; }
QScrollBar:vertical   { background: #2b2b2b; width: 8px;  margin: 0; }
QScrollBar:horizontal { background: #2b2b2b; height: 8px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #555555; border-radius: 4px;
    min-height: 24px; min-width: 24px;
}
QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover { background: #666666; }
QScrollBar::add-line:vertical,  QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { height: 0; width: 0; }
QProgressBar {
    background: #3c3c3c; border: 1px solid #555555; border-radius: 4px;
    text-align: center; color: #ffffff; font-size: 8pt; max-height: 18px;
}
QProgressBar::chunk { background: #7eb3ff; border-radius: 3px; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: #2b2b2b; }
QLabel { color: #ffffff; }
QTableWidget {
    background: #1a1a2e; color: #cdd6f4; border: none;
    gridline-color: #2a2a3e; alternate-background-color: #12121e;
    selection-background-color: #313244;
}
QTableWidget::item { padding: 3px 5px; border: none; font-size: 8pt; }
QHeaderView::section {
    background: #2b2b2b; color: #7eb3ff; border: 1px solid #45475a;
    padding: 3px 5px; font-weight: bold; font-size: 7pt;
}
QTableCornerButton::section { background: #2b2b2b; border: 1px solid #45475a; }
QFormLayout QLabel { color: #aaaaaa; }
QTabBar::tab { min-height: 30px; padding: 5px 10px; }
"""

# Exported so main_qt.py can apply it to QApplication (global scope)
GLOBAL_QSS = _QSS


# ── Cd curve preview dialog (Phase D) ─────────────────────────────────────────

class CdCurvePreviewDialog(QDialog):
    """
    Modal preview window for a Mach-dependent Cd curve loaded into AppState.

    Plots Mach (X) vs Cd (Y) on a Matplotlib canvas embedded in a small,
    resizable Qt dialog.  The figure is owned by the dialog (NOT by pyplot)
    so it never enters the global pyplot registry — closing the dialog
    releases the figure deterministically.

    Memory hygiene
    --------------
    Matplotlib figures can leak when:
      *  Held by ``pyplot`` indefinitely (we avoid this by using
         :class:`Figure` directly).
      *  Their canvas keeps a back-reference after Qt has destroyed the
         widget hierarchy.
    ``closeEvent`` explicitly clears the figure and schedules the canvas
    for deletion, guaranteeing both reference paths drop on close.
    """

    def __init__(
        self,
        title: str,
        curve: list[tuple[float, float]],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(560, 420)
        self.setModal(True)

        from matplotlib.figure                import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        # Use ``Figure`` directly (no pyplot) so the figure is never added to
        # the global pyplot figure registry — releasing it here is sufficient
        # to free its memory.
        self._fig    = Figure(figsize=(5.4, 3.6), constrained_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        ax           = self._fig.add_subplot(111)

        machs = [pt[0] for pt in curve]
        cds   = [pt[1] for pt in curve]
        ax.plot(machs, cds, marker="o", linewidth=1.5, color="#89b4fa")
        ax.set_xlabel("Mach")
        ax.set_ylabel("Cd")
        ax.set_title(f"{len(curve)} interpolation points")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._canvas, 1)

        btn_close = QPushButton("Close")
        btn_close.setToolTip("Close the preview (Esc)")
        btn_close.setShortcut("Esc")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, 0, Qt.AlignmentFlag.AlignRight)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        # Explicitly tear down Matplotlib resources.  ``Figure.clear()``
        # detaches all axes/artists; ``deleteLater`` schedules the Qt
        # widget for destruction on the next event loop tick.  Together
        # these drop the only two references to the Figure object, so
        # garbage collection can reclaim its arrays immediately.
        try:
            self._fig.clear()
        except Exception:
            pass
        try:
            self._canvas.deleteLater()
        except Exception:
            pass
        super().closeEvent(event)


# ── Advanced Settings Dialog ──────────────────────────────────────────────────

class AdvancedSettingsDialog(QDialog):
    """
    Modal dialog for rarely-changed simulation parameters.

    All input widgets are public attributes so AppWindow can expose them
    directly on ``self`` for SimController compatibility.  Values are live
    — they persist between dialog open/close cycles and are read by
    SimController._collect_params() at run time.

    The OK button simply closes the dialog (values already updated).
    Cancel restores the snapshot taken when the dialog was last opened.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.setMinimumWidth(440)
        self.setModal(True)
        self._snapshot: dict = {}
        # Bi-directional binding to AppState — installed lazily via
        # ``bind_app_state()`` so the dialog can be constructed before the
        # global AppState exists.  None means "no binding active yet".
        self._bound_state = None
        self._build()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 10)

        # ── Wind group ────────────────────────────────────────────────────────
        grp_wind = QGroupBox("Wind Parameters")
        frm = QFormLayout(grp_wind)
        frm.setSpacing(6)
        frm.setContentsMargins(10, 12, 10, 8)

        self.surf_spd_input = QDoubleSpinBox()
        self.surf_spd_input.setRange(0, 50); self.surf_spd_input.setDecimals(1)
        self.surf_spd_input.setValue(4.0);   self.surf_spd_input.setSuffix(" m/s")
        self.surf_spd_input.wheelEvent = lambda event: event.ignore()

        self.surf_dir_input = QDoubleSpinBox()
        self.surf_dir_input.setRange(0, 360); self.surf_dir_input.setDecimals(1)
        self.surf_dir_input.setValue(100.0);  self.surf_dir_input.setSuffix("°")
        self.surf_dir_input.setWrapping(True)
        self.surf_dir_input.wheelEvent = lambda event: event.ignore()

        self.up_spd_input = QDoubleSpinBox()
        self.up_spd_input.setRange(0, 100); self.up_spd_input.setDecimals(1)
        self.up_spd_input.setValue(8.0);    self.up_spd_input.setSuffix(" m/s")
        self.up_spd_input.wheelEvent = lambda event: event.ignore()

        self.up_dir_input = QDoubleSpinBox()
        self.up_dir_input.setRange(0, 360); self.up_dir_input.setDecimals(1)
        self.up_dir_input.setValue(90.0);   self.up_dir_input.setSuffix("°")
        self.up_dir_input.setWrapping(True)
        self.up_dir_input.wheelEvent = lambda event: event.ignore()

        frm.addRow("Surface Wind Speed (0 m):",   self.surf_spd_input)
        frm.addRow("Surface Wind From  (0 m):",   self.surf_dir_input)
        frm.addRow("Upper Wind Speed (500 m):",   self.up_spd_input)
        frm.addRow("Upper Wind From  (500 m):",   self.up_dir_input)

        # ── Monte Carlo group ─────────────────────────────────────────────────
        grp_mc = QGroupBox("Monte Carlo / Statistics")
        frm2 = QFormLayout(grp_mc)
        frm2.setSpacing(6)
        frm2.setContentsMargins(10, 12, 10, 8)

        self.cep_prob_input = QSpinBox()
        self.cep_prob_input.setRange(50, 99); self.cep_prob_input.setValue(90)
        self.cep_prob_input.setSuffix(" %")
        self.cep_prob_input.wheelEvent = lambda event: event.ignore()

        self.mc_runs_input = QSpinBox()
        self.mc_runs_input.setRange(10, 5000); self.mc_runs_input.setValue(200)
        self.mc_runs_input.setSingleStep(50)
        self.mc_runs_input.wheelEvent = lambda event: event.ignore()

        self.wind_unc_input = QDoubleSpinBox()
        self.wind_unc_input.setRange(0, 1);  self.wind_unc_input.setDecimals(2)
        self.wind_unc_input.setValue(0.20);  self.wind_unc_input.setSingleStep(0.01)
        self.wind_unc_input.setSuffix("  (±ratio)")
        self.wind_unc_input.wheelEvent = lambda event: event.ignore()

        self.thrust_unc_input = QDoubleSpinBox()
        self.thrust_unc_input.setRange(0, 1);  self.thrust_unc_input.setDecimals(2)
        self.thrust_unc_input.setValue(0.05);  self.thrust_unc_input.setSingleStep(0.01)
        self.thrust_unc_input.setSuffix("  (±ratio)")
        self.thrust_unc_input.wheelEvent = lambda event: event.ignore()

        frm2.addRow("CEP Probability:",    self.cep_prob_input)
        frm2.addRow("MC Runs:",            self.mc_runs_input)
        frm2.addRow("Wind Uncertainty:",   self.wind_unc_input)
        frm2.addRow("Thrust Uncertainty:", self.thrust_unc_input)

        # ── Aerodynamics & Motor group ────────────────────────────────────────
        # These four parameters were previously hard-coded inside core/.
        # Exposing them lets the operator pick the correct drag model for
        # boost vs coast and the correct propellant chemistry (default = BP).
        grp_aero = QGroupBox("Aerodynamics & Motor")
        frm3 = QFormLayout(grp_aero)
        frm3.setSpacing(6)
        frm3.setContentsMargins(10, 12, 10, 8)

        self.power_on_cd_input = QDoubleSpinBox()
        self.power_on_cd_input.setRange(0.0, 2.0)
        self.power_on_cd_input.setDecimals(3)
        self.power_on_cd_input.setSingleStep(0.01)
        self.power_on_cd_input.setValue(0.45)
        self.power_on_cd_input.wheelEvent = lambda event: event.ignore()

        self.power_off_cd_input = QDoubleSpinBox()
        self.power_off_cd_input.setRange(0.0, 2.0)
        self.power_off_cd_input.setDecimals(3)
        self.power_off_cd_input.setSingleStep(0.01)
        self.power_off_cd_input.setValue(0.40)
        self.power_off_cd_input.wheelEvent = lambda event: event.ignore()

        self.motor_isp_input = QDoubleSpinBox()
        self.motor_isp_input.setRange(40.0, 300.0)
        self.motor_isp_input.setDecimals(1)
        self.motor_isp_input.setSingleStep(1.0)
        self.motor_isp_input.setValue(80.0)
        self.motor_isp_input.setSuffix(" s")
        self.motor_isp_input.wheelEvent = lambda event: event.ignore()

        self.motor_propellant_density_input = QDoubleSpinBox()
        self.motor_propellant_density_input.setRange(500.0, 2500.0)
        self.motor_propellant_density_input.setDecimals(0)
        self.motor_propellant_density_input.setSingleStep(10.0)
        self.motor_propellant_density_input.setValue(1700.0)
        self.motor_propellant_density_input.setSuffix(" kg/m³")
        self.motor_propellant_density_input.wheelEvent = lambda event: event.ignore()

        # ── Mach-dependent Cd curve rows (Phase C) ────────────────────────────
        # Each row groups: [Load CSV] [Clear] [status label].
        # Status label shows ``Curve Loaded`` (with point count) or
        # ``Using Static Value`` and is the canonical readout of whether
        # the curve overrides the scalar Cd above.
        def _build_curve_row() -> tuple[QPushButton, QPushButton, QPushButton, QLabel, QWidget]:
            host  = QWidget()
            row   = QHBoxLayout(host)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            btn_load    = QPushButton("Load Cd Curve…")
            btn_load.setToolTip("Load a Mach-dependent Cd curve from a CSV file")
            btn_preview = QPushButton("Preview")
            btn_preview.setToolTip("Preview the currently loaded Cd curve")
            btn_clear   = QPushButton("Clear")
            btn_clear.setToolTip("Clear the loaded Cd curve and fallback to the static scalar value")
            lbl         = QLabel("Using Static Value")
            lbl.setStyleSheet("color: #888888;")
            row.addWidget(btn_load)
            row.addWidget(btn_preview)
            row.addWidget(btn_clear)
            row.addWidget(lbl, 1)
            return btn_load, btn_preview, btn_clear, lbl, host

        (self.power_on_cd_curve_load_btn,
         self.power_on_cd_curve_preview_btn,
         self.power_on_cd_curve_clear_btn,
         self.power_on_cd_curve_label,
         _row_on)  = _build_curve_row()

        (self.power_off_cd_curve_load_btn,
         self.power_off_cd_curve_preview_btn,
         self.power_off_cd_curve_clear_btn,
         self.power_off_cd_curve_label,
         _row_off) = _build_curve_row()

        self.power_on_cd_curve_load_btn.clicked.connect(
            lambda: self._on_load_cd_curve("power_on"))
        self.power_on_cd_curve_preview_btn.clicked.connect(
            lambda: self._on_preview_cd_curve("power_on"))
        self.power_on_cd_curve_clear_btn.clicked.connect(
            lambda: self._on_clear_cd_curve("power_on"))
        self.power_off_cd_curve_load_btn.clicked.connect(
            lambda: self._on_load_cd_curve("power_off"))
        self.power_off_cd_curve_preview_btn.clicked.connect(
            lambda: self._on_preview_cd_curve("power_off"))
        self.power_off_cd_curve_clear_btn.clicked.connect(
            lambda: self._on_clear_cd_curve("power_off"))

        frm3.addRow("Power-On  Cd (boost):",  self.power_on_cd_input)
        frm3.addRow("Power-On  Cd Curve:",    _row_on)
        frm3.addRow("Power-Off Cd (coast):",  self.power_off_cd_input)
        frm3.addRow("Power-Off Cd Curve:",    _row_off)
        frm3.addRow("Motor Isp:",             self.motor_isp_input)
        frm3.addRow("Propellant Density:",    self.motor_propellant_density_input)

        # ── OK / Cancel ───────────────────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self._on_cancel)

        root.addWidget(grp_wind)
        root.addWidget(grp_mc)
        root.addWidget(grp_aero)
        root.addWidget(btns)

    # ── Cancel / snapshot helpers ─────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._snapshot = self._take_snapshot()

    def _take_snapshot(self) -> dict:
        return {
            "surf_spd":      self.surf_spd_input.value(),
            "surf_dir":      self.surf_dir_input.value(),
            "up_spd":        self.up_spd_input.value(),
            "up_dir":        self.up_dir_input.value(),
            "cep_prob":      self.cep_prob_input.value(),
            "mc_runs":       self.mc_runs_input.value(),
            "wind_unc":      self.wind_unc_input.value(),
            "thr_unc":       self.thrust_unc_input.value(),
            "power_on_cd":   self.power_on_cd_input.value(),
            "power_off_cd":  self.power_off_cd_input.value(),
            "motor_isp":     self.motor_isp_input.value(),
            "prop_density":  self.motor_propellant_density_input.value(),
        }

    def _on_cancel(self) -> None:
        s = self._snapshot
        self.surf_spd_input.setValue(s["surf_spd"])
        self.surf_dir_input.setValue(s["surf_dir"])
        self.up_spd_input.setValue(s["up_spd"])
        self.up_dir_input.setValue(s["up_dir"])
        self.cep_prob_input.setValue(s["cep_prob"])
        self.mc_runs_input.setValue(s["mc_runs"])
        self.wind_unc_input.setValue(s["wind_unc"])
        self.thrust_unc_input.setValue(s["thr_unc"])
        self.power_on_cd_input.setValue(s["power_on_cd"])
        self.power_off_cd_input.setValue(s["power_off_cd"])
        self.motor_isp_input.setValue(s["motor_isp"])
        self.motor_propellant_density_input.setValue(s["prop_density"])
        self.reject()

    # ── AppState binding ─────────────────────────────────────────────────────

    def bind_app_state(self, state) -> None:
        """Establish bi-directional binding between the four advanced inputs
        and the corresponding AppState properties.

        Direction widget → state: the spinbox's ``valueChanged`` signal writes
        through to the AppState property setter, which emits its own change
        signal so other observers (workers, plot views) stay in sync.

        Direction state → widget: the AppState change signal pushes the new
        value back into the spinbox.  ``QSignalBlocker`` is used during the
        push to avoid re-emitting ``valueChanged`` and creating a feedback loop.

        Safe to call exactly once with a real ``AppState``; subsequent calls
        are no-ops.  Idempotent re-binding is intentionally not supported in
        Phase B (a single AppState lives for the whole session).
        """
        if state is None or self._bound_state is state:
            return
        self._bound_state = state

        # Seed widgets with the current AppState values so the dialog opens
        # showing the authoritative state, not the hard-coded spinbox defaults.
        from PySide6.QtCore import QSignalBlocker
        for widget, attr in (
            (self.power_on_cd_input,              "power_on_cd"),
            (self.power_off_cd_input,             "power_off_cd"),
            (self.motor_isp_input,                "motor_isp"),
            (self.motor_propellant_density_input, "motor_propellant_density"),
        ):
            with QSignalBlocker(widget):
                widget.setValue(float(getattr(state, attr)))

        # widget → state
        self.power_on_cd_input.valueChanged.connect(
            lambda v: setattr(state, "power_on_cd", float(v)))
        self.power_off_cd_input.valueChanged.connect(
            lambda v: setattr(state, "power_off_cd", float(v)))
        self.motor_isp_input.valueChanged.connect(
            lambda v: setattr(state, "motor_isp", float(v)))
        self.motor_propellant_density_input.valueChanged.connect(
            lambda v: setattr(state, "motor_propellant_density", float(v)))

        # state → widget (guarded with QSignalBlocker to avoid loops)
        def _push(widget, value):
            if widget.value() != float(value):
                with QSignalBlocker(widget):
                    widget.setValue(float(value))

        state.power_on_cd_changed.connect(
            lambda v: _push(self.power_on_cd_input, v))
        state.power_off_cd_changed.connect(
            lambda v: _push(self.power_off_cd_input, v))
        state.motor_isp_changed.connect(
            lambda v: _push(self.motor_isp_input, v))
        state.motor_propellant_density_changed.connect(
            lambda v: _push(self.motor_propellant_density_input, v))

        # ── Cd curve labels (Phase C) ────────────────────────────────────────
        # The Load / Clear buttons mutate AppState directly; the labels are
        # purely view-side reflections of the curve property.  Connecting
        # here means the readout updates regardless of who triggered the
        # change (UI button, programmatic load, deserialised session, …).
        state.cd_curve_power_on_changed.connect(
            lambda curve: self._refresh_curve_label(
                self.power_on_cd_curve_label, curve))
        state.cd_curve_power_off_changed.connect(
            lambda curve: self._refresh_curve_label(
                self.power_off_cd_curve_label, curve))
        self._refresh_curve_label(
            self.power_on_cd_curve_label,  state.cd_curve_power_on)
        self._refresh_curve_label(
            self.power_off_cd_curve_label, state.cd_curve_power_off)

    # ── Cd curve handlers (Phase C) ──────────────────────────────────────────

    @staticmethod
    def _refresh_curve_label(label: QLabel, curve) -> None:
        """Update *label* to reflect whether a Cd curve is currently loaded."""
        if curve is not None and len(curve) >= 2:
            label.setText(f"Curve Loaded ({len(curve)} pts)")
            label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        else:
            label.setText("Using Static Value")
            label.setStyleSheet("color: #888888;")

    def _on_load_cd_curve(self, which: str) -> None:
        """Open a file dialog, parse the chosen CSV, and write the result back
        into AppState's ``cd_curve_power_on`` / ``cd_curve_power_off``.

        ``which`` is ``"power_on"`` or ``"power_off"``.  Errors are surfaced
        through :class:`QMessageBox` warnings without raising, so a bad file
        leaves the existing curve (or static-value fallback) untouched.
        """
        if self._bound_state is None:
            QMessageBox.warning(
                self, "Cd Curve Loader",
                "AppState is not yet wired to this dialog. "
                "The Cd curve cannot be stored.")
            return

        title = ("Load Power-On Cd Curve" if which == "power_on"
                 else "Load Coast (Power-Off) Cd Curve")
        filepath, _ = QFileDialog.getOpenFileName(
            self, title, "", "CSV files (*.csv);;All files (*.*)")
        if not filepath:
            return

        try:
            from utils.data_loader import parse_cd_curve_csv
            curve = parse_cd_curve_csv(filepath)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(
                self, "Cd Curve Parse Error",
                f"Failed to load Cd curve from:\n{filepath}\n\n{exc}")
            return

        attr = "cd_curve_power_on" if which == "power_on" else "cd_curve_power_off"
        setattr(self._bound_state, attr, curve)

    def _on_clear_cd_curve(self, which: str) -> None:
        """Reset the chosen Cd curve to ``None`` so the simulation falls back
        to the corresponding scalar (``power_on_cd`` / ``power_off_cd``)."""
        if self._bound_state is None:
            return
        attr = "cd_curve_power_on" if which == "power_on" else "cd_curve_power_off"
        setattr(self._bound_state, attr, None)

    def _on_preview_cd_curve(self, which: str) -> None:
        """Pop a Matplotlib preview of the currently-loaded Cd curve.

        Shows an informational message box (rather than an error) when no
        curve has been loaded — the scalar fallback is the *intended* state,
        not a misconfiguration.
        """
        if self._bound_state is None:
            QMessageBox.warning(
                self, "Cd Curve Preview",
                "AppState is not yet wired to this dialog.")
            return

        attr  = "cd_curve_power_on" if which == "power_on" else "cd_curve_power_off"
        curve = getattr(self._bound_state, attr, None)

        if curve is None or len(curve) < 2:
            QMessageBox.information(
                self, "Cd Curve Preview",
                "No curve loaded. Using static scalar value.")
            return

        title = ("Power-On Cd Curve" if which == "power_on"
                 else "Coast (Power-Off) Cd Curve")
        # ``self`` parents the preview dialog so Qt destroys it automatically
        # if the Advanced Settings dialog is closed while it is still open.
        preview = CdCurvePreviewDialog(title, list(curve), parent=self)
        preview.exec()


# ── Manual rocket geometry dialog ─────────────────────────────────────────────

class ManualSetupDialog(QDialog):
    """
    Modal dialog for manually entering all 12 rocket airframe geometry parameters.

    Spinbox widgets are public attributes; AppWindow exposes them as proxy
    attributes so SimController._wire_airframe_spinboxes() requires zero changes.

    The Load JSON / Save JSON buttons emit signals that AppWindow forwards to the
    controller — no file I/O occurs inside the dialog itself.
    """

    sig_load_json = Signal()   # forwarded → AppWindow.sig_load_rocket_json_clicked
    sig_save_json = Signal()   # forwarded → AppWindow.sig_save_rocket_json_clicked
    sig_reset     = Signal()   # forwarded to handle reset

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manual Rocket Configuration")
        self.setMinimumWidth(440)
        self.setModal(True)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 10)

        # ── Scrollable airframe form ──────────────────────────────────────────
        inner  = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(0, 0, 0, 0)
        inner_lay.setSpacing(6)

        grp = QGroupBox("Airframe  (m · kg · s  from nose tip)")
        frm = QFormLayout(grp)
        frm.setSpacing(5)
        frm.setContentsMargins(2, 4, 2, 2)

        def _dsb(hi, dec, step, suffix):
            sb = QDoubleSpinBox()
            sb.setDecimals(dec); sb.setSingleStep(step); sb.setSuffix(suffix)
            sb.setRange(-9999.0, hi)
            sb.setSpecialValueText("")
            sb.setValue(-9999.0)
            sb.wheelEvent = lambda event: event.ignore()
            return sb

        self.af_mass_input      = _dsb(50.0, 4, 0.001, " kg")
        self.af_cg_input        = _dsb( 5.0, 3, 0.001, " m")
        self.af_len_input       = _dsb( 5.0, 3, 0.001, " m")
        self.af_radius_input    = _dsb( 0.5, 4, 0.001, " m")
        self.af_nose_input      = _dsb( 2.0, 3, 0.001, " m")
        self.af_finroot_input   = _dsb( 1.0, 3, 0.001, " m")
        self.af_fintip_input    = _dsb( 1.0, 3, 0.001, " m")
        self.af_finspan_input   = _dsb( 1.0, 3, 0.001, " m")
        self.af_finpos_input    = _dsb( 5.0, 3, 0.001, " m")
        self.af_motorpos_input  = _dsb( 5.0, 3, 0.001, " m")
        self.af_motormass_input = _dsb( 5.0, 4, 0.001, " kg")

        self.lbl_mass      = QLabel("Mass [kg]:")
        self.lbl_cg        = QLabel("CG from Nose [m]:")
        self.lbl_len       = QLabel("Length [m]:")
        self.lbl_radius    = QLabel("Body Radius [m]:")
        self.lbl_nose      = QLabel("Nose Length [m]:")
        self.lbl_finroot   = QLabel("Fin Root Chord [m]:")
        self.lbl_fintip    = QLabel("Fin Tip Chord [m]:")
        self.lbl_finspan   = QLabel("Fin Semi-Span [m]:")
        self.lbl_finpos    = QLabel("Fin LE Position [m]:")
        self.lbl_motorpos  = QLabel("Motor CG Pos. [m]:")
        self.lbl_motormass = QLabel("Motor Dry Mass [kg]:")

        frm.addRow(self.lbl_mass,      self.af_mass_input)
        frm.addRow(self.lbl_cg,        self.af_cg_input)
        frm.addRow(self.lbl_len,       self.af_len_input)
        frm.addRow(self.lbl_radius,    self.af_radius_input)
        frm.addRow(self.lbl_nose,      self.af_nose_input)
        frm.addRow(self.lbl_finroot,   self.af_finroot_input)
        frm.addRow(self.lbl_fintip,    self.af_fintip_input)
        frm.addRow(self.lbl_finspan,   self.af_finspan_input)
        frm.addRow(self.lbl_finpos,    self.af_finpos_input)
        frm.addRow(self.lbl_motorpos,  self.af_motorpos_input)
        frm.addRow(self.lbl_motormass, self.af_motormass_input)


        inner_lay.addWidget(grp)



        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        sa.setWidget(inner)

        # ── Load / Save JSON buttons ──────────────────────────────────────────
        btn_load = QPushButton("📂  Load JSON (rocket.json)")
        btn_load.setToolTip("Load rocket geometry configuration from a JSON file")
        btn_save = QPushButton("💾  Save JSON")
        btn_save.setToolTip("Save the current rocket geometry configuration to a JSON file")
        self.btn_reset = QPushButton("Reset Configuration")
        self.btn_reset.setToolTip("Restore all tracking parameters back to the original values from the loaded .rkt file")

        self.btn_reset.clicked.connect(self.sig_reset.emit)
        btn_load.clicked.connect(self.sig_load_json.emit)
        btn_save.clicked.connect(self.sig_save_json.emit)
        btn_row = QHBoxLayout()
        btn_row.addWidget(btn_load)
        btn_row.addWidget(btn_save)

        # ── Close ─────────────────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)

        root.addWidget(sa, stretch=1)
        root.addLayout(btn_row)
        root.addWidget(self.btn_reset)
        root.addWidget(btns)

        self.setMinimumHeight(500)


# ── Matplotlib canvas wrapper ─────────────────────────────────────────────────

class _MplCanvas(FigureCanvasQTAgg):
    def __init__(self, fig: Figure, parent: Optional[QWidget] = None) -> None:
        FigureCanvasQTAgg.__init__(self, fig)
        if parent is not None:
            self.setParent(parent)


class _AzimCanvas(FigureCanvasQTAgg):
    """3-D profile canvas: mouse-wheel adjusts the linked azimuth QSlider (±3° per notch) or scrubs animation."""

    def __init__(self, fig: Figure, parent: Optional[QWidget] = None) -> None:
        FigureCanvasQTAgg.__init__(self, fig)
        if parent is not None:
            self.setParent(parent)
        self._current_time_idx = 0
        self._marker_artist = None
        self._traj_x = None
        self._traj_y = None
        self._traj_z = None
        self._traj_t = None
        self._time_label = None

    def wheelEvent(self, event) -> None:
        from PySide6.QtCore import Qt
        if event.modifiers() & Qt.ShiftModifier:
            if self._traj_t is not None and len(self._traj_t) > 0:
                delta = event.angleDelta().y()
                step = SCROLL_STEP if delta > 0 else -SCROLL_STEP
                self._current_time_idx = max(0, min(len(self._traj_t) - 1, self._current_time_idx + step))
                self._update_marker()
                event.accept()
                return

        sl = getattr(self, '_azim_slider', None)
        if sl is not None:
            delta = event.angleDelta().y()
            step = AZIMUTH_STEP if delta > 0 else -AZIMUTH_STEP
            sl.setValue(max(sl.minimum(), min(sl.maximum(), sl.value() + step)))
        event.accept()

    def set_trajectory(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.ndarray, ax: object) -> None:
        """Set the trajectory arrays and reset the time index.

        Args:
            x: Array of X coordinates.
            y: Array of Y coordinates.
            z: Array of Z coordinates.
            t: Array of time values.
            ax: Matplotlib 3D axes object.
        """
        self._traj_x = x
        self._traj_y = y
        self._traj_z = z
        self._traj_t = t
        self._current_time_idx = 0

        if self._marker_artist:
            try:
                self._marker_artist.remove()
            except:
                pass
        if self._time_label:
            try:
                self._time_label.remove()
            except:
                pass

        # Draw the marker point
        self._marker_artist = ax.scatter([x[0]], [y[0]], [z[0]], color='red', s=MARKER_SIZE, zorder=10)
        # Use text2D on the axis to place it at the top-left
        self._time_label = ax.text2D(0.05, 0.95, f"T+ {t[0]:.1f}s | Alt: {z[0]:.1f}m", transform=ax.transAxes, color='#cdd6f4', fontsize=10, weight='bold')
        self.draw_idle()

    def _update_marker(self) -> None:
        """Update the position of the 3D marker and the time label based on current index."""
        if self._marker_artist and self._traj_x is not None:
            idx = self._current_time_idx
            self._marker_artist._offsets3d = ([self._traj_x[idx]], [self._traj_y[idx]], [self._traj_z[idx]])
            if self._time_label:
                self._time_label.set_text(f"T+ {self._traj_t[idx]:.1f}s | Alt: {self._traj_z[idx]:.1f}m")
            self.draw_idle()


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
    limits  = np.array([ax.get_xlim3d(), ax.get_ylim3d()])
    centers = limits.mean(axis=1)
    max_r   = max((limits[:, 1] - limits[:, 0]).max() / 2.0, 1.0)
    ax.set_xlim3d(centers[0] - max_r, centers[0] + max_r)
    ax.set_ylim3d(centers[1] - max_r, centers[1] + max_r)


def _make_altitude_lc(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> object:
    """Create a 3D LineCollection coloured by altitude.

    Args:
        x: Array of X coordinates.
        y: Array of Y coordinates.
        z: Array of Z coordinates.

    Returns:
        A Line3DCollection object.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import matplotlib.cm as _cm
    pts  = np.column_stack([x, y, z])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    z_mid = (z[:-1] + z[1:]) / 2.0
    norm  = (z_mid - z.min()) / max(z.max() - z.min(), 1e-6)
    return Line3DCollection(segs, colors=_cm.cool(norm), linewidth=2.0, alpha=0.92)


def _draw_ellipse_3d(ax: object, *, cx: float, cy: float, a: float, b: float, angle_rad: float = 0.0,
                     color: str = "#cba6f7", lw: float = 1.6, label: str = "") -> None:
    """Draw an ellipse projected onto the Z=0 plane in a 3D plot.

    Args:
        ax: Matplotlib 3D axes object.
        cx: Center X coordinate.
        cy: Center Y coordinate.
        a: Semi-major axis length.
        b: Semi-minor axis length.
        angle_rad: Rotation angle in radians.
        color: Stroke color.
        lw: Line width.
        label: Legend label.
    """
    t  = np.linspace(0.0, 2.0 * np.pi, 120)
    xe = a * np.cos(t) * np.cos(angle_rad) - b * np.sin(t) * np.sin(angle_rad)
    ye = a * np.cos(t) * np.sin(angle_rad) + b * np.sin(t) * np.cos(angle_rad)
    ax.plot(cx + xe, cy + ye, np.zeros(120),
            color=color, lw=lw, linestyle="--", alpha=0.90,
            label=label if label else "_nolegend_")


def _draw_ellipse_2d(ax: object, *, cx: float, cy: float, a: float, b: float, angle_rad: float = 0.0,
                     color: str = "#cba6f7", lw: float = 1.6, alpha: float = 0.90, label: str = "") -> object:
    """Draw a 2D ellipse.

    Args:
        ax: Matplotlib 2D axes object.
        cx: Center X coordinate.
        cy: Center Y coordinate.
        a: Semi-major axis length.
        b: Semi-minor axis length.
        angle_rad: Rotation angle in radians.
        color: Stroke color.
        lw: Line width.
        alpha: Transparency.
        label: Legend label.

    Returns:
        The created Line2D artist.
    """
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
    surf_spd_input, surf_dir_input    : QDoubleSpinBox  (in AdvancedSettingsDialog)
    up_spd_input,   up_dir_input      : QDoubleSpinBox  (in AdvancedSettingsDialog)
    cep_prob_input                    : QSpinBox        (in AdvancedSettingsDialog)
    mc_runs_input                     : QSpinBox        (in AdvancedSettingsDialog)
    wind_unc_input, thrust_unc_input  : QDoubleSpinBox  (in AdvancedSettingsDialog)
    wind_speed_input, wind_dir_input  : QDoubleSpinBox  (aliases → surf_spd/dir)
    lat_input, lon_input              : QDoubleSpinBox  (in Launch Settings tab)
    elev_input, azim_input            : QDoubleSpinBox  (in Launch Settings tab)
    motor_label                       : QLabel          (in Airframe tab)
    mode_combo                        : QComboBox       (pinned in Parameters dock)
    rmax_input                        : QDoubleSpinBox  (pinned in Parameters dock)
    map_widget                        : _MapCoordProxy

    Signals
    -------
    sig_load_rocket_json_clicked : emitted when the "Load Rocket JSON" button is clicked.

    Window-internal reactive state
    ------------------------------
    state : AppState  — drives profile / map / wind canvases via needs_redraw
    """

    sig_load_rocket_json_clicked = Signal()   # Load rocket.json (from ManualSetupDialog)
    sig_load_rkt_clicked         = Signal()   # Load .rkt file
    sig_load_para_json_clicked   = Signal()   # Load parachute-only JSON
    sig_save_rocket_json_clicked = Signal()   # Export rocket.json

    OPERATION_MODES = ("定点滞空", "高度", "有翼", "自由")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = AppState()
        print(f"=== AppWindow.__init__ === Created local AppWindow State: id={id(self.state)}")

        self.setWindowTitle("Kazamidori Project")
        self.resize(1600, 900)
        self.setMinimumSize(960, 640)

        # Task 1: QSplitter is the central widget — no QDockWidgets anywhere.
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._main_splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self._main_splitter)

        self._apply_theme()
        self._build_figures()
        self._build_menu_bar()
        self._build_tool_bar()
        self._build_status_bar()

        # Create the persistent Advanced Settings dialog and expose its widgets
        # at window level so SimController._collect_params() can read them without
        # knowing where they live in the widget hierarchy.
        self._adv_dialog = AdvancedSettingsDialog(self)
        self.surf_spd_input     = self._adv_dialog.surf_spd_input
        self.surf_dir_input     = self._adv_dialog.surf_dir_input
        self.up_spd_input       = self._adv_dialog.up_spd_input
        self.up_dir_input       = self._adv_dialog.up_dir_input
        self.cep_prob_input     = self._adv_dialog.cep_prob_input
        self.mc_runs_input      = self._adv_dialog.mc_runs_input
        self.wind_unc_input     = self._adv_dialog.wind_unc_input
        self.thrust_unc_input   = self._adv_dialog.thrust_unc_input
        # Aliases used by _bind_state for the local reactive AppState
        self.wind_speed_input   = self._adv_dialog.surf_spd_input
        self.wind_dir_input     = self._adv_dialog.surf_dir_input

        # Create the persistent ManualSetupDialog and expose its airframe spinboxes
        # at window level so SimController._wire_airframe_spinboxes() can find them
        # by attribute name without knowing they live inside a dialog.
        self._manual_dialog = ManualSetupDialog(self)
        self._manual_dialog.sig_load_json.connect(self._on_load_airframe_json)
        self._manual_dialog.sig_save_json.connect(self.sig_save_rocket_json_clicked.emit)
        self._manual_dialog.sig_reset.connect(self._on_manual_config_reset)
        self.af_mass_input      = self._manual_dialog.af_mass_input
        self.af_cg_input        = self._manual_dialog.af_cg_input
        self.af_len_input       = self._manual_dialog.af_len_input
        self.af_radius_input    = self._manual_dialog.af_radius_input
        self.af_nose_input      = self._manual_dialog.af_nose_input
        self.af_finroot_input   = self._manual_dialog.af_finroot_input
        self.af_fintip_input    = self._manual_dialog.af_fintip_input
        self.af_finspan_input   = self._manual_dialog.af_finspan_input
        self.af_finpos_input    = self._manual_dialog.af_finpos_input
        self.af_motorpos_input  = self._manual_dialog.af_motorpos_input
        self.af_motormass_input = self._manual_dialog.af_motormass_input
        # Create backfire delay input directly on the main window
        self.af_backfire_input  = QDoubleSpinBox(self)
        self.af_backfire_input.setDecimals(2); self.af_backfire_input.setSingleStep(0.1)
        self.af_backfire_input.setSuffix(" s"); self.af_backfire_input.setRange(-9999.0, 10.0)
        self.af_backfire_input.setSpecialValueText(""); self.af_backfire_input.setValue(-9999.0)
        self.af_backfire_input.clear()
        self.af_backfire_input.wheelEvent = lambda event: event.ignore()
        self._setup_splitter()

        # self.af_backfire_input.valueChanged.connect(lambda v: self._mark_modified())
        self.af_mass_input.valueChanged.connect(lambda v: self._mark_modified())
        self.af_cg_input.valueChanged.connect(lambda v: self._mark_modified())
        self.af_len_input.valueChanged.connect(lambda v: self._mark_modified())
        self.af_radius_input.valueChanged.connect(lambda v: self._mark_modified())
        self.af_nose_input.valueChanged.connect(lambda v: self._mark_modified())
        self.af_finroot_input.valueChanged.connect(lambda v: self._mark_modified())
        self.af_fintip_input.valueChanged.connect(lambda v: self._mark_modified())
        self.af_finspan_input.valueChanged.connect(lambda v: self._mark_modified())
        self.af_finpos_input.valueChanged.connect(lambda v: self._mark_modified())
        self.af_motorpos_input.valueChanged.connect(lambda v: self._mark_modified())
        self.af_motormass_input.valueChanged.connect(lambda v: self._mark_modified())

        # Phase 2 Tracker evaluation hooks
        self.af_mass_input.valueChanged.connect(self._evaluate_config_deltas)
        self.af_cg_input.valueChanged.connect(self._evaluate_config_deltas)
        self.af_len_input.valueChanged.connect(self._evaluate_config_deltas)
        self.af_radius_input.valueChanged.connect(self._evaluate_config_deltas)
        self.af_nose_input.valueChanged.connect(self._evaluate_config_deltas)
        self.af_finroot_input.valueChanged.connect(self._evaluate_config_deltas)
        self.af_fintip_input.valueChanged.connect(self._evaluate_config_deltas)
        self.af_finspan_input.valueChanged.connect(self._evaluate_config_deltas)
        self.af_finpos_input.valueChanged.connect(self._evaluate_config_deltas)
        self.af_motorpos_input.valueChanged.connect(self._evaluate_config_deltas)
        self.af_motormass_input.valueChanged.connect(self._evaluate_config_deltas)

        self._bind_state()

        # Motor data persisted here after _on_load_motor(); read by SimController._collect_params()
        self._motor_thrust_data: list | None = None
        self._motor_burn_time:   float | None = None

    # ── Theme ──────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        self.setStyleSheet(_QSS)

    # ── Figures ────────────────────────────────────────────────────────────────

    def _build_figures(self) -> None:
        self.profile_fig    = Figure(figsize=(5, 5), facecolor="#1e1e2e")
        self.profile_ax     = self.profile_fig.add_subplot(111, projection="3d")
        self.profile_ax.view_init(elev=25, azim=DEFAULT_AZIMUTH)
        self.profile_canvas = _AzimCanvas(self.profile_fig)

        self.map_view = MapView(self.state, self)

        # Dual wind panel: left = Cartesian speed profile, right = polar compass.
        self.wind_fig        = Figure(figsize=(9, 3.5), facecolor="#1e1e1e")
        self.wind_profile_ax = self.wind_fig.add_subplot(121)
        self.wind_ax         = self.wind_fig.add_subplot(122, projection="polar")
        self.wind_canvas     = _MplCanvas(self.wind_fig)

        # Overlay artist tracking — populated by update_map_plot() and
        # _render_overlays() so partial redraws can remove exactly these
        # artists without touching the base scatter or trajectory layers.
        self._overlay_artists: list = []
        # KDE ContourSets + error ellipse patch tracked separately so
        # update_ellipse_layer() can remove and redraw them without ax.cla().
        self._ellipse_layer_artists: list = []
        # Wind history buffer: populated by update_wind_history() from the
        # global AppState wind_history_updated signal.  Keyed by altitude (m);
        # each value is a list of (relative_time_s, speed_ms) pairs.
        self._wind_hist_buf: dict = {}

    # ── Menu bar ───────────────────────────────────────────────────────────────

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        fm = mb.addMenu("&File")
        fm.addAction(QAction("Load Motor File…", self, triggered=self._on_load_motor))
        fm.addAction(QAction("Export Results…",  self))
        fm.addSeparator()
        # Phase E: persist the operator-facing AppState (Cd / motor settings,
        # Cd curves) to JSON so a calibrated launch profile can be reloaded
        # in a future session without re-entering every value.
        fm.addAction(QAction("Save Session…", self, triggered=self._on_save_session))
        fm.addAction(QAction("Load Session…", self, triggered=self._on_load_session))
        fm.addSeparator()
        fm.addAction(QAction("Quit", self, triggered=self.close))

        sm = mb.addMenu("&Simulation")
        sm.addAction(QAction("▶  Run Simulation (F5)", self, triggered=self._on_run))
        sm.addAction(QAction("⏹  Stop (Esc)",           self, triggered=self._on_stop))

        self._view_menu = mb.addMenu("&View")

        # Checkbox actions for plot toggles
        self.action_show_kde = QAction("KDE Contour", self, checkable=True)
        self.action_show_kde.setChecked(self.state.show_kde)
        self.action_show_kde.toggled.connect(lambda c: setattr(self.state, "show_kde", c))
        self._view_menu.addAction(self.action_show_kde)

        self.action_show_cep = QAction("CEP 90% Ellipse", self, checkable=True)
        self.action_show_cep.setChecked(self.state.show_cep)
        self.action_show_cep.toggled.connect(lambda c: setattr(self.state, "show_cep", c))
        self._view_menu.addAction(self.action_show_cep)

        self.action_show_scatter = QAction("Monte Carlo Scatter", self, checkable=True)
        self.action_show_scatter.setChecked(self.state.show_scatter)
        self.action_show_scatter.toggled.connect(lambda c: setattr(self.state, "show_scatter", c))
        self._view_menu.addAction(self.action_show_scatter)

        self.action_show_burnout = QAction("Motor Burnout Point", self, checkable=True)
        self.action_show_burnout.setChecked(self.state.show_burnout)
        self.action_show_burnout.toggled.connect(lambda c: setattr(self.state, "show_burnout", c))
        self._view_menu.addAction(self.action_show_burnout)

        self.action_show_apogee = QAction("Apogee", self, checkable=True)
        self.action_show_apogee.setChecked(self.state.show_apogee)
        self.action_show_apogee.toggled.connect(lambda c: setattr(self.state, "show_apogee", c))
        self._view_menu.addAction(self.action_show_apogee)

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

        btn_run  = QPushButton("▶  Run",  tb); btn_run.setObjectName("btn_run")
        btn_stop = QPushButton("⏹  Stop", tb); btn_stop.setObjectName("btn_stop")

        btn_run.setFixedWidth(90);  btn_run.clicked.connect(self._on_run)
        btn_run.setToolTip("Run Simulation (F5)")
        btn_run.setShortcut("F5")
        btn_stop.setFixedWidth(74); btn_stop.clicked.connect(self._on_stop)
        btn_stop.setToolTip("Stop Simulation (Esc)")
        btn_stop.setShortcut("Esc")

        tb.addWidget(btn_run)
        _vline()
        tb.addWidget(btn_stop)
        _vline()

        self.btn_download_map = QPushButton("Download Map", tb)
        self.btn_download_map.setObjectName("btn_download_map")
        self.btn_download_map.setToolTip("Download offline map tiles for current location")
        tb.addWidget(self.btn_download_map)

        spacer = QWidget(tb)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        self._progress = QProgressBar(tb)
        self._progress.setFixedWidth(140)
        self._progress.setValue(0)
        self._progress.setFormat("%p%")
        self._progress.setTextVisible(True)
        tb.addWidget(self._progress)

        self._phase_label = QLabel("Idle", tb)
        self._phase_label.setFixedWidth(120)
        self._phase_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._phase_label.setStyleSheet(
            "color: #a6adc8; font-size: 8pt; background: transparent; padding: 0 4px;")
        tb.addWidget(self._phase_label)

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
        self._wind_status.setStyleSheet("color: #7eb3ff;")

        sb.addWidget(self._status_label, stretch=1)
        sb.addPermanentWidget(self._wind_status)

    # ── Nested splitter layout ─────────────────────────────────────────────────
    #
    # Build order: map panel FIRST so self.map_widget exists before
    # _build_parameters_panel() wires the lat/lon lambda closures.
    #
    # Structure:
    #   _main_splitter (H)
    #   ├── params_panel          (left, full height)
    #   └── _right_splitter (V)
    #       ├── _top_splitter (H)
    #       │   ├── profile_panel   (3-D trajectory)
    #       │   └── map_panel       (2-D landing map)
    #       └── wind_panel          (wind history + compass + table, full width)

    def _setup_splitter(self) -> None:
        self._map_panel     = self._build_map_dock_widget()
        self._params_panel  = self._build_parameters_panel()
        self._profile_panel = self._build_profile_dock_widget()
        self._wind_panel    = self._build_wind_panel()

        self._top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._top_splitter.setChildrenCollapsible(False)
        self._top_splitter.setHandleWidth(3)
        self._top_splitter.addWidget(self._profile_panel)
        self._top_splitter.addWidget(self._map_panel)

        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setChildrenCollapsible(False)
        self._right_splitter.setHandleWidth(3)
        self._right_splitter.addWidget(self._top_splitter)
        self._right_splitter.addWidget(self._wind_panel)

        self._main_splitter.addWidget(self._params_panel)
        self._main_splitter.addWidget(self._right_splitter)

    # ── Column sizing (deferred to first paint) ───────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._apply_initial_sizes)

    def _apply_initial_sizes(self) -> None:
        # Main: params 300 | right rest
        self._main_splitter.setSizes([300, 1300])
        # Right vertical: top (3D+map) 60% | wind 40%
        self._right_splitter.setSizes([560, 340])
        # Top horizontal: profile | map equal
        self._top_splitter.setSizes([650, 650])

    # ── Profile dock content (3-D trajectory + wind) ──────────────────────────

    def _build_profile_dock_widget(self) -> QWidget:
        """3-D trajectory panel (azimuth slider, no wind section — that is in wind_panel)."""
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)


        lay.addWidget(self.profile_canvas)
        lay.setStretchFactor(self.profile_canvas, 1)

        # ── Azimuth control row ───────────────────────────────────────────────
        azim_row = QWidget(container)
        azim_row.setFixedHeight(26)
        azim_lay = QHBoxLayout(azim_row)
        azim_lay.setContentsMargins(8, 0, 8, 0)
        azim_lay.setSpacing(6)

        azim_lbl = QLabel("Azimuth:", azim_row)
        azim_lbl.setStyleSheet("color: #6c7086; font-size: 7pt;")
        azim_lbl.setFixedWidth(48)

        self._azim_slider = QSlider(Qt.Orientation.Horizontal, azim_row)
        self._azim_slider.setMinimum(-180)
        self._azim_slider.setMaximum(180)
        self._azim_slider.setValue(DEFAULT_AZIMUTH)
        self._azim_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self._azim_slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 4px; background: #3c3c3c;"
            "  border-radius: 2px; }"
            "QSlider::handle:horizontal  { width: 12px; height: 12px;"
            "  background: #7eb3ff; border-radius: 6px; margin: -4px 0; }"
            "QSlider::sub-page:horizontal { background: #7eb3ff; border-radius: 2px; }")

        self._azim_val_lbl = QLabel(f"{DEFAULT_AZIMUTH}°", azim_row)
        self._azim_val_lbl.setStyleSheet("color: #a6adc8; font-size: 7pt;")
        self._azim_val_lbl.setFixedWidth(28)

        self._azim_slider.valueChanged.connect(self._on_azim_changed)
        self._azim_slider.valueChanged.connect(
            lambda v: self._azim_val_lbl.setText(f"{v}°"))

        azim_lay.addWidget(azim_lbl)
        azim_lay.addWidget(self._azim_slider)
        azim_lay.addWidget(self._azim_val_lbl)
        lay.addWidget(azim_row)

        self.profile_canvas._azim_slider = self._azim_slider
        return container

    # ── Parameters panel ──────────────────────────────────────────────────────
    #
    # Layout (top → bottom):
    #   QToolBox  [Airframe | Launch Settings]  ← expands to fill
    #   Advanced Settings button
    #   Launch Mode group box                   ← pinned, fixed height
    #   GO / NO-GO indicator
    #   RUN PHASE 1 button
    #
    # QSizePolicy.Maximum prevents the panel from growing taller than its
    # natural content size when the window is maximised, eliminating the
    # "mystery gap" caused by Qt distributing leftover vertical space to
    # the parameters container instead of the adjacent graph dock.

    def _build_parameters_panel(self) -> QWidget:
        container = QWidget()
        # Task 2: Expanding in both axes so the panel fills its splitter column.
        container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 6)
        lay.setSpacing(4)

        # ── Two-tab toolbox ───────────────────────────────────────────────────
        tb = QToolBox(container)
        tb.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        tb.addItem(self._build_airframe_page(),        "🚀  Airframe")
        tb.addItem(self._build_launch_settings_page(), "📍  Launch Settings")
        # QSS min-height on QToolBox::tab is ignored at widget-level; set height
        # directly on the internal QAbstractButton children instead.
        for _tab_btn in tb.findChildren(QAbstractButton):
            _tab_btn.setMinimumHeight(44)
        lay.addWidget(tb, stretch=1)

        # ── Advanced Settings button ──────────────────────────────────────────
        btn_adv = QPushButton("⚙  Advanced Settings…", container)
        btn_adv.setObjectName("btn_adv_settings")
        btn_adv.setToolTip("Configure Monte Carlo parameters and simulation limits")
        btn_adv.clicked.connect(self._on_advanced_settings)
        lay.addWidget(btn_adv)

        # ── Launch Mode (pinned above Run button) ─────────────────────────────
        mode_grp = QGroupBox("Launch Mode", container)
        mode_grp.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        mode_lay     = QFormLayout(mode_grp)
        mode_lay.setSpacing(5)
        mode_lay.setContentsMargins(2, 4, 2, 2)

        self.mode_combo = QComboBox(mode_grp)
        self.mode_combo.addItems(self.OPERATION_MODES)
        self.mode_combo.setCurrentText("自由")

        self.rmax_input  = QDoubleSpinBox(mode_grp)
        self.rmax_input.setRange(0, 9999); self.rmax_input.setDecimals(1)
        self.rmax_input.setValue(50.0);    self.rmax_input.setSuffix(" m")
        self.rmax_input.setToolTip(
            "Target landing radius for GO/NO-GO assessment and map display")
        self.rmax_input.wheelEvent = lambda event: event.ignore()

        mode_lay.addRow("飛行モード:",    self.mode_combo)
        mode_lay.addRow("Target Radius:", self.rmax_input)

        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._on_mode_changed("自由")

        lay.addWidget(mode_grp)

        # ── GO / NO-GO indicator ──────────────────────────────────────────────
        self._go_nogo_label = QLabel("⬤  STANDBY", container)
        self._go_nogo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._go_nogo_label.setStyleSheet(
            "font-size: 14pt; font-weight: bold; color: #7a7e9a; padding: 8px;"
            "background: #1a1a2e; border-radius: 8px; border: 2px solid #3a3a52;")
        lay.addWidget(self._go_nogo_label)


        # ── Results Panel (Hidden by Default) ─────────────────────────────────
        self._results_grp = QGroupBox("Optimization Results", container)
        self._results_grp.setVisible(False)
        self._results_grp.setStyleSheet("QGroupBox { border: 1px solid #7eb3ff; margin-top: 1ex; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #7eb3ff; }")

        res_lay = QFormLayout(self._results_grp)
        res_lay.setSpacing(4)
        res_lay.setContentsMargins(10, 10, 10, 8)

        _res_tag = "QLabel { font-weight: bold; color: #a6e3a1; font-family: 'Consolas', monospace; }"
        self.lbl_res_angle = QLabel("—")
        self.lbl_res_best  = QLabel("—")
        self.lbl_res_avg   = QLabel("—")
        self.lbl_res_min   = QLabel("—")
        self.lbl_res_alt   = QLabel("—")
        self.lbl_res_hang  = QLabel("—")

        for _lbl in (self.lbl_res_angle, self.lbl_res_best, self.lbl_res_avg, self.lbl_res_min, self.lbl_res_alt, self.lbl_res_hang):
            _lbl.setStyleSheet(_res_tag)

        res_lay.addRow("Optimal Angle:", self.lbl_res_angle)
        res_lay.addRow("Best Score:", self.lbl_res_best)
        res_lay.addRow("MC Avg Score:", self.lbl_res_avg)
        res_lay.addRow("MC Min Score:", self.lbl_res_min)
        res_lay.addRow("MC Avg Alt:", self.lbl_res_alt)
        res_lay.addRow("MC Avg Hang:", self.lbl_res_hang)

        lay.addWidget(self._results_grp)

        # ── Run button ────────────────────────────────────────────────────────
        btn_run = QPushButton("🚀   RUN PHASE 1 SIMULATION", container)
        btn_run.setObjectName("btn_phase1_run")
        btn_run.setMinimumHeight(48)
        btn_run.clicked.connect(self._on_phase1)
        btn_run.setToolTip("Run Phase 1 Optimization (F6)")
        btn_run.setShortcut("F6")
        lay.addWidget(btn_run)

        return container

    # ── Airframe tab ──────────────────────────────────────────────────────────
    # Contains: Load-JSON button, motor load + specs, 12 CGS airframe params.
    # Units: CGMS — lengths in cm from nose tip, mass in g, delay in s.

    def _mark_modified(self) -> None:
        if 'loaded' not in self.rkt_label.text() and '(Modified)' not in self.rkt_label.text():
            self.rkt_label.setText(f"{self.rkt_label.text()} (Modified)")
            self.rkt_label.setStyleSheet("color: #f38ba8; font-style: italic; font-size: 8pt; padding: 2px 4px;")

    def _build_airframe_page(self) -> QScrollArea:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(6)

        # ── Model loading buttons ─────────────────────────────────────────────
        btn_rkt = QPushButton("📂  Load .rkt File", w)
        btn_rkt.setToolTip("Load an OpenRocket .rkt file")
        btn_rkt.clicked.connect(self.sig_load_rkt_clicked.emit)

        self.rkt_label = QLabel("(no .rkt loaded)", w)
        self.rkt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rkt_label.setStyleSheet(
            "color: #f9a86b; font-style: italic; font-size: 8pt; padding: 2px 4px;")
        self.rkt_label.setWordWrap(True)

        btn_manual = QPushButton("⚙  Manual Config…", w)
        btn_manual.setToolTip("Manually enter all rocket geometry parameters")
        btn_manual.clicked.connect(self._on_manual_config)

        btn_motor = QPushButton("📂  Load Thrust Curve (.csv)", w)
        btn_motor.setToolTip("Load a custom motor thrust curve from a CSV file")
        btn_motor.clicked.connect(self._on_load_motor)

        self.motor_label = QLabel("(no motor loaded)", w)
        self.motor_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.motor_label.setStyleSheet(
            "color: #f9a86b; font-style: italic; font-size: 8pt; padding: 2px 4px;")
        self.motor_label.setWordWrap(True)

        # ── Motor specification summary (read-only) ───────────────────────────
        grp_motor     = QGroupBox("Motor Specifications", w)
        grp_motor_lay = QFormLayout(grp_motor)
        grp_motor_lay.setSpacing(5)
        grp_motor_lay.setContentsMargins(2, 4, 2, 2)

        _tag = (
            "QLabel { color: #eef0f8; background: #12121e; font-weight: bold; "
            "font-family: 'Consolas', monospace; padding: 3px 8px; "
            "border-radius: 4px; border: 1px solid #3a3a52; }"
        )
        self.lbl_avg_thrust    = QLabel("—", grp_motor)
        self.lbl_max_thrust    = QLabel("—", grp_motor)
        self.lbl_burn_time     = QLabel("—", grp_motor)
        self.lbl_total_impulse = QLabel("—", grp_motor)
        for _lbl in (self.lbl_avg_thrust, self.lbl_max_thrust,
                     self.lbl_burn_time,  self.lbl_total_impulse):
            _lbl.setStyleSheet(_tag)
        grp_motor_lay.addRow("Avg Thrust:",    self.lbl_avg_thrust)
        grp_motor_lay.addRow("Max Thrust:",    self.lbl_max_thrust)
        grp_motor_lay.addRow("Burn Time:",     self.lbl_burn_time)
        grp_motor_lay.addRow("Total Impulse:", self.lbl_total_impulse)
        grp_motor_lay.addRow("Backfire Delay [s]:", self.af_backfire_input)

        # ── Recovery / Parachute parameters ──────────────────────────────────
        grp_para     = QGroupBox("Recovery / Parachute", w)
        frm_para     = QFormLayout(grp_para)
        frm_para.setSpacing(5)
        frm_para.setContentsMargins(2, 4, 2, 2)

        def _psb(hi, dec, step, suffix):
            sb = QDoubleSpinBox(grp_para)
            sb.setDecimals(dec); sb.setSingleStep(step); sb.setSuffix(suffix)
            sb.setRange(-9999.0, hi)
            sb.setSpecialValueText("")
            sb.setValue(-9999.0)
            sb.wheelEvent = lambda event: event.ignore()
            return sb

        self.para_cd_input   = _psb(2.00,  2, 0.01,  "")
        self.para_area_input = _psb(10.0,  4, 0.001, " m²")
        self.para_lag_input  = _psb(30.0,  2, 0.1,   " s")

        frm_para.addRow("Drag Coeff. Cd:",     self.para_cd_input)
        frm_para.addRow("Canopy Area [m²]:",   self.para_area_input)
        frm_para.addRow("Deployment Lag [s]:", self.para_lag_input)

        btn_para_json = QPushButton("📂  Load Parachute JSON", w)
        btn_para_json.setToolTip("Load a parachute-only JSON config file")
        btn_para_json.clicked.connect(self.sig_load_para_json_clicked.emit)
        frm_para.addRow(btn_para_json)

        lay.addWidget(btn_rkt)
        lay.addWidget(self.rkt_label)
        lay.addWidget(btn_manual)
        lay.addWidget(btn_motor)
        lay.addWidget(self.motor_label)
        lay.addWidget(grp_motor)
        lay.addWidget(grp_para)

        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        sa.setWidget(w)
        return sa

    # ── Launch Settings tab ───────────────────────────────────────────────────

    def _build_launch_settings_page(self) -> QWidget:
        w   = QWidget()
        frm = QFormLayout(w)
        frm.setSpacing(6)
        frm.setContentsMargins(8, 8, 8, 8)

        self.lat_input = QDoubleSpinBox(w)
        self.lat_input.setDecimals(6); self.lat_input.setSuffix("°")
        self.lat_input.setRange(-9999.0, 90)
        self.lat_input.setSpecialValueText("")
        self.lat_input.setValue(35.42215789)
        self.lat_input.wheelEvent = lambda event: event.ignore()
        self.lat_input.valueChanged.connect(
            lambda v: self.map_widget.update_launch(v, self.lon_input.value())
            if v != -9999.0 else None)

        self.lon_input = QDoubleSpinBox(w)
        self.lon_input.setDecimals(6); self.lon_input.setSuffix("°")
        self.lon_input.setRange(-9999.0, 180)
        self.lon_input.setSpecialValueText("")
        self.lon_input.setValue(139.42268826)
        self.lon_input.wheelEvent = lambda event: event.ignore()
        self.lon_input.valueChanged.connect(
            lambda v: self.map_widget.update_launch(self.lat_input.value(), v)
            if v != -9999.0 else None)

        self.elev_input = QDoubleSpinBox(w)
        self.elev_input.setRange(0.0, 90.0)
        self.elev_input.setDecimals(1); self.elev_input.setSuffix("°")
        self.elev_input.setValue(85.0)
        self.elev_input.wheelEvent = lambda event: event.ignore()

        self.rail_len_input = QDoubleSpinBox(w)
        self.rail_len_input.setRange(0.1, 20.0)
        self.rail_len_input.setDecimals(2); self.rail_len_input.setSuffix(" m")
        self.rail_len_input.setSingleStep(0.1)
        self.rail_len_input.setValue(1.0)
        self.rail_len_input.wheelEvent = lambda event: event.ignore()

        self.azim_input = QDoubleSpinBox(w)
        self.azim_input.setDecimals(1); self.azim_input.setSuffix("°")
        self.azim_input.setRange(-9999.0, 360)
        self.azim_input.setSpecialValueText("")
        self.azim_input.setValue(0.0)
        self.azim_input.wheelEvent = lambda event: event.ignore()
        self.azim_input.setWrapping(True)

        btn_dl_map = QPushButton("🗺️  Download Offline Map", w)
        btn_dl_map.setObjectName("btn_download_map")
        btn_dl_map.setToolTip("Download OSM tiles for the current coordinates to use offline")

        btn_gps = QPushButton("📍  Get Current Location", w)
        btn_gps.setToolTip("Attempt to fetch launch coordinates using IP-based geolocation")
        btn_gps.clicked.connect(self._on_get_location)

        frm.addRow("Latitude:",         self.lat_input)
        frm.addRow("Longitude:",        self.lon_input)
        frm.addRow("",                  btn_gps)
        frm.addRow(QLabel(""))
        frm.addRow("Rail Elevation:",   self.elev_input)
        frm.addRow("Rail Length [m]:",  self.rail_len_input)
        frm.addRow("Rail Azimuth:",     self.azim_input)
        return w

    # ── Map dock content ───────────────────────────────────────────────────────

    def _build_map_dock_widget(self) -> QWidget:
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        info = QFrame(container)
        info.setObjectName("MapInfoBar")
        info.setFixedHeight(32)
        info.setStyleSheet(
            "QFrame#MapInfoBar {"
            "  background: #12121e; border-bottom: 1px solid #2a2a3e;"
            "}")
        ilay = QHBoxLayout(info)
        ilay.setContentsMargins(14, 0, 14, 0)
        ilay.setSpacing(6)

        self._map_launch_lbl = QLabel("Launch:  35.682800°N, 139.759000°E", info)
        self._map_launch_lbl.setStyleSheet(
            "color: #7eb3ff; font-size: 9pt; background: transparent;")

        _sep = QLabel("|", info)
        _sep.setStyleSheet("color: #3a3a52; background: transparent;")

        self._map_landing_lbl = QLabel("Landing:  —", info)
        self._map_landing_lbl.setStyleSheet(
            "color: #f38ba8; font-size: 9pt; background: transparent;")

        ilay.addWidget(self._map_launch_lbl)
        ilay.addStretch()
        ilay.addWidget(_sep)
        ilay.addStretch()
        ilay.addWidget(self._map_landing_lbl)

        lay.addWidget(info)
        lay.addWidget(self.map_view)
        lay.setStretchFactor(self.map_view, 1)

        self.map_widget = _MapCoordProxy(self._map_launch_lbl, self._map_landing_lbl)
        return container

    # ── Wind panel (history graph + compass + current-values table) ──────────

    def _build_wind_panel(self) -> QWidget:
        """Bottom row: wind speed history  ·  polar compass  ·  current-values table."""
        container = QWidget()
        lay = QHBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        # ── Left: matplotlib canvas (2 subplots) ──────────────────────────────
        canvas_wrap = QWidget(container)
        cwl = QVBoxLayout(canvas_wrap)
        cwl.setContentsMargins(0, 0, 0, 0)
        cwl.setSpacing(0)
        hdr = QLabel(
            "  Wind  ·  Speed History  &  Compass  ·  5 Altitude Nodes",
            canvas_wrap)
        hdr.setStyleSheet("color: #6c7086; font-size: 7pt; padding: 1px 4px;")

        cwl.addWidget(self.wind_canvas, stretch=1)
        lay.addWidget(canvas_wrap, stretch=1)

        # ── Right column: wind table on top, status labels directly below ────
        # Previously the table and the two status labels lived as three
        # separate slots inside the outer QHBoxLayout, which pushed the
        # Koinobori / GPV labels to the far-right of the panel and produced
        # visible horizontal dead space.  Wrapping them into a dedicated
        # QVBoxLayout keeps the compass canvas at stretch=1 (so it absorbs
        # the reclaimed width) and stacks the labels neatly under the table.
        right_col = QWidget(container)
        right_col.setSizePolicy(
            QSizePolicy.Policy.Maximum,    # do not let the column eat canvas width
            QSizePolicy.Policy.Preferred,
        )
        right_lay = QVBoxLayout(right_col)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)

        self._wind_table = QTableWidget(5, 3, right_col)
        self._wind_table.setObjectName("WindTable")
        self._wind_table.setHorizontalHeaderLabels(["Alt", "Speed (m/s)", "Dir (°)"])
        self._wind_table.verticalHeader().setVisible(False)
        self._wind_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._wind_table.setSelectionMode(
            QTableWidget.SelectionMode.NoSelection)
        hh = self._wind_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._wind_table.setMaximumWidth(260)
        self._wind_table.setMinimumWidth(160)
        self._wind_table.setAlternatingRowColors(True)
        # Pre-populate with dashes; cells are reused (never re-created) for speed
        for r in range(5):
            for c in range(3):
                item = QTableWidgetItem("—")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._wind_table.setItem(r, c, item)

        right_lay.addWidget(self._wind_table)

        # ── System Status Labels (stacked under the table) ───────────────────
        # Koinobori on the left edge, GPV timestamp on the right edge of the
        # same row.  ``addStretch`` keeps them pinned to opposite sides so
        # the row stays compact when the column is narrow and the labels
        # remain readable when the column widens.  Signal/slot bindings to
        # AppState.koinobori_status_changed and gpv_last_fetch_time_changed
        # are wired elsewhere (search for ``lbl_koinobori_status`` /
        # ``lbl_gpv_status``); only the layout placement is touched here.
        status_lay = QHBoxLayout()
        status_lay.setContentsMargins(4, 0, 4, 0)
        status_lay.setSpacing(8)

        self.lbl_koinobori_status = QLabel("Koinobori: Disconnected", right_col)
        self.lbl_koinobori_status.setStyleSheet("color: #a6adc8; font-size: 8pt;")
        self.lbl_koinobori_status.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.lbl_gpv_status = QLabel("GPV Updated: N/A", right_col)
        self.lbl_gpv_status.setStyleSheet("color: #a6adc8; font-size: 8pt;")
        self.lbl_gpv_status.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        status_lay.addWidget(self.lbl_koinobori_status)
        status_lay.addStretch()
        status_lay.addWidget(self.lbl_gpv_status)

        right_lay.addLayout(status_lay)

        telemetry_lay = QHBoxLayout()
        telemetry_lay.setContentsMargins(4, 4, 4, 0)
        self.lbl_max_gust = QLabel("Max Gust: —", right_col)
        self.lbl_mean_wind = QLabel("Mean Wind: —", right_col)
        self.lbl_std_dev = QLabel("Std Dev: —", right_col)
        for _lbl in (self.lbl_max_gust, self.lbl_mean_wind, self.lbl_std_dev):
            _lbl.setStyleSheet("color: #cdd6f4; font-size: 8pt;")
            _lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            telemetry_lay.addWidget(_lbl)

        right_lay.addLayout(telemetry_lay)
        right_lay.addStretch()   # keep table + status pinned to the top

        lay.addWidget(right_col)

        return container

    # ── Reactive binding ───────────────────────────────────────────────────────

    def _bind_state(self) -> None:
        s = self.state
        self.wind_speed_input.valueChanged.connect(lambda v: setattr(s, "wind_speed", v))
        self.wind_dir_input.valueChanged.connect(  lambda v: setattr(s, "wind_dir",   v))
        self.cep_prob_input.valueChanged.connect(  lambda v: setattr(s, "cep_prob",   v))
        self.cep_prob_input.valueChanged.connect(lambda v: self.refresh_visuals())
        self.mode_combo.currentTextChanged.connect(
            lambda v: setattr(s, "sim_mode", v))

        self.lat_input.valueChanged.connect(lambda v: setattr(s, 'launch_lat', v))
        self.lon_input.valueChanged.connect(lambda v: setattr(s, 'launch_lon', v))

        s.launch_lat = self.lat_input.value()
        s.launch_lon = self.lon_input.value()

        s.needs_redraw.connect(self.update_profile_plot)
        s.needs_redraw.connect(self.update_map_plot)

        s.needs_redraw.connect(self.update_wind_plot)

        self.update_profile_plot()
        self.update_map_plot()
        self.update_wind_plot()

    # ══ Plot: 3-D Flight Profile ══════════════════════════════════════════════
    #
    # ROADMAP - 3D Flight Profile Integration:
    # Schedule the following adjustments for the 3D Plot:
    # 1. Re-orient the 3D axis so that North (the Y axis) points directly into
    #    the screen (the depth/background axis).
    # 2. Recalibrate the camera rotation limits strictly between -90° and 90°
    #    (representing a sweep from true West to true East).

    def update_profile_plot(self) -> None:
        ax = self.profile_ax
        ax.cla()
        _style_3d(ax, self.profile_fig)

        s   = self.state
        res = s.simulation_result

        if res is not None:
            self._draw_real_result(ax, res)
        else:
            self._draw_empty_profile(ax)

        ax.set_xlabel("East  (m)",  color="#6c7086", fontsize=8, labelpad=4)
        ax.set_ylabel("North  (m)", color="#6c7086", fontsize=8, labelpad=4)
        ax.set_zlabel("Alt  (m)",   color="#6c7086", fontsize=8, labelpad=4)
        azim = getattr(self, '_azim_slider', None)
        ax.view_init(elev=25, azim=azim.value() if azim is not None else DEFAULT_AZIMUTH)
        if res is not None:
            _equalise_3d_axes(ax)

        # Draw Compass Rose in 3D
        cx, cy, cz = 0, 0, 0
        span = max(10, ax.get_xlim3d()[1] * 0.1)
        ax.quiver(cx, cy, cz, 0, span, 0, color="#a6e3a1", arrow_length_ratio=0.1, alpha=0.8) # North
        ax.quiver(cx, cy, cz, span, 0, 0, color="#f38ba8", arrow_length_ratio=0.1, alpha=0.8) # East
        ax.text(cx, cy + span*1.1, cz, "N", color="#a6e3a1", fontsize=8, fontweight="bold", ha="center")
        ax.text(cx + span*1.1, cy, cz, "E", color="#f38ba8", fontsize=8, fontweight="bold", ha="center")

        self.profile_fig.tight_layout(pad=0.5)
        self.profile_canvas.draw_idle()

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
        ax.text2D(0.5, 0.40, "Configure parameters and click\n'Run' (F5) to begin simulation.",
                  transform=ax.transAxes, ha="center", va="center",
                  color="#45475a", fontsize=10, linespacing=1.8)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.05), fontsize=7,
                  facecolor="#1e1e2e", edgecolor="#45475a",
                  labelcolor="#cdd6f4", framealpha=0.85, markerscale=1.5)
        mode_str = getattr(self.state, "flight_mode", "Free")
        ax.set_title(f"3D Flight Profile  |  Mode: {mode_str}", color="#a6adc8", fontsize=9, pad=6)

    def _draw_real_result(self, ax, res: dict) -> None:
        s = self.state
        tx = np.asarray(res.get("trajectory_x", [0.0]), dtype=float)
        ty = np.asarray(res.get("trajectory_y", [0.0]), dtype=float)
        tz = np.clip(np.asarray(res.get("trajectory_z", [0.0]), dtype=float), 0.0, None)
        land_x   = float(res.get("land_x", tx[-1] if len(tx) else 0.0))
        land_y   = float(res.get("land_y", ty[-1] if len(ty) else 0.0))

        apex_z = float(res.get("apogee_m",
                               float(tz.max()) if len(tz) > 0 else 0.0))
        phases = res.get("phases")
        events = res.get("events")

        # ── KDE density projected flat onto the ground plane (z = 0) ─────────
        # contourf with zdir='z', offset=0 paints the density heatmap on the
        # floor before any trajectory lines are drawn — lowest visual layer.
        # Levels start at 0.05 (5 % of peak) so the near-zero padding area
        # outside the scatter cloud is never filled — eliminates the purple square.
        kde_grid = res.get("kde")
        if kde_grid:
            try:
                _X = np.asarray(kde_grid["X_m"], dtype=float)
                _Y = np.asarray(kde_grid["Y_m"], dtype=float)
                _Z = np.asarray(kde_grid["Z"],   dtype=float)
                _lev3d = np.linspace(0.05, 1.0, 9)
                ax.contourf(_X, _Y, _Z, zdir='z', offset=0.0,
                            levels=_lev3d, cmap="plasma", alpha=0.50, zorder=1)
            except Exception as e:
                print(f"Drawing Error (KDE 3D projection): {e}")

        # Ground-track projection (always shown)
        ax.plot(tx, ty, np.zeros_like(tz),
                color="#45475a", lw=0.8, linestyle=":", alpha=0.35)

        # Send raw arrays to canvas for Shift+Scroll animation
        if "t_hist" in res:
            self.profile_canvas.set_trajectory(tx, ty, tz, res["t_hist"], ax)

        # Draw Solid Error Ellipse on Z=0
        ellipse = res.get("ellipse")
        if ellipse:
            try:
                # Plot outline of ellipse
                theta = np.linspace(0, 2*np.pi, 100)
                a = ellipse["width"] / 2
                b = ellipse["height"] / 2
                cx = ellipse["cx"]
                cy = ellipse["cy"]
                angle = ellipse["angle_rad"]

                # Parametric ellipse equations
                x_ell = cx + a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle)
                y_ell = cy + a * np.cos(theta) * np.sin(angle) + b * np.sin(theta) * np.cos(angle)
                z_ell = np.zeros_like(x_ell)

                _cep_pct = getattr(self.state, "landing_prob",
                                   getattr(self.state, "cep_prob", 90))
                ax.plot(x_ell, y_ell, z_ell, color="#f9e2af", lw=2.2, linestyle="-", alpha=0.9, zorder=6, label=f"CEP {_cep_pct}%")
            except Exception as e:
                print(f"Drawing Error (3D Ellipse): {e}")


        if phases:
            # ── Phase-coloured trajectory: Thrust / Coast / Parachute ─────────
            _PH = [
                ("thrust", "#f38ba8", "Thrust  (推進)"),
                ("coast",  "#89b4fa", "Coast  (滑空)"),
                ("chute",  "#a6e3a1", "Parachute  (降下)"),
            ]
            for ph_key, ph_col, ph_lbl in _PH:
                ph = phases.get(ph_key, {})
                px = np.asarray(ph.get("x", []), dtype=float)
                py = np.asarray(ph.get("y", []), dtype=float)
                pz = np.clip(np.asarray(ph.get("z", []), dtype=float), 0.0, None)
                if len(px) > 1:
                    ax.plot(px, py, pz, color=ph_col, lw=2.2, alpha=0.90,
                            label=ph_lbl)

            if events:
                # ── Key-event markers: Burnout / Apogee / Chute Deploy ────────
                _EV = [
                    ("burnout", "#fab387", "X", 90,  "Burnout"),
                    ("apogee",  "#f9e2af", "*", 120, f"Apogee  {apex_z:.0f} m"),
                    ("chute",   "#a6e3a1", "^", 80,  "Chute Deploy"),
                ]
                for ev_key, ev_col, ev_mk, ev_sz, ev_lbl in _EV:
                    ev = events.get(ev_key)
                    if ev is None:
                        continue
                    ex = float(ev[0]); ey = float(ev[1])
                    ez = max(0.0, float(ev[2]))
                    ax.scatter([ex], [ey], [ez], c=ev_col, s=ev_sz,
                               marker=ev_mk, zorder=9, label=ev_lbl)
                    ax.text(ex, ey, ez * 1.04 + 1.5,
                            f"  {ez:.0f} m", color=ev_col, fontsize=6)
        else:
            # ── Fallback: altitude-gradient single line ───────────────────────
            if len(tx) > 1:
                ax.add_collection3d(_make_altitude_lc(tx, ty, tz))
            ax.plot([], [], [], color="#89b4fa", lw=2.0,
                    label="Trajectory  (cool = alt)")
            apex_i = int(np.argmax(tz))
            ax.scatter([tx[apex_i]], [ty[apex_i]], [apex_z],
                       c="#f9e2af", s=90, marker="*", zorder=6,
                       label=f"Apogee  {apex_z:.0f} m")
            ax.text(tx[apex_i], ty[apex_i], apex_z * 1.04,
                    f"  {apex_z:.0f} m", color="#f9e2af", fontsize=7)

        # ── Wind quivers (shown in both phase and fallback modes) ─────────────
        # ``wind_profile`` only exists on the local AppWindow AppState; after
        # ``bind_app_state`` overwrites ``self.state`` with the global AppState
        # (which has no such attribute) we fall back to an empty list so the
        # quiver block stays a no-op rather than raising.
        profile = getattr(s, "wind_profile", [])
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

        ax.scatter([land_x], [land_y], [0.0], c="#f38ba8", s=130,
                   marker="v", zorder=7, label="Nominal landing")
        ax.scatter([0.0], [0.0], [0.0], c="#a6e3a1", s=130,
                   marker="^", zorder=8, label="Launch  (0, 0, 0)")

        # MC landing scatter at Z=0 — single scatter3D call, max 500 pts
        if "mc_scatter_x" in res and res["mc_scatter_x"]:
            try:
                _sc_x = np.asarray(res["mc_scatter_x"][:MAX_SCATTER_POINTS], dtype=float)
                _sc_y = np.asarray(res["mc_scatter_y"][:MAX_SCATTER_POINTS], dtype=float)
                ax.scatter(_sc_x, _sc_y, np.zeros(len(_sc_x)),
                           c="#f38ba8", s=3, alpha=0.40, zorder=3,
                           linewidths=0, label=f"MC impacts  ({len(res['mc_scatter_x'])})")
            except Exception as e:
                print(f"Drawing Error (3D MC scatter): {e}")

        h_dist = float(np.hypot(land_x, land_y))
        ax.text2D(0.98, 0.98,
                  f"Apogee:  {apex_z:.0f} m\nH-dist:  {h_dist:.0f} m",
                  transform=ax.transAxes, ha="right", va="top",
                  color="#cdd6f4", fontsize=7.5,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#313244",
                            edgecolor="#45475a", alpha=0.88))
        ax.legend(loc="upper left", fontsize=7,
                  facecolor="#1e1e2e", edgecolor="#45475a",
                  labelcolor="#cdd6f4", framealpha=0.88, borderpad=0.6)

        tx_min, tx_max = float(tx.min()), float(tx.max())
        ty_min, ty_max = float(ty.min()), float(ty.max())
        tz_min, tz_max = float(tz.min()), float(tz.max())

        pad_x = max((tx_max - tx_min) * 0.15, 10.0)
        pad_y = max((ty_max - ty_min) * 0.15, 10.0)
        pad_z = max((tz_max - tz_min) * 0.15, 10.0)

        ax.set_xlim3d(tx_min - pad_x, tx_max + pad_x)
        ax.set_ylim3d(ty_min - pad_y, ty_max + pad_y)
        ax.set_zlim3d(max(0.0, tz_min), max(tz_max + pad_z, 10.0))
        _mode = getattr(s, "operation_mode", getattr(s, "sim_mode", ""))
        _spd  = getattr(s, "surf_wind_speed", getattr(s, "wind_speed", 0.0))
        _dir  = getattr(s, "surf_wind_dir",   getattr(s, "wind_dir",   0.0))
        _cep  = getattr(s, "landing_prob",    getattr(s, "cep_prob",   90))
        ax.set_title(
            f"Mode: {_mode}   ·   "
            f"Wind: {_spd:.1f} m/s @ {_dir:.0f}°   ·   "
            f"CEP: {_cep} %",
            color="#a6adc8", fontsize=9, pad=8,
        )

    # ══ Plot: 2-D Landing Map ═════════════════════════════════════════════════

    def update_map_plot(self) -> None:
        pass # Migrated to MapView

    # ── Wind-node colour / label constants ───────────────────────────────────
    # One entry per WIND_SAMPLE_ALTS level: [3, 10, 150, 300, 600] m
    # Warm (low alt) → cool (high alt) so the profile and compass share one key.
    _NODE_COLORS = ["#f38ba8", "#fab387", "#f9e2af", "#a6e3a1", "#89b4fa"]
    _NODE_LABELS = ["3 m", "10 m", "150 m", "300 m", "600 m"]

    # ══ Plot: Dual Wind Panel (Speed Profile  +  Polar Compass) ══════════════
    #
    # wind_fig has a 1×2 subplot layout:
    #   wind_profile_ax (121) — Cartesian speed-vs-altitude profile
    #   wind_ax         (122) — Polar compass with 5 altitude-node arrows
    #
    # Arrow convention: tip points in the direction the wind TRAVELS
    # (meteorological FROM direction + 180°), length ∝ wind speed.
    # Compass orientation: North at top, clockwise (aviation standard).

    def update_wind_plot(self) -> None:
        fig  = self.wind_fig
        ax_p = self.wind_profile_ax   # left — Cartesian speed profile
        ax_c = self.wind_ax           # right — polar compass
        ax_p.cla()
        ax_c.cla()
        fig.patch.set_facecolor("#1e1e1e")

        # ── Gather wind-node data ─────────────────────────────────────────────
        res   = self.state.simulation_result
        nodes = res.get("wind_nodes", []) if res is not None else []

        # Fallback: synthesise a surface node from current spinbox values.
        # ``self.state`` is the local lightweight AppState during __init__
        # (exposes ``wind_speed`` / ``wind_dir``) and the global AppState
        # after ``bind_app_state`` (exposes ``surf_wind_speed`` / ``surf_wind_dir``).
        if not nodes:
            if hasattr(self.state, "surf_wind_speed"):
                _spd = self.state.surf_wind_speed
                _dir = self.state.surf_wind_dir
            else:
                _spd = self.state.wind_speed
                _dir = self.state.wind_dir
            nodes = [{
                "alt_m":    3.0,
                "speed_ms": _spd,
                "dir_deg":  _dir,
            }]

        slice5   = nodes[:5]
        max_spd  = max((float(n.get("speed_ms", 0.0)) for n in slice5), default=1.0)
        max_spd  = max(max_spd, 1.0)
        speeds   = [float(n.get("speed_ms", 0.0)) for n in slice5]
        alts     = [float(n.get("alt_m",    0.0)) for n in slice5]
        dirs     = [float(n.get("dir_deg",  0.0)) for n in slice5]
        colors   = [
            self._NODE_COLORS[i] if i < len(self._NODE_COLORS) else "#cdd6f4"
            for i in range(len(slice5))
        ]
        labels   = [
            self._NODE_LABELS[i] if i < len(self._NODE_LABELS)
            else f"{n.get('alt_m', '?'):.0f} m"
            for i, n in enumerate(slice5)
        ]

        # ════════════════════════════════════════════════════════════════════
        # LEFT SUBPLOT: Wind Speed History (60 s rolling) or static profile
        # ════════════════════════════════════════════════════════════════════
        ax_p.set_facecolor("#1a1a2e")
        ax_p.tick_params(colors="#a6adc8", labelsize=6)
        for spine in ax_p.spines.values():
            spine.set_edgecolor("#45475a")
        ax_p.grid(True, color="#333355", linewidth=0.5, alpha=0.7)

        _HIST_ALTS = [3.0, 10.0, 150.0, 300.0, 600.0]
        hist_buf   = self._wind_hist_buf

        if hist_buf:
            # ── Time-series mode: X = relative time (s), Y = speed (m/s) ──────
            # pts are 3-tuples: (relative_t, speed_ms, dir_deg)
            for i, alt in enumerate(_HIST_ALTS):
                pts = hist_buf.get(alt, [])

                col = (self._NODE_COLORS[i] if i < len(self._NODE_COLORS) else "#cdd6f4")
                lbl = (self._NODE_LABELS[i] if i < len(self._NODE_LABELS) else f"{alt:.0f} m")

                if not pts:
                    # ZOH Fallback if deque is empty: draw horizontal line at last known speed
                    fallback_speed = speeds[i] if i < len(speeds) else 0.0
                    ax_p.axhline(fallback_speed, color=col, lw=1.4, alpha=0.85)
                    ax_p.plot([0.0], [fallback_speed], linestyle='', marker='o', color=col, markersize=5, zorder=5)
                    ax_p.plot([], [], color=col, lw=1.4, alpha=0.85, label=lbl, marker='o', markersize=4)
                    continue

                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                # Zero-Order Hold (ZOH): extend horizontal line to current time (t=0.0)
                if xs and xs[-1] < 0.0:
                    xs.append(0.0)
                    ys.append(ys[-1])

                # Plot the continuous line connecting all historical data points
                # Task 1: The Line Layer
                ax_p.plot(xs, ys, color=col, lw=1.4, alpha=0.85, linestyle='-', marker='')

                # Add proxy artist for legend
                ax_p.plot([], [], color=col, lw=1.4, alpha=0.85, label=lbl, marker='o', markersize=4)

                # Draw markers ONLY at the precise timestamps where data was actively fetched
                # (exclude the synthesized ZOH points where t == 0.0 unless it genuinely is the fetch time)
                fetch_xs = []
                fetch_ys = []
                # Filter to only keep timestamps where the speed value actually changed,
                # as `pts` contains 1Hz samples that may just be repeats.
                last_val = None
                for p in pts:
                    current_val = p[1]
                    if current_val != last_val:
                        fetch_xs.append(p[0])
                        fetch_ys.append(p[1])
                        last_val = current_val

                if fetch_xs:
                    # Task 1: The Scatter/Dot Layer
                    ax_p.plot(fetch_xs, fetch_ys, linestyle='', marker='o', color=col, markersize=5, zorder=5)

                # 10-second rolling average horizontal dotted line
                recent = [p[1] for p in pts if p[0] >= -10.0]
                if recent:
                    avg = sum(recent) / len(recent)
                    ax_p.axhline(avg, color=col, lw=1.0,
                                 linestyle="--", alpha=0.55, zorder=4)

            ax_p.axvline(0.0, color="#45475a", lw=0.8, linestyle=":", alpha=0.6)
            ax_p.set_xlabel("Time  (s)", color="#6c7086", fontsize=7, labelpad=3)
            ax_p.set_ylabel("Speed  (m/s)", color="#6c7086", fontsize=7, labelpad=3)
            ax_p.set_title(f"Wind Speed History  ({WIND_HISTORY_SAMPLES} s)",
                           color="#aaaaaa", fontsize=8, pad=6)
            ax_p.set_xlim(-60.0, 2.0)
            ax_p.set_ylim(bottom=0.0)

        else:
            # ── Static profile fallback (no history received yet) ─────────────
            if len(speeds) > 1:
                ax_p.plot(speeds, alts,
                          color="#44445a", lw=1.2, alpha=0.55, zorder=1,
                          linestyle="--")

            for spd, alt, col, lbl in zip(speeds, alts, colors, labels):
                marker = "D" if alt == 3.0 else "o"
                ax_p.scatter([spd], [alt], color=col, s=52, zorder=5,
                             marker=marker, edgecolors="#1a1a2e", linewidths=0.8)
                ax_p.text(spd + max_spd * 0.05, alt, f"{spd:.1f}",
                          color=col, fontsize=6, va="center")

            ax_p.set_xlabel("Speed  (m/s)", color="#6c7086", fontsize=7, labelpad=3)
            ax_p.set_ylabel("Altitude  (m)", color="#6c7086", fontsize=7, labelpad=3)
            ax_p.set_title("Wind Speed Profile", color="#aaaaaa", fontsize=8, pad=6)
            ax_p.set_xlim(0.0, max_spd * 1.40)
            ax_p.set_ylim(-30.0, max(alts, default=600.0) * 1.18 + 10.0)

            if alts and alts[0] == 3.0:
                ax_p.annotate(
                    "⬡ anemometer",
                    xy=(speeds[0], alts[0]),
                    xytext=(max_spd * 0.10, 60.0),
                    fontsize=5, color="#aaaaaa",
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                    lw=0.8, shrinkA=4, shrinkB=4),
                )

        # ════════════════════════════════════════════════════════════════════
        # RIGHT SUBPLOT: Polar Wind Compass
        # ════════════════════════════════════════════════════════════════════
        ax_c.set_facecolor("#1a1a2e")
        ax_c.tick_params(colors="#555555", labelsize=6)
        ax_c.grid(True, color="#333355", linewidth=0.6, alpha=0.7)

        # North at top, clockwise — aviation standard
        ax_c.set_theta_zero_location("N")
        ax_c.set_theta_direction(-1)
        ax_c.set_rlabel_position(135)
        ax_c.set_thetagrids(
            [0, 45, 90, 135, 180, 225, 270, 315],
            labels=["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"],
            fontsize=6, color="#888888",
        )

        # Radial axis: display speed ticks in m/s
        ax_c.set_rmax(1.05)
        ax_c.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax_c.set_yticklabels(
            [f"{max_spd * r:.1f}" for r in (0.25, 0.5, 0.75, 1.0)],
            color="#666666", fontsize=5,
        )

        for spd, alt, d_from, col, lbl in zip(speeds, alts, dirs, colors, labels):
            r_norm = spd / max_spd
            # Arrow points TO where wind travels (FROM + 180°)
            theta  = np.radians((d_from + 180.0) % 360.0)

            ax_c.annotate(
                "",
                xy=(theta, r_norm),
                xytext=(theta, 0.0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=col,
                    lw=2.0,
                    mutation_scale=16,
                ),
            )

            if r_norm > 0.05:
                ax_c.text(
                    theta, min(r_norm + 0.09, 1.02),
                    f"{spd:.1f}",
                    fontsize=6, color=col,
                    ha="center", va="center", fontweight="bold",
                )

            # Legend proxy
            ax_c.plot([], [], color=col, lw=2.5,
                      label=f"{lbl}  {spd:.1f} m/s @ {d_from:.0f}°")

        ax_c.set_title("Wind Compass", color="#aaaaaa", fontsize=8, pad=12)
        # Combined legend for both plots attached to the figure, not individual axes
        handles_p, labels_p = ax_p.get_legend_handles_labels()

        # Avoid duplicate labels (from ax_p history lines)
        by_label = dict(zip(labels_p, handles_p))
        if by_label:
            fig.legends.clear()
            fig.legend(
                by_label.values(), by_label.keys(),
                loc="center left", bbox_to_anchor=(0.01, 0.5),
                borderaxespad=0,
                fontsize=7,
                facecolor="#1a1a2e", edgecolor="#3a3a52",
                labelcolor="#cdd6f4", framealpha=0.88,
                ncol=1
            )

        # subplots_adjust reserves left margin for the legend
        fig.subplots_adjust(left=0.25, right=0.95, top=0.90, bottom=0.15, wspace=0.3)
        self.wind_canvas.draw_idle()
        self._update_wind_table(nodes)

    def update_wind_history(self, hist_dict) -> None:
        """Receive the global 5-altitude wind history and switch the left subplot to time-series.

        Called every second (via wind_history_updated) with the full
        dict[float, deque] from global AppState.  Converts absolute monotonic
        timestamps to relative seconds-from-now so the X axis always reads
        '−60 … 0'.
        """
        import time as _t
        now = _t.monotonic()
        self._wind_hist_buf = {
            float(alt): [
                (float(e["ts"]) - now, float(e["speed_ms"]), float(e["dir_deg"]))
                for e in entries
            ]
            for alt, entries in hist_dict.items()
        }
        self.update_wind_plot()

    def _update_wind_table(self, nodes: list | None = None) -> None:
        """Refresh the current-values table with the most recent wind readings.

        Priority: rolling hist_buf (live, per second) > nodes from last result.
        Cells are mutated in-place; no items are created after first population.
        """
        table = getattr(self, '_wind_table', None)
        if table is None:
            return

        hist_buf = self._wind_hist_buf
        if nodes is None:
            res   = self.state.simulation_result
            nodes = res.get("wind_nodes", []) if res is not None else []

        for r, (alt, lbl) in enumerate(zip(
                [3.0, 10.0, 150.0, 300.0, 600.0], self._NODE_LABELS)):
            spd_str = "—"
            dir_str = "—"

            pts = hist_buf.get(alt, [])
            if pts:
                # 3-tuple: (relative_t, speed_ms, dir_deg)
                spd_str = f"{pts[-1][1]:.1f}"
                dir_str = f"{pts[-1][2]:.0f}"
            elif nodes:
                for n in nodes:
                    if abs(float(n.get("alt_m", -1)) - alt) < 1.0:
                        spd_str = f"{float(n.get('speed_ms', 0)):.1f}"
                        dir_str = f"{float(n.get('dir_deg',  0)):.0f}"
                        break

            col = (self._NODE_COLORS[r]
                   if r < len(self._NODE_COLORS) else "#cdd6f4")
            for c, txt in enumerate([lbl, spd_str, dir_str]):
                item = table.item(r, c)
                if item is None:
                    item = QTableWidgetItem(txt)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    table.setItem(r, c, item)
                else:
                    item.setText(txt)
                if c == 0:
                    item.setData(Qt.ItemDataRole.ForegroundRole, QColor(col))

        # Update Telemetry Stats for 3.0m altitude
        surf_pts = hist_buf.get(3.0, [])
        if surf_pts:
            speeds = [p[1] for p in surf_pts]
            max_gust = max(speeds)
            mean_wind = sum(speeds) / len(speeds)
            std_dev = (sum((s - mean_wind) ** 2 for s in speeds) / len(speeds)) ** 0.5
            self.lbl_max_gust.setText(f"Max Gust: {max_gust:.1f}")
            self.lbl_mean_wind.setText(f"Mean Wind: {mean_wind:.1f}")
            self.lbl_std_dev.setText(f"Std Dev: {std_dev:.2f}")
        else:
            self.lbl_max_gust.setText("Max Gust: —")
            self.lbl_mean_wind.setText("Mean Wind: —")
            self.lbl_std_dev.setText("Std Dev: —")

    # ── Smart partial redraw ──────────────────────────────────────────────────

    def update_visual_overlays(self, state) -> None:
        """Partial redraw: update ellipse + KDE contours without ax.cla().

        Draw order (so artists stack correctly on a fixed axes background):
          1. _render_overlays  — clears stale artists, draws KDE contours + ellipse

        No re-simulation; no full axes clear.  The SimulationWorker is not involved.
        """
        scatter = getattr(state, 'cached_mc_scatter', None)
        if scatter is None:
            return

        if isinstance(scatter, np.ndarray):
            if scatter.ndim != 2 or scatter.shape[1] < 2:
                return
            pts = scatter[:, :2].astype(float)
        else:
            pts = np.array([(float(p[0]), float(p[1])) for p in scatter], dtype=float)

        if len(pts) < 4:
            return

        prob = int(getattr(state, 'cep_probability', 90))

        # ── Step 1: redraw ellipse + KDE contours from updated state ─────────────
        # _render_overlays clears _overlay_artists then redraws ellipse + KDE lines.
        # mc_ellipse was just updated by the controller before this call.
        self._render_overlays(
            getattr(state, 'mc_ellipse',    None),
            getattr(state, 'kde_contours',  None) or [],
            prob,
        )

        # ── Step 2: compute CEP radius and draw circle on top ────────────────────
        res = self.state.simulation_result
        _cx = float(pts[:, 0].mean())
        _cy = float(pts[:, 1].mean())
        lx  = float(res.get("land_x", _cx)) if res else _cx
        ly  = float(res.get("land_y", _cy)) if res else _cy

        dists = np.hypot(pts[:, 0] - lx, pts[:, 1] - ly)
        cep_r = float(np.percentile(dists, prob))

        ax = self.map_ax
        if cep_r > 0:
            theta_c = np.linspace(0.0, 2.0 * np.pi, 200)
            (line,) = ax.plot(
                lx + cep_r * np.cos(theta_c),
                ly + cep_r * np.sin(theta_c),
                color="#cba6f7", lw=2.0, alpha=0.90, zorder=5,
                label=f"CEP {prob} %  ({cep_r:.1f} m)",
            )
            ann = ax.text(
                lx, ly + cep_r * 1.08,
                f"CEP Radius: {cep_r:.1f} m",
                color="#cba6f7", fontsize=7.5, ha="center", zorder=7,
            )
            self._overlay_artists.extend([line, ann])

        self.map_canvas.draw_idle()

    def _render_overlays(
        self,
        ellipse,
        kde_contours: list,
        prob: int,
    ) -> None:
        """Remove stale overlay artists and draw fresh CEP/KDE/ellipse overlays.

        Called only from update_visual_overlays (partial redraw on prob change).
        update_map_plot handles its own overlay drawing inline after ax.cla().
        Caller is responsible for calling map_canvas.draw_idle() afterwards.
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

    # ── Partial redraw: swap KDE + ellipse layer in-place ─────────────────────

    def update_ellipse_layer(self, ellipse_data: dict | None) -> None:
        if self.state.simulation_result is not None:
            s = self.state
            _mode = getattr(s, "operation_mode", getattr(s, "sim_mode", ""))
            _spd  = getattr(s, "surf_wind_speed", getattr(s, "wind_speed", 0.0))
            _dir  = getattr(s, "surf_wind_dir",   getattr(s, "wind_dir",   0.0))
            _cep  = getattr(s, "landing_prob",    getattr(s, "cep_prob",   90))
            self.profile_ax.set_title(
                f"Mode: {_mode}   ·   "
                f"Wind: {_spd:.1f} m/s @ {_dir:.0f}°   ·   "
                f"CEP: {_cep} %",
                color="#a6adc8", fontsize=9, pad=8,
            )
            self.profile_canvas.draw_idle()

    def refresh_visuals(self) -> None:
        """Refresh KDE contours, error ellipse, and CEP circle from cached data.

        Called externally (e.g. after Advanced Settings OK) to update all
        statistical overlays without re-running the physics engine.
        No-op when no simulation result is cached.
        """
        res = self.state.simulation_result
        ellipse_data = res.get("ellipse") if isinstance(res, dict) else None
        self.update_ellipse_layer(ellipse_data)

    # ── Action handlers ────────────────────────────────────────────────────────

    def _on_azim_changed(self, value: int) -> None:
        """Rotate the 3-D profile to the new azimuth without a full redraw."""
        self.profile_ax.view_init(elev=25, azim=value)
        self.profile_canvas.draw_idle()

    def bind_app_state(self, state) -> None:
        """Forward the global :class:`AppState` to nested dialogs that need it.

        AppWindow itself is constructed before the global ``AppState`` exists
        (see ``main_qt.py``), so widgets that require bi-directional binding
        receive their state reference here, once both objects are alive.

        Wires:
          *  Advanced Settings dialog (Phase B aero/motor + Phase C Cd curves)
          *  File menu Save / Load Session actions (Phase E session manager)
          *  Phase 2 Map View (for global coordinates and target radius)
        """
        print(f"=== AppWindow.bind_app_state === Forwarding global State: id={id(state)}")
        self.state = state  # Overwrite with the true global instance
        self._app_state = state            # cached for the session menu slots
        self._adv_dialog.bind_app_state(state)
        if hasattr(self, 'map_view') and self.map_view:
            self.map_view.bind_app_state(state)

    # ── Session persistence (Phase E) ────────────────────────────────────────

    def _on_save_session(self) -> None:
        """Prompt for a JSON path and serialise the global AppState there.

        Strictly MVVM: this slot calls :mod:`utils.session_manager` which only
        reads AppState properties — no UI state leaks into the JSON, and no
        direct widget access happens here.
        """
        if getattr(self, "_app_state", None) is None:
            QMessageBox.warning(
                self, "Save Session",
                "AppState is not yet wired to AppWindow; session save is "
                "unavailable.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "kazamidori_session.json",
            "JSON files (*.json);;All files (*.*)")
        if not filepath:
            return

        try:
            from utils.session_manager import save_session
            save_session(self._app_state, filepath)
        except OSError as exc:
            QMessageBox.warning(
                self, "Save Session",
                f"Failed to write session file:\n{filepath}\n\n{exc}")
            return

        QMessageBox.information(
            self, "Save Session",
            f"Session saved successfully to:\n{filepath}")

    def _on_load_session(self) -> None:
        """Prompt for a JSON path and restore advanced settings from it.

        After ``load_session`` writes the values through the AppState property
        setters, every observer (the Advanced Settings dialog bindings, the
        Cd curve preview button, downstream workers) updates automatically
        via Qt signals — no manual UI refresh is needed.
        """
        if getattr(self, "_app_state", None) is None:
            QMessageBox.warning(
                self, "Load Session",
                "AppState is not yet wired to AppWindow; session load is "
                "unavailable.")
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            "JSON files (*.json);;All files (*.*)")
        if not filepath:
            return

        try:
            from utils.session_manager import load_session, SessionError
            load_session(self._app_state, filepath)
        except (OSError, SessionError) as exc:
            QMessageBox.warning(
                self, "Load Session",
                f"Failed to load session file:\n{filepath}\n\n{exc}")
            return

        QMessageBox.information(
            self, "Load Session",
            f"Session loaded successfully from:\n{filepath}")

    def _on_advanced_settings(self) -> None:
        """Open the Advanced Settings dialog modally."""
        self._adv_dialog.exec()

    def _on_load_airframe_json(self) -> None:
        """Emit sig_load_rocket_json_clicked so external consumers can handle file I/O."""
        self.sig_load_rocket_json_clicked.emit()

    def _on_manual_config(self) -> None:
        """Open the ManualSetupDialog modally."""
        self._evaluate_config_deltas()
        self._manual_dialog.exec()

    def _evaluate_config_deltas(self, *args) -> None:
        """Evaluate current widget values against AppState.original_rocket_config."""
        orig = getattr(self.state, "original_rocket_config", None)
        if orig is None:
            return

        def _check_and_style(widget, label, orig_key, scale=1.0):
            if widget.value() == -9999.0:
                return

            # Allow minor floating point deviations
            current_val = widget.value()
            orig_val = orig.get(orig_key, 0.0) * scale
            if abs(current_val - orig_val) > 1e-4:
                label.setStyleSheet("color: #ff5555; font-weight: bold;")
            else:
                label.setStyleSheet("")

        md = self._manual_dialog
        _check_and_style(self.af_mass_input,      md.lbl_mass,      "mass")
        _check_and_style(self.af_cg_input,        md.lbl_cg,        "cg")
        _check_and_style(self.af_len_input,       md.lbl_len,       "length")
        _check_and_style(self.af_radius_input,    md.lbl_radius,    "radius")
        _check_and_style(self.af_nose_input,      md.lbl_nose,      "nose_length")
        _check_and_style(self.af_finroot_input,   md.lbl_finroot,   "fin_root")
        _check_and_style(self.af_fintip_input,    md.lbl_fintip,    "fin_tip")
        _check_and_style(self.af_finspan_input,   md.lbl_finspan,   "fin_span")
        _check_and_style(self.af_finpos_input,    md.lbl_finpos,    "fin_pos")
        _check_and_style(self.af_motorpos_input,  md.lbl_motorpos,  "motor_pos")
        _check_and_style(self.af_motormass_input, md.lbl_motormass, "motor_dry_mass")

    def _on_manual_config_reset(self) -> None:
        """Reset values in ManualSetupDialog to match original_rocket_config."""
        orig = getattr(self.state, "original_rocket_config", None)
        if orig is None:
            return

        def _set(widget, orig_key, scale=1.0):
            val = orig.get(orig_key)
            if val is not None:
                widget.setValue(val * scale)

        _set(self.af_mass_input,      "mass")
        _set(self.af_cg_input,        "cg")
        _set(self.af_len_input,       "length")
        _set(self.af_radius_input,    "radius")
        _set(self.af_nose_input,      "nose_length")
        _set(self.af_finroot_input,   "fin_root")
        _set(self.af_fintip_input,    "fin_tip")
        _set(self.af_finspan_input,   "fin_span")
        _set(self.af_finpos_input,    "fin_pos")
        _set(self.af_motorpos_input,  "motor_pos")
        _set(self.af_motormass_input, "motor_dry_mass")

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
                burn_time = thrust_data[-1][0]  # absolute end time (s); t=0 is ignition
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
                # Persist for SimController._collect_params()
                self._motor_thrust_data = [list(pt) for pt in thrust_data]
                self._motor_burn_time   = burn_time
            else:
                self.motor_label.setText(f"{name}  (no data)")
                self.set_status(f"No valid thrust rows found in {name}", "#f38ba8")

        except Exception as exc:
            self.motor_label.setText(f"{name}  (error)")
            self.set_status(f"Motor load error: {exc}", "#f38ba8")

    def _on_mode_changed(self, mode: str) -> None:
        is_free = "free" in mode.lower() or "自由" in mode
        self.rmax_input.setEnabled(not is_free)
        if is_free:
            self.update_status_indicator("🟢 GO (FREE FLIGHT MODE)")
        else:
            self.update_status_indicator("⬤  STANDBY")
        if mode == "定点滞空":
            self.rmax_input.setValue(50.0)
        elif mode in ("高度", "有翼"):
            self.rmax_input.setValue(250.0)

    def _on_about(self) -> None:
        QMessageBox.information(
            self, "About Kazamidori",
            "Kazamidori  —  Trajectory & Landing Point Simulator\n\n"
            "Qt6 / PySide6  (ui_qt/)   |   Tkinter legacy (ui/)\n\n"
            "Both UIs share the same core/ simulation engine.")

    # ── Public API ─────────────────────────────────────────────────────────────

    @Slot(str)
    def show_error_message(self, message: str) -> None:
        """Display a critical error dialog. Called by the Controller on file-parse failures."""
        QMessageBox.critical(self, "Error", message)

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

    def update_status_indicator(self, status_text: str) -> None:
        """Update the GO/NO-GO label to an arbitrary string, styled by keyword.

        No-op if called before the label is constructed (early __init__ calls).

        Color rules (case-insensitive keyword match):
            'NO-GO' or 'UNSAFE'    → bold red    (danger — do not launch)
            'WARNING' or 'CAUTION' → bold amber  (early warning while MC runs)
            'GO' (not 'NO-GO')     → bold green  (clear to proceed)
            anything else          → neutral grey (standby / calculating)

        Safe to call at any point in the simulation cycle, including while the
        MC progress bar is still advancing.  Qt label setText() is a synchronous
        property write that bypasses the canvas render queue, so the operator
        sees the update immediately on the next event-loop tick.
        """
        if not hasattr(self, "_go_nogo_label"):
            return
        t = status_text.upper()
        if "NO-GO" in t or "UNSAFE" in t:
            style = (
                "font-size: 12pt; font-weight: bold; color: #f38ba8; padding: 8px;"
                "background: #1f0d0d; border-radius: 8px; border: 2px solid #f38ba8;"
            )
        elif "WARNING" in t or "CAUTION" in t:
            style = (
                "font-size: 11pt; font-weight: bold; color: #f9e2af; padding: 8px;"
                "background: #1f1a0d; border-radius: 8px; border: 2px solid #f9e2af;"
            )
        elif "GO" in t:
            style = (
                "font-size: 13pt; font-weight: bold; color: #a8e6a1; padding: 8px;"
                "background: #0d1f0d; border-radius: 8px; border: 2px solid #a8e6a1;"
            )
        else:
            style = (
                "font-size: 11pt; font-weight: bold; color: #a6adc8; padding: 8px;"
                "background: #1e1e2e; border-radius: 8px; border: 2px solid #45475a;"
            )
        self._go_nogo_label.setText(status_text)
        self._go_nogo_label.setStyleSheet(style)

    def set_go_nogo(self, go: bool) -> None:
        """Binary GO/NO-GO update — delegates to update_status_indicator."""
        if go:
            self.update_status_indicator("🟢  GO  (LAUNCH CLEAR)")
        else:
            self.update_status_indicator("🔴  NO-GO  (WIND LIMIT EXCEEDED)")

    def set_progress(self, value: int, label: str = "") -> None:
        self._progress.setValue(max(0, min(100, value)))
        lbl = getattr(self, '_phase_label', None)
        if label:
            self._progress.setFormat("%p%")
            if lbl is not None:
                lbl.setText(label)
        elif lbl is not None and value == 0:
            lbl.setText("Idle")


# ── Standalone entry point ────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = AppWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
