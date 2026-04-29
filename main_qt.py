"""
main_qt.py
PySide6 entry point for the Kazamidori Project.

Responsibilities (and nothing else):
  1. Construct QApplication
  2. Build AppState with the canonical default config
  3. Inject AppState into AppWindow and show it
  4. Hand off to Qt's event loop
"""

import sys

from PySide6.QtWidgets import QApplication

from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow

DEFAULT_CONFIG = {
    "wind_uncertainty":      0.20,
    "thrust_uncertainty":    0.05,
    "allowable_uncertainty": 20.0,
    "landing_prob":          90,
    "mc_n_runs":             200,
}

if __name__ == "__main__":
    app = QApplication(sys.argv)
    state = AppState(config=DEFAULT_CONFIG)
    window = AppWindow(state=state)
    window.show()
    sys.exit(app.exec())
