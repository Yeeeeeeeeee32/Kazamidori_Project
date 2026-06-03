"""
main_qt.py
PySide6 entry point and top-level controller for the Kazamidori Project.

Responsibilities
----------------
1. Construct QApplication.
2. Build the shared AppState (cross-component data bus for computed results).
3. Show AppWindow (which owns its own reactive plot-state internally).
4. Wire RUN / STOP buttons -> SimulationWorker via SimController.
5. Route worker signals back to AppWindow public API and AppState properties.
6. Start the Qt event loop.

Architecture note
-----------------
AppWindow defines its own lightweight AppState (with a needs_redraw Signal)
to drive the live 3-D plot without polling.  The ui_qt.app_state.AppState
created here is the broader application data bus that holds final simulation
results and will be consumed by future views (map overlay, Phase 2 panel,
etc.).  These are intentionally separate objects; the controller mediates
writes to both.
"""

import os
# Cap BLAS/OMP threads in the parent process before any numpy/scipy import occurs
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",    "1")

import faulthandler
import sys

# Dump C-level tracebacks on native crashes (segfault, access violation, etc.)
# so silent "window disappears" failures become diagnosable in stderr.
faulthandler.enable()

# GUI imports are moved inside __main__ to prevent child processes from importing them

DEFAULT_CONFIG: dict = {
    "wind_uncertainty":   0.20,
    "thrust_uncertainty": 0.05,
    "landing_prob":        90,
    # 50 runs is statistically sufficient for the CEP ellipse (CLT holds at n≥30)
    # and keeps MC wall-time under ~30 s on a mid-range laptop.
    # Increase to 100-200 for final competition analysis if time allows.
    "mc_n_runs":          50,
}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    from PySide6.QtWidgets import QApplication
    from ui_qt.app_state import AppState
    from ui_qt.app_window import AppWindow, GLOBAL_QSS
    from ui_qt.sim_controller import SimController

    def global_excepthook(exc_type, exc_value, exc_traceback):
        import traceback
        import sys
        print("--- GLOBAL EXCEPTION CAUGHT ---", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
        print("-------------------------------", file=sys.stderr)
        sys.exit(1)
    sys.excepthook = global_excepthook

    app = QApplication(sys.argv)
    # Task 4: enforce high-contrast dark theme globally before any widget is shown.
    app.setStyleSheet(GLOBAL_QSS)

    # Shared data bus: holds computed simulation results for all future views.
    app_state = AppState(config=DEFAULT_CONFIG)
    print(f"=== main_qt.py === Created global AppState: id={id(app_state)}")

    # Inject the global app_state to unify instances and resolve the map view unresponsiveness
    window = AppWindow(app_state)
    window.show()

    # Controller wires the run/stop buttons to the background worker.
    # Must be assigned to a variable so it is not garbage-collected while
    # the event loop runs.
    controller = SimController(window, app_state)

    sys.exit(app.exec())



