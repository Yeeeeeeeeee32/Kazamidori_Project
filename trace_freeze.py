import sys
import time
import os
from PySide6.QtWidgets import QApplication
from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.sim_controller import SimController
from core.pool_manager import get_global_pool

def log(msg):
    with open('freeze_trace.log', 'a', encoding='utf-8') as f:
        f.write(f"[{time.time():.2f}] {msg}\n")
    print(msg, flush=True)

if os.path.exists('freeze_trace.log'):
    os.remove('freeze_trace.log')

log("Starting test script...")
get_global_pool()
log("Global pool initialized.")

app = QApplication(sys.argv)
state = AppState()
win = AppWindow()
ctrl = SimController(win, state)

state.flight_mode = 'Launch Angle Optimization'
log("Flight mode set.")

log("Mocking thrust data so it doesn't fail on validation...")
state.motor_thrust_data = [[0, 10], [1, 10], [2, 0]]
state.motor_burn_time = 2.0
state.motor_cg_pos = 0.38
state.motor_dry_mass = 0.015

log("Triggering run clicked...")
ctrl._on_run_clicked()
log("Run clicked triggered successfully.")

import PySide6.QtCore as qc

def check_progress():
    log(f"Current progress: {win.progress_bar.value()}")

timer = qc.QTimer()
timer.timeout.connect(check_progress)
timer.start(1000)

log("Starting app.exec()...")
qc.QTimer.singleShot(10000, app.quit)
app.exec()
log("app.exec() finished!")
