import sys
import time
import os
from PySide6.QtWidgets import QApplication
from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.sim_controller import SimController
from core.pool_manager import get_global_pool
import PySide6.QtCore as qc

def log(msg):
    with open('worker_trace.log', 'a', encoding='utf-8') as f:
        f.write(f"[{time.time():.2f}] {msg}\n")
    print(msg, flush=True)

if os.path.exists('worker_trace.log'):
    os.remove('worker_trace.log')
if os.path.exists('worker_internal_trace.log'):
    os.remove('worker_internal_trace.log')

log("Starting worker_trace.py")
get_global_pool()

app = QApplication(sys.argv)
state = AppState()
win = AppWindow()
ctrl = SimController(win, state)

state.flight_mode = 'Free Flight'  # Use Free Flight to ensure SimulationWorker!
state.motor_thrust_data = [[0, 10], [1, 10], [2, 0]]
state.motor_burn_time = 2.0
state.motor_cg_pos = 0.38
state.motor_dry_mass = 0.015
state.parachute_cd = 0.8
state.parachute_area = 0.126
state.parachute_lag = 0.5
state.backfire_delay = 4.0
state.launch_lat = 35.6
state.launch_lon = 139.7
state.wind_uncertainty = 0.2
state.thrust_uncertainty = 0.05
state.target_radius = 50.0

win.lat_input.setValue(35.6)
win.lon_input.setValue(139.7)
win.azim_input.setValue(0.0)

# Mock window to bypass _validate_run_prerequisites motor thrust curve check
win._motor_thrust_data = state.motor_thrust_data
win._motor_burn_time = 2.0

log("Triggering run...")
ctrl._on_run_clicked()

qc.QTimer.singleShot(10000, app.quit)
log("Running app.exec()...")
app.exec()
log("Done.")
