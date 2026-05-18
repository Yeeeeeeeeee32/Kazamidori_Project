import sys
from PySide6.QtWidgets import QApplication
from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from main_qt import SimController

app = QApplication(sys.argv)
app_state = AppState(config={})
window = AppWindow()
controller = SimController(window, app_state)

print("Make state ready via loaders...")

# Mock parsed config from .rkt file
cfg = {
    "airframe": {
        "mass": 1.0,
        "cg": 1.0,
        "length": 1.0,
        "radius": 0.5,
        "nose_length": 1.0,
        "fin_root": 1.0,
        "fin_tip": 1.0,
        "fin_span": 1.0,
        "fin_pos": 1.0,
        "motor_pos": 1.0,
        "motor_dry_mass": 1.0,
        "backfire_delay": 1.0
    },
    "parachute": {
        "cd": 1.0,
        "area": 1.0,
        "lag": 1.0
    },
    "moi": {
        "ixx": 1.0,
        "iyy": 1.0,
        "izz": 1.0
    }
}

af = cfg["airframe"]
par = cfg["parachute"]

# Simulating _on_load_rkt EXACTLY how it is in the code
# The code in main_qt.py -> SimController._on_load_rkt does this:
app_state.rocket_dry_mass = af["mass"]
app_state.rocket_cg       = af["cg"]
app_state.rocket_length   = af["length"]
app_state.rocket_diameter = af["radius"] * 2.0
app_state.nose_length     = af["nose_length"]
app_state.fin_root_chord  = af["fin_root"]
app_state.fin_tip_chord   = af["fin_tip"]
app_state.fin_span        = af["fin_span"]
app_state.fin_position    = af["fin_pos"]
app_state.motor_cg        = af["motor_pos"]
app_state.motor_dry_mass  = af["motor_dry_mass"]
app_state.backfire_delay  = af["backfire_delay"]
app_state.parachute_cd   = par["cd"]
app_state.parachute_area = par["area"]
app_state.parachute_lag  = par["lag"]

# Wait, what if someone uses the UI directly?
window.lat_input.setValue(35.0)
window.lon_input.setValue(135.0)

# Simulate _on_load_motor
window._motor_thrust_data = [[0, 100], [1, 100], [2, 0]]
window._motor_burn_time = 2.0

# Simulate getting some wind data
for _ in range(5):
    controller._on_wind_tick()

print("Is ready?", app_state.is_ready_to_run)
