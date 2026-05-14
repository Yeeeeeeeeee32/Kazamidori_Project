import sys
from PySide6.QtWidgets import QApplication
from ui_qt.app_window import AppWindow

app = QApplication(sys.argv)
window = AppWindow()

dummy_result = {
    "trajectory_x": [0, 10],
    "trajectory_y": [0, 10],
    "trajectory_z": [0, 100],
    "kde": {"X_m": [[0, 1], [0, 1]], "Y_m": [[0, 0], [1, 1]], "Z": [[0.5, 0.5], [0.5, 0.5]]},
    "ellipse": {"width": 10, "height": 5, "cx": 5, "cy": 5, "angle_rad": 0, "a": 5, "b": 2.5},
    "mc_scatter_x": [1, 2, 3],
    "mc_scatter_y": [1, 2, 3],
    "events": {"burnout": [5, 5, 50], "apogee": [10, 10, 100]},
    "apogee_m": 100
}
window.state.simulation_result = dummy_result

def get_count():
    return len(window.profile_ax.collections) + len(window.profile_ax.lines) + len(window.profile_ax.patches)

print(f"--- Initial State --- Artists: {get_count()}")
window.action_show_kde.setChecked(False)
app.processEvents()
print(f"--- KDE False --- Artists: {get_count()}")
window.action_show_cep.setChecked(False)
app.processEvents()
print(f"--- CEP False --- Artists: {get_count()}")
