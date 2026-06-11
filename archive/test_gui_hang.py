import os
import sys
import time

workspace_dir = r"c:\Users\yezic\OneDrive\Desktop\2026年度\Kazamidori_Project"
sys.path.insert(0, workspace_dir)

os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",    "1")

# Use offscreen platform for headless Qt run
sys.argv.append("-platform")
sys.argv.append("offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QCoreApplication
from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.sim_controller import SimController
from utils.data_loader import load_rocket_config, load_motor_csv
from PIL import Image

DEFAULT_CONFIG: dict = {
    "wind_uncertainty":   0.20,
    "thrust_uncertainty": 0.05,
    "landing_prob":        90,
    "mc_n_runs":          50, # Use standard 50 runs
}

def create_dummy_map():
    map_dir = os.path.join(workspace_dir, "assets", "offline_map")
    os.makedirs(map_dir, exist_ok=True)
    
    meta_path = os.path.join(map_dir, "map_meta.json")
    img_path = os.path.join(map_dir, "background.png")
    
    # Create 3000x3000px image (approx 9MB raw) to simulate realistic map file size
    print("Creating dummy 3000x3000px background.png...")
    img = Image.new("RGBA", (3000, 3000), color=(30, 30, 46, 255))
    img.save(img_path)
    
    import json
    meta_data = {
        "magnetic_declination": -7.5,
        "extent_meters": [-1000.0, 1000.0, -1000.0, 1000.0]
    }
    with open(meta_path, "w") as f:
        json.dump(meta_data, f)
    print("Dummy map files created.")

def main():
    create_dummy_map()
    
    print("Initializing QApp...")
    app = QApplication(sys.argv)
    
    print("Initializing AppState, AppWindow, SimController...")
    state = AppState(config=DEFAULT_CONFIG)
    window = AppWindow()
    window.bind_app_state(state)
    controller = SimController(window, state)
    
    print("Loading rocket config...")
    cfg = load_rocket_config(os.path.join(workspace_dir, "Rocket.json"))
    af = cfg["airframe"]
    par = cfg["parachute"]
    
    state.rocket_dry_mass = af["mass"]
    state.rocket_cg = af["cg"]
    state.rocket_length = af["length"]
    state.rocket_diameter = af["radius"] * 2.0
    state.nose_length = af["nose_length"]
    state.fin_root_chord = af["fin_root"]
    state.fin_tip_chord = af["fin_tip"]
    state.fin_span = af["fin_span"]
    state.fin_position = af["fin_pos"]
    
    state.parachute_cd = par["cd"]
    state.parachute_area = par["area"]
    state.parachute_lag = par["lag"]
    
    state.motor_cg_pos = 0.38
    state.motor_dry_mass = 0.015
    state.backfire_delay = 4.0
    
    # Explicitly set uncertainties
    state.wind_uncertainty = 0.20
    state.thrust_uncertainty = 0.05
    
    print("Loading motor CSV...")
    motor_data = load_motor_csv(os.path.join(workspace_dir, "Estes_B4.csv"))
    window._motor_thrust_data = [list(pt) for pt in motor_data.thrust_points]
    window._motor_burn_time = motor_data.burn_time
    
    print("Setting up wind profile and other parameters...")
    state.wind_profile_data = [
         {"alt_m": 3.0, "speed_ms": 4.0, "dir_deg": 100.0},
         {"alt_m": 10.0, "speed_ms": 4.5, "dir_deg": 100.0},
         {"alt_m": 150.0, "speed_ms": 5.0, "dir_deg": 100.0},
         {"alt_m": 300.0, "speed_ms": 5.5, "dir_deg": 100.0},
         {"alt_m": 600.0, "speed_ms": 6.0, "dir_deg": 100.0},
    ]
    # Populate wind history buffer
    for _ in range(10):
        state.append_wind_reading(4.0, 100.0)
        
    state.flight_mode = "定点滞空"
    state.target_radius = 50.0
    state.landing_prob = 90
    
    print("Warm up the pool...")
    from core.pool_manager import get_global_pool, warmup_pool
    get_global_pool()
    warmup_pool()
    
    # Track execution
    start_time = time.time()
    
    def on_nominal_done(payload):
        print(f"[{time.time()-start_time:.2f}s] Nominal done callback fired!")
        
    def on_progress_updated(current, total, msg):
        print(f"[{time.time()-start_time:.2f}s] Progress: {current}/{total} ({msg})")
        
    def on_finished(result):
        print(f"[{time.time()-start_time:.2f}s] Finished! Result cancelled: {result.get('cancelled')}")
        app.quit()
        
    def on_error(msg):
        print(f"[{time.time()-start_time:.2f}s] Worker Error: {msg}")
        app.quit()
        
    # Connect directly to controller worker slots to monitor
    original_on_run_clicked = controller._on_run_clicked
    
    def hooked_on_run_clicked():
        print("Run clicked!")
        original_on_run_clicked()
        # Connect signals of the newly created worker
        if controller._worker:
            controller._worker.sig_nominal_done.connect(on_nominal_done)
            controller._worker.sig_progress.connect(on_progress_updated)
            controller._worker.sig_finished.connect(on_finished)
            controller._worker.error.connect(on_error)
            print("Signals hooked successfully.")
        else:
            print("No worker created!")
            
    # Set a timeout timer to prevent infinite hang of the test script
    timeout_timer = QTimer()
    timeout_timer.setInterval(40000) # 40 seconds timeout
    def on_timeout():
        print(f"[{time.time()-start_time:.2f}s] TIMEOUT! App hung.")
        if controller._worker:
            print("Worker isRunning:", controller._worker.isRunning())
            print("Worker stop event set:", controller._worker._stop_event.is_set())
        # Print where main thread is stuck by checking thread stack frames
        import traceback
        import threading
        print("Main thread stack trace:")
        for thread_id, stack in sys._current_frames().items():
            if thread_id == threading.main_thread().ident:
                print("".join(traceback.format_stack(stack)))
        app.quit()
    timeout_timer.timeout.connect(on_timeout)
    timeout_timer.start()
    
    # Trigger run after event loop starts
    QTimer.singleShot(100, hooked_on_run_clicked)
    
    print("Entering Qt event loop...")
    app.exec()
    print("Qt event loop exited.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
