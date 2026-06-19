# Code Health Audit Report

**Date**: $(date)
**Grade**: A

## Evaluation Criteria

* **Root Clutter**: Found `test_physics_core.py` in the root directory. Moved to `archive/`. Also found `scratch/test_gui_hang.py` which was floating around outside the `tests/` directory; moved to `archive/`.
* **Domain Leakage**: Scanned imports within `core/`. No file imports PySide6, PyQt, Matplotlib, Folium, or other GUI components. The only match was a comment in `core/mc_worker.py` indicating that it must not import PySide6.
* **Coordinate Violation**: Scanned `core/` for coordinate processing. `dist_to_target` and related parameters are computed using ENU (X,Y) offsets (e.g., `math.hypot(impact_x - target_x, impact_y - target_y)`). Lat/Lon usage in `core/` is restricted to Environment/map mapping and initialization logic (e.g. `launch_lat`, `launch_lon`). No explicit distance calculations using Lat/Lon instead of ENU were found.
* **Test Placement**: Tests are properly structured inside `tests/` (e.g. `tests/core/test_optimization.py`). Floating tests like `test_physics_core.py` and `scratch/test_gui_hang.py` have been moved to `archive/`.

## Corrective Actions
1. `mv test_physics_core.py archive/`
2. `mv scratch/test_gui_hang.py archive/`
3. Staged changes using `git add -u` and `git add archive/...`
