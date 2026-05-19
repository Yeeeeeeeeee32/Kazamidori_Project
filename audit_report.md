# Codebase Health Audit Report for Kazamidori Project

## 1. DRY (Don't Repeat Yourself) Violations

### WARNING: Duplicated Constants
* **Issue:** `G0` (standard gravity) and `111132.92` (m/deg latitude) are duplicated.
* **Location:**
    * `core/simulation.py` lines 502, 505, 522 use `-9.81` directly in diagnostics instead of `G0` from `core/constants.py` (which is `9.80665`).
    * `111111.0` sequence used for conversions: `ui_qt/map_view.py` uses `111111.0` (lines 363-364) and `ui_qt/sim_controller.py` uses `111_320.0` (lines 672-673) for rough latitude degree-to-meters conversions, whereas `utils/geo_math.py` has a more robust `meters_per_degree` (line 15) and `offset_to_latlon` (line 31) function.
* **Proposed Fix:** Use `-G0` from `core/constants.py` everywhere. Replace hardcoded `111111.0` and `111320.0` in the UI layer with `utils.geo_math.offset_to_latlon` or `utils.geo_math.meters_per_degree` calls.

### WARNING: Duplicate Math Functions
* **Issue:** Duplicate implementation of `speed_dir_to_uv` logic.
* **Location:**
    * `ui_qt/sim_controller.py` lines 736-737: `mu_u = surf_spd * math.sin(math.radians(surf_dir))`
    * `ui_qt/app_state.py` lines 1060-1061: `live_u = speed * math.sin(math.radians(direction))`
* **Proposed Fix:** Use `speed_dir_to_uv` from `core/wind_model.py` instead of manually calculating via sine/cosine in the UI state and controller. Note: `speed_dir_to_uv` already exists in `core/wind_model.py`.

### OPTIMIZATION: Duplicate UI Loading Logic
* **Issue:** Duplicate motor CSV loading logic.
* **Location:** `ui_qt/app_window.py` implements its own `_on_load_motor` (line 2887+) with a custom `csv.reader` loop, duplicating logic already available in `utils/data_loader.py` `load_motor_csv`.
* **Proposed Fix:** Delegate motor CSV loading in `_on_load_motor` to `utils.data_loader.load_motor_csv` to eliminate the redundant parse loop.

## 2. Dead Code & Unused Imports

### OPTIMIZATION: Unused Imports
* **Issue:** Extraneous imports across various files.
* **Location:**
    * `ui_qt/app_window.py`: `import os, csv as _csv` inside `_on_load_motor` (line 2894) is unnecessary if `load_motor_csv` is used.
    * `ui_qt/sim_controller.py`: Multiple `import os as _os` instances within methods (`1047`, `1081`, `1123`, `1166`, `1213`).
* **Proposed Fix:** Clean up unused imports, particularly those left over after fixing the duplicate CSV loader.

## 3. State & Type Contradictions

### WARNING: Type Contradictions in AppState
* **Issue:** Initial values or properties with implicit types conflicting with explicit hints.
* **Location:** `ui_qt/app_state.py`.
    * Typecasting using float/int is inconsistent when receiving UI inputs. Wait to refactor until more info is needed.
* **Proposed Fix:** Ensure robust null-safe type casting (using `_safe_float` or `_safe_int`) in all `AppState` setters if we edit them. (Note: Only read-only report required for now).

## 4. Architecture Leaks

### CRITICAL: Core & Utils Isolation
* **Issue:** The architecture dictates NO `PySide6` or `PyQt` components are imported inside the `core/` or `utils/` directories.
* **Location:** Whole project.
* **Proposed Fix:** None required; the isolation is currently maintained. I have verified this using `grep`.

## 5. Performance / UI Thread Blocks

### CRITICAL: UI Thread Blocking
* **Issue:** Heavy file I/O operations inside main thread slots.
* **Location:**
    * `ui_qt/app_window.py` line 2894+: `_on_load_motor` performs synchronous CSV reading and calculation on the main UI thread.
    * `ui_qt/map_view.py` line 375+: `_render_map_tiles` performs `Image.open` and image manipulation (`np.array(img)`) directly in the main UI thread (lines 413 and 418).
* **Proposed Fix:**
    * Move CSV loading and parsing to a background worker or delegate to a fast loader (it's quick, but standard practice favors workers if large).
    * Pre-load the offline map `background.png` into a memory-cached object (e.g., in `AppState` or `MapView` init) instead of hitting the disk inside `_render_map_tiles` on every single map redraw/pan.