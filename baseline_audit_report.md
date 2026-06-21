# Baseline Audit Report

## 1. Physics & Math Validation
- **Violation**: Magic numbers used for Earth's radius and degree-to-meter conversion instead of standard constants.
  - `utils/map_downloader.py`: `R_EARTH = 6_378_137.0` is hardcoded.
  - `utils/geo_math.py`: `R_EARTH = 6_378_137.0` is hardcoded.
  - `ui_qt/sim_controller.py`: `R = 6371.0 # km` and `111_320.0` (meters per degree approx) are hardcoded.
  - These should be centralized in `core/constants.py` (e.g. `R_EARTH_M = 6_378_137.0`).

## 2. Strict Unit Consistency
- **Violation**: The parameter name `angle` in `geo_math.py`'s `circle_polygon` does not specify `rad` or `deg`, although it seems to be in radians.
- **Violation**: `ui_qt/sim_controller.py` uses variables `R = 6371.0 # km` and returns `R * c` in km instead of meters, and the function is named `get_distance` without unit suffix (should be e.g., `_m` or `_km`).

## 3. Coordinate System Integrity
- **Violation**: `ui_qt/sim_controller.py` defines a local `get_distance(lat1, lon1, lat2, lon2)` function inside `_on_map_coordinates_picked` using the Haversine formula (with lat/lon). It should instead use the centralized metric ENU coordinate system conversion via `utils.geo_math`.
- **Violation**: `ui_qt/sim_controller.py` computes offset-to-latlon manually using `111_320.0` inside `_update_app_state_with_nominal_result` (around line 780). This should use `utils.geo_math.offset_to_latlon`.

## 4. DRY Principle (Don't Repeat Yourself)
- **Violation**: `R_EARTH` is defined in both `utils/map_downloader.py` and `utils/geo_math.py`.
- **Violation**: The Haversine distance function in `ui_qt/sim_controller.py` duplicates functionality that should be handled in `utils/geo_math.py` or by using the ENU coordinate system offsets.

## Proposed Cleanup Plan
1. Centralize physical constants: Move `R_EARTH = 6_378_137.0` to `core/constants.py` and import it in `utils/map_downloader.py` and `utils/geo_math.py`.
2. Refactor `ui_qt/sim_controller.py` to remove the local `get_distance` function and manual coordinate conversions. Instead, use `utils.geo_math.offset_to_latlon` and `utils.geo_math.latlon_to_offset` for converting between Lat/Lon and metric ENU offsets.
3. Ensure explicit unit suffixes for variables representing physical quantities (e.g., `angle` to `angle_rad`).
