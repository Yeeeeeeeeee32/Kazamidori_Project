# Baseline Audit Report

## 1. Physics & Math Validation
* **Flagged Magic Numbers & Formulas:**
    * In `utils/map_downloader.py`: Usage of `(180.0 / math.pi)` and `(math.pi / 180.0)` for degree/radian conversions instead of explicitly using `math.radians()` and `math.degrees()`.
    * In `utils/geo_math.py`: Same usage of `(180.0 / math.pi)` for degree conversions instead of explicit math library functions.
    * In `ui_qt/sim_controller.py`: A manual Haversine formula (`get_distance`) is defined directly in the UI controller instead of using existing generic or ENU conversions.

## 2. Strict Unit Consistency
* **Missing Conversions:**
    * In `ui_qt/sim_controller.py` inside `get_distance`: `math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2` does not explicitly declare the units of variables or outputs.
* **Naming Conventions:**
    * Many variables representing angles or distances don't strictly have `_deg`, `_rad`, `_m` or similar suffixes (e.g., `speed` and `direction` in `app_state.py` without units).

## 3. Coordinate System Integrity
* **Coordinate Standard Violation:**
    * In `ui_qt/sim_controller.py` (lines 1577-1583): Direct calculation of distance using Latitude/Longitude via a Haversine formula. The protocol explicitly forbids direct Haversine calculations outside of generic loading, and demands using ENU conversions (e.g. `latlon_to_offset`) for physical distances.

## 4. DRY Principle (Don't Repeat Yourself)
* **Duplicated Constants:**
    * `R_EARTH = 6_378_137.0` is hardcoded separately in both `utils/geo_math.py` and `utils/map_downloader.py`. It should be centralized in `core/constants.py`.
* **Overlapping Functions / Redundancy:**
    * The manual `get_distance(lat1, lon1, lat2, lon2)` function in `ui_qt/sim_controller.py` can be removed in favor of using `latlon_to_offset` to calculate East/North offset, then `math.hypot(dx, dy)`.
    * `meters_per_degree` approximation logic is redundantly implemented inline in `utils/map_downloader.py` (lines 207-208) despite being defined in `utils/geo_math.py`.
