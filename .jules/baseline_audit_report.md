# Baseline Audit Report

## 1. Physics & Math Validation

### Magic Numbers in Formulas
*   `utils/map_downloader.py`: `math.pi / 180.0` and `180.0 / math.pi` are heavily used instead of `math.radians()` and `math.degrees()`. Examples: lines 64, 65, 73, 74, 277, 278.
*   `utils/geo_math.py`: `math.pi / 180.0` and `180.0 / math.pi` are used instead of `math.radians()` and `math.degrees()`. Examples: lines 86, 87.
*   `utils/map_downloader.py` and `utils/geo_math.py`: Hardcoded `R_EARTH` and the magic numbers for WGS-84 `meters_per_degree` approximation.

## 2. Strict Unit Consistency

### Missing/Incorrect Conversions
*   `ui_qt/app_state.py`: Variable names are sometimes missing units. In `app_state.py` line 1041-1042: `live_u = speed * math.sin(math.radians(direction))` and `live_v = speed * math.cos(math.radians(direction))`. Although the math is correct, the parameter names `speed` and `direction` do not indicate units (e.g. `speed_mps`, `direction_deg`).

## 3. Coordinate System Integrity

### Separation between WGS84 and ENU
*   The system largely appears to adhere to the separation correctly, performing heavy simulation in ENU coordinates within `core/` modules (e.g., `monte_carlo.py` explicitly states it has zero geographic coordinate logic).

## 4. DRY Principle (Don't Repeat Yourself)

### Duplicated Constants
*   `R_EARTH` is duplicated in multiple places:
    *   `utils/map_downloader.py`: `R_EARTH = 6378137.0` (line 55)
    *   `utils/geo_math.py`: `R_EARTH = 6_378_137.0` (line 80)
    It should be consolidated in a constants file like `core/constants.py` or similar.

### Duplicated Logic
*   `meters_per_degree` calculation logic is duplicated.
    *   `utils/geo_math.py` has a dedicated `meters_per_degree` function using constants like `111132.92`, `559.82`, etc.
    *   `utils/map_downloader.py` (lines 207-208) implements the *exact same formula* manually.
    *   `utils/map_downloader.py` should import and use `meters_per_degree` from `utils/geo_math.py`.
