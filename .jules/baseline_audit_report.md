# Baseline Audit Report

## 1. Physics & Math Validation

### Magic Numbers in Formulas
*   `utils/map_downloader.py`: `math.pi / 180.0` and `180.0 / math.pi` are heavily used instead of `math.radians()` and `math.degrees()`.
*   `utils/geo_math.py`: `math.pi / 180.0` and `180.0 / math.pi` are used instead of `math.radians()` and `math.degrees()`.
*   `utils/map_downloader.py` and `utils/geo_math.py`: Hardcoded `R_EARTH` (6378137.0).

## 2. Strict Unit Consistency

### Missing/Incorrect Conversions
*   `ui_qt/app_state.py`: Variable names are missing explicit units. Specifically, `speed` and `direction` in `check_tolerance` do not indicate units (e.g., `speed_mps`, `direction_deg`).

## 3. Coordinate System Integrity

### Separation between WGS84 and ENU
*   The system largely appears to adhere to the separation correctly, performing heavy simulation in ENU coordinates within `core/` modules (e.g., `monte_carlo.py` explicitly states it has zero geographic coordinate logic).

## 4. DRY Principle (Don't Repeat Yourself)

### Duplicated Constants
*   `R_EARTH` is duplicated in multiple places:
    *   `utils/map_downloader.py`: `R_EARTH = 6378137.0`
    *   `utils/geo_math.py`: `R_EARTH = 6_378_137.0`

### Proposed Cleanup Plan (Not to be executed yet)
1. **Extract `R_EARTH`:** Add `R_EARTH_M = 6378137.0` to `core/constants.py` and import it into `utils/map_downloader.py` and `utils/geo_math.py`.
2. **Fix Magic Numbers:** Replace `180.0 / math.pi` and `math.pi / 180.0` in `utils/map_downloader.py` and `utils/geo_math.py` with `math.degrees()` and `math.radians()`.
3. **Fix Explicit Unit Naming:** In `ui_qt/app_state.py`, rename the `speed` and `direction` parameters in the `check_tolerance` function to `speed_mps` and `direction_deg` respectively.
