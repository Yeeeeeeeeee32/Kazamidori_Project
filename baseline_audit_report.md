## Baseline Audit Report

### 1. Physics & Math Validation
- **Missing Constants:** The value `101325.0` (Standard sea level pressure) is hardcoded in `core/simulation.py` (`p0 = params.get('env_pressure', 101325.0)`). It should be defined in `core/constants.py`.
- **Duplicate Logic:** The `build_wind_profile` and `_hellmann_alpha` logic in `core/optimization.py` appears to be a redundant copy (comment explicitly states it is copied from `main.py` / `WindProfileBuilder` to avoid circular dependencies). This violates the DRY principle and should be centralized in `utils/` or `core/wind_model.py`.

### 2. Strict Unit Consistency
- **Magic Conversions:** In `utils/geo_math.py` and `utils/map_downloader.py`, manual conversions `* (180.0 / math.pi)` and `* (math.pi / 180.0)` are used. These must be replaced with `math.degrees()` and `math.radians()`.
- **Missing Unit Suffixes:** Some variables could be clearer, but primarily the math functions violate the SOP.

### 3. Coordinate System Integrity
- **Map View ENU Usage:** Checked `ui_qt/map_view.py`. It correctly uses the ENU system (e.g., `impact_x`, `impact_y` directly on axes, no geographic conversion within the rendering path itself except for the required exceptions).

### 4. DRY Principle (Don't Repeat Yourself)
- **Misplaced Test Files:** `test_physics_core.py` is floating in the root directory. Tests must strictly be placed inside the `tests/` directory.
- **Constant Duplication:** `R_EARTH = 6_378_137.0` is redefined in `utils/geo_math.py` and `utils/map_downloader.py`. This must be extracted to `core/constants.py` as `R_EARTH_M = 6_378_137.0`.
