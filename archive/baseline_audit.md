# Kazamidori Project - Baseline Audit Report

## 1. Physics & Math Validation
**Violations:**
- `core/monte_carlo.py`: Several magic numbers used directly in calculations without being defined as constants.
  - `4.605` used for Chi-squared upper quantile. (Wait, `4.605` is in `constants.py` as `CHI2_90`, but `core/monte_carlo.py` might be using magic numbers for other quantiles like `3.219`, `9.210`, etc.).
  - `core/optimization.py`: Lots of un-extracted magic numbers for weighting and progress (`0.6`, `0.75`, `0.9`).
  - `core/geometry_math.py`: Inertia calculations use `12.0`, `3.0`, `4.0`, `18.0` without context in some places, but these are standard geometrical constants (e.g. `mr^2 / 4 + mh^2 / 12`).

## 2. Strict Unit Consistency
**Violations:**
- Found potential degree/radian unit mismatches:
  - `ui_qt/sim_controller.py`: Check if `math.atan2` results are correctly converted to degrees if expected.
  - Variable naming in `core/simulation.py` could be more explicit (e.g., `isp_s`, `power_on_cd` are okay, but `launch_lat`, `launch_lon` should maybe be `launch_lat_deg`).

## 3. Coordinate System Integrity
**Violations:**
- `math.atan2(y, x)` vs `math.atan2(x, y)`:
  - Navigational standard dictates `math.atan2(x, y)` where `x` is East and `y` is North.
  - `core/optimization.py` uses `math.atan2(eig_vc[1, 1], eig_vc[0, 1])`. Need to verify if the matrix is `[x, y]` or `[y, x]`. If it's `[x, y]`, then `atan2(y, x)` is standard math, which might be a violation of the Navigational Standard.
  - `core/monte_carlo.py` line 309 uses `math.atan2(major_vec[0], major_vec[1])`. If `major_vec[0]` is East (X) and `major_vec[1]` is North (Y), this correctly implements `atan2(x, y)`.

## 4. DRY Principle (Don't Repeat Yourself)
**Violations:**
- Duplicated Constants:
  - `utils/map_downloader.py`: Hardcodes `R_EARTH = 6378137.0` (WGS84 Earth Radius) on line 55.
  - `ui_qt/sim_controller.py`: Hardcodes `R = 6371.0` (Earth Radius in km) on line 1450.
  These should be centralized in `core/constants.py`.
  - `utils/geo_math.py` or similar might need Earth radius too.

## Proposed Cleanup Plan
1. Move `R_EARTH = 6378137.0` to `core/constants.py` and update `utils/map_downloader.py` to import it.
2. Review `math.atan2` usages in `core/optimization.py` to ensure they follow the Navigational Standard.
3. Review `core/monte_carlo.py` and `core/optimization.py` for magic numbers and extract them to named constants if they represent physical/statistical parameters.
