# Kazamidori Project: Baseline Code Health & Physics Integrity Audit Report

## 1. Physics & Math Validation
- **Status:** **PASS / MINOR ISSUES**
- `core/optimization.py` line 649 explicitly states a previous bug fix `BUG-01 FIX: sin(t), not cos(t)`, but the trig mapping appears logically sound for error ellipses.
- `utils/data_loader.py` line 505: `area = _math.pi * (dia_mm / 2000.0) ** 2` - This calculates the area correctly but contains magic numbers (`2000.0` instead of `/ 1000.0 / 2.0`).

## 2. Strict Unit Consistency
- **Status:** **FAIL**
- `ui_qt/app_state.py` lines 1190-1191 uses `speed` and `direction` without unit suffixes (e.g., `speed_mps`, `direction_deg`). The variables `live_u` and `live_v` lack units as well.
- Manual radian/degree conversions still exist in multiple places:
  - `utils/map_downloader.py`: `(180.0 / math.pi)` and `(math.pi / 180.0)` are heavily used in lines 132, 133, 141, 142.
  - `utils/geo_math.py`: `(180.0 / math.pi)` is used in lines 86, 87.
  - `core/geo_math.py`: Missing use of `math.radians()` in coordinate translation.

## 3. Coordinate System Integrity
- **Status:** **FAIL**
- `core/wind_model.py` and `ui_qt/app_state.py` construct ENU coordinates using `u = speed * math.sin(math.radians(dir))` and `v = speed * math.cos(math.radians(dir))`. But the standard Navigational coordinates specify +Y is North, +X is East, meaning `x = r * sin(angle)` and `y = r * cos(angle)`. The current usage is technically correct for the wind vector if direction is "where it's coming from", but wait, `ui_qt/sim_controller.py` lines 873-874 has:
  ```python
  mu_u = surf_spd * math.sin(math.radians(surf_dir))
  mu_v = surf_spd * math.cos(math.radians(surf_dir))
  ```
  Is this correct based on ENU standards? Yes, standard Navigational (0=North=+Y, 90=East=+X) uses `X = sin(theta)` and `Y = cos(theta)`. The usage looks consistent.
- `ui_qt/sim_controller.py` contains Lat/Lon to ENU math directly inside the UI layer (lines 1579-1581 have Haversine/Distance logic). This is an MVVM violation. The math belongs in `utils/geo_math.py` or `core/`.

## 4. DRY Principle (Don't Repeat Yourself)
- **Status:** **FAIL**
- Overlapping logic: Distance calculations like Haversine are potentially duplicated between `ui_qt/sim_controller.py` and `utils/geo_math.py`.
- Redundant math imports and aliases: `_math.pi` is used in `utils/data_loader.py`.
- Earth's radius (`R_EARTH`) is likely redefined multiple times in `utils/geo_math.py` and `utils/map_downloader.py` instead of being imported from `core/constants.py` (which currently doesn't define it).

## Proposed Cleanup Plan (Pending Approval)
1. **Move Constants:** Add `R_EARTH = 6371000.0` to `core/constants.py` and update `utils/geo_math.py` and `utils/map_downloader.py` to use it.
2. **Refactor Math:** Replace all instances of `* (180.0 / math.pi)` and `* (math.pi / 180.0)` with `math.degrees()` and `math.radians()`.
3. **Variable Naming:** Rename variables in `ui_qt/app_state.py` (e.g., `speed` -> `speed_mps`, `direction` -> `direction_deg`).
4. **Architectural Fix:** Move Haversine/distance math out of `ui_qt/sim_controller.py` and into `utils/geo_math.py`.
