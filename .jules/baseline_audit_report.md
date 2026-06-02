# Baseline Audit Report

## 1. Physics & Math Validation
- **Magic Numbers in Conversions**: `utils/map_downloader.py` and `utils/geo_math.py` manually compute degree/radian conversions using `* (180.0 / math.pi)` and `* (math.pi / 180.0)`. These violate our SOP which mandates the use of Python's built-in `math.degrees()` and `math.radians()`.
- **Magic Numbers for Earth Radius**: The WGS-84 Earth Radius (`6378137.0`) is hardcoded independently in `utils/map_downloader.py` (lines 41, 50) and `utils/geo_math.py` (line 80).
- **Magic Numbers for WGS-84 Approximation**: `utils/geo_math.py` uses raw literal coefficients (`111132.92`, `559.82`, `111412.84`, etc.) in `meters_per_degree` which lack explanatory variables but are functionally correct approximations.

## 2. Strict Unit Consistency
- No glaring unit violations (e.g. mismatched passing of angles to trig functions without conversion) were discovered in `core/simulation.py` or `core/geometry_math.py` directly, as explicit naming (e.g., `angle_deg`, `angle_rad`, `radius_m`) seems generally well maintained in helpers.

## 3. Coordinate System Integrity
- The system correctly isolates Lat/Lon to ENU conversion via `meters_per_degree` and bounding/polygon helpers. No instances were found where Lat/Lon is mistakenly used to calculate scalar distances like `math.hypot(x, y)` in `core/`.

## 4. DRY Principle (Don't Repeat Yourself)
- **Duplicated Constants**: `R_EARTH` is duplicated across multiple modules.
- **Overlapping Functions**: `utils/map_downloader.py` duplicates geodetic-to-ENU coordinate conversions (`latlon_to_enu`, `enu_to_latlon`) instead of reusing `latlon_to_offset` and `offset_to_latlon` from `utils/geo_math.py`.

## Proposed Cleanup Plan (Do Not Execute Yet)
1. **Centralize `R_EARTH`**: Add `R_EARTH_M = 6_378_137.0` to `core/constants.py`.
2. **Remove Redundant Conversions**: Update `utils/geo_math.py` to import `R_EARTH_M` from `core/constants.py` and use `math.radians()` and `math.degrees()`.
3. **Refactor Map Downloader**: Update `utils/map_downloader.py` to import `R_EARTH_M` from `core/constants.py`, use `math.radians()` and `math.degrees()`, and ideally replace `latlon_to_enu`/`enu_to_latlon` with the canonical `latlon_to_offset`/`offset_to_latlon` from `utils/geo_math.py`.