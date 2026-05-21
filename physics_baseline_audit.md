# Kazamidori Project - Baseline Audit Report

## 1. Physics & Math Validation
- **Magic Number**: The Earth's radius (`6378137.0` meters) is hardcoded repeatedly. It should be extracted to a centralized constants file or block, e.g., `utils.geo_math.R_EARTH`.
- **Magic Numbers**: WGS-84 metric conversions in `utils/geo_math.py` (`111132.92`, `559.82`, `111412.84`, etc.) lack explanatory context or a clear source reference. While correct, they are technically "magic numbers".
- **Math/Unit Practice**: Widespread use of manual degree/radian conversions like `math.pi * lat / 180.0` or `180.0 / math.pi` rather than using Python's built-in `math.radians()` and `math.degrees()`. This increases the risk of math errors and clutter.

## 2. Strict Unit Consistency
- Generally well-named variables exist (e.g., `radius_m`, `angle_rad`, `lat_deg`).
- Inconsistent usage: In `utils/geo_math.py:circle_polygon`, `angle` is in radians but lacks the `_rad` suffix (unlike `angle_rad` in `ellipse_polygon`).

## 3. Coordinate System Integrity
- The system correctly relies on WGS-84 mapping to ENU, but the map downloader logic (e.g., `utils/map_downloader.py`) duplicates conversion logic instead of reusing `utils/geo_math.py`.

## 4. DRY Principle (Don't Repeat Yourself)
- **Duplicated Coordinate Math**: `utils/map_downloader.py` duplicates the exact `111132.92 - 559.82 * math.cos(...)` polynomial math found in `utils/geo_math.py` (`meters_per_degree`).
- **Duplicated Earth Radius**: `R_EARTH` is defined independently in `utils/geo_math.py` and `utils/map_downloader.py`.

## Proposed Cleanup Plan
1. **Consolidate Constants**: Create a `core/constants.py` (or define at the top of `utils/geo_math.py`) to hold `R_EARTH` and WGS-84 constants.
2. **Refactor Map Downloader**: Update `utils/map_downloader.py` to import and use `meters_per_degree`, `latlon_to_offset`, and `offset_to_latlon` from `utils/geo_math.py` instead of re-implementing them.
3. **Replace Manual Math Conversions**: Traverse `utils/geo_math.py` and `utils/map_downloader.py` to replace `* (math.pi / 180.0)` and `* (180.0 / math.pi)` with `math.radians()` and `math.degrees()`.
4. **Fix Variable Names**: Ensure local angles are explicitly named with `_rad` or `_deg` suffixes (e.g., in `circle_polygon`).
