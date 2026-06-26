# Baseline Audit Report

## 1. Physics & Math Validation
- **Magic Numbers**:
  - `180.0 / math.pi` and `math.pi / 180.0` are heavily used manually in `utils/map_downloader.py` and `utils/geo_math.py`. These should use `math.degrees()` and `math.radians()`.
  - `R_EARTH = 6_378_137.0` is hardcoded as a local variable/constant in `utils/geo_math.py` and `utils/map_downloader.py`.

## 2. Strict Unit Consistency
- **Unit Conversions**:
  - Manual conversion from radians to degrees (`* (180.0 / math.pi)`) and vice versa (`* (math.pi / 180.0)`) is used instead of the explicit and safer Python `math.degrees()` and `math.radians()`.
  - In `utils/geo_math.py` line 87: `math.cos(math.pi * lat / 180.0)` is missing an explicit `math.radians()` conversion and instead uses manual math.

## 3. Coordinate System Integrity
- Passes. There is no domain leakage of `lat` or `lon` into the math processing `core/` files outside of the allowed usage in `simulation.py` for `RocketPy`.

## 4. DRY Principle (Don't Repeat Yourself)
- **Duplicated Constants**:
  - `R_EARTH` is duplicated in `utils/geo_math.py` and `utils/map_downloader.py`. It should be centralized in `core/constants.py`.

## Proposed Cleanup Plan (Do Not Execute Yet):
1. Extract `R_EARTH` to `core/constants.py` and import it into `utils/geo_math.py` and `utils/map_downloader.py`.
2. Refactor all manual radian/degree conversions (`* (180.0 / math.pi)` and `* (math.pi / 180.0)`) to use `math.degrees()` and `math.radians()` across `utils/geo_math.py` and `utils/map_downloader.py`.
