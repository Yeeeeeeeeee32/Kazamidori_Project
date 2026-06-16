# Baseline Audit Report

This report outlines the violations found during the initial audit of the Kazamidori codebase, based on the `audit_protocol.md`.

## 1. Physics & Math Validation
- **Bug in `ellipse_polygon` (core/optimization.py, line 649):**
  - **Issue:** The calculation for `ye` incorrectly uses `math.sin(t) * ca` for the second term, marked with `# BUG-01 FIX: sin(t), not cos(t)`. However, the mathematical definition of standard 2D rotation of an ellipse $x_e = a \cos t$, $y_e = b \sin t$ by angle $\alpha$ is:
    $x' = x_e \cos \alpha - y_e \sin \alpha = a \cos t \cos \alpha - b \sin t \sin \alpha$
    $y' = x_e \sin \alpha + y_e \cos \alpha = a \cos t \sin \alpha + b \sin t \cos \alpha$
    So `ye = a * math.cos(t) * sa + b * math.sin(t) * ca` is actually mathematically correct. However, `core/monte_carlo.py` and `utils/geo_math.py` have their own ellipse/circle definitions. There is code duplication across files.

## 2. Strict Unit Consistency
- **Variable naming and degrees vs. radians:** In several places like `utils/map_downloader.py`, formulas directly use `180.0 / math.pi` and `math.pi / 180.0` instead of `math.degrees()` and `math.radians()`, violating memory rules.

## 3. Coordinate System Integrity
- **Distance calculation:** The rule "Distance calculations within the `core/` module must strictly use the ENU (X,Y) metric coordinate system (e.g., `math.hypot(impact_x, impact_y)`)" seems to be followed.
- The rule "For coordinate and distance calculations ... strictly use the centralized utilities in `utils/geo_math.py`" should be checked.

## 4. DRY Principle (Don't Repeat Yourself)
- **Duplicated Constants:** `R_EARTH = 6_378_137.0` is hardcoded in both `utils/geo_math.py` and `utils/map_downloader.py`. It should be centralized in `core/constants.py`.
- **Duplicated Math Formulas:** Both `utils/geo_math.py` and `core/optimization.py` have ellipse and circle logic.

## Proposed Cleanup Plan (Do not execute yet)
1. Move `R_EARTH = 6_378_137.0` to `core/constants.py` as `R_EARTH_M = 6_378_137.0`.
2. Update `utils/geo_math.py` and `utils/map_downloader.py` to import and use `R_EARTH_M` from `core/constants.py`.
3. Replace manual degree/radian conversions (e.g. `* 180.0 / math.pi`) with `math.degrees()` and `math.radians()` in `utils/map_downloader.py` and `utils/geo_math.py`.
4. Ensure `core/optimization.py` and `utils/geo_math.py` share ellipse generation code, potentially moving it strictly to `core/geometry_math.py` or removing duplication.
