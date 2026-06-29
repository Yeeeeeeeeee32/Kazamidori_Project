# Kazamidori Project: Codebase Health & Physics Integrity Audit Protocol

## 1. Physics & Math Validation
- Check algorithms in `core/` for logical correctness.
- Flag any magic numbers hidden in formulas (these should be in `core/constants.py`).

## 2. Strict Unit Consistency
- Ensure explicit naming conventions for physical quantities (e.g., `angle_deg` vs `angle_rad`, `velocity_mps`, `length_m`).
- Flag any missing conversions (e.g., passing degrees into `math.cos()` without `math.radians()`).
- Verify math usage, such as using `math.radians()` instead of `* (math.pi / 180.0)`.

## 3. Coordinate System Integrity
- Verify strict separation between WGS84 (Latitude/Longitude) and local ENU (East/North/Up in meters).
- Ensure the UI/Map strictly renders in ENU. Lat/Lon should only be used for data loading or text display.
- Ensure Navigational Standard azimuth logic (0=North=+Y, 90=East=+X) using correct trigonometric mappings (e.g., `x = r * math.sin(angle_rad)` and `y = r * math.cos(angle_rad)`) and `math.atan2(East, North)`.

## 4. DRY Principle (Don't Repeat Yourself)
- Scan for duplicated constants across multiple files.
- Identify overlapping helper functions or redundant class definitions (e.g., math utilities defined in both `core/` and `utils/`).
- Flag unused imports or dead code.

---
**Execution:** Run this checklist whenever the "Run Health Audit" command is given, or automatically before major pull requests.
