# Codebase Health & Physics Integrity Audit Protocol

This Standard Operating Procedure (SOP) defines the checklist for maintaining codebase health and physics integrity for the Kazamidori Project.

## 1. Physics & Math Validation
- Check algorithms in `core/` for logical correctness.
- Flag any magic numbers hidden in formulas. Ensure physical constants (e.g., Earth's radius, gravity) use `core.constants`.

## 2. Strict Unit Consistency
- Ensure explicit naming conventions (e.g., `angle_deg` vs `angle_rad`, `velocity_mps`).
- Flag any missing conversions (e.g., passing degrees into `math.cos()` without `math.radians()`).

## 3. Coordinate System Integrity
- Verify strict separation between WGS84 (Latitude/Longitude) and local ENU (East/North/Up in meters).
- Ensure the UI/Map strictly renders in ENU, and Lat/Lon is only used for data loading or text display. Distance calculations within the `core/` module and UI components must strictly use the ENU (X,Y) metric coordinate system. Direct distance calculations using Latitude/Longitude are forbidden.
- For coordinate and distance calculations (such as replacing manual Haversine formulas), strictly use the centralized utilities in `utils/geo_math.py` (e.g., `latlon_to_offset`, `offset_to_latlon`) instead of redefining geometric conversions in UI controllers.

## 4. DRY Principle (Don't Repeat Yourself)
- Scan for duplicated constants across multiple files.
- Identify overlapping helper functions or redundant class definitions (e.g., math utilities defined in both `core/` and `utils/`).
- Flag unused imports or dead code.
