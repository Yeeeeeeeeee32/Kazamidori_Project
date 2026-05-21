# Codebase Health & Physics Integrity Audit Protocol

This document outlines the strict checklist for the Kazamidori Project's codebase health and physics integrity audits.
It must be executed manually when requested, or automatically before major pull requests.

## 1. Physics & Math Validation
- Check algorithms in `core/` for logical correctness.
- Flag any magic numbers hidden in formulas. Ensure they are extracted into well-named constants.

## 2. Strict Unit Consistency
- Ensure explicit naming conventions (e.g., `angle_deg` vs `angle_rad`, `velocity_mps`, `mass_kg`).
- Flag any missing conversions (e.g., passing degrees into `math.cos()` without `math.radians()`, or mismatched metric/imperial conversions).
- Verify standard SI units are used natively within calculation contexts, unless explicitly dealing with UI inputs/outputs.

## 3. Coordinate System Integrity
- Verify strict separation between WGS84 (Latitude/Longitude) and local ENU (East/North/Up in meters).
- Ensure the UI/Map strictly renders in ENU. Lat/Lon is only used for data loading, text display, or initial map center calibration.
- Explicit checks for reverse coordinates or incorrectly swapped X/Y vs Lat/Lon parameters.

## 4. DRY Principle (Don't Repeat Yourself)
- Scan for duplicated constants across multiple files.
- Identify overlapping helper functions or redundant class definitions (e.g., math utilities defined in both `core/` and `utils/`).
- Flag unused imports or dead code across the repository.
