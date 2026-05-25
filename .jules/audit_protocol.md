# Codebase Health & Physics Integrity Audit Protocol

**Purpose:** Ensure ongoing code health by preventing technical debt, unit mismatch errors, and coordinate system confusion in the Kazamidori Project.

**Execution:** Run this audit protocol before major pull requests or when manually triggered by the user with the prompt "Run Health Audit".

## Checklist

### 1. Physics & Math Validation
- [ ] Check algorithms in `core/` for logical correctness.
- [ ] Flag any magic numbers hidden in formulas. Replace them with named constants.

### 2. Strict Unit Consistency
- [ ] Ensure explicit naming conventions for physical quantities (e.g., `angle_deg` vs `angle_rad`, `velocity_mps`, `length_m`).
- [ ] Flag any missing conversions (e.g., passing degrees into `math.cos()` without `math.radians()`).
- [ ] Ensure standard units are used throughout the simulation and math routines (SI units preferred unless otherwise specified).

### 3. Coordinate System Integrity
- [ ] Verify strict separation between WGS84 (Latitude/Longitude) and local ENU (East/North/Up in meters).
- [ ] Ensure the UI/Map strictly renders in ENU. Lat/Lon should ONLY be used for data loading or text display.
- [ ] Confirm that Navigational Standard math conventions are followed (Azimuth 0° is True North +Y, 90° is East +X, angles increase clockwise). `x = r * math.sin(angle)` (East), `y = r * math.cos(angle)` (North), and azimuth recovery uses `math.atan2(x, y)`.

### 4. DRY Principle (Don't Repeat Yourself)
- [ ] Scan for duplicated constants across multiple files. Centralize them in `core/constants.py` or similar.
- [ ] Identify overlapping helper functions or redundant class definitions (e.g., math utilities defined in both `core/` and `utils/`).
- [ ] Flag unused imports or dead code. Remove them or comment on why they are kept.
