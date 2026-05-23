# Codebase Health & Physics Integrity Audit

**Trigger:** "Run Health Audit" or before major pull requests.

## Checklist

### 1. Physics & Math Validation
- [ ] Check algorithms in `core/` for logical correctness.
- [ ] Flag any magic numbers hidden in formulas.

### 2. Strict Unit Consistency
- [ ] Ensure explicit naming conventions (e.g., `angle_deg` vs `angle_rad`, `velocity_mps`).
- [ ] Flag any missing conversions (e.g., passing degrees into `math.cos()` without `math.radians()`).

### 3. Coordinate System Integrity
- [ ] Verify strict separation between WGS84 (Latitude/Longitude) and local ENU (East/North/Up in meters).
- [ ] Ensure the UI/Map strictly renders in ENU, and Lat/Lon is only used for data loading or text display.

### 4. DRY Principle (Don't Repeat Yourself)
- [ ] Scan for duplicated constants across multiple files.
- [ ] Identify overlapping helper functions or redundant class definitions (e.g., math utilities defined in both `core/` and `utils/`).
- [ ] Flag unused imports or dead code.
