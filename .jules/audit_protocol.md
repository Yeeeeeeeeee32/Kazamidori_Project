# Codebase Health & Physics Integrity Audit

This protocol defines the standard operating procedure (SOP) checklist for the 'Codebase Health & Physics Integrity Audit' of the Kazamidori Project.

**Trigger:** Manually via "Run Health Audit" or automatically prior to major pull requests.

## Checklist

### 1. Physics & Math Validation
- [ ] **Algorithm Logic**: Check algorithms in `core/` for logical correctness (e.g., rigid body dynamics, aerodynamics, stability limits).
- [ ] **Magic Numbers**: Flag any magic numbers hidden in formulas; extract them to named constants in `core/constants.py` (e.g., `G0 = 9.80665`).
- [ ] **Validity of Physics Conversions**: Verify geometric relationships and assumptions.

### 2. Strict Unit Consistency
- [ ] **Naming Conventions**: Ensure explicit naming conventions with unit suffixes (e.g., `angle_deg` vs `angle_rad`, `velocity_mps`, `length_m`).
- [ ] **Mathematical Functions**: Flag any missing conversions (e.g., passing degrees into `math.cos()` without converting via `math.radians()`).
- [ ] **System of Units**: Enforce strict SI units across all internal calculations (metres, kilograms, seconds, Newtons).

### 3. Coordinate System Integrity
- [ ] **Strict Separation**: Verify strict separation between WGS84 (Latitude/Longitude) and local ENU (East/North/Up in meters).
- [ ] **Metric ENU in Core**: The `core/` directory MUST NOT process or calculate using Lat/Lon directly; all internal trajectory logic uses ENU Cartesian coordinates.
- [ ] **UI Layer Lat/Lon**: Ensure the UI/Map strictly renders using ENU metrics. Lat/Lon should strictly be used for data loading, map initialization, or text displays (e.g., `utils/geo_math.py` conversions).

### 4. DRY Principle (Don't Repeat Yourself)
- [ ] **Constants Duplication**: Scan for duplicated physical/math constants across multiple files.
- [ ] **Redundant Definitions**: Identify overlapping helper functions or redundant class definitions (e.g., math utilities defined in both `core/` and `utils/`).
- [ ] **Dead Code & Imports**: Flag unused imports or dead code.
