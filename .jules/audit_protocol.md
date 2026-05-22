# Codebase Health & Physics Integrity Audit Protocol

**Purpose:** To prevent technical debt, unit mismatch errors, and coordinate system confusion in the "Kazamidori Project".
**Trigger:** Run automatically before major pull requests or manually when requested with "Run Health Audit".

## 1. Physics & Math Validation
- [ ] **Check algorithms in `core/` for logical correctness.** Are the equations physically consistent?
- [ ] **Flag any magic numbers hidden in formulas.** E.g. Earth's radius (`6378137.0`), gravity (`9.81` instead of `G0`), `math.pi / 180.0` instead of `math.radians()`, etc. Constants must be defined in a dedicated module (e.g., `core/constants.py`) and imported.

## 2. Strict Unit Consistency
- [ ] **Ensure explicit naming conventions.** Suffixes must be used when ambiguous (e.g., `angle_deg` vs `angle_rad`, `velocity_mps`).
- [ ] **Flag any missing conversions.** For instance, ensure `math.cos()` or `math.sin()` are not accidentally passed degrees without `math.radians()`, and `math.degrees()` is used appropriately.
- [ ] **Verify SI Units in Core.** Ensure calculations in `core/` strictly use metric SI units unless explicitly named otherwise.

## 3. Coordinate System Integrity
- [ ] **Verify strict separation between WGS84 and ENU.** Ensure geodetic coordinates (Latitude/Longitude) and local East/North/Up (ENU in meters) are not mixed in calculations.
- [ ] **Check UI/Map Rendering.** Ensure the UI/Map strictly renders in local ENU (East/North), and Lat/Lon is *only* used for data loading or text display.

## 4. DRY Principle (Don't Repeat Yourself)
- [ ] **Scan for duplicated constants across multiple files.** Ensure values like `R_EARTH`, `G0`, or standard conversion factors are not redefined locally.
- [ ] **Identify overlapping helper functions or redundant class definitions.** e.g., coordinate conversion or math utilities defined in both `core/` and `utils/`.
- [ ] **Flag unused imports or dead code.** Keep the codebase clean.
