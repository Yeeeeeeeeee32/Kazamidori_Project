# Codebase Health & Physics Integrity Audit Protocol

This checklist defines the Standard Operating Procedure (SOP) for auditing the Kazamidori Project codebase. Run this audit manually when requested ("Run Health Audit") or automatically before major pull requests.

## 1. Physics & Math Validation
- [ ] **Algorithm Correctness:** Check algorithms in `core/` for logical correctness.
- [ ] **No Magic Numbers:** Flag any magic numbers hidden in formulas. Ensure physical and mathematical constants are centralized (e.g., in `core/constants.py`).

## 2. Strict Unit Consistency
- [ ] **Explicit Naming:** Ensure explicit naming conventions are used for variables representing physical quantities (e.g., `angle_deg` vs `angle_rad`, `velocity_mps`).
- [ ] **Valid Conversions:** Flag any missing conversions (e.g., passing degrees into trigonometric functions like `math.cos()` without explicitly using `math.radians()`).

## 3. Coordinate System Integrity
- [ ] **WGS84 vs ENU Separation:** Verify strict separation between WGS84 (Latitude/Longitude) and local ENU (East/North/Up in meters) coordinate systems.
- [ ] **UI/Map Rendering:** Ensure the UI/Map strictly renders in ENU coordinates.
- [ ] **Lat/Lon Usage:** Ensure Latitude/Longitude is strictly only used for data loading or text display. No distance calculations using Lat/Lon directly in `core/` (unless permitted exception for rocketpy init).
- [ ] **Coordinate Math:** Use centralized utilities (e.g., `utils/geo_math.py`) instead of redefining geometric conversions.

## 4. DRY Principle (Don't Repeat Yourself)
- [ ] **Constants:** Scan for duplicated constants across multiple files.
- [ ] **Redundancy:** Identify overlapping helper functions or redundant class definitions (e.g., math utilities defined in both `core/` and `utils/`).
- [ ] **Code Quality:** Flag unused imports or dead code.
