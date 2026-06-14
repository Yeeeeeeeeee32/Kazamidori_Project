# Codebase Health & Physics Integrity Audit Protocol

This protocol serves as the Standard Operating Procedure (SOP) for the "Codebase Health & Physics Integrity Audit". It must be executed whenever a manual "Run Health Audit" is requested, or automatically prior to major pull requests.

## Checklist

### 1. Physics & Math Validation
- [ ] **Algorithm Correctness**: Verify that mathematical operations and physical simulations in `core/` are logically sound and match expected physical behaviors.
- [ ] **No Magic Numbers**: Ensure there are no undocumented or floating numeric constants hidden in formulas. All constants must be clearly defined and centralized where appropriate.

### 2. Strict Unit Consistency
- [ ] **Explicit Naming Conventions**: Verify variables representing physical quantities use explicit unit suffixes (e.g., `angle_deg`, `angle_rad`, `velocity_mps`, `distance_m`).
- [ ] **Missing Conversions**: Check for incorrect unit handling, especially in trigonometric functions (e.g., passing degrees directly to `math.cos()` or `math.sin()` without `math.radians()`, or failing to convert radians back to degrees).

### 3. Coordinate System Integrity
- [ ] **WGS84 vs. ENU Separation**: Ensure strict separation between WGS84 (Latitude/Longitude) and local ENU (East/North/Up in meters) coordinate systems.
- [ ] **UI/Map Rendering**: Verify that UI and Map components strictly render in ENU coordinates. Latitude/Longitude should only be used for data loading or textual display. (Distance calculations in `core/` must strictly use ENU metric coordinates).

### 4. DRY Principle (Don't Repeat Yourself)
- [ ] **Centralized Constants**: Scan for duplicated constants (e.g., `R_EARTH_M`) across multiple files and ensure they are centralized (e.g., in `core/constants.py`).
- [ ] **Helper Overlap**: Identify overlapping helper functions or redundant class definitions (e.g., math utilities defined in both `core/` and `utils/`).
- [ ] **Dead Code**: Flag unused imports, redundant assignments, or dead code segments.
