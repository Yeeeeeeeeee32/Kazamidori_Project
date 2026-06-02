# Codebase Health & Physics Integrity Audit Protocol

This checklist serves as the Standard Operating Procedure (SOP) for running a manual or automated health audit on the codebase, specifically targeting physics integrity, math validation, unit consistency, coordinate system rules, and the DRY principle.

## 1. Physics & Math Validation
- [ ] **Algorithm Verification**: Check algorithms in `core/` for logical correctness.
- [ ] **No Magic Numbers**: Flag any magic numbers hidden in formulas; extract them to named constants in `core/constants.py` or use explicitly named local variables.

## 2. Strict Unit Consistency
- [ ] **Explicit Naming Conventions**: Ensure variables are named with explicit units where applicable (e.g., `angle_deg` vs `angle_rad`, `velocity_mps`).
- [ ] **Conversion Safety**: Flag any missing conversions (e.g., passing degrees into trigonometric functions like `math.cos()` without using `math.radians()`).

## 3. Coordinate System Integrity
- [ ] **WGS84 vs. ENU Separation**: Verify strict separation between WGS84 (Latitude/Longitude) and local ENU (East/North/Up in meters).
- [ ] **UI Rendering Compliance**: Ensure the UI and Map views strictly render in ENU coordinates. Latitude/Longitude should only be used for data loading, coordinate transformations, or text display.

## 4. DRY Principle (Don't Repeat Yourself)
- [ ] **Centralized Constants**: Scan for duplicated constants across multiple files. Universal physical and mathematical constants must be strictly centralized in `core/constants.py`.
- [ ] **No Overlapping Helpers**: Identify overlapping helper functions or redundant class definitions (e.g., math utilities defined in both `core/` and `utils/`).
- [ ] **Dead Code & Unused Imports**: Flag unused imports, redundant variables, or dead code blocks.
