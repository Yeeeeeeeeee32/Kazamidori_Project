# Kazamidori Project - Codebase Health & Physics Integrity Audit

This protocol defines the strict checks performed during a manual "Health Audit" of the Kazamidori Project codebase.

## 1. Physics & Math Validation
*   **Algorithms in `core/`**: Verify that physical models (e.g., rigid body mechanics, aerodynamics, coordinate transformations) are logically sound and mathematically correct.
*   **Magic Numbers**: Scan formulas for undocumented or hardcoded numeric literals. All constants must be extracted to `core/constants.py` or defined explicitly with descriptive comments.

## 2. Strict Unit Consistency
*   **Explicit Naming**: Ensure all variable and parameter names carrying physical quantities include their units (e.g., `angle_deg`, `angle_rad`, `velocity_mps`, `mass_kg`).
*   **Missing Conversions**: Check for logic passing degrees to trigonometric functions expecting radians (e.g., `math.cos()`, `math.sin()`) without explicit conversion (`math.radians()`), or vice-versa (`math.degrees()`).

## 3. Coordinate System Integrity
*   **Separation of WGS84 and ENU**: Ensure strict boundaries between Geodetic coordinates (Latitude/Longitude in WGS84) and Local Cartesian coordinates (East/North/Up in meters).
*   **UI/Map Rendering**: Verify that the Matplotlib/PySide6 map views render **only** using ENU Cartesian coordinates. Lat/Lon should exclusively be used for data ingest, initial origin setup, or text annotations.

## 4. DRY Principle (Don't Repeat Yourself)
*   **Duplicated Constants**: Identify constants defined in multiple places. Ensure single-source-of-truth from `core/constants.py`.
*   **Redundant Helpers/Classes**: Find functions or classes duplicated across modules (e.g., math utilities in both `core/geometry_math.py` and `utils/geo_math.py` that perform identical tasks or violate the `core`/`utils` separation of concerns).
*   **Dead Code**: Flag unused imports, unused local variables, or obsolete functions.

---
*To execute this audit, the Lead QA agent will systematically scan the current working tree against these criteria and produce a Baseline Audit Report.*
