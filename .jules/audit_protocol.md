# Codebase Health & Physics Integrity Audit Protocol

This checklist acts as the Standard Operating Procedure (SOP) for the Kazamidori Project's "Codebase Health & Physics Integrity Audit". Run this audit whenever instructed to "Run Health Audit", or automatically prior to major pull requests.

## 1. Physics & Math Validation
- [ ] **Algorithm Review:** Check all algorithms in the `core/` directory for logical correctness and robust bounds checking.
- [ ] **No Magic Numbers:** Flag any undocumented or "magic" numbers hidden within formulas. Constants should be extracted, named meaningfully, and preferably placed at the module level or in a dedicated configuration file.

## 2. Strict Unit Consistency
- [ ] **Explicit Naming Conventions:** Ensure variables representing physical quantities have explicit unit suffixes (e.g., `angle_deg` vs `angle_rad`, `velocity_mps`, `distance_m`).
- [ ] **Conversion Accuracy:** Flag any missing or incorrect mathematical conversions (e.g., passing degrees into `math.cos()` without wrapping in `math.radians()`).
- [ ] **Standard Functions:** Prefer Python's built-in `math.radians()` and `math.degrees()` over manual degree/radian math conversions (e.g., `* (180.0 / math.pi)`).

## 3. Coordinate System Integrity
- [ ] **WGS84 vs. ENU Separation:** Verify strict separation between WGS84 (Latitude/Longitude) and local ENU (East/North/Up in meters) coordinate systems.
- [ ] **UI/Map Standards:** Ensure that the UI and Map components strictly render and compute using ENU coordinates.
- [ ] **Lat/Lon Restrictions:** Ensure WGS84 (Latitude/Longitude) is ONLY used for data loading, initializing environments (e.g., `rocketpy.Environment`), or display as text. No calculations or core logic outside of initialization should use Lat/Lon.

## 4. DRY Principle (Don't Repeat Yourself)
- [ ] **Duplicated Constants:** Scan for and flag constants duplicated across multiple files.
- [ ] **Redundant Definitions:** Identify overlapping helper functions or redundant class definitions across the codebase (e.g., math utilities defined in both `core/` and `utils/`).
- [ ] **Dead Code & Unused Imports:** Flag any unused imports, deprecated functions, or unreachable code for removal.
- [ ] **Architectural Separation:** Verify adherence to the MVVM architecture (e.g., ensure `core/` and `utils/` do not import GUI libraries like PySide6, PyQt, Matplotlib UI, or Folium).
