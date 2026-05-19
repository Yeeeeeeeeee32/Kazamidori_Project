# Comprehensive System Audit & UX Report: Kazamidori Project

## 1. Architectural Disconnects (Data Pipeline)

### 1.1 The "Manual Config" Breakdown
**Diagnosis:** The Manual Configuration dialog (`ManualSetupDialog`) in `ui_qt/app_window.py` allows users to input raw airframe values. However, the data flow is broken because the `original_rocket_config` attribute in `AppState` is either unpopulated or improperly managed.
- The `_evaluate_config_deltas` method attempts to compare current spinbox values against `self.state.original_rocket_config` to highlight changed fields in red. Because the reference state is missing or stale, it silently fails.
- The `_on_manual_config_reset` method attempts to read from this same dictionary to push values back into the widgets, failing for the same reason.
- Furthermore, while `SimController._wire_airframe_spinboxes` correctly binds the main window widgets to `AppState`, the modal dialog inputs are completely decoupled from this pipeline unless specifically mapped.

### 1.2 Destructive `.rkt` Loading
**Diagnosis:** The `.rkt` parser function `_on_load_rkt` inside `ui_qt/sim_controller.py` bypasses the initial empty states designed for safety.
- **Backfire Delay:** The parser extracts `"backfire_delay"` from the airframe definition and unconditionally injects it (`s.backfire_delay = af["backfire_delay"]`). This violates the core requirement that `backfire_delay` must strictly remain a manual, real-time input.
- **Motor & Parachute:** While the current parser primarily targets airframe mass and geometry, it lacks explicit safeguards. Any automatic loading mechanism must actively block overwriting parachute (Cd, area, lag) and motor configurations. If a `.rkt` file attempts to provide these, the UI must intercept it, discard those specific values, and issue a warning instructing the user to configure them manually.

### 1.3 Telemetry & Results Data Mismatch
**Diagnosis:** The "Simulation Results (諸元)" panel displays "-" because the data keys in the `nominal_data` and `mc_data` dictionaries do not match what `SimController.populate_results` expects.
- **Key Mismatches:** For instance, the UI requests `mc_data.get("mc_avg_apogee")`, but the Monte Carlo worker likely outputs a differently named key (e.g., `avg_apogee_m`). Similarly, `nominal_data.get("elev")` or `score` might be missing or nested differently depending on the flight mode dictionary schema.
- **Result Visibility Toggle:** `_update_results_layout` in `ui_qt/app_window.py` correctly toggles the *visibility* of these labels, but because the data binding fails silently (`set_val` falls back to `"-"`), the user perceives a broken UI.

---

## 2. UX & Interaction Flaws

### 2.1 2D Map Interactions
**Diagnosis & GIS Recommendation:**
- **Current State:** `ui_qt/map_view.py` uses plain Drag for panning, Mouse Scroll for zoom, and `Shift+Drag` for relocating the launch site.
- **UX Conflict & Resolution:** Standard GIS applications (like QGIS or ArcGIS) traditionally reserve `Shift+Drag` for bounding-box (marquee) zooming. Using it to relocate a highly sensitive element like the Launch Site is risky and unconventional.
- **Recommendation:**
    - **Pan:** Plain `Left-Click + Drag`.
    - **Zoom:** `Scroll` (centered strictly on the mouse cursor using inverted affine transforms, not a generic axes multiplier).
    - **Launch Site Relocation:** **`Ctrl+Click + Drag`** (or `Alt+Drag`). This requires a deliberate, secondary modifier, preventing accidental relocations during intense navigation.
- **Coordinate Sync Desync:** When the launch site is dragged, `MapView` updates visually but fails to push the new projected Lat/Lon back into `AppState.launch_lat/lon`. Because the state isn't updated, the UI coordinate input boxes (`AppWindow.lat_input`) remain frozen.
- **UI Blocking:** Redrawing a ghost marker on every `motion_notify_event` stresses the main thread. We must rely on `set_data()` without full canvas redraws, or execute coordinate state mutations *strictly* on `button_release_event` to prevent layout update loops.

### 2.2 3D Map Visuals & Controls
**Diagnosis:**
- **Graph Title:** The `ax.set_title(...)` is hardcoded in `update_profile_plot` and creates visual clutter. It should be removed.
- **Camera Azimuth Constraint:** The `_azim_slider` is locked between -90 and 90 degrees. To unrestrict this, we need to allow a full 360° rotation (e.g., -180 to 180) and add a dedicated "Reset to True North" (Azimuth = -90) button in the dock.
- **Auto-Zoom Flattening:** The `_equalise_3d_axes` function in `ui_qt/app_window.py` computes a maximum radius across all axes (`[xlim, ylim]`). If a Monte Carlo simulation scatters rockets 500m horizontally, the Z-axis (Altitude) is forced to scale proportionally. If the apogee is only 100m, the vertical trajectory is visually flattened. The Z-axis must be explicitly excluded from equalization bounds.
- **Time-Seek UI Block:** The `ProfileCanvas` handles `Shift+Scroll` to scrub through time (`set_trajectory`). Re-rendering a 3D marker on every scroll tick can block the Qt Event loop. It must use `set_data()` and `set_3d_properties()` on a persistent `Line3D` object rather than clearing or recreating artists.

### 2.3 Wind Graph Telemetry
**Diagnosis:**
- The Continuous Wind Speed graph plots both a line layer and a scatter layer. While the logic attempts to filter redundant timestamps (`fetch_xs`), it still iterates over the 1Hz history buffer inefficiently. It must plot the continuous line using `linestyle='-'` for Zero-Order Hold (ZOH) representation, and overlay dots `marker='o'` *only* at the precise timestamps where the raw data changed.

---

## 3. Redundancy & Clutter

- **Duplicate Coordinate Display:** The 2D map canvas features raw coordinate text drawn directly on the axes (`MapView._render_current_state`), duplicating the explicit "Launch Coordinates" QSpinBoxes in the side panel. The canvas text should be stripped.
- **State Properties:** The lightweight local `self.state` vs global `AppState` paradigm causes confusion in `update_wind_plot` (`hasattr(self.state, "surf_wind_speed")` fallback). This redundancy should be cleaned up; `AppWindow` should bind the global state earlier or rely entirely on safe getter methods.

---

## 4. Phased Execution Plan

To execute these fixes safely without causing regressions, we will follow a strict, 3-phase roadmap.

### Phase 1: Data Pipeline & State Binding (The Core)
1. **Fix `.rkt` Loader (`ui_qt/sim_controller.py`):**
   - Strip `backfire_delay` from the payload injection.
   - Add explicit safeguards preventing modification of `parachute_cd`, `parachute_area`, `parachute_lag`, and motor parameters.
   - Add warning `QMessageBox` if unsupported fields exist in the `.rkt`.
2. **Fix Manual Config (`ui_qt/app_window.py`):**
   - Implement correct syncing of `original_rocket_config` inside `AppState` when initial files are loaded.
   - Repair `_evaluate_config_deltas` and `_on_manual_config_reset` to correctly read/write from this state.
3. **Fix Results Dictionary Keys (`ui_qt/sim_controller.py`):**
   - Audit the worker output schemas vs `populate_results` expected keys and correct the mappings so that Free, Precision, Altitude, and Winged modes populate accurately.

### Phase 2: 2D Map Interactions & UX (The Control Surface)
1. **Implement `Ctrl+Drag` Relocation (`ui_qt/map_view.py`):**
   - Refactor `_on_button_press` and `_on_motion_notify` to use `event.key == 'control'` for relocation.
   - Bind `button_release_event` to fire coordinate translation via `geo_math.offset_to_latlon` and strictly push to `AppState.launch_lat/lon`.
2. **Refine Zoom & Pan (`ui_qt/map_view.py`):**
   - Fix mouse-scroll zoom to lock onto the cursor coordinate (`event.xdata/ydata`).
   - Add "Reset View" auto-zoom function.
3. **Remove Redundancy (`ui_qt/map_view.py`):**
   - Delete the duplicate coordinate text artist from the canvas drawing logic.

### Phase 3: 3D Visuals & Telemetry (The Polish)
1. **Fix 3D Camera & Title (`ui_qt/app_window.py`):**
   - Remove `ax.set_title` from `update_profile_plot`.
   - Update `_azim_slider` range to [-180, 180] and wire a "Reset to True North (-90°)" button.
2. **Fix 3D Auto-Zoom Flattening (`ui_qt/app_window.py`):**
   - Modify `_equalise_3d_axes` to calculate bounds only on X and Y, manually setting Z limits based purely on the `apogee_m` + 15% padding.
3. **Refine Wind Telemetry Graph (`ui_qt/app_window.py`):**
   - Finalize the `update_wind_plot` line/scatter separation logic to ensure only delta-timestamps render a point.
4. **Optimize Time-Seek (`ui_qt/app_window.py`):**
   - Verify `_update_marker` strictly uses `set_data` and `set_3d_properties` without artist recreation.