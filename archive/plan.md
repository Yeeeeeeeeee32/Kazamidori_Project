1. **Change Widget Types in `ui_qt/app_window.py`**:
   - For all Airframe configuration properties (`af_mass_input`, `af_cg_input`, `af_len_input`, `af_radius_input`, `af_nose_input`, `af_finroot_input`, `af_fintip_input`, `af_finspan_input`, `af_finpos_input`, `af_motorpos_input`, `af_motormass_input`, `af_backfire_input`, `para_cd_input`, `para_area_input`, `para_lag_input`), replace `QDoubleSpinBox` with `QLineEdit`.
   - Apply a `QDoubleValidator` to each `QLineEdit`.
   - Set placeholders (e.g., `"入力必須"`) to guide the user.
   - Initialize the `text` of the fields to empty strings `""`.
   - Update `_evaluate_config_deltas` and `_on_manual_config_reset` to handle text extraction instead of `.value()`, and handle parsing as floats.

2. **Update Bindings in `ui_qt/sim_controller.py`**:
   - In `_bind` function inside `SimController.bind_app_state`, handle `QLineEdit` specifically for string to float conversion with `""` -> `None`.
   - Change connection to `textChanged.connect()`.
   - Reverse update mapping: when the application state becomes `None`, set `widget.setText("")`. Otherwise, use `widget.setText(str(_g(v)))`.

3. **Simulation Guard in `core/simulation.py` & `ui_qt/sim_controller.py`**:
   - The validation in `_validate_run_prerequisites` correctly uses `None` for internal data types but currently guards against `-9999.0` for `launch_lat`, `launch_lon`, and `azim_input`. For `azim_input`, it's not part of the airframe, but we should double check if the prompt requires removing `-9999.0` entirely. The prompt strictly limits scope: `The initial state of these parameter fields must be a GENUINE blank/empty state, not a masked dummy number. Refactor the input components in the Airframe / Rocket Configuration section so they initialize as completely empty fields`
   - Only modify the Airframe properties. Check that `_validate_run_prerequisites` correctly spots missing variables as `None`.
   - Update `missing.append` logic for `None` properties if it doesn't already catch them. (It already uses `if any(v is None for v in (...))`).

4. **Verify TypeErrors**:
   - Execute tests/run tests.

5. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**
