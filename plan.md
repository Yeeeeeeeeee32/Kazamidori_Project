1. **Create the Audit Protocol**
   - Save the audit checklist to `.jules/audit_protocol.md`. (Already done, but will state in output).
2. **Execute Baseline Audit and Output Report**
   - The report identified that Earth's radius is hardcoded differently in `utils/map_downloader.py` (6378137.0) and `ui_qt/sim_controller.py` (6371.0).
   - Some missing unit conversions and ambiguous names were flagged.
3. **Propose Cleanup Plan (Do not execute)**
   - Recommend extracting Earth radius `R_EARTH` to `core/constants.py` and standardizing to standard WGS84 radius.
   - Fix unit suffixes (e.g. `speed_mps`, `angle_rad`).
   - Fix missing `math.radians()` instead of custom pi arithmetic.
4. **Complete Pre-commit Steps**
   - Ensure proper testing, verification, review, and reflection are done (though this task does not require modifying code, only analyzing and creating documentation).
