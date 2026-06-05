## Cleanup Proposal

Based on the Baseline Audit Report, here is the proposed (but not yet executed) cleanup plan:

1. **Centralize Constants:**
   - Move `R_EARTH = 6_378_137.0` into `core/constants.py` as `R_EARTH_M = 6_378_137.0`. Update `utils/geo_math.py` and `utils/map_downloader.py` to import and use this.
   - Extract `P0_PA = 101325.0` (Standard sea level pressure) into `core/constants.py` and update `core/simulation.py` and `core/koinobori_api.py`.

2. **Fix Math Conversions:**
   - In `utils/geo_math.py` and `utils/map_downloader.py`, replace instances of `* (180.0 / math.pi)` and `* (math.pi / 180.0)` with explicit calls to `math.degrees()` and `math.radians()`.

3. **Resolve Misplaced Files:**
   - Move `test_physics_core.py` from the root directory into `archive/` or `tests/` depending on whether it's an active test or an orphaned script. (Will move to `archive/` to clean the root as per memory rules).

4. **DRY `wind_model` functions:**
   - Investigate moving `_hellmann_alpha` and `build_wind_profile` currently duplicated in `core/optimization.py` into `core/wind_model.py` so that it's centralized and imported cleanly.

*Note: This cleanup is proposed as a separate follow-up task and will not be executed in this PR.*
