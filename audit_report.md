# Baseline Audit Report

## 1. Duplicated Constants (DRY Violation)
- Found Earth radius hardcoded in `utils/map_downloader.py`
- Found Earth radius hardcoded in `ui_qt/sim_controller.py`

## 2. Missing Unit Conversions
- Potential missing conversion in `core/monte_carlo.py` line 107: `(cx + radius * math.sin(step * i),`
- Potential missing conversion in `core/monte_carlo.py` line 108: `cy + radius * math.cos(step * i))`
- Potential missing conversion in `utils/geo_math.py` line 87: `d_lon = (dx / (R_EARTH * math.cos(math.pi * lat / 180.0))) * (180.0 / math.pi)`

## 3. Ambiguous Variable Naming
- Missing unit suffix in `core/wind_model.py` line 128: `speed = math.hypot(u, v)`
- Missing unit suffix in `utils/geo_math.py` line 83: `angle = math.pi * 2 * i / n`
- Missing unit suffix in `ui_qt/sim_controller.py` line 1376: `speed     = max(0.0, base_spd + random.gauss(0.0, base_spd * 0.05 + 0.1))`
