# Kazamidori Project: Architecture & Directory Structure Manifest

## 1. Absolute Directives
- **Rule 1 (MVVM Separation):** The `core/` and `utils/` directories MUST NOT contain any GUI libraries (No PySide6, No Matplotlib UI, No Folium). The `ui_qt/` directory handles all views and reacts strictly to `AppState`.
- **Rule 2 (Coordinate System):** ALL internal core calculations must use the local metric ENU coordinate system (X=East, Y=North, Z=Up). Launch pad is (0,0,0). `Lat/Lon` is strictly forbidden in `core/` and is only calculated at the UI mapping layer.
- **Rule 3 (Root Cleanliness):** NO arbitrary Python scripts or temporary profiling files are allowed in the root directory.

## 2. Authorized Directory Structure
Kazamidori_Project/
├── main_qt.py               # The ONLY authorized entry point in the root
├── core/                    # Pure Physics, Math, and Optimization engines
│   ├── constants.py
│   ├── geometry_math.py
│   ├── monte_carlo.py
│   ├── optimization.py
│   ├── simulation.py
│   └── wind_model.py
├── ui_qt/                   # PySide6 Views, State, and Workers
│   ├── init.py
│   ├── app_state.py
│   ├── app_window.py
│   ├── map_view.py
│   ├── sim_controller.py
│   └── workers.py
├── utils/                   # Helpers (Geo Math, Loaders, Session)
│   ├── data_loader.py
│   ├── geo_math.py
│   ├── map_downloader.py
│   └── session_manager.py
├── tests/                   # Official unit/integration tests (PyTest)
├── assets/                  # Static assets (Maps, Images, Icons)
├── .jules/                  # AI Agent configs and manifests
└── archive/                 # Quarantined/Deprecated scripts
