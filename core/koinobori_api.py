"""
core/koinobori_api.py
Mock interface for the Koinobori atmospheric sensor hardware.
Currently returns dummy data until the hardware integration is complete.
"""

def get_surface_data() -> dict:
    """
    Fetch current atmospheric data from the local Koinobori sensor.
    Returns:
        dict: {
            "pressure_pa": float (e.g. 101325.0),
            "temperature_c": float (e.g. 15.0),
            "humidity": float (0-100 percentage, e.g. 50.0),
            "wind_speed_ms": float,
            "wind_dir_deg": float (0 = North)
        }
    """
    # TODO: Replace with serial/USB or network query to actual Koinobori hardware
    return {
        "pressure_pa": 101325.0, # Standard sea level pressure
        "temperature_c": 15.0,   # 15 deg C
        "humidity": 50.0,        # 50% relative humidity
        "wind_speed_ms": 4.0,    # Mock wind speed
        "wind_dir_deg": 0.0      # Mock wind dir (North)
    }
