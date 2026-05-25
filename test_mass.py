import sys
from core.simulation import simulate_once
import numpy as np

params = {
    'airframe_mass': 1.0,
    'airframe_len': 1.0,
    'radius': 0.1,
    'airframe_cg': 0.5,
    'nose_len': 0.2,
    'fin_root': 0.1,
    'fin_tip': 0.05,
    'fin_span': 0.1,
    'fin_pos': 0.9,
    'motor_pos': 1.0,
    'motor_dry_mass': 0.2,
    'backfire_delay': 2.0,
    'para_cd': 1.0,
    'para_area': 1.0,
    'para_lag': 1.0,
    'rail': 2.0,
    'wind_u_prof': np.zeros((2, 2)),
    'wind_v_prof': np.zeros((2, 2)),
    'thrust_data': [[0, 100], [1, 100], [2, 0]],
    'motor_burn_time': 2.0,
    'power_on_cd': 0.45,
    'power_off_cd': 0.40,
}

simulate_once(90.0, 0.0, params)
