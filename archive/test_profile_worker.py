import cProfile
import pstats
from ui_qt.workers import _mc_worker_task
import random

sim_params = {
    "elev": 100,
    "azi": 0,
    "airframe_mass": 10,
    "airframe_len": 2,
    "radius": 0.1,
    "airframe_cg": 1,
    "nose_len": 0.5,
    "fin_root": 0.2,
    "fin_tip": 0.1,
    "fin_span": 0.15,
    "fin_pos": 1.8,
    "motor_pos": 2,
    "motor_dry_mass": 1,
    "backfire_delay": 0,
    "para_cd": 1.5,
    "para_area": 2,
    "para_lag": 1,
    "rail": 5,
    "wind_u_prof": [(float(i), 10.0) for i in range(100)],
    "wind_v_prof": [(float(i), 5.0) for i in range(100)],
    "thrust_data": [[0, 1000], [1, 1000], [2, 0]],
    "motor_burn_time": 2,
    "launch_lat": 35,
    "launch_lon": 135,
}

def run():
    for i in range(100):
        _mc_worker_task(
            sim_params,
            wind_unc=0.2,
            gust_sigma=1.0,
            tu=0.05,
            raw_thrust=sim_params["thrust_data"],
            elev=sim_params["elev"],
            azi=sim_params["azi"],
            base_u=sim_params["wind_u_prof"],
            base_v=sim_params["wind_v_prof"],
            flight_mode="Altitude Competition",
            target_radius=1000,
            seed=i
        )

cProfile.run('run()', 'stats_worker')
p = pstats.Stats('stats_worker')
p.sort_stats('tottime').print_stats(15)
