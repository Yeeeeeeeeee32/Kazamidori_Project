import cProfile
import pstats
from core.monte_carlo import run_mc_scatter
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
    run_mc_scatter(
        sim_params,
        n_runs=100,
        wind_uncertainty=0.2,
        thrust_uncertainty=0.05,
        gust_intensity=0.0,
    )

cProfile.run('run()', 'stats')
p = pstats.Stats('stats')
p.sort_stats('tottime').print_stats(20)
