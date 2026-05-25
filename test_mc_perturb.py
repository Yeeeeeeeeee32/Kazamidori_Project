import sys; sys.path.insert(0,'.')
from ui_qt.workers import SimulationWorker
from core.simulation import simulate_once
from core.monte_carlo import _perturb_wind_profile
import random

p = {
    'airframe_mass': 1.0,
    'airframe_len': 2.0,
    'radius': 0.1,
    'airframe_cg': 1.0,
    'nose_len': 0.5,
    'fin_root': 0.2,
    'fin_tip': 0.1,
    'fin_span': 0.1,
    'fin_pos': 1.8,
    'motor_pos': 2.0,
    'motor_dry_mass': 1.0,
    'backfire_delay': 2.0,
    'power_on_cd': 0.45,
    'power_off_cd': 0.4,
    'para_cd': 1.2,
    'para_area': 1.0,
    'para_lag': 1.0,
    'rail': 2.0,
    'thrust_data': [[0,100], [1,100]],
    'motor_burn_time': 1.0,
    'elev': 85.0,
    'azim': 0.0,
    'mc_runs': 5,
    'wind_unc': 10.0,
    'thrust_unc': 5.0,
    'gust_sigma': 2.0,
}

u_prof, v_prof = SimulationWorker._build_wind_profiles(p)
sim_params = SimulationWorker._build_sim_params(p, u_prof, v_prof)
rng = random.Random(42)

base_u = sim_params.get('wind_u_prof', [])
base_v = sim_params.get('wind_v_prof', [])

for i in range(1, 6):
    u_new, v_new, _ = _perturb_wind_profile(base_u, base_v, rng, 10.0, 2.0)
    trial_p = dict(sim_params)
    trial_p['wind_u_prof'] = u_new
    trial_p['wind_v_prof'] = v_new
    try:
        r = simulate_once(85.0, 0.0, trial_p, trial_idx=i)
        print(f'Trial {i} OK: {r["ok"]}')
        if not r["ok"]: print(f'  Error: {r["error"]}')
    except Exception as e:
        print(f'Trial {i} EXCEPTION:', type(e).__name__, str(e))
