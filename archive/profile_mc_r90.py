import time
import math
import random as _random_mod
import concurrent.futures
import os

def build_perturbed_wind_prof(base_params, rng, wu):
    return [], [], 0, 0, []

def simulate_once(elev, azi, p):
    return {'ok': True, 'impact_x': 10, 'impact_y': 20}

def _mc_r90_worker(elev, azi, base_params, seed, wu, tu, raw_thrust):
    rng = _random_mod.Random(seed)
    u_prof, v_prof, _, _, _ = build_perturbed_wind_prof(base_params, rng, wu)
    thrust_scale     = max(0.1, 1.0 + rng.gauss(0.0, tu))
    perturbed_thrust = [[t, T * thrust_scale] for (t, T) in raw_thrust]

    p = dict(base_params)
    p['wind_u_prof'] = u_prof
    p['wind_v_prof'] = v_prof
    p['thrust_data'] = perturbed_thrust

    r = simulate_once(elev, azi, p)
    if r['ok']:
        return math.hypot(r['impact_x'], r['impact_y'])
    return None

def _monte_carlo_r90(
    elev: float, azi: float,
    base_params: dict,
    n_trials: int,
    landing_prob: int,
    wind_uncertainty: float,
    thrust_uncertainty: float,
    stop_flag = None,
) -> tuple[float, float]:
    distances: list[float] = []
    succeeded = 0
    wu  = max(wind_uncertainty, 0.0)
    tu  = max(thrust_uncertainty, 0.0)
    raw_thrust = base_params['thrust_data']

    base_seed = _random_mod.randint(0, 2**31 - 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for i in range(n_trials):
            if stop_flag is not None and stop_flag.is_set():
                break
            futures.append(executor.submit(_mc_r90_worker, elev, azi, base_params, base_seed + i, wu, tu, raw_thrust))

        for future in concurrent.futures.as_completed(futures):
            if stop_flag is not None and stop_flag.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                break
            res = future.result()
            if res is not None:
                distances.append(res)
                succeeded += 1

    if not distances:
        return float('inf'), 0.0
    distances.sort()
    p_idx = max(0, min(
        len(distances) - 1,
        int(round((landing_prob / 100.0) * len(distances))) - 1))
    return distances[p_idx], succeeded / n_trials

base_params = {
    'surf_spd': 5.0,
    'surf_dir': 90.0,
    'up_spd': 10.0,
    'up_dir': 90.0,
    'thrust_data': [[0, 10], [1, 10]]
}

t0 = time.perf_counter()
_monte_carlo_r90(80, 0, base_params, 40, 90, 0.2, 0.05)
print(time.perf_counter() - t0)
