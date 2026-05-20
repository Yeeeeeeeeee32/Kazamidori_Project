import time
import math
import random as _random_mod

def build_wind_profile(surf_spd, surf_dir, obs_alt, up_spd, up_dir, blend_alt):
    # Dummy mock
    return [], []

def simulate_once(elev, azi, p):
    # Dummy mock
    return {'ok': True, 'impact_x': 10, 'impact_y': 20}

def p1_mc_points(
    elev: float, azi: float,
    base_params: dict,
    mu: float, sigma: float,
    n: int,
    stop_flag = None,
) -> list[tuple[float, float]]:
    rng        = _random_mod.Random()
    mu_nominal = max(base_params['surf_spd'], 1e-6)
    points: list[tuple[float, float]] = []

    for _ in range(n):
        if stop_flag is not None and stop_flag.is_set():
            break
        surf_spd = max(0.0, rng.gauss(mu, sigma))
        ratio    = surf_spd / mu_nominal
        up_spd   = max(0.0, rng.gauss(base_params['up_spd'] * ratio, sigma * 0.5))
        u_prof, v_prof = build_wind_profile(
            surf_spd, base_params['surf_dir'], 1.5,
            up_spd,   base_params['up_dir'],   100.0,
        )
        p = dict(base_params)
        p['wind_u_prof'] = u_prof
        p['wind_v_prof'] = v_prof
        p['surf_spd']    = surf_spd
        r = simulate_once(elev, azi, p)
        if r['ok']:
            points.append((r['impact_x'], r['impact_y']))

    return points

base_params = {
    'surf_spd': 5.0,
    'surf_dir': 90.0,
    'up_spd': 10.0,
    'up_dir': 90.0
}

t0 = time.perf_counter()
p1_mc_points(80, 0, base_params, 5.0, 1.0, 40)
print(time.perf_counter() - t0)
