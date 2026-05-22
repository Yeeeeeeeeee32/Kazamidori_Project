import cProfile
import pstats
from core.monte_carlo import _perturb_wind_profile
import random
import math

base_u = [(float(i), 10.0) for i in range(100)]
base_v = [(float(i), 5.0) for i in range(100)]
rng = random.Random(42)

def _perturb_wind_profile_opt2(
    u_prof: list[tuple[float, float]],
    v_prof: list[tuple[float, float]],
    rng: random.Random,
    wind_uncertainty: float,
    gust_intensity: float = 0.0,
):
    if not u_prof or not v_prof:
        return list(u_prof), list(v_prof), []

    wu = max(wind_uncertainty, 0.0)

    speed_factor = max(0.05, 1.0 + rng.gauss(0.0, wu))
    dir_rot      = rng.gauss(0.0, wu * math.pi / 6.0)
    cos_r, sin_r = math.cos(dir_rot), math.sin(dir_rot)

    has_gust = gust_intensity > 0.0
    gust_sigma = float(gust_intensity)

    rng_gauss = rng.gauss
    math_hypot = math.hypot
    math_sqrt = math.sqrt

    n = len(u_prof)
    u_new = [None] * n
    v_new = [None] * n
    spd_out = [None] * n

    gust_sigma_sq = gust_sigma * gust_sigma
    wu_030 = wu * 0.30

    for i in range(n):
        alt_u, u_nom = u_prof[i]
        _, v_nom = v_prof[i]

        # 1. Global (synoptic) rotation & scaling
        u_g = (u_nom * cos_r - v_nom * sin_r) * speed_factor
        v_g = (u_nom * sin_r + v_nom * cos_r) * speed_factor

        # 2. Local (mesoscale) turbulence
        local_spd = math_hypot(u_nom, v_nom)
        sigma     = wu_030 * max(local_spd, 1.0)

        if has_gust:
            # Combine variances: Var(sum) = Var(X) + Var(Y) = sigma^2 + gust_sigma^2
            total_sigma = math_sqrt(sigma*sigma + gust_sigma_sq)
            u_val = u_g + rng_gauss(0.0, total_sigma)
            v_val = v_g + rng_gauss(0.0, total_sigma)
        else:
            u_val = u_g + rng_gauss(0.0, sigma)
            v_val = v_g + rng_gauss(0.0, sigma)

        u_new[i] = (alt_u, u_val)
        v_new[i] = (alt_u, v_val)
        spd_out[i] = (alt_u, math_hypot(u_val, v_val))

    return u_new, v_new, spd_out

def run():
    for _ in range(10000):
        _perturb_wind_profile_opt2(base_u, base_v, rng, 0.2, 1.0)

cProfile.run('run()', 'stats')
p = pstats.Stats('stats')
p.sort_stats('tottime').print_stats(10)
