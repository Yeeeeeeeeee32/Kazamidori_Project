import cProfile
import pstats
from core.monte_carlo import _perturb_wind_profile
import random

base_u = [(float(i), 10.0) for i in range(100)]
base_v = [(float(i), 5.0) for i in range(100)]
rng = random.Random(42)

def run():
    for _ in range(10000):
        _perturb_wind_profile(base_u, base_v, rng, 0.2, 0.0)

cProfile.run('run()', 'stats')
p = pstats.Stats('stats')
p.sort_stats('tottime').print_stats(10)
