import time, os, sys
sys.path.insert(0, '.')
from core.simulation import simulate_once
from core.wind_model import create_wind_profile

profile = [
    {'alt_m':3,   'speed_ms':4, 'dir_deg':0},
    {'alt_m':600, 'speed_ms':8, 'dir_deg':0},
]
u, v = create_wind_profile(profile)
params = {
    'wind_u_prof': u,
    'wind_v_prof': v,
    'thrust_data': [[0,10],[0.5,10],[1,5],[2,0]],
    'motor_burn_time': 2.0,
    'launch_lat': 35.0,
    'launch_lon': 135.0,
}

print("=== Single simulate_once test ===")
t0 = time.perf_counter()
r = simulate_once(85.0, 0.0, params)
t1 = time.perf_counter()
print(f"ok={r['ok']}  time={t1-t0:.2f}s")
if not r['ok']:
    print("ERROR:", r.get('error'))
else:
    print(f"apogee={r['apogee_m']:.1f}m  hang={r['hang_time']:.1f}s")

# Test parallel with ThreadPoolExecutor
import concurrent.futures
print("\n=== ThreadPoolExecutor (4 workers) test ===")
def run_one(i):
    t = time.perf_counter()
    res = simulate_once(85.0, float(i), params)
    return i, res['ok'], time.perf_counter()-t

t0 = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    futures = [ex.submit(run_one, i) for i in range(8)]
    for f in concurrent.futures.as_completed(futures):
        i, ok, dt = f.result()
        print(f"  trial {i}: ok={ok}  {dt:.2f}s")
t1 = time.perf_counter()
print(f"Total wall time for 8 runs: {t1-t0:.2f}s (sequential would be ~{8*(t1-t0)/8:.2f}s each)")
