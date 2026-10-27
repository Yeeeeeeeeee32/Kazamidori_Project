"""Pure validation shared by session loading and Qt state."""
import math

def normalize_phase4(value, current):
    if not isinstance(value, dict):
        raise ValueError('Phase 4 settings must be an object')
    options = dict(current)
    options.update({k: v for k, v in value.items() if k in options})
    for key, lo, hi in (('score_a', -1e6, 1e6), ('score_b', -1e6, 1e6),
                        ('launch_alt_m', -1000, 10000), ('manual_declination', -180, 180)):
        options[key] = float(options[key])
        if not math.isfinite(options[key]) or not lo <= options[key] <= hi:
            raise ValueError(f'{key} must be between {lo} and {hi}')
    budget = float(options['sa_budget'])
    if not budget.is_integer() or not 8 <= budget <= 10000:
        raise ValueError('SA budget must be 8–10000')
    options['sa_budget'] = int(budget)
    seed = float(options['random_seed'])
    if not seed.is_integer() or not 0 <= seed <= 2147483647:
        raise ValueError('Random seed must be an integer between 0 and 2147483647')
    options['random_seed'] = int(seed)
    if not all(isinstance(options[k], str) for k in ('wind_csv', 'wind_time_utc')):
        raise ValueError('Wind CSV path and time must be strings')
    if options['wind_csv']:
        from utils.wind_data import utc_time
        utc_time(options['wind_time_utc'])
    if options['declination_mode'] not in ('auto', 'manual') or options['azimuth_reference'] not in ('true', 'magnetic'):
        raise ValueError('Invalid bearing reference or declination mode')
    return options
