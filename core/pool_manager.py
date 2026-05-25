import os
import concurrent.futures

_global_pool = None

def get_global_pool():
    global _global_pool
    if _global_pool is None:
        _global_pool = concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count())
    return _global_pool
