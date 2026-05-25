import os
import concurrent.futures

_global_pool = None

def get_global_pool():
    global _global_pool
    if _global_pool is None:
        # Prevent SciPy/NumPy from oversubscribing the CPU in worker processes
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        _global_pool = concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count())
    return _global_pool
