import os
import concurrent.futures

_global_pool = None

def get_global_pool():
    global _global_pool
    if _global_pool is None:
        # RocketPy's Flight integrator uses scipy.integrate.solve_ivp with the
        # LSODA method, which relies on legacy Fortran code holding global state.
        # This makes the physics core strictly NOT THREAD-SAFE. We MUST use
        # ProcessPoolExecutor to guarantee memory isolation, despite the Windows
        # spawn overhead, to prevent memory corruption and UI freezes.
        n_workers = os.cpu_count() or 1
        _global_pool = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)
    return _global_pool


def shutdown_global_pool(wait: bool = True) -> None:
    """Gracefully shut down the global pool. Call at application exit."""
    global _global_pool
    if _global_pool is not None:
        _global_pool.shutdown(wait=wait)
        _global_pool = None
