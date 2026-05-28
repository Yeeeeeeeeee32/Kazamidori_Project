"""
core/pool_manager.py

Global ProcessPoolExecutor for CPU-bound Monte Carlo simulations.

Performance notes
-----------------
RocketPy's Flight integrator uses scipy.integrate.solve_ivp (LSODA solver),
which wraps legacy Fortran code that holds global state and is strictly
NOT THREAD-SAFE.  ProcessPoolExecutor provides full memory isolation via
OS-level process separation.

CPU oversubscription prevention
--------------------------------
scipy / numpy rely on BLAS / OpenBLAS / MKL for matrix operations.  By
default those libraries spawn N_CORES threads per process.  Running
N_WORKERS processes each with N_CORES BLAS threads produces:
    N_WORKERS × N_CORES threads competing for N_CORES physical cores
This causes thrashing, massively increased context-switch overhead, and
effectively starves the Qt GUI thread — freezing the UI.

The env-var block below limits every BLAS/OMP library to a single thread
so the worker pool runs as exactly N_WORKERS clean processes.  The vars
must be set BEFORE any numpy/scipy import in the parent process; they are
inherited by spawned children automatically.
"""

import os
import concurrent.futures
import multiprocessing

# ── BLAS / OpenMP thread cap ──────────────────────────────────────────────────
# Must be set before numpy/scipy are imported anywhere in this process.
# 'setdefault' avoids overriding an explicit user override (e.g. via shell).
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",    "1")

_global_pool = None


def _init_child_process() -> None:
    """Initializer run once in every spawned worker process.

    Sets the process scheduling priority to BELOW_NORMAL so that the Qt /
    OS GUI thread is never fully starved when all cores are busy with MC
    simulations.  Silently ignores any failure (e.g. on Linux where the
    Windows constant is irrelevant).
    """
    import sys
    if sys.platform == "win32":
        try:
            import ctypes
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                BELOW_NORMAL_PRIORITY_CLASS,
            )
        except Exception:
            pass


def get_global_pool():
    global _global_pool
    if _global_pool is None:
        # Reserve 1 core for the OS and Qt GUI event loop so the window
        # remains responsive while workers run.  Minimum 1 worker even on
        # single-core machines.
        n_workers = max(1, (os.cpu_count() or 2) - 1)

        # 'spawn' context: child processes are created fresh without inheriting
        # the parent's memory (including any PySide6 / Qt state), which is the
        # only safe option on Windows (Windows has no fork(2)) and recommended
        # on Linux to prevent GUI deadlocks from forking a Qt event loop.
        ctx = multiprocessing.get_context('spawn')
        _global_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_child_process,
        )
    return _global_pool



def warmup_pool() -> None:
    """
    Submits dummy tasks to the pool to force worker processes to spawn and load
    their heavy modules (like numpy, scipy, RocketPy) BEFORE the first actual
    simulation. This removes the 5-10s cold-start latency from the UI thread.
    """
    pool = get_global_pool()
    from core.mc_worker import _noop_warmup
    # Submit one task per worker to ensure all processes in the pool are spawned
    n_workers = pool._max_workers
    for _ in range(n_workers):
        pool.submit(_noop_warmup)


def shutdown_global_pool(wait: bool = True) -> None:
    """Gracefully shut down the global pool. Call at application exit."""
    global _global_pool
    if _global_pool is not None:
        _global_pool.shutdown(wait=wait)
        _global_pool = None
