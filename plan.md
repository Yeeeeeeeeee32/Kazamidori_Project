1. **Optimize `_monte_carlo_r90` in `core/optimization.py`**
   - The `_monte_carlo_r90` function runs multiple RocketPy simulations in a sequential loop. RocketPy simulations are CPU-bound and slow.
   - We will replace the sequential `for` loop with a `ProcessPoolExecutor` to run the simulations in parallel, bypassing the GIL.
   - We will create a picklable worker function `_mc_r90_worker` to execute the simulations.
   - We will add `import concurrent.futures`, `import os`, and use `os.cpu_count()`.
   - We will add a benchmark comment, as per the bolt persona rules.

2. **Optimize `p1_mc_points` in `core/optimization.py`**
   - The `p1_mc_points` function runs multiple RocketPy simulations in a sequential loop.
   - We will replace the sequential `for` loop with a `ProcessPoolExecutor` to run the simulations in parallel.
   - We will create a picklable worker function `_p1_mc_worker` to execute the simulations.
   - We will add a benchmark comment.

3. **Complete pre-commit steps**
   - Run tests and linting to ensure everything works as expected.

4. **Submit PR**
   - Present the changes as requested by Bolt.
