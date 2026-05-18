## YYYY-MM-DD - [Python vs Numpy for small data]
**Learning:** Instantiating `numpy.array` from lists solely to calculate means on small-to-medium datasets introduces significant Python-to-C memory translation overhead that outweighs vectorization benefits.
**Action:** Use native Python sums (`sum_x += x`, `cx = sum_x / n`) for simple statistics on short lists (like landing zone scatter points) instead of forcing them into NumPy arrays inside hot paths.

## YYYY-MM-DD - [Pre-allocating lists in hot loops]
**Learning:** Calling `.append()` in a hot loop has noticeable overhead due to dynamic array resizing.
**Action:** When the length is known in advance (e.g., matching a zip length), pre-allocate lists using `[None] * n` and index into them to improve loop execution speed.

## YYYY-MM-DD - [Caching function references]
**Learning:** Method lookups (like `math.hypot` and `rng.gauss`) inside tight inner loops cost dictionary lookup overhead on every single iteration.
**Action:** Cache method references to local variables (e.g., `math_hypot = math.hypot`) before the loop starts to shave off milliseconds per thousand iterations in heavy computations.
