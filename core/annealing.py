"""Bounded, reproducible simulated annealing; no GUI or flight-model dependency."""
import math
import random


def anneal(evaluate, *, budget=48, seed=0, initial=(90.0, 0.0), cancelled=lambda: False,
           progress=lambda *_: None):
    """Maximise an objective over elevation [60,90] and periodic azimuth.

    Keep every finite candidate, accept downhill steps with exp(delta/T),
    and use periodic global proposals to explore disconnected feasible regions.
    The callback returns (score, payload); failures are represented by -inf.
    """
    if not 8 <= budget <= 10000:
        raise ValueError('SA evaluation budget must be between 8 and 10000')
    rng = random.Random(seed)
    current = (min(90., max(60., initial[0])), initial[1] % 360)
    current_score = -math.inf
    scale = 1.0
    candidates = []
    for i in range(budget):
        if cancelled():
            raise RuntimeError('cancelled')
        cooling = (1 - i / max(1, budget - 1))
        temperature = max(.01, cooling ** 2)
        if i == 0:
            proposal = current
        elif i % 8 == 0 or not math.isfinite(current_score):
            proposal = (rng.uniform(60, 90), rng.uniform(0, 360))
        else:
            # Reflect at elevation boundaries instead of accumulating at clamps.
            e = (current[0] - 60 + rng.gauss(0, 1 + 10 * cooling)) % 60
            proposal = (60 + (e if e <= 30 else 60 - e),
                        (current[1] + rng.gauss(0, 3 + 80 * cooling)) % 360)
        score, payload = evaluate(*proposal)
        if math.isfinite(score):
            candidates.append((score, *proposal, payload))
            if math.isfinite(current_score):
                scale = max(scale, abs(score - current_score))
            delta = score - current_score
            if delta >= 0 or rng.random() < math.exp(max(-700, delta / (scale * temperature))):
                current, current_score = proposal, score
        progress(i + 1, budget)
    return sorted(candidates, key=lambda row: row[0], reverse=True)
