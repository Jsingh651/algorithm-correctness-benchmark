"""Empirical time-complexity profiling.

For every algorithm that supplies a ``scaler`` we time it across a geometric
spread of input sizes, then fit a straight line to ``log(time)`` vs
``log(n)``. The slope is the observed growth exponent; comparing it to the
algorithm's declared ``expected_exponent`` surfaces implementations whose
real-world scaling deviates from their textbook complexity (e.g. an accidental
quadratic in something that should be linearithmic).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from algobench.core.registry import Algorithm, get_registry


def _time_once(algo: Algorithm, n: int, repeats: int) -> float:
    payload = algo.scaler(n)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        algo.func(**payload)
        best = min(best, time.perf_counter() - start)
    return best


def profile_algorithm(
    algo: Algorithm,
    sizes=(64, 128, 256, 512, 1024, 2048),
    repeats: int = 3,
) -> Optional[dict]:
    """Return a record describing observed vs expected scaling, or ``None``
    when the algorithm has no scaler."""
    if algo.scaler is None:
        return None

    xs: List[int] = []
    ys: List[float] = []
    for n in sizes:
        t = _time_once(algo, n, repeats)
        # Guard against unmeasurably small timings on tiny inputs.
        if t > 0:
            xs.append(n)
            ys.append(t)

    if len(xs) < 3:
        return None

    log_n = np.log(np.asarray(xs, dtype=float))
    log_t = np.log(np.asarray(ys, dtype=float))
    fit = stats.linregress(log_n, log_t)

    observed = float(fit.slope)
    expected = float(algo.expected_exponent)
    return {
        "algorithm": algo.name,
        "category": algo.category,
        "complexity_label": algo.complexity_label,
        "expected_exponent": expected,
        "observed_exponent": round(observed, 3),
        "deviation": round(observed - expected, 3),
        "r_squared": round(float(fit.rvalue) ** 2, 4),
        "n_points": len(xs),
    }


def profile_all(registry: Optional[Dict[str, Algorithm]] = None, **kwargs) -> List[dict]:
    registry = registry or get_registry()
    out: List[dict] = []
    for algo in registry.values():
        rec = profile_algorithm(algo, **kwargs)
        if rec is not None:
            out.append(rec)
    return out
