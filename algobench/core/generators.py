"""Random and adversarial test-case generators.

Random generators provide broad coverage; adversarial generators deliberately
target the brittle assumptions of each algorithm class:

* **empty / singleton inputs** — off-by-one and base-case bugs,
* **integer overflow** — values near ``np.int64`` limits that overflow naive
  fixed-width arithmetic,
* **degenerate structure** — already-sorted / reverse-sorted arrays, all-equal
  elements, cyclic and disconnected graphs, negative edge weights.

All randomness is seeded for reproducibility.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from algobench.core.case import TestCase

SEED = 1234
_rng = np.random.default_rng(SEED)


def reseed(seed: int = SEED) -> None:
    """Reset the module RNG so a full run is byte-for-byte reproducible."""
    global _rng
    _rng = np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Array generators (sorting / searching / DP over sequences)
# --------------------------------------------------------------------------- #
def random_arrays(
    count: int = 40,
    sizes: Sequence = (0, 1, 2, 5, 16, 64, 256),
    low: int = -1000,
    high: int = 1000,
) -> List[TestCase]:
    """A spread of random integer arrays across several sizes."""
    cases: List[TestCase] = []
    for i in range(count):
        n = int(sizes[i % len(sizes)])
        arr = _rng.integers(low, high, size=n).tolist()
        cases.append(
            TestCase(
                name=f"random[{i}]n={n}",
                payload={"arr": arr},
                kind="random",
                size=n,
            )
        )
    return cases
