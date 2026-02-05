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


def search_cases() -> List[TestCase]:
    """Sorted arrays paired with present/absent targets (binary search needs
    sorted input; we feed sorted arrays and probe both hit and miss paths)."""
    cases: List[TestCase] = []
    for i in range(30):
        n = (i + 1) * 8
        arr = sorted(_rng.integers(-500, 500, size=n).tolist())
        # alternate between a target that exists and one that (likely) does not
        if i % 2 == 0 and arr:
            target = int(arr[_rng.integers(0, len(arr))])
            tag = {"hit"}
        else:
            target = 10_000  # guaranteed outside the [-500,500) range
            tag = {"miss"}
        cases.append(
            TestCase(
                f"search[{i}]n={n}",
                {"arr": arr, "target": target},
                "random",
                tag,
                size=n,
            )
        )
    # adversarial search edges
    cases += [
        TestCase("search_empty", {"arr": [], "target": 5}, "adversarial", {"empty"}, size=0),
        TestCase("search_single_hit", {"arr": [9], "target": 9}, "adversarial", {"singleton", "hit"}, size=1),
        TestCase("search_single_miss", {"arr": [9], "target": 1}, "adversarial", {"singleton", "miss"}, size=1),
        TestCase("search_dups", {"arr": [1, 1, 1, 1, 1], "target": 1}, "adversarial", {"duplicates", "hit"}, size=5),
    ]
    return cases

# --------------------------------------------------------------------------- #
# Graph generators
# --------------------------------------------------------------------------- #
def _random_dag(n: int, density: float) -> Dict[int, list]:
    """Random directed acyclic graph as an adjacency dict (edges go low->high)."""
    adj: Dict[int, list] = {v: [] for v in range(n)}
    for u in range(n):
        for v in range(u + 1, n):
            if _rng.random() < density:
                adj[u].append(v)
    return adj


def _random_weighted(n: int, density: float, low: int = 1, high: int = 20) -> Dict[int, list]:
    """Random weighted directed graph; weights are non-negative (Dijkstra)."""
    adj: Dict[int, list] = {v: [] for v in range(n)}
    for u in range(n):
        for v in range(n):
            if u != v and _rng.random() < density:
                w = int(_rng.integers(low, high))
                adj[u].append((v, w))
    return adj


def random_graphs() -> List[TestCase]:
    cases: List[TestCase] = []
    for i, n in enumerate([1, 2, 4, 8, 16, 32]):
        adj = _random_dag(n, density=0.3)
        cases.append(
            TestCase(f"dag[{i}]n={n}", {"graph": adj, "n": n, "start": 0}, "random", {"dag"}, size=n)
        )
    return cases


def random_weighted_graphs() -> List[TestCase]:
    cases: List[TestCase] = []
    for i, n in enumerate([1, 2, 4, 8, 16, 32]):
        adj = _random_weighted(n, density=0.4)
        cases.append(
            TestCase(
                f"wgraph[{i}]n={n}",
                {"graph": adj, "n": n, "start": 0},
                "random",
                {"weighted"},
                size=n,
            )
        )
    return cases


def adversarial_graphs() -> List[TestCase]:
    """Empty, cyclic, disconnected and self-looping graphs."""
    return [
        TestCase("graph_empty", {"graph": {}, "n": 0, "start": 0}, "adversarial", {"empty"}, size=0),
        TestCase(
            "graph_single", {"graph": {0: []}, "n": 1, "start": 0}, "adversarial", {"singleton"}, size=1
        ),
        TestCase(
            "graph_self_loop",
            {"graph": {0: [0]}, "n": 1, "start": 0},
            "adversarial",
            {"cyclic", "self-loop"},
            size=1,
        ),
        TestCase(
            "graph_cycle3",
            {"graph": {0: [1], 1: [2], 2: [0]}, "n": 3, "start": 0},
            "adversarial",
            {"cyclic"},
            size=3,
        ),
        TestCase(
            "graph_disconnected",
            {"graph": {0: [1], 1: [], 2: [3], 3: []}, "n": 4, "start": 0},
            "adversarial",
            {"disconnected"},
            size=4,
        ),
    ]


def adversarial_weighted_graphs() -> List[TestCase]:
    return [
        TestCase(
            "wgraph_empty", {"graph": {}, "n": 0, "start": 0}, "adversarial", {"empty"}, size=0
        ),
        TestCase(
            "wgraph_unreachable",
            {"graph": {0: [(1, 5)], 1: [], 2: [(3, 1)], 3: []}, "n": 4, "start": 0},
            "adversarial",
            {"disconnected", "unreachable"},
            size=4,
        ),
        TestCase(
            "wgraph_cycle",
            {"graph": {0: [(1, 2)], 1: [(2, 2)], 2: [(0, 2)]}, "n": 3, "start": 0},
            "adversarial",
            {"cyclic"},
            size=3,
        ),
    ]

# --------------------------------------------------------------------------- #
# Graph generators
# --------------------------------------------------------------------------- #
def _random_dag(n: int, density: float) -> Dict[int, list]:
    """Random directed acyclic graph as an adjacency dict (edges go low->high)."""
    adj: Dict[int, list] = {v: [] for v in range(n)}
    for u in range(n):
        for v in range(u + 1, n):
            if _rng.random() < density:
                adj[u].append(v)
    return adj


def _random_weighted(n: int, density: float, low: int = 1, high: int = 20) -> Dict[int, list]:
    """Random weighted directed graph; weights are non-negative (Dijkstra)."""
    adj: Dict[int, list] = {v: [] for v in range(n)}
    for u in range(n):
        for v in range(n):
            if u != v and _rng.random() < density:
                w = int(_rng.integers(low, high))
                adj[u].append((v, w))
    return adj


def random_graphs() -> List[TestCase]:
    cases: List[TestCase] = []
    for i, n in enumerate([1, 2, 4, 8, 16, 32]):
        adj = _random_dag(n, density=0.3)
        cases.append(
            TestCase(f"dag[{i}]n={n}", {"graph": adj, "n": n, "start": 0}, "random", {"dag"}, size=n)
        )
    return cases


def random_weighted_graphs() -> List[TestCase]:
    cases: List[TestCase] = []
    for i, n in enumerate([1, 2, 4, 8, 16, 32]):
        adj = _random_weighted(n, density=0.4)
        cases.append(
            TestCase(
                f"wgraph[{i}]n={n}",
                {"graph": adj, "n": n, "start": 0},
                "random",
                {"weighted"},
                size=n,
            )
        )
    return cases


def adversarial_graphs() -> List[TestCase]:
    """Empty, cyclic, disconnected and self-looping graphs."""
    return [
        TestCase("graph_empty", {"graph": {}, "n": 0, "start": 0}, "adversarial", {"empty"}, size=0),
        TestCase(
            "graph_single", {"graph": {0: []}, "n": 1, "start": 0}, "adversarial", {"singleton"}, size=1
        ),
        TestCase(
            "graph_self_loop",
            {"graph": {0: [0]}, "n": 1, "start": 0},
            "adversarial",
            {"cyclic", "self-loop"},
            size=1,
        ),
        TestCase(
            "graph_cycle3",
            {"graph": {0: [1], 1: [2], 2: [0]}, "n": 3, "start": 0},
            "adversarial",
            {"cyclic"},
            size=3,
        ),
        TestCase(
            "graph_disconnected",
            {"graph": {0: [1], 1: [], 2: [3], 3: []}, "n": 4, "start": 0},
            "adversarial",
            {"disconnected"},
            size=4,
        ),
    ]


def adversarial_weighted_graphs() -> List[TestCase]:
    return [
        TestCase(
            "wgraph_empty", {"graph": {}, "n": 0, "start": 0}, "adversarial", {"empty"}, size=0
        ),
        TestCase(
            "wgraph_unreachable",
            {"graph": {0: [(1, 5)], 1: [], 2: [(3, 1)], 3: []}, "n": 4, "start": 0},
            "adversarial",
            {"disconnected", "unreachable"},
            size=4,
        ),
        TestCase(
            "wgraph_cycle",
            {"graph": {0: [(1, 2)], 1: [(2, 2)], 2: [(0, 2)]}, "n": 3, "start": 0},
            "adversarial",
            {"cyclic"},
            size=3,
        ),
    ]

def adversarial_arrays() -> List[TestCase]:
    """Degenerate arrays that break naive sorting/searching assumptions."""
    big = int(np.iinfo(np.int64).max)
    cases = [
        TestCase("empty", {"arr": []}, "adversarial", {"empty"}, size=0),
        TestCase("singleton", {"arr": [42]}, "adversarial", {"singleton"}, size=1),
        TestCase("all_equal", {"arr": [7] * 64}, "adversarial", {"duplicates"}, size=64),
        TestCase(
            "already_sorted",
            {"arr": list(range(128))},
            "adversarial",
            {"sorted", "quicksort-worst-case"},
            size=128,
        ),
        TestCase(
            "reverse_sorted",
            {"arr": list(range(128, 0, -1))},
            "adversarial",
            {"reverse", "quicksort-worst-case"},
            size=128,
        ),
        TestCase(
            "negatives",
            {"arr": [-5, -100, -1, -50, 0, -3]},
            "adversarial",
            {"negative"},
            size=6,
        ),
        TestCase(
            "int64_overflow",
            {"arr": [big, big - 1, big - 2, -big, 0]},
            "adversarial",
            {"overflow"},
            size=5,
        ),
        TestCase(
            "two_elements_swapped",
            {"arr": [2, 1]},
            "adversarial",
            {"minimal"},
            size=2,
        ),
    ]
    return cases


def search_cases() -> List[TestCase]:
    """Sorted arrays paired with present/absent targets (binary search needs
    sorted input; we feed sorted arrays and probe both hit and miss paths)."""
    cases: List[TestCase] = []
    for i in range(30):
        n = (i + 1) * 8
        arr = sorted(_rng.integers(-500, 500, size=n).tolist())
        # alternate between a target that exists and one that (likely) does not
        if i % 2 == 0 and arr:
            target = int(arr[_rng.integers(0, len(arr))])
            tag = {"hit"}
        else:
            target = 10_000  # guaranteed outside the [-500,500) range
            tag = {"miss"}
        cases.append(
            TestCase(
                f"search[{i}]n={n}",
                {"arr": arr, "target": target},
                "random",
                tag,
                size=n,
            )
        )
    # adversarial search edges
    cases += [
        TestCase("search_empty", {"arr": [], "target": 5}, "adversarial", {"empty"}, size=0),
        TestCase("search_single_hit", {"arr": [9], "target": 9}, "adversarial", {"singleton", "hit"}, size=1),
        TestCase("search_single_miss", {"arr": [9], "target": 1}, "adversarial", {"singleton", "miss"}, size=1),
        TestCase("search_dups", {"arr": [1, 1, 1, 1, 1], "target": 1}, "adversarial", {"duplicates", "hit"}, size=5),
    ]
    return cases


