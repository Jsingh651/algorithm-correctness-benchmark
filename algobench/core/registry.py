"""Algorithm registry.

Each algorithm is described once and registered globally. The registry binds
together the implementation, a constraint checker, case generators, and the
expected asymptotic complexity used for empirical deviation analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from algobench.core.case import TestCase

# A checker receives (payload, output) and returns a list of human-readable
# constraint-violation strings. An empty list means "all constraints hold".
Checker = Callable[[dict, object], List[str]]

# A generator yields TestCase objects for one algorithm.
Generator = Callable[[], List[TestCase]]

# A scaler maps a problem size n -> a payload of that size, used by the
# empirical complexity profiler to fit an observed runtime exponent.
Scaler = Callable[[int], dict]


@dataclass
class Algorithm:
    """Everything the runner needs to benchmark one algorithm."""

    name: str
    category: str
    func: Callable
    checker: Checker
    generators: List[Generator] = field(default_factory=list)
    # Expected complexity exponent k such that runtime ~ n**k (log-log slope).
    # Use 1.0 for O(n), 2.0 for O(n^2). For O(n log n) the empirical slope sits
    # slightly above 1.0; we record 1.1 as the nominal target.
    expected_exponent: float = 1.0
    complexity_label: str = "O(n)"
    # Optional payload factory for the empirical complexity profiler.
    scaler: Optional[Scaler] = None

    def cases(self) -> List[TestCase]:
        out: List[TestCase] = []
        for gen in self.generators:
            out.extend(gen())
        return out


REGISTRY: Dict[str, Algorithm] = {}


def register(algo: Algorithm) -> Algorithm:
    """Register an algorithm, guarding against duplicate names."""
    if algo.name in REGISTRY:
        raise ValueError(f"duplicate algorithm name: {algo.name!r}")
    REGISTRY[algo.name] = algo
    return algo


def get_registry() -> Dict[str, Algorithm]:
    """Import every algorithm module (populating ``REGISTRY``) and return it."""
    # Importing the package triggers registration as a side effect.
    import algobench.algorithms  # noqa: F401  (import-for-side-effect)

    return REGISTRY
