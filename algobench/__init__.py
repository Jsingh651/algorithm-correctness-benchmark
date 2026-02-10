"""algobench — Algorithm Correctness Benchmark Suite.

A structured framework for evaluating classical computer-science algorithms
against expected *constraints* (invariants), not just reference outputs. Every
algorithm ships with:

* one or more implementations,
* a constraint checker that validates output invariants using NumPy,
* random and adversarial test-case generators,
* an expected asymptotic complexity used for empirical deviation analysis.

The :class:`~algobench.core.runner.BenchmarkRunner` executes every case,
records timings and constraint violations, and the reporting layer aggregates
the results into Pandas DataFrames.
"""

from algobench.core.case import TestCase, CaseResult
from algobench.core.registry import REGISTRY, Algorithm, register, get_registry
from algobench.core.runner import BenchmarkRunner

__all__ = [
    "TestCase",
    "CaseResult",
    "REGISTRY",
    "Algorithm",
    "register",
    "get_registry",
    "BenchmarkRunner",
]

__version__ = "1.0.0"
