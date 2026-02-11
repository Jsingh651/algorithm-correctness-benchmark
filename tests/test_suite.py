"""Tests for the algobench framework itself.

These verify two things:

1. The framework correctly *accepts* correct implementations (every shipped
   algorithm passes all of its generated constraints).
2. The framework correctly *rejects* incorrect ones — we inject deliberately
   buggy algorithms and assert the checkers flag them. A benchmark that only
   ever says "pass" is worthless; these tests prove it has teeth.
"""

from __future__ import annotations

import numpy as np
import pytest

from algobench.algorithms.sorting import merge_sort
from algobench.core.case import TestCase
from algobench.core.checkers import check_search, check_sort, is_permutation, is_sorted
from algobench.core.complexity import profile_algorithm
from algobench.core.generators import reseed
from algobench.core.registry import Algorithm, get_registry
from algobench.core.runner import BenchmarkRunner


@pytest.fixture(scope="module")
def registry():
    reseed()
    return get_registry()


# --------------------------------------------------------------------------- #
# Registry / coverage
# --------------------------------------------------------------------------- #
def test_registry_has_15_plus_algorithms(registry):
    assert len(registry) >= 15


def test_suite_has_500_plus_cases(registry):
    total = sum(len(algo.cases()) for algo in registry.values())
    assert total >= 500, f"only {total} cases"


def test_multiple_categories(registry):
    cats = {a.category for a in registry.values()}
    assert {"sorting", "searching", "graph", "dp", "numeric", "string"} <= cats


# --------------------------------------------------------------------------- #
# Checker primitives
# --------------------------------------------------------------------------- #
def test_is_sorted():
    assert is_sorted([1, 2, 2, 3])
    assert is_sorted([])
    assert not is_sorted([3, 1])


def test_is_permutation():
    assert is_permutation([3, 1, 2], [1, 2, 3])
    assert not is_permutation([1, 2, 2], [1, 2, 3])


def test_check_sort_flags_dropped_element():
    violations = check_sort({"arr": [3, 1, 2]}, [1, 2])  # missing an element
    assert violations  # non-empty -> failure detected


def test_check_search_flags_false_negative():
    violations = check_search({"arr": [1, 2, 3], "target": 2}, -1)
    assert violations


# --------------------------------------------------------------------------- #
# End-to-end: correct implementations pass
# --------------------------------------------------------------------------- #
def test_all_shipped_algorithms_pass(registry):
    runner = BenchmarkRunner()
    results = runner.run_all(registry)
    failures = [r for r in results if not r.passed]
    assert not failures, f"{len(failures)} unexpected failures: " + \
        ", ".join(f"{r.algorithm}/{r.case}: {r.violations or r.error}" for r in failures[:5])


# --------------------------------------------------------------------------- #
# The framework actually catches bugs
# --------------------------------------------------------------------------- #
def _buggy_sort(arr):
    """Drops the last element — a subtle off-by-one a naive eyeball test
    (output *looks* sorted) would miss but the permutation invariant catches."""
    return sorted(arr)[:-1] if arr else arr


def test_framework_rejects_buggy_sort():
    buggy = Algorithm(
        "buggy_sort", "sorting", _buggy_sort, check_sort,
        generators=[lambda: [TestCase("t", {"arr": [5, 3, 1, 4, 2]}, size=5)]],
    )
    runner = BenchmarkRunner()
    result = runner.run_case(buggy, buggy.cases()[0])
    assert not result.passed
    assert any("permutation" in v or "length" in v for v in result.violations)


def test_framework_records_expected_errors():
    """A case that expects an exception passes when that exception is raised."""
    def explodes(x):
        raise ValueError("boom")

    algo = Algorithm(
        "explodes", "misc", explodes, lambda p, o: [],
        generators=[lambda: [TestCase("e", {"x": 1}, expect_error=ValueError)]],
    )
    runner = BenchmarkRunner()
    result = runner.run_case(algo, algo.cases()[0])
    assert result.passed


# --------------------------------------------------------------------------- #
# Complexity profiling separates growth classes
# --------------------------------------------------------------------------- #
def test_complexity_distinguishes_quadratic_from_linearithmic(registry):
    ins = profile_algorithm(registry["insertion_sort"], sizes=(64, 128, 256, 512))
    mrg = profile_algorithm(registry["merge_sort"], sizes=(64, 128, 256, 512, 1024))
    assert ins["observed_exponent"] > mrg["observed_exponent"]
    assert ins["observed_exponent"] > 1.5  # clearly super-linear
