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
