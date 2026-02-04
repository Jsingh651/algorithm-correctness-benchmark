"""Search algorithms returning an index of ``target`` in ``arr`` or ``-1``."""

from __future__ import annotations

from typing import List

import numpy as np

from algobench.core.checkers import check_search
from algobench.core.generators import search_cases
from algobench.core.registry import Algorithm, register


def linear_search(arr: List[int], target: int) -> int:
    for i, v in enumerate(arr):
        if v == target:
            return i
    return -1


def binary_search(arr: List[int], target: int) -> int:
    """Classic binary search. Assumes ``arr`` is sorted ascending. Written to
    avoid the ``(lo + hi)`` overflow idiom even though Python ints are
    unbounded — the suite explicitly stresses overflow-prone styles."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def _search_scaler(n: int) -> dict:
    arr = list(range(n))
    return {"arr": arr, "target": n - 1}  # worst case for linear, present


register(
    Algorithm(
        "linear_search", "searching", linear_search, check_search,
        [search_cases], scaler=_search_scaler,
        expected_exponent=1.0, complexity_label="O(n)",
    )
)
register(
    Algorithm(
        "binary_search", "searching", binary_search, check_search,
        [search_cases], scaler=_search_scaler,
        expected_exponent=0.1, complexity_label="O(log n)",
    )
)
