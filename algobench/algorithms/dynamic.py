"""Dynamic-programming algorithms.

Generators embed an independently-computed ``expected`` answer in each payload
(brute force where feasible, an alternative DP otherwise) so the checker is a
genuine cross-check rather than a tautology against the same method.
"""

from __future__ import annotations

import bisect
from functools import lru_cache
from itertools import combinations
from typing import List, Tuple

import numpy as np

from algobench.core.case import TestCase
from algobench.core.checkers import check_equals
from algobench.core.registry import Algorithm, register

_rng = np.random.default_rng(2024)


# --------------------------------------------------------------------------- #
# Implementations under test
# --------------------------------------------------------------------------- #
def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        w, v = weights[i], values[i]
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[capacity]


def longest_increasing_subsequence(arr: List[int]) -> int:
    """O(n log n) patience-sorting length of the strictly increasing LIS."""
    tails: List[int] = []
    for x in arr:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


register(Algorithm("knapsack_01", "dp", knapsack_01, check_equals("expected"), [_knapsack_cases],
                   scaler=_knap_scaler, expected_exponent=2.0, complexity_label="O(n*W)"))
register(Algorithm("longest_increasing_subsequence", "dp", longest_increasing_subsequence, check_equals("expected"),
                   [_lis_cases], scaler=_lis_scaler, expected_exponent=1.1, complexity_label="O(n log n)"))
