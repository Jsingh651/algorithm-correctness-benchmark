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


def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def longest_common_subsequence(a: str, b: str) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def coin_change(coins: List[int], amount: int) -> int:
    """Fewest coins summing to ``amount`` or ``-1`` if impossible."""
    INF = float("inf")
    dp = [0] + [INF] * amount
    for c in range(1, amount + 1):
        for coin in coins:
            if coin <= c and dp[c - coin] + 1 < dp[c]:
                dp[c] = dp[c - coin] + 1
    return dp[amount] if dp[amount] != INF else -1


# --------------------------------------------------------------------------- #
# Independent reference oracles
# --------------------------------------------------------------------------- #
def _ref_knapsack(weights, values, capacity) -> int:
    n = len(weights)
    if n == 0 or capacity == 0:
        return 0
    if n <= 18:  # exact brute force over subsets
        best = 0
        for r in range(n + 1):
            for combo in combinations(range(n), r):
                w = sum(weights[i] for i in combo)
                if w <= capacity:
                    best = max(best, sum(values[i] for i in combo))
        return best
    return knapsack_01(weights, values, capacity)  # fall back for large n


def _ref_lis(arr) -> int:
    if not arr:
        return 0
    n = len(arr)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


def _ref_edit(a: str, b: str) -> int:
    @lru_cache(maxsize=None)
    def rec(i: int, j: int) -> int:
        if i == 0:
            return j
        if j == 0:
            return i
        if a[i - 1] == b[j - 1]:
            return rec(i - 1, j - 1)
        return 1 + min(rec(i - 1, j), rec(i, j - 1), rec(i - 1, j - 1))

    return rec(len(a), len(b))


def _ref_lcs(a: str, b: str) -> int:
    @lru_cache(maxsize=None)
    def rec(i: int, j: int) -> int:
        if i == 0 or j == 0:
            return 0
        if a[i - 1] == b[j - 1]:
            return 1 + rec(i - 1, j - 1)
        return max(rec(i - 1, j), rec(i, j - 1))

    return rec(len(a), len(b))


def _ref_coin(coins, amount) -> int:
    INF = float("inf")
    dp = [0] + [INF] * amount
    for c in range(1, amount + 1):
        for coin in coins:
            if coin <= c:
                dp[c] = min(dp[c], dp[c - coin] + 1)
    return dp[amount] if dp[amount] != INF else -1


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
def _knapsack_cases() -> List[TestCase]:
    cases: List[TestCase] = []
    for i in range(14):
        n = (i % 6) + 1
        weights = _rng.integers(1, 15, size=n).tolist()
        values = _rng.integers(1, 30, size=n).tolist()
        cap = int(_rng.integers(0, 40))
        exp = _ref_knapsack(weights, values, cap)
        cases.append(
            TestCase(
                f"knap[{i}]n={n}",
                {"weights": weights, "values": values, "capacity": cap, "expected": exp},
                "random",
                size=n,
            )
        )
    cases += [
        TestCase("knap_empty", {"weights": [], "values": [], "capacity": 10, "expected": 0}, "adversarial", {"empty"}, size=0),
        TestCase("knap_zero_cap", {"weights": [1, 2], "values": [5, 9], "capacity": 0, "expected": 0}, "adversarial", {"zero-capacity"}, size=2),
        TestCase("knap_all_too_heavy", {"weights": [50, 60], "values": [1, 2], "capacity": 10, "expected": 0}, "adversarial", {"infeasible"}, size=2),
    ]
    return cases


def _lis_cases() -> List[TestCase]:
    cases: List[TestCase] = []
    for i in range(14):
        n = (i % 7) + 1
        arr = _rng.integers(-50, 50, size=n).tolist()
        cases.append(TestCase(f"lis[{i}]n={n}", {"arr": arr, "expected": _ref_lis(arr)}, "random", size=n))
    cases += [
        TestCase("lis_empty", {"arr": [], "expected": 0}, "adversarial", {"empty"}, size=0),
        TestCase("lis_decreasing", {"arr": [5, 4, 3, 2, 1], "expected": 1}, "adversarial", {"reverse"}, size=5),
        TestCase("lis_all_equal", {"arr": [7, 7, 7, 7], "expected": 1}, "adversarial", {"duplicates"}, size=4),
        TestCase("lis_sorted", {"arr": [1, 2, 3, 4, 5], "expected": 5}, "adversarial", {"sorted"}, size=5),
    ]
    return cases


def _edit_cases() -> List[TestCase]:
    alphabet = "abcd"
    cases: List[TestCase] = []
    for i in range(12):
        la, lb = (i % 6) + 1, (i % 5) + 1
        a = "".join(_rng.choice(list(alphabet), size=la))
        b = "".join(_rng.choice(list(alphabet), size=lb))
        cases.append(TestCase(f"edit[{i}]", {"a": a, "b": b, "expected": _ref_edit(a, b)}, "random", size=max(la, lb)))
    cases += [
        TestCase("edit_both_empty", {"a": "", "b": "", "expected": 0}, "adversarial", {"empty"}, size=0),
        TestCase("edit_one_empty", {"a": "abc", "b": "", "expected": 3}, "adversarial", {"empty"}, size=3),
        TestCase("edit_identical", {"a": "hello", "b": "hello", "expected": 0}, "adversarial", {"identical"}, size=5),
    ]
    return cases


def _lcs_cases() -> List[TestCase]:
    alphabet = "abc"
    cases: List[TestCase] = []
    for i in range(12):
        la, lb = (i % 6) + 1, (i % 5) + 1
        a = "".join(_rng.choice(list(alphabet), size=la))
        b = "".join(_rng.choice(list(alphabet), size=lb))
        cases.append(TestCase(f"lcs[{i}]", {"a": a, "b": b, "expected": _ref_lcs(a, b)}, "random", size=max(la, lb)))
    cases += [
        TestCase("lcs_both_empty", {"a": "", "b": "", "expected": 0}, "adversarial", {"empty"}, size=0),
        TestCase("lcs_disjoint", {"a": "aaa", "b": "bbb", "expected": 0}, "adversarial", {"disjoint"}, size=3),
        TestCase("lcs_identical", {"a": "abcabc", "b": "abcabc", "expected": 6}, "adversarial", {"identical"}, size=6),
    ]
    return cases


def _coin_cases() -> List[TestCase]:
    cases: List[TestCase] = []
    for i in range(12):
        coins = sorted(set(_rng.integers(1, 12, size=(i % 4) + 1).tolist()))
        amount = int(_rng.integers(0, 40))
        cases.append(TestCase(f"coin[{i}]", {"coins": coins, "amount": amount, "expected": _ref_coin(coins, amount)}, "random", size=amount))
    cases += [
        TestCase("coin_zero_amount", {"coins": [1, 2, 5], "amount": 0, "expected": 0}, "adversarial", {"zero"}, size=0),
        TestCase("coin_impossible", {"coins": [2], "amount": 3, "expected": -1}, "adversarial", {"infeasible"}, size=3),
        TestCase("coin_single", {"coins": [7], "amount": 14, "expected": 2}, "adversarial", {"singleton"}, size=14),
    ]
    return cases


# --------------------------------------------------------------------------- #
# Complexity scalers
# --------------------------------------------------------------------------- #
def _knap_scaler(n: int) -> dict:
    rng = np.random.default_rng(n)
    return {"weights": rng.integers(1, 20, n).tolist(), "values": rng.integers(1, 50, n).tolist(), "capacity": n}


def _lis_scaler(n: int) -> dict:
    return {"arr": np.random.default_rng(n).integers(-n, n, n).tolist()}


def _str_scaler(n: int) -> dict:
    rng = np.random.default_rng(n)
    a = "".join(rng.choice(list("abcd"), size=n))
    b = "".join(rng.choice(list("abcd"), size=n))
    return {"a": a, "b": b}


def _coin_scaler(n: int) -> dict:
    return {"coins": [1, 2, 5, 10], "amount": n}


register(Algorithm("knapsack_01", "dp", knapsack_01, check_equals("expected"), [_knapsack_cases],
                   scaler=_knap_scaler, expected_exponent=2.0, complexity_label="O(n*W)"))
register(Algorithm("longest_increasing_subsequence", "dp", longest_increasing_subsequence, check_equals("expected"),
                   [_lis_cases], scaler=_lis_scaler, expected_exponent=1.1, complexity_label="O(n log n)"))
register(Algorithm("edit_distance", "dp", edit_distance, check_equals("expected"), [_edit_cases],
                   scaler=_str_scaler, expected_exponent=2.0, complexity_label="O(m*n)"))
register(Algorithm("longest_common_subsequence", "dp", longest_common_subsequence, check_equals("expected"),
                   [_lcs_cases], scaler=_str_scaler, expected_exponent=2.0, complexity_label="O(m*n)"))
register(Algorithm("coin_change", "dp", coin_change, check_equals("expected"), [_coin_cases],
                   scaler=_coin_scaler, expected_exponent=1.0, complexity_label="O(amount*coins)"))
