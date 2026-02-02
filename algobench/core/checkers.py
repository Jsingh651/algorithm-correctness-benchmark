"""Reusable constraint checkers built on NumPy array comparisons.

A checker validates *invariants* of an algorithm's output rather than diffing
against a single reference answer. This catches a broader class of bugs: an
implementation can return the "right shape" while violating an invariant
(e.g. a sort that drops duplicates), and it lets us score partial constraint
violations independently of strict correctness.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def is_sorted(arr: Sequence) -> bool:
    """True iff ``arr`` is non-decreasing (NumPy vectorised comparison)."""
    a = np.asarray(arr)
    if a.size <= 1:
        return True
    return bool(np.all(a[:-1] <= a[1:]))


def is_permutation(out: Sequence, src: Sequence) -> bool:
    """True iff ``out`` is a permutation of ``src`` (multiset equality)."""
    a = np.sort(np.asarray(out))
    b = np.sort(np.asarray(src))
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b))


def check_sort(payload: dict, output) -> List[str]:
    """Constraints for any sorting routine."""
    src = payload["arr"]
    violations: List[str] = []
    if output is None:
        return ["output is None"]
    out = np.asarray(output)
    if out.size != len(src):
        violations.append(
            f"length changed: expected {len(src)}, got {out.size}"
        )
    if not is_sorted(out):
        violations.append("output is not non-decreasing")
    if not is_permutation(out, src):
        violations.append("output is not a permutation of the input")
    return violations


def check_equals(expected_key: str):
    """Build a checker asserting ``output`` equals ``payload[expected_key]``.

    Comparison is done with NumPy so it handles scalars, lists and arrays
    uniformly.
    """

    def checker(payload: dict, output) -> List[str]:
        expected = payload[expected_key]
        if np.isscalar(expected) or isinstance(expected, (int, float, bool)):
            if output != expected:
                return [f"expected {expected!r}, got {output!r}"]
            return []
        exp = np.asarray(expected)
        got = np.asarray(output)
        if exp.shape != got.shape:
            return [f"shape mismatch: expected {exp.shape}, got {got.shape}"]
        if not np.array_equal(exp, got):
            return [f"value mismatch: expected {exp.tolist()}, got {got.tolist()}"]
        return []

    return checker


def check_search(payload: dict, output) -> List[str]:
    """Constraints for search routines returning an index or -1."""
    arr = payload["arr"]
    target = payload["target"]
    a = np.asarray(arr)
    present = bool(np.any(a == target)) if a.size else False
    if output == -1:
        if present:
            return [f"target {target!r} is present but search returned -1"]
        return []
    if not (0 <= output < len(arr)):
        return [f"index {output} out of bounds for length {len(arr)}"]
    if arr[output] != target:
        return [f"arr[{output}]={arr[output]!r} != target {target!r}"]
    return []
