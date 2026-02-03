"""Sorting algorithms and their constraint-based registrations.

Every routine returns a *new* list and must satisfy the sorting invariants in
:func:`algobench.core.checkers.check_sort`: same length, non-decreasing order,
and a permutation of the input.
"""

from __future__ import annotations

from typing import List

import numpy as np

from algobench.core.checkers import check_sort
from algobench.core.generators import adversarial_arrays, random_arrays
from algobench.core.registry import Algorithm, register


def insertion_sort(arr: List[int]) -> List[int]:
    a = list(arr)
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


def bubble_sort(arr: List[int]) -> List[int]:
    a = list(arr)
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a
