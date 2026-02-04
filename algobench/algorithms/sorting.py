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


def merge_sort(arr: List[int]) -> List[int]:
    a = list(arr)
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    merged: List[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def quick_sort(arr: List[int]) -> List[int]:
    """Iterative quicksort with median-of-three pivot to dodge the classic
    already-sorted O(n^2) trap."""
    a = list(arr)
    if len(a) <= 1:
        return a
    stack = [(0, len(a) - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue
        mid = (lo + hi) // 2
        # median-of-three pivot selection
        trio = sorted((a[lo], a[mid], a[hi]))
        pivot = trio[1]
        i, j = lo, hi
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        stack.append((lo, j))
        stack.append((i, hi))
    return a
