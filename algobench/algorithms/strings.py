"""String-matching algorithms."""

from __future__ import annotations

from typing import List

import numpy as np

from algobench.core.case import TestCase
from algobench.core.registry import Algorithm, register

_rng = np.random.default_rng(7)


def kmp_search(text: str, pattern: str) -> int:
    """Index of the first occurrence of ``pattern`` in ``text`` (Knuth-Morris-
    Pratt), or ``-1``. An empty pattern matches at index 0 by convention."""
    if pattern == "":
        return 0
    # build longest-proper-prefix-suffix table
    lps = [0] * len(pattern)
    k = 0
    for i in range(1, len(pattern)):
        while k > 0 and pattern[i] != pattern[k]:
            k = lps[k - 1]
        if pattern[i] == pattern[k]:
            k += 1
        lps[i] = k
    # scan
    j = 0
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = lps[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == len(pattern):
            return i - j + 1
    return -1


def check_substring(payload: dict, output) -> List[str]:
    """Cross-check against Python's built-in ``str.find``."""
    expected = payload["text"].find(payload["pattern"])
    if output != expected:
        return [f"expected index {expected}, got {output}"]
    return []


def _kmp_cases() -> List[TestCase]:
    alphabet = "ab"
    cases: List[TestCase] = []
    for i in range(16):
        tn = (i % 8) + 4
        pn = (i % 3) + 1
        text = "".join(_rng.choice(list(alphabet), size=tn))
        # half the time, splice the pattern in so we exercise the hit path
        if i % 2 == 0:
            pattern = text[i % max(1, tn - pn): (i % max(1, tn - pn)) + pn]
        else:
            pattern = "".join(_rng.choice(list(alphabet), size=pn))
        cases.append(TestCase(f"kmp[{i}]", {"text": text, "pattern": pattern}, "random", size=tn))
    cases += [
        TestCase("kmp_empty_text", {"text": "", "pattern": "a"}, "adversarial", {"empty"}, size=0),
        TestCase("kmp_empty_pattern", {"text": "abc", "pattern": ""}, "adversarial", {"empty"}, size=3),
        TestCase("kmp_both_empty", {"text": "", "pattern": ""}, "adversarial", {"empty"}, size=0),
        TestCase("kmp_no_match", {"text": "aaaa", "pattern": "b"}, "adversarial", {"miss"}, size=4),
        TestCase("kmp_overlap", {"text": "aaaaa", "pattern": "aaa"}, "adversarial", {"overlap"}, size=5),
        TestCase("kmp_full", {"text": "abcabc", "pattern": "abcabc"}, "adversarial", {"identical"}, size=6),
    ]
    return cases


register(Algorithm("kmp_search", "string", kmp_search, check_substring, [_kmp_cases],
                   scaler=lambda n: {"text": "ab" * (n // 2), "pattern": "ab" * (max(1, n // 8))},
                   expected_exponent=1.0, complexity_label="O(n+m)"))
