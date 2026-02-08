"""Number-theoretic / numeric algorithms."""

from __future__ import annotations

import math
from typing import List

import numpy as np

from algobench.core.case import TestCase
from algobench.core.registry import Algorithm, register

_rng = np.random.default_rng(99)


# --------------------------------------------------------------------------- #
# Implementations under test
# --------------------------------------------------------------------------- #
def gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def binary_exponentiation(base: int, exp: int, mod: int) -> int:
    """Compute ``base**exp % mod`` via fast exponentiation."""
    if mod == 1:
        return 0
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result


def check_gcd(payload: dict, output) -> List[str]:
    exp = math.gcd(payload["a"], payload["b"])
    return [] if output == exp else [f"expected gcd {exp}, got {output}"]


def check_modpow(payload: dict, output) -> List[str]:
    exp = pow(payload["base"], payload["exp"], payload["mod"])
    return [] if output == exp else [f"expected {exp}, got {output}"]


def _gcd_cases() -> List[TestCase]:
    big = int(np.iinfo(np.int64).max)
    cases = [
        TestCase(f"gcd[{i}]", {"a": int(_rng.integers(0, 10_000)), "b": int(_rng.integers(1, 10_000))}, "random", size=None)
        for i in range(16)
    ]
    cases += [
        TestCase("gcd_zero_zero", {"a": 0, "b": 0}, "adversarial", {"empty", "zero"}),
        TestCase("gcd_one_zero", {"a": 17, "b": 0}, "adversarial", {"zero"}),
        TestCase("gcd_negatives", {"a": -48, "b": -36}, "adversarial", {"negative"}),
        TestCase("gcd_int64", {"a": big, "b": big - 1}, "adversarial", {"overflow"}),
        TestCase("gcd_coprime", {"a": 13, "b": 17}, "adversarial", {"coprime"}),
    ]
    return cases


def _modpow_cases() -> List[TestCase]:
    cases = [
        TestCase(
            f"modpow[{i}]",
            {"base": int(_rng.integers(0, 1000)), "exp": int(_rng.integers(0, 64)), "mod": int(_rng.integers(1, 1000)) + 1},
            "random",
        )
        for i in range(16)
    ]
    cases += [
        TestCase("modpow_exp_zero", {"base": 7, "exp": 0, "mod": 13}, "adversarial", {"zero"}),
        TestCase("modpow_mod_one", {"base": 7, "exp": 5, "mod": 1}, "adversarial", {"mod-one"}),
        TestCase("modpow_large_exp", {"base": 2, "exp": 1_000_000, "mod": 1_000_000_007}, "adversarial", {"overflow", "large"}),
        TestCase("modpow_base_zero", {"base": 0, "exp": 10, "mod": 5}, "adversarial", {"zero"}),
    ]
    return cases


def _fib_cases() -> List[TestCase]:
    cases = [TestCase(f"fib[{i}]", {"n": int(_rng.integers(0, 40))}, "random", size=None) for i in range(14)]
    cases += [
        TestCase("fib_zero", {"n": 0}, "adversarial", {"empty", "zero"}),
        TestCase("fib_one", {"n": 1}, "adversarial", {"base-case"}),
        TestCase("fib_large", {"n": 500}, "adversarial", {"overflow", "large"}),
    ]
    return cases


def _sieve_cases() -> List[TestCase]:
    cases = [TestCase(f"sieve[{i}]", {"n": int(_rng.integers(0, 500))}, "random", size=None) for i in range(14)]
    cases += [
        TestCase("sieve_zero", {"n": 0}, "adversarial", {"empty"}),
        TestCase("sieve_one", {"n": 1}, "adversarial", {"empty"}),
        TestCase("sieve_two", {"n": 2}, "adversarial", {"base-case"}),
        TestCase("sieve_large", {"n": 5000}, "adversarial", {"large"}, size=5000),
    ]
    return cases


# --------------------------------------------------------------------------- #
# Complexity scalers
# --------------------------------------------------------------------------- #
register(Algorithm("gcd", "numeric", gcd, check_gcd, [_gcd_cases],
                   expected_exponent=0.1, complexity_label="O(log min(a,b))"))
register(Algorithm("binary_exponentiation", "numeric", binary_exponentiation, check_modpow, [_modpow_cases],
                   scaler=lambda n: {"base": 3, "exp": n, "mod": 1_000_000_007},
                   expected_exponent=0.1, complexity_label="O(log exp)"))
