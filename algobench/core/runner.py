"""Execute test cases against algorithms and collect timed results."""

from __future__ import annotations

import time
import traceback
from typing import Dict, List, Optional

from algobench.core.case import CaseResult, TestCase
from algobench.core.registry import Algorithm, get_registry

# Payload keys that are metadata for the checker rather than function
# arguments. They are visible to the checker but stripped before the call.
RESERVED_KEYS = frozenset({"expected"})  # checker-only metadata


class BenchmarkRunner:
    """Runs every registered algorithm over its generated cases.

    Parameters
    ----------
    repeats:
        How many times to execute each case; the *minimum* wall-clock time is
        kept to reduce scheduler noise.
    """

    def __init__(self, repeats: int = 1) -> None:
        self.repeats = max(1, repeats)

    # ------------------------------------------------------------------ #
    def run_case(self, algo: Algorithm, case: TestCase) -> CaseResult:
        violations: List[str] = []
        error: Optional[str] = None
        passed = False
        best = float("inf")
        output = None
        call_kwargs = {
            k: v for k, v in case.payload.items() if k not in RESERVED_KEYS
        }

        for _ in range(self.repeats):
            start = time.perf_counter()
            try:
                output = algo.func(**call_kwargs)
            except Exception as exc:  # noqa: BLE001 — we record any failure
                elapsed = time.perf_counter() - start
                best = min(best, elapsed)
                if case.expect_error and isinstance(exc, case.expect_error):
                    # Defensive behaviour was *expected*: this is a pass.
                    passed = True
                    error = None
                else:
                    error = f"{type(exc).__name__}: {exc}"
                    error_tb = traceback.format_exc(limit=1)
                    error = error or error_tb
                break
            else:
                elapsed = time.perf_counter() - start
                best = min(best, elapsed)
        else:
            # loop completed without break -> no exception on the last run
            pass

        if error is None and not (case.expect_error and passed):
            if case.expect_error:
                # We expected an error but none was raised.
                violations.append(
                    f"expected {case.expect_error.__name__} but none was raised"
                )
                passed = False
            else:
                violations = algo.checker(case.payload, output)
                passed = len(violations) == 0

        return CaseResult(
            algorithm=algo.name,
            category=algo.category,
            case=case.name,
            kind=case.kind,
            tags=case.tags,
            size=case.size,
            passed=passed,
            elapsed_s=best if best != float("inf") else 0.0,
            violations=violations,
            error=error,
        )

    # ------------------------------------------------------------------ #
    def run_all(self, registry: Optional[Dict[str, Algorithm]] = None) -> List[CaseResult]:
        registry = registry or get_registry()
        results: List[CaseResult] = []
        for algo in registry.values():
            for case in algo.cases():
                results.append(self.run_case(algo, case))
        return results
