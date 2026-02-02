"""Data structures describing a single benchmark case and its result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TestCase:
    """A single input fed to an algorithm.

    Attributes
    ----------
    name:
        Human-readable identifier, unique within an algorithm.
    payload:
        Keyword arguments passed to the algorithm implementation.
    kind:
        ``"random"`` or ``"adversarial"``. Used to slice reports.
    tags:
        Free-form labels describing the stress scenario, e.g.
        ``{"empty", "overflow", "cyclic"}``.
    size:
        The problem size *n* used for complexity analysis. ``None`` when the
        notion of size does not apply.
    expect_error:
        When set, the algorithm is *expected* to raise this exception type
        (adversarial cases that probe defensive behaviour). A case passes if
        the matching error is raised.
    """

    # Tell pytest this is data, not a test class to collect.
    __test__ = False

    name: str
    payload: Dict[str, Any]
    kind: str = "random"
    tags: frozenset = field(default_factory=frozenset)
    size: Optional[int] = None
    expect_error: Optional[type] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tags, frozenset):
            self.tags = frozenset(self.tags)


@dataclass
class CaseResult:
    """The outcome of running one :class:`TestCase` against one algorithm."""

    algorithm: str
    category: str
    case: str
    kind: str
    tags: frozenset
    size: Optional[int]
    passed: bool
    elapsed_s: float
    violations: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        """Flatten into a dict suitable for a Pandas DataFrame row."""
        return {
            "algorithm": self.algorithm,
            "category": self.category,
            "case": self.case,
            "kind": self.kind,
            "tags": ",".join(sorted(self.tags)),
            "size": self.size,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "n_violations": len(self.violations),
            "violations": "; ".join(self.violations),
            "error": self.error or "",
        }
