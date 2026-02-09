"""Aggregate raw results into Pandas DataFrames and printable summaries."""

from __future__ import annotations

from typing import List

import pandas as pd

from algobench.core.case import CaseResult


def results_frame(results: List[CaseResult]) -> pd.DataFrame:
    """One row per executed test case."""
    return pd.DataFrame([r.to_row() for r in results])


def correctness_by_algorithm(df: pd.DataFrame) -> pd.DataFrame:
    """Correctness rate + timing summary per algorithm."""
    g = df.groupby(["category", "algorithm"], as_index=False).agg(
        cases=("passed", "size"),
        passed=("passed", "sum"),
        violations=("n_violations", "sum"),
        mean_ms=("elapsed_s", lambda s: round(s.mean() * 1e3, 4)),
        max_ms=("elapsed_s", lambda s: round(s.max() * 1e3, 4)),
    )
    g["correctness_rate"] = (g["passed"] / g["cases"]).round(4)
    return g.sort_values(["category", "algorithm"]).reset_index(drop=True)


def correctness_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Roll correctness up to the algorithm category."""
    g = df.groupby("category", as_index=False).agg(
        cases=("passed", "size"),
        passed=("passed", "sum"),
        violations=("n_violations", "sum"),
    )
    g["correctness_rate"] = (g["passed"] / g["cases"]).round(4)
    return g.sort_values("category").reset_index(drop=True)


def adversarial_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Compare pass rates on random vs adversarial cases per category."""
    pivot = (
        df.pivot_table(
            index="category",
            columns="kind",
            values="passed",
            aggfunc="mean",
        )
        .round(4)
        .reset_index()
    )
    pivot.columns.name = None
    return pivot


def violations_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Every case that recorded at least one constraint violation or error."""
    bad = df[(df["n_violations"] > 0) | (df["error"] != "")]
    cols = ["category", "algorithm", "case", "kind", "violations", "error"]
    return bad[cols].reset_index(drop=True)


def complexity_frame(records: List[dict]) -> pd.DataFrame:
    """Observed vs expected complexity exponents."""
    if not records:
        return pd.DataFrame(
            columns=[
                "algorithm",
                "category",
                "complexity_label",
                "expected_exponent",
                "observed_exponent",
                "deviation",
                "r_squared",
                "n_points",
            ]
        )
    df = pd.DataFrame(records)
    return df.sort_values(["category", "algorithm"]).reset_index(drop=True)
