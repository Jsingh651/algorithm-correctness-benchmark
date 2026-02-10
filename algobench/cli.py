"""Command-line entry point: run the suite and emit DataFrame reports.

Usage
-----
    python -m algobench [--repeats N] [--no-complexity] [--outdir results]

Outputs printed tables to stdout and writes CSV artefacts to ``--outdir``.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from algobench.core.complexity import profile_all
from algobench.core.generators import reseed
from algobench.core.registry import get_registry
from algobench.core.report import (
    adversarial_breakdown,
    complexity_frame,
    correctness_by_algorithm,
    correctness_by_category,
    results_frame,
    violations_frame,
)
from algobench.core.runner import BenchmarkRunner


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="algobench", description=__doc__)
    parser.add_argument("--repeats", type=int, default=1, help="times to run each case (minimum elapsed time is kept)")
    parser.add_argument("--no-complexity", action="store_true", help="skip empirical complexity profiling")
    parser.add_argument("--outdir", default="results", help="directory for CSV output artefacts")
    args = parser.parse_args(argv)

    reseed()
    registry = get_registry()
    print(f"Loaded {len(registry)} algorithms across "
          f"{len(set(a.category for a in registry.values()))} categories.")

    runner = BenchmarkRunner(repeats=args.repeats)
    results = runner.run_all(registry)
    df = results_frame(results)

    os.makedirs(args.outdir, exist_ok=True)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)

    _print_header(f"CORRECTNESS BY ALGORITHM  ({len(df)} total test cases)")
    by_algo = correctness_by_algorithm(df)
    print(by_algo.to_string(index=False))

    _print_header("CORRECTNESS BY CATEGORY")
    by_cat = correctness_by_category(df)
    print(by_cat.to_string(index=False))

    _print_header("RANDOM vs ADVERSARIAL PASS RATE (by category)")
    adv = adversarial_breakdown(df)
    print(adv.to_string(index=False))

    _print_header("CONSTRAINT VIOLATIONS / ERRORS")
    viol = violations_frame(df)
    if viol.empty:
        print("None — every constraint held across all cases. ✅")
    else:
        print(viol.to_string(index=False))

    comp = pd.DataFrame()
    if not args.no_complexity:
        _print_header("EMPIRICAL TIME-COMPLEXITY DEVIATION")
        comp = complexity_frame(profile_all(registry))
        print(comp.to_string(index=False))

    # ----- artefacts -----
    df.to_csv(os.path.join(args.outdir, "raw_results.csv"), index=False)
    by_algo.to_csv(os.path.join(args.outdir, "correctness_by_algorithm.csv"), index=False)
    by_cat.to_csv(os.path.join(args.outdir, "correctness_by_category.csv"), index=False)
    adv.to_csv(os.path.join(args.outdir, "adversarial_breakdown.csv"), index=False)
    viol.to_csv(os.path.join(args.outdir, "violations.csv"), index=False)
    if not comp.empty:
        comp.to_csv(os.path.join(args.outdir, "complexity.csv"), index=False)

    _print_header("SUMMARY")
    total = len(df)
    passed = int(df["passed"].sum())
    print(f"Total cases     : {total}")
    print(f"Passed          : {passed}  ({passed / total:.2%})")
    print(f"Adversarial     : {int((df['kind'] == 'adversarial').sum())}")
    print(f"Artefacts written to: {os.path.abspath(args.outdir)}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
