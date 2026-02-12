# Algorithm Correctness Benchmark Suite

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-013243.svg?logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.x-8CAAE6.svg?logo=scipy)](https://scipy.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458.svg?logo=pandas)](https://pandas.pydata.org/)
[![Tests](https://img.shields.io/badge/tests-11%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A structured benchmark suite that evaluates **23 classical computer-science
algorithms** across **550+ test cases**, scoring outputs against
**expected constraints** (invariants) rather than a single reference answer.
It ships **adversarial edge-case generators**, validates results with
**NumPy array comparisons**, cross-checks against **SciPy** oracles, runs an
**empirical time-complexity profiler**, and summarises everything in
**Pandas DataFrames**.

```
Total cases     : 550
Passed          : 550  (100.00%)
Adversarial     : 114
```

---

## Table of Contents

- [Why constraint-based testing?](#why-constraint-based-testing)
- [Key features](#key-features)
- [Architecture](#architecture)
- [Algorithm coverage](#algorithm-coverage)
- [Adversarial edge cases](#adversarial-edge-cases)
- [Quickstart](#quickstart)
- [Sample output](#sample-output)
- [How each layer works](#how-each-layer-works)
- [Output artefacts](#output-artefacts)
- [Testing](#testing)
- [Extending the suite](#extending-the-suite)
- [Project layout](#project-layout)
- [License](#license)

---

## Why constraint-based testing?

Most algorithm test harnesses compare an output to one precomputed "expected"
value. That misses an entire class of bugs and couples your test to one input.
This suite instead validates the **invariants** an output must satisfy:

| Algorithm class | Constraints checked (not just equality) |
| --------------- | --------------------------------------- |
| Sorting         | output length unchanged **and** non-decreasing **and** a permutation of the input |
| Searching       | returned index actually holds the target, or `-1` *only* when the target is truly absent |
| Graph traversal | visited set equals the set reachable from `start`, no vertex visited twice |
| Shortest paths  | distances match an independent **SciPy** Dijkstra oracle |
| Topological sort| every edge `u→v` places `u` before `v`; cyclic graphs must yield `None` |
| DP              | result equals an **independent** oracle (brute force / alternate recurrence) |

A sort that silently drops a duplicate still *looks* sorted — the permutation
invariant catches it. That "teeth" is verified by the test suite itself, which
injects deliberately buggy implementations and asserts they are rejected
(see [`tests/test_suite.py`](tests/test_suite.py)).

## Key features

- **23 algorithms / 6 categories**, each registered once with its
  implementation, constraint checker, generators, and declared complexity.
- **550+ generated test cases**, split into `random` (broad coverage) and
  `adversarial` (targeted stress) — fully seeded for reproducibility.
- **Adversarial generators**: empty/singleton inputs, `int64` overflow values,
  already/reverse-sorted arrays (quicksort worst case), cyclic & disconnected
  graphs, negative weights, infeasible coin-change, and more.
- **NumPy-powered checkers** — vectorised `is_sorted` / permutation / equality
  comparisons that handle scalars, lists and arrays uniformly.
- **SciPy oracle cross-checks** — Dijkstra distances are validated against
  `scipy.sparse.csgraph.dijkstra`.
- **Empirical complexity profiler** — times each algorithm across a geometric
  spread of sizes and fits a log-log slope (`scipy.stats.linregress`) to expose
  scaling that deviates from the textbook bound.
- **Pandas reporting** — correctness rates by algorithm and category,
  random-vs-adversarial pass rates, a violations ledger, and an observed-vs-
  expected complexity table — all also written to CSV.

## Architecture

```
                         ┌─────────────────────┐
        register() ─────▶│   Algorithm Registry │  (impl + checker + gens + Big-O)
                         └──────────┬──────────┘
                                    │
        generators  ───────────────┤ random + adversarial TestCases
                                    │
                          ┌─────────▼─────────┐
                          │  BenchmarkRunner  │  time each case, run checker
                          └─────────┬─────────┘
                                    │ CaseResult[]
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
       NumPy checkers        Complexity profiler      Pandas report
   (invariant violations)  (scipy log-log slope)   (DataFrames + CSV)
```

Each algorithm is a single declarative `Algorithm` record:

```python
register(Algorithm(
    name="merge_sort",
    category="sorting",
    func=merge_sort,
    checker=check_sort,                       # NumPy invariant checker
    generators=[random_arrays, adversarial_arrays],
    scaler=_sort_scaler,                       # payload factory for profiling
    expected_exponent=1.1, complexity_label="O(n log n)",
))
```

## Algorithm coverage

| Category   | Algorithms |
| ---------- | ---------- |
| **Sorting** (5)   | insertion sort, bubble sort, merge sort, quick sort (median-of-three), heap sort |
| **Searching** (2) | linear search, binary search |
| **Graph** (6)     | BFS, DFS (iterative), Dijkstra, topological sort (Kahn), cycle detection, connected components |
| **Dynamic programming** (5) | 0/1 knapsack, longest increasing subsequence, edit distance, longest common subsequence, coin change |
| **Numeric** (4)   | GCD (Euclid), modular exponentiation, Fibonacci (fast doubling), Sieve of Eratosthenes |
| **String** (1)    | KMP substring search |

**Total: 23 algorithms across 6 categories.**

## Adversarial edge cases

The generators deliberately probe the brittle assumptions of each class:

- **Empty / singleton inputs** — base-case and off-by-one bugs.
- **Integer overflow** — values at `np.iinfo(np.int64).max` that overflow naive
  fixed-width arithmetic (and the `(lo+hi)` binary-search midpoint idiom).
- **Degenerate order** — already-sorted and reverse-sorted arrays (the classic
  quicksort O(n²) trap), all-equal elements, all-negative arrays.
- **Cyclic graphs** — self-loops and 3-cycles that must be detected; cyclic
  input to topological sort must return `None`.
- **Disconnected / unreachable** — components and unreachable Dijkstra targets
  (distance must stay `inf`).
- **Infeasible instances** — coin change with no solution (`-1`), knapsack where
  every item exceeds capacity (`0`).

## Quickstart

```bash
# 1. clone
git clone https://github.com/Jsingh651/algorithm-correctness-benchmark.git
cd algorithm-correctness-benchmark

# 2. create an environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # numpy, scipy, pandas

# 3. run the full suite
python -m algobench
```

Useful flags:

```bash
python -m algobench --repeats 5        # keep min time over 5 runs (less noise)
python -m algobench --no-complexity    # skip the profiling phase
python -m algobench --outdir out       # write CSV artefacts to ./out
```

Random and adversarial generators use a fixed seed (`SEED = 1234`) so every run
is reproducible across machines.

## Sample output

```
Loaded 23 algorithms across 6 categories.

==============================================================================
CORRECTNESS BY ALGORITHM  (550 total test cases)
==============================================================================
 category                      algorithm  cases  passed  mean_ms  max_ms  correctness_rate
       dp                    coin_change     15      15   0.0048  0.0098               1.0
    graph                       dijkstra      9       9   0.0144  0.0627               1.0
  sorting                    bubble_sort     48      48   0.2251  1.9266               1.0
  sorting                     merge_sort     48      48   0.0577  0.3837               1.0
  ...

==============================================================================
RANDOM vs ADVERSARIAL PASS RATE (by category)
==============================================================================
 category  adversarial  random
       dp          1.0     1.0
    graph          1.0     1.0
  sorting          1.0     1.0

==============================================================================
CONSTRAINT VIOLATIONS / ERRORS
==============================================================================
None — every constraint held across all cases. ✅

==============================================================================
EMPIRICAL TIME-COMPLEXITY DEVIATION
==============================================================================
                     algorithm  complexity_label  expected_exponent  observed_exponent  deviation  r_squared
                insertion_sort            O(n^2)                2.0              2.042      0.042     0.9996
                    bubble_sort            O(n^2)                2.0              2.059      0.059     0.9996
                    merge_sort        O(n log n)                1.1              1.116      0.016     1.0000
                    quick_sort        O(n log n)                1.1              1.120      0.020     0.9998
                     heap_sort        O(n log n)                1.1              1.207      0.107     1.0000
                 linear_search              O(n)                1.0              0.999     -0.001     0.9984
                   knapsack_01            O(n*W)                2.0              2.086      0.086     0.9999
```

The profiler cleanly separates the **quadratic** sorts (observed exponent ≈ 2.0)
from the **linearithmic** ones (≈ 1.1) by fitting the slope of `log(time)` vs
`log(n)` — and the high `r_squared` confirms the fit. A real quadratic bug in an
algorithm advertised as `O(n log n)` would show up here as a large positive
`deviation`.

## How each layer works

| Module | Responsibility |
| ------ | -------------- |
| [`core/case.py`](algobench/core/case.py)         | `TestCase` (input + metadata + optional expected error) and `CaseResult`. |
| [`core/registry.py`](algobench/core/registry.py) | Global `Algorithm` registry binding impl + checker + generators + Big-O. |
| [`core/checkers.py`](algobench/core/checkers.py) | Reusable NumPy invariant checkers (`is_sorted`, permutation, equality, search). |
| [`core/generators.py`](algobench/core/generators.py) | Seeded random + adversarial case factories for arrays and graphs. |
| [`core/runner.py`](algobench/core/runner.py)     | Executes each case, times it (min of N repeats), runs the checker, handles expected errors. |
| [`core/complexity.py`](algobench/core/complexity.py) | Times across sizes and fits a log-log exponent with `scipy.stats.linregress`. |
| [`core/report.py`](algobench/core/report.py)     | Aggregates `CaseResult`s into Pandas DataFrames. |
| [`cli.py`](algobench/cli.py)                      | Orchestrates a run, prints tables, writes CSVs. |

## Output artefacts

Running `python -m algobench` writes (to `results/` by default):

| File | Contents |
| ---- | -------- |
| `raw_results.csv`               | One row per test case: pass/fail, timing, tags, violations. |
| `correctness_by_algorithm.csv`  | Per-algorithm correctness rate and timing summary. |
| `correctness_by_category.csv`   | Correctness rolled up to category. |
| `adversarial_breakdown.csv`     | Random vs adversarial pass rates per category. |
| `violations.csv`                | Every case that violated a constraint or raised an error. |
| `complexity.csv`                | Observed vs expected complexity exponents + R². |

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests assert both directions of correctness:

- **Acceptance** — every shipped algorithm passes all of its generated cases.
- **Rejection** — injected buggy implementations (e.g. a sort that drops the
  last element) are *caught* by the checkers, and the complexity profiler
  distinguishes a quadratic from a linearithmic implementation.

```
11 passed in 1.45s
```

## Extending the suite

Add a new algorithm in three steps:

```python
# 1. implement it
def counting_sort(arr): ...

# 2. (re)use a checker — sorting invariants already exist
from algobench.core.checkers import check_sort

# 3. register it
from algobench.core.registry import Algorithm, register
from algobench.core.generators import random_arrays, adversarial_arrays

register(Algorithm(
    "counting_sort", "sorting", counting_sort, check_sort,
    [random_arrays, adversarial_arrays],
    scaler=lambda n: {"arr": list(range(n, 0, -1))},
    expected_exponent=1.0, complexity_label="O(n+k)",
))
```

It is immediately picked up by the runner, reports, and complexity profiler.

## Project layout

```
algobench/
├── __init__.py
├── __main__.py            # python -m algobench
├── cli.py                 # orchestration + printing + CSV
├── algorithms/            # implementations + registrations
│   ├── sorting.py
│   ├── searching.py
│   ├── graphs.py
│   ├── dynamic.py
│   ├── numeric.py
│   └── strings.py
└── core/
    ├── case.py            # TestCase / CaseResult
    ├── registry.py        # Algorithm registry
    ├── checkers.py        # NumPy invariant checkers
    ├── generators.py      # random + adversarial generators
    ├── runner.py          # execution + timing
    ├── complexity.py      # scipy log-log exponent fit
    └── report.py          # pandas aggregation
tests/
└── test_suite.py          # framework self-tests (accept + reject)
```

## License

[MIT](LICENSE)
