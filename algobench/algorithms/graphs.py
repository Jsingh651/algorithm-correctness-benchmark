"""Graph algorithms over adjacency-dict representations."""

from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np

from algobench.core.generators import adversarial_graphs, random_graphs
from algobench.core.registry import Algorithm, register


# Reference helpers
 (independent of the implementations under test)
# --------------------------------------------------------------------------- #
def _reachable(graph: dict, start: int, n: int) -> set:
    if n == 0 or start not in graph:
        return set()
    seen, stack = set(), [start]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        for v in graph.get(u, []):
            seen_target = v[0] if isinstance(v, tuple) else v
            stack.append(seen_target)
    return seen


def _is_dag(graph: dict, n: int) -> bool:
    color = {v: 0 for v in range(n)}  # 0=unseen,1=active,2=done

    def visit(u: int) -> bool:
        color[u] = 1
        for v in graph.get(u, []):
            if color.get(v, 0) == 1:
                return False
            if color.get(v, 0) == 0 and not visit(v):
                return False
        color[u] = 2
        return True

    return all(color[v] != 0 or visit(v) for v in range(n))


def _wcc_count(graph: dict, n: int) -> int:
    """Weakly connected component count via union-find."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for u in range(n):
        for v in graph.get(u, []):
            t = v[0] if isinstance(v, tuple) else v
            union(u, t)
    return len({find(v) for v in range(n)})


# --------------------------------------------------------------------------- #
# Implementations under test

# --------------------------------------------------------------------------- #
def bfs(graph: dict, n: int, start: int) -> List[int]:
    if n == 0 or start not in graph:
        return []
    seen = {start}
    order: List[int] = []
    q = deque([start])
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return order


def dfs(graph: dict, n: int, start: int) -> List[int]:
    """Iterative DFS (explicit stack) so it survives deep/large graphs without
    hitting Python's recursion limit."""
    if n == 0 or start not in graph:
        return []
    seen: set = set()
    order: List[int] = []
    stack = [start]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        order.append(u)
        # reverse so neighbours are explored in natural order
        for v in reversed(graph.get(u, [])):
            if v not in seen:
                stack.append(v)
    return order


# traversal checker: visit order and reachability
def check_traversal(payload: dict, output) -> List[str]:
    graph, n, start = payload["graph"], payload["n"], payload["start"]
    if output is None:
        return ["traversal returned None"]
    out = list(output)
    violations: List[str] = []
    if len(set(out)) != len(out):
        violations.append("traversal visited a vertex more than once")
    if set(out) != _reachable(graph, start, n):
        violations.append("traversal set != set of vertices reachable from start")
    if out and out[0] != start and n > 0:
        violations.append(f"traversal did not begin at start ({start})")
    return violations


_UNW_GENS = [random_graphs, adversarial_graphs]

register(Algorithm("bfs", "graph", bfs, check_traversal, _UNW_GENS,
                   expected_exponent=1.0, complexity_label="O(V+E)"))
register(Algorithm("dfs", "graph", dfs, check_traversal, _UNW_GENS,
                   expected_exponent=1.0, complexity_label="O(V+E)"))
