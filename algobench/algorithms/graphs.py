"""Graph algorithms over adjacency-dict representations.

Unweighted graphs: ``{u: [v, ...]}``. Weighted graphs: ``{u: [(v, w), ...]}``.
Payloads always carry ``n`` (vertex count) and ``start``.

Checkers validate structural invariants and, for Dijkstra, cross-check against
SciPy's ``scipy.sparse.csgraph.dijkstra`` as an independent oracle.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, List

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from algobench.core.generators import (
    adversarial_graphs,
    adversarial_weighted_graphs,
    random_graphs,
    random_weighted_graphs,
)
from algobench.core.registry import Algorithm, register


# --------------------------------------------------------------------------- #
# Reference helpers (independent of the implementations under test)
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


def dijkstra(graph: dict, n: int, start: int) -> Dict[int, float]:
    """Shortest-path distances from ``start`` to every vertex (``inf`` when
    unreachable). Non-negative weights assumed."""
    dist = {v: float("inf") for v in range(n)}
    if n == 0 or start not in graph:
        return dist
    dist[start] = 0.0
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def topological_sort(graph: dict, n: int, start: int):
    """Kahn's algorithm. Returns a valid topological order, or ``None`` when
    the graph contains a cycle (no valid order exists)."""
    indeg = {v: 0 for v in range(n)}
    for u in range(n):
        for v in graph.get(u, []):
            indeg[v] += 1
    q = deque([v for v in range(n) if indeg[v] == 0])
    order: List[int] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else None


def has_cycle(graph: dict, n: int, start: int) -> bool:
    return not _is_dag(graph, n)


def connected_components(graph: dict, n: int, start: int) -> int:
    return _wcc_count(graph, n)


# --------------------------------------------------------------------------- #
# Checkers
# --------------------------------------------------------------------------- #
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


def check_dijkstra(payload: dict, output) -> List[str]:
    graph, n, start = payload["graph"], payload["n"], payload["start"]
    if n == 0:
        return [] if not output else ["expected empty distances for empty graph"]
    # Build dense matrix for SciPy oracle.
    mat = np.zeros((n, n), dtype=float)
    for u in range(n):
        for v, w in graph.get(u, []):
            mat[u, v] = w
    ref = scipy_dijkstra(csr_matrix(mat), indices=start)
    violations: List[str] = []
    for v in range(n):
        got = output.get(v, float("inf"))
        exp = ref[v]
        if np.isinf(exp) and np.isinf(got):
            continue
        if not np.isclose(got, exp):
            violations.append(f"dist[{v}]={got} but SciPy oracle says {exp}")
    return violations


def check_topo(payload: dict, output) -> List[str]:
    graph, n = payload["graph"], payload["n"]
    if not _is_dag(graph, n):
        return [] if output is None else ["cyclic graph must yield None"]
    if output is None:
        return ["DAG must yield a topological order, got None"]
    if sorted(output) != list(range(n)):
        return ["topological order is not a permutation of all vertices"]
    pos = {v: i for i, v in enumerate(output)}
    violations: List[str] = []
    for u in range(n):
        for v in graph.get(u, []):
            if pos[u] > pos[v]:
                violations.append(f"edge {u}->{v} violates topological order")
    return violations


def check_has_cycle(payload: dict, output) -> List[str]:
    expected = not _is_dag(payload["graph"], payload["n"])
    return [] if bool(output) == expected else [f"expected has_cycle={expected}, got {output}"]


def check_components(payload: dict, output) -> List[str]:
    expected = _wcc_count(payload["graph"], payload["n"])
    return [] if output == expected else [f"expected {expected} components, got {output}"]


# --------------------------------------------------------------------------- #
# Complexity scalers (sparse line-ish graphs)
# --------------------------------------------------------------------------- #
def _unweighted_scaler(n: int) -> dict:
    rng = np.random.default_rng(n)
    graph = {v: [] for v in range(n)}
    for u in range(n - 1):
        graph[u].append(u + 1)  # backbone path keeps it connected
        # a few random forward edges for realistic branching
        for _ in range(2):
            graph[u].append(int(rng.integers(0, n)))
    return {"graph": graph, "n": n, "start": 0}


def _weighted_scaler(n: int) -> dict:
    rng = np.random.default_rng(n)
    graph = {v: [] for v in range(n)}
    for u in range(n - 1):
        graph[u].append((u + 1, int(rng.integers(1, 20))))
        graph[u].append((int(rng.integers(0, n)), int(rng.integers(1, 20))))
    return {"graph": graph, "n": n, "start": 0}


_UNW_GENS = [random_graphs, adversarial_graphs]
_W_GENS = [random_weighted_graphs, adversarial_weighted_graphs]

register(Algorithm("bfs", "graph", bfs, check_traversal, _UNW_GENS,
                   scaler=_unweighted_scaler, expected_exponent=1.0, complexity_label="O(V+E)"))
register(Algorithm("dfs", "graph", dfs, check_traversal, _UNW_GENS,
                   scaler=_unweighted_scaler, expected_exponent=1.0, complexity_label="O(V+E)"))
register(Algorithm("topological_sort", "graph", topological_sort, check_topo, _UNW_GENS,
                   scaler=_unweighted_scaler, expected_exponent=1.0, complexity_label="O(V+E)"))
register(Algorithm("has_cycle", "graph", has_cycle, check_has_cycle, _UNW_GENS,
                   expected_exponent=1.0, complexity_label="O(V+E)"))
register(Algorithm("connected_components", "graph", connected_components, check_components, _UNW_GENS,
                   expected_exponent=1.0, complexity_label="O(V+E)"))
register(Algorithm("dijkstra", "graph", dijkstra, check_dijkstra, _W_GENS,
                   scaler=_weighted_scaler, expected_exponent=1.2, complexity_label="O((V+E)logV)"))
