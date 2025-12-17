from __future__ import annotations

import numpy as np

from .graph import D0Graph
from .info import mutual_information


def mi_matrix(rho: np.ndarray, dims: list[int] | tuple[int, ...]) -> np.ndarray:
    n = len(dims)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            mij = float(mutual_information(rho, A=(i,), B=(j,), dims=dims))
            M[i, j] = mij
            M[j, i] = mij
    return M


def rewire_topk(
    graph: D0Graph,
    mi: np.ndarray,
    *,
    k_edges: int,
    keep_fraction: float = 0.5,
) -> D0Graph:
    """
    Build a new graph with k_edges undirected edges:
    - keep_fraction of current edges are preserved (highest MI among existing edges)
    - remaining edges are filled by highest MI pairs overall (excluding self, duplicates)
    """
    n = int(mi.shape[0])
    if n < 2:
        return graph

    # List all candidate pairs (i<j)
    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(mi[i, j]), i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    # Rank existing edges by MI (normalize to i<j)
    existing: list[tuple[float, int, int]] = []
    for (i, j) in graph.edges:
        a, b = (i, j) if i < j else (j, i)
        existing.append((float(mi[a, b]), a, b))
    existing.sort(reverse=True, key=lambda x: x[0])

    keep_n = int(round(keep_fraction * min(len(existing), k_edges)))
    kept = {(i, j) for _, i, j in existing[:keep_n]}

    # Fill remaining slots from global top MI pairs
    new_edges: set[tuple[int, int]] = set(kept)
    for _, i, j in pairs:
        if len(new_edges) >= k_edges:
            break
        if (i, j) in new_edges:
            continue
        new_edges.add((i, j))

    return D0Graph(n, new_edges)
