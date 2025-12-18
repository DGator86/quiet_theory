from __future__ import annotations

import numpy as np

from .info import mutual_information
from .graph import D0Graph


def objective_locality(rho: np.ndarray, graph: D0Graph, dims: list[int] | tuple[int, ...], lam: float = 0.25) -> float:
    """
    F = sum_{(i,j) in E} MI(i:j) - lam * sum_{i<j} MI(i:j)

    Reward edge-local mutual information; penalize global mutual information.
    """
    n = len(dims)

    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            mij = float(mutual_information(rho, A=(i,), B=(j,), dims=dims))
            total -= lam * mij

    for (i, j) in graph.edges:
        mij = float(mutual_information(rho, A=(i,), B=(j,), dims=dims))
        total += mij

    return float(total)
