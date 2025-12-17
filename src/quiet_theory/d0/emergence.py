from __future__ import annotations

import numpy as np

from .info import mutual_information


def mi_matrix(rho: np.ndarray, dims: list[int] | tuple[int, ...]) -> np.ndarray:
    """
    Pairwise mutual information matrix MI[i,j] for single-site subsystems.
    """
    n = len(dims)
    MI = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            mij = mutual_information(rho, A=(i,), B=(j,), dims=dims)
            MI[i, j] = MI[j, i] = float(mij)
    return MI


def mi_distance(MI: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert MI matrix to a distance-like matrix using negative log scaling.
    """
    MI = np.asarray(MI, dtype=float)
    mmax = float(np.max(MI))
    if mmax <= eps:
        # no correlations -> large distances
        return np.full_like(MI, fill_value=50.0, dtype=float)
    X = MI / mmax
    return -np.log(eps + X)


def emergent_edges(MI: np.ndarray, tau: float) -> set[tuple[int, int]]:
    """
    Build an emergent undirected edge set by thresholding mutual information.
    """
    MI = np.asarray(MI, dtype=float)
    n = MI.shape[0]
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if MI[i, j] > tau:
                edges.add((i, j))
    return edges
