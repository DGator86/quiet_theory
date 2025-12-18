from __future__ import annotations

import numpy as np

from .graph import D0Graph
from .model import D0Model
from .objective import objective_locality
from .ops import apply_two_site_unitary


def random_unitary_4(rng: np.random.Generator) -> np.ndarray:
    """
    Haar-ish random 4x4 unitary via QR decomposition of random complex matrix.
    """
    X = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    Q, R = np.linalg.qr(X)
    d = np.diag(R)
    ph = d / np.where(np.abs(d) == 0, 1.0, np.abs(d))
    return Q * ph


def greedy_edge_update(
    model: D0Model,
    i: int,
    j: int,
    *,
    trials: int = 32,
    lam: float = 0.25,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Try `trials` random two-site unitaries on edge (i,j). Keep the best improvement.
    Returns delta_F (new - old). If no improvement, does nothing and returns <= 0.
    """
    if rng is None:
        rng = np.random.default_rng()

    rho0 = model.rho()
    f0 = objective_locality(rho0, model.graph, model.dims, lam=lam)

    best_U = None
    best_f = f0

    psi_orig = model.psi.copy()

    for _ in range(trials):
        U = random_unitary_4(rng)
        model.psi = psi_orig.copy()
        model.apply_edge_unitary(i, j, U)
        f = objective_locality(model.rho(), model.graph, model.dims, lam=lam)
        if f > best_f + 1e-12:
            best_f = f
            best_U = U

    model.psi = psi_orig.copy()
    if best_U is not None:
        model.apply_edge_unitary(i, j, best_U)

    return float(best_f - f0)


def _haar_random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Haar-random unitary via QR decomposition of a complex Ginibre matrix.
    """
    X = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))).astype(np.complex128)
    Q, R = np.linalg.qr(X)
    # Fix phases to ensure unitary distributed correctly
    diag = np.diag(R)
    ph = diag / np.where(np.abs(diag) == 0, 1.0, np.abs(diag))
    Q = Q * ph.conj()
    return Q


def apply_random_two_site_unitary_step(
    *,
    psi: np.ndarray,
    graph: D0Graph,
    dims: list[int] | tuple[int, ...] | np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Pick a random existing edge (i,j), apply a Haar-random 2-site unitary on that pair.
    """
    dims_list = [int(d) for d in dims]
    n = len(dims_list)

    edges = [
        (int(a), int(b))
        for a, b in graph.edges
        if 0 <= int(a) < n and 0 <= int(b) < n and int(a) != int(b)
    ]
    if not edges:
        return psi

    i, j = edges[int(rng.integers(0, len(edges)))]
    di = dims_list[i]
    dj = dims_list[j]
    U = _haar_random_unitary(di * dj, rng)
    return apply_two_site_unitary(psi, U, i, j, dims_list)
