from __future__ import annotations

import numpy as np

from .model import D0Model
from .objective import objective_locality


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
