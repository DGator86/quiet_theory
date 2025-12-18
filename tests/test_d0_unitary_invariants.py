from __future__ import annotations

import numpy as np

from quiet_theory.d0 import D0Graph, D0Model, apply_two_site_unitary, mi_matrix


def _haar_unitary(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(X)
    phases = np.diag(R)
    phases = phases / np.where(np.abs(phases) == 0, 1.0, np.abs(phases))
    return Q * phases


def test_two_site_unitary_preserves_norm_and_accepts_aliases() -> None:
    dims = [2, 2, 2]
    rng = np.random.default_rng(42)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi = psi.astype(np.complex128)
    psi = psi / np.linalg.norm(psi)

    U = _haar_unitary(4, seed=7)
    psi2 = apply_two_site_unitary(psi, U=U, dims=dims, i=0, j=2)

    assert np.isclose(np.linalg.norm(psi2), 1.0, atol=1e-10)


def test_rho_is_hermitian_with_unit_trace() -> None:
    g = D0Graph.chain(3)
    dims = [2, 2, 2]
    psi = np.zeros(8, dtype=np.complex128)
    psi[0] = 1.0
    model = D0Model(graph=g, dims=dims, psi=psi)

    rho = model.rho()
    assert np.allclose(rho, rho.conj().T)
    assert np.isclose(np.trace(rho), 1.0)


def test_mi_matrix_symmetry_and_zero_diagonal() -> None:
    rng = np.random.default_rng(0)
    dims = [2, 2, 2, 2]
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi = psi.astype(np.complex128)
    psi = psi / np.linalg.norm(psi)

    model = D0Model(graph=D0Graph.chain(len(dims)), dims=dims, psi=psi)
    M = mi_matrix(model.rho(), dims)

    assert np.allclose(M, M.T, atol=1e-12)
    assert np.allclose(np.diag(M), 0.0, atol=1e-12)
