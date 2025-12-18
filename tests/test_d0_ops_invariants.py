from __future__ import annotations

import numpy as np
import pytest

from quiet_theory.d0 import (
    D0Graph,
    D0Model,
    EvolutionConfig,
    apply_two_site_unitary,
    evolve,
    mi_matrix,
)


def _random_unitary(dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(X)
    ph = np.diag(R) / np.where(np.abs(np.diag(R)) == 0, 1.0, np.abs(np.diag(R)))
    return Q * ph


def test_apply_edge_unitary_matches_expected_entangled_state() -> None:
    g = D0Graph.chain(2)
    dims = [2, 2]
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = 1.0

    model = D0Model(graph=g, dims=dims, psi=psi)

    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    U_hi = np.kron(H, np.eye(2, dtype=np.complex128))
    model.apply_edge_unitary(0, 1, U_hi)

    CNOT = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex128,
    )
    model.apply_edge_unitary(0, 1, CNOT)

    expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    assert np.allclose(model.psi, expected)
    assert np.isclose(np.linalg.norm(model.psi), 1.0)


def test_apply_two_site_unitary_preserves_norm() -> None:
    rng = np.random.default_rng(1)
    dims = [2, 2, 2]
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi = psi.astype(np.complex128)
    psi = psi / np.linalg.norm(psi)

    U = _random_unitary(4, seed=2)
    psi2 = apply_two_site_unitary(psi, U, 1, 2, dims)

    assert np.isclose(np.linalg.norm(psi2), 1.0, atol=1e-10)


def test_apply_two_site_unitary_out_of_range_raises() -> None:
    psi = np.ones(4, dtype=np.complex128) / 2.0
    U = np.eye(4, dtype=np.complex128)
    with pytest.raises(ValueError):
        apply_two_site_unitary(psi, U, -1, 1, [2, 2])
    with pytest.raises(ValueError):
        apply_two_site_unitary(psi, U, 0, 2, [2, 2])


def test_mi_matrix_is_symmetric_with_zero_diagonal() -> None:
    rng = np.random.default_rng(3)
    dims = [2, 2, 2]
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi = psi.astype(np.complex128)
    psi = psi / np.linalg.norm(psi)

    model = D0Model(graph=D0Graph.chain(3), dims=dims, psi=psi)
    M = mi_matrix(model.rho(), dims)

    assert np.allclose(M, M.T, atol=1e-12)
    assert np.allclose(np.diag(M), 0.0, atol=1e-12)


def test_evolution_config_accepts_state_updates_and_rewires() -> None:
    rng = np.random.default_rng(4)
    dims = [2, 2, 2]
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi = psi.astype(np.complex128)
    psi = psi / np.linalg.norm(psi)

    model = D0Model(graph=D0Graph.chain(3), dims=dims, psi=psi)

    cfg = EvolutionConfig(
        steps=1,
        k_edges=2,
        keep_fraction=0.5,
        state_updates_per_step=1,
        unitary_trials=2,
        seed=5,
    )

    evolve(model, cfg, rng=rng)

    assert isinstance(cfg.state_updates_per_step, int)
    assert len(model.graph.edges) == 2
