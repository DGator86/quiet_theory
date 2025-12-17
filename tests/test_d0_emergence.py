from __future__ import annotations

import numpy as np

from quiet_theory.d0 import D0Graph, D0Model, emergent_edges, mi_matrix


def _H() -> np.ndarray:
    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


def _CNOT() -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.complex128,
    )


def test_emergent_graph_adds_edge_after_entangling() -> None:
    g = D0Graph.chain(3)
    dims = [2, 2, 2]

    psi = np.zeros(8, dtype=np.complex128)
    psi[0] = 1.0
    m = D0Model(graph=g, dims=dims, psi=psi)

    MI0 = mi_matrix(m.rho(), dims=dims)
    e0 = emergent_edges(MI0, tau=0.1)
    assert (0, 1) not in e0

    # Entangle (0,1)
    U_hi = np.kron(_H(), np.eye(2, dtype=np.complex128))
    m.apply_edge_unitary(0, 1, U_hi)
    m.apply_edge_unitary(0, 1, _CNOT())

    MI1 = mi_matrix(m.rho(), dims=dims)
    e1 = emergent_edges(MI1, tau=0.1)
    assert (0, 1) in e1
