from __future__ import annotations

import numpy as np

from quiet_theory.d0 import D0Graph, D0Model


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


def test_three_qubit_bell_pair_increases_mi_and_entropy() -> None:
    g = D0Graph.chain(3)
    dims = [2, 2, 2]

    psi = np.zeros(8, dtype=np.complex128)
    psi[0] = 1.0

    model = D0Model(graph=g, dims=dims, psi=psi)

    mi_before = model.mi((0,), (1,))
    s0_before = model.entropy((0,))

    # Create Bell pair between qubits 0 and 1
    U_hi = np.kron(_H(), np.eye(2, dtype=np.complex128))
    model.apply_edge_unitary(0, 1, U_hi)
    model.apply_edge_unitary(0, 1, _CNOT())

    mi_after = model.mi((0,), (1,))
    s0_after = model.entropy((0,))

    # Bell pair should yield ~2 bits of MI and ~1 bit of entropy on qubit 0
    assert mi_after - mi_before > 0.5
    assert mi_after > 1.5
    assert s0_after - s0_before > 0.25
    assert s0_after > 0.8
