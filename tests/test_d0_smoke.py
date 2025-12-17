from __future__ import annotations

import numpy as np

from quiet_theory.d0 import D0Graph, D0Model


def _H() -> np.ndarray:
    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


def _CNOT() -> np.ndarray:
    # |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10|
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.complex128,
    )


def test_d0_entanglement_increases_mi_and_entropy() -> None:
    # 3-qubit chain 0-1-2
    g = D0Graph.chain(3)
    dims = [2, 2, 2]

    # |000>
    psi = np.zeros(8, dtype=np.complex128)
    psi[0] = 1.0

    m = D0Model(graph=g, dims=dims, psi=psi)

    mi_before = m.mi((0,), (1,))
    s0_before = m.entropy((0,))

    # Create Bell pair between 0 and 1:
    # Apply H on qubit 0 (as a 2-site unitary on (0,1): H ⊗ I)
    U_hi = np.kron(_H(), np.eye(2, dtype=np.complex128))
    m.apply_edge_unitary(0, 1, U_hi)

    # Apply CNOT (0 control, 1 target)
    m.apply_edge_unitary(0, 1, _CNOT())

    mi_after = m.mi((0,), (1,))
    s0_after = m.entropy((0,))

    assert mi_after > mi_before + 1e-6
    assert s0_after > s0_before + 1e-6
