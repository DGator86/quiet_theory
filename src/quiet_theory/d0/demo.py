from __future__ import annotations

import numpy as np

from . import D0Graph, D0Model


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


def main() -> None:
    # Simple Bell-pair demo on a 3-qubit chain.
    g = D0Graph.chain(3)
    dims = [2, 2, 2]
    psi = np.zeros(8, dtype=np.complex128)
    psi[0] = 1.0

    model = D0Model(graph=g, dims=dims, psi=psi)
    print("Initial MI(0,1):", model.mi((0,), (1,)))

    U_hi = np.kron(_H(), np.eye(2, dtype=np.complex128))
    model.apply_edge_unitary(0, 1, U_hi)
    model.apply_edge_unitary(0, 1, _CNOT())

    print("After entangling MI(0,1):", model.mi((0,), (1,)))
    print("Entropy of site 0:", model.entropy((0,)))


if __name__ == "__main__":
    main()

