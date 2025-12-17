from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import D0Graph
from .info import mutual_information, partial_trace, von_neumann_entropy
from .ops import apply_two_site_unitary


@dataclass
class D0Model:
    """
    Minimal D0 model container:
    - Graph substrate
    - Local dimensions
    - Pure state vector
    """
    graph: D0Graph
    dims: list[int]
    psi: np.ndarray

    def rho(self) -> np.ndarray:
        psi = self.psi.reshape(-1, 1)
        return psi @ psi.conj().T

    def entropy(self, A: tuple[int, ...]) -> float:
        rho_A = partial_trace(self.rho(), keep=A, dims=self.dims)
        return von_neumann_entropy(rho_A)

    def mi(self, A: tuple[int, ...], B: tuple[int, ...]) -> float:
        return mutual_information(self.rho(), A=A, B=B, dims=self.dims)

    def apply_edge_unitary(self, i: int, j: int, U: np.ndarray) -> None:
        self.psi = apply_two_site_unitary(self.psi, dims=self.dims, i=i, j=j, U=U)
        # renormalize (numerical drift guard)
        nrm = np.linalg.norm(self.psi)
        if nrm == 0 or not np.isfinite(nrm):
            raise ValueError("State became invalid after applying unitary.")
        self.psi = self.psi / nrm
