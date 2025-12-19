from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .graph import D0Graph
from .info import mutual_information, von_neumann_entropy, partial_trace
from .ops import apply_two_site_unitary


@dataclass
class D0Model:
    graph: D0Graph
    dims: Sequence[int]
    psi: np.ndarray

    def rho(self) -> np.ndarray:
        psi = np.asarray(self.psi, dtype=np.complex128)
        return np.outer(psi, np.conjugate(psi))

    def entropy(self, A: Sequence[int]) -> float:
        rhoA = partial_trace(self.rho(), dims=self.dims, keep=tuple(A))
        return float(von_neumann_entropy(rhoA))

    def mi(self, A: Sequence[int], B: Sequence[int]) -> float:
        return float(mutual_information(self.rho(), A=tuple(A), B=tuple(B), dims=self.dims))

    def apply_edge_unitary(self, i: int, j: int, U: np.ndarray) -> None:
        # Standardize on keyword style
        self.psi = apply_two_site_unitary(self.psi, dims=self.dims, i=i, j=j, U=U)
