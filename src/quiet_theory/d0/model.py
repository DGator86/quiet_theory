from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .graph import D0Graph
from .info import mutual_information, partial_trace, von_neumann_entropy
from .ops import apply_two_site_unitary


@dataclass
class D0Model:
    graph: D0Graph
    dims: list[int]
    psi: np.ndarray

    def rho(self) -> np.ndarray:
        psi = self.psi.reshape((-1, 1))
        return psi @ psi.conj().T

    def entropy(self, A: Iterable[int]) -> float:
        rhoA = partial_trace(self.rho(), dims=self.dims, keep=tuple(A))
        return float(von_neumann_entropy(rhoA))

    def mi(self, A: Iterable[int], B: Iterable[int]) -> float:
        return float(mutual_information(self.rho(), A=tuple(A), B=tuple(B), dims=self.dims))

    def apply_edge_unitary(self, i: int, j: int, U: np.ndarray) -> None:
        self.psi = apply_two_site_unitary(self.psi, U, dims=self.dims, i=i, j=j)
