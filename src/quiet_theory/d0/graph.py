from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple


@dataclass(frozen=True)
class D0Graph:
    """
    Minimal graph substrate for D0.

    - Nodes are labeled 0..n-1.
    - Edges are undirected by default.
    """
    n: int
    edges: Set[Tuple[int, int]]

    @staticmethod
    def chain(n: int) -> "D0Graph":
        edges: Set[Tuple[int, int]] = set()
        for i in range(n - 1):
            edges.add((i, i + 1))
        return D0Graph(n=n, edges=edges)

    def neighbors(self, v: int) -> List[int]:
        nbrs: List[int] = []
        for a, b in self.edges:
            if a == v:
                nbrs.append(b)
            elif b == v:
                nbrs.append(a)
        return nbrs
