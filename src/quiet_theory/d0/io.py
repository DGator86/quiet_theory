from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .graph import D0Graph
from .model import D0Model


def save_model(model: D0Model, path: str | Path) -> None:
    path = Path(path)
    payload = {
        "dims": list(model.dims),
        "edges": [list(e) for e in sorted(model.graph.edges)],
        "psi_real": model.psi.real.tolist(),
        "psi_imag": model.psi.imag.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model(path: str | Path) -> D0Model:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    dims = list(payload["dims"])
    n = len(dims)

    edges: set[tuple[int, int]] = set()
    for i, j in payload["edges"]:
        a = int(i)
        b = int(j)
        edges.add((a, b) if a < b else (b, a))

    re = np.array(payload["psi_real"], dtype=float)
    im = np.array(payload["psi_imag"], dtype=float)
    psi = (re + 1j * im).astype(np.complex128)

    return D0Model(graph=D0Graph(n, edges), dims=dims, psi=psi)
