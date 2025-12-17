from __future__ import annotations

import numpy as np

from quiet_theory.d0 import D0Graph, D0Model, mi_matrix, rewire_topk, save_model, load_model


def test_rewire_topk_builds_correct_edge_count() -> None:
    rng = np.random.default_rng(0)
    dims = [2, 2, 2, 2]
    g = D0Graph.chain(4)

    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi = psi.astype(np.complex128)
    psi = psi / np.linalg.norm(psi)

    m = D0Model(graph=g, dims=dims, psi=psi)
    M = mi_matrix(m.rho(), dims)

    g2 = rewire_topk(g, M, k_edges=3, keep_fraction=0.5)
    assert len(g2.edges) == 3


def test_save_load_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(1)
    dims = [2, 2, 2, 2]
    g = D0Graph.chain(4)

    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi = psi.astype(np.complex128)
    psi = psi / np.linalg.norm(psi)

    m = D0Model(graph=g, dims=dims, psi=psi)

    p = tmp_path / "model.json"
    save_model(m, p)
    m2 = load_model(p)

    assert list(m2.dims) == list(m.dims)
    assert set(m2.graph.edges) == set(m.graph.edges)
    assert np.allclose(m2.psi, m.psi)
