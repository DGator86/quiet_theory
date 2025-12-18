from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quiet_theory.d0 import (
    D0Graph,
    D0Model,
    EvolutionConfig,
    evolve,
    mi_matrix,
    objective_locality,
)


def test_pipeline_rewire_then_unitary_updates(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    dims = [2, 2, 2, 2]
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi = psi.astype(np.complex128)
    psi = psi / np.linalg.norm(psi)

    model = D0Model(graph=D0Graph.chain(len(dims)), dims=dims, psi=psi)
    cfg = EvolutionConfig(
        steps=3,
        k_edges=3,
        keep_fraction=0.5,
        state_updates_per_step=1,
        unitary_trials=4,
        save_dir=tmp_path,
        run_name="pipeline",
        seed=7,
    )

    initial_obj = objective_locality(model.rho(), model.graph, model.dims)
    evolved_model = evolve(model, cfg, rng=rng)

    metrics_path = tmp_path / "pipeline_metrics.jsonl"
    assert metrics_path.exists()

    with metrics_path.open() as f:
        metrics = [json.loads(line) for line in f if line.strip()]

    assert len(metrics) == cfg.steps
    for entry in metrics:
        assert np.isfinite(entry["objective_before"]) and np.isfinite(entry["objective_after"])
        assert abs(entry["delta"]) < 1e3

    assert np.isfinite(objective_locality(evolved_model.rho(), evolved_model.graph, evolved_model.dims))
    assert not np.any(np.isnan(mi_matrix(evolved_model.rho(), evolved_model.dims)))
    assert np.isfinite(initial_obj)
