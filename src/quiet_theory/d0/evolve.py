from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .model import D0Model
from .objective import objective_locality
from .rewire import mi_matrix, rewire_topk
from .update import apply_random_two_site_unitary_step
from .io import save_model


@dataclass
class EvolutionConfig:
    steps: int
    k_edges: int
    keep_fraction: float = 0.5

    # REQUIRED BY TESTS:
    state_updates_per_step: int = 0
    unitary_trials: int = 8

    save_dir: str | Path | None = "artifacts"
    run_name: str = "d0_evolve_state"
    seed: int | None = None

    # Optional backward-compat alias
    psi_updates_per_step: int = 0

    def __post_init__(self) -> None:
        if self.state_updates_per_step == 0 and self.psi_updates_per_step != 0:
            self.state_updates_per_step = int(self.psi_updates_per_step)
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.k_edges < 0:
            raise ValueError("k_edges must be non-negative")
        if self.state_updates_per_step < 0:
            raise ValueError("state_updates_per_step must be non-negative")
        if not (0 <= self.keep_fraction <= 1):
            raise ValueError("keep_fraction must be in [0,1]")
        if self.save_dir is not None:
            self.save_dir = Path(self.save_dir)


def evolve(model: D0Model, config: EvolutionConfig, *, rng: np.random.Generator | None = None) -> D0Model:
    """
    Evolve a D0Model by alternating greedy state updates and MI-based rewiring.

    The number of greedy updates per step is controlled by ``state_updates_per_step``.
    After each round of updates, the graph is rewired to the top ``k_edges`` pairs
    by mutual information while preserving a ``keep_fraction`` of existing edges.
    Optionally saves intermediate models to ``save_dir``.
    """

    if rng is None:
        rng = np.random.default_rng(config.seed)

    metrics_path = None
    if config.save_dir is not None:
        config.save_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = Path(config.save_dir) / f"{config.run_name}_metrics.jsonl"
        metrics_path.write_text("")

    for step in range(config.steps):
        objective_before = objective_locality(model.rho(), model.graph, model.dims)

        if config.state_updates_per_step > 0:
            for _ in range(config.state_updates_per_step):
                model.psi = apply_random_two_site_unitary_step(
                    psi=model.psi,
                    graph=model.graph,
                    dims=model.dims,
                    rng=rng,
                )

        M = mi_matrix(model.rho(), model.dims)
        model.graph = rewire_topk(
            model.graph,
            M,
            k_edges=config.k_edges,
            keep_fraction=config.keep_fraction,
        )

        objective_after = objective_locality(model.rho(), model.graph, model.dims)
        delta = float(objective_after - objective_before)
        if not np.isfinite(delta):
            raise ValueError("Objective change became non-finite during evolution")

        if metrics_path is not None:
            with metrics_path.open("a", encoding="utf-8") as f:
                json.dump(
                    {
                        "step": step,
                        "objective_before": float(objective_before),
                        "objective_after": float(objective_after),
                        "delta": delta,
                    },
                    f,
                )
                f.write("\n")

        if config.save_dir is not None:
            config.save_dir.mkdir(parents=True, exist_ok=True)
            save_model(model, config.save_dir / f"{config.run_name}_step{step:03d}.json")

    return model


__all__ = ["EvolutionConfig", "evolve"]
