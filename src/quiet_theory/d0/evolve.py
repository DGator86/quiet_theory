from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import D0Model
from .rewire import mi_matrix, rewire_topk
from .update import greedy_edge_update
from .io import save_model


@dataclass
class EvolutionConfig:
    """Configuration for a simple D0 evolution loop."""

    steps: int = 8
    k_edges: int = 4
    keep_fraction: float = 0.5
    state_updates_per_step: int = 2
    unitary_trials: int = 16
    save_dir: str | Path | None = None
    run_name: str = "d0_run"
    seed: int | None = None

    def __post_init__(self) -> None:
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

    for step in range(config.steps):
        edges: list[tuple[int, int]] = list(model.graph.edges)
        if edges and config.state_updates_per_step > 0:
            for u in range(config.state_updates_per_step):
                i, j = edges[u % len(edges)]
                greedy_edge_update(
                    model,
                    i,
                    j,
                    trials=config.unitary_trials,
                    rng=rng,
                )

        M = mi_matrix(model.rho(), model.dims)
        model.graph = rewire_topk(
            model.graph,
            M,
            k_edges=config.k_edges,
            keep_fraction=config.keep_fraction,
        )

        if config.save_dir is not None:
            config.save_dir.mkdir(parents=True, exist_ok=True)
            save_model(model, config.save_dir / f"{config.run_name}_step{step:03d}.json")

    return model


__all__ = ["EvolutionConfig", "evolve"]
