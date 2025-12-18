from __future__ import annotations

import numpy as np

from . import D0Graph, D0Model, EvolutionConfig, evolve, mi_matrix


def main() -> None:
    rng = np.random.default_rng(0)
    dims = [2, 2, 2, 2]
    g = D0Graph.chain(len(dims))

    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi = psi.astype(np.complex128)
    psi = psi / np.linalg.norm(psi)

    model = D0Model(graph=g, dims=dims, psi=psi)

    cfg = EvolutionConfig(
        steps=3,
        k_edges=4,
        keep_fraction=0.5,
        state_updates_per_step=2,
        unitary_trials=8,
        run_name="evolve_demo",
        seed=1,
        save_dir="artifacts",
    )

    evolve(model, cfg, rng=rng)

    print("Final edges:", sorted(model.graph.edges))
    M = mi_matrix(model.rho(), model.dims)
    print("MI matrix:\n", np.round(M, 4))
    if cfg.save_dir is not None:
        metrics_path = f"{cfg.save_dir}/{cfg.run_name}_metrics.jsonl"
        print("Metrics written to", metrics_path)


if __name__ == "__main__":
    main()

