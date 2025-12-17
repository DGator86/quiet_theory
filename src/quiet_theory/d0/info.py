from __future__ import annotations

import numpy as np


def _as_dims(dims: list[int] | tuple[int, ...]) -> list[int]:
    dims_l = list(dims)
    if any(d <= 1 for d in dims_l):
        raise ValueError("All local dimensions must be >= 2.")
    return dims_l


def partial_trace(rho: np.ndarray, keep: tuple[int, ...], dims: list[int] | tuple[int, ...]) -> np.ndarray:
    """
    Partial trace of density matrix rho over subsystems not in `keep`.

    dims: local dimensions [d0, d1, ..., d_{n-1}]
    rho: shape (D, D) with D = prod(dims)

    Returns: rho_keep with shape (D_keep, D_keep)
    """
    dims_l = _as_dims(dims)
    n = len(dims_l)
    keep = tuple(sorted(keep))
    traced = tuple(i for i in range(n) if i not in keep)

    D = int(np.prod(dims_l))
    if rho.shape != (D, D):
        raise ValueError(f"rho must be shape ({D},{D}); got {rho.shape}")

    # reshape into tensor with indices (i0..i_{n-1}, j0..j_{n-1})
    rho_t = rho.reshape(dims_l + dims_l)

    # permute into (i_keep, i_traced, j_keep, j_traced)
    perm = (
        list(keep)
        + list(traced)
        + [q + n for q in keep]
        + [q + n for q in traced]
    )
    rho_p = np.transpose(rho_t, axes=perm)

    Dk = int(np.prod([dims_l[i] for i in keep])) if keep else 1
    Dt = int(np.prod([dims_l[i] for i in traced])) if traced else 1

    rho_p = rho_p.reshape((Dk, Dt, Dk, Dt))
    rho_keep = np.trace(rho_p, axis1=1, axis2=3)
    return rho_keep


def von_neumann_entropy(rho: np.ndarray, tol: float = 1e-12) -> float:
    """
    S(rho) = -Tr(rho log2 rho)
    """
    # Hermitize for numerical stability
    rho_h = 0.5 * (rho + rho.conj().T)
    evals = np.linalg.eigvalsh(rho_h)
    evals = np.real_if_close(evals).astype(float)
    evals = np.clip(evals, 0.0, 1.0)
    evals = evals[evals > tol]
    if evals.size == 0:
        return 0.0
    return float(-np.sum(evals * np.log2(evals)))


def mutual_information(
    rho: np.ndarray,
    A: tuple[int, ...],
    B: tuple[int, ...],
    dims: list[int] | tuple[int, ...],
) -> float:
    """
    I(A:B) = S(A) + S(B) - S(AB)
    """
    A = tuple(sorted(A))
    B = tuple(sorted(B))
    AB = tuple(sorted(set(A) | set(B)))

    rho_A = partial_trace(rho, keep=A, dims=dims)
    rho_B = partial_trace(rho, keep=B, dims=dims)
    rho_AB = partial_trace(rho, keep=AB, dims=dims)

    return von_neumann_entropy(rho_A) + von_neumann_entropy(rho_B) - von_neumann_entropy(rho_AB)
