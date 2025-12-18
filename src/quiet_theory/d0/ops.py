from __future__ import annotations

import numpy as np


def apply_two_site_unitary(
    psi: np.ndarray,
    U: np.ndarray,
    i: int,
    j: int,
    dims: list[int] | tuple[int, ...],
) -> np.ndarray:
    """
    Apply a two-site unitary U on sites (i, j) to a pure state psi.

    psi: shape (D,)
    dims: local dims [d0..d_{n-1}]
    U: shape (d_i*d_j, d_i*d_j)

    Returns: new psi (shape D,)
    """
    dims_l = list(dims)
    n = len(dims_l)
    if i == j:
        raise ValueError("Sites i and j must be different.")
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError("Site index out of range.")

    di, dj = dims_l[i], dims_l[j]
    Dij = di * dj
    if U.shape != (Dij, Dij):
        raise ValueError(f"U must be shape ({Dij},{Dij}); got {U.shape}")

    D = int(np.prod(dims_l))
    psi = np.asarray(psi, dtype=np.complex128).reshape(D)

    # reshape state into tensor
    amp = psi.reshape(dims_l)

    # bring i,j to the end
    rest = [k for k in range(n) if k not in (i, j)]
    perm = rest + [i, j]
    inv_perm = np.argsort(perm)

    amp_p = np.transpose(amp, axes=perm)
    Dr = int(np.prod([dims_l[k] for k in rest])) if rest else 1
    amp_p = amp_p.reshape((Dr, Dij))

    # apply U on last factor
    amp_p2 = amp_p @ U.T  # column action in computational basis convention

    # reshape back
    amp_p2 = amp_p2.reshape([dims_l[k] for k in rest] + [di, dj])
    amp2 = np.transpose(amp_p2, axes=inv_perm)

    return amp2.reshape(D)
