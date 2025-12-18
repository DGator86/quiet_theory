from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def apply_two_site_unitary(
    psi: np.ndarray,
    U: np.ndarray | None = None,
    i: int | None = None,
    j: int | None = None,
    dims: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Apply a 2-site unitary ``U`` to the full statevector ``psi`` on sites ``(i, j)``.

    Returns a new statevector (same shape as psi).
    """
    if U is None:
        raise TypeError("U must be provided")
    if dims is None:
        raise TypeError("dims must be provided")

    if i is None or j is None:
        raise TypeError("Must provide site indices i and j.")

    i_idx = int(i)
    j_idx = int(j)

    if i_idx == j_idx:
        raise ValueError("Sites must be distinct (a != b).")

    n = len(dims)
    if not (0 <= i_idx < n and 0 <= j_idx < n):
        raise ValueError("Site index out of range.")

    da = int(dims[i_idx])
    db = int(dims[j_idx])
    if U.shape != (da * db, da * db):
        raise ValueError(
            f"U must have shape {(da*db, da*db)} for dims[i]*dims[j]={da*db}."
        )

    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    if psi.size != int(np.prod(dims)):
        raise ValueError("psi length does not match product(dims).")

    # Reshape to tensor
    psi_t = psi.reshape(tuple(int(d) for d in dims))

    # Move axes a,b to the front
    axes = [i_idx, j_idx] + [k for k in range(n) if k not in (i_idx, j_idx)]
    inv = np.argsort(axes)

    front = np.transpose(psi_t, axes=axes).reshape(da * db, -1)

    # Apply U on the (a,b) block
    front2 = U @ front

    # Restore original ordering
    rest_dims = [int(dims[k]) for k in range(n) if k not in (i_idx, j_idx)]
    out_t = front2.reshape([da, db] + rest_dims)
    out = np.transpose(out_t, axes=inv).reshape(-1)

    return out
