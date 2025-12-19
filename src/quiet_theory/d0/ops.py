from __future__ import annotations

from typing import Sequence

import numpy as np


def apply_two_site_unitary(
    psi: np.ndarray,
    U: np.ndarray | None = None,
    a: int | None = None,
    b: int | None = None,
    dims: Sequence[int] | None = None,
    *,
    # Compatibility aliases (your tests use these)
    i: int | None = None,
    j: int | None = None,
) -> np.ndarray:
    """
    Apply a 2-site unitary U to statevector psi on sites (a,b).

    Supports BOTH call styles:
      1) apply_two_site_unitary(psi, U, a, b, dims)
      2) apply_two_site_unitary(psi, dims=dims, i=i, j=j, U=U)

    psi is a flat statevector of length prod(dims).
    U must be shape (d_a*d_b, d_a*d_b) where d_a=dims[a], d_b=dims[b].
    """
    if dims is None:
        raise TypeError("dims must be provided (positional or keyword).")

    # Resolve indices from either naming scheme
    if a is None and i is not None:
        a = i
    if b is None and j is not None:
        b = j

    if U is None:
        raise TypeError("U must be provided (positional or keyword).")
    if a is None or b is None:
        raise TypeError("Must provide site indices (a,b) or (i,j).")

    dims = list(int(x) for x in dims)
    n = len(dims)

    a = int(a)
    b = int(b)

    if a == b:
        raise ValueError("Sites must be distinct.")
    if a < 0 or b < 0 or a >= n or b >= n:
        raise ValueError("Site index out of range.")

    psi = np.asarray(psi, dtype=np.complex128)
    D = int(np.prod(dims))
    if psi.shape != (D,):
        raise ValueError(f"psi must have shape ({D},), got {psi.shape}.")

    da = dims[a]
    db = dims[b]
    dloc = da * db
    U = np.asarray(U, dtype=np.complex128)
    if U.shape != (dloc, dloc):
        raise ValueError(f"U must have shape ({dloc},{dloc}), got {U.shape}.")

    # Reshape psi into tensor with one axis per site
    tensor = psi.reshape(dims)

    # Move target axes to the end, preserving order (a then b)
    # If a > b, moveaxis order needs to follow current axis indices.
    if a < b:
        axes = (a, b)
    else:
        axes = (b, a)

    moved = np.moveaxis(tensor, axes, (-2, -1))  # (..., d?, d?)

    # If we swapped axes order (because a>b), we must also swap local dims
    if a < b:
        moved_da, moved_db = da, db
        U_eff = U
    else:
        # moved axes are (b,a) now. Local basis order is b then a.
        # We need U to act on (a,b) ordering but tensor is (b,a).
        # Easiest: permute U to match (b,a) ordering.
        moved_da, moved_db = db, da

        # permutation between (a,b) and (b,a) basis: swap tensor product factors
        # Implement swap on the local Hilbert space by reshaping.
        U4 = U.reshape(da, db, da, db)  # (a,b)->(a,b)
        # map to (b,a)->(b,a)
        U_eff = np.transpose(U4, (1, 0, 3, 2)).reshape(dloc, dloc)

    # Batch-apply U on the last axis-pair
    rest = int(np.prod(moved.shape[:-2]))
    local = moved.reshape(rest, moved_da * moved_db)  # (rest, dloc)

    # Column-vector action: |psi'> = U |psi|  => local' = local @ U^T
    local2 = local @ U_eff.T

    moved2 = local2.reshape(*moved.shape[:-2], moved_da, moved_db)

    # Move axes back to original positions
    tensor2 = np.moveaxis(moved2, (-2, -1), axes)

    out = tensor2.reshape(D)
    return out
