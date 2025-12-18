from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def apply_two_site_unitary(
    psi: np.ndarray,
    U: np.ndarray | None = None,
    a: int | None = None,
    b: int | None = None,
    dims: Sequence[int] | None = None,
    *,
    # Back-compat keywords (older call sites / tests)
    i: int | None = None,
    j: int | None = None,
) -> np.ndarray:
    """
    Apply a 2-site unitary U to the full statevector psi on sites (a,b).

    Accepts either:
      - positional/keyword: (psi, U, a, b, dims)
      - keyword legacy:     (psi, dims=..., i=..., j=..., U=...)

    Returns a new statevector (same shape as psi).
    """
    if U is None:
        raise TypeError("U must be provided")
    if dims is None:
        raise TypeError("dims must be provided")

    # Allow legacy i/j aliases
    if a is None and i is not None:
        a = i
    if b is None and j is not None:
        b = j

    if a is None or b is None:
        raise TypeError("Must provide site indices (a,b) or (i,j).")

    a = int(a)
    b = int(b)

    if a == b:
        raise ValueError("Sites must be distinct (a != b).")

    n = len(dims)
    if not (0 <= a < n and 0 <= b < n):
        raise ValueError("Site index out of range.")

    da = int(dims[a])
    db = int(dims[b])
    if U.shape != (da * db, da * db):
        raise ValueError(f"U must have shape {(da*db, da*db)} for dims[a]*dims[b]={da*db}.")

    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    if psi.size != int(np.prod(dims)):
        raise ValueError("psi length does not match product(dims).")

    # Reshape to tensor
    psi_t = psi.reshape(tuple(int(d) for d in dims))

    # Move axes a,b to the front
    axes = [a, b] + [k for k in range(n) if k not in (a, b)]
    inv = np.argsort(axes)

    front = np.transpose(psi_t, axes=axes).reshape(da * db, -1)

    # Apply U on the (a,b) block
    front2 = U @ front

    # Restore original ordering
    rest_dims = [int(dims[k]) for k in range(n) if k not in (a, b)]
    out_t = front2.reshape([da, db] + rest_dims)
    out = np.transpose(out_t, axes=inv).reshape(-1)

    return out
