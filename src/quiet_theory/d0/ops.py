from __future__ import annotations

import numpy as np


def apply_two_site_unitary(
    psi: np.ndarray,
    U: np.ndarray | None = None,
    a: int | None = None,
    b: int | None = None,
    dims: list[int] | tuple[int, ...] | None = None,
    *,
    i: int | None = None,
    j: int | None = None,
) -> np.ndarray:
    """
    Apply a two-site unitary ``U`` to a pure state ``psi``.

    The API is intentionally tolerant to both positional and keyword-based
    invocations. Indices can be provided as ``a``/``b`` or ``i``/``j`` and the
    unitary can be supplied positionally or via the ``U`` keyword.

    Parameters
    ----------
    psi:
        State vector of shape ``(D,)`` where ``D = prod(dims)``.
    U:
        Two-site unitary of shape ``(d_a * d_b, d_a * d_b)``.
    a, b / i, j:
        Site indices. ``a``/``b`` take precedence; ``i``/``j`` are aliases.
    dims:
        Iterable of local dimensions ``[d0, d1, ..., d_{n-1}]``.
    """

    if dims is None:
        raise ValueError("dims must be provided to apply_two_site_unitary")

    dims_l = list(dims)
    n = len(dims_l)

    if a is None:
        a = i
    if b is None:
        b = j

    if a is None or b is None:
        raise ValueError("Site indices must be provided via a/b or i/j")

    if U is None:
        raise ValueError("Unitary U must be provided")

    if a == b:
        raise ValueError("Sites a and b must be different.")
    if not (0 <= a < n and 0 <= b < n):
        raise ValueError(
            f"Site indices out of range: a={a}, b={b}, len(dims)={n}"
        )

    di, dj = dims_l[a], dims_l[b]
    Dij = di * dj
    if U.shape != (Dij, Dij):
        raise ValueError(
            f"U must be shape ({Dij},{Dij}); got {U.shape} for a={a}, b={b}, len(dims)={n}"
        )

    D = int(np.prod(dims_l))
    psi = np.asarray(psi, dtype=np.complex128).reshape(D)

    # reshape state into tensor
    amp = psi.reshape(dims_l)

    # bring a,b to the end
    rest = [k for k in range(n) if k not in (a, b)]
    perm = rest + [a, b]
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
