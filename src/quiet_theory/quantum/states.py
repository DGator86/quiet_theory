from __future__ import annotations

from importlib import import_module

import numpy as np


def xi62_state() -> np.ndarray:
    """
    Returns the AME(6,2) state vector (length 64).
    For now, we reuse the known-good generator in the repo root: ame62_check.py.
    """
    try:
        m = import_module("ame62_check")
    except Exception as e:
        raise RuntimeError(
            "Could not import ame62_check.py. Run pytest from the repo root: C:\\dev\\quiet_theory"
        ) from e

    if not hasattr(m, "xi62_state"):
        raise RuntimeError("ame62_check.py does not define xi62_state().")

    psi = m.xi62_state()
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    if psi.shape != (64,):
        raise RuntimeError(f"xi62_state() returned shape {psi.shape}, expected (64,).")

    return psi


__all__ = ["xi62_state"]
