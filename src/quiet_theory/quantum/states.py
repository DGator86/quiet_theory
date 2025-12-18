from __future__ import annotations

from importlib import import_module
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


def xi62_state() -> np.ndarray:
    """
    Returns the AME(6,2) state vector (length 64).
    For now, we reuse the known-good generator in the repo root: ame62_check.py.
    """
    try:
        m = import_module("ame62_check")
    except Exception:
        # Fallback: load the helper directly from the repository root so tests
        # and callers work even when the working directory is different.
        root = Path(__file__).resolve().parents[3]
        helper = root / "ame62_check.py"
        if not helper.exists():
            raise RuntimeError(
                "Could not import ame62_check.py. Run pytest from the repo root: C:\\dev\\quiet_theory"
            )
        m = SourceFileLoader("ame62_check", str(helper)).load_module()

    if not hasattr(m, "xi62_state"):
        raise RuntimeError("ame62_check.py does not define xi62_state().")

    psi = m.xi62_state()
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    if psi.shape != (64,):
        raise RuntimeError(f"xi62_state() returned shape {psi.shape}, expected (64,).")

    return psi


__all__ = ["xi62_state"]
