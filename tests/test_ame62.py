from __future__ import annotations

import numpy as np

from quiet_theory.quantum.ame import check_ame62
from quiet_theory.quantum.states import xi62_state


def test_xi62_is_ame62() -> None:
    psi = xi62_state()
    assert psi.shape == (64,)
    assert np.isfinite(psi).all()
    assert check_ame62(psi, tol=1e-10, verbose=False)


def test_basis_state_is_not_ame62() -> None:
    psi = np.zeros(64, dtype=np.complex128)
    psi[0] = 1.0
    assert not check_ame62(psi, tol=1e-10, verbose=False)
