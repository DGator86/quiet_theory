import numpy as np

from quiet_theory.d0.ops import apply_two_site_unitary


def test_apply_two_site_unitary_accepts_legacy_keywords() -> None:
    dims = [2, 2]
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = 1.0
    U = np.eye(4, dtype=np.complex128)

    out = apply_two_site_unitary(psi, dims=dims, i=0, j=1, U=U)
    assert out.shape == psi.shape
