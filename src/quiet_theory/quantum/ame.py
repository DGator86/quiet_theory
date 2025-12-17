from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


def normalize_state(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(psi)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("State has invalid norm.")
    return psi / norm


def density_matrix(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1, 1)
    return psi @ psi.conj().T


def partial_trace_rho(rho: np.ndarray, keep: tuple[int, ...], n_qubits: int) -> np.ndarray:
    """
    Partial trace over qubits not in 'keep' for an n_qubits density matrix.
    Returns reduced density matrix on the 'keep' subsystem.

    Assumes each site is a qubit (dimension 2).
    """
    keep = tuple(sorted(keep))
    traced = tuple(i for i in range(n_qubits) if i not in keep)

    k = len(keep)
    t = n_qubits - k

    rho_t = rho.reshape([2] * n_qubits + [2] * n_qubits)

    # Permute to (i_keep, i_traced, j_keep, j_traced)
    perm = (
        list(keep)
        + list(traced)
        + [q + n_qubits for q in keep]
        + [q + n_qubits for q in traced]
    )
    rho_p = np.transpose(rho_t, axes=perm)

    # Reshape to (2^k, 2^t, 2^k, 2^t)
    rho_p = rho_p.reshape((2**k, 2**t, 2**k, 2**t))

    # Trace out traced subsystem: trace axes 1 and 3
    rho_keep = np.trace(rho_p, axis1=1, axis2=3)
    return rho_keep


def reshape_isometry_matrix(psi: np.ndarray, A: tuple[int, ...], n_qubits: int) -> np.ndarray:
    """
    Build M_A of shape (2^(n-k), 2^k) by reshaping the amplitude tensor.
    This is a standard “perfect tensor / isometry” reshape check.
    """
    A = tuple(sorted(A))
    B = tuple(i for i in range(n_qubits) if i not in A)

    amp = np.asarray(psi, dtype=np.complex128).reshape([2] * n_qubits)
    perm = B + A  # B first, then A
    amp_perm = np.transpose(amp, axes=perm)

    k = len(A)
    return amp_perm.reshape((2 ** (n_qubits - k), 2**k))


@dataclass(frozen=True)
class AmeCheckResult:
    ok: bool
    norm_err: float
    worst_rho_err: float
    worst_purity_err: float
    worst_iso_err: float


def check_ame_qubits(
    psi_in: np.ndarray,
    *,
    n_qubits: int,
    max_k: int,
    tol: float = 1e-10,
    verbose: bool = True,
) -> AmeCheckResult:
    """
    Checks the AME condition for an n_qubits qubit pure state:
      For all subsets A with |A| = k, for k = 1..max_k,
      rho_A should be maximally mixed: I / 2^k.

    Returns worst-case errors across all tested subsystems.
    """
    psi = normalize_state(psi_in)

    # Test 0: normalization
    norm_err = float(abs(np.vdot(psi, psi) - 1.0))
    if norm_err > tol:
        if verbose:
            print(f"FAIL norm: |<psi|psi>-1| = {norm_err:.3e}")
        return AmeCheckResult(False, norm_err, np.inf, np.inf, np.inf)

    rho = density_matrix(psi)

    worst_rho_err = 0.0
    worst_purity_err = 0.0
    worst_iso_err = 0.0
    all_ok = True

    for k in range(1, max_k + 1):
        target = np.eye(2**k, dtype=np.complex128) / (2**k)
        purity_target = 2.0 ** (-k)

        for A in itertools.combinations(range(n_qubits), k):
            rho_A = partial_trace_rho(rho, keep=A, n_qubits=n_qubits)

            # Reduced state equals maximally mixed
            err_rho = float(np.linalg.norm(rho_A - target, ord="fro"))
            worst_rho_err = max(worst_rho_err, err_rho)
            if err_rho > tol:
                all_ok = False
                if verbose:
                    print(f"FAIL rho_A for A={A}, k={k}: ||rho-target||_F = {err_rho:.3e}")

            # Purity check: Tr(rho_A^2) = 2^{-k}
            purity = float(np.real_if_close(np.trace(rho_A @ rho_A)))
            err_purity = float(abs(purity - purity_target))
            worst_purity_err = max(worst_purity_err, err_purity)
            if err_purity > 10 * tol:
                all_ok = False
                if verbose:
                    print(f"FAIL purity for A={A}, k={k}: {purity} vs {purity_target}")

            # Isometry reshape check: M_A^† M_A = I / 2^k
            M = reshape_isometry_matrix(psi, A=A, n_qubits=n_qubits)
            gram = M.conj().T @ M
            err_iso = float(np.linalg.norm(gram - target, ord="fro"))
            worst_iso_err = max(worst_iso_err, err_iso)
            if err_iso > tol:
                all_ok = False
                if verbose:
                    print(f"FAIL isometry for A={A}, k={k}: ||M^†M-target||_F = {err_iso:.3e}")

    if verbose:
        print("PASS AME check" if all_ok else "FAIL AME check")

    return AmeCheckResult(all_ok, norm_err, worst_rho_err, worst_purity_err, worst_iso_err)


def check_ame62(psi_in: np.ndarray, tol: float = 1e-10, verbose: bool = True) -> bool:
    """
    Convenience wrapper: AME(6,2) means n_qubits=6 and all subsystems up to k=3 are maximally mixed.
    """
    res = check_ame_qubits(psi_in, n_qubits=6, max_k=3, tol=tol, verbose=verbose)
    if verbose:
        print("PASS AME(6,2)" if res.ok else "FAIL AME(6,2)")
    return res.ok
