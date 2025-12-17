import itertools
import numpy as np


def normalize_state(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(psi)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("State has invalid norm.")
    return psi / norm


def density_matrix(psi: np.ndarray) -> np.ndarray:
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T


def partial_trace_rho(rho: np.ndarray, keep: tuple[int, ...], n_qubits: int = 6) -> np.ndarray:
    """
    Partial trace over qubits not in 'keep' for an n_qubits density matrix.
    Returns reduced density matrix on the 'keep' subsystem.

    Qubit order convention: computational basis index is int(b0b1...b_{n-1}, 2)
    where b0 is the most-significant bit.
    """
    keep = tuple(sorted(keep))
    traced = tuple(i for i in range(n_qubits) if i not in keep)

    k = len(keep)
    t = n_qubits - k

    # rho as tensor with 2n indices: (i0..i_{n-1}, j0..j_{n-1})
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

    # Trace over traced subsystem: trace axes 1 and 3
    rho_keep = np.trace(rho_p, axis1=1, axis2=3)
    return rho_keep


def reshape_isometry_matrix(psi: np.ndarray, A: tuple[int, ...], n_qubits: int = 6) -> np.ndarray:
    """
    Reshape the amplitude tensor into a matrix M_A of shape (2^(n-k), 2^k)
    where columns correspond to subsystem A and rows correspond to its complement.
    """
    A = tuple(sorted(A))
    B = tuple(i for i in range(n_qubits) if i not in A)

    amp = psi.reshape([2] * n_qubits)
    perm = B + A  # B first, then A
    amp_perm = np.transpose(amp, axes=perm)

    k = len(A)
    return amp_perm.reshape((2 ** (n_qubits - k), 2**k))


def xi62_state() -> np.ndarray:
    """
    Construct an AME(6,2) 6-qubit perfect tensor state using the 5-qubit perfect code.

    We build the Choi state:
        |Xi> = (|0> |0_L> + |1> |1_L>) / sqrt(2),
    where |1_L> = X^{⊗5} |0_L>.

    Qubit order is b0..b5 (b0 MSB) with index = int(b0b1b2b3b4b5, 2).
    We take b0 as the "logical" qubit in the Choi construction.
    """
    # 5-qubit perfect code |0_L> expansion (16 terms, amplitudes ±1/4)
    # Each term is (bitstring length-5, sign)
    terms0 = [
        ("00000", +1),
        ("10010", +1),
        ("01001", +1),
        ("10100", +1),
        ("01010", +1),
        ("11011", -1),
        ("00110", -1),
        ("11000", -1),
        ("11101", -1),
        ("00011", -1),
        ("11110", -1),
        ("01111", -1),
        ("10001", -1),
        ("01100", -1),
        ("10111", -1),
        ("00101", +1),
    ]

    psi = np.zeros(64, dtype=np.complex128)
    amp = 1.0 / (4.0 * np.sqrt(2.0))  # (1/4) from codeword, (1/sqrt2) from Choi superposition

    for b, sgn in terms0:
        # |0> ⊗ |0_L>
        idx0 = int("0" + b, 2)
        psi[idx0] = sgn * amp

        # |1> ⊗ |1_L> where |1_L> = X^{⊗5} |0_L> = bitwise NOT of basis strings
        b_flip = "".join("1" if c == "0" else "0" for c in b)
        idx1 = int("1" + b_flip, 2)
        psi[idx1] = sgn * amp

    return psi


def check_ame62(psi_in: np.ndarray, tol: float = 1e-10, verbose: bool = True) -> bool:
    psi = normalize_state(psi_in)

    # Test 0: normalization
    norm_err = abs(np.vdot(psi, psi) - 1.0)
    if norm_err > tol:
        if verbose:
            print(f"FAIL norm: |<psi|psi>-1| = {norm_err:.3e}")
        return False

    rho = density_matrix(psi)
    all_ok = True

    # For AME(6,2), every k<=3 subsystem is maximally mixed.
    for k in (1, 2, 3):
        target = np.eye(2**k, dtype=np.complex128) / (2**k)

        for A in itertools.combinations(range(6), k):
            rho_A = partial_trace_rho(rho, keep=A, n_qubits=6)

            # Reduced state equals maximally mixed
            err_rho = np.linalg.norm(rho_A - target, ord="fro")
            if err_rho > tol:
                all_ok = False
                if verbose:
                    print(f"FAIL rho_A for A={A}, k={k}: ||rho-target||_F = {err_rho:.3e}")

            # Purity check: Tr(rho_A^2) = 2^{-k}
            purity = float(np.real_if_close(np.trace(rho_A @ rho_A)))
            purity_target = 2.0 ** (-k)
            err_purity = abs(purity - purity_target)
            if err_purity > 10 * tol:
                all_ok = False
                if verbose:
                    print(f"FAIL purity for A={A}, k={k}: {purity} vs {purity_target}")

            # Isometry reshape check: M_A^† M_A = I / 2^k
            M = reshape_isometry_matrix(psi, A=A, n_qubits=6)
            gram = M.conj().T @ M
            err_iso = np.linalg.norm(gram - target, ord="fro")
            if err_iso > tol:
                all_ok = False
                if verbose:
                    print(f"FAIL isometry for A={A}, k={k}: ||M^†M-target||_F = {err_iso:.3e}")

    if verbose:
        print("PASS AME(6,2)" if all_ok else "FAIL AME(6,2)")
    return all_ok


if __name__ == "__main__":
    # Test the actual AME(6,2) candidate
    psi = xi62_state()
    ok = check_ame62(psi, tol=1e-10, verbose=True)
    raise SystemExit(0 if ok else 1)
