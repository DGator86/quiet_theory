from __future__ import annotations

import sys

from quiet_theory.quantum.ame import check_ame62


def main() -> int:
    # We reuse your working AME(6,2) state definition from the repo root script.
    try:
        import ame62_check  # repo-root module
    except Exception as e:
        print("ERROR: Could not import ame62_check.py from repo root.")
        print("Run this from C:\\dev\\quiet_theory, not from another folder.")
        print(f"Details: {e}")
        return 2

    if not hasattr(ame62_check, "xi62_state"):
        print("ERROR: ame62_check.py does not define xi62_state().")
        print("Your AME state generator must be a function named xi62_state() that returns a 64-length vector.")
        return 2

    psi = ame62_check.xi62_state()
    ok = check_ame62(psi, tol=1e-10, verbose=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
