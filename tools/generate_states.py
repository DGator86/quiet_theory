from __future__ import annotations

import inspect
from pathlib import Path

try:
    import ame62_check
except Exception as e:
    raise SystemExit(f"ERROR: Could not import ame62_check.py from repo root. Details: {e}")

if not hasattr(ame62_check, "xi62_state"):
    raise SystemExit("ERROR: ame62_check.py does not define xi62_state().")

src = inspect.getsource(ame62_check.xi62_state).rstrip() + "\n"

out_path = Path("src/quiet_theory/quantum/states.py")
out_path.parent.mkdir(parents=True, exist_ok=True)

content = (
    "from __future__ import annotations\n\n"
    "import numpy as np\n\n\n"
    + src
    + "\n\n__all__ = [\"xi62_state\"]\n"
)

out_path.write_text(content, encoding="utf-8")
print(f"WROTE {out_path}")
