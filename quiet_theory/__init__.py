from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))
    package_path = SRC / "quiet_theory"
    if package_path.is_dir():
        __path__.append(str(package_path))
