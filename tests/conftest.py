"""Pytest fixtures shared across all tests."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src-layout package is importable when running pytest from a clean shell.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
