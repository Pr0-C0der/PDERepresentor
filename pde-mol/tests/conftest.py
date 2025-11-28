"""
Pytest configuration for the pde-mol project.

This ensures that the project root (containing the `pde` package) is on
`sys.path`, so tests can simply use `from pde import ...` without needing
to set PYTHONPATH in the shell.
"""

import os
import sys


def _ensure_project_root_on_path() -> None:
    # tests/ -> project root is one directory up from this file's directory
    root = os.path.dirname(os.path.dirname(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root_on_path()


