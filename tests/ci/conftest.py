"""Make the CI helper scripts importable by tests in this directory.

``scripts/ci/`` is not an installed package, so put it on ``sys.path`` here
rather than relying on packaging — keeps ``import import_graph`` working under
``pytest -n auto``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_CI = Path(__file__).resolve().parents[2] / "scripts" / "ci"
if str(_SCRIPTS_CI) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_CI))
