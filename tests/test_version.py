from __future__ import annotations

import re
import tomllib
from pathlib import Path

import probpipe


def test_version_resolves_from_distribution():
    """``probpipe.__version__`` is read from installed metadata, not the fallback.

    Regression guard for the distribution-name lookup in
    ``probpipe/__init__.py``: the rename ``probpipe`` -> ``probpipe-core``
    silently broke ``importlib.metadata.version(...)`` until CI caught it. If the
    looked-up name drifts from the distribution that ships the package again, the
    version falls back to the ``"0.0.0+unknown"`` placeholder and this fails.
    """
    assert probpipe.__version__ != "0.0.0+unknown"


def test_the_metapackage_moves_in_lockstep_with_the_core():
    """The ``probpipe`` metapackage carries ``probpipe-core``'s version and pins it exactly.

    CONTRIBUTING.md § Package Structure states the lockstep: both
    ``pyproject.toml`` files carry one version, and each ``probpipe-core``
    requirement of the metapackage pins that version with ``==``.
    """
    root = Path(__file__).resolve().parents[1]
    core = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    meta = tomllib.loads((root / "packaging" / "probpipe" / "pyproject.toml").read_text())[
        "project"
    ]
    assert meta["version"] == core["version"]
    pins = [req for req in meta["dependencies"] if req.startswith("probpipe-core")]
    assert pins
    exact = re.compile(rf"==\s*{re.escape(core['version'])}\s*(;|$)")
    assert [req for req in pins if not exact.search(req)] == []
