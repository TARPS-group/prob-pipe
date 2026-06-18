from __future__ import annotations

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
