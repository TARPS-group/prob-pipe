"""The skip of a test that constructs a StanModel, which reads its program with BridgeStan's stanc."""

from __future__ import annotations

import pytest


def require_stanc() -> None:
    """Skip the calling test unless BridgeStan's stanc compiler is installed."""
    pytest.importorskip("bridgestan")
    from probpipe.families._programs import _stanc

    try:
        _stanc(fetch=False)
    except ImportError as exc:
        pytest.skip(f"stanc is unavailable: {exc}")
