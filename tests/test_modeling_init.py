"""Tests for probpipe.modeling lazy imports."""

import pytest


class TestLazyImports:
    def test_stanmodel_importable(self):
        from probpipe.modeling import StanModel
        assert StanModel is not None

    def test_pymcmodel_importable(self):
        from probpipe.modeling import PyMCModel
        assert PyMCModel is not None

    def test_unknown_attr_raises(self):
        import probpipe.modeling as mod
        with pytest.raises(AttributeError, match="has no attribute"):
            mod.NonExistent
