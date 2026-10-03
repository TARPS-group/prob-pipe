"""Tests for the program-defined families' exports from ``probpipe``."""

import pytest

import probpipe


class TestLazyExports:
    """``probpipe`` exports the program-defined families lazily."""

    def test_stanmodel_lazy_load(self):
        from probpipe import StanModel
        from probpipe.families import StanModel as family

        assert StanModel is family

    def test_pymcmodel_lazy_load(self):
        from probpipe import PyMCModel
        from probpipe.families import PyMCModel as family

        assert PyMCModel is family

    def test_unknown_attr_raises(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = probpipe.NonExistent
