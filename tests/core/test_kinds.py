"""The kind table: one tracked class and one batch form per value spec.

The correspondence was open-coded at six sites, each knowing a different
subset, so widening one left the others behind. These tests pin the table
itself and the property that made the duplication a bug: every registered kind
answers both questions the same way wherever it is asked.
"""

from __future__ import annotations

import pytest

from probpipe import (
    FunctionBatch,
    FunctionSpec,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    Opaque,
    OpaqueBatch,
    OpaqueSpec,
)
from probpipe.core._kinds import batch_class_for_spec, register_kind, term_class_for_spec


class TestEveryKindDeclaresItsClasses:
    @pytest.mark.parametrize(
        ("spec", "term", "batch"),
        [
            pytest.param(NumericArraySpec(shape=()), NumericArray, NumericArrayBatch, id="numeric"),
            pytest.param(OpaqueSpec(), Opaque, OpaqueBatch, id="opaque"),
            pytest.param(FunctionSpec(), None, FunctionBatch, id="function"),
        ],
    )
    def test_a_spec_resolves_to_its_pair(self, spec, term, batch):
        assert term_class_for_spec(spec) is term
        assert batch_class_for_spec(spec) is batch

    def test_a_kind_with_no_tracked_class_says_so(self):
        """`FunctionSpec` has a batch form and no separate tracked class — a
        callable is already a `Function`."""
        assert term_class_for_spec(FunctionSpec()) is None


class TestTheTableIsSingleValued:
    def test_a_contradicting_registration_is_refused(self):
        """One spec, one kind. A second registration is a contradiction, not an
        override, so it raises rather than silently winning."""
        with pytest.raises(ValueError, match="already registered"):
            register_kind(OpaqueSpec, term_class=int, batch_class=int)

    def test_re_registering_the_same_pair_is_harmless(self):
        """Importing a module twice must not raise."""
        register_kind(OpaqueSpec, term_class=Opaque, batch_class=OpaqueBatch)

        assert batch_class_for_spec(OpaqueSpec()) is OpaqueBatch

    def test_a_spec_subclass_inherits_its_bases_kind(self):
        """Resolution walks the MRO, so a refinement need not re-register."""

        class _NarrowerOpaque(OpaqueSpec):
            pass

        assert batch_class_for_spec(_NarrowerOpaque()) is OpaqueBatch

    def test_an_unregistered_spec_resolves_to_nothing(self):
        class _Unregistered:
            pass

        assert batch_class_for_spec(_Unregistered()) is None
