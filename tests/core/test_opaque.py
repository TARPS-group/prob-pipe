"""Tests for Opaque — the tracked class of the opaque kind."""

from __future__ import annotations

import copy
import pickle

import pytest

from probpipe import Opaque, OpaqueBatch, OpaqueSpec
from probpipe.core.provenance import Provenance


class _Payload:
    """A stand-in for the sort of thing an opaque value holds."""

    def __init__(self, tag: str = "x") -> None:
        self.tag = tag

    def shout(self) -> str:
        return self.tag.upper()

    def __call__(self) -> str:
        return "called"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Payload) and other.tag == self.tag


class TestOpaqueHoldsOneValue:
    def test_the_value_is_reachable_and_unchanged(self):
        payload = _Payload()

        assert Opaque(payload).value is payload

    def test_the_spec_defaults_to_a_bare_one(self):
        assert Opaque(_Payload()).spec == OpaqueSpec()

    def test_a_declared_spec_carries_its_meta(self):
        spec = OpaqueSpec(meta="fitted-model")

        assert Opaque(_Payload(), spec=spec).spec.meta == "fitted-model"

    def test_a_spec_of_another_kind_is_refused(self):
        with pytest.raises(TypeError, match="must be an OpaqueSpec"):
            Opaque(_Payload(), spec="not a spec")

    def test_a_mapping_is_refused(self):
        """The value layer reads a mapping as a subtree, not a leaf."""
        with pytest.raises(TypeError, match="reads a mapping as a subtree"):
            Opaque({"a": 1})

    @pytest.mark.parametrize("value", [1, "text", None, [1, 2], (1, 2), _Payload()])
    def test_anything_else_is_admitted(self, value):
        assert Opaque(value).value == value


class TestOpaqueAddsIdentityAndNothingElse:
    """A box that forwards is indistinguishable from what it wraps."""

    def test_it_does_not_forward_attributes(self):
        wrapped = Opaque(_Payload())

        assert not hasattr(wrapped, "shout")
        with pytest.raises(AttributeError):
            wrapped.shout()

    def test_it_is_not_callable_even_when_its_value_is(self):
        wrapped = Opaque(_Payload())

        assert not callable(wrapped)
        with pytest.raises(TypeError):
            wrapped()

    def test_the_value_affords_what_it_always_did_once_out(self):
        assert Opaque(_Payload("hi")).value.shout() == "HI"

    def test_it_carries_no_array_surface(self):
        wrapped = Opaque(_Payload())

        for absent in ("shape", "dtype", "ndim", "__array__", "as_jax"):
            assert not hasattr(wrapped, absent)


class TestOpaqueCarriesIdentity:
    def test_a_name_is_kept_and_marked_user_given(self):
        wrapped = Opaque(_Payload(), name="model")

        assert (wrapped.name, wrapped.name_is_auto) == ("model", False)

    def test_an_omitted_name_is_auto_derived(self):
        wrapped = Opaque(_Payload())

        assert (wrapped.name, wrapped.name_is_auto) == ("opaque", True)

    def test_provenance_is_write_once(self):
        wrapped = Opaque(_Payload()).with_provenance(Provenance.create("fit", parents=[]))

        assert wrapped.provenance.operation == "fit"
        with pytest.raises(RuntimeError, match="already set"):
            wrapped.with_provenance(Provenance.create("again", parents=[]))

    def test_it_is_immutable(self):
        with pytest.raises(AttributeError, match="immutable"):
            Opaque(_Payload())._value = _Payload("other")

    @pytest.mark.parametrize(
        "roundtrip",
        [copy.copy, copy.deepcopy, lambda o: pickle.loads(pickle.dumps(o))],
    )
    def test_it_survives_copy_and_pickle(self, roundtrip):
        wrapped = Opaque(_Payload("kept"), name="model")

        rebuilt = roundtrip(wrapped)

        assert isinstance(rebuilt, Opaque)
        assert rebuilt.name == "model"
        assert rebuilt.value == _Payload("kept")


class TestOpaqueAndItsBatch:
    """The batch existed first: a collection is tracked whatever it holds."""

    def test_a_batch_of_opaque_values_still_hands_back_what_was_put_in(self):
        """`OpaqueBatch`'s element contract is unchanged by the new class.

        Its elements are the caller's own objects, stored rather than
        materialized, so it does not start wrapping them in `Opaque`.
        """
        payloads = [_Payload("a"), _Payload("b")]

        batch = OpaqueBatch(payloads, "draw")

        assert batch[0] is payloads[0]

    def test_a_batch_may_hold_opaque_terms_as_its_elements(self):
        """Nothing stops one, since an `Opaque` is itself a non-mapping value."""
        terms = [Opaque(_Payload("a"), name="first"), Opaque(_Payload("b"), name="second")]

        batch = OpaqueBatch(terms, "draw")

        assert batch[1] is terms[1]
        assert batch[1].name == "second"
