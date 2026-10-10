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

        assert (
            Opaque(
                payload,
                label="p",
            ).value
            is payload
        )

    def test_the_spec_defaults_to_the_values_type(self):
        assert Opaque(
            _Payload(),
            label="p",
        ).spec == OpaqueSpec(type=_Payload)
        assert Opaque(
            "north",
            label="s",
        ).spec == OpaqueSpec(type=str)

    def test_a_declared_type_the_value_lacks_is_refused(self):
        with pytest.raises(
            TypeError, match=r"value of type _Payload does not match OpaqueSpec\(type=str\)"
        ):
            Opaque(
                _Payload(),
                spec=OpaqueSpec(type=str),
                label="p",
            )

    def test_a_declared_spec_carries_its_meta(self):
        spec = OpaqueSpec(meta="fitted-model")

        assert (
            Opaque(
                _Payload(),
                spec=spec,
                label="p",
            ).spec.meta
            == "fitted-model"
        )

    def test_a_spec_of_another_kind_is_refused(self):
        with pytest.raises(TypeError, match="must be an OpaqueSpec"):
            Opaque(
                _Payload(),
                spec="not a spec",
                label="p",
            )

    def test_a_mapping_is_refused(self):
        with pytest.raises(TypeError, match="Opaque cannot hold a mapping, got dict"):
            Opaque(
                {"a": 1},
                label="p",
            )

    @pytest.mark.parametrize("value", [1, "text", None, [1, 2], (1, 2), _Payload()])
    def test_anything_else_is_admitted(self, value):
        assert (
            Opaque(
                value,
                label="p",
            ).value
            == value
        )


class TestOpaqueAddsIdentityAndNothingElse:
    """Its interface is `value` and the identity a tracked term carries."""

    def test_it_does_not_forward_attributes(self):
        wrapped = Opaque(
            _Payload(),
            label="p",
        )

        assert not hasattr(wrapped, "shout")
        with pytest.raises(AttributeError):
            wrapped.shout()

    def test_it_is_not_callable_even_when_its_value_is(self):
        wrapped = Opaque(
            _Payload(),
            label="p",
        )

        assert not callable(wrapped)
        with pytest.raises(TypeError):
            wrapped()

    def test_the_value_affords_what_it_always_did_once_out(self):
        assert (
            Opaque(
                _Payload("hi"),
                label="p",
            ).value.shout()
            == "HI"
        )

    def test_it_carries_no_array_surface(self):
        wrapped = Opaque(
            _Payload(),
            label="p",
        )

        for absent in ("shape", "dtype", "ndim", "__array__", "as_jax"):
            assert not hasattr(wrapped, absent)


class TestOpaqueCarriesIdentity:
    def test_a_name_is_kept(self):
        wrapped = Opaque(
            _Payload(),
            label="model",
        )

        assert wrapped.label == "model"

    def test_a_name_is_required(self):
        """The label is what says which opaque value this is."""
        with pytest.raises(TypeError):
            Opaque(label=_Payload())

    def test_a_derived_name_is_kept(self):
        wrapped = Opaque(
            _Payload(),
            label="batch[draw=0]",
        )

        assert wrapped.label == "batch[draw=0]"

    def test_provenance_is_write_once(self):
        wrapped = Opaque(
            _Payload(),
            label="p",
        ).with_provenance(Provenance.create("fit", parents=[]))

        assert wrapped.provenance.operation == "fit"
        with pytest.raises(RuntimeError, match="already set"):
            wrapped.with_provenance(Provenance.create("again", parents=[]))

    def test_it_is_immutable(self):
        with pytest.raises(AttributeError, match="immutable"):
            Opaque(
                _Payload(),
                label="p",
            )._value = _Payload("other")

    @pytest.mark.parametrize(
        "roundtrip",
        [copy.copy, copy.deepcopy, lambda o: pickle.loads(pickle.dumps(o))],
    )
    def test_it_survives_copy_and_pickle(self, roundtrip):
        wrapped = Opaque(
            _Payload("kept"),
            label="model",
        )

        rebuilt = roundtrip(wrapped)

        assert isinstance(rebuilt, Opaque)
        assert rebuilt.label == "model"
        assert rebuilt.value == _Payload("kept")


class TestOpaqueAndItsBatch:
    """A collection is tracked whatever it holds."""

    def test_a_batch_of_opaque_values_hands_back_a_view_of_what_was_put_in(self):
        """Its elements are stored, so an element is an Opaque holding the caller's object."""
        payloads = [_Payload("a"), _Payload("b")]

        batch = OpaqueBatch(
            payloads,
            "draw",
            label="batch",
        )

        assert batch[0].value is payloads[0]
        assert batch[0].label == "batch[draw=0]"

    def test_a_batch_may_hold_opaque_terms_as_its_elements(self):
        """An `Opaque` is itself a non-mapping value."""
        terms = [
            Opaque(
                _Payload("a"),
                label="first",
            ),
            Opaque(
                _Payload("b"),
                label="second",
            ),
        ]

        batch = OpaqueBatch(
            terms,
            "draw",
            label="batch",
        )

        assert batch[1].value is terms[1].value
        assert batch[1].label == "batch[draw=1]"
        assert terms[1].label == "second"
