"""Numeric, the flat-vector interface of the numeric kinds (design II.3)."""

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Numeric,
    NumericArray,
    NumericArrayBatch,
    NumericRecord,
    NumericRecordBatch,
)


def _values():
    return (
        NumericArray(
            jnp.arange(6.0).reshape(2, 3),
            label="x",
        ),
        NumericRecord(
            {"a": jnp.arange(2.0), "b": {"c": jnp.ones((2, 2))}},
            label="r",
        ),
    )


class TestTheNumericKinds:
    def test_an_array_and_a_record_are_numeric(self):
        assert all(isinstance(value, Numeric) for value in _values())

    def test_the_batch_forms_are_not(self):
        assert not issubclass(NumericArrayBatch, Numeric)
        assert not issubclass(NumericRecordBatch, Numeric)

    def test_a_batch_names_its_layout_spec_as_a_value_does(self):
        spec = NumericRecord(
            {"a": jnp.arange(2.0)},
            label="r",
        ).spec
        batch = NumericRecordBatch.from_vector(
            "b", spec=spec, vec=jnp.zeros((3, 2)), level_names="draw"
        )
        assert batch.batch_shape == (3,)

    @pytest.mark.parametrize("value", _values(), ids=["array", "record"])
    def test_from_vector_inverts_to_vector(self, value):
        vector = value.to_vector()
        assert vector.shape == (value.vector_size,)
        rebuilt = type(value).from_vector("rebuilt", value.spec, vector)
        assert rebuilt.label == "rebuilt"
        np.testing.assert_array_equal(rebuilt.to_vector(), vector)

    def test_an_array_presents_its_shape(self):
        # Its coordinates are its elements in row-major order, presented unraveled.
        value = NumericArray(
            jnp.arange(6.0).reshape(2, 3),
            label="x",
        )
        np.testing.assert_array_equal(np.asarray(value), np.arange(6.0).reshape(2, 3))
        assert jnp.asarray(value).shape == (2, 3)
        np.testing.assert_array_equal(value.to_vector(), jnp.ravel(jnp.asarray(value)))


class _Pair(Numeric):
    """A minimal numeric kind: two coordinates."""

    __slots__ = ("_vector",)

    def __init__(self, vector):
        self._vector = jnp.asarray(vector)

    @property
    def vector_size(self):
        return 2

    def to_vector(self):
        return self._vector

    @classmethod
    def from_vector(cls, name, spec, vec):
        return cls(vec)


class TestTheBase:
    def test_the_coordinates_present_the_vector(self):
        pair = _Pair([1.0, 2.0])
        np.testing.assert_array_equal(np.asarray(pair), [1.0, 2.0])
        assert np.asarray(pair, dtype=np.float64).dtype == np.float64
        np.testing.assert_array_equal(jnp.asarray(pair), jnp.array([1.0, 2.0]))

    def test_a_kind_missing_a_member_cannot_be_constructed(self):
        class _NoVector(Numeric):
            @property
            def vector_size(self):
                return 0

            @classmethod
            def from_vector(cls, name, spec, vec):
                return cls()

        with pytest.raises(TypeError, match="abstract"):
            _NoVector()
