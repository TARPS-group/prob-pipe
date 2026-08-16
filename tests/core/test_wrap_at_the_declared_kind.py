"""The Function output boundary wraps a raw return into its own kind."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Function,
    Normal,
    NumericArray,
    Opaque,
    Record,
    function,
    log_prob,
    mean,
    sample,
)


class TestARawReturnWrapsIntoItsOwnKind:
    """No kind is presented by wrapping it in another."""

    def test_an_array_return_is_a_numeric_array(self):
        @function
        def total(x):
            return x * 2

        result = total(jnp.arange(3.0))

        assert isinstance(result, NumericArray)
        np.testing.assert_array_equal(np.asarray(result), np.arange(3.0) * 2)

    def test_a_scalar_return_is_a_numeric_array(self):
        @function
        def half(x):
            return x / 2

        assert float(half(5.0)) == 2.5

    def test_a_mapping_return_is_a_record(self):
        @function
        def both(x):
            return {"lo": x - 1, "hi": x + 1}

        result = both(jnp.asarray(1.0))

        assert isinstance(result, Record)
        assert result.fields == ("lo", "hi")

    def test_a_callable_return_is_a_function(self):
        """Callable because it is the function kind."""

        @function
        def make_adder(k):
            return lambda x: x + k

        adder = make_adder(jnp.asarray(2.0))

        assert isinstance(adder, Function)
        assert float(adder(jnp.asarray(1.0))) == 3.0

    def test_any_other_return_is_opaque(self):
        @function
        def label(x):
            return f"value-{int(x)}"

        result = label(jnp.asarray(2))

        assert isinstance(result, Opaque)
        assert result.value == "value-2"

    def test_the_result_is_named_after_the_function(self):
        @function
        def scaled(x):
            return x * 3

        assert scaled(jnp.asarray(1.0)).name == "scaled"


class TestTheKindsAreOrderedNotDisjoint:
    def test_a_callable_takes_the_function_kind_before_the_opaque_fallback(self):
        """A callable is also a non-mapping value, and the specific rule wins."""

        @function
        def make(x):
            del x
            return lambda: 1

        assert not isinstance(make(jnp.asarray(0.0)), Opaque)

    def test_every_tracked_term_keeps_its_kind(self):
        """Whatever the kind, a term keeps it."""
        inner = Function(func=lambda x: x + 1, name="inner")
        outer = Function(func=lambda: inner, name="outer")

        assert isinstance(outer(), Function)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: NumericArray(jnp.arange(3.0), name="held"),
            lambda: Opaque("held", object()),
            lambda: Record("held", {"x": jnp.asarray(1.0)}, name_is_auto=True),
        ],
    )
    def test_the_rule_is_the_same_for_every_kind(self, make):
        kind = type(make())
        returning = Function(func=make, name="returning")

        assert isinstance(returning(), kind)


class TestTheOperationsReturnTheirDeclaredKind:
    def test_log_prob_is_a_numeric_array(self):
        law = Normal(loc=0.0, scale=1.0, name="x")

        assert isinstance(log_prob(law, 0.0), NumericArray)

    def test_mean_of_a_scalar_law_is_a_numeric_array(self):
        assert isinstance(mean(Normal(loc=2.0, scale=1.0, name="x")), NumericArray)

    def test_a_scalar_draw_is_a_numeric_array(self):
        drawn = sample(Normal(loc=0.0, scale=1.0, name="x"), key=jax.random.PRNGKey(0))

        assert isinstance(drawn, NumericArray)

    def test_a_numeric_array_result_still_computes(self):
        """A result computes directly, which is what the array surface is for."""
        law = Normal(loc=0.0, scale=1.0, name="x")

        assert float(log_prob(law, 0.0) * 2) == pytest.approx(
            float(np.asarray(log_prob(law, 0.0))) * 2
        )


class TestASampleShapeGetsADrawLevel:
    """Design V.2: the leading dimensions go on a level named `draw`."""

    def test_no_sample_shape_is_one_value(self):
        drawn = sample(Normal(loc=0.0, scale=1.0, name="x"), key=jax.random.PRNGKey(0))

        assert isinstance(drawn, NumericArray)

    @pytest.mark.parametrize("sample_shape", [(5,), (2, 3)])
    def test_draws_land_on_one_draw_level(self, sample_shape):
        from probpipe import NumericArrayBatch

        drawn = sample(
            Normal(loc=0.0, scale=1.0, name="x"),
            sample_shape=sample_shape,
            key=jax.random.PRNGKey(0),
        )

        assert isinstance(drawn, NumericArrayBatch)
        assert drawn.batch_shape == sample_shape
        assert drawn.level_names == ("sample",)

    def test_the_event_shape_is_kept_out_of_the_draw_level(self):
        """A vector law draws vectors, so its event axes stay with the element."""
        from probpipe import MultivariateNormal, NumericArrayBatch

        law = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="v")

        drawn = sample(law, sample_shape=(5,), key=jax.random.PRNGKey(0))

        assert isinstance(drawn, NumericArrayBatch)
        assert drawn.batch_shape == (5,)
        assert tuple(drawn.element_spec.shape) == (3,)
        assert drawn.shape == (5, 3)

    def test_an_element_is_one_draw(self):
        drawn = sample(
            Normal(loc=0.0, scale=1.0, name="x"), sample_shape=(5,), key=jax.random.PRNGKey(0)
        )

        assert isinstance(drawn[2], NumericArray)
        assert drawn[2].shape == ()

    def test_a_record_law_still_draws_its_own_batch(self):
        """A record law builds its own batch."""
        from probpipe import NumericRecordBatch, ProductDistribution

        law = ProductDistribution(
            Normal(loc=0.0, scale=1.0, name="a"), Normal(loc=1.0, scale=1.0, name="b")
        )

        drawn = sample(law, sample_shape=(5,), key=jax.random.PRNGKey(0))

        assert isinstance(drawn, NumericRecordBatch)

    def test_a_law_that_does_not_prepend_its_draws_is_left_alone(self):
        """Rather than mis-splitting axes the law never laid out that way."""
        from probpipe.core.ops import _drawn_at_its_batch_form

        # Leading axes that are not the requested sample_shape: nothing here can
        # say which axes are draws, so the value is handed back untouched.
        drawn = _drawn_at_its_batch_form(jnp.zeros((2, 7)), (5,), name="law", name_is_auto=False)

        assert not isinstance(drawn, NumericArray)
        assert drawn.shape == (2, 7)
