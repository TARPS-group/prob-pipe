"""The Function and its engine (design V.1).

A Function wraps one callable and presents it as a graph node. Every call runs
the stack of eight steps in order, a failure ending the call at its step, and
on concrete values the engine agrees with plain evaluation. ``check`` runs the
first six steps without numerical execution or random events.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Function,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    function,
    workflow_run,
)
from probpipe.functions import _function
from probpipe.values import _function_base

from ._design_helpers import atom_leaves, error_of, standard_normal


class TestTheFunction:
    def test_the_decorator_wraps_one_callable_as_a_function(self):
        def predict(theta, x):
            return x * theta

        wrapped = function(predict)

        assert isinstance(wrapped, Function)
        assert wrapped.label == "predict"
        assert wrapped.raw() is predict

    def test_the_decorator_takes_construction_time_controls(self):
        @function(n_broadcast_samples=500, dispatch="jax")
        def predict(theta, x):
            return x * theta

        assert predict.options["n_broadcast_samples"] == 500
        assert predict.options["dispatch"] == "jax"

    def test_a_call_agrees_with_plain_evaluation_adding_the_wrap_and_the_provenance(self):
        @function
        def predict(theta, x):
            return x * theta

        x = jnp.arange(3.0)
        result = predict(2.0, x)

        assert isinstance(result, NumericArray)
        np.testing.assert_allclose(result.value, predict.apply(2.0, x))
        assert result.provenance is not None
        assert result.provenance.parents[0].label == "predict"


class TestTheEngine:
    def test_the_engine_is_installed_once(self):
        with pytest.raises(RuntimeError, match="already installed"):
            _function_base.install_call_engine(lambda function, *args, **kwargs: None)

    def test_installing_the_same_engine_again_changes_nothing(self):
        _function_base.install_call_engine(_function._call_engine)

        assert _function_base._call_engine is _function._call_engine

    def test_a_non_callable_engine_is_refused(self):
        with pytest.raises(TypeError, match="callable"):
            _function_base.install_call_engine(object())

    def test_a_binding_failure_ends_the_call_before_the_body_runs(self):
        calls = []

        @function
        def record(x):
            calls.append(x)
            return x

        with pytest.raises(TypeError):
            record(1.0, 2.0)
        assert calls == []

    @pytest.mark.parametrize(
        "controls", [{"exact_only": True}, {"method": "elementwise_sweep"}], ids=lambda c: str(c)
    )
    def test_a_lift_failure_ends_the_call_before_the_route_is_selected(self, controls):
        @function(dispatch="sequential")
        def add(x, y):
            return x + y

        def misaligned():
            scalar = NumericArraySpec(())
            return (
                NumericArrayBatch("x", jnp.arange(2.0), "row", element_spec=scalar),
                NumericArrayBatch("y", jnp.arange(3.0), "row", element_spec=scalar),
            )

        unrestricted = error_of(lambda: add(*misaligned()))
        restricted = error_of(lambda: add.with_options(**controls)(*misaligned()))

        assert unrestricted is not None
        assert type(restricted) is type(unrestricted)
        assert str(restricted) == str(unrestricted)


class TestCheck:
    """``check`` probes steps 1 to 6 and never executes the route it selects."""

    def test_check_never_invokes_the_body(self):
        calls = []

        @function
        def record(x):
            calls.append(x)
            return x

        record.check(1.0)

        assert calls == []

    def test_check_causes_no_random_event(self):
        @function(n_broadcast_samples=8, dispatch="sequential")
        def identity(x):
            return x

        law = standard_normal()
        with workflow_run(seed=3):
            baseline = identity(law)
        with workflow_run(seed=3):
            identity.check(law)
            probed = identity(law)

        for probed_leaf, baseline_leaf in zip(
            atom_leaves(probed), atom_leaves(baseline), strict=True
        ):
            np.testing.assert_array_equal(probed_leaf, baseline_leaf)

    def test_a_view_probes_under_the_controls_of_its_call(self):
        @function
        def identity(x):
            return x

        identity.with_options(n_broadcast_samples=16).check(standard_normal())

    def test_an_argument_that_does_not_bind_raises_as_the_call_would(self):
        @function
        def identity(x):
            return x

        with pytest.raises(TypeError):
            identity.check(1.0, 2.0)
