"""Labels of functions and laws stay independent of their numerical declarations.

The value-kind constructor tests live in the files that mirror their modules.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Distribution,
    Function,
    Normal,
    NumericArray,
    NumericArraySpec,
    OutputSpec,
    function,
    workflow_run,
)
from probpipe.core._fingerprint import fingerprint
from probpipe.core.tracked import _NO_DESCRIPTION


class TestFunctionLabels:
    def test_function_defaults_and_explicit_aliases(self):
        def predict(temperature):
            return temperature + 1

        assert Function(predict).notation == "predict(temperature)"
        assert Function(lambda temperature: temperature + 1).notation == "f(temperature)"
        assert function(lambda temperature: temperature + 1).notation == "f(temperature)"
        assert Function(predict)(NumericArray(2.0, label="ambient")).label == "predict(ambient)"
        assert Function(predict, output_label="prediction")(2.0).label == "prediction"

    def test_output_components_are_declared_independently_of_aliases(self):
        inferred = Function(lambda x: x, output_spec=NumericArraySpec(()))
        assert tuple(inferred.output_spec.components) == ("f",)
        declared = OutputSpec(prediction=NumericArraySpec(()))
        f = Function(
            lambda x: x + 1, label="predict", output_label="forecast", output_spec=declared
        )
        g = Function(f.raw(), label="other", output_label="estimate", output_spec=declared)
        assert f.output_spec == g.output_spec
        assert fingerprint(f) == fingerprint(g)
        with workflow_run(seed=42):
            first = f(Normal("temperature", 0, 1))
        with workflow_run(seed=42):
            second = g(Normal("temperature", 0, 1))
        assert tuple(first.event_spec.components) == ("prediction",)
        assert tuple(second.event_spec.components) == ("prediction",)
        np.testing.assert_array_equal(first.atoms.raw(), second.atoms.raw())

    @pytest.mark.parametrize("transform", [jax.jit, jax.vmap], ids=["jit", "vmap"])
    def test_a_function_applies_to_a_rebuilt_array_in_a_transform(self, transform):
        rebuilt = transform(lambda x: x)(NumericArray(jnp.arange(3.0), label="temperature"))
        assert rebuilt.label == _NO_DESCRIPTION
        increment = Function(lambda x: x + 1)
        np.testing.assert_array_equal(
            np.asarray(transform(increment)(rebuilt)), jnp.arange(3.0) + 1
        )

    def test_a_managed_result_derives_its_label_from_a_compiled_callable(self):
        @jax.jit
        def compiled(x):
            return x + 1

        first = NumericArray(1.0, label="temperature")
        wrapped = Function(compiled, label="increment")
        assert wrapped(first).label == "increment(temperature)"
        assert float(wrapped(first)) == 2.0

    def test_lifting_infers_components_independently_of_display_aliases(self):
        def predict(x):
            return x + 1

        inferred = Function(predict, n_broadcast_samples=5)
        aliased = Function(
            predict, label="forecast", output_label="estimate", n_broadcast_samples=5
        )
        assert fingerprint(inferred) == fingerprint(aliased)
        with workflow_run(seed=42):
            first = inferred(Normal("temperature", 0, 1))
        with workflow_run(seed=42):
            second = aliased.with_label("other")(Normal("temperature", 0, 1))
        assert tuple(first.event_spec.components) == ("predict",)
        assert tuple(second.event_spec.components) == ("predict",)
        np.testing.assert_array_equal(first.atoms.raw(), second.atoms.raw())
        unnamed = Function(lambda x: x + 1, label="predict", n_broadcast_samples=5)
        assert tuple(unnamed(Normal("temperature", 0, 1)).event_spec.components) == ("f",)
        exposed = Function(lambda x: {"prediction": x + 1}, n_broadcast_samples=5)
        assert tuple(exposed(Normal("temperature", 0, 1)).event_spec.components) == ("prediction",)


class TestLawLabels:
    def test_generic_law_default_uses_a_declared_component(self):
        law = Distribution(OutputSpec(tau=NumericArraySpec(())))
        assert law.notation == "p(tau)"
        assert law.with_label("prior").event_spec == law.event_spec
