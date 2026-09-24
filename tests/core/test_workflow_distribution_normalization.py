"""Characterization tests for Function distribution normalization."""

from __future__ import annotations

import inspect
import types
from typing import Any, Protocol, runtime_checkable
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    Distribution,
    DistributionArray,
    EmpiricalDistribution,
    KDEDistribution,
    Normal,
    NumericArrayBatch,
    NumericRecordDistribution,
    OpaqueSpec,
    converter_registry,
    log_prob,
    mean,
    workflow_run,
)
from probpipe.core._workflow_call import make_signature_info_from_signature
from probpipe.core._workflow_distribution_normalization import (
    DISTRIBUTION_HINT_PROTOCOLS,
    normalize_distribution_values,
)
from probpipe.core.node import Function
from probpipe.core.protocols import SupportsLogProb


@pytest.fixture
def normal_external():
    return tfd.Normal(loc=2.0, scale=0.5)


@pytest.fixture
def empirical_dist():
    return EmpiricalDistribution(
        "x",
        jnp.asarray([[0.0], [1.0], [2.0]]),
    )


@pytest.fixture
def mean_recorder():
    seen = []

    def mean_of_normal(dist: Normal):
        seen.append(dist)
        return mean(dist)

    return mean_of_normal, seen


class TestNormalizeDistributionValues:
    """Direct helper coverage mirrors facade cases for clearer failure locality."""

    def test_concrete_distribution_hint_converts_external_distribution(
        self,
        normal_external,
    ):
        normalized = normalize_distribution_values(
            values={"dist": normal_external},
            signature_info=_signature_info(("dist",), {"dist": Normal}),
        )

        assert isinstance(normalized["dist"], Normal)
        assert normalized["dist"] is not normal_external

    def test_concrete_distribution_hint_without_converter_raises(self):
        class UnsupportedDistribution(Distribution):
            pass

        source = Normal(loc=0.0, scale=1.0, name="source")

        with pytest.raises(
            TypeError,
            match=r"No converter registered for Normal -> UnsupportedDistribution",
        ):
            normalize_distribution_values(
                values={"dist": source},
                signature_info=_signature_info(("dist",), {"dist": UnsupportedDistribution}),
            )

    def test_protocol_hint_converts_distribution_that_lacks_protocol(
        self,
        empirical_dist,
    ):
        normalized = normalize_distribution_values(
            values={"dist": empirical_dist},
            signature_info=_signature_info(("dist",), {"dist": SupportsLogProb}),
        )

        assert normalized["dist"] is not empirical_dist
        assert isinstance(normalized["dist"], KDEDistribution)
        assert isinstance(normalized["dist"], SupportsLogProb)

    def test_protocol_hint_preserves_value_when_conversion_raises_type_error(self, empirical_dist):
        with patch.object(converter_registry, "convert", side_effect=TypeError("unsupported")):
            normalized = normalize_distribution_values(
                values={"dist": empirical_dist},
                signature_info=_signature_info(("dist",), {"dist": SupportsLogProb}),
            )

        assert normalized["dist"] is empirical_dist

    def test_protocol_hint_propagates_invalid_conversion_plan(self, empirical_dist):
        error = RuntimeError("a sampled conversion requires a sample shape")
        with (
            patch.object(converter_registry, "convert", side_effect=error),
            pytest.raises(RuntimeError, match="requires a sample shape") as exc_info,
        ):
            normalize_distribution_values(
                values={"dist": empirical_dist},
                signature_info=_signature_info(("dist",), {"dist": SupportsLogProb}),
            )

        assert exc_info.value is error

    def test_zero_dimensional_distribution_array_unwraps_to_scalar_component(self):
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(),
            loc=jnp.asarray(3.0),
            scale=jnp.asarray(1.0),
            name="zero_d",
        )

        normalized = normalize_distribution_values(
            values={"dist": da},
            signature_info=_signature_info(("dist",), {"dist": Normal}),
        )

        assert isinstance(normalized["dist"], Normal)
        assert float(normalized["dist"].loc) == 3.0

    def test_size_one_distribution_array_remains_a_sweep(self):
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(1,),
            loc=jnp.asarray([3.0]),
            scale=jnp.asarray([1.0]),
            name="one_cell",
        )

        normalized = normalize_distribution_values(
            values={"dist": da},
            signature_info=_signature_info(("dist",), {"dist": Normal}),
        )

        assert normalized["dist"] is da

    def test_unhinted_external_distribution_converts_for_broadcast(
        self,
        normal_external,
    ):
        normalized = normalize_distribution_values(
            values={"x": normal_external},
            signature_info=_signature_info(("x",)),
        )

        assert isinstance(normalized["x"], NumericRecordDistribution)
        assert normalized["x"] is not normal_external


# Facade checks below are retained to verify Function wiring.
class TestHintedDistributionConversion:
    def test_concrete_distribution_hint_converts_external_distribution(
        self,
        mean_recorder,
        normal_external,
    ):
        mean_of_normal, seen = mean_recorder
        wf = Function(func=mean_of_normal, dispatch="sequential")

        result = wf(dist=normal_external)

        assert isinstance(seen[0], Normal)
        assert float(result) == 2.0

    def test_protocol_hint_converts_distribution_that_lacks_protocol(self, empirical_dist):
        seen = []

        def log_prob_at_zero(dist: SupportsLogProb):
            seen.append(dist)
            return log_prob(dist, jnp.asarray([0.0]))

        wf = Function(func=log_prob_at_zero, dispatch="sequential")

        result = wf(dist=empirical_dist)

        assert seen[0] is not empirical_dist
        assert isinstance(seen[0], KDEDistribution)
        assert isinstance(seen[0], SupportsLogProb)
        assert jnp.isfinite(jnp.asarray(result)).all()


class TestDistributionArrayHandling:
    def test_zero_dimensional_distribution_array_unwraps_to_scalar_component(
        self,
        mean_recorder,
    ):
        mean_of_normal, seen = mean_recorder
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(),
            loc=jnp.asarray(3.0),
            scale=jnp.asarray(1.0),
            name="zero_d",
        )
        wf = Function(func=mean_of_normal, dispatch="sequential")

        result = wf(dist=da)

        assert isinstance(seen[0], Normal)
        assert float(result) == 3.0

    def test_size_one_distribution_array_remains_a_sweep(self, mean_recorder):
        mean_of_normal, _ = mean_recorder
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(1,),
            loc=jnp.asarray([3.0]),
            scale=jnp.asarray([1.0]),
            name="one_cell",
        )
        wf = Function(func=mean_of_normal, dispatch="sequential")

        result = wf(dist=da)

        assert isinstance(result, NumericArrayBatch)
        assert result.batch_shape == (1,)
        np.testing.assert_allclose(result.values, jnp.asarray([3.0]))


class TestUnhintedExternalDistribution:
    def test_unhinted_external_distribution_converts_then_broadcasts(self):
        seen_values = []

        def double(x):
            value = jnp.asarray(x)
            seen_values.append(value)
            return value * 2.0

        wf = Function(
            func=double,
            n_broadcast_samples=8,
            dispatch="sequential",
        )
        external = tfd.Normal(loc=1.0, scale=0.1)

        with workflow_run(seed=42):
            result = wf(external)

        assert result.num_atoms == 8
        assert [value.shape for value in seen_values] == [()] * 8
        # ``atol=0.0`` is deliberate: multiplication by 2.0 is bit-exact
        # under IEEE 754, so the broadcast output should match the
        # recorded per-call inputs exactly.
        np.testing.assert_allclose(result.samples, 2.0 * jnp.stack(seen_values), atol=0.0)


@runtime_checkable
class _SupportsApply(Protocol):
    def apply(self, *args: Any, **kwargs: Any) -> Any: ...


def _signature_info(names, hints=None):
    signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in names]
    )
    return make_signature_info_from_signature(signature, hints=hints)


def test_non_distribution_capability_protocol_does_not_disable_lifting():
    seen = []

    def consume(x: _SupportsApply):
        seen.append(x)
        return x

    wrapped = Function(
        func=consume,
        n_broadcast_samples=8,
        dispatch="sequential",
    )

    with workflow_run(seed=12):
        result = wrapped(Normal("x", 0, 1))

    assert result.num_atoms == 8
    assert len(seen) == 8
    assert all(not isinstance(value, Distribution) for value in seen)


def _unreachable(self, *args, **kwargs):
    raise AssertionError("a law that passes through unlifted is not evaluated")


def _law_claiming(capability: type) -> Distribution:
    """Return a minimal law whose class inherits *capability*."""
    # The conditioning capabilities are abstract, so the class implements
    # ``_condition_on``; the structural protocols need nothing further.
    law_type = types.new_class(
        f"_Claims{capability.__name__}",
        (Distribution, capability),
        exec_body=lambda namespace: namespace.update(_condition_on=_unreachable),
    )
    return law_type("law", OpaqueSpec())


@pytest.mark.parametrize("capability", DISTRIBUTION_HINT_PROTOCOLS, ids=lambda c: c.__name__)
def test_capability_annotation_passes_the_law_through(capability):
    seen = []

    def consume(law):
        seen.append(law)
        return 0.0

    consume.__annotations__ = {"law": capability}
    law = _law_claiming(capability)
    wrapped = Function(func=consume, n_broadcast_samples=8, dispatch="sequential")

    wrapped(law)

    assert len(seen) == 1
    assert seen[0] is law
