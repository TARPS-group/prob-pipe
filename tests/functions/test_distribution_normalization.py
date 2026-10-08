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
    DistributionBatch,
    EmpiricalDistribution,
    Function,
    KDEDistribution,
    Normal,
    NumericArrayBatch,
    NumericDistribution,
    OpaqueSpec,
    ResolutionError,
    converter_registry,
    log_prob,
    mean,
    workflow_run,
)
from probpipe.distributions._capabilities import SupportsLogProb
from probpipe.functions._normalization import (
    DISTRIBUTION_HINT_PROTOCOLS,
    normalize_distribution_values,
)
from probpipe.values._binding import make_signature_info_from_signature


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

        source = Normal(loc=0.0, scale=1.0, label="source")

        with pytest.raises(
            ResolutionError,
            match=r"cannot convert parameter 'dist' to UnsupportedDistribution: no method is registered for "
            r"\(Normal, UnsupportedDistribution\)",
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

    def test_protocol_hint_raises_when_no_conversion_applies(self, empirical_dist):
        """A failed conversion raises, naming the parameter, rather than passing the law through."""
        failure = ResolutionError("no converter applies")
        with (
            patch.object(converter_registry, "convert", side_effect=failure),
            pytest.raises(
                ResolutionError, match="cannot convert parameter 'dist' to SupportsLogProb"
            ),
        ):
            normalize_distribution_values(
                values={"dist": empirical_dist},
                signature_info=_signature_info(("dist",), {"dist": SupportsLogProb}),
            )

    def test_a_conversions_entry_selects_the_converter_and_passes_its_options(self):
        normalized = normalize_distribution_values(
            values={"dist": Normal("x", 0.0, 1.0)},
            signature_info=_signature_info(("dist",), {"dist": SupportsLogProb}),
        )
        assert isinstance(normalized["dist"], Normal)
        with workflow_run(seed=0):
            normalized = normalize_distribution_values(
                values={"dist": Normal("x", 0.0, 1.0)},
                signature_info=_signature_info(("dist",), {"dist": EmpiricalDistribution}),
                conversions={"dist": {"method": "empirical", "num_samples": 7}},
            )
        assert normalized["dist"].num_atoms == 7

    def test_a_conversions_entry_restricted_to_exact_converters_refuses_an_approximation(
        self, empirical_dist
    ):
        with pytest.raises(ResolutionError, match="exact_only"):
            normalize_distribution_values(
                values={"dist": empirical_dist},
                signature_info=_signature_info(("dist",), {"dist": SupportsLogProb}),
                conversions={"dist": {"exact_only": True}},
            )

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

    def test_a_batch_of_one_law_remains_a_batch(self):
        batch = DistributionBatch("one_cell", [Normal("x", 3.0, 1.0)], "cell")

        normalized = normalize_distribution_values(
            values={"dist": batch},
            signature_info=_signature_info(("dist",), {"dist": Normal}),
        )

        assert normalized["dist"] is batch

    def test_unhinted_external_distribution_converts_for_broadcast(
        self,
        normal_external,
    ):
        normalized = normalize_distribution_values(
            values={"x": normal_external},
            signature_info=_signature_info(("x",)),
        )

        assert isinstance(normalized["x"], NumericDistribution)
        assert normalized["x"] is not normal_external


# Facade checks below are retained to verify Function wiring.
class TestHintedDistributionConversion:
    def test_concrete_distribution_hint_converts_external_distribution(
        self,
        mean_recorder,
        normal_external,
    ):
        mean_of_normal, seen = mean_recorder
        wf = Function(label="mean_of_normal", fn=mean_of_normal, dispatch="sequential")

        result = wf(dist=normal_external)

        assert isinstance(seen[0], Normal)
        assert float(result) == 2.0

    def test_protocol_hint_converts_distribution_that_lacks_protocol(self, empirical_dist):
        seen = []

        def log_prob_at_zero(dist: SupportsLogProb):
            seen.append(dist)
            return log_prob(dist, jnp.asarray([0.0]))

        wf = Function(label="log_prob_at_zero", fn=log_prob_at_zero, dispatch="sequential")

        result = wf(dist=empirical_dist)

        assert seen[0] is not empirical_dist
        assert isinstance(seen[0], KDEDistribution)
        assert isinstance(seen[0], SupportsLogProb)
        assert jnp.isfinite(jnp.asarray(result)).all()


class TestDistributionBatchHandling:
    def test_a_batch_of_one_law_is_swept(self, mean_recorder):
        mean_of_normal, seen = mean_recorder
        batch = DistributionBatch("one_cell", [Normal("x", 3.0, 1.0)], "cell")
        wf = Function(label="mean_of_normal", fn=mean_of_normal, dispatch="sequential")

        result = wf(dist=batch)

        assert isinstance(seen[0], Normal)

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
            label="double",
            fn=double,
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
        np.testing.assert_allclose(result.atoms, 2.0 * jnp.stack(seen_values), atol=0.0)


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
        label="consume",
        fn=consume,
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
    wrapped = Function(label="consume", fn=consume, n_broadcast_samples=8, dispatch="sequential")

    wrapped(law)

    assert len(seen) == 1
    assert seen[0] is law
