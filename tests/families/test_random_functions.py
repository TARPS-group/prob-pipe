"""The random functions and random measures (VII.5)."""

from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Distribution,
    DistributionSpec,
    FunctionSpec,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    OpaqueSpec,
    OutputSpec,
    RandomFunction,
    RandomMeasure,
    ResolutionError,
    SupportsMean,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    Weights,
    log_prob,
    mean,
    random_log_prob,
    random_unnormalized_log_prob,
    sample,
)
from probpipe.core.constraints import real
from probpipe.distributions import DistributionBatch


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Random functions
# ---------------------------------------------------------------------------


class _MinimalRandomFunction(RandomFunction):
    """A random function whose value at every point is one law."""

    def __call__(self, x):
        return Normal("y", jnp.zeros(3), jnp.ones(3))


class TestRandomFunction:
    def test_cannot_instantiate_directly(self):
        """The base is abstract until ``__call__`` is implemented."""
        with pytest.raises(TypeError, match="abstract"):
            RandomFunction("f")

    def test_minimal_subclass_instantiates(self):
        rf = _MinimalRandomFunction("rf")
        assert isinstance(rf, RandomFunction)
        assert isinstance(rf, Distribution)

    def test_call_returns_distribution(self):
        assert isinstance(_MinimalRandomFunction("rf")(jnp.array([1.0, 2.0, 3.0])), Distribution)

    def test_a_draw_is_an_unspecified_callable_by_default(self):
        assert _MinimalRandomFunction("rf").event_spec == OutputSpec(rf=FunctionSpec())

    def test_a_hole_in_the_event_is_an_unspecified_callable(self):
        assert _MinimalRandomFunction("rf", OutputSpec(g=None)).event_spec == OutputSpec(
            g=FunctionSpec()
        )

    @pytest.mark.parametrize(
        "event_spec",
        [OutputSpec(rf=NumericArraySpec(())), NumericArraySpec(())],
        ids=["declaration", "term-spec"],
    )
    def test_an_event_that_is_not_a_function_raises(self, event_spec):
        with pytest.raises(TypeError, match="FunctionSpec"):
            _MinimalRandomFunction("rf", event_spec)

    def test_sample_raises(self, key):
        with pytest.raises(ResolutionError, match="does not claim SupportsSampling"):
            sample(_MinimalRandomFunction("rf"))

    def test_sample_with_shape_raises(self, key):
        with pytest.raises(ResolutionError, match="does not claim SupportsSampling"):
            sample(_MinimalRandomFunction("rf"), sample_shape=(5,))

    def test_log_prob_raises(self):
        with pytest.raises(ResolutionError, match="does not claim SupportsLogProb"):
            log_prob(_MinimalRandomFunction("rf"), lambda x: x)


# ---------------------------------------------------------------------------
# Random measures
# ---------------------------------------------------------------------------


class _Mixture(Distribution, SupportsMean):
    """The finite mixture ``Σᵢ wᵢ pᵢ`` of laws that share one declaration, by its mean."""

    def __init__(self, components, weights, *, name="mixture"):
        super().__init__(name, components[0].event_spec)
        self._components = list(components)
        self._w = weights

    def _mean(self):
        return self._w.mean(jnp.stack([component._mean() for component in self._components]))


class _DiracLogProbFunction(RandomFunction):
    """The random log-density of a finite mixture: ``log p_i(x)`` with weight ``w_i``."""

    def __init__(self, components, weights, *, name="dirac_log_prob"):
        super().__init__(name=name)
        self._components = components
        self._w = weights

    def __call__(self, x):
        scalar_dists = [
            Normal(loc=c._log_prob(x), scale=jnp.array(1e-8), name="lp") for c in self._components
        ]
        return _Mixture(scalar_dists, self._w)


class _DiracRandomMeasure(
    RandomMeasure,
    SupportsSampling,
    SupportsMean,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
):
    """A weighted finite set of laws; a draw is one of them, picked by weight.

    The drawn laws share the first one's event shape and support, which the
    measure's ``DistributionSpec`` declares.
    """

    def __init__(self, components, weights=None, *, name=None):
        components = list(components)
        if not components:
            raise ValueError("_DiracRandomMeasure requires at least one component")
        first = components[0]
        for i, c in enumerate(components):
            if c.event_shape != first.event_shape:
                raise ValueError(
                    f"All components must share event_shape; components[0]="
                    f"{first.event_shape} but components[{i}]={c.event_shape}"
                )
            if c.support != first.support:
                raise ValueError("All components must share support")
        self._components = components
        self._w = Weights(n=len(components), weights=weights)
        super().__init__(name or "dirac_random_measure", DistributionSpec(first.event_spec))

    @property
    def components(self):
        return self._components

    def _sample(self, key, sample_shape=()):
        if sample_shape == ():
            return self._components[int(self._w.choice(key))]
        indices = self._w.choice(key, shape=(prod(sample_shape),))
        return _object_array([self._components[int(i)] for i in indices], sample_shape)

    def _mean(self):
        return _Mixture(self._components, self._w, name=f"{self.label}_expected")

    def _random_log_prob(self):
        return _DiracLogProbFunction(self._components, self._w)

    def _random_unnormalized_log_prob(self):
        return self._random_log_prob()


class _SamplingOnlyRandomMeasure(RandomMeasure, SupportsSampling):
    """A random measure that only draws laws."""

    def __init__(self, component, name="sampling_only_rm"):
        super().__init__(name, DistributionSpec(component.event_spec))
        self._component = component

    def _sample(self, key, sample_shape=()):
        if sample_shape == ():
            return self._component
        return _object_array([self._component] * prod(sample_shape), sample_shape)


def _object_array(laws, sample_shape):
    """*laws* as an object array of shape *sample_shape*, the raw form of a batch of laws."""
    store = np.empty(len(laws), dtype=object)
    for position, law in enumerate(laws):
        store[position] = law
    return store.reshape(tuple(sample_shape))


def _normals(n):
    return [Normal(loc=float(i), scale=1.0, name="n") for i in range(n)]


class TestInheritance:
    def test_random_measure_is_distribution(self):
        rm = _DiracRandomMeasure(_normals(1))
        assert isinstance(rm, RandomMeasure)
        assert isinstance(rm, Distribution)

    def test_a_draw_is_an_opaque_law_by_default(self):
        assert RandomMeasure("m").event_spec.spec == DistributionSpec(OutputSpec(m=OpaqueSpec()))

    def test_a_hole_in_the_event_is_an_opaque_law(self):
        assert RandomMeasure("m", OutputSpec(g=None)).event_spec == OutputSpec(
            g=DistributionSpec(OutputSpec(m=OpaqueSpec()))
        )

    @pytest.mark.parametrize(
        "event_spec",
        [OutputSpec(m=NumericArraySpec(())), NumericArraySpec(())],
        ids=["declaration", "term-spec"],
    )
    def test_an_event_that_is_not_a_law_raises(self, event_spec):
        with pytest.raises(TypeError, match="DistributionSpec"):
            RandomMeasure("m", event_spec)

    def test_no_outer_event_shape_or_support(self):
        """A law-valued draw has no array shape or support of its own."""
        rm = _DiracRandomMeasure(_normals(1))
        assert not hasattr(rm, "support")
        assert not hasattr(rm, "event_shape")
        with pytest.raises(AttributeError, match="does not draw a single array"):
            _ = rm.event_shape


class TestSampling:
    def test_single_sample_returns_distribution(self, key):
        drawn = sample(_DiracRandomMeasure(_normals(3)))
        assert isinstance(drawn, Distribution)
        assert float(mean(drawn)) in {0.0, 1.0, 2.0}

    def test_batched_sample_returns_distribution_batch(self, key):
        batch = sample(_DiracRandomMeasure(_normals(4)), sample_shape=(5,))
        assert isinstance(batch, DistributionBatch)
        assert batch.batch_shape == (5,)
        assert len(batch) == 5
        for i in range(5):
            assert isinstance(batch[i], Distribution)

    def test_multi_d_batched_sample(self, key):
        batch = sample(_DiracRandomMeasure(_normals(4)), sample_shape=(2, 3))
        assert isinstance(batch, DistributionBatch)
        assert batch.batch_shape == (2, 3)
        assert batch.batch_size == 6

    def test_sampling_protocol_opt_in_present(self):
        assert isinstance(_DiracRandomMeasure(_normals(1)), SupportsSampling)


class TestMean:
    def test_returns_distribution(self):
        """The mean of a random measure is the marginalized law ``D̄(A) = ∫ D(A) dM(D)``."""
        assert isinstance(mean(_DiracRandomMeasure(_normals(3))), Distribution)

    def test_outer_mean_matches_weighted_inner_mean(self):
        locs = [0.0, 2.0, 5.0]
        comps = [Normal(loc=loc, scale=1.0, name=f"n{i}") for i, loc in enumerate(locs)]
        weights = jnp.array([0.2, 0.3, 0.5])
        m = mean(mean(_DiracRandomMeasure(comps, weights=weights)))
        assert jnp.allclose(m, float((weights * jnp.array(locs)).sum()), atol=1e-6)

    def test_protocol_isinstance(self):
        assert isinstance(_DiracRandomMeasure(_normals(1)), SupportsMean)


class TestRandomLogProb:
    def test_random_log_prob_returns_random_function(self):
        assert isinstance(random_log_prob(_DiracRandomMeasure(_normals(3))), RandomFunction)

    def test_random_log_prob_call_returns_distribution(self):
        rf = random_log_prob(_DiracRandomMeasure(_normals(3)))
        assert isinstance(rf(jnp.array(1.0)), Distribution)

    def test_random_unnormalized_log_prob_returns_random_function(self):
        rf = random_unnormalized_log_prob(_DiracRandomMeasure(_normals(3)))
        assert isinstance(rf, RandomFunction)

    def test_protocols_isinstance(self):
        rm = _DiracRandomMeasure(_normals(1))
        assert isinstance(rm, SupportsRandomLogProb)
        assert isinstance(rm, SupportsRandomUnnormalizedLogProb)


class TestProtocolOptIn:
    def test_the_base_claims_no_random_log_density(self):
        rm = _SamplingOnlyRandomMeasure(Normal(loc=0.0, scale=1.0, name="n0"))
        assert isinstance(rm, SupportsSampling)
        assert not isinstance(rm, SupportsMean)
        assert not isinstance(rm, SupportsRandomLogProb)
        assert not isinstance(rm, SupportsRandomUnnormalizedLogProb)

    def test_a_sampling_measure_has_a_monte_carlo_mean_and_no_random_density(self):
        """The mean of a measure that only samples is the mixture of its draws."""
        rm = _SamplingOnlyRandomMeasure(Normal(loc=0.0, scale=1.0, name="n0"))
        assert isinstance(mean(rm), Distribution)
        with pytest.raises(ResolutionError, match="SupportsRandomLogProb"):
            random_log_prob(rm)
        with pytest.raises(ResolutionError, match="SupportsRandomUnnormalizedLogProb"):
            random_unnormalized_log_prob(rm)


class TestTheDrawnLawsDeclaration:
    """The ``DistributionSpec`` of the event declares the drawn laws' event."""

    @staticmethod
    def _drawn(rm):
        return rm.event_spec.spec.event_spec.spec

    def test_the_support_of_the_drawn_laws(self):
        comps = _normals(3)
        assert self._drawn(_DiracRandomMeasure(comps)).support == comps[0].support == real

    def test_a_scalar_event_shape(self):
        assert self._drawn(_DiracRandomMeasure(_normals(3))).shape == ()

    def test_a_vector_event_shape(self):
        comps = [
            MultivariateNormal(loc=jnp.zeros(3) + i, cov=jnp.eye(3), name=f"mvn{i}")
            for i in range(2)
        ]
        assert self._drawn(_DiracRandomMeasure(comps)).shape == (3,)

    def test_mismatched_event_shapes_raise(self):
        comps = [
            Normal(loc=0.0, scale=1.0, name="scalar"),
            MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="vector"),
        ]
        with pytest.raises(ValueError, match="event_shape"):
            _DiracRandomMeasure(comps)


class TestBatchOfRandomMeasures:
    def test_a_distribution_batch_of_random_measures(self):
        rm1 = _DiracRandomMeasure([Normal(loc=0.0, scale=1.0, name="x")], name="rm")
        rm2 = _DiracRandomMeasure([Normal(loc=5.0, scale=1.0, name="x")], name="rm")
        batch = DistributionBatch("measures", [rm1, rm2], "measure")
        assert len(batch) == 2
        assert batch[0].components is rm1.components
        assert batch[1].components is rm2.components
        assert isinstance(batch[0], RandomMeasure)
