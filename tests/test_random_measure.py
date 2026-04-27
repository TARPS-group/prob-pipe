"""Tests for RandomMeasure and NumericRandomMeasure base classes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from probpipe import (
    Distribution,
    MultivariateNormal,
    Normal,
    NumericRandomMeasure,
    RandomFunction,
    RandomMeasure,
    SupportsExpectedDistribution,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    Weights,
    expected_distribution,
    mean,
    random_log_prob,
    random_unnormalized_log_prob,
    sample,
)
from probpipe.core._broadcast_distributions import _make_mixture_marginal
from probpipe.core._distribution_array import DistributionArray
from probpipe.core.constraints import real
from probpipe._utils import prod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Test fixtures: synthetic concrete subclasses
# ---------------------------------------------------------------------------


class _DiracLogProbFunction(RandomFunction):
    """Test-only RandomFunction whose call site returns a finite mixture
    of log-densities ``log p_i(x)``."""

    def __init__(self, components, weights, *, name="dirac_log_prob"):
        super().__init__(name=name)
        self._components = components
        self._w = weights

    def __call__(self, x):
        # log D(x) is a random scalar varying in D ~ M; return the
        # finite mixture of per-component log-densities.
        scalar_dists = [
            Normal(loc=c._log_prob(x), scale=jnp.array(1e-8), name=f"lp{i}")
            for i, c in enumerate(self._components)
        ]
        return _make_mixture_marginal(scalar_dists, weights=self._w)


class _DiracRandomMeasure(
    NumericRandomMeasure,
    SupportsSampling,
    SupportsExpectedDistribution,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
):
    """Finite-support random measure: weighted Dirac on inner ``Distribution[Array]``s.

    A draw is one of the inner components, picked by weight.  This is
    the simplest possible random measure — useful as a test fixture
    and as the ``DiracRandomMeasure`` baseline that sabi's empirical
    ``SurrogatePosterior`` will subclass.
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

    def __init__(self, components, weights=None, *, name=None):
        components = list(components)
        if not components:
            raise ValueError("DiracRandomMeasure requires at least one component")
        first = components[0]
        first_es = getattr(first, "event_shape", ())
        first_supp = getattr(first, "support", real)
        for i, c in enumerate(components):
            if getattr(c, "event_shape", ()) != first_es:
                raise ValueError(
                    f"All components must share event_shape; "
                    f"components[0]={first_es} but components[{i}]="
                    f"{getattr(c, 'event_shape', ())}"
                )
            if getattr(c, "support", real) != first_supp:
                raise ValueError("All components must share support")
        self._components = components
        self._w = Weights(n=len(components), weights=weights)
        self._inner_event_shape = first_es
        self._inner_support = first_supp
        super().__init__(name=name or "dirac_random_measure")
        self._approximate = True

    @property
    def inner_support(self):
        return self._inner_support

    @property
    def inner_event_shape(self):
        return self._inner_event_shape

    @property
    def components(self):
        return self._components

    @property
    def weights(self):
        return self._w.normalized

    def _sample(self, key, sample_shape=()):
        if sample_shape == ():
            idx = self._w.choice(key)
            return self._components[int(idx)]
        n_draws = prod(sample_shape)
        indices = self._w.choice(key, shape=(n_draws,))
        drawn = [self._components[int(i)] for i in indices]
        return DistributionArray(drawn, batch_shape=tuple(sample_shape))

    def _expected_distribution(self):
        return _make_mixture_marginal(
            self._components, weights=self._w, name=f"{self.name}_expected",
        )

    def _random_log_prob(self):
        return _DiracLogProbFunction(self._components, self._w)

    def _random_unnormalized_log_prob(self):
        return self._random_log_prob()


class _SamplingOnlyRandomMeasure(RandomMeasure, SupportsSampling):
    """Minimal subclass implementing only sampling — for opt-in protocol checks."""

    _sampling_cost = "low"
    _preferred_orchestration = None

    def __init__(self, component, name="sampling_only_rm"):
        super().__init__(name=name)
        self._component = component

    def _sample(self, key, sample_shape=()):
        if sample_shape == ():
            return self._component
        return DistributionArray(
            [self._component] * prod(sample_shape),
            batch_shape=tuple(sample_shape),
        )


# ---------------------------------------------------------------------------
# Inheritance & generics
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_random_measure_is_distribution(self):
        rm = _DiracRandomMeasure(
            [Normal(loc=0.0, scale=1.0, name="n0")],
        )
        assert isinstance(rm, RandomMeasure)
        assert isinstance(rm, Distribution)

    def test_numeric_subclass_relation(self):
        rm = _DiracRandomMeasure(
            [Normal(loc=0.0, scale=1.0, name="n0")],
        )
        assert isinstance(rm, NumericRandomMeasure)
        assert isinstance(rm, RandomMeasure)

    def test_no_outer_event_shape_or_support(self):
        """Outer ``support`` / ``event_shape`` are deliberately absent."""
        rm = _DiracRandomMeasure(
            [Normal(loc=0.0, scale=1.0, name="n0")],
        )
        # Inner versions exist; outer versions do not.
        assert hasattr(rm, "inner_support")
        assert hasattr(rm, "inner_event_shape")
        assert not hasattr(rm, "support")
        assert not hasattr(rm, "event_shape")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    def test_single_sample_returns_distribution(self, key):
        comps = [Normal(loc=float(i), scale=1.0, name=f"n{i}") for i in range(3)]
        rm = _DiracRandomMeasure(comps)
        drawn = sample(rm, key=key)
        assert isinstance(drawn, Distribution)
        assert drawn in comps  # one of the components, by weight

    def test_batched_sample_returns_distribution_array(self, key):
        comps = [Normal(loc=float(i), scale=1.0, name=f"n{i}") for i in range(4)]
        rm = _DiracRandomMeasure(comps)
        batch = sample(rm, key=key, sample_shape=(5,))
        assert isinstance(batch, DistributionArray)
        assert batch.batch_shape == (5,)
        assert batch.n == 5
        for i in range(5):
            assert isinstance(batch[i], Distribution)

    def test_multi_d_batched_sample(self, key):
        comps = [Normal(loc=float(i), scale=1.0, name=f"n{i}") for i in range(4)]
        rm = _DiracRandomMeasure(comps)
        batch = sample(rm, key=key, sample_shape=(2, 3))
        assert isinstance(batch, DistributionArray)
        assert batch.batch_shape == (2, 3)
        assert batch.n == 6

    def test_sampling_protocol_opt_in_present(self, key):
        rm = _DiracRandomMeasure(
            [Normal(loc=0.0, scale=1.0, name="n0")],
        )
        assert isinstance(rm, SupportsSampling)


# ---------------------------------------------------------------------------
# Expected distribution
# ---------------------------------------------------------------------------


class TestExpectedDistribution:
    def test_returns_distribution(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"n{i}") for i in range(3)]
        rm = _DiracRandomMeasure(comps)
        ed = expected_distribution(rm)
        assert isinstance(ed, Distribution)

    def test_mean_matches_weighted_mean(self):
        """E[D̄] under the marginalised distribution is the weighted mean
        of the component means.
        """
        locs = [0.0, 2.0, 5.0]
        comps = [
            Normal(loc=loc, scale=1.0, name=f"n{i}") for i, loc in enumerate(locs)
        ]
        weights = jnp.array([0.2, 0.3, 0.5])
        rm = _DiracRandomMeasure(comps, weights=weights)
        ed = expected_distribution(rm)
        m = mean(ed)
        expected_mean = float((weights * jnp.array(locs)).sum())
        assert jnp.allclose(m, expected_mean, atol=1e-6)

    def test_protocol_isinstance(self):
        rm = _DiracRandomMeasure(
            [Normal(loc=0.0, scale=1.0, name="n0")],
        )
        assert isinstance(rm, SupportsExpectedDistribution)


# ---------------------------------------------------------------------------
# Random log-prob protocols
# ---------------------------------------------------------------------------


class TestRandomLogProb:
    def test_random_log_prob_returns_random_function(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"n{i}") for i in range(3)]
        rm = _DiracRandomMeasure(comps)
        rf = random_log_prob(rm)
        assert isinstance(rf, RandomFunction)

    def test_random_log_prob_call_returns_distribution(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"n{i}") for i in range(3)]
        rm = _DiracRandomMeasure(comps)
        rf = random_log_prob(rm)
        marginal = rf(jnp.array(1.0))
        assert isinstance(marginal, Distribution)

    def test_random_unnormalized_log_prob_returns_random_function(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"n{i}") for i in range(3)]
        rm = _DiracRandomMeasure(comps)
        rf = random_unnormalized_log_prob(rm)
        assert isinstance(rf, RandomFunction)

    def test_protocols_isinstance(self):
        rm = _DiracRandomMeasure(
            [Normal(loc=0.0, scale=1.0, name="n0")],
        )
        assert isinstance(rm, SupportsRandomLogProb)
        assert isinstance(rm, SupportsRandomUnnormalizedLogProb)


# ---------------------------------------------------------------------------
# Optional protocol opt-out
# ---------------------------------------------------------------------------


class TestProtocolOptIn:
    def test_sampling_only_rm_opts_out_of_other_protocols(self):
        rm = _SamplingOnlyRandomMeasure(Normal(loc=0.0, scale=1.0, name="n0"))
        assert isinstance(rm, SupportsSampling)
        assert not isinstance(rm, SupportsExpectedDistribution)
        assert not isinstance(rm, SupportsRandomLogProb)
        assert not isinstance(rm, SupportsRandomUnnormalizedLogProb)

    def test_unsupported_op_raises_typerror(self):
        rm = _SamplingOnlyRandomMeasure(Normal(loc=0.0, scale=1.0, name="n0"))
        with pytest.raises(TypeError, match="expected_distribution"):
            expected_distribution(rm)
        with pytest.raises(TypeError, match="random_log_prob"):
            random_log_prob(rm)
        with pytest.raises(TypeError, match="random_unnormalized_log_prob"):
            random_unnormalized_log_prob(rm)


# ---------------------------------------------------------------------------
# Inner support / inner event shape
# ---------------------------------------------------------------------------


class TestInnerMetadata:
    def test_inner_support_propagates(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"n{i}") for i in range(3)]
        rm = _DiracRandomMeasure(comps)
        assert rm.inner_support == comps[0].support

    def test_inner_event_shape_scalar(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"n{i}") for i in range(3)]
        rm = _DiracRandomMeasure(comps)
        assert rm.inner_event_shape == ()

    def test_inner_event_shape_vector(self):
        comps = [
            MultivariateNormal(
                loc=jnp.zeros(3) + i,
                cov=jnp.eye(3),
                name=f"mvn{i}",
            )
            for i in range(2)
        ]
        rm = _DiracRandomMeasure(comps)
        assert rm.inner_event_shape == (3,)

    def test_mismatched_inner_event_shape_raises(self):
        comps = [
            Normal(loc=0.0, scale=1.0, name="scalar"),
            MultivariateNormal(
                loc=jnp.zeros(3),
                cov=jnp.eye(3),
                name="vector",
            ),
        ]
        with pytest.raises(ValueError, match="event_shape"):
            _DiracRandomMeasure(comps)


# ---------------------------------------------------------------------------
# Batch of random measures via DistributionArray
# ---------------------------------------------------------------------------


class TestBatchOfRandomMeasures:
    def test_distribution_array_of_random_measures(self, key):
        rm1 = _DiracRandomMeasure(
            [Normal(loc=0.0, scale=1.0, name="a")],
            name="rm1",
        )
        rm2 = _DiracRandomMeasure(
            [Normal(loc=5.0, scale=1.0, name="b")],
            name="rm2",
        )
        batch = DistributionArray([rm1, rm2])
        assert batch.n == 2
        assert batch[0] is rm1
        assert batch[1] is rm2
        # Each element is itself a random measure
        assert isinstance(batch[0], RandomMeasure)
