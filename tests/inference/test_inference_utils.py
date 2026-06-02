"""Tests for the backend-agnostic inference utilities in
``probpipe.inference._inference_utils``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

from probpipe import (
    Normal,
    ProductDistribution,
    SimpleModel,
)
from probpipe.core._distribution_base import Distribution
from probpipe.core._numeric_record import NumericRecord
from probpipe.core.protocols import SupportsSampling
from probpipe.inference._inference_utils import (
    as_prng_key,
    build_likelihood_flat,
    build_target_log_prob,
    build_target_log_prob_flat,
    get_init_state,
    get_prior,
    is_jax_traceable,
    run_chain_scan,
)
from probpipe.modeling._likelihood import Likelihood


class _IdentityLikelihood(Likelihood):
    """Trivial likelihood for unit-test fixtures: ``log p(y | theta) = 0``."""

    def log_likelihood(self, params, data) -> float:
        return jnp.asarray(0.0)


class _GaussianMeanLikelihood(Likelihood):
    """Gaussian likelihood with known scale, mean = flat parameter vector.

    ``log p(y | theta) = sum_i log N(y_i; mu, scale^2)`` with ``mu`` the
    (scalar) flat parameter. Used to check ``build_likelihood_flat``
    against an independently computed value.
    """

    def __init__(self, scale: float = 2.0):
        self.scale = scale

    def log_likelihood(self, params, data):
        mu = jnp.reshape(jnp.asarray(params), ())
        s = self.scale
        return jnp.sum(
            -0.5 * ((jnp.asarray(data) - mu) / s) ** 2
            - jnp.log(s) - 0.5 * jnp.log(2 * jnp.pi)
        )


@pytest.fixture
def small_model() -> SimpleModel:
    """SimpleModel with a 2-field ProductDistribution prior."""
    prior = ProductDistribution(
        a=Normal(loc=0.0, scale=1.0, name="a"),
        b=Normal(loc=2.0, scale=0.5, name="b"),
    )
    return SimpleModel(prior, _IdentityLikelihood(), name="m")


class TestBuildTargetLogProbFlat:
    """Characterise the flat-vector target builder used by BlackJAX backends."""

    def test_flat_target_matches_record_target(self, small_model):
        observed = jnp.zeros((4,))
        target_record = build_target_log_prob(small_model, observed)
        target_flat, flat_init, template = build_target_log_prob_flat(
            small_model, observed,
        )
        # Round-trip: unflatten the flat init back to a Record and confirm
        # the two callables agree.
        record_init = NumericRecord.unflatten(flat_init, template=template)
        np.testing.assert_allclose(
            float(target_flat(flat_init)),
            float(target_record(record_init)),
            rtol=0, atol=1e-6,
        )

    def test_flat_init_dim_matches_template_flat_size(self, small_model):
        _, flat_init, template = build_target_log_prob_flat(
            small_model, observed=None,
        )
        # Both fields are scalar Normals: flat_size == 2.
        assert flat_init.shape == (template.flat_size,) == (2,)

    def test_template_field_order_preserved(self, small_model):
        _, _, template = build_target_log_prob_flat(small_model, observed=None)
        # Insertion order from the ProductDistribution constructor.
        assert template.fields == ("a", "b")

    def test_bare_distribution_falls_through_unwrapped(self):
        """A target with no Record-shaped prior round-trips its log-prob unchanged.

        For a bare ``SupportsLogProb`` whose ``_unnormalized_log_prob``
        already takes a flat array, ``build_target_log_prob_flat``
        passes the callable through verbatim and returns
        ``record_template=None``. This is the path BlackJAX MCMC uses
        for hand-rolled distributions that don't carry a Record-shaped
        prior.
        """
        class _FlatGaussian:
            event_shape = (2,)

            def _unnormalized_log_prob(self, x):
                return -0.5 * jnp.sum(jnp.asarray(x) ** 2)

        target_flat, flat_init, template = build_target_log_prob_flat(
            _FlatGaussian(), observed=None,
        )
        assert template is None
        assert flat_init.shape == (2,)
        np.testing.assert_allclose(
            float(target_flat(jnp.asarray([1.0, -1.0]))), -1.0,
        )


class TestGetPrior:
    """``get_prior`` returns the SimpleModel's prior or the dist itself."""

    def test_simple_model_returns_prior(self, small_model):
        assert get_prior(small_model) is small_model._prior

    def test_bare_distribution_returns_self(self):
        d = Normal(loc=0.0, scale=1.0, name="x")
        assert get_prior(d) is d


# ---------------------------------------------------------------------------
# Stub pytree nodes for run_chain_scan (mimic the BlackJAX state/info contract)
# ---------------------------------------------------------------------------


@jtu.register_pytree_node_class
class _FakeState:
    """Minimal BlackJAX-style sampler state: carries a ``position``."""

    def __init__(self, position):
        self.position = position

    def tree_flatten(self):
        return (self.position,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jtu.register_pytree_node_class
class _FakeInfo:
    """Minimal BlackJAX-style per-step info object with one field."""

    def __init__(self, accept):
        self.accept = accept

    def tree_flatten(self):
        return (self.accept,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


class _FakeSampler:
    """Fake BlackJAX sampler: each ``step`` advances the position by one
    and reports a constant ``accept`` rate, so the stacked outputs are
    trivially predictable.
    """

    def step(self, key, state):
        return _FakeState(state.position + 1.0), _FakeInfo(jnp.asarray(0.5))


class TestAsPRNGKey:
    """``as_prng_key`` upgrades int seeds and passes keys through."""

    def test_int_seed_becomes_prng_key(self):
        key = as_prng_key(0)
        ref = jax.random.PRNGKey(0)
        assert key.shape == ref.shape
        assert key.dtype == ref.dtype
        np.testing.assert_array_equal(np.asarray(key), np.asarray(ref))

    def test_existing_key_passes_through(self):
        ref = jax.random.PRNGKey(7)
        # Passthrough is by identity: no copy, no re-derivation.
        assert as_prng_key(ref) is ref


class TestRunChainScan:
    """``run_chain_scan`` drives a sampler under ``lax.scan`` and stacks
    positions / infos along the leading axis.
    """

    def test_positions_and_infos_shapes(self):
        event_shape = (2,)
        num_results = 5
        init = _FakeState(jnp.zeros(event_shape))
        positions, infos = run_chain_scan(
            _FakeSampler(), init, num_results, jax.random.PRNGKey(0),
        )
        # Positions: (num_results, *event_shape).
        assert positions.shape == (num_results, *event_shape)
        # Infos: pytree stacked along axis 0, length num_results.
        assert infos.accept.shape == (num_results,)

    def test_positions_track_the_sampler_recurrence(self):
        # Each step adds one, starting from zeros, so row i == i + 1.
        init = _FakeState(jnp.zeros((2,)))
        positions, _ = run_chain_scan(
            _FakeSampler(), init, 4, jax.random.PRNGKey(0),
        )
        expected = np.arange(1, 5)[:, None] * np.ones((1, 2))
        np.testing.assert_allclose(np.asarray(positions), expected)


class TestIsJaxTraceable:
    """``is_jax_traceable`` probes whether a fn traces at an init state."""

    def test_traceable_fn(self):
        init = jnp.array([1.0, 2.0])
        assert is_jax_traceable(lambda x: jnp.sum(x ** 2), init) is True

    def test_non_traceable_fn(self):
        # ``float(np.asarray(x))`` forces concretisation of a traced value,
        # which raises during ``make_jaxpr``.
        def host_side(x):
            return jnp.asarray(float(np.asarray(x).sum()))

        init = jnp.array([1.0, 2.0])
        assert is_jax_traceable(host_side, init) is False


class TestBuildLikelihoodFlat:
    """``build_likelihood_flat`` returns the *likelihood alone* as a flat-
    vector callable (the ESS entry point).
    """

    @pytest.fixture
    def gaussian_model(self):
        prior = ProductDistribution(mu=Normal(loc=0.0, scale=1.0, name="mu"))
        return SimpleModel(prior, _GaussianMeanLikelihood(scale=2.0), name="g")

    def test_returns_scalar_log_likelihood(self, gaussian_model):
        data = jnp.array([1.0, -1.0, 0.5])
        llf = build_likelihood_flat(
            gaussian_model._prior, gaussian_model._likelihood, data,
        )
        assert callable(llf)
        out = llf(jnp.array([0.3]))
        # Scalar (rank-0) log-likelihood.
        assert jnp.ndim(out) == 0

    def test_matches_independent_gaussian(self, gaussian_model):
        data = jnp.array([1.0, -1.0, 0.5])
        llf = build_likelihood_flat(
            gaussian_model._prior, gaussian_model._likelihood, data,
        )
        mu, scale = 0.3, 2.0
        expected = np.sum(
            -0.5 * ((np.asarray(data) - mu) / scale) ** 2
            - np.log(scale) - 0.5 * np.log(2 * np.pi)
        )
        np.testing.assert_allclose(
            float(llf(jnp.array([mu]))), expected, rtol=0, atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Stub distributions for get_init_state branch coverage
# ---------------------------------------------------------------------------


class _EventShapeOnlyDist(Distribution):
    """Distribution with ``event_shape`` but no ``_sample`` (not
    ``SupportsSampling``) — exercises the Stan ``Uniform(-2, 2)`` fallback.
    """

    event_shape = (3,)

    def __init__(self):
        super().__init__(name="event_shape_only")

    def _unnormalized_log_prob(self, value):
        return -0.5 * jnp.sum(jnp.asarray(value) ** 2)


class _NoInitHeuristicDist(Distribution):
    """Distribution with neither sampling nor ``event_shape`` — exercises
    the ``get_init_state`` raise branch.
    """

    def __init__(self):
        super().__init__(name="no_init_heuristic")

    def _unnormalized_log_prob(self, value):
        return jnp.asarray(0.0)


class TestGetInitState:
    """Cover every documented branch of ``get_init_state``."""

    def test_explicit_init_passthrough(self):
        # Branch 1: explicit init returned verbatim (cast to prior dtype).
        prior = Normal(loc=0.0, scale=1.0, name="x")
        out = get_init_state(prior, init=jnp.array([3.0, 4.0]))
        np.testing.assert_array_equal(np.asarray(out), np.array([3.0, 4.0]))
        # Cast to the prior dtype: a default-float Normal yields a float
        # array (not, e.g., int) regardless of the input dtype.
        assert jnp.issubdtype(out.dtype, jnp.floating)

    def test_explicit_init_casts_dtype(self):
        # An integer-valued init is cast to the prior's float dtype.
        prior = Normal(loc=0.0, scale=1.0, name="x")
        out = get_init_state(prior, init=np.array([1, 2], dtype=np.int32))
        assert jnp.issubdtype(out.dtype, jnp.floating)
        np.testing.assert_allclose(np.asarray(out), np.array([1.0, 2.0]))

    def test_prior_sample_path(self):
        # Branch 2: prior implements SupportsSampling -> draw a sample.
        prior = Normal(loc=0.0, scale=1.0, name="x")
        assert isinstance(prior, SupportsSampling)
        out = get_init_state(prior, init=None, random_seed=0)
        assert out.shape == (1,)            # scalar Normal -> length-1 vector
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_stan_uniform_fallback(self):
        # Branch 3: no sampling path, but event_shape exposed -> Uniform(-2, 2).
        dist = _EventShapeOnlyDist()
        assert not isinstance(dist, SupportsSampling)
        out = get_init_state(dist, init=None, random_seed=0)
        assert out.shape == dist.event_shape
        assert bool(jnp.all((out >= -2.0) & (out <= 2.0)))

    def test_raises_without_sampling_or_event_shape(self):
        # Branch 4: neither heuristic applies -> ValueError.
        with pytest.raises(ValueError, match="Cannot determine initial state"):
            get_init_state(_NoInitHeuristicDist(), init=None)

    def test_data_not_consulted_so_seed_determines_init(self):
        # By design get_init_state takes no observed-data argument: the
        # init depends only on (dist, init, random_seed). Same seed ->
        # identical init across calls; different seed -> (generally)
        # different init.
        dist = _EventShapeOnlyDist()
        a = get_init_state(dist, init=None, random_seed=7)
        b = get_init_state(dist, init=None, random_seed=7)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        c = get_init_state(dist, init=None, random_seed=8)
        assert not bool(jnp.all(a == c))
