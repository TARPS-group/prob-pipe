"""Tests for the backend-agnostic inference utilities in
``probpipe.inference._inference_utils``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    Dirichlet,
    Gamma,
    HalfNormal,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    NumericRecord,
    NumericRecordSpec,
    OpaqueSpec,
    condition_on,
    conditional_distribution,
    inference_method_registry,
    workflow_run,
)
from probpipe.distributions import FactoredDistribution
from probpipe.distributions._capabilities import SupportsSampling
from probpipe.distributions._distribution import Distribution
from probpipe.inference._inference_utils import (
    as_prng_key,
    build_target_log_prob,
    build_target_log_prob_flat,
    extract_event_spec,
    get_init_state,
    is_jax_traceable,
    likelihood_flat,
    model_factors,
    observed_target,
    parameter_given,
    posterior_var_order,
    run_chain_scan,
    unconstrained_chain,
)
from tests._posterior import flat_chains, flat_draws
from tests.inference.canonical import ObservationKernel


def _gaussian_mean(prior, n, scale=1.0):
    """``y_i ~ N(mu, scale^2)`` for ``n`` observations, times *prior* over ``mu``."""
    likelihood = ObservationKernel(
        "y",
        dict(prior.event_spec.components),
        NumericArraySpec((n,)),
        lambda mu: tfd.Independent(tfd.Normal(jnp.broadcast_to(mu, (n,)), scale), 1),
    )
    return likelihood * prior


@pytest.fixture
def small_model():
    """The unnormalized conditional of a joint with a 2-field factored prior."""
    prior = Normal(loc=0.0, scale=1.0, label="a") * Normal(loc=2.0, scale=0.5, label="b")
    likelihood = ObservationKernel(
        "y",
        dict(prior.event_spec.components),
        NumericArraySpec((4,)),
        lambda a, b: tfd.Independent(tfd.Normal(jnp.zeros(4), 1.0), 1),
    )
    return observed_target(likelihood * prior, {"y": jnp.zeros((4,))})


class TestBuildTargetLogProbFlat:
    """Characterise the flat-vector target builder used by BlackJAX backends."""

    def test_flat_target_matches_record_target(self, small_model):
        target_record = build_target_log_prob(small_model, None)
        target_flat, flat_init, event_spec = build_target_log_prob_flat(small_model, None)
        # Round-trip: unflatten the flat init back to a Record and confirm
        # the two callables agree.
        record_init = NumericRecord.from_vector("nr", event_spec.spec, flat_init)
        np.testing.assert_allclose(
            float(target_flat(flat_init)),
            float(target_record(record_init)),
            rtol=0,
            atol=1e-6,
        )

    def test_flat_init_dim_matches_the_declared_vector_size(self, small_model):
        _, flat_init, event_spec = build_target_log_prob_flat(
            small_model,
            observed=None,
        )
        # Both fields are scalar Normals: vector_size == 2.
        assert flat_init.shape == (event_spec.spec.vector_size,) == (2,)

    def test_the_declared_component_order_is_preserved(self, small_model):
        _, _, event_spec = build_target_log_prob_flat(small_model, observed=None)
        # The order of the joint's factors.
        assert tuple(event_spec.components) == ("a", "b")

    def test_bare_distribution_falls_through_unwrapped(self):
        """A target with no Record-shaped prior round-trips its log-prob unchanged.

        For a bare ``SupportsLogProb`` whose ``_unnormalized_log_prob``
        already takes a flat array, ``build_target_log_prob_flat``
        passes the callable through verbatim and returns no declaration.
        This is the path BlackJAX MCMC uses
        for hand-rolled distributions that don't carry a Record-shaped
        prior.
        """

        class _FlatGaussian:
            event_shape = (2,)

            def _unnormalized_log_prob(self, x):
                return -0.5 * jnp.sum(jnp.asarray(x) ** 2)

        target_flat, flat_init, event_spec = build_target_log_prob_flat(
            _FlatGaussian(),
            observed=None,
        )
        assert event_spec is None
        assert flat_init.shape == (2,)
        np.testing.assert_allclose(
            float(target_flat(jnp.asarray([1.0, -1.0]))),
            -1.0,
        )


class _FlatTarget(Distribution):
    """A law over a flat array that has no flat-vector view of its own."""

    def __init__(self):
        super().__init__("target", NumericArraySpec((2,)))

    def _log_prob(self, value):
        return -0.5 * jnp.sum(jnp.asarray(value) ** 2)

    def _unnormalized_log_prob(self, value):
        return self._log_prob(value)


class TestExtractEventSpec:
    def test_a_target_over_a_record_gives_its_declaration(self, small_model):
        assert extract_event_spec(small_model) == small_model.event_spec

    def test_a_target_with_no_flat_view_gives_none(self):
        assert extract_event_spec(_FlatTarget()) is None

    @pytest.mark.parametrize("method", ["blackjax_nuts", "blackjax_rwmh", "tfp_nuts"])
    def test_the_methods_agree_on_a_bare_target(self, method):
        """Each names the posterior and draws it as build_target_log_prob_flat does."""
        posterior = inference_method_registry.execute(
            _FlatTarget(), method=method, num_results=20, num_warmup=20, num_chains=1, random_seed=0
        )
        assert list(posterior.event_spec.components) == ["posterior"]
        assert isinstance(flat_draws(posterior), jax.Array)


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
            _FakeSampler(),
            init,
            num_results,
            jax.random.PRNGKey(0),
        )
        # Positions: (num_results, *event_shape).
        assert positions.shape == (num_results, *event_shape)
        # Infos: pytree stacked along axis 0, length num_results.
        assert infos.accept.shape == (num_results,)

    def test_positions_track_the_sampler_recurrence(self):
        # Each step adds one, starting from zeros, so row i == i + 1.
        init = _FakeState(jnp.zeros((2,)))
        positions, _ = run_chain_scan(
            _FakeSampler(),
            init,
            4,
            jax.random.PRNGKey(0),
        )
        expected = np.arange(1, 5)[:, None] * np.ones((1, 2))
        np.testing.assert_allclose(np.asarray(positions), expected)


class TestIsJaxTraceable:
    """``is_jax_traceable`` probes whether a fn traces at an init state."""

    def test_traceable_fn(self):
        init = jnp.array([1.0, 2.0])
        assert is_jax_traceable(lambda x: jnp.sum(x**2), init) is True

    def test_non_traceable_fn(self):
        # ``float(np.asarray(x))`` forces concretisation of a traced value,
        # which raises during ``make_jaxpr``.
        def host_side(x):
            return jnp.asarray(float(np.asarray(x).sum()))

        init = jnp.array([1.0, 2.0])
        assert is_jax_traceable(host_side, init) is False


class TestModelFactors:
    """The prior and likelihood of a joint at observed fields, and the flat log-likelihood."""

    @pytest.fixture
    def gaussian_target(self):
        prior = FactoredDistribution("prior", [Normal(loc=0.0, scale=1.0, label="mu")])
        return observed_target(
            _gaussian_mean(prior, 3, scale=2.0), {"y": jnp.array([1.0, -1.0, 0.5])}
        )

    def test_the_factors_split_at_the_observed_fields(self, gaussian_target):
        factors = model_factors(gaussian_target)
        assert list(factors.prior.event_spec.components) == ["mu"]
        assert list(factors.likelihood.event_spec.components) == ["y"]
        np.testing.assert_allclose(factors.observed, [1.0, -1.0, 0.5])

    def test_a_target_that_is_no_conditioned_joint_has_no_factors(self):
        assert model_factors(Normal(loc=0.0, scale=1.0, label="x")) is None

    def test_returns_scalar_log_likelihood(self, gaussian_target):
        llf = likelihood_flat(model_factors(gaussian_target))
        assert jnp.ndim(llf(jnp.array([0.3]))) == 0

    def test_matches_independent_gaussian(self, gaussian_target):
        llf = likelihood_flat(model_factors(gaussian_target))
        data = np.array([1.0, -1.0, 0.5])
        mu, scale = 0.3, 2.0
        expected = np.sum(
            -0.5 * ((data - mu) / scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi)
        )
        np.testing.assert_allclose(float(llf(jnp.array([mu]))), expected, rtol=0, atol=1e-5)


Y = jnp.array([1.0, 2.0, 0.5, 1.5, 2.5])


def _shifted(**given_spec):
    """``y_i ~ N(mu + shift, 2^2)`` over five observations, with ``shift`` optional at 0."""
    return conditional_distribution(
        "lik",
        lambda mu, shift=0.0: Normal("y", (mu + shift) * jnp.ones(5), 2.0),
        given_spec={"mu": NumericArraySpec(()), **given_spec},
    )


class TestModelFactorsWithOptionalSlots:
    """An optional slot of the likelihood takes its default unless the prior produces it."""

    def test_an_unmet_optional_slot_is_left_to_its_default(self):
        factors = model_factors(observed_target(_shifted() * Normal("mu", 0.0, 1.0), {"y": Y}))
        # The prior is the whole-term law of mu, so its draw is the value itself.
        given = parameter_given(factors, jnp.asarray(1.0))
        assert list(given) == ["mu"]
        np.testing.assert_allclose(given["mu"], 1.0)

    def test_the_prior_meets_an_optional_slot(self):
        prior = Normal("mu", 0.0, 1.0) * Normal("shift", 0.0, 1.0)
        factors = model_factors(observed_target(_shifted() * prior, {"y": Y}))
        given = parameter_given(factors, {"mu": jnp.asarray(1.0), "shift": jnp.asarray(0.5)})
        assert set(given) == {"mu", "shift"}

    def test_a_prior_kernel_at_its_defaults_is_the_prior(self):
        hyper = conditional_distribution("mu", lambda loc=0.0: Normal("mu", loc, 1.0))
        factors = model_factors(observed_target(_shifted() * hyper, {"y": Y}))
        assert isinstance(factors.prior, Normal)

    @pytest.mark.parametrize("met", [False, True], ids=["default", "prior"])
    def test_elliptical_slice_fits_the_conjugate_posterior(self, met):
        """The posterior mean of the location ``mu + shift`` is conjugate under either prior.

        With the default the location is ``mu`` with prior variance 1; with a
        prior on ``shift`` it is ``mu + shift`` with prior variance 2. The
        observations have variance 4, so the posterior mean is
        ``(sum(y) / 4) / (1 / v + 5 / 4)`` for prior variance ``v``.
        """
        prior = (
            Normal("mu", 0.0, 1.0) * Normal("shift", 0.0, 1.0) if met else Normal("mu", 0.0, 1.0)
        )
        budget = {"num_results": 2000, "num_warmup": 200, "num_chains": 2}
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_elliptical_slice", method_options=budget
            )(_shifted() * prior, {"y": Y})
        draws = flat_draws(posterior)
        location = np.asarray(draws["mu"]) + (np.asarray(draws["shift"]) if met else 0.0)
        variance = 2.0 if met else 1.0
        expected = float(jnp.sum(Y)) / 4.0 / (1.0 / variance + 5.0 / 4.0)
        np.testing.assert_allclose(location.mean(), expected, atol=0.1)


# ---------------------------------------------------------------------------
# Stub distributions for get_init_state branch coverage
# ---------------------------------------------------------------------------


class _EventShapeOnlyDist(Distribution):
    """Distribution with ``event_shape`` but no ``_sample`` (not
    ``SupportsSampling``) — exercises the Stan ``Uniform(-2, 2)`` fallback.
    """

    def __init__(self):
        super().__init__("event_shape_only", NumericArraySpec((3,)))

    def _unnormalized_log_prob(self, value):
        return -0.5 * jnp.sum(jnp.asarray(value) ** 2)


class _NoInitHeuristicDist(Distribution):
    """Distribution with neither sampling nor ``event_shape`` — exercises
    the ``get_init_state`` raise branch.
    """

    def __init__(self):
        super().__init__("no_init_heuristic", OpaqueSpec())

    def _unnormalized_log_prob(self, value):
        return jnp.asarray(0.0)


class _MappingDrawDist(Distribution):
    """A law over the record ``(a, b)`` whose draw is a mapping keyed ``b`` first."""

    def __init__(self):
        super().__init__("mapping_draw", NumericRecordSpec(a=(), b=(2,)))

    def _sample(self, key, sample_shape=()):
        return {"b": jnp.array([3.0, 4.0]), "a": jnp.asarray(1.0)}

    def _unnormalized_log_prob(self, value):
        return jnp.asarray(0.0)


class TestGetInitState:
    """Cover every documented branch of ``get_init_state``."""

    def test_explicit_init_passthrough(self):
        # Branch 1: explicit init returned verbatim (cast to prior dtype).
        prior = Normal(loc=0.0, scale=1.0, label="x")
        out = get_init_state(prior, init=jnp.array([3.0, 4.0]))
        np.testing.assert_array_equal(np.asarray(out), np.array([3.0, 4.0]))
        # Cast to the prior dtype: a default-float Normal yields a float
        # array (not, e.g., int) regardless of the input dtype.
        assert jnp.issubdtype(out.dtype, jnp.floating)

    def test_explicit_init_casts_dtype(self):
        # An integer-valued init is cast to the prior's float dtype.
        prior = Normal(loc=0.0, scale=1.0, label="x")
        out = get_init_state(prior, init=np.array([1, 2], dtype=np.int32))
        assert jnp.issubdtype(out.dtype, jnp.floating)
        np.testing.assert_allclose(np.asarray(out), np.array([1.0, 2.0]))

    def test_prior_sample_path(self):
        # Branch 2: prior implements SupportsSampling -> draw a sample.
        prior = Normal(loc=0.0, scale=1.0, label="x")
        assert isinstance(prior, SupportsSampling)
        out = get_init_state(prior, init=None, random_seed=0)
        assert out.shape == (1,)  # scalar Normal -> length-1 vector
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_a_factored_prior_starts_at_its_own_draw(self):
        # A factored prior draws a nested mapping, which is flattened rather
        # than replaced by the Uniform(-2, 2) box.
        prior = Normal("a", 0.0, 1.0) * MultivariateNormal(
            "b", jnp.array([10.0, -10.0]), cov=jnp.eye(2)
        )
        out = get_init_state(prior, init=None, random_seed=0)
        draw = prior._sample(as_prng_key(0), sample_shape=())
        expected = jnp.concatenate([jnp.ravel(draw["a"]), jnp.ravel(draw["b"])])
        np.testing.assert_allclose(np.asarray(out), np.asarray(expected))

    @pytest.mark.parametrize("seed", range(12))
    def test_a_factored_prior_starts_inside_its_support(self, seed):
        prior = Normal("a", 0.0, 1.0) * HalfNormal("scale", 1.0)
        out = get_init_state(prior, init=None, random_seed=seed)
        assert float(out[1]) > 0.0

    def test_a_mapping_draw_flattens_in_the_order_of_the_declaration(self):
        out = get_init_state(_MappingDrawDist(), init=None, random_seed=0)
        np.testing.assert_allclose(np.asarray(out), np.array([1.0, 3.0, 4.0]))

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


# ---------------------------------------------------------------------------
# The seed of a run
# ---------------------------------------------------------------------------


_SEEDED_METHODS = ["blackjax_rwmh", "blackjax_nuts", "tfp_nuts"]


def _first_draws(method, seed, **options):
    """The first chain's draws of *method* on a Gaussian mean, in a scope seeded by *seed*."""
    model = _gaussian_mean(Normal("mu", 0.0, 1.0), 4)
    data = jnp.array([0.3, -0.2, 0.5, 0.1])
    with workflow_run(seed=seed):
        posterior = condition_on.with_options(
            method=method, method_options={"num_results": 8, "num_warmup": 4, **options}
        )(model, {"y": data})
    return np.asarray(flat_chains(posterior)[0])


class TestRunSeed:
    """A run is seeded by the workflow scope unless its options set ``random_seed``."""

    @pytest.mark.parametrize("method", _SEEDED_METHODS)
    def test_a_seeded_scope_reproduces_the_run(self, method):
        np.testing.assert_array_equal(_first_draws(method, 0), _first_draws(method, 0))

    @pytest.mark.parametrize("method", _SEEDED_METHODS)
    def test_scopes_with_different_seeds_run_different_chains(self, method):
        assert not np.array_equal(_first_draws(method, 0), _first_draws(method, 1))

    @pytest.mark.parametrize("method", _SEEDED_METHODS)
    def test_an_explicit_random_seed_wins_over_the_scope(self, method):
        np.testing.assert_array_equal(
            _first_draws(method, 0, random_seed=3), _first_draws(method, 1, random_seed=3)
        )


# ---------------------------------------------------------------------------
# parallel_chain_map
# ---------------------------------------------------------------------------


class TestParallelChainMap:
    """``parallel_chain_map`` picks pmap when enough devices, vmap otherwise.

    The pmap-when-available path matters for two reasons:

    1. **NUTS bit-identicalness.** ``jax.vmap`` of NUTS masks data-dependent
       trajectory length, so vmap'd draws don't match sequential at the same
       seed. ``jax.pmap`` runs each chain on its own device and is
       bit-identical. The pmap path is exercised in a subprocess test below.
    2. **CPU multi-device parallelism.** Users who set
       ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` get linear
       chain scaling on CPU without code changes.
    """

    def test_uses_vmap_when_single_device(self, monkeypatch):
        """With ``local_device_count() == 1``, multi-chain calls hit ``jax.vmap``.

        Patches both ``jax.pmap`` and ``jax.vmap`` so we can witness
        the dispatch directly (a runtime ``Array`` doesn't distinguish
        the two on a single device).
        """
        import jax

        from probpipe.inference import _inference_utils as iu

        if jax.local_device_count() != 1:
            import pytest as _pytest

            _pytest.skip(
                "test requires the default 1-device CPU; got "
                f"{jax.local_device_count()} — set XLA_FLAGS in your shell?"
            )

        called = {"pmap": 0, "vmap": 0}
        real_vmap = jax.vmap

        def _spy_pmap(fn):
            called["pmap"] += 1
            raise AssertionError("pmap path should not fire on single device")

        def _spy_vmap(fn):
            called["vmap"] += 1
            return real_vmap(fn)

        monkeypatch.setattr(iu.jax, "pmap", _spy_pmap)
        monkeypatch.setattr(iu.jax, "vmap", _spy_vmap)

        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        out = iu.parallel_chain_map(lambda k: jax.random.normal(k, (3,)), keys)
        assert out.shape == (4, 3)
        assert called == {"pmap": 0, "vmap": 1}

    def test_pmap_path_via_subprocess(self):
        """Subprocess with virtual devices exercises the pmap branch.

        Done as a subprocess because ``XLA_FLAGS`` is read once at JAX
        import time — setting it inside this process has no effect.
        """
        import os
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent("""
            import jax, jax.numpy as jnp
            from probpipe.inference._inference_utils import parallel_chain_map

            assert jax.local_device_count() == 4, f"expected 4 devices, got {jax.local_device_count()}"
            keys = jax.random.split(jax.random.PRNGKey(0), 4)
            out = parallel_chain_map(lambda k: jax.random.normal(k, (3,)), keys)
            assert out.shape == (4, 3), f"shape {out.shape}"

            # Bit-identical to sequential single-chain runs at the same per-key seed.
            expected = jnp.stack([jax.random.normal(k, (3,)) for k in keys])
            assert jnp.allclose(out, expected, atol=1e-6), "pmap result diverges from sequential"
            print("OK")
        """)
        env = os.environ.copy()
        env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "OK" in result.stdout


class _StubPosterior:
    def __init__(self, var_names):
        # dict preserves insertion order, mimicking a trace's data_vars.
        self.data_vars = dict.fromkeys(var_names)


class _StubTrace:
    """Minimal stand-in: ``posterior_var_order`` reads only
    ``trace.posterior.data_vars`` (ordered)."""

    def __init__(self, var_names):
        self.posterior = _StubPosterior(var_names)


class TestPosteriorVarOrder:
    """``posterior_var_order`` returns a trace's posterior variables in the
    backend's natural order filtered to *keep*, and fails loudly when a
    kept name is missing."""

    def test_preserves_trace_order_not_keep_order(self):
        trace = _StubTrace(["slope", "intercept", "lp__"])
        # The trace's data_vars order wins; the keep argument's order does not.
        assert posterior_var_order(trace, ["intercept", "slope"]) == ["slope", "intercept"]

    def test_alphabetical_backend_order_returned_for_realignment(self):
        # nutpie, e.g., sorts data_vars alphabetically; the helper returns
        # that order so field_order can realign columns by name.
        trace = _StubTrace(["a", "b", "c"])
        assert posterior_var_order(trace, ["c", "a", "b"]) == ["a", "b", "c"]

    def test_ignores_unkept_trace_vars(self):
        trace = _StubTrace(["mu", "sigma", "lp__", "diverging"])
        assert posterior_var_order(trace, ["mu", "sigma"]) == ["mu", "sigma"]

    def test_missing_expected_var_raises_valueerror(self):
        """A kept name absent from the trace raises a clear ValueError here,
        not a cryptic 'not a permutation' error later in make_posterior."""
        trace = _StubTrace(["mu"])
        with pytest.raises(ValueError, match="missing expected variable"):
            posterior_var_order(trace, ["mu", "sigma"])

    def test_error_message_names_the_missing_vars(self):
        trace = _StubTrace(["mu"])
        with pytest.raises(ValueError, match="sigma"):
            posterior_var_order(trace, ["mu", "sigma"])


# ---------------------------------------------------------------------------
# Unconstrained chain coordinates
# ---------------------------------------------------------------------------


class TestUnconstrainedChain:
    """A flat chain over a constrained leaf runs in unconstrained coordinates."""

    def test_a_law_of_the_reals_keeps_its_coordinates(self):
        law = Normal("x", 0.0, 1.0)
        density = law._unnormalized_log_prob
        same, init, constrain = unconstrained_chain(density, jnp.zeros(()), law)
        assert same is density
        assert float(init) == 0.0
        np.testing.assert_array_equal(constrain(jnp.ones((4, 1))), jnp.ones((4, 1)))

    def test_a_positive_leaf_adds_the_log_jacobian(self):
        law = Gamma("g", 2.0, 1.0)

        def log_density(theta):
            return law._log_prob(jnp.reshape(theta, ()))

        density, init, constrain = unconstrained_chain(log_density, jnp.ones(1), law)
        z = jnp.array([0.3])
        x = float(jnp.squeeze(constrain(z[None])))
        slope = jax.grad(lambda u: jnp.squeeze(constrain(jnp.reshape(u, (1, 1)))))(0.3)
        assert x > 0
        assert init.shape == (1,)
        expected = float(log_density(jnp.asarray(x))) + float(jnp.log(slope))
        np.testing.assert_allclose(float(density(z)), expected, rtol=1e-5)

    def test_a_simplex_leaf_has_one_fewer_coordinate_and_its_draws_sum_to_one(self):
        law = Dirichlet("p", jnp.ones(3))
        density, init, constrain = unconstrained_chain(law._log_prob, jnp.full(3, 1.0 / 3.0), law)
        assert init.shape == (2,)
        draws = constrain(jax.random.normal(jax.random.PRNGKey(0), (5, 2)))
        assert draws.shape == (5, 3)
        np.testing.assert_allclose(jnp.sum(draws, axis=-1), 1.0, rtol=1e-5)
        assert bool(jnp.all(jnp.isfinite(jax.vmap(density)(jnp.zeros((2, 2))))))

    def test_an_initial_state_outside_the_support_starts_at_the_center(self):
        law = Gamma("g", 2.0, 1.0)
        _, init, constrain = unconstrained_chain(lambda theta: 0.0, jnp.array([-1.0]), law)
        assert bool(jnp.all(jnp.isfinite(init)))
        assert float(jnp.squeeze(constrain(init[None]))) > 0
