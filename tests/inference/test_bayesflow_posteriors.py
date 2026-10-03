"""Tests for the BayesFlow amortized-SBI backend (NPE / FMPE / CMPE).

Requires the ``[bayesflow]`` extra (Python 3.12-3.13); skipped otherwise. Uses a
small, fast-training toy fixture so the suite stays tractable.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
pytest.importorskip("bayesflow")

import jax
import jax.numpy as jnp
import numpy as np

import probpipe as pp
from probpipe import (
    EmpiricalDistribution,
    Normal,
    NumericRecord,
    NumericRecordBatch,
    condition_on,
    learn_amortized_posterior,
    log_prob,
    sample,
    workflow_run,
)
from probpipe.core._dispatch import ResolutionError
from probpipe.core._specs import NumericArraySpec, OutputSpec
from probpipe.distributions import ConditionalDistribution
from probpipe.distributions._capabilities import (
    SupportsApproximateConditioning,
    SupportsConditionalLogProb,
    SupportsConditionalSampling,
    SupportsLogProb,
    SupportsSampling,
    _is_normalized,
)
from probpipe.inference._bayesflow_posteriors import _AmortizedPosterior
from probpipe.operations._condition import condition_on as condition_on_operation
from tests._posterior import law_draws, method_of

from ._bayesflow_helpers import SimulatorKernel, theta_vec

pytestmark = pytest.mark.bayesflow


def _toy(params, key):
    """Identifiable 2-parameter model: ``y = [a + b, a - b] + small noise``."""
    t = theta_vec(params)  # structured record (training) or raw array (direct)
    a, b = t[0], t[1]
    return jnp.stack([a + b, a - b]) + 0.1 * jax.random.normal(key, (2,))


def _toy_simulator():
    return SimulatorKernel(_prior(), (2,), _toy)


class _UniformBelowKernel(ConditionalDistribution, SupportsConditionalSampling):
    """``x | z ~ Uniform(0, z)``, whose support depends on ``z`` and so is not declared."""

    def __init__(self):
        spec = NumericArraySpec((), "float32")
        super().__init__("x", {"z": spec}, OutputSpec(x=spec))

    def _condition_on(self, given, /, **options):
        return pp.Uniform("x", 0.0, given["z"])

    def _conditional_sample(self, given, key, sample_shape=()):
        return pp.Uniform("x", 0.0, given["z"])._sample(key, sample_shape)


def _prior():
    return Normal(loc=0.0, scale=1.0, label="a") * Normal(loc=0.0, scale=1.0, label="b")


def _observe(a, b, seed):
    return _toy(jnp.array([a, b]), jax.random.PRNGKey(seed))


def _vec(params, key):
    """Vector + scalar params [m0, m1, s]; y = [m0 + s, m1 - s] + small noise."""
    t = theta_vec(params)
    m0, m1, s = t[0], t[1], t[2]
    return jnp.stack([m0 + s, m1 - s]) + 0.1 * jax.random.normal(key, (2,))


def _vec_prior():
    return pp.MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="m") * Normal(
        loc=0.0, scale=1.0, label="s"
    )


def _single_field(params, key):
    """Single (vector) parameter field: ``y = theta + small noise``."""
    return theta_vec(params) + 0.1 * jax.random.normal(key, (2,))


def _non_jax(params, key):
    """A deliberately non-vmappable simulator: it concretizes the parameters
    (``float(...)``) and draws noise with numpy, so it runs only on the eager
    path (``sim_backend="sequential"``) -- ``jax.vmap`` would raise a tracer error."""
    t = np.asarray(theta_vec(params))
    a, b = float(t[0]), float(t[1])
    seed = int(jax.random.randint(key, (), 0, 2**16))
    noise = np.random.default_rng(seed).standard_normal(2)
    return np.array([a + b, a - b]) + 0.1 * noise


def _scalar(params, key):
    """One-parameter model: ``y = a + small noise`` (a single scalar param)."""
    return theta_vec(params)[:1] + 0.1 * jax.random.normal(key, (1,))


def _multi_field(params, key):
    """Three-field params (flat ``[a, b0, b1, c]``) -> an 8-d observation built
    from several param combinations, to exercise multiple mixed-shape parameter
    fields and higher-dimensional data."""
    t = theta_vec(params)
    a, b0, b1, c = t[0], t[1], t[2], t[3]
    mean = jnp.stack([a + b0, a - b0, b1 + c, b1 - c, a + c, b0 * b1, a, c])
    return mean + 0.1 * jax.random.normal(key, mean.shape)


def _multi_field_prior():
    return (
        Normal(loc=0.0, scale=1.0, label="a")
        * pp.MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="b")
        * Normal(loc=0.0, scale=1.0, label="c")
    )


# Conjugate Gaussian with an analytic posterior, used to check calibration.
_CONJ_SIGMA = 0.5


def _conjugate(params, key):
    """Conjugate model (any dimension): prior ``theta ~ N(0, I)``, ``y = theta +
    sigma * noise``. The posterior is analytic -- ``N(y / (1 + sigma^2), sigma^2 /
    (1 + sigma^2) I)`` -- so the amortized posterior's mean *and* spread can be
    checked against it. The observation dimension follows the parameter dimension,
    so one simulator serves both the 2-D and 1-D calibration tests."""
    t = theta_vec(params)
    return t + _CONJ_SIGMA * jax.random.normal(key, t.shape)


def _conjugate_simulator(prior):
    return SimulatorKernel(prior, (prior.event_spec.spec.vector_size,), _conjugate)


def _named_field(params, key):
    """Accesses params strictly by field name (``params["a"]``), never positionally
    -- locks the contract that training passes the simulator the prior's
    structured per-draw record (a flattened-vector regression raises here)."""
    a, b = params["a"], params["b"]
    return jnp.stack([a + b, a - b]) + 0.1 * jax.random.normal(key, (2,))


def _positive(params, key):
    """Simulator for a constrained prior (positive ``r``, real ``m``):
    ``y = [r + m, r - m] + small noise``."""
    t = theta_vec(params)
    r, m = t[0], t[1]
    return jnp.stack([r + m, r - m]) + 0.1 * jax.random.normal(key, (2,))


def _nested_prior():
    """Nested joint: a sub-record ``outer`` (a
    positive leaf ``r`` and a real leaf ``m``) plus a top-level real ``c`` --
    leaves ``outer/r``, ``outer/m``, ``c``. The ``Gamma`` leaf exercises a
    per-leaf bijector *under* nesting; ``flatten`` order is ``[r, m, c]``."""
    outer = (pp.Gamma("r", 3.0, 1.0) * Normal(loc=0.0, scale=1.0, label="m")).with_path_names(
        {"r": "outer/r", "m": "outer/m"}
    )
    return (outer * Normal(loc=0.0, scale=1.0, label="c")).with_label("joint")


def _nested(params, key):
    """Nested params (``outer={r, m}``, ``c``) -> ``y = [r + c, m - c, r - m]`` +
    small noise. Reads the per-draw record by *leaf path* (``params["outer/r"]``)
    -- the leaf-keyed access the redesigned Record requires -- locking the
    structured-record contract under nesting; a flattened-vector regression
    would raise here."""
    r, m, c = params["outer/r"], params["outer/m"], params["c"]
    return jnp.stack([r + c, m - c, r - m]) + 0.1 * jax.random.normal(key, (3,))


def _nested_observe(r, m, c, seed):
    """Observe ``_NestedLikelihood`` at a given (r, m, c) by building the nested
    per-draw record via ``from_vector`` (leaf order ``[r, m, c]``) -- the same
    structured object the offline simulator passes the simulator at train time."""
    rec = NumericRecord.from_vector("nr", _nested_prior().event_spec.spec, jnp.array([r, m, c]))
    return _nested(rec, jax.random.PRNGKey(seed))


@pytest.fixture(scope="module")
def npe_model():
    """A briefly-trained NPE estimator, shared across the NPE tests."""
    return learn_amortized_posterior(
        _prior(),
        _toy_simulator(),
        method="npe",
        num_simulations=3000,
        epochs=6,
        batch_size=256,
        random_seed=0,
        verbose=0,
    )


class TestBayesFlowNPE:
    def test_recovery(self, npe_model):
        """The amortized posterior concentrates near the truth -- in *both*
        parameters. Both truths sit >0.5 from the prior mean (0), so neither
        assertion passes without the model actually learning the parameter."""
        post = condition_on(npe_model, {"observation": _observe(0.6, -0.6, seed=7)})
        draws = law_draws(post)
        a = float(np.mean(np.asarray(draws["a"])))
        b = float(np.mean(np.asarray(draws["b"])))
        # Loose, calibration-style tolerance (brief training, stochastic).
        assert abs(a - 0.6) < 0.5
        assert abs(b - (-0.6)) < 0.5

    def test_amortization(self, npe_model):
        """The same trained model conditions on distinct observations, and the
        posterior means land on the correct side of zero (the defining property
        of amortized inference)."""
        mean_a_hi = float(
            np.mean(
                np.asarray(
                    law_draws(condition_on(npe_model, {"observation": _observe(1.0, 0.0, 2)}))["a"]
                )
            )
        )
        mean_a_lo = float(
            np.mean(
                np.asarray(
                    law_draws(condition_on(npe_model, {"observation": _observe(-1.0, 0.0, 3)}))["a"]
                )
            )
        )
        assert mean_a_hi > 0 > mean_a_lo

    def test_contract(self, npe_model):
        """``condition_on`` returns the learned law at the observation, which samples
        the network and whose annotations name the method."""
        post = condition_on(npe_model, {"observation": _observe(0.0, 0.0, 1)})
        assert isinstance(post, SupportsSampling)
        assert not isinstance(post, EmpiricalDistribution)
        assert method_of(post) == "bayesflow_npe"
        draws = law_draws(post, 300)
        # Fields named by the prior's declaration, one row per draw.
        assert np.asarray(draws["a"]).reshape(-1).shape[0] == 300
        assert np.isfinite(np.asarray(draws["b"])).all()

    def test_a_coupling_flow_posterior_has_the_flow_density(self, npe_model):
        """The kernel and its law claim the density, which equals the approximator's
        own ``log_prob`` for a batch of values and for each value alone."""
        observation = np.asarray(_observe(0.6, -0.6, seed=7), dtype="float32")
        law = condition_on(npe_model, {"observation": observation})
        assert isinstance(npe_model, SupportsConditionalLogProb)
        assert isinstance(law, SupportsLogProb)
        with workflow_run(seed=0):
            draws = sample(law, sample_shape=(5,))
        a, b = np.asarray(draws["a"]), np.asarray(draws["b"])
        reference = npe_model._approximator.log_prob(
            data={
                "theta_0": a[:, None],
                "theta_1": b[:, None],
                "observation": np.tile(observation, (5, 1)),
            }
        )
        scores = np.asarray(log_prob(law, draws))
        np.testing.assert_allclose(scores, np.ravel(reference), rtol=1e-5, atol=1e-5)
        one = float(np.asarray(log_prob(law, {"a": a[0], "b": b[0]})))
        np.testing.assert_allclose(one, scores[0], rtol=1e-5)
        given = {"observation": observation}
        conditional = npe_model._conditional_log_prob(given, {"a": a[0], "b": b[0]})
        np.testing.assert_allclose(float(conditional), one, rtol=1e-6)

    def test_the_density_integrates_to_one(self, npe_model):
        law = condition_on(npe_model, {"observation": _observe(0.6, -0.6, seed=7)})
        grid = np.linspace(-3.0, 3.0, 241)
        a, b = np.meshgrid(grid, grid, indexing="ij")
        points = NumericRecordBatch(
            "grid", {"a": jnp.asarray(a.ravel()), "b": jnp.asarray(b.ravel())}, "point"
        )
        density = np.exp(np.asarray(log_prob(law, points)))
        assert density.sum() * (grid[1] - grid[0]) ** 2 == pytest.approx(1.0, abs=0.01)

    def test_an_mcmc_option_is_refused(self, npe_model):
        """Conditioning the posterior takes no method options and refuses them."""
        with pytest.raises(TypeError, match=r"\['num_chains', 'num_warmup'\].*takes none"):
            condition_on.with_options(method_options={"num_warmup": 99, "num_chains": 4})(
                npe_model, {"observation": _observe(0.0, 0.0, 1)}
            )

    def test_observation_dim_mismatch(self, npe_model):
        """Conditioning on wrong-size observed data raises a clear error rather
        than an opaque keras shape failure."""
        with pytest.raises(ValueError, match="conditioning shape is fixed"):
            condition_on(npe_model, {"observation": np.zeros(5, dtype="float32")})

    def test_the_model_claims_no_direct_sampling(self, npe_model):
        """The amortized posterior is a kernel, so it claims SupportsSampling only
        at an observation; its prior and simulator properties are the joint it
        was trained on."""
        assert not isinstance(npe_model, SupportsSampling)
        assert tuple(npe_model.prior.event_spec.components) == ("a", "b")
        assert isinstance(npe_model.simulator, SimulatorKernel)

    def test_it_is_a_kernel_from_the_observation_to_the_parameters(self, npe_model):
        assert isinstance(npe_model, ConditionalDistribution)
        assert isinstance(npe_model, SupportsApproximateConditioning)
        assert isinstance(npe_model, SupportsConditionalSampling)
        assert list(npe_model.given_spec) == ["observation"]
        assert npe_model.event_spec == npe_model.prior.event_spec

    def test_conditioning_evaluates_it_with_no_inference(self, npe_model):
        """Conditioning on an observation curries the kernel, which is approximate,
        and returns a law that samples, so no inference method runs."""
        observation = {"observation": _observe(0.5, 0.0, 0)}
        report = condition_on_operation.check(npe_model, observation)
        assert (report.route, report.method, report.exact) == ("curry", None, False)
        law = condition_on_operation(npe_model, observation)
        assert isinstance(law, SupportsSampling)
        assert _is_normalized(law)
        assert tuple(law.event_spec.components) == ("a", "b")

    def test_a_parameter_given_with_the_observation_is_conditioned_by_bayes_rule(self, npe_model):
        """Conditioning a parameter is Bayes' rule on the learned law, which inference
        normalizes through the coupling flow's density, so the result is the law of
        the other parameter."""
        given = {"observation": _observe(0.5, 0.0, 0), "a": 0.5}
        budgets = {"num_warmup": 200, "num_results": 400}
        with workflow_run(seed=0):
            law = condition_on_operation.with_options(method_options=budgets)(npe_model, given)
        assert isinstance(law, EmpiricalDistribution)
        assert tuple(law.event_spec.components) == ("b",)
        assert abs(float(np.asarray(pp.mean(law)["mean(b)"]).ravel()[0])) < 0.5

    def test_without_a_density_a_given_parameter_is_not_left_free(self, npe_model):
        """A posterior whose network gives no density has no route for Bayes' rule, so
        the call raises rather than drop the parameter."""
        bare = _AmortizedPosterior(
            npe_model._approximator,
            npe_model.prior,
            npe_model.simulator,
            method="npe",
            data_dim=2,
            bijectors=npe_model._bijectors,
        )
        given = {"observation": _observe(0.5, 0.0, 0), "a": 0.5}
        report = condition_on_operation.check(bare, given)
        assert report.route != "approximate_conditioning"
        with pytest.raises(ResolutionError):
            condition_on_operation(bare, given)

    @pytest.mark.parametrize("keys", [("observation", "typo"), ("obsrevation",)])
    def test_a_key_that_names_no_field_raises(self, npe_model, keys):
        given = dict.fromkeys(keys, _observe(0.5, 0.0, 0))
        with pytest.raises(ResolutionError):
            condition_on_operation(npe_model, given)

    def test_its_conditioning_reads_the_observation_slot_alone(self, npe_model):
        with pytest.raises(KeyError, match="typo"):
            npe_model._condition_on({"observation": _observe(0.5, 0.0, 0), "typo": 1.0})

    def test_exact_only_refuses_it(self, npe_model):
        view = condition_on_operation.with_options(exact_only=True)
        with pytest.raises(ResolutionError, match="SupportsApproximateConditioning"):
            view(npe_model, {"observation": _observe(0.5, 0.0, 0)})

    def test_the_draws_follow_the_workflow_seed(self, npe_model):
        law = condition_on(npe_model, {"observation": _observe(0.3, 0.1, 4)})
        with workflow_run(seed=11):
            first = np.asarray(law_draws(law, 300)["a"]).reshape(-1)
        with workflow_run(seed=11):
            second = np.asarray(law_draws(law, 300)["a"]).reshape(-1)
        with workflow_run(seed=12):
            third = np.asarray(law_draws(law, 300)["a"]).reshape(-1)
        assert first.shape == (300,)
        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, third)

    def test_a_draw_at_an_observation_has_the_parameters_kind(self, npe_model):
        given = {"observation": _observe(0.5, 0.0, 0)}
        draws = npe_model._conditional_sample(given, jax.random.PRNGKey(0), (3,))
        assert {name: np.shape(draws[name]) for name in ("a", "b")} == {"a": (3,), "b": (3,)}

    def test_provenance_names_the_joint_it_was_trained_on(self, npe_model):
        record = npe_model.provenance
        parents = [parent.label for parent in record.parents]
        assert npe_model.prior.label in parents
        assert npe_model.simulator.label in parents

    def test_repr(self, npe_model):
        """``repr`` names the method the posterior was learned with."""
        r = repr(npe_model)
        assert "method='npe'" in r


class TestBayesFlowMethods:
    @pytest.mark.parametrize("method", ["npe", "fmpe", "cmpe"])
    def test_methods_smoke(self, method):
        """Each amortized method trains and conditions, returning named draws."""
        model = learn_amortized_posterior(
            _prior(),
            _toy_simulator(),
            method=method,
            num_simulations=1500,
            epochs=3,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        post = condition_on(model, {"observation": _observe(0.5, 0.0, 0)})
        assert method_of(post) == f"bayesflow_{method}"
        # Only NPE's coupling flow computes the learned law's density.
        assert isinstance(model, SupportsConditionalLogProb) == (method == "npe")
        assert isinstance(post, SupportsLogProb) == (method == "npe")
        draws = law_draws(post, 200)
        assert np.isfinite(np.asarray(draws["a"])).all()
        assert np.asarray(draws["a"]).reshape(-1).shape[0] == 200

    def test_vector_valued_field(self):
        """A vector-valued parameter field round-trips through the per-field
        reshape/concatenate and returns named draws of the right shape."""
        model = learn_amortized_posterior(
            _vec_prior(),
            SimulatorKernel(_vec_prior(), (2,), _vec),
            method="npe",
            num_simulations=2000,
            epochs=4,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        obs = _vec(jnp.array([0.5, -0.5, 0.2]), jax.random.PRNGKey(5))
        draws = law_draws(condition_on(model, {"observation": obs}), 200)
        m = np.asarray(draws["m"]).reshape(200, -1)
        s = np.asarray(draws["s"]).reshape(200, -1)
        assert m.shape == (200, 2)  # the (2,)-vector field is preserved
        assert s.shape == (200, 1)
        assert np.isfinite(m).all() and np.isfinite(s).all()

    def test_custom_inference_network(self):
        """A caller-supplied ``inference_network`` overrides the method default
        and is the network actually wired into the trained approximator."""
        import bayesflow as bf

        net = bf.networks.CouplingFlow()
        model = learn_amortized_posterior(
            _prior(),
            _toy_simulator(),
            method="npe",
            inference_network=net,
            num_simulations=1500,
            epochs=3,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        # The exact instance passed in is the one used (not a method default).
        assert model._approximator.inference_network is net
        assert isinstance(model, SupportsConditionalLogProb)
        post = condition_on(model, {"observation": _observe(0.5, 0.0, 0)})
        assert np.asarray(law_draws(post, 200)["a"]).reshape(-1).shape[0] == 200

    def test_single_field_prior(self):
        """A single-field prior (not a factored joint) is supported: its
        draws are not field-indexable, but the canonical flat layout drives the
        per-field split, so it round-trips end-to-end to named draws."""

        prior = pp.MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="theta")
        model = learn_amortized_posterior(
            prior,
            SimulatorKernel(prior, (2,), _single_field),
            method="npe",
            num_simulations=1500,
            epochs=3,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        obs = _single_field(jnp.array([0.5, -0.5]), jax.random.PRNGKey(4))
        draws = law_draws(condition_on(model, {"observation": obs}), 200)
        assert np.asarray(draws["theta"]).reshape(200, -1).shape == (200, 2)
        assert np.isfinite(np.asarray(draws["theta"])).all()

    def test_non_jax_simulator(self):
        """A non-vmappable (non-JAX) simulator trains via the eager path
        (``sim_backend="sequential"``) and conditions to named draws -- ``vmap``
        would fail on it, so success proves the eager loop ran."""
        model = learn_amortized_posterior(
            _prior(),
            SimulatorKernel(_prior(), (2,), _non_jax),
            method="npe",
            sim_backend="sequential",
            num_simulations=800,
            epochs=2,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        post = condition_on(model, {"observation": _observe(0.5, 0.0, 0)})
        assert method_of(post) == "bayesflow_npe"
        draws = law_draws(post, 200)
        assert np.asarray(draws["a"]).reshape(-1).shape[0] == 200
        assert np.isfinite(np.asarray(draws["a"])).all()

    def test_scalar_prior_fmpe(self):
        """A one-parameter (scalar) prior round-trips with FMPE: the single-field
        split handles ``event_shape=()``, and flow matching (unlike NPE's
        coupling flow) has no >= 2-parameter requirement."""
        model = learn_amortized_posterior(
            Normal(loc=0.0, scale=1.0, label="a"),
            SimulatorKernel(Normal(loc=0.0, scale=1.0, label="a"), (1,), _scalar),
            method="fmpe",
            num_simulations=1500,
            epochs=3,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        obs = _scalar(jnp.array([0.7]), jax.random.PRNGKey(4))
        draws = law_draws(condition_on(model, {"observation": obs}), 200)
        assert np.asarray(draws["a"]).reshape(-1).shape[0] == 200
        assert np.isfinite(np.asarray(draws["a"])).all()

    def test_npe_one_param_fallback_calibration(self):
        """The NPE one-parameter fallback is held to the same calibration standard
        as the multi-parameter path. A coupling flow can't split a 1-D vector, so
        NPE at d=1 falls back to a flow-matching network; here, against a 1-D
        conjugate Gaussian (analytic posterior), the amortized posterior mean and
        spread are checked against the analytic mean and std. The bounds are looser
        than ``test_calibration_against_conjugate_gaussian`` because flow matching
        at d=1 is a measurably noisier estimator -- its ODE sampling is less exact
        than a coupling flow's density (across training seeds the mean error spans
        ~0.04-0.19 posterior-std and the std ratio ~0.92-1.18). Training is seeded,
        so a given run is reproducible; the band absorbs that estimator imprecision
        plus cross-platform / library-version drift."""
        import bayesflow as bf

        prior = Normal(loc=0.0, scale=1.0, label="a")  # event_size 1 -> FlowMatching
        model = learn_amortized_posterior(
            prior,
            _conjugate_simulator(prior),
            method="npe",
            num_simulations=5000,
            epochs=40,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        assert isinstance(model._approximator.inference_network, bf.networks.FlowMatching)
        assert not isinstance(model, SupportsConditionalLogProb)
        s2 = _CONJ_SIGMA**2
        post_std = (s2 / (1 + s2)) ** 0.5  # analytic posterior std
        mean_errs, std_ratios = [], []
        for i in range(6):
            theta = jax.random.normal(jax.random.PRNGKey(100 + i), (1,))
            obs = _conjugate(theta, jax.random.PRNGKey(900 + i))
            x = np.asarray(law_draws(condition_on(model, {"observation": obs}))["a"]).reshape(-1)
            mean_errs.append(abs(float(x.mean()) - float(obs[0]) / (1 + s2)))
            std_ratios.append(float(x.std()) / post_std)
        # Estimate: mean posterior-mean error under 0.5 posterior-std.
        assert np.mean(mean_errs) < 0.5 * post_std
        # Uncertainty: mean std ratio in [0.7, 1.4] (flow matching at d=1 tends to
        # slightly under-disperse; band bounds the measured cross-seed spread).
        assert 0.7 < np.mean(std_ratios) < 1.4

    def test_multi_field_prior_and_data(self):
        """A richer scenario: three parameter fields (scalar + 2-vector + scalar)
        and a higher-dimensional (8-d) observation. Exercises the per-field
        split, adapter routing, and posterior assembly with multiple mixed-shape
        fields and bigger data, and checks the posterior responds to the data."""
        model = learn_amortized_posterior(
            _multi_field_prior(),
            SimulatorKernel(_multi_field_prior(), (8,), _multi_field),
            method="npe",
            num_simulations=2500,
            epochs=5,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        obs = _multi_field(jnp.array([0.5, -0.5, 0.3, -0.2]), jax.random.PRNGKey(6))
        assert obs.shape == (8,)  # higher-dimensional observation
        draws = law_draws(condition_on(model, {"observation": obs}), 200)
        assert np.asarray(draws["a"]).reshape(200, -1).shape == (200, 1)
        assert np.asarray(draws["b"]).reshape(200, -1).shape == (200, 2)  # 2-vector field
        assert np.asarray(draws["c"]).reshape(200, -1).shape == (200, 1)
        assert all(np.isfinite(np.asarray(draws[f])).all() for f in ("a", "b", "c"))
        # Amortized response: the posterior mean of `a` tracks the observation.
        obs_hi = _multi_field(jnp.array([1.0, 0.0, 0.0, 0.0]), jax.random.PRNGKey(1))
        obs_lo = _multi_field(jnp.array([-1.0, 0.0, 0.0, 0.0]), jax.random.PRNGKey(2))
        mean_a_hi = float(
            np.mean(np.asarray(law_draws(condition_on(model, {"observation": obs_hi}))["a"]))
        )
        mean_a_lo = float(
            np.mean(np.asarray(law_draws(condition_on(model, {"observation": obs_lo}))["a"]))
        )
        assert mean_a_hi > mean_a_lo

    def test_calibration_against_conjugate_gaussian(self):
        """Estimate *and* uncertainty are roughly correct: against a conjugate
        Gaussian whose posterior is analytic, the amortized posterior mean tracks
        the analytic mean and its std matches the analytic std (averaged over
        several observations). Trains a bit longer than the smoke tests so the
        estimator is near-converged."""
        prior = Normal(loc=0.0, scale=1.0, label="a") * Normal(loc=0.0, scale=1.0, label="b")
        model = learn_amortized_posterior(
            prior,
            _conjugate_simulator(prior),
            method="npe",
            num_simulations=5000,
            epochs=40,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        s2 = _CONJ_SIGMA**2
        post_std = (s2 / (1 + s2)) ** 0.5  # analytic posterior std
        mean_errs, std_ratios = [], []
        for i in range(6):
            theta = jax.random.normal(jax.random.PRNGKey(100 + i), (2,))
            obs = _conjugate(theta, jax.random.PRNGKey(900 + i))
            draws = law_draws(condition_on(model, {"observation": obs}))
            for j, f in enumerate(("a", "b")):
                x = np.asarray(draws[f]).reshape(-1)
                analytic_mean = float(obs[j]) / (1 + s2)
                mean_errs.append(abs(float(x.mean()) - analytic_mean))
                std_ratios.append(float(x.std()) / post_std)
        # Training is seeded (reproducible); the margins absorb cross-platform /
        # library-version numerical drift. Estimate: mean posterior-mean error under
        # 0.3 posterior-std (observed ~0.03-0.11 across training seeds).
        assert np.mean(mean_errs) < 0.3 * post_std
        # Uncertainty: mean std ratio in [0.8, 1.25] (observed ~1.01-1.03 across seeds).
        assert 0.8 < np.mean(std_ratios) < 1.25

    def test_nested_prior_end_to_end(self):
        """A nested prior trains and conditions end to end. The
        simulator receives the structured *nested* record (read by nested name),
        posterior draws come back under the same nested leaf names, and the
        constrained leaf ``outer/r`` is mapped back through its per-leaf bijector
        so every draw lands in the positive support. NPE no longer rejects nested
        priors -- it lifts them via per-leaf bijectors and adapter keying."""
        model = learn_amortized_posterior(
            _nested_prior(),
            SimulatorKernel(_nested_prior(), (3,), _nested),
            method="npe",
            num_simulations=3000,
            epochs=8,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        draws = law_draws(
            condition_on(model, {"observation": _nested_observe(2.0, -0.5, 0.4, seed=7)}), 400
        )
        r = np.asarray(draws["outer/r"]).reshape(-1)
        m = np.asarray(draws["outer/m"]).reshape(-1)
        c = np.asarray(draws["c"]).reshape(-1)
        assert r.shape == (400,) and m.shape == (400,) and c.shape == (400,)
        assert np.isfinite(r).all() and np.isfinite(m).all() and np.isfinite(c).all()
        assert (r > 0).all()  # per-leaf forward bijector (Exp) under nesting
        # Amortized response: the posterior mean of `c` tracks the observation.
        mean_c_hi = float(
            np.mean(
                np.asarray(
                    law_draws(
                        condition_on(model, {"observation": _nested_observe(2.0, 0.0, 1.0, 2)})
                    )["c"]
                )
            )
        )
        mean_c_lo = float(
            np.mean(
                np.asarray(
                    law_draws(
                        condition_on(model, {"observation": _nested_observe(2.0, 0.0, -1.0, 3)})
                    )["c"]
                )
            )
        )
        assert mean_c_hi > mean_c_lo

    def test_nested_prior_calibration_against_conjugate(self):
        """Decisive correctness check for the nested lift: against a
        conjugate Gaussian with an analytic posterior, each *nested* leaf's
        posterior mean and spread match the analytic values. The leaves round-trip
        in flatten order (``outer/a``, ``outer/b``, ``m``); a mis-ordered column or
        a mis-keyed per-leaf bijector would land a leaf's mass on the wrong
        coordinate and fail here."""
        outer = (
            Normal(loc=0.0, scale=1.0, label="a") * Normal(loc=0.0, scale=1.0, label="b")
        ).with_path_names({"a": "outer/a", "b": "outer/b"})
        prior = (outer * Normal(loc=0.0, scale=1.0, label="m")).with_label("joint")
        model = learn_amortized_posterior(
            prior,
            _conjugate_simulator(prior),
            method="npe",
            num_simulations=5000,
            epochs=40,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        s2 = _CONJ_SIGMA**2
        post_std = (s2 / (1 + s2)) ** 0.5
        leaves = ("outer/a", "outer/b", "m")
        mean_errs, std_ratios = [], []
        for i in range(6):
            theta = jax.random.normal(jax.random.PRNGKey(100 + i), (3,))
            obs = _conjugate(theta, jax.random.PRNGKey(900 + i))
            draws = law_draws(condition_on(model, {"observation": obs}))
            for j, leaf in enumerate(leaves):
                x = np.asarray(draws[leaf]).reshape(-1)
                analytic_mean = float(obs[j]) / (1 + s2)
                mean_errs.append(abs(float(x.mean()) - analytic_mean))
                std_ratios.append(float(x.std()) / post_std)
        # Same margins as the flat conjugate test (seed-reproducible; absorbs drift).
        assert np.mean(mean_errs) < 0.3 * post_std
        assert 0.8 < np.mean(std_ratios) < 1.25

    def test_nested_wishart_matrix_leaf(self):
        """A matrix-valued constrained leaf (Wishart, positive-definite) nested
        under ``outer``: the per-leaf inverse bijector runs at the leaf's native
        (n, n) event shape under nesting (a flat layout would crash the
        CholeskyOuterProduct chain), and the forward map at sample time returns
        SPD draws under the nested leaf name ``outer/cov`` -- the nested analogue
        of test_wishart_matrix_prior_round_trip."""
        outer = (
            pp.Wishart(df=4.0, scale=jnp.eye(2), label="cov")
            * Normal(loc=0.0, scale=1.0, label="m")
        ).with_path_names({"cov": "outer/cov", "m": "outer/m"})
        prior = (outer * Normal(loc=0.0, scale=1.0, label="c")).with_label("joint")
        model = learn_amortized_posterior(
            prior,
            _conjugate_simulator(prior),
            method="fmpe",
            num_simulations=600,
            epochs=2,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        obs = jnp.array([1.5, 0.3, 0.3, 1.2, 0.5, 0.0])  # flat (cov 2x2, m, c)
        cov = np.asarray(law_draws(condition_on(model, {"observation": obs}))["outer/cov"]).reshape(
            -1, 2, 2
        )
        assert np.isfinite(cov).all()
        np.testing.assert_allclose(cov, np.swapaxes(cov, -1, -2), atol=1e-5)  # symmetric
        assert np.linalg.eigvalsh(cov).min() > 0  # positive definite

    def test_simulator_receives_named_record(self):
        """The simulator kernel receives the prior's structured per-draw sample
        (named fields) as its given values -- the simulator uses params["a"]/["b"]
        exclusively, so training succeeds only when the structured record is
        passed."""
        model = learn_amortized_posterior(
            _prior(),
            SimulatorKernel(_prior(), (2,), _named_field),
            method="npe",
            num_simulations=800,
            epochs=2,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        draws = law_draws(condition_on(model, {"observation": jnp.array([0.8, 0.2])}), 200)
        assert np.asarray(draws["a"]).reshape(-1).shape[0] == 200
        assert np.isfinite(np.asarray(draws["a"])).all()

    def test_constrained_prior_draws_respect_support(self):
        """A constrained (positive) prior field is trained in unconstrained space and
        its draws are mapped back through the forward bijector, so they land in the
        support -- here all positive. The accompanying real-valued field is unaffected."""
        prior = pp.Gamma("r", 3.0, 1.0) * Normal(loc=0.0, scale=1.0, label="m")
        model = learn_amortized_posterior(
            prior,
            SimulatorKernel(prior, (2,), _positive),
            method="npe",
            num_simulations=1500,
            epochs=5,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        r = np.asarray(
            law_draws(condition_on(model, {"observation": jnp.array([3.0, 1.0])}))["r"]
        ).reshape(-1)
        assert np.isfinite(r).all()
        assert (r > 0).all()  # forward bijector (Exp) keeps every draw in support

    def test_a_positive_leaf_density_changes_variables(self):
        """A positive leaf trains on its log, so its density is the flow's density at
        the log less the log-Jacobian of ``exp``, which is the log itself."""
        prior = pp.Gamma("r", 3.0, 1.0) * Normal(loc=0.0, scale=1.0, label="m")
        model = learn_amortized_posterior(
            prior,
            SimulatorKernel(prior, (2,), _positive),
            method="npe",
            num_simulations=1500,
            epochs=3,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        observation = np.array([3.0, 1.0], dtype="float32")
        law = condition_on(model, {"observation": observation})
        with workflow_run(seed=0):
            draws = sample(law, sample_shape=(4,))
        r, m = np.asarray(draws["r"]), np.asarray(draws["m"])
        unconstrained = model._approximator.log_prob(
            data={
                "theta_0": np.log(r)[:, None],
                "theta_1": m[:, None],
                "observation": np.tile(observation, (4, 1)),
            }
        )
        np.testing.assert_allclose(
            np.asarray(log_prob(law, draws)), np.ravel(unconstrained) - np.log(r), atol=1e-4
        )

    def test_a_simplex_leaf_density_integrates_to_one(self):
        """A 3-simplex trains on two coordinates, so its coupling flow's density, taken
        in the simplex's first two coordinates, integrates to one over the triangle."""
        prior = pp.Dirichlet("p", jnp.ones(3))
        model = learn_amortized_posterior(
            prior,
            _conjugate_simulator(prior),
            method="npe",
            num_simulations=2000,
            epochs=4,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        law = condition_on(model, {"observation": jnp.array([0.5, 0.3, 0.2])})
        assert isinstance(law, SupportsLogProb)
        cells = 400
        grid = (np.arange(cells) + 0.5) / cells
        first, second = np.meshgrid(grid, grid, indexing="ij")
        inside = first + second < 1
        points = np.stack(
            [first[inside], second[inside], 1 - first[inside] - second[inside]], axis=-1
        )
        density = np.exp(np.asarray(law._log_prob(jnp.asarray(points, dtype=jnp.float32))))
        assert density.sum() / cells**2 == pytest.approx(1.0, abs=0.02)

    def test_adapter_internal_names_avoid_collisions(self):
        """Theta fields are re-keyed away from BayesFlow adapter internals.

        Fields named like BayesFlow's own keys (``"observation"``,
        ``"inference_variables"``) train and condition, with draws returned
        under the user's names.
        """
        prior = Normal(loc=0.0, scale=1.0, label="observation") * Normal(
            loc=0.0, scale=1.0, label="inference_variables"
        )
        model = learn_amortized_posterior(
            prior,
            SimulatorKernel(prior, (2,), _toy, label="y"),
            method="npe",
            num_simulations=800,
            epochs=2,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        (slot,) = model.given_spec
        assert slot == "observation_"
        draws = law_draws(condition_on(model, {slot: jnp.array([0.5, 0.1])}), 200)
        for f in ("observation", "inference_variables"):
            x = np.asarray(draws[f]).reshape(-1)
            assert x.shape[0] == 200
            assert np.isfinite(x).all()

    def test_interval_prior_draws_respect_support(self):
        """A bounded-interval prior field (Beta, unit-interval support) rounds
        through the Sigmoid bijector: trained unconstrained, every posterior
        draw lands strictly inside (0, 1)."""
        prior = pp.Beta("q", 2.0, 2.0) * Normal(loc=0.0, scale=1.0, label="m")
        model = learn_amortized_posterior(
            prior,
            _conjugate_simulator(prior),
            method="npe",
            num_simulations=800,
            epochs=2,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        q = np.asarray(
            law_draws(condition_on(model, {"observation": jnp.array([0.5, 0.0])}))["q"]
        ).reshape(-1)
        assert np.isfinite(q).all()
        assert ((q > 0) & (q < 1)).all()  # Sigmoid forward keeps draws in (0, 1)

    def test_wishart_matrix_prior_round_trip(self):
        """A matrix-valued constrained field (Wishart, positive-definite support)
        round-trips: the inverse bijector runs at the field's native (n, n) event
        shape at train time -- a flattened input would crash the
        CholeskyOuterProduct chain -- and the forward map at sample time returns
        draws that are symmetric positive definite."""
        prior = pp.Wishart(df=4.0, scale=jnp.eye(2), label="cov") * Normal(
            loc=0.0, scale=1.0, label="m"
        )
        model = learn_amortized_posterior(
            prior,
            _conjugate_simulator(prior),
            method="fmpe",
            num_simulations=600,
            epochs=2,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        obs = jnp.array([1.5, 0.3, 0.3, 1.2, 0.5])  # flattened (cov, m) observation
        cov = np.asarray(law_draws(condition_on(model, {"observation": obs}))["cov"]).reshape(
            -1, 2, 2
        )
        assert np.isfinite(cov).all()
        np.testing.assert_allclose(cov, np.swapaxes(cov, -1, -2), atol=1e-5)  # symmetric
        assert np.linalg.eigvalsh(cov).min() > 0  # positive definite

    def test_dirichlet_simplex_prior_npe(self):
        """A single 2-simplex (Dirichlet) prior under method='npe': the coupling-flow
        guard counts *unconstrained* dimensions (one here, not the constrained
        event_size of two), so NPE falls back to flow matching instead of building a
        units=0 coupling flow; draws land on the simplex."""
        import bayesflow as bf

        model = learn_amortized_posterior(
            pp.Dirichlet("p", jnp.ones(2)),
            _conjugate_simulator(pp.Dirichlet("p", jnp.ones(2))),
            method="npe",
            num_simulations=600,
            epochs=2,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        assert isinstance(model._approximator.inference_network, bf.networks.FlowMatching)
        p = np.asarray(
            law_draws(condition_on(model, {"observation": jnp.array([0.7, 0.3])}))["p"]
        ).reshape(-1, 2)
        np.testing.assert_allclose(p.sum(axis=-1), 1.0, atol=1e-5)
        assert ((p > 0) & (p < 1)).all()

    def test_training_deterministic_for_seed(self):
        """Two same-seed trainings produce bit-identical models: keras init + fit
        are seeded via ``keras.utils.set_random_seed`` (the calibration tests'
        tolerance rationale relies on this reproducibility)."""

        def _fit():
            return learn_amortized_posterior(
                _prior(),
                _toy_simulator(),
                method="npe",
                num_simulations=400,
                epochs=1,
                batch_size=256,
                random_seed=0,
                verbose=0,
            )

        obs = _observe(0.4, -0.2, 5)
        # The draws are seeded by the workflow scope, so each call has the same one.
        with pp.workflow_run(seed=0):
            d1 = np.asarray(law_draws(condition_on(_fit(), {"observation": obs}))["a"]).reshape(-1)
        with pp.workflow_run(seed=0):
            d2 = np.asarray(law_draws(condition_on(_fit(), {"observation": obs}))["a"]).reshape(-1)
        np.testing.assert_array_equal(d1, d2)

    def test_global_rng_state_restored(self):
        """Training seeds keras via the global RNG but snapshots and restores the
        caller's NumPy / Python random state, so a call does not silently make the
        caller's unrelated random streams deterministic."""
        import random as pyrandom

        np.random.seed(123)
        pyrandom.seed(7)
        expected_np = np.random.random()
        expected_py = pyrandom.random()
        np.random.seed(123)
        pyrandom.seed(7)
        learn_amortized_posterior(
            _prior(),
            _toy_simulator(),
            method="fmpe",
            num_simulations=256,
            epochs=1,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        assert np.random.random() == expected_np
        assert pyrandom.random() == expected_py


class TestBayesFlowValidation:
    """Train-time input validation -- each raises before any simulation runs."""

    def test_rejects_non_generative_simulator(self):
        """A simulator that is not a sampling kernel is rejected with a clear TypeError."""

        class _NoGenerate:
            pass

        with pytest.raises(TypeError, match="ConditionalDistribution that samples"):
            learn_amortized_posterior(_prior(), _NoGenerate(), num_simulations=8, epochs=1)

    def test_rejects_a_prior_that_is_not_numeric(self):
        """A prior that is not a numeric distribution (here a raw array) is
        rejected with a clear TypeError."""
        with pytest.raises(TypeError, match="requires a numeric prior"):
            learn_amortized_posterior(
                jnp.zeros(2),
                _toy_simulator(),
                num_simulations=8,
                epochs=1,
            )

    def test_rejects_unknown_method(self):
        """An unsupported amortized method is rejected up front."""
        with pytest.raises(ValueError, match="Unknown amortized SBI method"):
            learn_amortized_posterior(
                _prior(),
                _toy_simulator(),
                method="bogus",
                num_simulations=8,
                epochs=1,
            )

    def test_rejects_unknown_sim_backend(self):
        """An unsupported simulation backend is rejected up front."""
        with pytest.raises(ValueError, match="Unknown sim_backend"):
            learn_amortized_posterior(
                _prior(),
                _toy_simulator(),
                sim_backend="bogus",
                num_simulations=8,
                epochs=1,
            )

    @pytest.mark.parametrize(
        "override",
        [{"num_simulations": 0}, {"batch_size": 0}, {"epochs": 0}],
    )
    def test_rejects_nonpositive_counts(self, override):
        """num_simulations / batch_size / epochs must be positive."""
        kwargs = {"num_simulations": 8, "epochs": 1, **override}
        with pytest.raises(ValueError, match="positive integer"):
            learn_amortized_posterior(_prior(), _toy_simulator(), **kwargs)

    @pytest.mark.parametrize(
        "override",
        [{"num_simulations": 100.5}, {"batch_size": "64"}, {"epochs": 2.0}, {"epochs": None}],
    )
    def test_rejects_non_integer_counts(self, override):
        """Count parameters must be integers -- floats (even integral ones),
        strings, and None are rejected with a TypeError rather than truncating
        or failing deep inside keras."""
        kwargs = {"num_simulations": 8, "epochs": 1, **override}
        with pytest.raises(TypeError, match="must be an integer"):
            learn_amortized_posterior(_prior(), _toy_simulator(), **kwargs)

    def test_rejects_discrete_prior(self):
        """A discrete prior field has no smooth bijector to R^d and is rejected up
        front with a clear error (here a Poisson count parameter)."""
        bad_prior = pp.Poisson("k", 3.0) * Normal(loc=0.0, scale=1.0, label="m")
        with pytest.raises(ValueError, match="discrete"):
            learn_amortized_posterior(bad_prior, _toy_simulator(), num_simulations=8, epochs=1)

    def test_rejects_a_prior_parameter_whose_support_is_not_declared(self):
        """A support that depends on another parameter is not declared, so no
        bijector to R^d can be chosen for it."""
        prior = _UniformBelowKernel() * pp.Exponential("z", 1.0)
        with pytest.raises(ValueError, match="'x': its support is not declared"):
            learn_amortized_posterior(prior, _toy_simulator(), num_simulations=8, epochs=1)
