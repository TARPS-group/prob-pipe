"""Tests for protocol compliance across distribution classes."""

import jax
import jax.numpy as jnp
import pytest

from probpipe import (
    Bernoulli,
    Beta,
    BootstrapDistribution,
    EmpiricalDistribution,
    Gamma,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    NumericRecordBatch,
    NumericRecordSpec,
    OpaqueBatch,
    OutputSpec,
)
from probpipe.distributions import ConditionalDistribution
from probpipe.distributions._capabilities import (
    SupportsApproximateConditioning,
    SupportsCovariance,
    SupportsExactConditioning,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
)
from probpipe.families import BijectorTransformedDistribution


class _ShiftKernel(ConditionalDistribution):
    """``y | x ~ Normal(x, 1)``, the dependent factor of a joint."""

    def __init__(self):
        super().__init__("y", {"x": NumericArraySpec(())}, OutputSpec(y=NumericArraySpec(())))

    def _condition_on(self, given, /, **options):
        return Normal("y", given["x"], 1.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normal():
    return Normal(loc=0.0, scale=1.0, label="x")


@pytest.fixture
def empirical():
    samples = jax.random.normal(jax.random.PRNGKey(0), (100, 2))
    return EmpiricalDistribution("x", samples)


@pytest.fixture
def bootstrap():
    data = jax.random.normal(jax.random.PRNGKey(1), (50,))
    return BootstrapDistribution("bootstrap", EmpiricalDistribution("y", data))


@pytest.fixture
def joint():
    return Normal("x", 0, 1) * Normal("y", 1, 2)


# ---------------------------------------------------------------------------
# SupportsSampling
# ---------------------------------------------------------------------------


class TestSupportsSampling:
    """All distributions should support sampling."""

    @pytest.mark.parametrize(
        "dist_cls,kwargs",
        [
            (Normal, {"loc": 0.0, "scale": 1.0, "label": "x"}),
            (Beta, {"alpha": 2.0, "beta": 5.0, "label": "b"}),
            (Gamma, {"concentration": 3.0, "rate": 1.0, "label": "g"}),
            (Bernoulli, {"probs": 0.5, "label": "d"}),
            (MultivariateNormal, {"loc": jnp.zeros(2), "cov": jnp.eye(2), "label": "z"}),
        ],
    )
    def test_tfp_distributions(self, dist_cls, kwargs):
        dist = dist_cls(**kwargs)
        assert isinstance(dist, SupportsSampling)

    def test_empirical(self, empirical):
        assert isinstance(empirical, SupportsSampling)

    def test_bootstrap(self, bootstrap):
        assert isinstance(bootstrap, SupportsSampling)

    def test_joint(self, joint):
        assert isinstance(joint, SupportsSampling)


# ---------------------------------------------------------------------------
# SupportsExpectation
# ---------------------------------------------------------------------------


class TestSupportsExpectation:
    def test_a_continuous_law_claims_no_exact_expectation(self, normal):
        assert not isinstance(normal, SupportsExpectation)

    def test_empirical(self, empirical):
        assert isinstance(empirical, SupportsExpectation)

    def test_a_joint_of_continuous_laws_claims_no_exact_expectation(self, joint):
        assert not isinstance(joint, SupportsExpectation)


# ---------------------------------------------------------------------------
# SupportsLogProb
# ---------------------------------------------------------------------------


class TestSupportsLogProb:
    @pytest.mark.parametrize(
        "dist_cls,kwargs",
        [
            (Normal, {"loc": 0.0, "scale": 1.0, "label": "x"}),
            (Beta, {"alpha": 2.0, "beta": 5.0, "label": "b"}),
            (MultivariateNormal, {"loc": jnp.zeros(2), "cov": jnp.eye(2), "label": "z"}),
        ],
    )
    def test_tfp_distributions(self, dist_cls, kwargs):
        dist = dist_cls(**kwargs)
        assert isinstance(dist, SupportsLogProb)

    def test_empirical_not_log_prob(self, empirical):
        assert not isinstance(empirical, SupportsLogProb)


# ---------------------------------------------------------------------------
# Protocol hierarchy
# ---------------------------------------------------------------------------


class TestProtocolHierarchy:
    """Verify that protocol inheritance relationships hold."""

    def test_sampling_and_expectation_independent(self, normal):
        """A law samples without claiming the exact expectation, which needs finite support."""
        assert isinstance(normal, SupportsSampling)
        assert not isinstance(normal, SupportsExpectation)

    def test_log_prob_implies_unnormalized(self, normal):
        """SupportsLogProb extends SupportsUnnormalizedLogProb."""
        assert isinstance(normal, SupportsLogProb)
        assert isinstance(normal, SupportsUnnormalizedLogProb)

    def test_mean_independent_of_expectation(self):
        """SupportsMean does NOT extend SupportsExpectation."""
        assert not issubclass(SupportsMean, SupportsExpectation)

    def test_variance_independent_of_expectation(self):
        """SupportsVariance does NOT extend SupportsExpectation."""
        assert not issubclass(SupportsVariance, SupportsExpectation)

    def test_covariance_independent_of_expectation(self):
        """SupportsCovariance does NOT extend SupportsExpectation."""
        assert not issubclass(SupportsCovariance, SupportsExpectation)

    def test_a_continuous_law_has_a_mean_but_no_exact_expectation(self, normal):
        """The mean is exact in closed form, while an arbitrary expectation is not."""
        assert isinstance(normal, SupportsMean)
        assert not isinstance(normal, SupportsExpectation)

    def test_sampling_independent_of_expectation(self):
        """SupportsSampling does NOT extend SupportsExpectation."""
        assert not issubclass(SupportsSampling, SupportsExpectation)

    def test_log_prob_subclass_check(self):
        assert issubclass(SupportsLogProb, SupportsUnnormalizedLogProb)


# ---------------------------------------------------------------------------
# SupportsMean / SupportsVariance / SupportsCovariance
# ---------------------------------------------------------------------------


class TestSupportsMean:
    """Only distributions with exact (non-MC) moments satisfy these."""

    def test_tfp_normal(self, normal):
        assert isinstance(normal, SupportsMean)
        assert isinstance(normal, SupportsVariance)

    def test_empirical_numeric_has_moments(self, empirical):
        """Numeric EmpiricalDistribution dispatches to Array variant with moments."""
        assert isinstance(empirical, SupportsMean)
        assert isinstance(empirical, SupportsVariance)
        assert isinstance(empirical, SupportsCovariance)

    def test_empirical_generic_no_moments(self):
        """Non-numeric EmpiricalDistribution does not support moments."""
        dist = EmpiricalDistribution("x", OpaqueBatch("labels", ["a", "b", "c"], "atom"))
        assert not isinstance(dist, SupportsMean)
        assert not isinstance(dist, SupportsVariance)
        assert not isinstance(dist, SupportsCovariance)

    def test_array_empirical(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100, 2))
        dist = EmpiricalDistribution("x", samples)
        assert isinstance(dist, SupportsMean)
        assert isinstance(dist, SupportsVariance)
        assert isinstance(dist, SupportsCovariance)

    def test_bootstrap(self, bootstrap):
        """The bootstrap measure's mean is its source; it claims no event-typed variance."""
        assert isinstance(bootstrap, SupportsMean)
        assert not isinstance(bootstrap, SupportsVariance)


# ---------------------------------------------------------------------------
# Conditioning capabilities
# ---------------------------------------------------------------------------


class TestConditioningCapabilities:
    def test_gaussian_joint(self, joint):
        assert isinstance(joint, SupportsExactConditioning)

    def test_a_dependent_joint_claims_no_conditioning_capability(self):
        """A dependent joint claims no conditioning capability, so the inference registry conditions it."""
        joint = _ShiftKernel() * Normal("x", 0, 1)
        assert not isinstance(joint, SupportsExactConditioning)
        assert not isinstance(joint, SupportsApproximateConditioning)

    def test_multivariate_gaussian_joint(self):
        jg = MultivariateNormal("x", jnp.zeros(2), jnp.eye(2)) * MultivariateNormal(
            "y", jnp.zeros(2), jnp.eye(2)
        )
        assert isinstance(jg, SupportsExactConditioning)

    def test_normal_not_conditionable(self, normal):
        assert not isinstance(normal, SupportsExactConditioning)
        assert not isinstance(normal, SupportsApproximateConditioning)

    def test_the_exact_implementations_do_not_claim_approximate(self, joint):
        assert not isinstance(joint, SupportsApproximateConditioning)

    def test_the_capability_is_claimed_by_inheriting_not_by_the_method(self):
        """Exactness is a claim about the result, so defining ``_condition_on`` claims nothing."""

        class DefinesTheMethod:
            def _condition_on(self, observed, /, **kwargs):
                return observed

        instance = DefinesTheMethod()
        assert not isinstance(instance, SupportsExactConditioning)
        assert not isinstance(instance, SupportsApproximateConditioning)


# ---------------------------------------------------------------------------
# Named components (duck-typing check)
# ---------------------------------------------------------------------------


class TestNamedComponents:
    def test_joint_components_are_its_factors_names(self, joint):
        assert tuple(joint.event_spec.components) == ("x", "y")

    def test_normal_components_have_its_name(self, normal):
        assert tuple(normal.event_spec.components) == ("x",)


# ---------------------------------------------------------------------------
# Dynamic-protocol views (regression for hard-coded SupportsMean/Variance/
# Sampling on a field view)
# ---------------------------------------------------------------------------


class TestFieldViewDynamicProtocols:
    """A view over a field must only claim protocols its parent supports."""

    def test_view_over_log_prob_only_parent_is_not_sampling(self):
        """Build a parent with a RecordSpec that supports only
        log_prob, and verify the view doesn't claim to be
        SupportsSampling / SupportsMean / SupportsVariance.

        The view's density derives from the parent's marginals, which this
        parent lacks, so the view claims no density either."""
        from probpipe.core._specs import RecordSpec
        from probpipe.distributions import Distribution

        class _LogProbOnlyParent(Distribution, SupportsLogProb):
            def __init__(self):
                super().__init__("lp_only", RecordSpec(x=(), y=()))

            def _log_prob(self, value):
                import jax.numpy as jnp

                return jnp.asarray(0.0)

        parent = _LogProbOnlyParent()
        view = parent["x"]
        assert not isinstance(view, SupportsLogProb)
        assert not isinstance(view, SupportsSampling)
        assert not isinstance(view, SupportsMean)
        assert not isinstance(view, SupportsVariance)

    def test_view_over_full_parent_gets_all_protocols(self):
        """A joint of normals supports sampling and (through its TFP
        factors) mean / variance — its views should match."""
        dist = Normal(loc=0.0, scale=1.0, label="intercept") * Normal(
            loc=0.0, scale=1.0, label="slope"
        )
        view = dist["intercept"]
        assert isinstance(view, SupportsSampling)
        assert isinstance(view, SupportsMean)
        assert isinstance(view, SupportsVariance)


# ---------------------------------------------------------------------------
# Sample / sample_one return-type convention
# ---------------------------------------------------------------------------


class TestSampleReturnTypeConvention:
    """Pin down the contract documented on ``SupportsSampling``.

    - Numeric distributions return ``Array`` (sample_shape + event_shape).
    - Record-based joints return the raw form, the mapping of their
      components with the sample axes leading; the ``sample`` operation
      returns a ``Record`` / ``NumericRecord`` for an unbatched draw
      (``sample_shape == ()``) and a ``NumericRecordBatch`` for a batched
      draw.
    """

    def test_numeric_distribution_returns_array(self):
        import jax.numpy as jnp

        dist = Normal(loc=0.0, scale=1.0, label="x")
        k = jax.random.PRNGKey(0)
        assert isinstance(dist._sample(k, ()), jnp.ndarray)
        assert dist._sample(k, (5,)).shape == (5,)
        assert dist._sample(k, (3, 4)).shape == (3, 4)

    def test_joint_return_types(self):
        from probpipe import Record, sample
        from probpipe.core._numeric_record_batch import NumericRecordBatch

        dist = Normal(loc=0.0, scale=1.0, label="x") * Normal(loc=0.0, scale=1.0, label="y")
        k = jax.random.PRNGKey(0)
        # unbatched
        s0 = dist._sample(k, ())
        assert set(s0) == {"x", "y"} and s0["x"].shape == ()
        assert isinstance(sample(dist), Record)
        # batched
        s1 = dist._sample(k, (5,))
        assert s1["x"].shape == (5,) and s1["y"].shape == (5,)
        batch = sample(dist, sample_shape=(5,))
        assert isinstance(batch, NumericRecordBatch)
        assert batch.batch_shape == (5,)

    def test_no_distribution_exposes_sample_one(self):
        """``_sample_one`` was removed from the distribution surface —
        ``_sample(key, ())`` is the sole entry point for a single draw."""
        distributions = [
            Normal(loc=0.0, scale=1.0, label="x"),
            Normal(loc=0.0, scale=1.0, label="a") * Normal(loc=0.0, scale=1.0, label="b"),
            EmpiricalDistribution("x", jnp.arange(5.0)),
            BootstrapDistribution("bootstrap", EmpiricalDistribution("y", jnp.arange(5.0))),
        ]
        for d in distributions:
            assert not hasattr(d, "_sample_one"), (
                f"{type(d).__name__} should not expose _sample_one"
            )

    def test_record_empirical_return_types(self):
        """An empirical law over records draws the raw form, the nested mapping of its leaves."""
        rows = NumericRecordBatch(
            "rows",
            {"x": jnp.asarray([[1.0], [2.0], [3.0]]), "y": jnp.asarray([[0.5], [1.5], [2.5]])},
            "row",
            element_spec=NumericRecordSpec(x=(1,), y=(1,)),
        )
        law = EmpiricalDistribution("joint", rows)
        k = jax.random.PRNGKey(0)
        one, many = law._sample(k, ()), law._sample(k, (4,))
        assert set(one) == {"x", "y"} and one["x"].shape == (1,)
        assert many["x"].shape == (4, 1) and many["y"].shape == (4, 1)

    def test_multivariate_gaussian_joint_return_types(self):
        from probpipe import Record, sample
        from probpipe.core._numeric_record_batch import NumericRecordBatch

        jg = MultivariateNormal("x", jnp.zeros(1), jnp.eye(1)) * MultivariateNormal(
            "y", jnp.zeros(1), jnp.eye(1)
        )
        k = jax.random.PRNGKey(0)
        assert jg._sample(k, ())["x"].shape == (1,)
        assert jg._sample(k, (5,))["y"].shape == (5, 1)
        assert isinstance(sample(jg), Record)
        assert isinstance(sample(jg, sample_shape=(5,)), NumericRecordBatch)
        assert sample(jg, sample_shape=(5,)).batch_shape == (5,)


# ---------------------------------------------------------------------------
# Dynamic protocol claims on concrete distributions
# ---------------------------------------------------------------------------


class TestTransformedDistributionDynamicProtocols:
    """BijectorTransformedDistribution claims only the protocols its base supports."""

    def test_over_full_tfp_base_has_all_protocols(self):
        import tensorflow_probability.substrates.jax.bijectors as tfb

        from probpipe import Normal

        td = BijectorTransformedDistribution("td", Normal(loc=0.0, scale=1.0, label="x"), tfb.Exp())
        assert isinstance(td, SupportsSampling)
        assert isinstance(td, SupportsLogProb)
        assert not isinstance(td, SupportsMean)
        assert not isinstance(td, SupportsVariance)

    def test_over_log_prob_only_base_no_sampling(self):
        """A base with log_prob but no sampling → transform has no SupportsSampling."""
        import tensorflow_probability.substrates.jax.bijectors as tfb

        from probpipe import NumericDistribution
        from probpipe.core._specs import NumericArraySpec
        from probpipe.core.constraints import real
        from probpipe.distributions._capabilities import SupportsLogProb

        class _LogProbOnly(NumericDistribution, SupportsLogProb):
            def __init__(self):
                super().__init__("lpo", NumericArraySpec((), "float32", real))

            def _log_prob(self, x):
                return jnp.asarray(0.0)

        base = _LogProbOnly()
        td = BijectorTransformedDistribution("td", base, tfb.Identity())
        assert isinstance(td, SupportsLogProb)
        assert not isinstance(td, SupportsSampling)


class TestJointDynamicProtocols:
    """A joint's protocol claims match its factors'."""

    def test_all_tfp_components_all_protocols(self):
        joint = Normal(loc=0.0, scale=1.0, label="z") * Normal(loc=0.0, scale=1.0, label="x")
        assert isinstance(joint, SupportsSampling)
        assert isinstance(joint, SupportsLogProb)
        assert isinstance(joint, SupportsMean)
        assert isinstance(joint, SupportsVariance)
        assert isinstance(joint, SupportsExactConditioning)

    def test_empirical_component_drops_log_prob(self):
        """``EmpiricalDistribution`` lacks ``SupportsLogProb``; a
        joint containing one should not claim it."""
        boot = EmpiricalDistribution("b", jnp.array([1.0, 2.0, 3.0]))
        joint = boot * Normal(loc=0.0, scale=1.0, label="z")
        # Sampling is always available; a joint that is not Gaussian claims no
        # conditioning capability, so the inference registry conditions it.
        assert isinstance(joint, SupportsSampling)
        assert not isinstance(joint, SupportsExactConditioning)
        # MRO-level claims reflect missing log-prob on a component.
        assert SupportsLogProb not in type(joint).__mro__


# ---------------------------------------------------------------------------
# SupportsArrayBackend protocol surface
# ---------------------------------------------------------------------------


class TestSupportsArrayBackendProtocolSurface:
    """Structural checks on :class:`SupportsArrayBackend`:

    * the protocol and ``_DistributionArrayBackend`` are importable, and only
      the protocol is exported;
    * every TFP-backed distribution inherits ``_make_array_backend`` from
      ``TFPDistribution``;
    * ``_DistributionArrayBackend`` declares its minimum members.
    """

    def test_protocol_is_importable(self):
        from probpipe.core.protocols import (
            SupportsArrayBackend,
            _DistributionArrayBackend,
        )

        assert SupportsArrayBackend is not None
        assert _DistributionArrayBackend is not None

    def test_protocol_in_public_all(self):
        from probpipe.core import protocols as proto_mod

        assert "SupportsArrayBackend" in proto_mod.__all__
        # Backend-interface stays private — leading underscore, not exported.
        assert "_DistributionArrayBackend" not in proto_mod.__all__

    def test_tfp_distributions_implement_protocol(self):
        """Every concrete TFP-backed distribution inherits
        ``_make_array_backend`` from ``TFPDistribution``.

        The protocol method is a classmethod, so the check is on the
        class itself: ``hasattr(Normal, "_make_array_backend")``.
        Non-TFP distributions do not implement it and stay on the
        literal-array fallback path.
        """
        for cls in (Normal, Beta, Gamma, MultivariateNormal):
            assert hasattr(cls, "_make_array_backend"), (
                f"{cls.__name__} should inherit _make_array_backend from TFPDistribution."
            )

    def test_backend_protocol_minimum_surface(self):
        """``_DistributionArrayBackend`` Protocol declares the agreed
        minimum surface."""
        from probpipe.core.protocols import _DistributionArrayBackend

        # Protocol attributes via __annotations__ / methods via vars.
        members = set(dir(_DistributionArrayBackend))
        for required in ("batch_shape", "event_shape", "cell_spec", "cell"):
            assert required in members, (
                f"_DistributionArrayBackend missing required attr {required!r}"
            )


@pytest.mark.pending(
    reason="predictive_check and add_ppc take a GenerativeLikelihood until they take a sampling "
    "kernel",
    raises=AssertionError,
)
def test_the_generative_likelihood_protocol_retires():
    """A model is a factored joint or a program family, so no likelihood protocol remains."""
    from probpipe.core import protocols

    assert not hasattr(protocols, "GenerativeLikelihood")
