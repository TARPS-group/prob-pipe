"""Contracts of the conditional families (VII.8).

The GLM assembly is implemented: the response families, their canonical links,
and ``glm_likelihood``. The linear-Gaussian kernel is pending.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Bernoulli,
    Function,
    HalfNormal,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    NumericDistribution,
    OutputSpec,
    Poisson,
    ResolutionError,
    condition_on,
    log_prob,
    sample,
)
from probpipe.distributions import (
    ConditionalDistribution,
    Distribution,
    FactoredConditionalDistribution,
    FactoredDistribution,
    FullyNumericConditionalDistribution,
)
from probpipe.distributions._capabilities import (
    SupportsConditionalCovariance,
    SupportsConditionalLogProb,
    SupportsConditionalMean,
    SupportsConditionalSampling,
    SupportsConditionalVariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from probpipe.families import (
    BernoulliFamily,
    GaussianFamily,
    GLMFamily,
    LinearGaussianConditional,
    PoissonFamily,
    glm_likelihood,
)
from probpipe.linalg import DenseLinOp, LinOp

_FAMILIES = [GaussianFamily, BernoulliFamily, PoissonFamily]


@pytest.fixture
def X():
    return jnp.array([[1.0, -1.0], [1.0, 0.0], [1.0, 0.5], [1.0, 2.0]])


@pytest.fixture
def beta():
    return jnp.array([0.3, 0.8])


def _dispersion(family: GLMFamily):
    return 0.5 if family.has_dispersion else None


class _UnitScaleGaussian(GLMFamily):
    """A family of unit-scale normal observations that defines only the members VII.8 declares."""

    canonical_link = GaussianFamily.canonical_link
    has_dispersion = False

    def build(self, label, mean, dispersion=None, *, event_spec=None):
        return MultivariateNormal(label, mean, cov=jnp.eye(mean.shape[0]), event_spec=event_spec)


# ---------------------------------------------------------------------------
# The response families
# ---------------------------------------------------------------------------


class TestTheResponseFamilies:
    def test_a_family_is_abstract(self):
        with pytest.raises(TypeError):
            GLMFamily()

    @pytest.mark.parametrize(
        ("family", "link", "has_dispersion"),
        [
            (GaussianFamily, "identity", True),
            (BernoulliFamily, "logit", False),
            (PoissonFamily, "log", False),
        ],
    )
    def test_the_canonical_links_and_dispersions(self, family, link, has_dispersion):
        assert isinstance(family.canonical_link, Function)
        assert family.canonical_link.label == link
        assert family.has_dispersion is has_dispersion

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_the_canonical_link_inverts(self, family):
        mean = jnp.array([0.2, 0.5, 0.9])
        link = family.canonical_link
        np.testing.assert_allclose(link._inverse(link.apply(mean)), mean, rtol=1e-5)

    def test_a_family_whose_canonical_link_is_not_invertible_raises_at_construction(self):
        class Opaque(GLMFamily):
            canonical_link = Function("square", lambda mean: mean**2)
            has_dispersion = False

            def build(self, label, mean, dispersion=None, *, event_spec=None):
                raise AssertionError("unreachable")

        with pytest.raises(ResolutionError, match="not invertible"):
            Opaque()

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_build_is_the_law_of_one_observation_per_mean(self, family):
        law = family().build("y", jnp.array([0.2, 0.5, 0.9]), _dispersion(family))
        assert isinstance(law, Distribution)
        assert isinstance(law, NumericDistribution)
        assert list(law.event_spec.components) == ["y"]
        assert law.event_spec.spec.shape == (3,)
        for protocol in (SupportsSampling, SupportsLogProb, SupportsMean, SupportsVariance):
            assert isinstance(law, protocol), protocol.__name__
        assert law._sample(jax.random.PRNGKey(0)).shape == (3,)

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_the_law_has_the_given_means(self, family):
        mean = jnp.array([0.2, 0.5, 0.9])
        law = family().build("y", mean, _dispersion(family))
        np.testing.assert_allclose(law._mean(), mean, rtol=1e-6)

    @pytest.mark.parametrize(
        ("family", "variance"),
        [
            (GaussianFamily, lambda m: jnp.full_like(m, 0.25)),
            (BernoulliFamily, lambda m: m * (1 - m)),
            (PoissonFamily, lambda m: m),
        ],
    )
    def test_the_law_has_the_family_variance(self, family, variance):
        mean = jnp.array([0.2, 0.5, 0.9])
        law = family().build("y", mean, _dispersion(family))
        np.testing.assert_allclose(law._variance(), variance(mean), rtol=1e-5)

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_the_covariance_is_a_diagonal_operator(self, family):
        mean = jnp.array([0.2, 0.5, 0.9])
        law = family().build("y", mean, _dispersion(family))
        covariance = law._cov()
        assert isinstance(covariance, LinOp)
        np.testing.assert_allclose(covariance.to_dense(), jnp.diag(law._variance()), rtol=1e-6)

    @pytest.mark.parametrize(
        ("family", "scalar"),
        [
            (GaussianFamily, lambda label, m: Normal(label, m, 0.5)),
            (BernoulliFamily, lambda label, m: Bernoulli(label, probs=m)),
            (PoissonFamily, lambda label, m: Poisson(label, m)),
        ],
    )
    def test_the_observations_are_conditionally_independent(self, family, scalar):
        mean = jnp.array([0.2, 0.5, 0.9])
        y = jnp.array([0.0, 1.0, 1.0])
        law = family().build("y", mean, _dispersion(family))
        separate = sum(scalar("y", m)._log_prob(v) for m, v in zip(mean, y, strict=True))
        np.testing.assert_allclose(law._log_prob(y), separate, rtol=1e-5)

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_event_spec_names_the_component(self, family):
        law = family().build(
            "y", jnp.array([0.2, 0.5]), _dispersion(family), event_spec=OutputSpec(counts=None)
        )
        assert law.label == "y"
        assert list(law.event_spec.components) == ["counts"]

    def test_a_missing_dispersion_raises(self):
        with pytest.raises(TypeError, match="requires a dispersion"):
            GaussianFamily().build("y", jnp.array([0.0, 1.0]))

    def test_an_integer_dispersion_takes_the_floating_dtype_of_the_mean(self):
        mean = jnp.zeros(3)
        law = GaussianFamily().build("y", mean, 2)
        np.testing.assert_allclose(law._variance(), jnp.full(3, 4.0), rtol=1e-6)
        assert law._log_prob(mean).dtype == mean.dtype

    @pytest.mark.parametrize("family", [BernoulliFamily, PoissonFamily])
    def test_a_dispersion_for_a_family_without_one_raises(self, family):
        with pytest.raises(TypeError, match="takes no dispersion"):
            family().build("y", jnp.array([0.5]), 1.0)

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_a_mean_that_is_not_a_vector_raises(self, family):
        with pytest.raises(ValueError, match="vector"):
            family().build("y", jnp.asarray(0.5), _dispersion(family))


# ---------------------------------------------------------------------------
# The GLM likelihood
# ---------------------------------------------------------------------------


class TestTheDeclarations:
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_the_given_slots_and_the_response(self, family):
        likelihood = glm_likelihood("y", family())
        assert isinstance(likelihood, ConditionalDistribution)
        expected = ["X", "beta", "dispersion"] if family.has_dispersion else ["X", "beta"]
        assert list(likelihood.given_spec) == expected
        assert likelihood.given_spec["X"].shape == ("obs", "features")
        assert likelihood.given_spec["beta"].shape == ("features",)
        assert list(likelihood.event_spec.components) == ["y"]
        assert likelihood.event_spec.spec.shape == ("obs",)

    def test_the_dimensions_are_symbolic_until_bound(self):
        likelihood = glm_likelihood("y", BernoulliFamily())
        assert likelihood.spec.free_dims == {"obs", "features"}
        bound = likelihood.with_dim_sizes(obs=4, features=2)
        assert bound.given_spec["X"].shape == (4, 2)
        assert bound.event_spec.spec.shape == (4,)

    def test_X_at_construction_binds_the_dimensions_and_leaves_its_slot(self, X):
        likelihood = glm_likelihood("y", PoissonFamily(), X=X)
        assert list(likelihood.given_spec) == ["beta"]
        assert likelihood.given_spec["beta"].shape == (2,)
        assert likelihood.event_spec.spec.shape == (4,)

    def test_the_dispersion_at_construction_leaves_its_slot(self, X):
        likelihood = glm_likelihood("y", GaussianFamily(), X=X, dispersion=0.5)
        assert list(likelihood.given_spec) == ["beta"]

    def test_event_spec_names_the_response(self, X):
        likelihood = glm_likelihood("y", PoissonFamily(), X=X, event_spec=OutputSpec(counts=None))
        assert list(likelihood.event_spec.components) == ["counts"]

    def test_both_sides_are_numeric(self, X):
        assert isinstance(
            glm_likelihood("y", PoissonFamily(), X=X), FullyNumericConditionalDistribution
        )

    def test_the_kernel_claims_the_conditional_capabilities(self, X):
        likelihood = glm_likelihood("y", BernoulliFamily(), X=X)
        for protocol in (
            SupportsConditionalSampling,
            SupportsConditionalLogProb,
            SupportsConditionalMean,
            SupportsConditionalVariance,
            SupportsConditionalCovariance,
        ):
            assert isinstance(likelihood, protocol), protocol.__name__

    def test_a_response_named_like_a_given_slot_raises(self):
        with pytest.raises(ValueError, match="given slot"):
            glm_likelihood("beta", PoissonFamily())

    def test_a_declared_response_type_binds_the_observations_on_both_sides(self):
        likelihood = glm_likelihood(
            "y", PoissonFamily(), event_spec=OutputSpec(counts=NumericArraySpec((5,)))
        )
        assert likelihood.event_spec.spec.shape == (5,)
        assert likelihood.given_spec["X"].shape == (5, "features")

    def test_a_declared_response_type_that_agrees_with_X_is_kept(self, X, beta):
        likelihood = glm_likelihood(
            "y", PoissonFamily(), X=X, event_spec=OutputSpec(counts=NumericArraySpec((4,)))
        )
        assert likelihood.event_spec.spec.shape == (4,)
        law = likelihood._condition_on({"beta": beta})
        assert list(law.event_spec.components) == ["counts"]

    def test_a_renamed_kernel_keeps_its_response_component(self, X):
        likelihood = glm_likelihood("y", PoissonFamily(), X=X).with_label("L")
        assert likelihood.label == "L"
        assert list(likelihood.event_spec.components) == ["y"]


class TestACustomFamily:
    """VII.8: a family needs only its canonical link, has_dispersion, and build."""

    def test_the_likelihood_assembles_from_the_declared_members(self, X):
        likelihood = glm_likelihood("y", _UnitScaleGaussian(), X=X)
        assert list(likelihood.given_spec) == ["beta"]
        assert list(likelihood.event_spec.components) == ["y"]
        assert likelihood.event_spec.spec.shape == (4,)

    def test_the_law_is_the_familys_law_at_the_mean(self, X, beta):
        law = glm_likelihood("y", _UnitScaleGaussian(), X=X)._condition_on({"beta": beta})
        y = jnp.array([0.0, 1.0, 0.5, 2.0])
        expected = MultivariateNormal("y", X @ beta, cov=jnp.eye(4))
        np.testing.assert_allclose(law._mean(), X @ beta, rtol=1e-6)
        np.testing.assert_allclose(law._log_prob(y), expected._log_prob(y), rtol=1e-6)


class TestTheConstructionErrors:
    def test_a_family_that_is_not_a_glm_family_raises(self):
        with pytest.raises(TypeError, match="GLMFamily"):
            glm_likelihood("y", Normal)

    def test_a_link_that_is_not_a_function_raises(self):
        with pytest.raises(TypeError, match="Function"):
            glm_likelihood("y", PoissonFamily(), jnp.exp)

    def test_a_link_that_is_not_invertible_raises(self):
        with pytest.raises(ResolutionError, match="not invertible"):
            glm_likelihood("y", PoissonFamily(), Function("square", lambda mean: mean**2))

    def test_a_dispersion_for_a_family_without_one_raises(self):
        with pytest.raises(TypeError, match="takes no dispersion"):
            glm_likelihood("y", BernoulliFamily(), dispersion=1.0)

    def test_an_X_that_is_not_a_matrix_raises(self):
        with pytest.raises(ValueError, match="obs"):
            glm_likelihood("y", PoissonFamily(), X=jnp.ones(3))

    def test_an_event_spec_that_is_not_an_output_spec_raises(self):
        with pytest.raises(TypeError, match="OutputSpec"):
            glm_likelihood("y", PoissonFamily(), event_spec=NumericArraySpec(("obs",)))

    def test_a_declared_response_type_that_disagrees_with_X_raises(self, X):
        with pytest.raises(ValueError, match="X"):
            glm_likelihood(
                "y", PoissonFamily(), X=X, event_spec=OutputSpec(counts=NumericArraySpec((5,)))
            )


class TestTheLaw:
    @pytest.mark.parametrize(
        ("family", "inverse_link"),
        [
            (GaussianFamily, lambda eta: eta),
            (BernoulliFamily, jax.nn.sigmoid),
            (PoissonFamily, jnp.exp),
        ],
    )
    def test_the_law_is_the_family_at_the_inverse_link_of_the_predictor(
        self, family, inverse_link, X, beta
    ):
        likelihood = glm_likelihood("y", family(), X=X, dispersion=_dispersion(family))
        law = likelihood._condition_on({"beta": beta})
        assert isinstance(law, Distribution)
        np.testing.assert_allclose(law._mean(), inverse_link(X @ beta), rtol=1e-5)
        direct = family().build("y", inverse_link(X @ beta), _dispersion(family))
        y = jnp.array([0.0, 1.0, 1.0, 2.0])
        np.testing.assert_allclose(law._log_prob(y), direct._log_prob(y), rtol=1e-5)

    def test_a_link_replaces_the_canonical_one(self, X, beta):
        likelihood = glm_likelihood("y", PoissonFamily(), GaussianFamily.canonical_link, X=X)
        np.testing.assert_allclose(
            likelihood._condition_on({"beta": beta})._mean(), X @ beta, rtol=1e-6
        )

    def test_the_law_carries_the_declared_response(self, X, beta):
        likelihood = glm_likelihood("y", PoissonFamily(), X=X, event_spec=OutputSpec(counts=None))
        law = likelihood._condition_on({"beta": beta})
        assert list(law.event_spec.components) == ["counts"]
        assert law.event_spec.spec.shape == (4,)

    def test_the_law_of_a_renamed_kernel_keeps_the_response_component(self, X, beta):
        law = (
            glm_likelihood("y", PoissonFamily(), X=X).with_label("L")._condition_on({"beta": beta})
        )
        assert law.label == "L"
        assert list(law.event_spec.components) == ["y"]

    def test_a_fixed_integer_dispersion_builds_the_law(self, X, beta):
        law = glm_likelihood("y", GaussianFamily(), X=X, dispersion=2)._condition_on({"beta": beta})
        np.testing.assert_allclose(law._variance(), jnp.full(4, 4.0), rtol=1e-6)

    def test_binding_every_slot_with_keywords(self, X, beta):
        likelihood = glm_likelihood("y", GaussianFamily())
        law = likelihood._condition_on({"X": X}, beta=beta, dispersion=0.5)
        np.testing.assert_allclose(law._mean(), X @ beta, rtol=1e-6)
        np.testing.assert_allclose(law._variance(), jnp.full(4, 0.25), rtol=1e-6)

    def test_binding_some_slots_curries(self, X, beta):
        likelihood = glm_likelihood("y", GaussianFamily())
        curried = likelihood._condition_on({"X": X})
        assert isinstance(curried, ConditionalDistribution)
        assert list(curried.given_spec) == ["beta", "dispersion"]
        assert curried.given_spec["beta"].shape == (2,)
        assert curried.event_spec.spec.shape == (4,)
        law = curried._condition_on({"beta": beta, "dispersion": 0.5})
        np.testing.assert_allclose(law._mean(), X @ beta, rtol=1e-6)

    def test_binding_beta_first_binds_the_features(self, X, beta):
        curried = glm_likelihood("y", BernoulliFamily())._condition_on({"beta": beta})
        assert list(curried.given_spec) == ["X"]
        assert curried.given_spec["X"].shape == ("obs", 2)

    def test_currying_leaves_the_original_kernel_unchanged(self, X):
        likelihood = glm_likelihood("y", BernoulliFamily())
        likelihood._condition_on({"X": X})
        assert list(likelihood.given_spec) == ["X", "beta"]

    def test_an_unknown_slot_raises(self, X, beta):
        with pytest.raises(KeyError, match="not given slots"):
            glm_likelihood("y", PoissonFamily(), X=X)._condition_on({"beta": beta, "gamma": 1.0})

    def test_a_value_of_the_wrong_shape_raises(self, X):
        with pytest.raises(ValueError, match="beta"):
            glm_likelihood("y", PoissonFamily(), X=X)._condition_on({"beta": jnp.ones(3)})


class TestTheCanonicalLink:
    """Under the canonical link the law is built from the linear predictor, not from the mean."""

    @pytest.fixture
    def predictor(self):
        return jnp.array([17.0, -17.0])

    def _logistic(self, predictor):
        return glm_likelihood("y", BernoulliFamily(), X=predictor[:, None])

    def test_the_logistic_log_density_is_exact_where_the_probability_rounds_to_one(self, predictor):
        law = self._logistic(predictor)._condition_on({"beta": jnp.array([1.0])})
        y = jnp.array([0.0, 1.0])
        # log p(y) = y η − softplus(η), which is −17 for each observation here.
        expected = y @ predictor - jnp.sum(jax.nn.softplus(predictor))
        np.testing.assert_allclose(law._log_prob(y), expected, rtol=1e-6)

    def test_the_logistic_gradient_is_finite(self, predictor):
        likelihood = self._logistic(predictor)
        y = jnp.array([0.0, 1.0])
        gradient = jax.grad(lambda b: likelihood._conditional_log_prob({"beta": b}, y))(
            jnp.array([1.0])
        )
        # d/dβ log p(y) = xᵀ (y − sigmoid(η)) for the one feature x = η.
        expected = predictor @ (y - jax.nn.sigmoid(predictor))
        assert bool(jnp.all(jnp.isfinite(gradient)))
        np.testing.assert_allclose(gradient, [expected], rtol=1e-5)

    def test_the_poisson_log_density_is_exact_where_the_rate_underflows(self):
        likelihood = glm_likelihood("y", PoissonFamily(), X=jnp.array([[-110.0]]))
        y = jnp.array([1.0])
        law = likelihood._condition_on({"beta": jnp.array([1.0])})
        # log p(1) = η − exp(η) − log 1!, which is −110 to float precision.
        np.testing.assert_allclose(law._log_prob(y), -110.0, rtol=1e-6)
        gradient = jax.grad(lambda b: likelihood._conditional_log_prob({"beta": b}, y))(
            jnp.array([1.0])
        )
        np.testing.assert_allclose(gradient, [-110.0], rtol=1e-5)


class TestTheConditionalCapabilities:
    """Each capability agrees with the same capability of the conditioned law."""

    @pytest.fixture
    def likelihood(self, X):
        return glm_likelihood("y", GaussianFamily(), X=X)

    @pytest.fixture
    def given(self, beta):
        return {"beta": beta, "dispersion": jnp.asarray(0.5)}

    def test_log_prob(self, likelihood, given):
        y = jnp.array([0.0, 1.0, 0.5, 2.0])
        np.testing.assert_allclose(
            likelihood._conditional_log_prob(given, y),
            likelihood._condition_on(given)._log_prob(y),
            rtol=1e-6,
        )

    def test_mean_and_variance(self, likelihood, given):
        law = likelihood._condition_on(given)
        np.testing.assert_allclose(likelihood._conditional_mean(given), law._mean(), rtol=1e-6)
        np.testing.assert_allclose(
            likelihood._conditional_variance(given), law._variance(), rtol=1e-6
        )

    def test_covariance(self, likelihood, given):
        covariance = likelihood._conditional_cov(given)
        assert isinstance(covariance, LinOp)
        np.testing.assert_allclose(
            covariance.to_dense(), likelihood._condition_on(given)._cov().to_dense(), rtol=1e-6
        )

    def test_sample(self, likelihood, given):
        key = jax.random.PRNGKey(3)
        np.testing.assert_allclose(
            likelihood._conditional_sample(given, key, (2,)),
            likelihood._condition_on(given)._sample(key, (2,)),
        )

    def test_a_capability_needs_every_given_slot(self, likelihood, beta):
        with pytest.raises(KeyError, match="every given slot"):
            likelihood._conditional_mean({"beta": beta})


class TestComposition:
    def test_the_likelihood_composes_with_its_priors(self, X):
        likelihood = glm_likelihood("y", GaussianFamily(), X=X)
        prior = MultivariateNormal("beta", jnp.zeros(2), cov=jnp.eye(2))
        scale = HalfNormal("dispersion", 1.0)
        joint = likelihood * prior * scale
        assert isinstance(joint, FactoredDistribution)
        assert joint.factors == (likelihood, prior, scale)
        assert list(joint.event_spec.components) == ["y", "beta", "dispersion"]

    def test_an_unbound_design_leaves_a_conditional_joint(self, X):
        likelihood = glm_likelihood("y", BernoulliFamily())
        joint = likelihood * MultivariateNormal("beta", jnp.zeros(2), cov=jnp.eye(2))
        assert isinstance(joint, FactoredConditionalDistribution)
        assert list(joint.given_spec) == ["X"]
        assert joint.given_spec["X"].shape == ("obs", 2)
        bound = joint._condition_on({"X": X})
        assert isinstance(bound, FactoredDistribution)
        assert bound.event_spec.components["y"].shape == (4,)

    def test_the_joint_samples(self, X):
        likelihood = glm_likelihood("y", PoissonFamily(), X=X)
        joint = likelihood * MultivariateNormal("beta", jnp.zeros(2), cov=jnp.eye(2))
        draw = sample(joint)
        assert draw["y"].shape == (4,)


class TestTheOperations:
    """III.9's fused conditional paths and conditioning, through the operations."""

    @pytest.mark.pending(reason="log_prob takes given= for a conditional operand", raises=TypeError)
    def test_log_prob_given(self, X, beta):
        likelihood = glm_likelihood("y", PoissonFamily(), X=X)
        y = jnp.array([0.0, 1.0, 1.0, 2.0])
        expected = likelihood._condition_on({"beta": beta})._log_prob(y)
        np.testing.assert_allclose(log_prob(likelihood, y, given={"beta": beta}), expected)

    def test_condition_on_binds_the_given_slots(self, X, beta):
        likelihood = glm_likelihood("y", PoissonFamily(), X=X)
        law = condition_on(likelihood, {"beta": beta})
        np.testing.assert_allclose(law._mean(), jnp.exp(X @ beta), rtol=1e-5)


# ---------------------------------------------------------------------------
# The linear-Gaussian kernel
# ---------------------------------------------------------------------------


class TestTheLinearGaussianKernel:
    @pytest.fixture
    def A(self):
        return DenseLinOp(jnp.array([[1.0, 2.0], [0.0, 1.0], [1.0, -1.0]]))

    @pytest.mark.pending(reason="the kernel reads its given slot and event type from A")
    def test_the_given_slot_is_the_operator_input_slot(self, A):
        kernel = LinearGaussianConditional("y", A, jnp.zeros(3), DenseLinOp(jnp.eye(3)))
        assert len(kernel.given_spec) == 1
        assert list(kernel.event_spec.components) == ["y"]
        assert kernel.event_spec.spec.shape == (3,)

    @pytest.mark.pending(reason="conditioning gives N(A @ s + b, cov)")
    def test_conditioning_gives_the_gaussian_law(self, A):
        b = jnp.array([0.5, 0.0, -1.0])
        kernel = LinearGaussianConditional("y", A, b, DenseLinOp(2.0 * jnp.eye(3)))
        (slot,) = kernel.given_spec
        s = jnp.array([1.0, 1.0])
        law = kernel._condition_on({slot: s})
        np.testing.assert_allclose(law._mean(), A.to_dense() @ s + b, rtol=1e-6)
        np.testing.assert_allclose(law._cov().to_dense(), 2.0 * jnp.eye(3), rtol=1e-6)
