"""The correctness of inference through condition_on: references, calibration, and exactness.

Three checks complement the per-method harness bindings in ``tests/inference``:

- the canonical references themselves: each closed-form case's posterior law,
  drawn independently, meets its own reference under the harness's contract,
  which validates the references and the comparison together;
- simulation-based calibration: under a calibrated method the rank of a
  parameter drawn from the prior among the posterior draws given data drawn
  from it is uniform, for a few dozen replications;
- exactness: ``condition_on`` returns the exact conditional of a law that
  claims exact conditioning, and every approximate method applied to the same
  law agrees with it within the harness's contract.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import MultivariateNormal
from probpipe.core._dispatch import ResolutionError
from probpipe.families import LinearGaussianConditional
from probpipe.linalg import DenseLinOp
from tests._ops import condition_on, inference_method_registry, mean, variance
from tests.correctness._laws import ExactGaussianRegression, exact_reference
from tests.inference import canonical
from tests.inference._harness import (
    PROFILES,
    assert_matches,
    calibration_ranks,
    uniformity_pvalues,
)
from tests.inference.canonical import ModelTestCase

#: The family-wise level of each calibration test: a calibrated method fails it
#: with probability at most 0.01, the level split across the parameter coordinates.
CALIBRATION_LEVEL = 0.01

#: The replications and the thinned posterior draws of each calibration test:
#: two dozen replications of a fitted method, and more of the exact route,
#: whose replications cost no fit.
REPLICATIONS, EXACT_REPLICATIONS, RANK_DRAWS = 24, 40, 99

#: The approximate methods the exact conditional is compared with.
APPROXIMATE_METHODS = ("blackjax_nuts", "blackjax_hmc", "blackjax_rwmh", "tfp_nuts")


def _regression() -> ExactGaussianRegression:
    X = jax.random.normal(jax.random.PRNGKey(3), (12, 2))
    return ExactGaussianRegression(X, prior_variance=2.0)


def _observation(model: ExactGaussianRegression) -> jnp.ndarray:
    beta = jnp.array([1.0, -0.5])
    noise = jax.random.normal(jax.random.PRNGKey(4), (model.X.shape[0],))
    return (model.X @ beta + noise).astype(jnp.float32)


def _regression_case() -> ModelTestCase:
    """The exactly conditioned regression as a case, with its closed-form reference."""
    model = _regression()
    y = _observation(model)
    posterior_mean, posterior_cov = model.posterior_moments(y)
    return ModelTestCase(
        name="exact_regression",
        model=model,
        data={"y": y},
        reference=exact_reference(posterior_mean, np.diag(posterior_cov)),
        tags=frozenset({"gaussian"}),
    )


class TestReferences:
    @pytest.mark.parametrize(
        "case_name", [name for name in canonical.CASES if canonical.case(name).posterior]
    )
    def test_the_closed_form_posterior_meets_its_own_reference(self, case_name):
        """Independent draws of the closed-form posterior law meet the case's reference.

        The law and the reference are computed separately, the law by the
        library's family and the reference in double precision by SciPy, so the
        test checks both and the harness's comparison of moments and quantiles.
        """
        case = canonical.case(case_name)
        assert_matches(case.posterior, case.reference, label=f"the posterior law of {case_name}")

    def test_the_eight_schools_reference_matches_the_known_posterior(self):
        """The quadrature gives the well-known eight-schools posterior means of mu and tau.

        Under ``mu ~ N(0, 5)`` and ``tau ~ HalfCauchy(0, 5)`` the posterior means
        are about 4.4 and 3.6.
        """
        leaves = canonical.case("eight_schools").reference.leaves
        assert float(leaves["mu"].mean) == pytest.approx(4.4, abs=0.05)
        assert float(leaves["tau"].mean) == pytest.approx(3.6, abs=0.05)

    def test_each_quadrature_interval_contains_the_mean(self):
        for name in ("poisson_regression", "eight_schools"):
            for path, leaf in canonical.case(name).reference.leaves.items():
                lower, upper = leaf.quantiles[0.05], leaf.quantiles[0.95]
                assert np.all(lower < leaf.mean) and np.all(leaf.mean < upper), (name, path)


class TestCalibration:
    def _assert_uniform(self, ranks: np.ndarray) -> None:
        pvalues = uniformity_pvalues(ranks, RANK_DRAWS)
        threshold = CALIBRATION_LEVEL / ranks.shape[1]
        assert np.all(pvalues >= threshold), f"KS p-values {pvalues.round(4)} below {threshold}"

    def test_the_exact_route_is_calibrated(self):
        """The ranks of draws of the exact conditional are uniform, as they must be.

        The exact conditional is calibrated by construction, so this checks the
        calibration loop itself: the joint's draws, the route condition_on
        selects, and the rank computation.
        """
        ranks = calibration_ranks(
            None, _regression_case(), replications=EXACT_REPLICATIONS, draws=RANK_DRAWS
        )
        self._assert_uniform(ranks)

    def test_the_uniformity_test_rejects_a_shifted_posterior(self):
        """Ranks among draws shifted by one and a half posterior deviations pile up, and are rejected.

        The draws are those of the exact conditional moved by 1.5 posterior
        standard deviations in every coordinate, so the test has power against
        a biased posterior at two dozen replications.
        """
        case = _regression_case()
        rows = []
        for replication in range(REPLICATIONS):
            draw = case.model._sample(jax.random.PRNGKey(replication))
            posterior_mean, cov = case.model.posterior_moments(draw["y"])
            shifted = np.random.default_rng(replication).multivariate_normal(
                posterior_mean + 1.5 * np.sqrt(np.diag(cov)), cov, size=RANK_DRAWS
            )
            rows.append((shifted < np.asarray(draw["beta"])).sum(axis=0))
        pvalues = uniformity_pvalues(np.stack(rows), RANK_DRAWS)
        assert np.all(pvalues < CALIBRATION_LEVEL / 2)

    @pytest.mark.parametrize("method", ["blackjax_nuts", "blackjax_rwmh"])
    def test_the_method_is_calibrated_on_the_gaussian_linear_case(self, method):
        """The method's ranks over two dozen replications of the Gaussian linear case are uniform.

        Each replication draws the coefficients and the responses from the
        case's joint and fits the posterior with the method, which compiles its
        kernel anew for each dataset, so the test takes twenty to forty seconds.
        """
        if method not in inference_method_registry.list_methods():
            pytest.skip(f"{method} is not registered here")
        budget = {"num_results": 500, "num_warmup": 300, "num_chains": 1}
        if method == "blackjax_rwmh":
            budget = {"num_results": 2000, "num_warmup": 1000, "num_chains": 1}
        ranks = calibration_ranks(
            method,
            canonical.case("gaussian_linear"),
            replications=REPLICATIONS,
            draws=RANK_DRAWS,
            method_options=budget,
        )
        self._assert_uniform(ranks)


class TestExactness:
    def test_the_exact_conditioning_route_is_selected_and_exact(self):
        model = _regression()
        report = condition_on.check(model, {"y": _observation(model)})
        assert (report.feasible, report.route, report.exact) == (True, "exact_conditioning", True)
        assert report.method is None

    def test_the_exact_conditional_is_the_closed_form_posterior(self):
        model = _regression()
        y = _observation(model)
        posterior = condition_on(model, {"y": y})
        expected_mean, expected_cov = model.posterior_moments(y)
        assert isinstance(posterior, MultivariateNormal)
        np.testing.assert_allclose(
            np.asarray(mean.with_options(raw=True)(posterior)), expected_mean, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(variance.with_options(raw=True)(posterior)),
            np.diag(expected_cov),
            rtol=1e-5,
        )

    def test_exact_only_returns_the_exact_conditional(self):
        model = _regression()
        y = _observation(model)
        posterior = condition_on.with_options(exact_only=True)(model, {"y": y})
        np.testing.assert_allclose(
            np.asarray(mean.with_options(raw=True)(posterior)),
            model.posterior_moments(y)[0],
            rtol=1e-5,
        )

    def test_exact_only_refuses_an_approximate_method(self):
        model = _regression()
        with pytest.raises(ResolutionError):
            condition_on.with_options(method="blackjax_nuts", exact_only=True)(
                model, {"y": _observation(model)}
            )

    @pytest.mark.parametrize("method", APPROXIMATE_METHODS)
    def test_each_approximate_method_agrees_with_the_exact_conditional(self, method):
        """An approximate method's posterior of the same law meets the exact conditional's moments.

        The exact conditional is computed through condition_on's exact route,
        and the method through its normalization stage on the law's density, so
        the two routes are compared on one law. Each fit takes a few seconds.
        """
        if method not in inference_method_registry.list_methods():
            pytest.skip(f"{method} is not registered here")
        model = _regression()
        given = {"y": _observation(model)}
        exact = condition_on(model, given)
        reference = exact_reference(
            np.asarray(mean.with_options(raw=True)(exact), np.float64),
            np.asarray(variance.with_options(raw=True)(exact), np.float64),
        )
        profile = PROFILES[method]
        approximate = condition_on.with_options(
            method=method, method_options=profile.method_options
        )(model, given)
        assert_matches(approximate, reference, label=f"{method} against the exact conditional")

    def test_exact_only_rejects_a_normalization_by_inference(self):
        """Under exact_only, a joint without an exact route raises, naming method="unnormalized"."""
        case = canonical.case("gaussian_linear")
        with pytest.raises(ResolutionError, match='method="unnormalized"'):
            condition_on.with_options(exact_only=True)(case.model, case.data)

    @pytest.mark.pending(
        reason="LinearGaussianConditional and the Gaussian algebra's exact conditioning"
    )
    def test_a_linear_gaussian_joint_is_conditioned_exactly(self):
        """A Gaussian prior and a linear-Gaussian observation compose to an exactly conditioned joint."""
        case = canonical.case("gaussian_linear")
        X = jnp.asarray(case.stan_data["X"], jnp.float32)
        likelihood = LinearGaussianConditional(
            "y", DenseLinOp(X), jnp.zeros(X.shape[0]), DenseLinOp(jnp.eye(X.shape[0]))
        )
        prior = MultivariateNormal("beta", jnp.zeros(X.shape[1]), cov=4.0 * jnp.eye(X.shape[1]))
        posterior = condition_on.with_options(exact_only=True)(likelihood * prior, case.data)
        np.testing.assert_allclose(
            np.asarray(mean.with_options(raw=True)(posterior)["mean(beta)"]),
            case.reference.leaves["beta"].mean,
            rtol=1e-4,
        )


class TestInitialState:
    def test_a_joints_conditional_starts_at_a_draw_of_the_joint(self):
        """The chain of the Beta-Bernoulli posterior starts inside the unit interval, at a joint draw."""
        from probpipe.inference._inference_utils import get_init_state

        case = canonical.case("beta_bernoulli")
        target = condition_on.with_options(method="unnormalized")(case.model, case.data)
        init = np.asarray(get_init_state(target, None, random_seed=0))
        assert init.shape == (1,)
        assert 0.0 < float(init[0]) < 1.0
