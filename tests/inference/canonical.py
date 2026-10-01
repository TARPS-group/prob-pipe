"""The canonical model set of the cross-method validation harness.

Each builder returns a :class:`ModelTestCase`: a model written as the joint law
of its parameters and its observations, the observed values ``condition_on``
binds, and the posterior reference a method's result is compared with. The
model is the joint the library composes with ``*``. A case that a native
modeling language expresses also carries the same model as a ``PyMCModel``
and as a Stan program, so that the backends that consume those languages are
validated against the same data and the same reference.

The six cases span the regimes the harness stresses:

- ``gaussian_linear``: a correlated Gaussian posterior, in closed form;
- ``beta_bernoulli``: a skewed posterior on the unit interval, in closed form;
- ``gamma_poisson``: a skewed posterior on the positive half-line, in closed form;
- ``dirichlet_multinomial``: a posterior on the simplex, in closed form;
- ``poisson_regression``: a non-conjugate GLM whose two-coefficient posterior
  is integrated on a grid;
- ``eight_schools``: a non-centered hierarchical model whose posterior given the
  group scale is Gaussian, so a quadrature over the scale gives every moment
  and quantile.

Every reference is computed in double precision from the observed values the
model holds, so it is the posterior of the data the method sees.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import scipy.stats
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    Beta,
    Dirichlet,
    Gamma,
    HalfCauchy,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    OutputSpec,
    Record,
)
from probpipe.core.constraints import Constraint, boolean, non_negative_integer, real
from probpipe.distributions import ConditionalDistribution, Distribution
from probpipe.distributions._capabilities import (
    SupportsConditionalLogProb,
    SupportsConditionalSampling,
)
from probpipe.families import GaussianFamily, PoissonFamily, TFPDistribution, glm_likelihood
from probpipe.validation import Reference

__all__ = [
    "CASES",
    "LeafReference",
    "ModelTestCase",
    "ObservationKernel",
    "PosteriorReference",
    "ScaleMixture",
    "beta_bernoulli",
    "case",
    "dirichlet_multinomial",
    "eight_schools",
    "gamma_poisson",
    "gaussian_linear",
    "poisson_regression",
]

#: The levels of the central intervals the harness compares: the 90% interval.
INTERVAL_LEVELS = (0.05, 0.95)


# ---------------------------------------------------------------------------
# The observation kernels
# ---------------------------------------------------------------------------


class _Observations(TFPDistribution):
    """The law of one observation vector, adapting a backend distribution whose event is it."""

    def __init__(
        self,
        name: str,
        backend: tfd.Distribution,
        support: Constraint,
        *,
        event_spec: OutputSpec | None = None,
    ) -> None:
        self._support = support
        super().__init__(name, backend, event_spec=event_spec)

    def _event_support(self) -> Constraint:
        return self._support


def _top(given: Any) -> dict[str, Any]:
    """The values of *given*, a record or a mapping, by slot name."""
    return dict(given.children if isinstance(given, Record) else given)


class ObservationKernel(
    ConditionalDistribution, SupportsConditionalSampling, SupportsConditionalLogProb
):
    """The kernel of an observation vector whose law a backend distribution gives at its givens.

    Its event is the response, a whole term under the kernel's name, and its law
    at a value of every given slot is ``build(**values)``, a backend distribution
    whose event is the response. It claims conditional sampling and the
    normalized conditional density, so a joint of it and normalized priors is
    normalized.

    Parameters
    ----------
    name : str
        The kernel's label and the component of the response.
    given : Mapping[str, NumericArraySpec]
        The given slots, each the event spec of the prior factor that produces it.
    response : NumericArraySpec
        The declared response vector.
    build : callable
        ``build(**values)``, the backend law of the response at the slot values.
    """

    def __init__(
        self,
        name: str,
        given: Mapping[str, NumericArraySpec],
        response: NumericArraySpec,
        build: Callable[..., tfd.Distribution],
    ) -> None:
        super().__init__(name, dict(given), OutputSpec(**{name: response}))
        object.__setattr__(self, "_build", build)
        object.__setattr__(self, "_support", response.support)

    def _law(self, given: Any) -> Distribution:
        values = {slot: jnp.asarray(value) for slot, value in _top(given).items()}
        return _Observations(
            self.name, self._build(**values), self._support, event_spec=self.event_spec
        )

    def _condition_on(self, given: Any, /, **options: Any) -> Distribution:
        return self._law(_top(given))

    def _conditional_sample(self, given: Any, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return self._law(given)._sample(key, sample_shape)

    def _conditional_log_prob(self, given: Any, value: Any) -> Any:
        return self._law(given)._log_prob(value)


# ---------------------------------------------------------------------------
# The references
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeafReference:
    """The posterior summaries of one leaf of the parameter record.

    Attributes
    ----------
    mean, variance : np.ndarray
        The posterior mean and variance of each coordinate, shaped like the leaf.
    quantiles : Mapping[float, np.ndarray]
        The posterior quantile of each coordinate at each level, shaped like the leaf.
    """

    mean: np.ndarray
    variance: np.ndarray
    quantiles: Mapping[float, np.ndarray]


@dataclass(frozen=True)
class PosteriorReference:
    """The posterior of a case, summarized leaf by leaf and keyed by event path.

    Attributes
    ----------
    leaves : Mapping[str, LeafReference]
        One entry per leaf of the parameter record, keyed by its event path.
    source : str
        How the summaries were computed: ``"closed form"`` or ``"quadrature"``.
    """

    leaves: Mapping[str, LeafReference]
    source: str

    def flat(self) -> Reference:
        """The flat moments of ``probpipe.validation``, in the order the leaves are listed.

        The covariance is the diagonal of the variances, since the harness
        compares coordinates one at a time.
        """
        mean = np.concatenate([np.ravel(leaf.mean) for leaf in self.leaves.values()])
        variance = np.concatenate([np.ravel(leaf.variance) for leaf in self.leaves.values()])
        return Reference.from_moments(mean=jnp.asarray(mean), cov=jnp.diag(jnp.asarray(variance)))


def _leaf(mean: Any, variance: Any, ppf: Callable[[float], Any]) -> LeafReference:
    return LeafReference(
        mean=np.asarray(mean, dtype=np.float64),
        variance=np.asarray(variance, dtype=np.float64),
        quantiles={q: np.asarray(ppf(q), dtype=np.float64) for q in INTERVAL_LEVELS},
    )


# ---------------------------------------------------------------------------
# The case
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTestCase:
    """One canonical model, its observed values, and its posterior reference.

    Attributes
    ----------
    name : str
        The case's name.
    model : Distribution
        The joint law of the parameters and the observations, composed with ``*``.
    data : Mapping[str, Any]
        The observed values, keyed by the components of the model they condition.
    reference : PosteriorReference
        The posterior of the parameters given *data*.
    tags : frozenset of str
        The regimes the case stresses, such as ``"gaussian"`` or ``"constrained"``.
    posterior : Distribution or None
        The posterior as a library law, when a closed form exists.
    pymc_model : callable or None
        ``pymc_model()`` builds the same joint as a ``PyMCModel``, whose observed
        variables are the keys of *data*.
    stan_program : str or None
        The same model as a Stan program.
    stan_data : Mapping[str, Any] or None
        The value of every entry of the program's data block, the observations
        included.
    """

    name: str
    model: Distribution
    data: Mapping[str, Any]
    reference: PosteriorReference
    tags: frozenset[str]
    posterior: Distribution | None = None
    pymc_model: Callable[[], Any] | None = None
    stan_program: str | None = None
    stan_data: Mapping[str, Any] | None = None

    @property
    def parameters(self) -> tuple[str, ...]:
        """The event paths of the parameter leaves, in the order the reference lists them."""
        return tuple(self.reference.leaves)

    def stan_model(self, directory: Path) -> Any:
        """The Stan program as a ``StanModel`` kernel over its data block, written into *directory*.

        Raises
        ------
        ValueError
            If the case has no Stan program.
        """
        from probpipe.families import StanModel

        if self.stan_program is None:
            raise ValueError(f"the case {self.name!r} has no Stan program")
        path = Path(directory) / f"{self.name}.stan"
        path.write_text(self.stan_program)
        return StanModel(self.name, str(path))


# ---------------------------------------------------------------------------
# gaussian_linear
# ---------------------------------------------------------------------------


def gaussian_linear(num_features: int = 3, num_observations: int = 40) -> ModelTestCase:
    """``y ~ N(X beta, I)`` with ``beta ~ N(0, 4 I)`` and a correlated design.

    The posterior is ``N(S Xᵀ y, S)`` with ``S = (I / 4 + Xᵀ X)⁻¹``. The columns
    of ``X`` have correlation 0.6, so the posterior coordinates are correlated
    and a method must recover the covariance, not only the scales.
    """
    prior_variance = 4.0
    k_x, k_beta, k_noise = jax.random.split(jax.random.PRNGKey(20260930), 3)
    correlation = 0.6 * np.ones((num_features, num_features)) + 0.4 * np.eye(num_features)
    factor = np.linalg.cholesky(correlation)
    X = jnp.asarray(np.asarray(jax.random.normal(k_x, (num_observations, num_features))) @ factor.T)
    X = X.astype(jnp.float32)
    beta_true = jax.random.normal(k_beta, (num_features,))
    y = (X @ beta_true + jax.random.normal(k_noise, (num_observations,))).astype(jnp.float32)

    X64, y64 = np.asarray(X, np.float64), np.asarray(y, np.float64)
    precision = np.eye(num_features) / prior_variance + X64.T @ X64
    cov = np.linalg.inv(precision)
    mean = cov @ X64.T @ y64
    sd = np.sqrt(np.diag(cov))
    reference = PosteriorReference(
        {"beta": _leaf(mean, np.diag(cov), lambda q: mean + sd * scipy.stats.norm.ppf(q))},
        "closed form",
    )
    prior = MultivariateNormal(
        "beta", jnp.zeros(num_features), cov=prior_variance * jnp.eye(num_features)
    )
    model = glm_likelihood("y", GaussianFamily(), X=X, dispersion=1.0) * prior

    def pymc_model() -> Any:
        import pymc as pm

        from probpipe.families import PyMCModel

        def build(y=None):
            with pm.Model() as m:
                beta = pm.MvNormal(
                    "beta",
                    mu=np.zeros(num_features),
                    cov=prior_variance * np.eye(num_features),
                )
                pm.Normal("y", mu=X64 @ beta, sigma=1.0, observed=y, shape=num_observations)
            return m

        return PyMCModel("gaussian_linear", build)

    program = """
    data {
      int<lower=0> N;
      int<lower=1> K;
      matrix[N, K] X;
      vector[N] y;
    }
    parameters {
      vector[K] beta;
    }
    model {
      beta ~ normal(0, 2);
      y ~ normal(X * beta, 1);
    }
    """
    return ModelTestCase(
        name="gaussian_linear",
        model=model,
        data={"y": y},
        reference=reference,
        tags=frozenset({"gaussian", "correlated"}),
        posterior=MultivariateNormal(
            "beta", jnp.asarray(mean, jnp.float32), cov=jnp.asarray(cov, jnp.float32)
        ),
        pymc_model=pymc_model,
        stan_program=program,
        stan_data={"N": num_observations, "K": num_features, "X": X64, "y": y64},
    )


# ---------------------------------------------------------------------------
# beta_bernoulli
# ---------------------------------------------------------------------------


def beta_bernoulli() -> ModelTestCase:
    """``theta ~ Beta(1, 1)`` and 2 successes in 13 Bernoulli trials, so ``theta | y ~ Beta(3, 12)``.

    The posterior is skewed (skewness about 0.7) and bounded, with its mode at
    about 0.15, well inside the unit interval.
    """
    successes, trials = 2, 13
    y = jnp.concatenate([jnp.ones(successes), jnp.zeros(trials - successes)]).astype(jnp.float32)
    a, b = 1.0 + successes, 1.0 + trials - successes
    posterior = scipy.stats.beta(a, b)
    reference = PosteriorReference(
        {"theta": _leaf(posterior.mean(), posterior.var(), posterior.ppf)}, "closed form"
    )
    prior = Beta("theta", 1.0, 1.0)
    likelihood = ObservationKernel(
        "y",
        {"theta": prior.event_spec.spec},
        NumericArraySpec((trials,), jnp.float32, boolean),
        lambda theta: tfd.Independent(
            tfd.Bernoulli(probs=jnp.full(trials, theta), dtype=jnp.float32), 1
        ),
    )

    def pymc_model() -> Any:
        import pymc as pm

        from probpipe.families import PyMCModel

        def build(y=None):
            with pm.Model() as m:
                theta = pm.Beta("theta", 1.0, 1.0)
                pm.Bernoulli("y", p=theta, observed=y, shape=trials)
            return m

        return PyMCModel("beta_bernoulli", build)

    program = """
    data {
      int<lower=0> N;
      array[N] int<lower=0, upper=1> y;
    }
    parameters {
      real<lower=0, upper=1> theta;
    }
    model {
      theta ~ beta(1, 1);
      y ~ bernoulli(theta);
    }
    """
    return ModelTestCase(
        name="beta_bernoulli",
        model=likelihood * prior,
        data={"y": y},
        reference=reference,
        tags=frozenset({"skewed", "constrained"}),
        posterior=Beta("theta", a, b),
        pymc_model=pymc_model,
        stan_program=program,
        stan_data={"N": trials, "y": np.asarray(y, int)},
    )


# ---------------------------------------------------------------------------
# gamma_poisson
# ---------------------------------------------------------------------------


def gamma_poisson() -> ModelTestCase:
    """``lam ~ Gamma(2, rate=1)`` and ten Poisson counts summing to 25, so ``lam | y ~ Gamma(27, 11)``."""
    counts = jnp.array([3.0, 1.0, 4.0, 2.0, 2.0, 5.0, 3.0, 0.0, 2.0, 3.0], jnp.float32)
    shape, rate = 2.0 + float(counts.sum()), 1.0 + counts.shape[0]
    posterior = scipy.stats.gamma(shape, scale=1.0 / rate)
    reference = PosteriorReference(
        {"lam": _leaf(posterior.mean(), posterior.var(), posterior.ppf)}, "closed form"
    )
    prior = Gamma("lam", 2.0, 1.0)
    n = counts.shape[0]
    likelihood = ObservationKernel(
        "y",
        {"lam": prior.event_spec.spec},
        NumericArraySpec((n,), jnp.float32, non_negative_integer),
        lambda lam: tfd.Independent(tfd.Poisson(rate=jnp.full(n, lam)), 1),
    )

    def pymc_model() -> Any:
        import pymc as pm

        from probpipe.families import PyMCModel

        def build(y=None):
            with pm.Model() as m:
                lam = pm.Gamma("lam", alpha=2.0, beta=1.0)
                pm.Poisson("y", mu=lam, observed=y, shape=n)
            return m

        return PyMCModel("gamma_poisson", build)

    program = """
    data {
      int<lower=0> N;
      array[N] int<lower=0> y;
    }
    parameters {
      real<lower=0> lam;
    }
    model {
      lam ~ gamma(2, 1);
      y ~ poisson(lam);
    }
    """
    return ModelTestCase(
        name="gamma_poisson",
        model=likelihood * prior,
        data={"y": counts},
        reference=reference,
        tags=frozenset({"skewed", "constrained", "discrete-likelihood"}),
        posterior=Gamma("lam", shape, rate),
        pymc_model=pymc_model,
        stan_program=program,
        stan_data={"N": n, "y": np.asarray(counts, int)},
    )


# ---------------------------------------------------------------------------
# dirichlet_multinomial
# ---------------------------------------------------------------------------


def dirichlet_multinomial() -> ModelTestCase:
    """``p ~ Dirichlet(1, 2, 3)`` and two multinomial draws of ten, so ``p | y ~ Dirichlet(8, 8, 10)``.

    Each coordinate's marginal posterior is ``Beta(a_i, a_0 - a_i)``, which gives
    its quantiles.
    """
    alpha = np.array([1.0, 2.0, 3.0])
    counts = jnp.array([[3.0, 5.0, 2.0], [4.0, 1.0, 5.0]], jnp.float32)
    total = 10.0
    concentration = alpha + np.asarray(counts, np.float64).sum(axis=0)
    a0 = concentration.sum()
    mean = concentration / a0
    variance = concentration * (a0 - concentration) / (a0**2 * (a0 + 1.0))
    reference = PosteriorReference(
        {
            "p": _leaf(
                mean,
                variance,
                lambda q: scipy.stats.beta.ppf(q, concentration, a0 - concentration),
            )
        },
        "closed form",
    )
    prior = Dirichlet("p", jnp.asarray(alpha, jnp.float32))
    draws, categories = counts.shape
    likelihood = ObservationKernel(
        "y",
        {"p": prior.event_spec.spec},
        NumericArraySpec((draws, categories), jnp.float32, non_negative_integer),
        lambda p: tfd.Independent(
            tfd.Multinomial(total_count=total, probs=jnp.broadcast_to(p, (draws, categories))), 1
        ),
    )

    def pymc_model() -> Any:
        import pymc as pm

        from probpipe.families import PyMCModel

        def build(y=None):
            with pm.Model() as m:
                p = pm.Dirichlet("p", a=alpha)
                pm.Multinomial("y", n=int(total), p=p, observed=y, shape=(draws, categories))
            return m

        return PyMCModel("dirichlet_multinomial", build)

    program = """
    data {
      int<lower=1> M;
      int<lower=2> K;
      vector<lower=0>[K] alpha;
      array[M, K] int<lower=0> y;
    }
    parameters {
      simplex[K] p;
    }
    model {
      p ~ dirichlet(alpha);
      for (m in 1:M) {
        y[m] ~ multinomial(p);
      }
    }
    """
    return ModelTestCase(
        name="dirichlet_multinomial",
        model=likelihood * prior,
        data={"y": counts},
        reference=reference,
        tags=frozenset({"simplex", "constrained", "discrete-likelihood"}),
        posterior=Dirichlet("p", jnp.asarray(concentration, jnp.float32)),
        pymc_model=pymc_model,
        stan_program=program,
        stan_data={"M": draws, "K": categories, "alpha": alpha, "y": np.asarray(counts, int)},
    )


# ---------------------------------------------------------------------------
# poisson_regression
# ---------------------------------------------------------------------------


def _grid_quantile(grid: np.ndarray, density: np.ndarray, q: float) -> float:
    """The quantile at *q* of the density tabulated on the uniform *grid*, by its interpolated CDF."""
    cdf = np.cumsum(density)
    cdf = (cdf - 0.5 * density) / cdf[-1]
    return float(np.interp(q, cdf, grid))


def poisson_regression(num_observations: int = 40) -> ModelTestCase:
    """``y_i ~ Poisson(exp(beta_0 + beta_1 z_i))`` with ``beta ~ N(0, I)``.

    The posterior has no closed form. It is two-dimensional, so it is tabulated
    on a 401-by-401 grid spanning eight Laplace standard deviations either side
    of the mode in each coordinate, which bounds the omitted mass far below the
    harness's Monte Carlo resolution; moments are grid sums and quantiles come
    from the interpolated marginal CDFs.
    """
    k_z, k_y = jax.random.split(jax.random.PRNGKey(7), 2)
    z = jax.random.normal(k_z, (num_observations,))
    X = jnp.stack([jnp.ones(num_observations), z], axis=1).astype(jnp.float32)
    beta_true = jnp.array([0.5, 0.4])
    y = jax.random.poisson(k_y, jnp.exp(X @ beta_true)).astype(jnp.float32)

    X64, y64 = np.asarray(X, np.float64), np.asarray(y, np.float64)

    def log_posterior(beta: np.ndarray) -> np.ndarray:
        eta = beta @ X64.T
        return (eta * y64 - np.exp(eta)).sum(-1) - 0.5 * (beta**2).sum(-1)

    def negative(beta: np.ndarray) -> float:
        return -float(log_posterior(beta[None, :])[0])

    mode = scipy.optimize.minimize(negative, np.zeros(2), method="BFGS").x
    rate = np.exp(X64 @ mode)
    hessian = X64.T @ (rate[:, None] * X64) + np.eye(2)
    scale = np.sqrt(np.diag(np.linalg.inv(hessian)))
    axes = [np.linspace(m - 8.0 * s, m + 8.0 * s, 401) for m, s in zip(mode, scale)]
    b0, b1 = np.meshgrid(*axes, indexing="ij")
    points = np.stack([b0, b1], axis=-1)
    log_density = log_posterior(points)
    weights = np.exp(log_density - log_density.max())
    weights /= weights.sum()
    mean = np.array([(weights * b0).sum(), (weights * b1).sum()])
    variance = np.array([(weights * b0**2).sum(), (weights * b1**2).sum()]) - mean**2
    marginals = [weights.sum(axis=1), weights.sum(axis=0)]

    def ppf(q: float) -> np.ndarray:
        return np.array([_grid_quantile(axes[i], marginals[i], q) for i in range(2)])

    reference = PosteriorReference({"beta": _leaf(mean, variance, ppf)}, "quadrature")
    model = glm_likelihood("y", PoissonFamily(), X=X) * MultivariateNormal(
        "beta", jnp.zeros(2), cov=jnp.eye(2)
    )

    def pymc_model() -> Any:
        import pymc as pm

        from probpipe.families import PyMCModel

        def build(y=None):
            with pm.Model() as m:
                beta = pm.MvNormal("beta", mu=np.zeros(2), cov=np.eye(2))
                pm.Poisson("y", mu=pm.math.exp(X64 @ beta), observed=y, shape=num_observations)
            return m

        return PyMCModel("poisson_regression", build)

    program = """
    data {
      int<lower=0> N;
      int<lower=1> K;
      matrix[N, K] X;
      array[N] int<lower=0> y;
    }
    parameters {
      vector[K] beta;
    }
    model {
      beta ~ normal(0, 1);
      y ~ poisson_log(X * beta);
    }
    """
    return ModelTestCase(
        name="poisson_regression",
        model=model,
        data={"y": y},
        reference=reference,
        tags=frozenset({"non-conjugate", "discrete-likelihood"}),
        pymc_model=pymc_model,
        stan_program=program,
        stan_data={"N": num_observations, "K": 2, "X": X64, "y": np.asarray(y, int)},
    )


# ---------------------------------------------------------------------------
# eight_schools
# ---------------------------------------------------------------------------

#: The eight-schools effects and their standard errors.
SCHOOL_EFFECTS = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
SCHOOL_ERRORS = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

#: The scales of the eight-schools priors: ``mu ~ N(0, 5)`` and ``tau ~ HalfCauchy(0, 5)``.
MU_SCALE, TAU_SCALE = 5.0, 5.0


@dataclass(frozen=True)
class ScaleMixture:
    """A posterior whose coefficients are Gaussian given one positive scale, tabulated over the scale.

    For a model ``y = A(s) z + e`` with ``z ~ N(0, diag(v))`` and ``e ~ N(0,
    diag(n(s)))``, the coefficients ``z`` given the scale ``s`` and ``y`` are
    Gaussian, and ``p(s | y)`` is the prior of ``s`` times the Gaussian marginal
    likelihood ``N(y; 0, A diag(v) Aᵀ + diag(n(s)))``. A grid uniform in
    ``log s`` carries the posterior mass of ``s``; a moment of ``z`` mixes the
    conditional moments over the grid, a quantile of ``z`` solves the mixed
    normal CDF, and a quantile of ``s`` interpolates its CDF.

    Attributes
    ----------
    log_scale : np.ndarray
        The grid, uniform in ``log s``.
    mass : np.ndarray
        The posterior mass of each grid point, summing to one.
    means, variances : np.ndarray
        The conditional mean and variance of each coefficient at each grid point.
    """

    log_scale: np.ndarray
    mass: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    @classmethod
    def tabulate(
        cls,
        y: np.ndarray,
        *,
        log_prior: Callable[[np.ndarray], np.ndarray],
        design: Callable[[float], np.ndarray],
        noise_variance: Callable[[float], np.ndarray],
        prior_variance: np.ndarray,
        log_scale: np.ndarray,
    ) -> ScaleMixture:
        """The mixture of the model ``y = design(s) z + e`` over the grid *log_scale*."""
        scale = np.exp(log_scale)
        log_weight = np.empty_like(scale)
        means = np.empty((scale.shape[0], prior_variance.shape[0]))
        variances = np.empty_like(means)
        for k, s in enumerate(scale):
            A, noise = design(s), noise_variance(s)
            marginal = (A * prior_variance) @ A.T + np.diag(noise)
            log_weight[k] = scipy.stats.multivariate_normal.logpdf(
                y, np.zeros(y.shape[0]), marginal
            )
            precision = np.diag(1.0 / prior_variance) + A.T @ (A / noise[:, None])
            cov = np.linalg.inv(precision)
            means[k] = cov @ A.T @ (y / noise)
            variances[k] = np.diag(cov)
        # The grid is uniform in log s, so each point carries the Jacobian s.
        log_mass = log_weight + log_prior(scale) + log_scale
        mass = np.exp(log_mass - log_mass.max())
        return cls(log_scale, mass / mass.sum(), means, variances)

    def coefficient(self, index: int) -> tuple[float, float, Callable[[float], float]]:
        """The posterior mean, variance, and quantile function of the coefficient *index*."""
        means, sd = self.means[:, index], np.sqrt(self.variances[:, index])
        mean = float(self.mass @ means)
        variance = float(self.mass @ (sd**2 + means**2)) - mean**2

        def ppf(q: float) -> float:
            def gap(x: float) -> float:
                return float(self.mass @ scipy.stats.norm.cdf((x - means) / sd)) - q

            spread = 20.0 * np.sqrt(variance)
            return scipy.optimize.brentq(gap, mean - spread, mean + spread, xtol=1e-10)

        return mean, variance, ppf

    def coefficients(
        self, indices: slice
    ) -> tuple[np.ndarray, np.ndarray, Callable[[float], np.ndarray]]:
        """The mean, variance, and quantile function of the coefficients *indices*, as vectors."""
        summaries = [self.coefficient(i) for i in range(self.means.shape[1])[indices]]
        return (
            np.array([m for m, _, _ in summaries]),
            np.array([v for _, v, _ in summaries]),
            lambda q: np.array([ppf(q) for _, _, ppf in summaries]),
        )

    def scale(self) -> tuple[float, float, Callable[[float], float]]:
        """The posterior mean, variance, and quantile function of the scale."""
        values = np.exp(self.log_scale)
        mean = float(self.mass @ values)
        variance = float(self.mass @ values**2) - mean**2
        cdf = np.cumsum(self.mass) - 0.5 * self.mass
        return mean, variance, lambda q: float(np.exp(np.interp(q, cdf, self.log_scale)))


@functools.cache
def eight_schools_quadrature() -> dict[str, Any]:
    """The eight-schools posterior of ``mu``, ``tau``, and ``theta_tilde``, as a scale mixture over ``tau``.

    Given ``tau``, the coefficients ``z = (mu, theta_tilde)`` enter linearly,
    ``y = mu 1 + tau theta_tilde + e``, so the posterior is a
    :class:`ScaleMixture` with ``A(tau) = [1 | tau I]``. The grid holds 4000
    points from ``tau = 1e-6`` to ``1e4``, beyond which the omitted mass is below
    1e-10.

    Returns
    -------
    dict
        The posterior mean, variance, and quantile function of each of ``mu``,
        ``tau``, and ``theta_tilde``.
    """
    J = SCHOOL_EFFECTS.shape[0]
    mixture = ScaleMixture.tabulate(
        SCHOOL_EFFECTS,
        log_prior=lambda tau: scipy.stats.halfcauchy.logpdf(tau, scale=TAU_SCALE),
        design=lambda tau: np.concatenate([np.ones((J, 1)), tau * np.eye(J)], axis=1),
        noise_variance=lambda tau: SCHOOL_ERRORS**2,
        prior_variance=np.concatenate([[MU_SCALE**2], np.ones(J)]),
        log_scale=np.linspace(np.log(1e-6), np.log(1e4), 4000),
    )
    return {
        "mu": mixture.coefficient(0),
        "tau": mixture.scale(),
        "theta_tilde": mixture.coefficients(slice(1, None)),
    }


def eight_schools() -> ModelTestCase:
    """The non-centered eight-schools model, with a quadrature reference.

    ``mu ~ N(0, 5)``, ``tau ~ HalfCauchy(0, 5)``, ``theta_tilde ~ N(0, I)``, and
    ``y_j ~ N(mu + tau theta_tilde_j, sigma_j)``. The posterior of ``tau`` has
    mass near zero and a heavy right tail, the regime that separates samplers
    that adapt to scale from those that do not.
    """
    J = SCHOOL_EFFECTS.shape[0]
    quadrature = eight_schools_quadrature()
    reference = PosteriorReference(
        {path: _leaf(*quadrature[path]) for path in ("mu", "tau", "theta_tilde")}, "quadrature"
    )
    sigma = jnp.asarray(SCHOOL_ERRORS, jnp.float32)
    theta_tilde = MultivariateNormal("theta_tilde", jnp.zeros(J), cov=jnp.eye(J))
    tau = HalfCauchy("tau", 0.0, TAU_SCALE)
    mu = Normal("mu", 0.0, MU_SCALE)
    likelihood = ObservationKernel(
        "y",
        {
            "mu": mu.event_spec.spec,
            "tau": tau.event_spec.spec,
            "theta_tilde": theta_tilde.event_spec.spec,
        },
        NumericArraySpec((J,), jnp.float32, real),
        lambda mu, tau, theta_tilde: tfd.Independent(
            tfd.Normal(loc=mu + tau * theta_tilde, scale=sigma), 1
        ),
    )

    def pymc_model() -> Any:
        import pymc as pm

        from probpipe.families import PyMCModel

        def build(y=None):
            with pm.Model() as m:
                mu_rv = pm.Normal("mu", 0.0, MU_SCALE)
                tau_rv = pm.HalfCauchy("tau", TAU_SCALE)
                theta_rv = pm.Normal("theta_tilde", 0.0, 1.0, shape=J)
                pm.Normal("y", mu_rv + tau_rv * theta_rv, SCHOOL_ERRORS, observed=y, shape=J)
            return m

        return PyMCModel("eight_schools", build)

    program = """
    data {
      int<lower=1> J;
      vector[J] y;
      vector<lower=0>[J] sigma;
    }
    parameters {
      real mu;
      real<lower=0> tau;
      vector[J] theta_tilde;
    }
    model {
      mu ~ normal(0, 5);
      tau ~ cauchy(0, 5);
      theta_tilde ~ normal(0, 1);
      y ~ normal(mu + tau * theta_tilde, sigma);
    }
    """
    return ModelTestCase(
        name="eight_schools",
        model=likelihood * theta_tilde * tau * mu,
        data={"y": jnp.asarray(SCHOOL_EFFECTS, jnp.float32)},
        reference=reference,
        tags=frozenset({"hierarchical", "constrained"}),
        pymc_model=pymc_model,
        stan_program=program,
        stan_data={"J": J, "y": SCHOOL_EFFECTS, "sigma": SCHOOL_ERRORS},
    )


# ---------------------------------------------------------------------------
# The set
# ---------------------------------------------------------------------------

#: Every canonical case, by name, with its builder.
CASES: Mapping[str, Callable[[], ModelTestCase]] = {
    "gaussian_linear": gaussian_linear,
    "beta_bernoulli": beta_bernoulli,
    "gamma_poisson": gamma_poisson,
    "dirichlet_multinomial": dirichlet_multinomial,
    "poisson_regression": poisson_regression,
    "eight_schools": eight_schools,
}


@functools.cache
def case(name: str) -> ModelTestCase:
    """The canonical case *name*, built once per process."""
    return CASES[name]()
