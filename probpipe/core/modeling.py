from .node import wf, abstractwf, Workflow, Module, AbstractModule
from ..distributions.distribution import Distribution, EmpiricalDistribution
from ..distributions.real_vector.gaussian import Gaussian

from typing import Any, Type, TypeVar, Callable
from numpy.typing import NDArray
import numpy as np
from abc import ABC


__all__ = [
    "Likelihood",
    "GenerativeLikelihood",
    "SimpleLikelihood",
    "PosteriorDistribution",
    "ApproximatePosterior",
    "RWMH",
    "IterativeForecaster",
    "PredictiveChecker",
    "PosteriorPredictiveChecker",
]

TDistribution = TypeVar("TDistribution", bound=Distribution)

class Likelihood(AbstractModule):
    """Abstract module node for computing the likelihood of a model given data and parameters"""
    @abstractwf
    def log_likelihood(self,  params: NDArray, data: NDArray) -> float:
        pass


class GenerativeLikelihood(AbstractModule):
    """Abstract module node for generating synthetic data given parameters."""
    @abstractwf
    def generate_data(self, params: NDArray, n_samples: int) -> NDArray:
        pass

class SimpleLikelihood(Likelihood, GenerativeLikelihood):
    """A simple likelihood module that wraps a Distribution class."""
    def __init__(self, dist_cls: Type[TDistribution], params_name: str, **other_params):
        super().__init__()  # no shared deps required anymore!
        self.dist_cls = dist_cls
        # XXX: need to validate params_name and other_params match with dist_cls requirements
        self.params_name = params_name
        self.other_params = other_params

    def _get_distribution_for_params(self, params: NDArray) -> Distribution:
        dist_params = dict(self.other_params)
        dist_params[self.params_name] = params
        mean = dist_params["mean"]
        cov = dist_params["cov"]
        #print("mean.shape:", np.asarray(mean).shape, "cov.shape:", np.asarray(cov).shape)
        return self.dist_cls(**dist_params)

    @wf
    def log_likelihood(self, params: NDArray, data: NDArray) -> float:
        dist = self._get_distribution_for_params(params)
        return float(dist.log_density(data).sum())

    @wf
    def generate_data(self, params: NDArray, n_samples: int) -> NDArray:
        dist = self._get_distribution_for_params(params)
        return dist.sample(n_samples=n_samples)


# Creating and updating approximate posterior distributions ###

# It is necessary to specify T this way.
T = TypeVar("T", bound=np.number)

class PosteriorDistribution(Distribution[T]):
    def __init__(self, posterior: Distribution[T], prior: Distribution[T], likelihood: Likelihood, data: NDArray):
        self.__class__.__name__ = posterior.__class__.__name__  # not sure if this is the best way!
        self.posterior = posterior
        self.prior = prior
        self.likelihood = likelihood
        self.data = data

    def log_density(self, x: NDArray) -> NDArray[np.floating]:
        # NOTE: likelihood.log_likelihood is a Workflow once wrapped, so calling it works
        return self.prior.log_density(x) + self.likelihood.log_likelihood(params=x, data=self.data)

    def density(self, x: NDArray) -> NDArray[np.floating]:
        return np.exp(self.log_density(x))

    def sample(self, n_samples: int) -> NDArray:
        return self.posterior.sample(n_samples=n_samples)

    def predictive_log_density(self, x: NDArray) -> NDArray[np.floating]:
        return self.likelihood.log_likelihood(params=x, data=self.data)

    def sample_predictive(self, n_samples: int) -> NDArray:
        
        params = np.asarray(self.posterior.sample(n_samples=1))
        if params.ndim == 2 and params.shape[0] == 1:
            params = params[0]
        if params.ndim != 1:
            raise ValueError(f"Expected posterior sample to be shape (d,) or (1,d), got {params.shape}")

        return self.likelihood.generate_data(params=params, n_samples=n_samples)

    def expectation(self, func: Callable[[NDArray[T]], NDArray]) -> Distribution[T]:
        return self.posterior.expectation(func)

    @classmethod
    def from_distribution(cls, convert_from: Distribution, **fit_kwargs: Any) -> Distribution[T]:
        raise NotImplementedError

    # Delegation
    def __getattr__(self, name: str):
        """
        Delegate all unknown attributes to the wrapped posterior.
        This preserves behavior like expectation(), variance(), etc.
        """
        return getattr(self.posterior, name)
    
    def __str__(self):
        return f"{self.posterior}"



class ApproximatePosterior(Workflow, ABC):
    def __init__(self, *, workflow_kind: str | None = None, name: str = "compute_posterior", **bind):
        super().__init__(func=self._compute_posterior, workflow_kind=workflow_kind, name=name, bind=bind)

    @abstractwf  
    def _compute_posterior(self, prior: Distribution, likelihood: Likelihood, data: NDArray) -> PosteriorDistribution:
        ...


class RWMH(ApproximatePosterior):
    def __init__(
        self,
        step_size: float = 1.0,
        n_steps: int = 10_000,
        burn_in: int = 2_000,
        thin: int = 5,
        init: NDArray | None = None,
        rng: np.random.Generator | None = None,
        workflow_kind: str | None = None,
    ):
        # Binding parameters into workflow node state
        super().__init__(
            workflow_kind=workflow_kind,
            step_size=step_size,
            n_steps=n_steps,
            burn_in=burn_in,
            thin=thin,
            init=init,
            rng=rng,
        )
        self.step_size = float(step_size)
        self.n_steps = int(n_steps)
        self.burn_in = int(burn_in)
        self.thin = int(thin)
        self.init = None if init is None else np.asarray(init, dtype=float)
        self.rng = rng or np.random.default_rng()

        if self.n_steps <= 0:
            raise ValueError("n_steps must be > 0")
        if self.burn_in < 0:
            raise ValueError("burn_in must be >= 0")
        if self.thin <= 0:
            raise ValueError("thin must be > 0")

    @wf
    def _compute_posterior(
        self,
        prior: Distribution,
        likelihood: Likelihood,
        data: NDArray,
    ) -> PosteriorDistribution:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"Expected data shape (n, d), got {data.shape}")
        d = data.shape[1]

        # target: unnormalized log posterior 
        def log_post(mu: NDArray) -> float:
            mu = np.asarray(mu, dtype=float)
            if mu.shape != (d,):
                raise ValueError(f"Expected params shape {(d,)}, got {mu.shape}")

            lp = prior.log_density(mu)
            lp_val = float(np.asarray(lp).sum())  # more robust if prior returns vector

            ll_val = float(likelihood.log_likelihood(params=mu, data=data))
            return lp_val + ll_val

        # initialize 
        if self.init is not None:
            mu_curr = self.init.copy()
            if mu_curr.shape != (d,):
                raise ValueError(f"init must have shape {(d,)}, got {mu_curr.shape}")
        else:
            # trying prior.mean if it exists, else taking data mean
            mu_curr = None
            if hasattr(prior, "mean"):
                try:
                    m = getattr(prior, "_mean")
                    m = np.asarray(m, dtype=float)
                    if m.shape == (d,):
                        mu_curr = m.copy()
                except Exception:
                    mu_curr = None
            if mu_curr is None:
                mu_curr = np.mean(data, axis=0).astype(float)

        logp_curr = log_post(mu_curr)

        # RWMH loop: mu' = mu + step_size * N(0, I)
        kept = []
        accepts = 0

        burn_in = min(self.burn_in, self.n_steps)

        for t in range(self.n_steps):
            mu_prop = mu_curr + self.step_size * self.rng.normal(size=d)
            logp_prop = log_post(mu_prop)

            log_alpha = logp_prop - logp_curr
            if np.log(self.rng.random()) < min(0.0, log_alpha):
                mu_curr = mu_prop
                logp_curr = logp_prop
                accepts += 1

            if t >= burn_in and ((t - burn_in) % self.thin == 0):
                kept.append(mu_curr.copy())

        if len(kept) == 0:
            raise ValueError("No samples retained; increase n_steps or reduce burn_in/thin.")

        samples = np.vstack(kept)  # (n_kept, d)

        # Represent the posterior as an EmpiricalDistribution
        post_approx = EmpiricalDistribution(x=samples, rng=self.rng)

        # This is optional: store diagnostics on the instance (not required, but handy)
        self.accept_rate = accepts / float(self.n_steps)

        return PosteriorDistribution(
            posterior=post_approx,
            prior=prior,
            likelihood=likelihood,
            data=data,
        )


class IterativeForecaster(Module):
    """Can iteratively update posterior given new data and generate predictions from posterior."""
    def __init__(self, *, prior: Distribution, likelihood: Likelihood, approx_post: ApproximatePosterior, workflow_kind: str | None = None):

        # Infer dimensionality from prior by sampling once
        try:
            sample = prior.sample(n_samples=1)
            sample = np.asarray(sample)
            if sample.ndim == 2:
                d = sample.shape[1]
            elif sample.ndim == 1:
                d = sample.shape[0]
            else:
                d = 1
            init_data = np.empty((0, d))
        except Exception:
            init_data = np.array([])  # fallback

        # state
        self._curr_posterior = PosteriorDistribution(
            posterior=prior,
            prior=prior,
            likelihood=likelihood,
            data=init_data,
        )

        # auto-split: Nodes become child_nodes; non-Nodes become inputs
        super().__init__(likelihood=likelihood, approx_post=approx_post, prior=prior, workflow_kind=workflow_kind)

    @property
    def curr_posterior(self) -> PosteriorDistribution:
        return self._curr_posterior

    @wf
    def update(self, approx_post: ApproximatePosterior, likelihood: Likelihood, data: NDArray) -> PosteriorDistribution:
        # Use only the posterior approximation as prior, not the full PosteriorDistribution
        # This ensures proper Bayesian updating without double-counting likelihoods
        prior_dist = self.curr_posterior.posterior
        post_dist = approx_post(prior=prior_dist, likelihood=likelihood, data=data)
        self._curr_posterior = post_dist
        return post_dist

    @wf
    def forecast(self, n_samples: int = 0) -> NDArray:
        return self.curr_posterior.sample_predictive(n_samples=n_samples)

# Predictive checks 

class PredictiveChecker(AbstractModule):
    @abstractwf
    def predictive_p_value(self, posterior: PosteriorDistribution) -> float:
        ...


class PosteriorPredictiveChecker(PredictiveChecker):
    """
    Perform a posterior predictive check (PPC) and compute a posterior
    predictive p-value for a scalar test statistic.

    This class compares a statistic computed on the observed data to the
    same statistic computed on datasets simulated from the posterior
    predictive distribution.

    For multivariate data (d > 1), ``statistic`` is applied independently
    to each dimension (column-wise), producing a vector of shape ``(d,)``.
    The ``reducer`` then collapses that vector into a single scalar so that
    observed and simulated statistics are compared on the same scale.

    For univariate data (d = 1), the statistic is already scalar and the
    reducer has no practical effect.

    The returned value estimates:

        p_ppc = P(T(y_rep) >= T(y_obs) | y_obs)

    where y_rep is drawn from the posterior predictive distribution.

    Args:
        statistic:
            A function ``T(x: NDArray) -> float`` applied column-wise to the data.
            Examples: ``np.mean``, ``np.std``, ``np.max``.

        n_rep:
            Number of posterior predictive replicate datasets.
            Larger values reduce Monte Carlo variability in the estimate.
            Default: 200.

        n_samples:
            Number of predictive draws used to construct each synthetic
            replicate dataset.

            If set to 0 (default), the observed dataset size is used,
            ensuring each replicated dataset matches the original sample
            size for a well-calibrated PPC.

            If positive, exactly that many predictive draws are used.

        reducer:
            A function ``r(v: NDArray[shape=(d,)]) -> float`` that collapses
            the per-dimension statistic vector to a scalar.
            Default: ``np.mean`` (average across dimensions).
            Example alternative: ``np.max`` to emphasize worst-case misfit.

    Returns:
        float:
            The posterior predictive p-value. Values near 0 or 1 suggest
            potential model misfit; values near 0.5 indicate consistency
            between the model and observed data.
    """
    def __init__(
        self,
        statistic: Callable[[NDArray], float],
        n_rep: int = 200,
        n_samples: int = 0,
        reducer: Callable[[NDArray], float] = lambda v: float(np.mean(v)),
        workflow_kind: str | None = None,
    ):
        super().__init__(
            statistic=statistic, 
            n_rep=n_rep, 
            n_samples=n_samples, 
            reducer=reducer, 
            workflow_kind=workflow_kind
        )

    @wf
    def predictive_p_value(
        self,
        posterior: PosteriorDistribution,
        statistic: Callable[[NDArray], float],
        n_samples: int,
        n_rep: int,
        reducer: Callable[[NDArray], float],
    ) -> float:
        if n_samples < 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if n_rep <= 0:
            raise ValueError(f"n_rep must be positive, got {n_rep}")

        obs_data = posterior.data
        if obs_data.size == 0:
            raise ValueError("Cannot compute PPC with empty observed data")
        
        # Match observed dataset size if n_samples == 0
        if n_samples == 0:
            n_samples = obs_data.shape[0]

        # per-dimension observed stats, then reduce to scalar
        obs_vec = np.apply_along_axis(statistic, 0, obs_data)   # shape (d,) or scalar
        # Handle both scalar and vector outputs
        if np.ndim(obs_vec) == 0:
            obs_stat = float(obs_vec)
        else:
            obs_stat = float(reducer(obs_vec))

        sim_stats = np.empty(int(n_rep), dtype=float)
        for r in range(int(n_rep)):
            samples = posterior.sample_predictive(n_samples=int(n_samples))  # (n_samples, d)
            sim_vec = np.apply_along_axis(statistic, 0, samples)             # (d,) or scalar
            # Handle both scalar and vector outputs
            if np.ndim(sim_vec) == 0:
                sim_stats[r] = float(sim_vec)
            else:
                sim_stats[r] = float(reducer(sim_vec))

        return float((sim_stats >= obs_stat).mean())
