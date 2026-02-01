from .node import wf, abstractwf, Workflow, Module, AbstractModule
from ..distributions.distribution import Distribution
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

#################################################################
### Creating and updating approximate posterior distributions ###
#################################################################

# XXX: it's awkward to have to specify T here; is there any alternative way to do this?
# XXX: no
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
        params = self.posterior.sample(n_samples=1)
        
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

    # -------------------------
    # Delegation
    # -------------------------

    def __getattr__(self, name: str):
        """
        Delegate all unknown attributes to the wrapped posterior.
        This preserves behavior like expectation(), variance(), etc.
        """
        return getattr(self._posterior, name)



class ApproximatePosterior(Workflow, ABC):
    def __init__(self, *, workflow_kind: str | None = None, name: str = "compute_posterior", **bind):
        super().__init__(func=self._compute_posterior, workflow_kind=workflow_kind, name=name, bind=bind)

    @abstractwf  
    def _compute_posterior(self, prior: Distribution, likelihood: Likelihood, data: NDArray) -> PosteriorDistribution:
        ...


class RWMH(ApproximatePosterior):
    def __init__(self, step_size: float = 1.0):
        # bind step_size into the workflow node (available as parameter if you include it in signature)
        super().__init__(step_size=step_size)
        self.step_size = step_size

    @wf
    def _compute_posterior(self, prior: Distribution, likelihood: Likelihood, data: NDArray) -> PosteriorDistribution:
        # crude: posterior approx centered at sample mean
        mu_hat = np.mean(data, axis=0)            # shape (d,)
        post_approx = Gaussian(mean=mu_hat, cov=np.eye(len(mu_hat)))  # shape (d,d)
        return PosteriorDistribution(posterior=post_approx, prior=prior, likelihood=likelihood, data=data)


class IterativeForecaster(Module):
    """Can iteratively update posterior given new data and generate predictions from posterior."""
    def __init__(self, *, prior: Distribution, likelihood: Likelihood, approx_post: ApproximatePosterior):

        # state
        self._curr_posterior = PosteriorDistribution(
            posterior=prior,
            prior=prior,
            likelihood=likelihood,
            data=np.array([]),
        )

        # auto-split: Nodes become child_nodes; non-Nodes become inputs
        super().__init__(likelihood=likelihood, approx_post=approx_post, prior=prior)

    @property
    def curr_posterior(self) -> PosteriorDistribution:
        return self._curr_posterior

    @wf
    def update(self, approx_post: ApproximatePosterior, likelihood: SimpleLikelihood, prior: Distribution, data: NDArray) -> PosteriorDistribution:
        # approx_post and likelihood and prior are resolved from module (deps/inputs) automatically
        post_dist = approx_post(prior=prior, likelihood=likelihood, data=data)
        self._curr_posterior = post_dist
        return post_dist

    @wf
    def forecast(self, n_samples: int) -> NDArray:
        return self.curr_posterior.sample_predictive(n_samples=n_samples)

#########################
### Predictive checks ###
#########################

class PredictiveChecker(AbstractModule):
    @abstractwf
    def predictive_p_value(self, posterior: PosteriorDistribution) -> float:
        ...


class PosteriorPredictiveChecker(PredictiveChecker):
    def __init__(self, statistic: Callable[[NDArray], float]):
        super().__init__(statistic=statistic)

    @wf
    def predictive_p_value(self, posterior: PosteriorDistribution, statistic: Callable[[NDArray], float]) -> float:
        obs_data = posterior.data
        obs_stat = statistic(obs_data)
        sim_stats = np.array([statistic(posterior.sample_predictive(n_samples=1)) for _ in range(200)])
        return float((sim_stats >= obs_stat).mean())

    
