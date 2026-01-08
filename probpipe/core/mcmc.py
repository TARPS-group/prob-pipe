from typing import Dict, Set, Any, Type, TypeVar, Callable
from probpipe.core.workflow_node import WorkflowNode
from probpipe.core.node import Node, wf
from probpipe.core.workflow_node import WorkflowNode
from probpipe.core.module_node import ModuleNode
import inspect
from typing import get_type_hints
from numpy.typing import NDArray
import numpy as np
from abc import ABC, abstractmethod
from probpipe.core.distributions import Distribution, EmpiricalDistribution



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

class Likelihood(ABC):
    """Abstract module node for (1) computing the likelihood of a model given data and parameters
    and (2) generating synthetic data given parameters.
    """
    # XXX: do abstactmethod and wf play nicely together?
    @abstractmethod
    def log_likelihood(self,  params: NDArray, data: NDArray) -> float:
        pass


class GenerativeLikelihood(ABC):
    @abstractmethod
    def generate_data(self, params: NDArray, n_samples: int) -> NDArray:
        pass


class SimpleLikelihood(ModuleNode, Likelihood, GenerativeLikelihood):
    """A simple likelihood module that wraps a Distribution class."""
    def __init__(self, dist_cls: Type[TDistribution], params_name: str, **other_params):
        super().__init__()
        self.dist_cls = dist_cls
        # XXX: need to validate params_name and other_params match with dist_cls requirements
        self.params_name = params_name
        self.other_params = other_params

    def _get_distribution_for_params(self, params: NDArray) -> Distribution:
        dist_params = dict(self.other_params)
        dist_params[self.params_name] = params
        return self.dist_cls(**dist_params)

    @wf
    def log_likelihood(self, params: NDArray, data: NDArray) -> float:
        dist = self._get_distribution_for_params(params)
        return dist.log_density(data).sum()
    
    @wf
    def generate_data(self, params: NDArray, n_samples: int) -> NDArray:
        dist = self._get_distribution_for_params(params)
        return dist.sample(n_samples=n_samples)

#################################################################
### Creating and updating approximate posterior distributions ###
#################################################################

# XXX: it's awkward to have to specify T here; is there any alternative way to do this?
T = TypeVar("T", bound=np.number)
### Wrapper around Distribution object that tracks data, prior, and likelihood
class PosteriorDistribution(Distribution[T]):
    """
    Wrapper around a posterior Distribution that augments it with
    prior, likelihood, and observed data.
    """

    def __init__(
        self,
        posterior: Distribution[T],
        prior: Distribution[T],
        likelihood: Likelihood,
        data: NDArray,
    ):
        self._posterior = posterior
        self.prior = prior
        self.likelihood = likelihood
        self.data = data

    # -------------------------
    # Core Distribution methods
    # -------------------------

    def log_density(self, params: NDArray) -> NDArray[np.floating]:
        return (
            self.prior.log_density(params)
            + self.likelihood.log_likelihood(params=params, data=self.data)
        )

    def density(self, params: NDArray) -> NDArray[np.floating]:
        return np.exp(self.log_density(params))

    def sample(self, n_samples: int) -> NDArray:
        return self._posterior.sample(n_samples=n_samples)

    # -------------------------
    # Predictive methods
    # -------------------------

    def expectation(self, func: Callable[[NDArray[T]], NDArray]) -> Distribution[T]:
        return self.posterior.expectation(func)
    
    def sample_predictive(self, n_samples: int) -> NDArray:
        # XXX: actually should generate n_samples from posterior, then generate one sample from likelihood for each posterior sample
        return self.likelihood.generate_data(params=self.posterior.sample(n_samples=1), n_samples=n_samples)

    def sample_predictive(self, n_samples: int) -> NDArray:
        # Correct Bayesian posterior predictive:
        # θᵢ ~ p(θ | data)
        # yᵢ ~ p(y | θᵢ)
        theta = self._posterior.sample(n_samples=n_samples)
        return self.likelihood.generate_data(params=theta, n_samples=1)

    # -------------------------
    # Delegation
    # -------------------------

    def __getattr__(self, name: str):
        """
        Delegate all unknown attributes to the wrapped posterior.
        This preserves behavior like expectation(), variance(), etc.
        """
        return getattr(self._posterior, name)



class ApproximatePosterior(WorkflowNode, ABC):
    def __init__(self):
        super().__init__(
            func=self.compute,  
            workflow_kind="task",
            name=type(self).__name__,
        )

    @abstractmethod
    def compute(
        self,
        prior: Distribution,
        likelihood: Likelihood,
        data: NDArray,
    ) -> PosteriorDistribution:
        pass


class RWMH(ApproximatePosterior):
    """Computes the posterior distribution using a Random Walk Metropolis-Hastings algorithm with a multivariate Gaussian proposal."""
    def __init__(self, step_size: float = 1):
        super().__init__()
        self.step_size = step_size

    def _compute_posterior(self, prior: Distribution, likelihood: Likelihood, data: NDArray) -> PosteriorDistribution:
        # XXX: implement RWMH algorithm here
        post_approx = EmpiricalDistribution(samples=np.array([]))  # XXX: placeholder
        return PosteriorDistribution(posterior=post_approx, prior=prior, likelihood=likelihood, data=data) 


class IterativeForecaster(ModuleNode):
    """Can iteratively update posterior given new data and generate predictions from posterior."""
    def __init__(self, prior: Distribution, likelihood: Likelihood, **kwargs):
        # XXX: not quite right but this is the general idea 
        super().__init__(child_nodes=dict(likelihood=likelihood), inputs={}, **kwargs)
        self._curr_posterior = PosteriorDistribution(posterior=prior, prior=prior, likelihood=likelihood, data=np.array([]))

    def current_posterior(self) -> PosteriorDistribution:
        return self._curr_posterior

    @wf
    def update(
        self,
        approx_post: ApproximatePosterior,
        likelihood: Likelihood,
        data: NDArray,
    ) -> Distribution:
        post_dist = approx_post(prior=self.curr_posterior, likelihood=likelihood, data=data)
        self._curr_posterior = post_dist
        return post_dist

    @wf
    def forecast(
        self, 
        n_samples: int) -> NDArray:
        return self.curr_posterior.sample_predictive(n_samples=n_samples)


#########################
### Predictive checks ###
#########################

class PredictiveChecker(ModuleNode):
    def __init__(self, statistic: Callable[[NDArray], float]):
        super().__init__(child_nodes={}, inputs={})
        self.statistic = statistic

    @abstractmethod
    @wf
    def predictive_p_value(
        self, 
        posterior: PosteriorDistribution,
    ) -> float:
        pass

class PosteriorPredictiveChecker(PredictiveChecker):
    @wf
    def predictive_p_value(self, posterior: PosteriorDistribution) -> float:
        obs_data = posterior.data
        obs_stat = self.statistic(obs_data)
        sim_stats = np.array([self.statistic(posterior.sample_predictive(n_samples=1)) for _ in range(1000)])
        return (sim_stats >= obs_stat).mean()