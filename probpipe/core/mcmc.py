from typing import Callable

import numpy as np

from .module import Module, InputSpec
from .multivariate import Normal1D

__all__ = [
    "Likelihood",
    "Prior",
    "MetropolisHastings",
    "MCMC",
]

class Likelihood(Module):
    """
    Basic likelihood module that computes the log-likelihood of observed data
    under a Normal1D distribution with a variable mean parameter.

    Notes
    -----
    - Expects a dependency named 'distribution' (a Normal1D instance).
    - The 'param' input is used as the mean (mu) for a temporary Normal1D with
      the same sigma as the dependency.
    """
    
    DEPENDENCIES = {'distribution'}

    def __init__(self, **dependencies):
        super().__init__(**dependencies)

        self.set_input(
            data=InputSpec(type=np.ndarray, required=True),
            param=InputSpec(type=float, required=True),
        )
        self.run_func(self._log_likelihood_task, name="log_likelihood")

    @property
    def distribution(self):
        """Return the distribution dependency used for evaluating likelihood."""
        return self.dependencies['distribution']

    def _log_likelihood_func(self, data: np.ndarray, param: float) -> float:
        """
        Compute the total log-likelihood of data under a Normal1D model.

        Parameters
        ----------
        data : np.ndarray
            Observed data values, shape (n,) or (n, 1).
        param : float
            Mean parameter (μ) for the Normal1D likelihood.

        Returns
        -------
        float
            Sum of log-likelihood values over all observations.
        """
        temp_dist = Normal1D(mu=param, sigma=self.distribution.sigma)
        xarr = np.asarray(data, dtype=float).reshape(-1, 1)
        log_probs = temp_dist.log_density(xarr)
        return float(np.sum(log_probs))

    def _log_likelihood_task(self, *, data: np.ndarray, param: float):
        """
        Prefect-compatible task wrapper for `_log_likelihood_func`.

        Parameters
        ----------
        data : np.ndarray
            Observed data values.
        param : float
            Mean parameter (μ) for the Normal1D likelihood.

        Returns
        -------
        float
            Log-likelihood of the data under the given parameter.
        """
        
        return self._log_likelihood_func(data, param)




# It bypasses the Prefect tasks entirely when doing MCMC actual sampling (which needs immediate float results).
# It only calls the pure computation methods returning floats.
# It preserves the Prefect tasks if you want to call them independently within Prefect workflows.
# This avoids the input missing errors and runtime confusion Prefect tasks incur when called like normal Python functions.




class Prior(Module):
    """
    Basic prior module that computes the log-probability (log-density)
    of a parameter under a given distribution.

    Notes
    -----
    - Expects a dependency named ``distribution`` (e.g., Normal1D, Beta, etc.).
    - Serves as a simple wrapper to expose prior log-probabilities as
      Prefect-compatible tasks within probabilistic workflows.
    """
    
    DEPENDENCIES = {'distribution'}

    def __init__(self, **dependencies):
        super().__init__(**dependencies)

        self.set_input(param=InputSpec(type=float, required=True))
        self.run_func(self._log_prob_task, name="log_prob")

    @property
    def distribution(self):
        """Return the distribution dependency representing the prior."""
        return self.dependencies['distribution']

    def _log_prob_func(self, param: float) -> float:
        """
        Compute the log-probability of a parameter under the prior.

        Parameters
        ----------
        param : float
            Parameter value at which to evaluate the log-density.

        Returns
        -------
        float
            Log-probability of the parameter under the prior distribution.
        """
        
        return self.distribution.log_density(param)

    def _log_prob_task(self, *, param: float):
        """
        Prefect-compatible task wrapper for `_log_prob_func`.

        Parameters
        ----------
        param : float
            Parameter value.

        Returns
        -------
        float
            Log-probability of the parameter under the prior.
        """
        
        return self._log_prob_func(param)




class MetropolisHastings(Module):
    """
    Simple Metropolis–Hastings (MH) sampler module.

    Performs random-walk MH sampling using a Normal(θ_t, proposal_std²) proposal.
    This module wraps a basic algorithm suitable for 1D or low-dimensional targets
    and can be embedded as a dependency of higher-level MCMC modules.

    Notes
    -----
    - Stateless: does not maintain internal dependencies.
    - Expects a callable `log_target` returning log-density values.
    """
    
    DEPENDENCIES = set()

    def __init__(self):
        super().__init__()
        self.set_input(
            log_target=InputSpec(type=Callable, required=True),
            num_samples=InputSpec(type=int, required=True),
            initial_state=InputSpec(type=float, required=True),
            proposal_std=InputSpec(type=float, required=False, default=1.0),
        )
        self.run_func(self._sample_posterior, name="sample_posterior")
        
        
    def _sample_posterior(self, *, log_target, num_samples, initial_state, proposal_std = 1.0):
        """
        Draw samples from a target distribution using the
        Metropolis–Hastings algorithm.

        Parameters
        ----------
        log_target : Callable[[float], float]
            Function returning the log-probability of a proposed state.
        num_samples : int
            Number of MCMC iterations to perform.
        initial_state : float
            Starting value for the Markov chain.
        proposal_std : float, default=1.0
            Standard deviation of the Normal proposal kernel.

        Returns
        -------
        list of float
            Sequence of sampled states approximating the target distribution.
        """
        
        samples = []
        current = initial_state
        current_log_prob = log_target(current)

        for _ in range(num_samples):
            proposal = np.random.normal(current, proposal_std)
            prop_log_prob = log_target(proposal)
            log_accept_ratio = prop_log_prob - current_log_prob

            if np.log(np.random.uniform()) < log_accept_ratio:
                current = proposal
                current_log_prob = prop_log_prob

            samples.append(current)
        return samples

        


class MCMC(Module):
    """
    Generic MCMC posterior estimation module.

    Combines user-specified prior, likelihood, and sampler modules
    (e.g., Metropolis–Hastings) to generate samples from a posterior
    distribution p(θ | data) and summarize them as a Normal1D.

    Notes
    -----
    - Dependencies: ``likelihood``, ``prior``, and ``sampler``.
    - Operates with automatic distribution conversion when applicable.
    - Returns a Normal1D summarizing the posterior mean and standard deviation.
    """

    DEPENDENCIES = {'likelihood', 'prior', 'sampler'}

    def __init__(self, **dependencies):
        super().__init__(**dependencies)

        self._conv_num_samples = 2048
        self._conv_by_kde = False
        self._conv_fit_kwargs = {}

        self.set_input(
            num_samples=InputSpec(type=int, required=True),
            initial_param=InputSpec(type=float, required=True),
            data=InputSpec(type=np.ndarray, required=True),
            proposal_std=InputSpec(type=float, required=False, default=1.0),
        )

        self.run_func(
            self._calculate_posterior, 
            name="calculate_posterior",
            )

    def _calculate_posterior(self, *, num_samples, initial_param, data, proposal_std=1.0):
        """
        Run MCMC to estimate the posterior distribution over parameters.

        Parameters
        ----------
        num_samples : int
            Number of MCMC samples to draw.
        initial_param : float
            Initial parameter value for the Markov chain.
        data : np.ndarray
            Observed data used in the likelihood.
        proposal_std : float, default=1.0
            Standard deviation for the proposal kernel used by the sampler.

        Returns
        -------
        Normal1D
            Gaussian summary of the posterior with mean and standard deviation
            computed from the sampled chain.
        """
        
        likelihood = self.dependencies['likelihood']
        prior = self.dependencies['prior']
        sampler = self.dependencies['sampler']

        def log_target(param):
            # Call pure functions, NOT Prefect tasks, to get floats synchronously
            ll = likelihood._log_likelihood_func(data, param)
            lp = prior._log_prob_func(param)
            return ll + lp

        samples = sampler.sample_posterior(
            log_target=log_target,
            num_samples=num_samples,
            initial_state=initial_param,
            proposal_std=proposal_std,
        )

        samples_array = np.asarray(samples).reshape(-1, 1)
         
        return Normal1D(mu = np.mean(samples_array), sigma = np.std(samples_array))    # or    EmpiricalDistribution(samples=samples_array)
    
