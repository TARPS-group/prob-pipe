from typing import Callable

import numpy as np

from .module import Module, InputSpec
from .multivariate import Normal1D
from numpy.typing import NDArray


__all__ = [
    "Likelihood",
    "Prior",
    "MetropolisHastings",
    "MCMC",
]

class Likelihood(Module):
    """Computes the log-likelihood of observed data under a user-specified distribution.

    This module provides a general-purpose interface for evaluating the
    log-likelihood of observed data given a parameterized probabilistic model.
    The distribution is provided as an argument at runtime, allowing flexible use
    with any distribution class that implements a ``log_density`` method.

    Attributes:
        None: This module does not define persistent dependencies.
            The ``distribution`` is passed dynamically when calling the method.

    Args:
        distribution: Any object implementing a ``log_density`` method.
            Defines the probabilistic model used to evaluate data likelihoods.
        data: Observed data points.
        param: Model parameter(s) at which to evaluate the likelihood.

    Notes:
        - The ``distribution`` object must expose a ``log_density(data)`` method
          returning log-probabilities for the provided samples.
        - Suitable for use in probabilistic workflows or MCMC-based pipelines
          where both model and data vary dynamically.
    """
    
    DEPENDENCIES = {'distribution'}

    def __init__(self, **dependencies):
        """Initializes the Likelihood module.

        Args:
            **dependencies: Optional module dependencies. Typically empty.
                The likelihood distribution is provided directly at runtime
                rather than registered as a dependency.
        """
        super().__init__(**dependencies)

        self.set_input(
            data=InputSpec(type=NDArray, required=True),
            param=InputSpec(type=float, required=True),
        )
        self.run_func(self._log_likelihood_task, name="log_likelihood")

    @property
    def distribution(self):
        """Distribution object used for evaluating likelihood.

        Returns:
            The ``distribution`` instance provided at runtime.
        """
        return self.dependencies['distribution']

    def _log_likelihood_func(self, data: NDArray, param: float) -> float:
        """Computes the total log-likelihood of observed data.

        Evaluates the sum of log-probabilities for all data points under
        the specified distribution, given a particular parameter value.

        Args:
            data: Observed data, shape (n,) or (n, 1).
            param: Model parameter value(s) used for evaluation.

        Returns:
            The total log-likelihood across all observations.

        Example:
            >>> dist = MyCustomDist(theta=2.0)
            >>> mod = Likelihood()
            >>> logL = mod._log_likelihood_func(data=np.array([1.0, 2.0]), param=1.5)
        """
        temp_dist = Normal1D(mu=param, sigma=self.distribution.sigma)
        xarr = np.asarray(data, dtype=float).reshape(-1, 1)
        log_probs = temp_dist.log_density(xarr)
        return float(np.sum(log_probs))

    def _log_likelihood_task(self, *, data: NDArray, param: float) -> float:
        """Prefect-compatible task wrapper for the log-likelihood computation.

        Wraps :meth:`_log_likelihood_func` for integration with Prefect
        task-based workflows.

        Args:
            data: Observed data points.
            param: Model parameter(s) for evaluation.

        Returns:
            Total log-likelihood of the observed data.

        Example:
            >>> wf = Likelihood()
            >>> result = wf._log_likelihood_task(
            ...     data=np.random.randn(10),
            ...     param=0.0,
            ... )
            >>> print(result)
            -12.3
        """
        
        return self._log_likelihood_func(data, param)




# It bypasses the Prefect tasks entirely when doing MCMC actual sampling (which needs immediate float results).
# It only calls the pure computation methods returning floats.
# It preserves the Prefect tasks if you want to call them independently within Prefect workflows.
# This avoids the input missing errors and runtime confusion Prefect tasks incur when called like normal Python functions.




class Prior(Module):
    """Computes the log-probability (log-density) of a parameter under a prior distribution.

    This module serves as a thin wrapper around a probability distribution
    (e.g., Normal1D, Beta, etc.) to expose prior log-probabilities as
    Prefect-compatible tasks. It is typically used within probabilistic
    workflows where prior evaluation is required as part of the posterior
    computation.

    Attributes:
        DEPENDENCIES: Required dependency names. Must include
            ``'distribution'`` representing the prior distribution.
    """
    
    DEPENDENCIES = {'distribution'}

    def __init__(self, **dependencies):
        """Initializes the Prior module.

        Args:
            **dependencies: Module dependencies. Must include a key
                ``'distribution'`` mapping to a distribution instance
                (e.g., ``Normal1D`` or ``Beta``).
        """
        super().__init__(**dependencies)

        self.set_input(param=InputSpec(type=float, required=True))
        self.run_func(self._log_prob_task, name="log_prob")

    @property
    def distribution(self):
        """Distribution: The dependency representing the prior distribution."""
        return self.dependencies['distribution']

    def _log_prob_func(self, param: float) -> float:
        """Computes the log-probability of a parameter under the prior.

        Args:
            param: Parameter value at which to evaluate the log-density.

        Returns:
            Log-probability of the parameter under the prior distribution.
        """
        
        return self.distribution.log_density(param)

    def _log_prob_task(self, *, param: float):
        """Prefect-compatible task wrapper for the log-probability computation.

        Wraps :meth:`_log_prob_func` for use in Prefect workflows, enabling
        asynchronous or task-based execution.

        Args:
            param: Parameter value to evaluate.

        Returns:
            Log-probability of the parameter under the prior.
        """
        
        return self._log_prob_func(param)




class MetropolisHastings(Module):
    """Implements a basic Metropolis–Hastings (MH) sampler.

    This module performs random-walk Metropolis–Hastings sampling using
    a Normal(delta_t, proposal_std^2) proposal distribution. It provides a minimal,
    general-purpose sampler for 1D or low-dimensional targets and can be
    embedded as a dependency within higher-level MCMC workflows.

    Attributes:
        DEPENDENCIES: The set of required dependencies (empty for this module).

    Notes:
        - Stateless: does not maintain internal dependencies.
        - Expects a callable ``log_target`` that returns the log-density of a state.
        - Suitable for pedagogical or small-scale use; for high-dimensional or
          correlated targets, more advanced samplers (e.g., HMC) are recommended.
    """
    
    DEPENDENCIES = set()

    def __init__(self):
        """Initializes the Metropolis–Hastings sampler.

        The module defines required input specifications for Prefect workflows:
        ``log_target``, ``num_samples``, ``initial_state``, and optionally ``proposal_std``.
        """
        super().__init__()
        self.set_input(
            log_target=InputSpec(type=Callable, required=True),
            num_samples=InputSpec(type=int, required=True),
            initial_state=InputSpec(type=float, required=True),
            proposal_std=InputSpec(type=float, required=False, default=1.0),
        )
        self.run_func(self._sample_posterior, name="sample_posterior")
        
        
    def _sample_posterior(self, *, log_target, num_samples, initial_state, proposal_std = 1.0):
        """Draws samples from a target distribution using the MH algorithm.

        Executes a random-walk Metropolis–Hastings loop over ``num_samples`` iterations.
        Each proposed sample is drawn from a Normal(θₜ, proposal_std²) proposal, and
        accepted or rejected according to the standard acceptance ratio.

        Args:
            log_target: Function that returns the
                log-probability (log-density) of a given state.
            num_samples: Number of MCMC iterations to perform.
            initial_state: Initial value of the Markov chain.
            proposal_std: Standard deviation of the Normal proposal
                kernel. Defaults to 1.0.

        Returns:
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
    """Generic Markov Chain Monte Carlo (MCMC) posterior estimation module.

    This module combines user-defined prior, likelihood, and sampler modules
    (e.g., :class:`MetropolisHastings`) to draw samples from a posterior
    distribution p(delta | data). The resulting samples are summarized as a
    :class:`Normal1D` distribution representing the posterior mean and
    standard deviation.

    Attributes:
        DEPENDENCIES: Required module dependencies:
            ``'likelihood'``, ``'prior'``, and ``'sampler'``.
        _conv_num_samples: Default number of samples for distribution conversion.
        _conv_by_kde: Whether to use kernel density estimation for conversion.
        _conv_fit_kwargs: Extra fitting keyword arguments.
    """

    DEPENDENCIES = {'likelihood', 'prior', 'sampler'}

    def __init__(self, **dependencies):
        """Initializes the MCMC module.

        Args:
            **dependencies: Module dependencies providing prior, likelihood,
                and sampler instances. Must include keys:
                ``'prior'`` (Prior), ``'likelihood'`` (Likelihood),
                and ``'sampler'`` (MetropolisHastings or compatible sampler).
        """
        super().__init__(**dependencies)

        self._conv_num_samples = 2048
        self._conv_by_kde = False
        self._conv_fit_kwargs = {}

        self.set_input(
            num_samples=InputSpec(type=int, required=True),
            initial_param=InputSpec(type=float, required=True),
            data=InputSpec(type=NDArray, required=True),
            proposal_std=InputSpec(type=float, required=False, default=1.0),
        )

        self.run_func(
            self._calculate_posterior, 
            name="calculate_posterior",
            )

    def _calculate_posterior(self, *, num_samples, initial_param, data, proposal_std=1.0):
        """Estimates the posterior distribution via MCMC sampling.

        Runs an MCMC chain to approximate the posterior p(θ | data)
        by iteratively combining the prior and likelihood in a
        user-provided sampler (e.g., Metropolis–Hastings).
        The resulting samples are summarized by a Normal1D distribution
        using their empirical mean and standard deviation.

        Args:
            num_samples: Number of MCMC samples to draw.
            initial_param: Initial parameter value for the Markov chain.
            data: Observed data used in the likelihood function.
            proposal_std: Standard deviation for the sampler’s
                proposal kernel. Defaults to 1.0.

        Returns:
            Posterior summary distribution with mean and standard deviation
            computed from the sampled chain.
        """
        
        likelihood = self.dependencies['likelihood']
        prior = self.dependencies['prior']
        sampler = self.dependencies['sampler']

        def log_target(param):
            """Computes unnormalized log posterior for a given parameter."""
            
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
    
