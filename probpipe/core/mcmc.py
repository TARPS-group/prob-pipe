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
    """Computes the log-likelihood of observed data under a Normal1D model.

    This module evaluates the total log-likelihood of observed data given
    a specified Normal1D distribution. The distribution is provided directly
    as an argument rather than as a module dependency, ensuring that this
    module remains self-contained and flexible for use with different
    likelihood formulations.

    The variance (sigma) is taken from the provided ``distribution``, while
    the input parameter ``param`` serves as the candidate mean (mu) of a
    temporary Normal1D distribution used to compute likelihood values.

    Attributes:
        None: This module does not maintain persistent dependencies.

    Args:
        distribution (Normal1D): The reference Normal1D distribution providing
            the fixed variance (sigma) used in likelihood evaluation.
        data (NDArray): Observed data points.
        param (float): Candidate mean parameter (mu) to evaluate the likelihood.

    Notes:
        - The ``distribution`` argument supplies the fixed variance term (σ^2)
          but is not stored as a dependency.
        - The input ``param`` value defines the mean for a temporary Normal1D
          constructed during log-likelihood computation.
    """
    
    DEPENDENCIES = {'distribution'}

    def __init__(self, **dependencies):
        """Initializes the Likelihood module.

        Args:
            **dependencies: Module dependencies. Must include a key
                ``'distribution'`` that maps to a ``Normal1D`` instance.
        """
        super().__init__(**dependencies)

        self.set_input(
            data=InputSpec(type=NDArray, required=True),
            param=InputSpec(type=float, required=True),
        )
        self.run_func(self._log_likelihood_task, name="log_likelihood")

    @property
    def distribution(self):
        """Normal1D: The distribution dependency used for evaluating likelihood."""
        return self.dependencies['distribution']

    def _log_likelihood_func(self, data: NDArray, param: float) -> float:
        """Computes the total log-likelihood of observed data.

        Creates a temporary Normal1D distribution using ``param`` as the
        mean (mu) and the dependency’s sigma. Then computes the sum of
        log-likelihood values for all observed data points.

        Args:
            data (NDArray): Observed data values, of shape (n,) or (n, 1).
            param (float): Mean parameter (mu) of the temporary Normal1D.

        Returns:
            float: Total log-likelihood of the data given the parameter.
        """
        temp_dist = Normal1D(mu=param, sigma=self.distribution.sigma)
        xarr = np.asarray(data, dtype=float).reshape(-1, 1)
        log_probs = temp_dist.log_density(xarr)
        return float(np.sum(log_probs))

    def _log_likelihood_task(self, *, data: NDArray, param: float):
        """Prefect-compatible task wrapper for the log-likelihood function.

        This task wraps the pure computation function
        :meth:`_log_likelihood_func` to make it compatible with Prefect
        workflows.

        Args:
            data (NDArray): Observed data values.
            param (float): Mean parameter (mu) for the Normal1D likelihood.

        Returns:
            float: Log-likelihood of the data under the given parameter.
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
        DEPENDENCIES (set[str]): Required dependency names. Must include
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
            param (float): Parameter value at which to evaluate the log-density.

        Returns:
            float: Log-probability of the parameter under the prior distribution.
        """
        
        return self.distribution.log_density(param)

    def _log_prob_task(self, *, param: float):
        """Prefect-compatible task wrapper for the log-probability computation.

        Wraps :meth:`_log_prob_func` for use in Prefect workflows, enabling
        asynchronous or task-based execution.

        Args:
            param (float): Parameter value to evaluate.

        Returns:
            float: Log-probability of the parameter under the prior.
        """
        
        return self._log_prob_func(param)




class MetropolisHastings(Module):
    """Implements a basic Metropolis–Hastings (MH) sampler.

    This module performs random-walk Metropolis–Hastings sampling using
    a Normal(delta_t, proposal_std^2) proposal distribution. It provides a minimal,
    general-purpose sampler for 1D or low-dimensional targets and can be
    embedded as a dependency within higher-level MCMC workflows.

    Attributes:
        DEPENDENCIES (set[str]): The set of required dependencies (empty for this module).

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
            log_target (Callable[[float], float]): Function that returns the
                log-probability (log-density) of a given state.
            num_samples (int): Number of MCMC iterations to perform.
            initial_state (float): Initial value (θ₀) of the Markov chain.
            proposal_std (float, optional): Standard deviation of the Normal proposal
                kernel. Defaults to 1.0.

        Returns:
            list[float]: Sequence of sampled states approximating the target distribution.
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
        DEPENDENCIES (set[str]): Required module dependencies:
            ``'likelihood'``, ``'prior'``, and ``'sampler'``.
        _conv_num_samples (int): Default number of samples for distribution conversion.
        _conv_by_kde (bool): Whether to use kernel density estimation for conversion.
        _conv_fit_kwargs (dict): Extra fitting keyword arguments.
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
            num_samples (int): Number of MCMC samples to draw.
            initial_param (float): Initial parameter value for the Markov chain.
            data (NDArray): Observed data used in the likelihood function.
            proposal_std (float, optional): Standard deviation for the sampler’s
                proposal kernel. Defaults to 1.0.

        Returns:
            Normal1D: Posterior summary distribution with mean and standard deviation
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
    
