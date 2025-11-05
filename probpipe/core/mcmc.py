from typing import Callable, ClassVar, Dict, Type, Any
from types import MappingProxyType


import numpy as np

from .module import Module, InputSpec
from .distributions import Distribution, EmpiricalDistribution
from numpy.typing import NDArray
from probpipe import Distribution


__all__ = [
    "Likelihood",
    "MetropolisHastings",
    "MCMC",
    "DistributionModule",
]

class DistributionModule(Module):
    DEPENDENCIES = MappingProxyType({})

    def __init__(self, distribution: Any):
        """
        Wraps an existing distribution instance to conform to the Module interface.

        Args:
            distribution (Any): An instance of a distribution class, such as Normal1D,
                Beta, or any custom distribution implementing required methods.
        """
        super().__init__()
        self._distribution = distribution

    def log_density(self, x):
        """Delegate to the wrapped distribution's log_density method."""
        return self._distribution.log_density(x)

    def sample(self, n_samples: int):
        """Delegate to the wrapped distribution's sample method."""
        return self._distribution.sample(n_samples)

    # Optionally expose additional distribution methods as needed
    def __getattr__(self, name):
        # Delegate attribute access to underlying distribution
        return getattr(self._distribution, name)
        


class Likelihood(Module):
    DEPENDENCIES = MappingProxyType({'distribution': Callable})  # factory returning distribution

    def __init__(self, **dependencies):
        super().__init__(**dependencies)

        self.set_input(
            data=InputSpec(type=np.ndarray, required=True),
            param=InputSpec(type=float, required=True),  # or NDArray if multi-parameter
        )

        self.run_func(self._log_likelihood_task, name="log_likelihood")

    @property
    def distribution_factory(self):
        return self.dependencies['distribution']

    def _log_likelihood_func(self, data, param):
        dist = self.distribution_factory(param)
        xarr = np.asarray(data)
        return float(np.sum(dist.log_density(xarr)))

    def _log_likelihood_task(self, *, data, param):
        return self._log_likelihood_func(data, param)
    
# It bypasses the Prefect tasks entirely when doing MCMC actual sampling (which needs immediate float results).
# It only calls the pure computation methods returning floats.
# It preserves the Prefect tasks if you want to call them independently within Prefect workflows.
# This avoids the input missing errors and runtime confusion Prefect tasks incur when called like normal Python functions.








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
    
    DEPENDENCIES = MappingProxyType({})

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

    DEPENDENCIES: ClassVar[Dict[str, Type[Module]]] = MappingProxyType({
        'likelihood': Likelihood,
        'distribution': DistributionModule,
        'sampler': MetropolisHastings,
    })


    def __init__(self, prior: Distribution, likelihood: Likelihood, sampler: MetropolisHastings, **dependencies):
        """Initializes the MCMC module.

        Args:
            prior: Prior distribution
            likelihood: Likelihood module
            sampler: Sampling module 
            **dependencies: Other module dependencies
        """
        super().__init__(likelihood=likelihood, sampler=sampler, **dependencies)
        self._prior = prior
        self._conv_num_samples = 2048
        self._conv_by_kde = False
        self._conv_fit_kwargs = {}

        self.set_input(
            num_samples=InputSpec(type=int, required=True),
            initial_param=InputSpec(type=NDArray, required=True),
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
        prior = self.dependencies['distribution']
        sampler = self.dependencies['sampler']

        def log_target(param):
            """Computes unnormalized log posterior for a given parameter."""
            
            ll = likelihood._log_likelihood_func(data, param)
            lp = prior.log_density(param)
            return ll + lp

        samples = sampler.sample_posterior(
            log_target=log_target,
            num_samples=num_samples,
            initial_state=initial_param,
            proposal_std=proposal_std,
        )

        D = initial_param.shape[0] 
        samples_array = np.asarray(samples).reshape(-1, D)
         
        return EmpiricalDistribution(samples=samples_array)
    
