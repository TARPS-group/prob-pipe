from core.module import module
from core.distributions import Distribution
import numpy as np
from core.multivariate import Normal1D
from core.distributions import EmpiricalDistribution
from core.mcmc import LikelihoodModule, PriorModule, MetropolisHastingsModule, MCMCModule
from typing import Callable, Any, Dict, Optional



likelihood = LikelihoodModule(distribution = Normal1D(mu=0, sigma=1))
prior = PriorModule(distribution = Normal1D(mu=0, sigma=5))
sampler = MetropolisHastingsModule()


mcmc = MCMCModule(likelihood=likelihood, prior=prior, sampler=sampler)


observed_data = np.random.normal(1.5, 1.0, size=100)

# Run MCMC to get samples
samples = mcmc.calculate_posterior(
    num_samples=10,
    initial_param=0.0,
    data=observed_data,
    proposal_std=0.5,
)

print(samples)


# Requesting Normal1D via module call with return_type:
posterior_norm = mcmc.calculate_posterior(
    num_samples=100,
    initial_param=0.0,
    data=observed_data,
    proposal_std=0.5,
    return_type = Normal1D, # this will trigger conversion 
)

print(type(posterior_norm))  # <class 'Normal1D'>
print(posterior_norm.mean(), posterior_norm.cov())  # mean and std dev of posterior