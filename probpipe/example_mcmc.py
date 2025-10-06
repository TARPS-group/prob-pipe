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

print("First 10 posterior samples:", samples[:10])