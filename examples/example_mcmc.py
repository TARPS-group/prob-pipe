"""
Example: MCMC Posterior Inference with Normal Prior and Likelihood
------------------------------------------------------------------

This example demonstrates how to use ProbPipe's built-in modules
(`Likelihood`, `Prior`, `MetropolisHastings`, and `MCMC`) to perform
posterior estimation for a simple Normal–Normal model.

Model:
    y_i ~ Normal(θ, 1)
    delta   ~ Normal(0, 5^2)

We observe noisy data generated around delta = 1.5 and infer the posterior
over delta using the Metropolis–Hastings sampler.
"""

import numpy as np
from probpipe import Likelihood, Prior, MetropolisHastings, MCMC, Normal1D
from probpipe import Module


class DistributionModule(Module):
    """Wraps a Normal1D distribution for use as a ProbPipe module dependency."""
    
    DEPENDENCIES = set()

    def __init__(self, dist):
        super().__init__()
        self.dist = dist

    @property
    def sigma(self):
        return self.dist.sigma

    def log_density(self, x):
        return self.dist.log_density(x)

    def sample(self, size=None):
        return self.dist.sample(size)
    

# Define modules for prior, likelihood, sampler, and MCMC inference 
likelihood = Likelihood(distribution=DistributionModule(Normal1D(mu=0, sigma=1)))
prior = Prior(distribution=DistributionModule(Normal1D(mu=0, sigma=5)))
sampler = MetropolisHastings()
mcmc = MCMC(likelihood=likelihood, prior=prior, sampler=sampler)

# Generate synthetic data
observed_data = np.random.normal(1.5, 1.0, size=100)

# Run MCMC to obtain posterior summary
posterior = mcmc.calculate_posterior(
    num_samples=100,
    initial_param=0.0,
    data=observed_data,
    proposal_std=0.5,       
)

print("Type of posterior:", type(posterior))
print("Posterior mean:", posterior.mu)
print("Posterior std:", posterior.sigma)
print("Posterior samples:\n", posterior.sample(10))
