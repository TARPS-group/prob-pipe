"""
Example: MCMC Posterior Inference with Normal Prior and Likelihood
------------------------------------------------------------------

This example demonstrates how to use ProbPipe's built-in modules
(`Likelihood`, `Prior`, `MetropolisHastings`, and `MCMC`) along with a
general distribution wrapped by `DistributionModule`.

Model:
    y_i ~ Normal(θ, 1)
    delta   ~ Normal(0, 5^2)

We observe noisy data generated around delta = 1.5 and infer the posterior
over delta using the Metropolis–Hastings sampler.
"""

import numpy as np
import scipy as sp
from probpipe import Likelihood, MetropolisHastings, MCMC, Normal1D, Module, DistributionModule


# Generate synthetic data
observed_data = np.random.normal(1.5, 1.0, size=100)

prior_dist = DistributionModule(Normal1D(mu=0.0, sigma=1.0))
likelihood = Likelihood(distribution = prior_dist)
sampler = MetropolisHastings()

mcmc = MCMC(likelihood=likelihood, distribution=prior_dist, sampler=sampler)

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
