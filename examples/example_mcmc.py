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
import scipy as sp
from probpipe import Likelihood, MetropolisHastings, MCMC, Normal1D
from probpipe import Module


# Define modules for prior, likelihood, sampler, and MCMC inference 
likelihood = Likelihood(lambda param, data: sp.stats.norm.logpdf(data, loc=param, scale=1).sum())
prior_dist=Normal1D(mu=0, sigma=5)
sampler = MetropolisHastings()
mcmc = MCMC(likelihood=likelihood, prior=prior_dist, sampler=sampler)

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
