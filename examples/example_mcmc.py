import numpy as np
from probpipe import Likelihood, Prior, MetropolisHastings, MCMC, Normal1D


likelihood = Likelihood(distribution=Normal1D(mu=0, sigma=1))
prior = Prior(distribution=Normal1D(mu=0, sigma=5))
sampler = MetropolisHastings()
mcmc = MCMC(likelihood=likelihood, prior=prior, sampler=sampler)

observed_data = np.random.normal(1.5, 1.0, size=100)

posterior = mcmc.calculate_posterior(
    num_samples=100,
    initial_param=0.0,
    data=observed_data,
    proposal_std=0.5,       
)

print("Type of posterior:", type(posterior))
print("First 10 posterior samples:", posterior.sample(10))