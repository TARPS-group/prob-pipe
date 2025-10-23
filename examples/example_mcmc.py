import numpy as np
from probpipe import Likelihood, Prior, MetropolisHastings, MCMC, Normal1D
from probpipe import Module


class DistributionModule(Module):
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
    


likelihood = Likelihood(distribution=DistributionModule(Normal1D(mu=0, sigma=1)))
prior = Prior(distribution=DistributionModule(Normal1D(mu=0, sigma=5)))
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