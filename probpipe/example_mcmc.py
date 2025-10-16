import numpy as np
from core.module import module
from core.multivariate import Multivariate, Normal1D
from core.dist_utils import _as_2d, _symmetrize_spd, _clip_unit_interval, _to_1d_vector
from core.distributions import Distribution
from core.mcmc import LikelihoodModule, PriorModule, MetropolisHastingsModule, MCMCModule






likelihood = LikelihoodModule(distribution=Normal1D(mu=0, sigma=1))
prior = PriorModule(distribution=Normal1D(mu=0, sigma=5))
sampler = MetropolisHastingsModule()
mcmc = MCMCModule(likelihood=likelihood, prior=prior, sampler=sampler)

observed_data = np.random.normal(1.5, 1.0, size=100)

posterior = mcmc.calculate_posterior(
    num_samples=100,
    initial_param=0.0,
    data=observed_data,
    proposal_std=0.5,       
)

print("Type of posterior:", type(posterior))
print("First 10 posterior samples:", posterior.sample(10))

