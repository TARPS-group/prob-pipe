from typing import Optional, Callable
from probpipe import WorkFlow, Module 
from prefect import flow, task
import numpy as np
from probpipe.core.multivariate import Normal1D

# Example usage of WorkFlow
with WorkFlow() as wf:

    @wf.run_func(as_task=True)
    def log_likelihood(data: np.ndarray, param: float, sigma: float) -> float:
        residuals = data - param
        # Log likelihood of normal model (ignoring constants)
        return -0.5 * np.sum((residuals / sigma) ** 2)

    @wf.run_func(as_task=True)
    def log_prior(param: float, mu_prior: float, sigma_prior: float) -> float:
        diff = param - mu_prior
        # Log prior (normal)
        return -0.5 * (diff / sigma_prior) ** 2

    @wf.run_func(as_task=True)
    def metropolis_hastings(
        log_target: Callable[[float], float],
        num_samples: int,
        initial_state: float,
        proposal_std: float = 1.0
    ):
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

    @wf.run_func(as_task=True)
    def calculate_posterior(
        data: np.ndarray,
        num_samples: int,
        initial_param: float,
        mu_prior: float,
        sigma_prior: float,
        sigma_likelihood: float,
        proposal_std: float = 1.0,
    ):
        def log_target(param: float) -> float:
            ll = wf.log_likelihood(data=data, param=param, sigma=sigma_likelihood)  # Prefect calls task
            lp = wf.log_prior(param=param, mu_prior=mu_prior, sigma_prior=sigma_prior)  # Prefect calls task
            return ll + lp

        samples = wf.metropolis_hastings(
            log_target=log_target,
            num_samples=num_samples,
            initial_state=initial_param,
            proposal_std=proposal_std,
        )

        samples_array = np.array(samples).reshape(-1, 1)
        mu_post = np.mean(samples_array)
        sigma_post = np.std(samples_array)
        return Normal1D(mu=mu_post, sigma=sigma_post)

# After exiting the `with` block, all run functions are registered on wf

print("Running MCMC posterior calculation...")
posterior_dist = wf.calculate_posterior(
    data=np.random.normal(3, 1, size=100),
    num_samples=2000,
    initial_param=0.0,
    mu_prior=0.0,
    sigma_prior=5.0,
    sigma_likelihood=1.0,
    proposal_std=0.5,
)

print(f"Posterior mean: {posterior_dist.mu:.3f}, Posterior std: {posterior_dist.sigma:.3f}")