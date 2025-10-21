from typing import Callable

import numpy as np

from .module import Module, InputSpec
from .multivariate import Normal1D

__all__ = [
    "Likelihood",
    "Prior",
    "MetropolisHastings",
    "MCMC",
]

class Likelihood(Module):
    DEPENDENCIES = {'distribution'}

    def __init__(self, **dependencies):
        super().__init__(**dependencies)

        self.set_input(
            data=InputSpec(type=np.ndarray, required=True),
            param=InputSpec(type=float, required=True),
        )
        self.run_func(self._log_likelihood_task, name="log_likelihood")

    @property
    def distribution(self):
        return self.dependencies['distribution']

    def _log_likelihood_func(self, data: np.ndarray, param: float) -> float:
        temp_dist = Normal1D(mu=param, sigma=self.distribution.sigma)
        xarr = np.asarray(data, dtype=float).reshape(-1, 1)
        log_probs = temp_dist.log_density(xarr)
        return float(np.sum(log_probs))

    def _log_likelihood_task(self, *, data: np.ndarray, param: float):
        return self._log_likelihood_func(data, param)




# It bypasses the Prefect tasks entirely when doing MCMC actual sampling (which needs immediate float results).
# It only calls the pure computation methods returning floats.
# It preserves the Prefect tasks if you want to call them independently within Prefect workflows.
# This avoids the input missing errors and runtime confusion Prefect tasks incur when called like normal Python functions.




class Prior(Module):
    DEPENDENCIES = {'distribution'}

    def __init__(self, **dependencies):
        super().__init__(**dependencies)

        self.set_input(param=InputSpec(type=float, required=True))
        self.run_func(self._log_prob_task, name="log_prob")

    @property
    def distribution(self):
        return self.dependencies['distribution']

    def _log_prob_func(self, param: float) -> float:
        return self.distribution.log_density(param)

    def _log_prob_task(self, *, param: float):
        # Prefect task wrapper calls pure function
        return self._log_prob_func(param)




class MetropolisHastings(Module):
    DEPENDENCIES = set()

    def __init__(self):
        super().__init__()
        self.set_input(
            log_target=InputSpec(type=Callable, required=True),
            num_samples=InputSpec(type=int, required=True),
            initial_state=InputSpec(type=float, required=True),
            proposal_std=InputSpec(type=float, required=False, default=1.0),
        )
        self.run_func(self._sample_posterior, name="sample_posterior")
        
        
    def _sample_posterior(self, *, log_target, num_samples, initial_state, proposal_std = 1.0):
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
    DEPENDENCIES = {'likelihood', 'prior', 'sampler'}

    def __init__(self, **dependencies):
        super().__init__(**dependencies)

        self._conv_num_samples = 2048
        self._conv_by_kde = False
        self._conv_fit_kwargs = {}

        self.set_input(
            num_samples=InputSpec(type=int, required=True),
            initial_param=InputSpec(type=float, required=True),
            data=InputSpec(type=np.ndarray, required=True),
            proposal_std=InputSpec(type=float, required=False, default=1.0),
        )

        self.run_func(
            self._calculate_posterior, 
            name="calculate_posterior",
            )

    def _calculate_posterior(self, *, num_samples, initial_param, data, proposal_std=1.0):
        likelihood = self.dependencies['likelihood']
        prior = self.dependencies['prior']
        sampler = self.dependencies['sampler']

        def log_target(param):
            # Call pure functions, NOT Prefect tasks, to get floats synchronously
            ll = likelihood._log_likelihood_func(data, param)
            lp = prior._log_prob_func(param)
            return ll + lp

        samples = sampler.sample_posterior(
            log_target=log_target,
            num_samples=num_samples,
            initial_state=initial_param,
            proposal_std=proposal_std,
        )

        samples_array = np.asarray(samples).reshape(-1, 1)
         
        return Normal1D(mu = np.mean(samples_array), sigma = np.std(samples_array))    # or    EmpiricalDistribution(samples=samples_array)
    