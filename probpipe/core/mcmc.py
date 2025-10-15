from core.module import module
import numpy as np
from core.multivariate import Normal1D
from core.distributions import EmpiricalDistribution
from typing import Callable



class LikelihoodModule(module):
    REQUIRED_DEPS = frozenset(['distribution'])

    def __init__(self, **dependencies):
        super().__init__(required_deps=self.REQUIRED_DEPS, **dependencies)
        self.set_input(
            data={'type': (list, np.ndarray), 'required': True},
            param={'type': float, 'required': True},
        )

        def log_likelihood(data: np.ndarray, param: float):
            dist: Normal1D = self.dependencies['distribution']
            # If param is not Normal1D instance but convertible, your type_check will handle conversion
            temp_dist = Normal1D(mu=param, sigma=dist.sigma)
            xarr = np.asarray(data, dtype=float).reshape(-1, 1)
            log_probs = temp_dist.log_density(xarr)
            return float(np.sum(log_probs))

        self.log_likelihood = self.run_func(log_likelihood)





class PriorModule(module):
    REQUIRED_DEPS = {'distribution'}

    def __init__(self, **dependencies):
        super().__init__(required_deps=self.REQUIRED_DEPS, **dependencies)
        self.set_input(param={'type': float, 'required': True})

        def log_prob(param):
            dist: Normal1D = self.dependencies['distribution']
            return dist.log_density(param)

        self.log_prob = self.run_func(log_prob)





class MetropolisHastingsModule(module):
    def __init__(self):
        super().__init__()
        self.set_input(
            log_target={'type': Callable, 'required': True},
            num_samples={'type': int, 'required': True},
            initial_state={'type': float, 'required': True},
            proposal_std={'type': float, 'default': 1.0},
        )

        def sample_posterior(log_target, num_samples, initial_state, proposal_std=1.0):
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

        self.sample_posterior = self.run_func(sample_posterior)


class MCMCModule(module):
    REQUIRED_DEPS = {'likelihood', 'prior', 'sampler'}

    def __init__(self, **dependencies):
        super().__init__(required_deps=self.REQUIRED_DEPS, **dependencies)
        self.set_input(
            num_samples={'type': int, 'required': True},
            initial_param={'type': float, 'required': True},
            data={'type': (list, np.ndarray), 'required': True},
            proposal_std={'type': float, 'default': 1.0},
            return_type={'type': str, 'default': 'empirical'},  # optional, for user override
        )

        def calculate_posterior(num_samples, initial_param, data, proposal_std=1.0, return_type='empirical'):
            likelihood = self.dependencies['likelihood']
            prior = self.dependencies['prior']
            sampler = self.dependencies['sampler']

            def log_target(param):
                return likelihood.log_likelihood(data=data, param=param) + prior.log_prob(param)

            samples = sampler.sample_posterior(
                log_target=log_target,
                num_samples=num_samples,
                initial_state=initial_param,
                proposal_std=proposal_std,
            )
            samples_array = np.asarray(samples).reshape(-1, 1)  # shape (n_samples, 1)
            return EmpiricalDistribution(samples=samples_array) 
     

        self.run_func(
            calculate_posterior,
            name="calculate_posterior",
            return_type=Normal1D,  # This enables output auto-conversion inside the wrapper
        )

