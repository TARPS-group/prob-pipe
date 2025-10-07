from core.module import module
from core.multivariate import Normal1D
from core.distributions import EmpiricalDistribution, Distribution
import numpy as np


class GaussianPosteriorModule(module):
    """
    Posterior for: y_i ~ Normal(mu, sigma^2), prior mu ~ Normal(mu0, var0)
    Returns posterior Normal1D(mu_n, var_n)
    """

    def __init__(self, **deps):
        super().__init__(**deps)

        self.set_input(
            prior={'type': Distribution, 'required': True},  # accept any distribution subclass
            y={'type': (list, np.ndarray), 'required': True},
            sigma={'type': float, 'default': 1.0},
        )

        self._conv_num_samples = 2048
        self._conv_by_kde = False
        self._conv_fit_kwargs = {}

        # Decorate and register internal calculate_posterior method
        self.run_func(self._calculate_posterior, name="calculate_posterior")

    def _calculate_posterior(self, *, prior: Distribution, y: np.ndarray, sigma: float = 1.0):
        """
        Compute posterior Normal1D given Normal likelihood with known sigma,
        prior Normal1D (mu0, var0), and data y.
        """

        # Convert y to numpy array if needed
        y = np.asarray(y, dtype=float)

        n = y.size
        var = sigma ** 2

        # Prior parameters
        mu0 = prior.mean()
        var0 = prior.cov()

        # Posterior parameters of Normal-Normal conjugacy
        var_n = 1 / (n / var + 1 / var0)
        mu_n = var_n * (y.sum() / var + mu0 / var0)

        return Normal1D(mu_n, np.sqrt(var_n))

gp = GaussianPosteriorModule()

prior_norm = Normal1D(0, 5)
post = gp.calculate_posterior(prior=prior_norm, y=np.array([1.2, 0.7, -0.3, 0.4]), sigma=1.0)
print(post.mu, post.sigma)

emp_prior = EmpiricalDistribution(samples= np.array([1.3, 2.0, 3.3, 4.1, 5.1]).reshape(-1,1) )
post2 = gp.calculate_posterior(prior=emp_prior, y=np.array([1.2, 0.7, 0.3, 0.4]), sigma=1.0)
print(post2.mu, post2.sigma)