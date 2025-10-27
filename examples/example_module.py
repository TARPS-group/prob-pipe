"""
Example: Gaussian Conjugate Posterior Computation
-------------------------------------------------

This example demonstrates how to define a lightweight `Module`
(`GaussianPosterior`) that analytically computes the posterior
for a Normal–Normal model:

    y_i ~ Normal(mu, sigma^2)
    mu  ~ Normal(mu_theta, var_theta)

The posterior is also Normal with parameters:

    var_n = 1 / (n / sigma^2 + 1 / var_theta)
    mu_n  = var_n * (Sigma_y / sigma^2 + mu_theta / var_theta)

The module automatically converts empirical distributions to
parametric Normals if necessary, illustrating ProbPipe’s type-safe
conversion logic.
"""

import numpy as np
from probpipe import Module, EmpiricalDistribution, Normal1D
from numpy.typing import NDArray

class GaussianPosterior(Module):
    """Posterior for a Normal–Normal conjugate model.

    Args:
        **deps: Optional dependencies (unused in this simple module).

    Attributes:
        _conv_num_samples: Number of samples used for conversion when
            fitting from an empirical prior.
        _conv_by_kde: Whether to use a KDE for conversion.
    """

    def __init__(self, **deps):
        """Initialize the GaussianPosterior module."""
        super().__init__(**deps)

        #self.set_input(
        #   prior={'type': Distribution, 'required': True},  # accept any distribution subclass
        #    y={'type': (list, NDArray), 'required': True},
        #    sigma={'type': float, 'default': 1.0},
        #)

        self._conv_num_samples = 2048
        self._conv_by_kde = False
        self._conv_fit_kwargs = {}

        # Decorate and register internal calculate_posterior method
        self.run_func(self._calculate_posterior, name="calculate_posterior", as_task=True)

    def _calculate_posterior(self, *, prior: Normal1D, y: NDArray, sigma: float = 1.0):
        """Compute the conjugate Normal posterior.

        Args:
            prior: Prior distribution over μ.
            y: Observed data array.
            sigma: Known standard deviation of the likelihood.

        Returns:
            Normal1D: Posterior distribution N(mu_n, sigma_n ^2).

        Examples:
            >>> gp = GaussianPosterior()
            >>> prior = Normal1D(0, 5)
            >>> y = np.array([1.2, 0.7, -0.3, 0.4])
            >>> post = gp.calculate_posterior(prior=prior, y=y, sigma=1.0)
            >>> post.mu, post.sigma
            (approx. 0.41, 0.47)
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

gp = GaussianPosterior()

prior_norm = Normal1D(0, 5)
post = gp.calculate_posterior(prior=prior_norm, y=np.array([1.2, 0.7, -0.3, 0.4]), sigma=1.0)
print(post.mu, post.sigma)

emp_prior = EmpiricalDistribution(samples=np.array([1.3, 2.0, 3.3, 4.1, 5.1]).reshape(-1,1))
post2 = gp.calculate_posterior(prior=emp_prior, y=np.array([1.2, 0.7, 0.3, 0.4]), sigma=1.0)
print(post2.mu, post2.sigma)
