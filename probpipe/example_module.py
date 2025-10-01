from core.module import module, InputSpec
from core.multivariate import Normal1D
from core.distributions import EmpiricalDistribution
import numpy as np



class GaussianPosteriorModule(module):
    """
    Posterior for: y_i ~ Normal(mu, sigma^2), prior mu ~ Normal(mu0, var0)
    Returns posterior Normal1D(mu_n, var_n)
    """

    # no required dependencies for this example, we you could do:
    # REQUIRED_DEPS = frozenset({"lik"})  # and later inject 'lik' by name

    def __init__(self, **deps):
        super().__init__(**deps)
        # declare required inputs
        self.set_input(
            prior=InputSpec(type=Normal1D, required=True),
            y=InputSpec(type=np.ndarray, required=True),      # observed data (1D)
            sigma=InputSpec(type=float, default=1.0),         # known obs std
        )

        # sensible defaults for conversion behaviour (user can override on module init)
        self._conv_num_samples = 2048
        self._conv_by_kde = False
        self._conv_fit_kwargs = {}


gp = GaussianPosteriorModule()

@gp.run_func  # wrapped as Prefect flow by default (as_task=False)
def run(prior: Normal1D, y: np.ndarray, sigma: float = 1.0) -> Normal1D:
    """
    If 'prior' isn’t a Normal1D, the decorator will convert it using Normal1D.from_distribution.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.size
    if n == 0:
        raise ValueError("y must contain at least one observation")

    mu0 = float(prior.mu)   # adjust attribute names if different in your class
    var0 = float(prior.sigma) ** 2   # prior variance
    s2 = float(sigma) ** 2    # known observation variance

    # conjugate update: posterior var = 1 / (1/var0 + n/s2)
    var_n = 1.0 / (1.0/var0 + n / s2)
    # posterior mean = var_n * (mu0/var0 + (sum y)/s2)
    mu_n  = var_n * (mu0/var0 + y.sum()/s2)

    return Normal1D(mu_n, var_n)


# Case A: passing a Normal1D directly
prior_ok = Normal1D(0.0, 5.0)
y = np.array([1.2, 0.7, -0.3, 0.4])
post = gp.run(prior=prior_ok, y=y, sigma=1.0)
print("Posterior (from Normal1D):", post.mean, post.sigma)

# Case B: passing an EmpiricalDistribution => should auto-convert to Normal1D via from_distribution
emp = EmpiricalDistribution(samples=np.array([-0.2, 0.0, 0.3, 0.1, -0.1]))  
post2 = gp.run(prior=emp, y=y, sigma=1.0)
print("Posterior (from Empirical → converted):", post2.mu, post2.sigma)



