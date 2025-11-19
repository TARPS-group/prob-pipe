try:
    from core.distributions import EmpiricalDistribution
    from core.multivariate import Normal1D, MvNormal
except ImportError:
    # Fallback 
    try:
        from distributions.distribution import EmpiricalDistribution
        from distributions.real_vector.gaussian import Gaussian
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Could not import distribution classes. Checked:\n"
            " - probpipe.core.distributions\n"
            " - probpipe.distributions.distribution\n"
            " - probpipe.core.multivariate\n"
            " - probpipe.distributions.real_vector.gaussian"
        )
