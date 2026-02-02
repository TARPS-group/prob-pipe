EmpiricalDistribution 
try:
    from probpipe.core.distributions import EmpiricalDistribution
except Exception:
    try:
        from probpipe.distributions.distribution import EmpiricalDistribution
    except Exception:
        EmpiricalDistribution = None


# Normal1D 
try:
    from probpipe.core.multivariate import Normal1D
except Exception:
    try:
        from probpipe.distributions.multivariate import Normal1D
    except Exception:
        Normal1D = None


# MvNormal
try:
    from probpipe.core.multivariate import MvNormal
except Exception:
    try:
        from probpipe.distributions.real_vector.gaussian import Gaussian
    except Exception:
        Gaussian = None

# Gaussian 
try:
    from probpipe.distributions.real_vector.gaussian import Gaussian
except Exception:
    Gaussian = None
