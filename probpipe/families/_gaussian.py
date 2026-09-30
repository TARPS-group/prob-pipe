"""The Gaussian algebra: the factored Gaussian joint and the Gaussian random functions.

Three families form a closed algebra built on ``LinOp``: the parametric
``MultivariateNormal``, the random-function member ``GaussianRandomFunction``,
and the factored joint ``FactoredMultivariateGaussian``. An affine
pushforward of a member is again a member, and conditioning a Gaussian prior on
a linear-Gaussian observation is exact.

Provides:
  - ``FactoredMultivariateGaussian`` – the factored joint of jointly Gaussian
    factors, which ``*`` and ``joint`` derive.
  - ``GaussianProcess`` – the random function specified by a mean function and
    a covariance kernel.

``MultivariateNormal`` is defined in :mod:`probpipe.distributions.multivariate`,
and ``GaussianRandomFunction`` and ``LinearBasisFunction`` in
:mod:`probpipe.distributions.gaussian_random_function`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from ..distributions._factored import FactoredNumericDistribution
from ..distributions.gaussian_random_function import GaussianRandomFunction

if TYPE_CHECKING:
    from ..core._specs import OutputSpec
    from ..custom_types import Array
    from ..distributions._conditional import ConditionalDistribution
    from ..distributions._distribution import Distribution

__all__ = ["FactoredMultivariateGaussian", "GaussianProcess"]


class FactoredMultivariateGaussian(FactoredNumericDistribution):
    """The factored joint whose factors are jointly Gaussian.

    ``*`` and ``joint`` derive it as the most specific class whenever every
    factor is a Gaussian or a linear-Gaussian conditional distribution, so it is
    never constructed by hand. Its log-density, moments, and sampling are in
    closed form, its conditioning and marginals are exact, and its pushforward
    to the flat coordinates is a ``MultivariateNormal``.

    Parameters
    ----------
    name : str
        The joint's label.
    factors : Sequence[Distribution | ConditionalDistribution]
        The jointly Gaussian factors, in conditional-first order.

    Raises
    ------
    NotImplementedError
        Always, until the Gaussian algebra is implemented.
    """

    def __init__(
        self,
        name: str,
        factors: Sequence[Distribution | ConditionalDistribution],
        *,
        _scope: Mapping[str, int] | None = None,
    ) -> None:
        raise NotImplementedError("FactoredMultivariateGaussian.__init__")


class GaussianProcess(GaussianRandomFunction):
    """The Gaussian random function specified by a mean function and a covariance kernel.

    Its finite-dimensional law at stacked inputs ``X`` has the mean
    ``mean_fn(X)`` and the covariance ``cov_kernel(X, X)``. The drawn
    function's output component and the function-valued event's component both
    default to the process's ``name``; ``output_spec`` names the former and
    ``event_spec`` the latter.

    Parameters
    ----------
    name : str
        The process's label.
    mean_fn : Callable[[Array], Array]
        The mean function, evaluated at stacked input points.
    cov_kernel : Callable[[Array, Array], Array]
        The covariance kernel, evaluated at two stacks of input points.
    output_spec : OutputSpec, optional
        The declaration of the drawn function's output.
    event_spec : OutputSpec, optional
        The declaration of one draw, a function.

    Raises
    ------
    NotImplementedError
        Always, until the Gaussian process is implemented.
    """

    def __init__(
        self,
        name: str,
        mean_fn: Callable[[Array], Array],
        cov_kernel: Callable[[Array, Array], Array],
        *,
        output_spec: OutputSpec | None = None,
        event_spec: OutputSpec | None = None,
    ) -> None:
        raise NotImplementedError("GaussianProcess.__init__")

    def predict_mean(self, X: Array) -> Array:
        """The mean function at the stacked input points *X*."""
        raise NotImplementedError("GaussianProcess.predict_mean")

    def predict_variance(self, X: Array) -> Array:
        """The marginal variance at each stacked input point of *X*."""
        raise NotImplementedError("GaussianProcess.predict_variance")
