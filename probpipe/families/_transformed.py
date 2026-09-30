"""The evaluation-result families: the lazy linear pushforward and the bijector transform.

Each evaluation rule returns a family of the catalog. The generic linear rule
returns a ``LinearPushforwardDistribution`` when no family-specific rule
applies, and the change-of-variables rule returns a
``BijectorTransformedDistribution``.

Provides:
  - ``LinearPushforwardDistribution`` – the law of ``op @ X`` for ``X ~ base``.
  - ``BijectorTransformedDistribution`` – the law of ``f(X)`` for an invertible
    ``f`` with a tractable Jacobian determinant.

The transformed distribution over backend bijectors is defined in
:mod:`probpipe.distributions.transformed`.
"""

from __future__ import annotations

from ..distributions._distribution import Distribution
from ..linalg import LinOp
from ..values import Function

__all__ = ["BijectorTransformedDistribution", "LinearPushforwardDistribution"]


class LinearPushforwardDistribution(Distribution):
    """The law of ``op @ X`` for ``X ~ base``, represented lazily.

    The event type is ``op``'s output type, under the pushforward's own
    component. Sampling pushes the base's draws through ``op``. The mean and
    the covariance delegate exactly, ``E[A X] = A E[X]`` and ``Cov(A X) = A
    Cov(X) Aᵀ``, lazily through the operator algebra. The log-density exists
    only when ``op`` is invertible, by change of variables.

    Parameters
    ----------
    name : str
        The pushforward's label, and the component of its event.
    base : Distribution
        The law of ``X``, whose numeric event is ``op``'s input.
    op : LinOp
        The linear map.

    Raises
    ------
    NotImplementedError
        Always, until the linear evaluation rule is implemented.
    """

    def __init__(self, name: str, base: Distribution, op: LinOp) -> None:
        raise NotImplementedError("LinearPushforwardDistribution.__init__")


class BijectorTransformedDistribution(Distribution):
    """The law of ``f(X)`` for ``X ~ base`` and an invertible map ``f`` with a Jacobian.

    Construction checks that ``bijector`` is invertible and claims the
    log-determinant of its Jacobian. Sampling pushes the base's draws through
    the bijector, and the log-density at ``y`` is the base's log-density at the
    preimage minus the log-determinant of the Jacobian there.

    Parameters
    ----------
    name : str
        The transformed law's label.
    base : Distribution
        The law of ``X``.
    bijector : Function
        The invertible map ``f``, claiming the inverse and the log-determinant
        of its Jacobian.

    Raises
    ------
    NotImplementedError
        Always, until the change-of-variables rule is implemented.
    """

    def __init__(self, name: str, base: Distribution, bijector: Function) -> None:
        raise NotImplementedError("BijectorTransformedDistribution.__init__")
