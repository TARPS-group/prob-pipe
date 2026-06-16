"""StanModel: wraps Stan models as ProbPipe distributions via BridgeStan.

Uses BridgeStan for log-probability evaluation and gradients.
Inference is handled by registered methods in ``probpipe.inference``.
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..core.distribution import Distribution
from ..core.protocols import SupportsLogProb
from ..custom_types import Array, ArrayLike
from ._base import ProbabilisticModel

logger = logging.getLogger(__name__)

__all__ = ["StanModel"]


def _to_f64(x: ArrayLike) -> np.ndarray:
    """Coerce *x* to the contiguous ``float64`` ndarray BridgeStan requires.

    BridgeStan's ctypes interface rejects anything that is not a NumPy
    ``float64`` array — a JAX array fails the ``ndarray`` check, and a
    ``float32`` array fails the dtype check — so every value crossing into
    ``param_constrain`` / ``param_unconstrain`` / ``log_density`` passes
    through here first.
    """
    return np.asarray(x, dtype=np.float64)


def _pack_parameters_value(owner: str, field_kwargs: dict[str, Any]) -> Array:
    """Shared ``_pack_value`` for the Stan keyword form (Tier 1).

    ``StanModel._log_prob`` (and the unconstrained view) consume a single
    flat parameter vector, so the keyword form takes one ``parameters=``
    argument rather than per-Stan-parameter fields. (Tier 2 — issue #228 —
    will expose the individual Stan parameters as fields built from
    BridgeStan's ``param_unc_names()``.) *owner* names the class in the
    error message.
    """
    if set(field_kwargs) != {"parameters"}:
        raise TypeError(
            f"{owner} keyword form takes a single 'parameters=' flat array; "
            f"got {sorted(field_kwargs)}."
        )
    return field_kwargs["parameters"]


class StanModel(ProbabilisticModel, SupportsLogProb):
    """Stan model via BridgeStan and CmdStanPy.

    Uses BridgeStan for log-probability evaluation and JAX-compatible
    gradients.  Posterior sampling (via ``condition_on``) uses
    CmdStanPy's interface to Stan's native NUTS sampler.

    Parameters are in the constrained space by default; use
    :meth:`as_unconstrained_distribution` for the unconstrained
    parameterization.

    Parameters
    ----------
    stan_file : str
        Path to a ``.stan`` file.
    data : dict or None
        Stan data dictionary.  Can also be provided at conditioning
        time.
    name : str or None
        Model name for provenance.

    Raises
    ------
    ImportError
        If ``bridgestan`` is not installed.
    """

    def __init__(
        self,
        stan_file: str,
        *,
        data: dict | None = None,
        name: str | None = None,
    ):
        try:
            import bridgestan
        except ImportError as e:
            raise ImportError(
                "bridgestan is required for StanModel. "
                "Install it with: pip install bridgestan"
            ) from e

        self._stan_file = stan_file
        self._stan_data = data
        # ``Distribution`` metaclass requires a non-empty name; default
        # to the class name when the caller doesn't supply one.
        self._name = name if name else "StanModel"

        # Compile and instantiate. BridgeStan's constructor takes the
        # ``.stan`` path directly (compiling on demand) and serializes a
        # data dict via stanio; ``data=None`` means no data.
        self._bs_model = bridgestan.StanModel(stan_file, data=data)
        self._num_params = self._bs_model.param_unc_num()

    # -- Distribution interface ---------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._num_params,)

    # -- Named components interface ------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        return self.parameter_names

    def __getitem__(self, key: str) -> Any:
        if key in self.parameter_names:
            return key  # placeholder — Stan doesn't expose sub-distributions
        raise KeyError(f"Unknown component: {key!r}")

    # -- ProbabilisticModel interface ---------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self._bs_model.param_names())

    # -- SupportsLogProb interface ------------------------------------------

    def _pack_value(self, **field_kwargs: Any) -> Array:
        """Keyword form (Tier 1): a single flat constrained ``parameters=``
        array (see :func:`_pack_parameters_value`)."""
        return _pack_parameters_value(type(self).__name__, field_kwargs)

    def _log_prob(self, value: Any) -> Array:
        """Log-density in the constrained parameter space.

        Internally unconstrains the parameters before calling Stan's
        log_density.
        """
        params_unc = self._bs_model.param_unconstrain(_to_f64(value))
        return jnp.asarray(self._bs_model.log_density(params_unc))

    def _unnormalized_log_prob(self, value: Any) -> Array:
        """Delegates to _log_prob (Stan provides the full joint)."""
        return self._log_prob(value)

    def _unnormalized_prob(self, value: Any) -> Array:
        return jnp.exp(self._unnormalized_log_prob(value))

    def _prob(self, value: Any) -> Array:
        return jnp.exp(self._log_prob(value))

    # -- Constrained / unconstrained transformations ------------------------

    def param_constrain(self, params_unc: ArrayLike) -> Array:
        """Transform unconstrained parameters to constrained space."""
        return jnp.asarray(self._bs_model.param_constrain(_to_f64(params_unc)))

    def param_unconstrain(self, params: ArrayLike) -> Array:
        """Transform constrained parameters to unconstrained space."""
        return jnp.asarray(self._bs_model.param_unconstrain(_to_f64(params)))

    def as_unconstrained_distribution(self) -> _UnconstrainedStanView:
        """Return a view of this model in the unconstrained parameter space."""
        return _UnconstrainedStanView(self)

    # -- BridgeStan model access (for nutpie integration) -------------------

    def _bridgestan_model(self, data: dict | None = None) -> Any:
        """Return the BridgeStan model, optionally with new data."""
        if data is not None:
            import bridgestan

            return bridgestan.StanModel(self._stan_file, data=data)
        return self._bs_model

    def __repr__(self) -> str:
        return f"StanModel(stan_file={self._stan_file!r}, num_params={self._num_params})"


class _UnconstrainedStanView(Distribution[Any], SupportsLogProb):
    """View of a StanModel in the unconstrained parameter space."""

    def __init__(self, model: StanModel):
        self._model = model
        # ``base.name`` is guaranteed non-empty by the Distribution
        # metaclass — wrap with an ``_unconstrained`` suffix.
        self._name = f"{model.name}_unconstrained"

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._model.event_shape

    def _pack_value(self, **field_kwargs: Any) -> Array:
        """Keyword form (Tier 1): a single flat ``parameters=`` array in the
        unconstrained space (see :func:`_pack_parameters_value`)."""
        return _pack_parameters_value(type(self).__name__, field_kwargs)

    def _log_prob(self, value: Any) -> Array:
        """Log-density directly in unconstrained space."""
        return jnp.asarray(self._model._bs_model.log_density(_to_f64(value)))

    def _unnormalized_log_prob(self, value: Any) -> Array:
        return self._log_prob(value)

    def _unnormalized_prob(self, value: Any) -> Array:
        return jnp.exp(self._unnormalized_log_prob(value))

    def _prob(self, value: Any) -> Array:
        return jnp.exp(self._log_prob(value))

    def __repr__(self) -> str:
        return f"UnconstrainedStanView({self._model!r})"
