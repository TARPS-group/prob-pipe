"""StanModel: wraps Stan models as ProbPipe distributions via BridgeStan.

Uses BridgeStan for log-probability evaluation and gradients.
Inference is handled by registered methods in ``probpipe.inference``.
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp

from ..core.distribution import Distribution
from ..core.protocols import SupportsLogProb
from ..custom_types import Array, ArrayLike
from ..inference._approximate_distribution import ApproximateDistribution
from ._base import ProbabilisticModel

logger = logging.getLogger(__name__)

__all__ = ["StanModel"]


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
        self._name_str = name

        # Compile the model
        self._bs_model = bridgestan.StanModel.from_stan_file(
            stan_file, data=data or {}
        )
        self._num_params = self._bs_model.param_unc_num()

    # -- Distribution interface ---------------------------------------------

    @property
    def name(self) -> str | None:
        return self._name_str

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._num_params,)

    # -- SupportsNamedComponents interface ----------------------------------

    @property
    def component_names(self) -> tuple[str, ...]:
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

    def _log_prob(self, value: Any) -> Array:
        """Log-density in the constrained parameter space.

        Internally unconstrains the parameters before calling Stan's
        log_density.
        """
        params_constrained = jnp.asarray(value)
        params_unc = self.param_unconstrain(params_constrained)
        return jnp.float32(self._bs_model.log_density(params_unc))

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
        return jnp.asarray(
            self._bs_model.param_constrain(jnp.asarray(params_unc))
        )

    def param_unconstrain(self, params: ArrayLike) -> Array:
        """Transform constrained parameters to unconstrained space."""
        return jnp.asarray(
            self._bs_model.param_unconstrain(jnp.asarray(params))
        )

    def as_unconstrained_distribution(self) -> _UnconstrainedStanView:
        """Return a view of this model in the unconstrained parameter space."""
        return _UnconstrainedStanView(self)

    # -- BridgeStan model access (for nutpie integration) -------------------

    def _bridgestan_model(self, data: dict | None = None) -> Any:
        """Return the BridgeStan model, optionally with new data."""
        if data is not None:
            import bridgestan

            return bridgestan.StanModel.from_stan_file(
                self._stan_file, data=data
            )
        return self._bs_model

    def __repr__(self) -> str:
        return f"StanModel(stan_file={self._stan_file!r}, num_params={self._num_params})"


class _UnconstrainedStanView(Distribution[Any], SupportsLogProb):
    """View of a StanModel in the unconstrained parameter space."""

    def __init__(self, model: StanModel):
        self._model = model

    @property
    def name(self) -> str | None:
        base = self._model.name
        return f"{base}_unconstrained" if base else "unconstrained"

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._model.event_shape

    def _log_prob(self, value: Any) -> Array:
        """Log-density directly in unconstrained space."""
        params_unc = jnp.asarray(value)
        return jnp.float32(self._model._bs_model.log_density(params_unc))

    def _unnormalized_log_prob(self, value: Any) -> Array:
        return self._log_prob(value)

    def _unnormalized_prob(self, value: Any) -> Array:
        return jnp.exp(self._unnormalized_log_prob(value))

    def _prob(self, value: Any) -> Array:
        return jnp.exp(self._log_prob(value))

    def __repr__(self) -> str:
        return f"UnconstrainedStanView({self._model!r})"
