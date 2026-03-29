"""StanModel: wraps Stan models as ProbPipe distributions via BridgeStan.

Uses BridgeStan for log-probability evaluation and gradients, and
CmdStanPy for posterior sampling via Stan's native NUTS sampler.
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp

from ..core.distribution import Distribution, Provenance
from ..core.protocols import SupportsLogProb
from ..custom_types import Array, ArrayLike
from ..inference._diagnostics import MCMCDiagnostics
from ..inference._mcmc_distribution import MCMCApproximateDistribution
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

    # -- SupportsConditionableComponents interface --------------------------

    @property
    def conditionable_components(self) -> dict[str, bool]:
        # Stan models condition via data passed at construction or conditioning
        return {"data": True}

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

    # -- Conditioning -------------------------------------------------------

    def _condition_on(self, observed: Any, /, **kwargs: Any) -> MCMCApproximateDistribution:
        """Condition on observed data using Stan's NUTS sampler.

        Parameters
        ----------
        observed : dict
            Stan data dictionary with observed values.
        **kwargs
            Sampling parameters passed to :func:`_cmdstanpy_condition`.
        """
        return _cmdstanpy_condition(
            self._stan_file,
            data={**(self._stan_data or {}), **(observed if isinstance(observed, dict) else {})},
            model_ref=self,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"StanModel(stan_file={self._stan_file!r}, num_params={self._num_params})"


# ---------------------------------------------------------------------------
# CmdStanPy helpers
# ---------------------------------------------------------------------------


def _ensure_cmdstanpy():
    """Import cmdstanpy or raise a helpful error."""
    try:
        import cmdstanpy
        return cmdstanpy
    except ImportError as e:
        raise ImportError(
            "cmdstanpy is required for Stan sampling. "
            "Install it with: pip install probpipe[stan]"
        ) from e


def _cmdstanpy_condition(
    stan_file: str,
    *,
    data: dict,
    model_ref: Any,
    num_results: int = 1000,
    num_warmup: int = 1000,
    num_chains: int = 4,
    random_seed: int = 0,
    **kwargs: Any,
) -> MCMCApproximateDistribution:
    """Run Stan's NUTS sampler and return an MCMCApproximateDistribution."""
    cmdstanpy = _ensure_cmdstanpy()

    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    fit = model.sample(
        data=data,
        chains=num_chains,
        iter_sampling=num_results,
        iter_warmup=num_warmup,
        seed=random_seed,
        show_console=False,
        **kwargs,
    )

    # Extract per-chain draws
    chains = []
    for c in range(num_chains):
        chain_draws = jnp.asarray(
            fit.draws(concat_chains=False)[c], dtype=jnp.float32,
        )
        chains.append(chain_draws)

    diagnostics = MCMCDiagnostics(
        log_accept_ratio=jnp.zeros(num_results * num_chains),
        step_size=0.0,
        is_accepted=None,
        algorithm="cmdstan_nuts",
    )

    result = MCMCApproximateDistribution(
        chains,
        diagnostics=diagnostics,
        name="posterior",
    )
    result.with_source(
        Provenance(
            "cmdstan_sample",
            parents=(model_ref,),
            metadata={
                "num_results": num_results,
                "num_warmup": num_warmup,
                "num_chains": num_chains,
            },
        )
    )
    return result


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
