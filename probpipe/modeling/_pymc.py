"""PyMCModel: wraps PyMC models as ProbPipe distributions."""

from __future__ import annotations

import logging
from typing import Any, Callable

import jax.numpy as jnp

from ..core.distribution import Distribution, Provenance
from ..custom_types import Array
from ..inference._diagnostics import MCMCDiagnostics
from ..inference._mcmc_distribution import MCMCApproximateDistribution
from ._base import ProbabilisticModel

logger = logging.getLogger(__name__)

__all__ = ["PyMCModel"]


class PyMCModel(ProbabilisticModel):
    """PyMC model wrapper.

    Wraps a PyMC model-building function as a ProbPipe
    :class:`ProbabilisticModel`.

    Parameters
    ----------
    model_fn : callable
        Function that takes ``**observed`` keyword arguments and
        returns a ``pymc.Model`` context.  Example::

            def my_model(y=None):
                with pm.Model() as m:
                    mu = pm.Normal("mu", 0, 1)
                    sigma = pm.HalfNormal("sigma", 1)
                    pm.Normal("y", mu, sigma, observed=y)
                return m
    name : str or None
        Model name for provenance.

    Raises
    ------
    ImportError
        If ``pymc`` is not installed.
    """

    def __init__(
        self,
        model_fn: Callable[..., Any],
        *,
        name: str | None = None,
    ):
        try:
            import pymc  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pymc is required for PyMCModel. "
                "Install it with: pip install pymc"
            ) from e

        self._model_fn = model_fn
        self._name_str = name

        # Discover observed variable names from the model function signature.
        # Parameters with default value None are treated as observed variables
        # that will receive data at conditioning time.
        import inspect

        sig = inspect.signature(model_fn)
        self._observed_names = tuple(
            name
            for name, p in sig.parameters.items()
            if p.default is None
        )

        # Build the model once without data to discover free parameters.
        # When called with no args, observed vars become free RVs.
        self._unconditioned_model = model_fn()
        all_free = {rv.name for rv in self._unconditioned_model.free_RVs}
        observed_set = set(self._observed_names)
        self._param_names = tuple(
            rv.name
            for rv in self._unconditioned_model.free_RVs
            if rv.name not in observed_set
        )

    # -- Distribution interface ---------------------------------------------

    @property
    def name(self) -> str | None:
        return self._name_str

    @property
    def event_shape(self) -> tuple[int, ...]:
        # Total number of scalar parameters (excluding observed)
        observed_set = set(self._observed_names)
        total = 0
        for rv in self._unconditioned_model.free_RVs:
            if rv.name in observed_set:
                continue
            shape = rv.type.shape
            size = 1
            for s in shape:
                if s is not None:
                    size *= s
            total += size
        return (total,)

    # -- SupportsNamedComponents interface ----------------------------------

    @property
    def component_names(self) -> tuple[str, ...]:
        return self._param_names + self._observed_names

    def __getitem__(self, key: str) -> Any:
        if key in self._param_names or key in self._observed_names:
            return key  # placeholder — PyMC doesn't expose sub-distributions easily
        raise KeyError(f"Unknown component: {key!r}")

    # -- SupportsConditionableComponents interface --------------------------

    @property
    def conditionable_components(self) -> dict[str, bool]:
        result = {}
        for name in self._param_names:
            result[name] = False
        for name in self._observed_names:
            result[name] = True
        return result

    # -- ProbabilisticModel interface ---------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self._param_names

    # -- Sampling (prior predictive) ----------------------------------------

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        """Prior predictive sampling via PyMC."""
        import pymc as pm

        n = 1
        for s in sample_shape:
            n *= s

        model = self._model_fn()
        with model:
            prior = pm.sample_prior_predictive(samples=max(n, 1))

        # Concatenate parameter values into a single array
        arrays = []
        for name in self._param_names:
            vals = prior.prior[name].values.reshape(n, -1)
            arrays.append(jnp.asarray(vals))
        samples = jnp.concatenate(arrays, axis=-1)

        if sample_shape == ():
            return samples[0]
        return samples.reshape(*sample_shape, -1)

    # -- PyMC model access (for nutpie integration) -------------------------

    def _pymc_model(self, data: Any = None) -> Any:
        """Build a PyMC model, optionally with data."""
        if data is not None:
            if isinstance(data, dict):
                return self._model_fn(**data)
            # If data is an array, pass as first observed variable
            return self._model_fn(**{self._observed_names[0]: data})
        return self._model_fn()

    # -- Conditioning -------------------------------------------------------

    def _condition_on(self, observed: Any, /, **kwargs: Any) -> MCMCApproximateDistribution:
        """Condition on observed data using PyMC's sampler.

        Parameters
        ----------
        observed : dict or array-like
            Observed data.  If a dict, keys should match observed variable
            names.  If an array, assigned to the first observed variable.
        **kwargs
            Sampling parameters: ``num_results`` (default 1000),
            ``num_warmup`` (default 500), ``num_chains`` (default 4),
            ``random_seed`` (default 0).
        """
        import pymc as pm

        num_results = kwargs.get("num_results", 1000)
        num_warmup = kwargs.get("num_warmup", 500)
        num_chains = kwargs.get("num_chains", 4)
        random_seed = kwargs.get("random_seed", 0)

        # Build conditioned model
        model = self._pymc_model(data=observed)

        with model:
            trace = pm.sample(
                draws=num_results,
                tune=num_warmup,
                chains=num_chains,
                random_seed=random_seed,
                return_inferencedata=True,
            )

        # Extract chains
        chains = []
        for c in range(num_chains):
            chain_arrays = []
            for name in self._param_names:
                vals = trace.posterior[name].values[c]
                if vals.ndim == 1:
                    vals = vals[:, None]
                else:
                    vals = vals.reshape(vals.shape[0], -1)
                chain_arrays.append(jnp.asarray(vals))
            chains.append(jnp.concatenate(chain_arrays, axis=-1))

        diagnostics = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(num_results * num_chains),
            step_size=0.0,
            is_accepted=None,
            algorithm="pymc_nuts",
        )

        result = MCMCApproximateDistribution(
            chains,
            diagnostics=diagnostics,
            name="posterior",
        )
        result.with_source(
            Provenance(
                "pymc_sample",
                parents=(self,),
                metadata={
                    "num_results": num_results,
                    "num_warmup": num_warmup,
                    "num_chains": num_chains,
                },
            )
        )
        return result

    def __repr__(self) -> str:
        params = ", ".join(self._param_names)
        return f"PyMCModel(params=[{params}])"
