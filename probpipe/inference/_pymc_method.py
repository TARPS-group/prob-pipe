"""PyMC inference methods for the registry: NUTS and ADVI."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..core._registry import MethodInfo
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._registry import InferenceMethod


def _extract_pymc_chains(trace: Any, param_names: list[str], num_chains: int) -> list:
    """Extract per-chain sample arrays from a PyMC ArviZ trace."""
    chains = []
    for c in range(num_chains):
        chain_arrays = []
        for name in param_names:
            vals = trace.posterior[name].values[c]
            if vals.ndim == 1:
                vals = vals[:, None]
            else:
                vals = vals.reshape(vals.shape[0], -1)
            chain_arrays.append(jnp.asarray(vals))
        chains.append(jnp.concatenate(chain_arrays, axis=-1))
    return chains


class PyMCNutsMethod(InferenceMethod):
    """PyMC NUTS sampler for PyMCModel."""

    def __init__(self) -> None:
        from ..modeling._pymc import PyMCModel
        self._model_type = PyMCModel

    @property
    def name(self) -> str:
        return "pymc_nuts"

    def supported_types(self) -> tuple[type, ...]:
        return (self._model_type,)

    @property
    def priority(self) -> int:
        return 60

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        if not isinstance(dist, self._model_type):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires PyMCModel")
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        import pymc as pm

        num_results = kwargs.get("num_results", 1000)
        num_warmup = kwargs.get("num_warmup", 500)
        num_chains = kwargs.get("num_chains", 4)
        random_seed = kwargs.get("random_seed", 0)

        model = dist._pymc_model(data=observed)
        with model:
            trace = pm.sample(
                draws=num_results,
                tune=num_warmup,
                chains=num_chains,
                cores=1,  # avoid os.fork() which deadlocks with JAX threads
                random_seed=random_seed,
                return_inferencedata=True,
            )

        chains = _extract_pymc_chains(trace, dist._param_names, num_chains)

        return make_posterior(
            chains, parents=(dist,), algorithm="pymc_nuts",
            auxiliary=trace,
            num_results=num_results, num_warmup=num_warmup, num_chains=num_chains,
        )


class PyMCADVIMethod(InferenceMethod):
    """PyMC ADVI (Automatic Differentiation Variational Inference)."""

    def __init__(self) -> None:
        from ..modeling._pymc import PyMCModel
        self._model_type = PyMCModel

    @property
    def name(self) -> str:
        return "pymc_advi"

    def supported_types(self) -> tuple[type, ...]:
        return (self._model_type,)

    @property
    def priority(self) -> int:
        return 35

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        if not isinstance(dist, self._model_type):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires PyMCModel")
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        import pymc as pm
        import numpy as np

        num_iterations = kwargs.get("num_iterations", 30000)
        num_results = kwargs.get("num_results", 1000)
        random_seed = kwargs.get("random_seed", 0)
        vi_method = kwargs.get("vi_method", "advi")

        model = dist._pymc_model(data=observed)
        with model:
            approx = pm.fit(n=num_iterations, method=vi_method, random_seed=random_seed)
            trace = approx.sample(num_results)

        chain_draws = [trace.posterior[n].values[0] for n in dist._param_names]
        samples = np.concatenate(
            [np.atleast_2d(d).reshape(num_results, -1) for d in chain_draws],
            axis=1,
        )
        chains = [jnp.asarray(samples)]
        algorithm = f"pymc_{vi_method}"

        return make_posterior(
            chains, parents=(dist,), algorithm=algorithm,
            auxiliary=trace,
            num_iterations=num_iterations,
        )
