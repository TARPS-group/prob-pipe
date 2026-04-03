"""PyMC inference methods for the registry: MCMC and ADVI."""

from __future__ import annotations

from typing import Any

from ..core._registry import MethodInfo
from ._delegating_method import make_delegating_method
from ._registry import InferenceMethod

# PyMC MCMC is a pure delegator — same pattern as CmdStan.
PyMCMCMCMethod = make_delegating_method(
    method_name="pymc_mcmc",
    model_path="probpipe.modeling._pymc.PyMCModel",
    method_priority=60,
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

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> Any:
        import pymc as pm
        import numpy as np
        import jax.numpy as jnp

        from ._diagnostics import InferenceDiagnostics
        from ._mcmc_distribution import MCMCApproximateDistribution
        from ..core.provenance import Provenance

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
        chains = [jnp.asarray(samples, dtype=jnp.float32)]

        algorithm = f"pymc_{vi_method}"
        diagnostics = InferenceDiagnostics(algorithm=algorithm)
        result = MCMCApproximateDistribution(
            chains, diagnostics=diagnostics, name="posterior",
        )
        result.with_source(Provenance(
            algorithm, parents=(),
            metadata={"num_iterations": num_iterations, "algorithm": algorithm},
        ))
        return result
