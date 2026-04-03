"""PyMC inference methods for the registry: MCMC and ADVI."""

from __future__ import annotations

from typing import Any

from ..core._registry import MethodInfo
from ._registry import InferenceMethod


class PyMCMCMCMethod(InferenceMethod):
    """PyMC's default MCMC sampler (NUTS) for PyMCModel."""

    @property
    def name(self) -> str:
        return "pymc_mcmc"

    def supported_types(self) -> tuple[type, ...]:
        from ..modeling._pymc import PyMCModel
        return (PyMCModel,)

    @property
    def priority(self) -> int:
        return 60

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        from ..modeling._pymc import PyMCModel
        if not isinstance(dist, PyMCModel):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires PyMCModel")
        return MethodInfo(feasible=True, method_name=self.name,
                          description="PyMC MCMC (default NUTS)")

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> Any:
        return dist._condition_on(observed, **kwargs)


class PyMCADVIMethod(InferenceMethod):
    """PyMC ADVI (Automatic Differentiation Variational Inference)."""

    @property
    def name(self) -> str:
        return "pymc_advi"

    def supported_types(self) -> tuple[type, ...]:
        from ..modeling._pymc import PyMCModel
        return (PyMCModel,)

    @property
    def priority(self) -> int:
        return 35

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        from ..modeling._pymc import PyMCModel
        if not isinstance(dist, PyMCModel):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires PyMCModel")
        return MethodInfo(feasible=True, method_name=self.name,
                          description="PyMC ADVI (variational inference)")

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> Any:
        import pymc as pm

        from ..inference._diagnostics import InferenceDiagnostics
        from ..inference._mcmc_distribution import MCMCApproximateDistribution
        from ..core.provenance import Provenance

        num_iterations = kwargs.get("num_iterations", 30000)
        random_seed = kwargs.get("random_seed", 0)
        vi_method = kwargs.get("vi_method", "advi")

        model = dist._pymc_model(data=observed)
        with model:
            approx = pm.fit(
                n=num_iterations,
                method=vi_method,
                random_seed=random_seed,
            )
            # Draw samples from the fitted approximation
            num_results = kwargs.get("num_results", 1000)
            trace = approx.sample(num_results)

        # Extract samples as a single chain
        param_names = dist._param_names
        chain_draws = []
        for name in param_names:
            chain_draws.append(trace.posterior[name].values[0])

        import jax.numpy as jnp
        import numpy as np
        samples = np.concatenate(
            [np.atleast_2d(d).reshape(num_results, -1) for d in chain_draws],
            axis=1,
        )
        chains = [jnp.asarray(samples, dtype=jnp.float32)]

        diagnostics = InferenceDiagnostics(algorithm=f"pymc_{vi_method}")
        result = MCMCApproximateDistribution(
            chains, diagnostics=diagnostics, name="posterior",
        )
        result.with_source(Provenance(
            f"pymc_{vi_method}", parents=(),
            metadata={"num_iterations": num_iterations, "algorithm": f"pymc_{vi_method}"},
        ))
        return result
