"""PyMC inference methods for the registry: NUTS and ADVI."""

from __future__ import annotations

from typing import Any

from ..core._registry import MethodInfo
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._inference_utils import extract_chain_columns
from ._registry import InferenceMethod


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
        # Tier 81-90 (optimised backend; native PyMC NUTS, tailored to
        # PyMCModel). At 82 alongside ``cmdstan_nuts``; the two apply to
        # disjoint model classes so the tie is documentary. Below
        # ``nutpie_nuts`` (88; Rust gradients are faster on PyMCModel
        # too when nutpie is installed).
        return 82

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
        # Build the template before sampling so a non-concrete or
        # dynamic-RV model fails fast with a clear error rather than an
        # opaque KeyError during chain extraction.
        record_template = dist._record_template_for(model)
        with model:
            trace = pm.sample(
                draws=num_results,
                tune=num_warmup,
                chains=num_chains,
                cores=1,  # avoid os.fork() which deadlocks with JAX threads
                random_seed=random_seed,
                return_inferencedata=True,
            )

        chains = extract_chain_columns(trace, dist._param_names, num_chains)

        return make_posterior(
            chains, parents=(dist,), algorithm="pymc_nuts",
            auxiliary=trace, record_template=record_template,
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
        # Tier 21-30 by algorithm category (parametric variational
        # approximation; quality bounded by the mean-field family), but
        # registered at the opt-in-only sentinel ``priority=0``. ADVI is
        # a deliberate bias-for-speed tradeoff the user should choose
        # explicitly; auto-dispatching into it when (e.g.) ``pymc_nuts``
        # happens to fail would surface VI silently in MCMC's place.
        # Callers who want ADVI pin ``method="pymc_advi"``.
        return 0

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        if not isinstance(dist, self._model_type):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires PyMCModel")
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        import pymc as pm

        num_iterations = kwargs.get("num_iterations", 30000)
        num_results = kwargs.get("num_results", 1000)
        random_seed = kwargs.get("random_seed", 0)
        vi_method = kwargs.get("vi_method", "advi")

        model = dist._pymc_model(data=observed)
        # Build the template before fitting so a non-concrete or
        # dynamic-RV model fails fast with a clear error.
        record_template = dist._record_template_for(model)
        with model:
            approx = pm.fit(n=num_iterations, method=vi_method, random_seed=random_seed)
            trace = approx.sample(num_results)

        # ADVI's approx.sample yields a single chain of `num_results` draws.
        chains = extract_chain_columns(trace, dist._param_names, num_chains=1)
        algorithm = f"pymc_{vi_method}"

        return make_posterior(
            chains, parents=(dist,), algorithm=algorithm,
            auxiliary=trace, record_template=record_template,
            num_iterations=num_iterations,
        )
