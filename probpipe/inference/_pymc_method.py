"""PyMC inference methods for the registry: NUTS and ADVI."""

from __future__ import annotations

import os
from typing import Any

from ..core._dispatch import Feasibility
from ..core._specs import OutputSpec
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._inference_utils import extract_chain_columns, posterior_var_order
from ._registry import InferenceMethod


class PyMCNutsMethod(InferenceMethod):
    """PyMC NUTS, registered as ``pymc_nuts`` at priority 82.

    Applies to a ``PyMCModel``.

    Notes
    -----
    An optimised backend: native PyMC NUTS, tailored to ``PyMCModel``. Tied
    with ``cmdstan_nuts`` (82), which applies to a disjoint model class, and
    below ``nutpie_nuts`` (88), whose Rust gradients are faster on a
    ``PyMCModel`` too when nutpie is installed.
    """

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
        return 82

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> Feasibility:
        if not isinstance(dist, self._model_type):
            return Feasibility(feasible=False, description="Requires PyMCModel")
        return Feasibility(feasible=True)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        import pymc as pm

        num_results = kwargs.get("num_results", 1000)
        num_warmup = kwargs.get("num_warmup", 500)
        num_chains = kwargs.get("num_chains", 4)
        # Multi-core sampling forces the "spawn" start method: this process
        # holds live JAX threads, and pymc's POSIX-"fork" default deadlocks
        # when a forked worker inherits a held thread lock. spawn pickles the
        # model to each worker; if the model is not picklable, pymc logs a
        # warning and falls back to single-process sampling — so multi-core
        # is the default without making serializability a hard requirement.
        cores = kwargs.get("cores", min(num_chains, os.cpu_count() or 1))
        random_seed = kwargs.get("random_seed", 0)

        model = dist._pymc_model(data=observed)
        # Build the parameter record in canonical field order before sampling
        # (fail fast on a dynamic-RV / non-concrete model).
        param_names = dist._conditioned_param_names(model)
        event_spec = OutputSpec(dist._parameter_record_for(model, param_names))
        with model:
            trace = pm.sample(
                draws=num_results,
                tune=num_warmup,
                chains=num_chains,
                cores=cores,
                mp_ctx="spawn" if cores > 1 else None,
                random_seed=random_seed,
                return_inferencedata=True,
            )

        # Extract in the trace's natural order; field_order lets
        # make_posterior realign columns to the parameters by name.
        order = posterior_var_order(trace, param_names)
        chains = extract_chain_columns(trace, order, num_chains)

        return make_posterior(
            chains,
            parents=(dist,),
            algorithm="pymc_nuts",
            annotations=trace,
            event_spec=event_spec,
            field_order=order,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=num_chains,
        )


class PyMCADVIMethod(InferenceMethod):
    """PyMC ADVI, registered as ``pymc_advi``, opt-in-only.

    Automatic Differentiation Variational Inference for a ``PyMCModel``;
    runs only when the caller pins ``method="pymc_advi"``.

    Notes
    -----
    A parametric variational approximation whose quality is bounded by the
    mean-field family. ADVI trades bias for speed, a tradeoff the user should
    choose explicitly; selecting it automatically when, for example,
    ``pymc_nuts`` fails would silently substitute VI for MCMC.
    """

    def __init__(self) -> None:
        from ..modeling._pymc import PyMCModel

        self._model_type = PyMCModel

    @property
    def name(self) -> str:
        return "pymc_advi"

    def supported_types(self) -> tuple[type, ...]:
        return (self._model_type,)

    @property
    def priority(self) -> int | None:
        return None

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> Feasibility:
        if not isinstance(dist, self._model_type):
            return Feasibility(feasible=False, description="Requires PyMCModel")
        return Feasibility(feasible=True)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        import pymc as pm

        num_iterations = kwargs.get("num_iterations", 30000)
        num_results = kwargs.get("num_results", 1000)
        random_seed = kwargs.get("random_seed", 0)
        vi_method = kwargs.get("vi_method", "advi")

        model = dist._pymc_model(data=observed)
        # Build the parameter record in canonical field order before fitting (fail
        # fast on a dynamic-RV / non-concrete model).
        param_names = dist._conditioned_param_names(model)
        event_spec = OutputSpec(dist._parameter_record_for(model, param_names))
        with model:
            approx = pm.fit(n=num_iterations, method=vi_method, random_seed=random_seed)
            trace = approx.sample(num_results)

        # ADVI's approx.sample yields a single chain of `num_results`
        # draws; extract in natural order and realign by name.
        order = posterior_var_order(trace, param_names)
        chains = extract_chain_columns(trace, order, num_chains=1)
        algorithm = f"pymc_{vi_method}"

        return make_posterior(
            chains,
            parents=(dist,),
            algorithm=algorithm,
            annotations=trace,
            event_spec=event_spec,
            field_order=order,
            num_iterations=num_iterations,
        )
