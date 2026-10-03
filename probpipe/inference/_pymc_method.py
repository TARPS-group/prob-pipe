"""PyMC inference methods for the registry: NUTS and ADVI."""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from ..core._dispatch import Feasibility
from ..core._specs import OutputSpec
from ..distributions._distribution import Distribution
from ..distributions._empirical import EmpiricalDistribution
from ..operations._condition import InferenceMethod, _UnnormalizedConditional
from ._approximate_distribution import _record_run, make_posterior
from ._inference_utils import (
    extract_chain_columns,
    integer_seed,
    joint_and_given,
    posterior_var_order,
    run_seed,
)


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

    _method_options = ("cores", "num_chains", "num_results", "num_warmup", "random_seed")

    def __init__(self) -> None:
        from ..families._programs import PyMCModel

        self._model_type = PyMCModel

    @property
    def name(self) -> str:
        return "pymc_nuts"

    def supported_types(self) -> tuple[type, ...]:
        return (self._model_type, _UnnormalizedConditional)

    @property
    def priority(self) -> int:
        return 82

    def check(self, target: Any, /, **kwargs: Any) -> Feasibility:
        """Whether the target is a PyMC model, or one at its observed values."""
        dist, _ = joint_and_given(target)
        if not isinstance(dist, self._model_type):
            return Feasibility(feasible=False, description="Requires PyMCModel")
        return Feasibility(feasible=True)

    def execute(self, target: Any, /, **kwargs: Any) -> EmpiricalDistribution:
        """PyMC inference on the model the target carries, at the observed values it binds."""
        self._check_options(kwargs)
        import pymc as pm

        dist, observed = joint_and_given(target)

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
        random_seed = integer_seed(run_seed(kwargs, self.name))

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
            parents=(target,),
            method="pymc_nuts",
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
    With ``vi_method="advi"``, the default, the result is the fitted
    mean-field family: a ``FactoredDistribution`` with one factor per
    parameter, the Gaussian ADVI fitted to the parameter's unconstrained value
    pushed through the bijector equal to PyMC's transform of it. The family
    covers the log, logodds, interval, and simplex transforms at constant
    bounds. A model with another transform, and the other values of
    ``vi_method``, give ``num_results`` draws of the approximation as an
    ``EmpiricalDistribution``.

    A parametric variational approximation whose quality is bounded by the
    mean-field family. ADVI trades bias for speed, a tradeoff the user should
    choose explicitly; selecting it automatically when, for example,
    ``pymc_nuts`` fails would silently substitute VI for MCMC.
    """

    _method_options = ("num_iterations", "num_results", "random_seed", "vi_method")

    def __init__(self) -> None:
        from ..families._programs import PyMCModel

        self._model_type = PyMCModel

    @property
    def name(self) -> str:
        return "pymc_advi"

    def supported_types(self) -> tuple[type, ...]:
        return (self._model_type, _UnnormalizedConditional)

    @property
    def priority(self) -> int | None:
        return None

    def check(self, target: Any, /, **kwargs: Any) -> Feasibility:
        """Whether the target is a PyMC model, or one at its observed values."""
        dist, _ = joint_and_given(target)
        if not isinstance(dist, self._model_type):
            return Feasibility(feasible=False, description="Requires PyMCModel")
        return Feasibility(feasible=True)

    def execute(self, target: Any, /, **kwargs: Any) -> Distribution:
        """PyMC variational inference on the model the target carries, at the observed values it binds."""
        self._check_options(kwargs)
        import pymc as pm

        dist, observed = joint_and_given(target)

        num_iterations = kwargs.get("num_iterations", 30000)
        num_results = kwargs.get("num_results", 1000)
        random_seed = integer_seed(run_seed(kwargs, self.name))
        vi_method = kwargs.get("vi_method", "advi")

        model = dist._pymc_model(data=observed)
        # Build the parameter record in canonical field order before fitting (fail
        # fast on a dynamic-RV / non-concrete model).
        param_names = dist._conditioned_param_names(model)
        event_spec = OutputSpec(dist._parameter_record_for(model, param_names))
        with model:
            approx = pm.fit(n=num_iterations, method=vi_method, random_seed=random_seed)
        name = f"pymc_{vi_method}"
        if vi_method == "advi":
            family = _mean_field_family(approx, model, param_names)
            if family is not None:
                return _record_run(family, (target,), name, num_iterations=num_iterations)
        with model:
            trace = approx.sample(num_results)

        # ADVI's approx.sample yields a single chain of `num_results`
        # draws; extract in natural order and realign by name.
        order = posterior_var_order(trace, param_names)
        chains = extract_chain_columns(trace, order, num_chains=1)

        return make_posterior(
            chains,
            parents=(target,),
            method=name,
            annotations=trace,
            event_spec=event_spec,
            field_order=order,
            num_iterations=num_iterations,
        )


# ---------------------------------------------------------------------------
# The mean-field family
# ---------------------------------------------------------------------------


class _PyMCSimplex(tfp.bijectors.SoftmaxCentered):
    """PyMC's simplex transform: ``y`` in ``R^(K-1)`` maps to ``softmax([y, -sum(y)])``.

    Its event shape changes from ``K - 1`` to ``K``, as ``SoftmaxCentered``'s does,
    and its log-determinant is the one PyMC's ``SimplexTransform`` computes.
    """

    def _forward(self, y: Any) -> Any:
        full = jnp.concatenate([y, -jnp.sum(y, axis=-1, keepdims=True)], axis=-1)
        return jax.nn.softmax(full, axis=-1)

    def _inverse(self, x: Any) -> Any:
        log_x = jnp.log(x)
        return log_x[..., :-1] - jnp.mean(log_x, axis=-1, keepdims=True)

    def _forward_log_det_jacobian(self, y: Any) -> Any:
        n = y.shape[-1] + 1
        total = jnp.sum(y, axis=-1, keepdims=True)
        expanded = jnp.concatenate([y + total, jnp.zeros_like(total)], axis=-1)
        logsumexp = jax.scipy.special.logsumexp(expanded, axis=-1, keepdims=True)
        return jnp.sum(jnp.log(n) + n * total - n * logsumexp, axis=-1)

    def _inverse_log_det_jacobian(self, x: Any) -> Any:
        return -self._forward_log_det_jacobian(self._inverse(x))


def _interval_bijector(transform: Any, rv: Any, model: Any) -> Any:
    """The bijector of PyMC's interval transform of *rv*, or ``NotImplemented``.

    Bounds that depend on another variable of the model, or that are finite at
    some coordinates and infinite at others, have no fixed bijector.
    """
    import pytensor.tensor as pt
    from pytensor.graph.basic import ancestors

    tfb = tfp.bijectors
    # A missing bound is a plain float infinity, so both become tensors.
    lower, upper = (pt.as_tensor_variable(b) for b in transform.get_a_and_b(rv.owner.inputs)[:2])
    variables = set(model.basic_RVs)
    if any(node in variables for node in ancestors([lower, upper])):
        return NotImplemented
    lower, upper = np.asarray(lower.eval()), np.asarray(upper.eval())
    finite_lower, finite_upper = np.isfinite(lower), np.isfinite(upper)
    if finite_lower.all() and finite_upper.all():
        return tfb.Sigmoid(low=jnp.asarray(lower), high=jnp.asarray(upper))
    if finite_lower.all() and not finite_upper.any():
        return tfb.Chain([tfb.Shift(jnp.asarray(lower)), tfb.Exp()])
    if finite_upper.all() and not finite_lower.any():
        return tfb.Chain([tfb.Shift(jnp.asarray(upper)), tfb.Scale(-1.0), tfb.Exp()])
    if not finite_lower.any() and not finite_upper.any():
        return None
    return NotImplemented


def _bijector_of(transform: Any, rv: Any, model: Any) -> Any:
    """The bijector equal to PyMC's backward transform of *rv*.

    ``None`` stands for no transform, and ``NotImplemented`` for a transform the
    mean-field family does not cover.
    """
    from pymc.distributions import transforms

    tfb = tfp.bijectors
    if transform is None:
        return None
    if isinstance(transform, transforms.LogTransform):
        return tfb.Exp()
    if isinstance(transform, transforms.LogOddsTransform):
        return tfb.Sigmoid()
    if isinstance(transform, transforms.SimplexTransform):
        return _PyMCSimplex(name="pymc_simplex")
    if isinstance(transform, transforms.IntervalTransform):
        return _interval_bijector(transform, rv, model)
    return NotImplemented


def _mean_field_family(approx: Any, model: Any, param_names: list[str]) -> Distribution | None:
    """The mean-field family ADVI fitted over *param_names*, or ``None`` when it has no bijector.

    Each parameter's factor is the Gaussian fitted to its unconstrained value,
    pushed through the bijector equal to PyMC's transform of it, so the family's
    draws and density are those of the approximation PyMC fitted.
    """
    from ..distributions._factored import FactoredDistribution
    from ..families import BijectorTransformedDistribution, Normal

    group = approx.groups[0]
    mean, std = np.asarray(approx.mean.eval()), np.asarray(approx.std.eval())
    factors = []
    for name in param_names:
        rv = model[name]
        bijector = _bijector_of(model.rvs_to_transforms.get(rv), rv, model)
        if bijector is NotImplemented:
            return None
        # Each entry is (value name, slice of the flat vector, shape, dtype).
        _, coordinates, shape, _ = group.ordering[model.rvs_to_values[rv].name]
        loc = jnp.reshape(jnp.asarray(mean[coordinates]), shape)
        scale = jnp.reshape(jnp.asarray(std[coordinates]), shape)
        if bijector is None:
            factors.append(Normal(name, loc, scale))
        else:
            base = Normal(f"{name}_unconstrained", loc, scale)
            factors.append(BijectorTransformedDistribution(name, base, bijector))
    return FactoredDistribution("posterior", factors)
