"""The condition_on operation and the inference-method registry behind Bayes' rule.

``condition_on(d, given)`` fixes fields of a distribution or of a conditional
distribution. ``given`` is field-keyed, a ``Record`` or a mapping of field
paths. Binding a given slot applies the kernel, and binding a produced field
conditions the law. The routes, in selection order:

1. ``curry`` binds given slots of a ``ConditionalDistribution`` through its
   ``_condition_on``, exactly.
2. ``slice`` assembles the conditional from a factored law's normalized factors
   when the conditioned fields admit an exact slice.
3. ``exact_conditioning`` calls ``_condition_on`` on a law claiming
   ``SupportsExactConditioning``.
4. ``bayes`` with an exact method of the inference-method registry.
5. ``approximate_conditioning`` calls ``_condition_on`` on a law claiming
   ``SupportsApproximateConditioning``.
6. ``bayes`` with an approximate method of the registry.

So no approximate route runs while an exact one applies, and ``exact_only``
excludes the approximate capability and the approximate methods alike.

The inference methods' own parameters, such as warmup lengths, are controls
set through ``with_options``. Each route declares the controls it reads:
``bayes`` passes the parameters of the registered inference methods to the
selected method, ``approximate_conditioning`` passes an amortized posterior's
sample count and seed to its ``_condition_on`` as keyword options, and the
exact routes read none. A control that no route declares raises ``TypeError``
at ``with_options``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ..core._dispatch import Feasibility
from ..core._record_spec import RecordSpec
from ..core._spec_base import TermSpec
from ..core._specs import InputSpec, OutputSpec
from ..core.record import Record
from ..distributions._capabilities import (
    SupportsApproximateConditioning,
    SupportsExactConditioning,
    _capability_guard,
)
from ..distributions._conditional import ConditionalDistribution, ConditionalDistributionSpec
from ..distributions._distribution import Distribution, DistributionSpec
from ..distributions._factored import SupportsFactors
from ..inference._registry import InferenceMethod, inference_method_registry
from ._operation import BoundCall, operation

__all__ = ["InferenceMethod", "condition_on", "inference_method_registry"]

_PATH_SEP = "/"

_MCMC_CONTROLS = ("init", "num_chains", "num_results", "num_warmup", "random_seed", "step_size")
_SGMCMC_CONTROLS = (
    "batch_size",
    "init",
    "num_results",
    "num_warmup",
    "random_seed",
    "step_size",
    "with_replacement",
)
_BACKEND_NUTS_CONTROLS = ("num_chains", "num_results", "num_warmup", "random_seed")

#: The parameters each inference method registered by :mod:`probpipe.inference`
#: reads, by method name; the ``bayes`` route declares every one of them.
_INFERENCE_METHOD_CONTROLS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "blackjax_nuts": (*_MCMC_CONTROLS, "num_integration_steps"),
        "blackjax_hmc": (*_MCMC_CONTROLS, "num_integration_steps"),
        "blackjax_rwmh": (*_MCMC_CONTROLS, "adapt", "n_windows", "proposal_cov"),
        "blackjax_elliptical_slice": (
            "init",
            "num_chains",
            "num_results",
            "num_warmup",
            "random_seed",
        ),
        "blackjax_sgld": _SGMCMC_CONTROLS,
        "blackjax_sghmc": (*_SGMCMC_CONTROLS, "alpha", "beta", "num_integration_steps"),
        "tfp_nuts": _MCMC_CONTROLS,
        "tfp_hmc": _MCMC_CONTROLS,
        "nutpie_nuts": _BACKEND_NUTS_CONTROLS,
        "cmdstan_nuts": _BACKEND_NUTS_CONTROLS,
        "pymc_nuts": (*_BACKEND_NUTS_CONTROLS, "cores"),
        "pymc_advi": ("num_iterations", "num_results", "random_seed", "vi_method"),
        "pyabc_smcabc": (
            "distance_fn",
            "eps",
            "eps_alpha",
            "max_populations",
            "max_total_nr_simulations",
            "max_walltime",
            "min_acceptance_rate",
            "minimum_epsilon",
            "n_particles",
            "random_seed",
            "sampler",
            "summary_fn",
            "transitions",
        ),
    }
)

#: The parameters an amortized posterior's ``_condition_on`` reads, as
#: :class:`~probpipe.inference.BayesFlowModel` does.
_AMORTIZED_CONDITIONING_CONTROLS = ("num_results", "random_seed")


def _given_keys(given: Any) -> tuple[str, ...] | None:
    """The field paths *given* names: a Record's fields or a mapping's keys, or None for neither."""
    if isinstance(given, Record):
        return tuple(given.fields)
    if isinstance(given, Mapping):
        return tuple(given)
    return None


def _condition_on_result(d: TermSpec, given: TermSpec) -> OutputSpec:
    """Binding given slots declares the kernel's law, or a kernel over the slots left.

    Conditioning on a produced field leaves the declaration to the law the
    selected route returns.
    """
    if isinstance(d, ConditionalDistributionSpec) and isinstance(given, RecordSpec):
        keys = tuple(given.children)
        if keys and all(key in d.given_spec for key in keys):
            left = {name: spec for name, spec in d.given_spec.items() if name not in keys}
            if not left:
                return OutputSpec(condition_on=DistributionSpec(d.event_spec))
            return OutputSpec(
                condition_on=ConditionalDistributionSpec(InputSpec(left), d.event_spec)
            )
    return OutputSpec(condition_on=None)


@operation(
    result=_condition_on_result,
    roles={"d": (DistributionSpec, ConditionalDistributionSpec), "given": (TermSpec,)},
)
def condition_on(d: Distribution, given: Any):
    """Fix fields of *d* at the values *given* holds, and return the resulting distribution.

    Parameters
    ----------
    d : Distribution or ConditionalDistribution
        The law or kernel to condition.
    given : Record or Mapping
        The values, keyed by field path: given slots of a kernel, or fields the
        law produces.

    Returns
    -------
    Distribution or ConditionalDistribution
        The conditional, as the selected route represents it: an ordinary law
        once every given slot is bound, and a kernel over the slots left
        otherwise.

    Raises
    ------
    ApplicabilityError
        If *d* is neither distribution kind.
    ResolutionError
        If no route applies under the controls.
    """


def _can_curry(call: BoundCall, result: OutputSpec | None) -> Any:
    """Every key of the given names a given slot of a conditional distribution."""
    d, keys = call.operands["d"], _given_keys(call.operands["given"])
    if not isinstance(d, ConditionalDistribution):
        return False
    if not keys:
        return False
    slots = set(d.given_spec)
    heads = {key.split(_PATH_SEP, 1)[0] for key in keys}
    if not heads <= slots:
        if heads & slots:
            raise NotImplementedError("condition_on.curry: given slots bound with produced fields")
        return False
    if any(_PATH_SEP in key for key in keys):
        raise NotImplementedError("condition_on.curry: binding part of a structured slot")
    return True


def _curry(call: BoundCall, result: OutputSpec | None) -> Any:
    """The kernel's ``_condition_on`` at the given slots."""
    return call.operands["d"]._condition_on(call.operands["given"])


def _can_slice(call: BoundCall, result: OutputSpec | None) -> Any:
    """The conditioned fields leave a conditional assembled from normalized factors.

    A field's slice is exact when the conditional is a product of available
    factors and exact local conditioning operations, which graph position alone
    does not establish.
    """
    d, keys = call.operands["d"], _given_keys(call.operands["given"])
    if not isinstance(d, SupportsFactors) or isinstance(d, ConditionalDistribution) or not keys:
        return False
    return Feasibility(
        False,
        "route 'slice' declined: assembling a conditional from the factors is not implemented",
    )


def _slice(call: BoundCall, result: OutputSpec | None) -> Any:
    """The conditional assembled from the factors the slice keeps."""
    raise NotImplementedError("condition_on.slice")


def _given_paths(d: Any, given: Any) -> tuple[str, ...]:
    """The paths *given* binds: its keys, or every component when the value is a whole draw."""
    keys = _given_keys(given)
    return keys if keys is not None else tuple(d.event_spec.components)


def _conditioning_guard(call: BoundCall, result: OutputSpec | None) -> Any:
    """The law's conditioning guard admits the given's paths, where its class defines one."""
    d = call.operands["d"]
    return _capability_guard(d, "_condition_on", _given_paths(d, call.operands["given"]))


condition_on.structural_route("curry", check=_can_curry, execute=_curry, exact=True)
condition_on.structural_route("slice", check=_can_slice, execute=_slice, exact=True)
condition_on.capability_route(
    "exact_conditioning",
    operand="d",
    protocol=SupportsExactConditioning,
    method="_condition_on",
    exact=True,
    check=_conditioning_guard,
)
condition_on.capability_route(
    "approximate_conditioning",
    operand="d",
    protocol=SupportsApproximateConditioning,
    method="_condition_on",
    exact=False,
    check=_conditioning_guard,
    controls=_AMORTIZED_CONDITIONING_CONTROLS,
)
condition_on.registry_route(
    "bayes",
    registry=inference_method_registry,
    controls={name for names in _INFERENCE_METHOD_CONTROLS.values() for name in names},
)
