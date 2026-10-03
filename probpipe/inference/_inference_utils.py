"""Backend-agnostic inference utilities.

Functions for building target log-density callables and initial chain
states from a :class:`~probpipe.Distribution` plus
observed data. Shared across every inference backend in
``probpipe.inference`` so they consume the same source of truth.

Two target builders:

- :func:`build_target_log_prob` returns a Record-shaped target
  (the TFP-flavoured interface).
- :func:`build_target_log_prob_flat` returns a flat-vector target —
  the BlackJAX entry point. It rebuilds the prior's event from each flat
  vector, the layout III.7 fixes for a numeric law, so kernels that operate
  on flat parameter vectors plug in without per-backend flatten / unflatten
  plumbing.

Scope: private to ``probpipe.inference``. Symbols are package-private
utilities shared across the backend modules; not re-exported through
``probpipe.inference.__init__`` or the top-level package.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from xarray import DataTree

import itertools
import logging

import jax
import jax.numpy as jnp
import numpy as np

from ..core._numeric_record import _reconstruct_from_vector
from ..core._record_spec import NumericRecordSpec
from ..core._spec_base import NumericArraySpec
from ..core._specs import OutputSpec
from ..core.record import Record
from ..custom_types import Array, ArrayLike
from ..distributions._capabilities import SupportsSampling, _is_normalized
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution
from ..distributions._factored import (
    FactoredConditionalDistribution,
    FactoredDistribution,
    _children,
    _components_of,
    _event_of,
    _factor_graph,
)
from ..families._backend import TFPDistribution
from ..operations._condition import _unnormalized_conditional, _UnnormalizedConditional

logger = logging.getLogger(__name__)

__all__ = [
    "as_prng_key",
    "build_mcmc_datatree",
    "build_target_log_prob",
    "build_target_log_prob_flat",
    "extract_chain_columns",
    "extract_event_spec",
    "flat_density",
    "flat_record",
    "flat_unflatten",
    "flat_vector",
    "get_init_state",
    "integer_seed",
    "is_jax_traceable",
    "joint_and_given",
    "likelihood_flat",
    "model_factors",
    "observed_parts",
    "observed_target",
    "parallel_chain_map",
    "parameter_given",
    "posterior_var_order",
    "run_chain_scan",
    "run_seed",
    "unconstrained_chain",
    "unconstrained_coordinates",
]


# ---------------------------------------------------------------------------
# Chain extraction
# ---------------------------------------------------------------------------


def extract_chain_columns(
    trace: Any,
    names: list[str],
    num_chains: int,
) -> list[Array]:
    """Per-chain flat sample matrices from an ArviZ-like trace.

    For each chain ``c`` and each variable in *names* (in that order),
    pulls ``trace.posterior[name].values[c]``, flattens trailing axes to
    2-D ``(draws, -1)``, and concatenates across variables. Columns are
    laid out in *names* order; pass the same order as ``field_order`` to
    :func:`make_posterior` so assembly aligns them to the template fields
    by name.

    Parameters
    ----------
    trace : ArviZ-like trace
        Object exposing a ``posterior`` group indexable by variable name.
    names : list of str
        Variables to extract, in the desired column order.
    num_chains : int
        Number of chains (leading axis of each ``values`` array).

    Returns
    -------
    list of Array
        One ``(draws, total_flat_dim)`` array per chain.
    """
    chains = []
    for c in range(num_chains):
        chain_arrays = []
        for name in names:
            vals = trace.posterior[name].values[c]
            if vals.ndim == 1:
                vals = vals[:, None]
            else:
                vals = vals.reshape(vals.shape[0], -1)
            chain_arrays.append(jnp.asarray(vals))
        chains.append(jnp.concatenate(chain_arrays, axis=-1))
    return chains


def posterior_var_order(trace: Any, keep: Iterable[str]) -> list[str]:
    """Variable names in *trace*'s ``posterior`` natural order, filtered
    to *keep*.

    Pass the result as both the extraction order
    (:func:`extract_chain_columns`) and ``field_order``
    (:func:`make_posterior`), so name-keyed assembly realigns columns to
    the template regardless of the backend's variable order — nutpie, for
    instance, sorts ``data_vars`` alphabetically.

    Raises
    ------
    ValueError
        If any name in *keep* is absent from ``trace.posterior``. Failing
        here names the missing variables, rather than letting them surface
        later as a cryptic "not a permutation" error in
        :func:`make_posterior`'s column realignment.
    """
    keep = list(keep)
    available = list(trace.posterior.data_vars)
    missing = [name for name in keep if name not in available]
    if missing:
        raise ValueError(
            f"trace posterior is missing expected variable name(s) "
            f"{missing}; available posterior variables are {available}. "
            f"Every parameter being assembled must be present in the trace."
        )
    keep_set = set(keep)
    return [name for name in available if name in keep_set]


# ---------------------------------------------------------------------------
# JAX-traceability probe
# ---------------------------------------------------------------------------


def is_jax_traceable(fn: Callable, init_state: jnp.ndarray) -> bool:
    """Probe whether *fn* can be traced by JAX at *init_state*.

    Used by gradient-based MCMC ``check()`` methods to filter out
    targets that would fail at ``execute()`` time. Costs ~one JAX
    trace — note that ``jax.make_jaxpr`` does not populate the JIT
    cache, so the subsequent ``lax.scan`` / ``vmap`` inside the
    runner re-traces from scratch.
    """
    try:
        jax.make_jaxpr(fn)(init_state)
        return True
    except Exception:
        logger.debug("is_jax_traceable: trace failed for %r", fn, exc_info=True)
        return False


def as_prng_key(seed: int | Array) -> Array:
    """Upgrade an ``int`` seed to a ``PRNGKey``; pass keys through.

    Centralises the ``isinstance(seed, int)`` branch repeated across
    the gradient-MCMC backends.
    """
    return jax.random.PRNGKey(seed) if isinstance(seed, int) else seed


#: The sampling ABI of the key that seeds an inference method's run.
_RUN_SEED_ABI = "probpipe.inference.run_seed/v1"


def run_seed(options: Mapping[str, Any], method: str) -> int | Array:
    """The seed of one run of the inference method *method*: its ``random_seed``, or a workflow key.

    A ``random_seed`` the call's options set is returned as it is. Otherwise
    the run's randomness is a workflow-owned random event (V.8), whose key the
    enclosing scope derives from its root seed and the call's structure, so
    ``workflow_run(seed=...)`` reproduces the run, and scopes with different
    seeds, or two unscoped calls, run different chains.
    """
    seed = options.get("random_seed")
    if seed is not None:
        return seed
    from ..functions import _broker

    return _broker._resolve_automatic_key(
        None,
        _broker._singleton_effect_plan(
            operation_kind="inference",
            execution_mode="sampled",
            sample_shape=None,
            sampling_abi=_RUN_SEED_ABI,
            provider_abi=f"probpipe.inference.{method}/v1",
        ),
    )


def integer_seed(seed: int | Array) -> int:
    """*seed* as the non-negative 32-bit integer a backend's own seed argument takes.

    An integer is returned as it is, and a key gives an integer drawn from it.
    """
    if isinstance(seed, int | np.integer):
        return int(seed)
    return int(jax.random.randint(as_prng_key(seed), (), 0, np.iinfo(np.int32).max))


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def observed_target(model: Any, observed: Any) -> Any:
    """The target of normalizing *model* at *observed*, as ``probpipe.condition_on`` passes them.

    A model with no data, and an object that is not a law, is its own target.
    Data keyed by the model's given slots and fields form the exact stage of
    ``condition_on`` (VI.6): a kernel binds the slots its data name first, as
    currying does, and the data of the fields it produces then form the
    unnormalized conditional of the result, whose joint and given values a
    method reads back. A law that currying leaves unnormalized, with no data
    left, is the target itself. Data that name no field, such as one array for
    a whole program, form the unnormalized conditional of the model at those
    data, from which :func:`observed_parts` reads them back, and so do data
    that only curry a kernel to a normalized law, which no method normalizes.
    """
    if observed is None or not isinstance(model, (Distribution, ConditionalDistribution)):
        return model
    if not isinstance(observed, (Record, Mapping)):
        return _UnnormalizedConditional(model, observed, model.event_spec, keyed=False)
    values = dict(observed.children if isinstance(observed, Record) else observed)
    law = model
    if isinstance(model, ConditionalDistribution):
        slots = {key: value for key, value in values.items() if key in model.given_spec}
        values = {key: value for key, value in values.items() if key not in slots}
        if slots:
            law = model._condition_on(slots)
    if not values:
        if isinstance(law, Distribution) and not _is_normalized(law):
            return law
        return _UnnormalizedConditional(model, observed, model.event_spec, keyed=False)
    if not set(values) <= set(law.event_spec.components):
        return _UnnormalizedConditional(law, values, law.event_spec, keyed=False)
    return _unnormalized_conditional(law, Record("given", values))


def observed_parts(target: Any) -> tuple[Any, Any]:
    """The model and the observed data *target* binds, as the model-and-data helpers take them.

    An unnormalized conditional at data its joint does not declare as fields
    is that joint and those data. Any other target is its own model, with its
    data already bound.
    """
    if isinstance(target, _UnnormalizedConditional) and not target.keyed:
        return target.joint, target.given
    return target, None


def joint_and_given(target: Any) -> tuple[Any, Any]:
    """The joint and the given values an unnormalized conditional carries, else *target* and None.

    A method backed by a program reads the program from the joint and its
    observed values from the given values.
    """
    if isinstance(target, _UnnormalizedConditional):
        return target.joint, target.given
    return target, None


def _has_flat_view(prior: Any) -> bool:
    """Whether *prior* is a parametric family, whose one array a flat vector lays out."""
    return isinstance(prior, TFPDistribution)


def _one_array(law: Any) -> NumericArraySpec | None:
    """The spec of the one numeric array of a concrete shape *law* draws, or None."""
    declaration = getattr(law, "event_spec", None)
    if not isinstance(law, Distribution) or declaration is None or declaration.exposes_record:
        return None
    spec = declaration.spec
    return spec if isinstance(spec, NumericArraySpec) and spec.is_concrete else None


def _reshape_to(shape: tuple[int, ...]) -> Callable[[Array], Array]:
    """The map from a flat vector, in row-major order, to an array of *shape*."""

    def unflatten(theta_flat: Array) -> Array:
        return jnp.reshape(theta_flat, shape)

    return unflatten


def _flat_view(prior: Any) -> Callable[[Array], Array] | None:
    """The map from a flat vector to a draw of *prior*, or ``None`` when it is no family.

    A parametric family draws one array, which the flat vector lays out in
    row-major order, so the map reshapes the vector to the event's shape.
    """
    if not _has_flat_view(prior):
        return None
    spec = prior.event_spec.spec
    return _reshape_to(spec.shape if isinstance(spec, NumericArraySpec) else ())


def flat_record(prior: Any) -> NumericRecordSpec | None:
    """The numeric record a flat chain over *prior* unflattens to, when it is no family.

    ``None`` for a parametric family, and for a law that draws no exposed
    numeric record, such as a law over one array.
    """
    if _has_flat_view(prior):
        return None
    declaration = getattr(prior, "event_spec", None)
    if not isinstance(declaration, OutputSpec) or not declaration.exposes_record:
        return None
    spec = declaration.spec
    return spec if isinstance(spec, NumericRecordSpec) else None


def flat_unflatten(law: Any) -> Callable[[Array], Any]:
    """The map from a flat vector to a draw of the numeric *law*, the inverse of :func:`flat_vector`.

    A draw of one array is the vector reshaped to the event, and an exposed
    numeric record's is the record whose leaves the vector lays out in
    canonical order.

    Raises
    ------
    TypeError
        If *law* draws neither one numeric array nor an exposed numeric record.
    """
    flat_prior = _flat_view(law)
    if flat_prior is not None:
        return flat_prior
    array = _one_array(law)
    if array is not None:
        return _reshape_to(array.shape)
    record = flat_record(law)
    if record is None:
        raise TypeError(f"{type(law).__name__} {law.label!r} draws no value a flat vector lays out")

    def unflatten(theta_flat: Array) -> Any:
        return _reconstruct_from_vector(law.label, record, theta_flat)

    return unflatten


def flat_vector(value: Any) -> Array:
    """*value*, a draw of a numeric law, as one flat vector in canonical order.

    A record, or the nested mapping of a record draw's raw leaves, gives its
    leaves' coordinates in canonical order, and an array is raveled.
    """
    if isinstance(value, Mapping):
        value = Record("draw", value)
    if isinstance(value, Record):
        return value.to_numeric().to_vector()
    return jnp.ravel(jnp.asarray(value))


def _declared_vector(law: Any, draw: Record | Mapping[str, Any]) -> Array:
    """*draw*, a record draw of *law*, as one flat vector laid out as *law* declares its leaves.

    The leaves follow the canonical order of the numeric record :func:`flat_record`
    names, which :func:`flat_unflatten` reads back, whatever order the draw's own
    mapping keeps. A law that declares no such record lays the draw out as
    :func:`flat_vector` does.
    """
    record = flat_record(law)
    if record is None:
        return flat_vector(draw)
    value = draw if isinstance(draw, Record) else Record("draw", draw)
    return jnp.concatenate([jnp.ravel(jnp.asarray(value.raw(path))) for path in record])


class ModelFactors(NamedTuple):
    """The prior and the likelihood of a factored joint at observed values of its fields.

    Attributes
    ----------
    prior : Distribution
        The law of the parameters, the joint of the factors that produce no
        observed field.
    likelihood : Distribution or ConditionalDistribution
        The law of the observed fields given the parameters, the joint of the
        factors that produce them.
    observed : Any
        The observed value of the likelihood's event.
    """

    prior: Distribution
    likelihood: Distribution | ConditionalDistribution
    observed: Any


def _joint_of(name: str, factors: list[Any]) -> Any:
    """The joint of *factors*, the one factor itself when there is one."""
    if len(factors) == 1:
        return factors[0]
    if _factor_graph(tuple(factors)).unmet is None:
        return FactoredDistribution(name, factors)
    return FactoredConditionalDistribution(name, factors)


def model_factors(target: Any) -> ModelFactors | None:
    """The prior and the likelihood factors of the joint *target* carries, or None.

    *target* is the unnormalized conditional of a factored joint at observed
    values of some of its fields (VI.6). The likelihood is the joint of the
    factors that produce the observed fields, and the prior is the joint of the
    others. None when *target* is no such conditional, when a factor produces
    observed and unobserved fields both, when the prior is not a law, or when the
    likelihood conditions on anything but the prior's fields.
    """
    if not isinstance(target, _UnnormalizedConditional) or not target.keyed:
        return None
    joint, given = target.joint, target.given
    factors = getattr(joint, "factors", None)
    if not isinstance(joint, Distribution) or not factors:
        return None
    observed = set(given.fields)
    prior_factors: list[Any] = []
    likelihood_factors: list[Any] = []
    for factor in factors:
        produced = set(factor.event_spec.components)
        if not produced & observed:
            prior_factors.append(factor)
        elif produced <= observed:
            likelihood_factors.append(factor)
        else:
            return None
    if not prior_factors or not likelihood_factors:
        return None
    prior = _joint_of(joint.label, prior_factors)
    likelihood = _joint_of(joint.label, likelihood_factors)
    if not isinstance(prior, Distribution):
        return None
    slots = set(likelihood.given_spec) if isinstance(likelihood, ConditionalDistribution) else set()
    if not slots <= set(prior.event_spec.components):
        return None
    children = dict(given.children)
    value = _event_of(
        likelihood.event_spec,
        {component: children[component] for component in likelihood.event_spec.components},
    )
    return ModelFactors(prior, likelihood, value)


def parameter_given(factors: ModelFactors, draw: Any) -> dict[str, Any]:
    """The likelihood's given values at *draw*, a draw of the prior."""
    if not isinstance(factors.likelihood, ConditionalDistribution):
        return {}
    components = _components_of(factors.prior.event_spec, draw)
    return {slot: components[slot] for slot in factors.likelihood.given_spec}


def likelihood_flat(factors: ModelFactors) -> Callable[[Array], Array]:
    """The log-likelihood at a flat parameter vector, up to a constant in the parameters.

    The vector unflattens to a draw of the prior, whose values bind the
    likelihood's given slots, and the likelihood's unnormalized density is read
    at the observed value.
    """
    unflatten = flat_unflatten(factors.prior)
    likelihood = factors.likelihood

    def loglikelihood_fn(theta_flat: Array) -> Array:
        if isinstance(likelihood, ConditionalDistribution):
            given = parameter_given(factors, unflatten(theta_flat))
            return likelihood._conditional_unnormalized_log_prob(given, factors.observed)
        return likelihood._unnormalized_log_prob(factors.observed)

    return loglikelihood_fn


def flat_density(dist: Any) -> Callable[[Array], Array]:
    """*dist*'s unnormalized log-density at a flat vector.

    The vector is unflattened to the numeric record :func:`flat_record` names,
    and taken as it is for any other law.
    """
    record = flat_record(dist)
    if record is None:
        return dist._unnormalized_log_prob

    def density(theta: Array) -> Array:
        return dist._unnormalized_log_prob(_reconstruct_from_vector(dist.label, record, theta))

    return density


def _joint_draw(target: Any, key: Array, record: NumericRecordSpec) -> Array | None:
    """A flat draw of *target*'s fields from the joint it conditions, or None when there is none.

    The draw is the joint's, restricted to the fields *record* declares, so it
    lies in the support of the unnormalized conditional. A factored joint that
    does not sample draws each field from the factor producing it, when that
    factor is a law that samples, as a prior is.
    """
    joint = target.joint if isinstance(target, _UnnormalizedConditional) else None
    if joint is None:
        return None
    try:
        if isinstance(joint, SupportsSampling):
            children = dict(_children(joint._sample(key, sample_shape=())))
        else:
            children = {}
            laws = [
                factor
                for factor in getattr(joint, "factors", ())
                if isinstance(factor, Distribution) and isinstance(factor, SupportsSampling)
            ]
            for law, subkey in zip(laws, jax.random.split(key, max(len(laws), 1))):
                draw = law._sample(subkey, sample_shape=())
                if isinstance(draw, Record):
                    children.update(draw.children)
                else:
                    (component,) = law.event_spec.components
                    children[component] = draw
        if not set(record.fields) <= set(children):
            return None
        fields = Record("init", {name: children[name] for name in record.fields})
        return fields.to_numeric().to_vector()
    except Exception:
        logger.debug("get_init_state: the joint's draw failed for %r", target, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Initial-state heuristics
# ---------------------------------------------------------------------------


def get_init_state(
    dist: Distribution,
    init: ArrayLike | None,
    *,
    random_seed: int | Array = 0,
) -> jnp.ndarray:
    """Determine an initial chain state.

    Pass the target, the law over the parameters from which init
    candidates are drawn.

    Resolution order:

    1. Explicit ``init`` — trusted, returned verbatim (cast to the
       prior's dtype).
    2. **Prior sample** — if the prior implements ``SupportsSampling``,
       draw a single sample with the supplied ``random_seed``. A record
       draw, or the nested mapping a factored prior draws, is flattened
       to a numeric vector in the order of the prior's declaration.
    3. **Joint draw** — if the prior is an unnormalized conditional
       over a numeric record, return the draw of its joint restricted
       to the unconditioned fields, flattened. A factored joint that
       does not sample draws each field from the factor producing it.
    4. **Stan default** — if the prior has no sampling path but
       exposes ``event_shape`` or a numeric record, return a
       coordinate-wise ``Uniform(-2, 2)`` draw, matching Stan's default
       init for unconstrained parameters. The gradient-based MCMC
       methods this helper feeds all assume an unconstrained parameter
       space, so the box is guaranteed to be inside the support.
    5. Raise — no init heuristic applies.

    Observed data is deliberately not consulted: a ``mean(observed)``
    heuristic would live in the *data* space while the chain state
    lives in the *parameter* space, and the two coincide only for
    pure location models. Callers that genuinely need a data-derived
    init should pass ``init=`` explicitly.
    """
    prior = dist

    target_dtype = getattr(prior, "dtype", None)
    if not isinstance(target_dtype, jnp.dtype):
        from .._dtype import _default_float_dtype

        target_dtype = _default_float_dtype()

    if init is not None:
        return jnp.atleast_1d(jnp.asarray(init, dtype=target_dtype))

    key = as_prng_key(random_seed)

    if isinstance(prior, SupportsSampling):
        try:
            s = prior._sample(key, sample_shape=())
            if isinstance(s, Record | Mapping):
                s = _declared_vector(prior, s)
            return jnp.atleast_1d(jnp.asarray(s, dtype=target_dtype))
        except Exception:
            logger.debug(
                "get_init_state: prior._sample failed for %r; falling back to Uniform(-2, 2)",
                prior,
                exc_info=True,
            )

    record = flat_record(prior)
    if record is not None:
        draw = _joint_draw(prior, key, record)
        if draw is not None:
            return jnp.atleast_1d(jnp.asarray(draw, dtype=target_dtype))

    try:
        shape = prior.event_shape
    except (AttributeError, ValueError):
        # A prior that draws no single concrete array has no box to draw from.
        shape = None
    if shape is None and record is not None:
        shape = (record.vector_size,)
    if shape is not None:
        return jax.random.uniform(
            key,
            shape=shape,
            minval=-2.0,
            maxval=2.0,
            dtype=target_dtype,
        )

    raise ValueError(
        "Cannot determine initial state: pass init= explicitly, or "
        "provide a distribution whose prior implements "
        "SupportsSampling or exposes event_shape."
    )


# ---------------------------------------------------------------------------
# The declaration of the posterior
# ---------------------------------------------------------------------------


def extract_event_spec(dist: Distribution) -> OutputSpec | None:
    """Return the declaration of *dist*'s prior, or ``None`` for a prior with no flat vector.

    The target is its own prior. A prior that is neither a parametric family nor
    an exposed numeric record, such as a bare ``SupportsLogProb`` target over a
    flat array, gives ``None``. :func:`build_target_log_prob_flat` uses the
    same condition, so every method names and shapes the posterior of such a
    target alike.
    """
    if not _has_flat_view(dist) and flat_record(dist) is None:
        return None
    return dist.event_spec


# ---------------------------------------------------------------------------
# Target log-density construction
# ---------------------------------------------------------------------------


def build_target_log_prob(
    dist: Distribution,
    observed: ArrayLike | Record | None,
) -> Callable[[Any], Array]:
    """Build a ``target_log_prob_fn(params)`` from *dist* and *observed*.

    Two cases, in the order the body dispatches them:

    1. **Bare target with data**: joint over ``(params, data)``,
       evaluated as ``dist._unnormalized_log_prob((params, data))``.
    2. **Target without data**: ``dist._unnormalized_log_prob``
       returned directly, the data already bound, as an unnormalized
       conditional binds them.

    The unnormalized accessor is used because MCMC
    samplers do not require a normalized density. Distributions that
    only implement ``_log_prob`` are unaffected: the
    ``SupportsUnnormalizedLogProb`` protocol provides a default
    ``_unnormalized_log_prob`` that delegates to ``_log_prob``.

    Observed data is passed through to the likelihood as-is (may be a
    raw array, a ``Record`` object, or a dict — the likelihood handles
    its own input types).
    """
    if observed is not None:
        return lambda params: dist._unnormalized_log_prob((params, observed))

    return dist._unnormalized_log_prob


def build_target_log_prob_flat(
    dist: Distribution,
    observed: ArrayLike | Record | None,
    *,
    init: ArrayLike | None = None,
    random_seed: int | Array = 0,
) -> tuple[Callable[[Array], Array], Array, OutputSpec | None]:
    """Build a flat-vector target + initial state + (optional) prior declaration.

    Returns ``(target_flat_fn, flat_init, event_spec)``:

    - ``target_flat_fn(theta_flat) -> log_prob``: a callable that
      consumes a flat parameter vector.
    - ``flat_init``: the flat-vector initial chain state from
      :func:`get_init_state`.
    - ``event_spec``: the prior's declaration when the prior is a
      numeric law with a flat-vector view; ``None`` for bare
      array-shaped targets. Passes through to
      :func:`~probpipe.inference._approximate_distribution.make_posterior`
      so the posterior names its fields by the prior's components.

    Three cases:

    1. **A parametric family prior**: ``target_flat_fn`` composes
       :func:`build_target_log_prob` with the reshape of the flat vector to
       the family's event, and the prior's declaration is returned for
       downstream lift-back.
    2. **A record-shaped prior or target**, such as a factored joint or the
       unnormalized conditional of one: ``target_flat_fn`` unflattens the
       vector to the numeric record the target declares, whose declaration is
       returned.
    3. **Bare ``SupportsLogProb`` target** with no Record-shaped prior
       (e.g., a hand-rolled ``Distribution`` subclass implementing
       ``_unnormalized_log_prob`` over a flat ``Array``). The target
       already takes a flat input; no flattening is needed.
       ``event_spec`` is returned as ``None``.

    Intended for use by BlackJAX-flavoured MCMC / VI backends.
    """
    prior = dist
    target_record = build_target_log_prob(dist, observed)
    flat_init = get_init_state(dist, init, random_seed=random_seed)

    flat_prior = _flat_view(prior)
    if flat_prior is not None:

        def target_flat(theta_flat: Array) -> Array:
            return target_record(flat_prior(theta_flat))

        return target_flat, flat_init, prior.event_spec

    record = flat_record(prior)
    if record is not None:

        def target_unflattened(theta_flat: Array) -> Array:
            return target_record(_reconstruct_from_vector(prior.label, record, theta_flat))

        return target_unflattened, flat_init, prior.event_spec

    # Bare array-shaped target: ``target_record`` already accepts a
    # flat array and no template is available to lift the chain.
    return target_record, flat_init, None


def _flat_leaves(law: Any) -> list[tuple[tuple[int, ...], Any]] | None:
    """The shape and declared support of each leaf a flat chain over *law* lays out, in its order.

    ``None`` when *law* declares no array or numeric record that a flat vector
    lays out, as a bare target over a flat array does.
    """
    if _has_flat_view(law):
        spec = law.event_spec.spec
        if isinstance(spec, NumericArraySpec):
            return [(tuple(spec.shape), spec.support)]
        return None
    record = flat_record(law)
    if record is not None:
        return [(tuple(record[path].shape), record[path].support) for path in record]
    array = _one_array(law)
    if array is not None:
        return [(tuple(array.shape), array.support)]
    return None


class _LeafMap(NamedTuple):
    """One leaf's segment of a flat state: its sizes, shapes, and bijector, if any."""

    size: int
    shape: tuple[int, ...]
    unconstrained_size: int
    unconstrained_shape: tuple[int, ...]
    bijector: Any


class UnconstrainedCoordinates(NamedTuple):
    """The maps between flat states of a law and its unconstrained coordinates, each at one point.

    ``forward`` maps unconstrained coordinates to a flat state in the support,
    ``inverse`` maps a flat state to its preimage, which is not finite outside
    the support, and ``log_jacobian`` is the log-determinant of the forward
    map's Jacobian. ``size`` is the number of unconstrained coordinates.
    """

    size: int
    forward: Callable[[Array], Array]
    inverse: Callable[[Array], Array]
    log_jacobian: Callable[[Array], Array]


def _leaf_maps(law: Any) -> list[_LeafMap] | None:
    """The segment map of each leaf of a flat state of *law*, by the rule of :func:`unconstrained_coordinates`.

    ``None`` when *law* declares no array or numeric record that a flat vector
    lays out.
    """
    from ..core._dispatch import MathematicalDomainError, ResolutionError
    from ..core.constraints import real
    from ..functions import bijector_for

    leaves = _flat_leaves(law)
    if leaves is None:
        return None
    maps: list[_LeafMap] = []
    for shape, support in leaves:
        size = int(np.prod(shape, dtype=int))
        bijector = None
        if support is not None and not isinstance(support, type(real)):
            try:
                bijector = bijector_for(support)
            except (MathematicalDomainError, ResolutionError):
                bijector = None
        if bijector is None:
            maps.append(_LeafMap(size, shape, size, shape, None))
            continue
        point = jax.ShapeDtypeStruct(shape, jnp.float32)
        unconstrained = tuple(jax.eval_shape(bijector._inverse, point).shape)
        maps.append(
            _LeafMap(size, shape, int(np.prod(unconstrained, dtype=int)), unconstrained, bijector)
        )
    return maps


def _segments(vector: Array, sizes: Iterable[int]) -> list[Array]:
    """*vector*'s consecutive segments of *sizes*, along its last axis."""
    bounds = np.cumsum([0, *sizes])
    return [vector[..., start:stop] for start, stop in itertools.pairwise(bounds)]


def _coordinates(maps: list[_LeafMap]) -> UnconstrainedCoordinates:
    """The maps that move each leaf of *maps* by its bijector and keep the others' coordinates."""
    sizes = [leaf.size for leaf in maps]
    unconstrained_sizes = [leaf.unconstrained_size for leaf in maps]

    def forward(z: Array) -> Array:
        parts = []
        for leaf, segment in zip(maps, _segments(z, unconstrained_sizes)):
            if leaf.bijector is None:
                parts.append(segment)
                continue
            value = leaf.bijector.raw()(jnp.reshape(segment, leaf.unconstrained_shape))
            parts.append(jnp.reshape(value, (leaf.size,)))
        return jnp.concatenate(parts)

    def inverse(x: Array) -> Array:
        parts = []
        for leaf, segment in zip(maps, _segments(x, sizes)):
            if leaf.bijector is None:
                parts.append(segment)
                continue
            preimage = leaf.bijector._inverse(jnp.reshape(segment, leaf.shape))
            parts.append(jnp.reshape(preimage, (leaf.unconstrained_size,)))
        return jnp.concatenate(parts)

    def log_jacobian(z: Array) -> Array:
        total = jnp.zeros(())
        for leaf, segment in zip(maps, _segments(z, unconstrained_sizes)):
            if leaf.bijector is not None:
                point = jnp.reshape(segment, leaf.unconstrained_shape)
                total = total + leaf.bijector._log_det_jacobian(point)
        return total

    return UnconstrainedCoordinates(sum(unconstrained_sizes), forward, inverse, log_jacobian)


def unconstrained_coordinates(law: Any) -> UnconstrainedCoordinates:
    """The unconstrained coordinates of a flat state of *law*.

    A leaf whose declared support is other than the reals takes the bijector
    :func:`~probpipe.bijector_for` gives it, which maps the leaf's unconstrained
    coordinates onto the support. A leaf of the reals, of no declared support,
    or of a support with no smooth bijector, such as a discrete one, keeps its
    coordinates.

    Raises
    ------
    TypeError
        If *law* declares no array or numeric record that a flat vector lays out.
    """
    maps = _leaf_maps(law)
    if maps is None:
        raise TypeError(f"{type(law).__name__} {law.label!r} draws no value a flat vector lays out")
    return _coordinates(maps)


def unconstrained_chain(
    density: Callable[[Array], Array], init: Array, law: Any
) -> tuple[Callable[[Array], Array], Array, Callable[[Array], Array]]:
    """The density, initial state, and draw map of a flat chain run in unconstrained coordinates.

    The chain runs in the coordinates :func:`unconstrained_coordinates` gives
    *law*, so its state ranges over all of ``R^n`` and every draw lies in the
    support. The density in these coordinates adds the forward map's
    log-Jacobian. An initial coordinate outside a leaf's support has no
    preimage, and that leaf starts at the preimage of the bijector's center,
    the origin.

    Parameters
    ----------
    density : callable
        The log-density at a flat state of *law*'s leaves.
    init : Array
        The flat initial state.
    law : Distribution
        The law whose declaration lays out the flat state.

    Returns
    -------
    tuple
        ``(density, init, constrain)``: the log-density at an unconstrained
        state, the unconstrained initial state, and the map from unconstrained
        states, with any leading axes, to flat states of *law*. Every leaf
        keeping its coordinates gives *density*, *init*, and the identity.
    """
    maps = _leaf_maps(law)
    if maps is None or all(leaf.bijector is None for leaf in maps):
        return density, init, lambda states: states
    coordinates = _coordinates(maps)

    def unconstrained_density(z: Array) -> Array:
        return density(coordinates.forward(z)) + coordinates.log_jacobian(z)

    parts = []
    for leaf, segment in zip(maps, _segments(jnp.asarray(init), [m.size for m in maps])):
        if leaf.bijector is None:
            parts.append(segment)
            continue
        preimage = jnp.ravel(leaf.bijector._inverse(jnp.reshape(segment, leaf.shape)))
        finite = bool(jnp.all(jnp.isfinite(preimage)))
        parts.append(preimage if finite else jnp.zeros(leaf.unconstrained_size, preimage.dtype))
    unconstrained_init = jnp.concatenate(parts)

    def constrain(states: Array) -> Array:
        states = jnp.asarray(states)
        flat = jnp.reshape(states, (-1, states.shape[-1]))
        return jnp.reshape(jax.vmap(coordinates.forward)(flat), (*states.shape[:-1], -1))

    return unconstrained_density, unconstrained_init, constrain


# ---------------------------------------------------------------------------
# ArviZ DataTree builder (backend-agnostic)
# ---------------------------------------------------------------------------


def build_mcmc_datatree(
    chains: list[Array],
    sample_stats: dict[str, np.ndarray] | None = None,
    warmup_chains: list[Array] | None = None,
) -> DataTree:
    """Build an arviz-convention DataTree from MCMC chains + diagnostics.

    Groups: ``posterior``, ``sample_stats`` (if provided), ``warmup``
    (if provided). Backend-agnostic — consumed by both the TFP and
    BlackJAX MCMC paths. The ``posterior`` and ``warmup`` groups hold the
    flat draws as the one variable ``params``, which :func:`make_posterior`
    names by the target's leaves.
    """
    import arviz_base as azb
    import xarray as xr

    def _stack(chain_list):
        return np.stack([np.asarray(c) for c in chain_list], axis=0)

    # arviz 1.x: positional ``{group: {var: array}}`` -> ``xarray.DataTree``.
    groups: dict = {"posterior": {"params": _stack(chains)}}
    if sample_stats:
        groups["sample_stats"] = sample_stats
    dt = azb.from_dict(groups)

    if warmup_chains is not None and all(w is not None for w in warmup_chains):
        warmup_array = _stack(warmup_chains)
        n_chains, n_warmup = warmup_array.shape[:2]
        event_dims = [f"params_dim_{i}" for i in range(warmup_array.ndim - 2)]
        dims = ["chain", "draw", *event_dims]
        warmup_ds = xr.Dataset(
            {
                "params": xr.DataArray(
                    warmup_array,
                    dims=dims,
                    coords={"chain": np.arange(n_chains), "draw": np.arange(n_warmup)},
                ),
            }
        )
        dt["warmup"] = xr.DataTree(dataset=warmup_ds)

    return dt


# ---------------------------------------------------------------------------
# Shared per-chain scan loop
# ---------------------------------------------------------------------------


def run_chain_scan(
    sampler: Any,
    init_state: Any,
    num_results: int,
    key: Array,
) -> tuple[Array, Any]:
    """Step a BlackJAX-style sampler under ``jax.lax.scan``.

    Drives the standard ``state, info = sampler.step(key, state)``
    contract for ``num_results`` iterations. Returns
    ``(positions, infos)`` where ``positions`` has shape
    ``(num_results, *event_shape)`` and ``infos`` is a pytree of
    per-step info objects stacked along the leading axis. The caller
    is responsible for packing ``infos`` into whatever output shape
    its consumer expects (e.g. an ArviZ-flavoured ``sample_stats``
    dict).

    Used by both the NUTS / HMC and the RWMH / ESS backends; the
    contract is identical because BlackJAX samplers share it.
    """

    def one_step(state, step_key):
        state, info = sampler.step(step_key, state)
        return state, (state.position, info)

    keys = jax.random.split(key, num_results)
    _, (positions, infos) = jax.lax.scan(one_step, init_state, keys)
    return positions, infos


# ---------------------------------------------------------------------------
# Multi-chain parallel dispatch
# ---------------------------------------------------------------------------


def parallel_chain_map(fn: Callable[[Array], Any], chain_keys: Array) -> Any:
    """Run ``fn`` across ``chain_keys`` using the best parallelism available.

    Picks between three strategies:

    * **Single chain** (``num_chains == 1``): apply ``fn`` directly to
      the lone key and add a leading axis. Skips both ``pmap`` and
      ``vmap`` since neither earns its tracing / dispatch cost for a
      one-element batch.
    * ``jax.pmap`` when ``num_chains >= 2`` and
      :func:`jax.local_device_count` >= ``num_chains``. Each chain runs
      independently on its own device — bit-identical to a single-chain
      sequential run at the same seed, with full per-device parallelism
      on GPU/TPU or on a CPU configured with multiple virtual devices
      (``XLA_FLAGS=--xla_force_host_platform_device_count=N``).
    * ``jax.vmap`` otherwise. Cheaper SIMD-style vectorization, no
      extra memory, but: (a) only a single core's worth of throughput
      on CPU, and (b) for kernels with data-dependent control flow
      like NUTS, vmap has to mask-pad divergent trajectories, so the
      per-chain draws no longer match the sequential reference at the
      same seed.

    Intended for top-of-runner use — the ``int(chain_keys.shape[0])``
    read requires a concrete shape and so is not safe inside ``jit``
    or ``scan``.

    Returns whatever ``fn`` returns, with a leading ``num_chains``
    axis. Both ``pmap`` and ``vmap`` backends produce the same logical
    output shape; pmap returns sharded arrays that downstream code can
    index per-chain transparently.
    """
    num_chains = int(chain_keys.shape[0])
    if num_chains == 1:
        return jax.tree.map(lambda a: a[None], fn(chain_keys[0]))
    if jax.local_device_count() >= num_chains:
        return jax.pmap(fn)(chain_keys)
    return jax.vmap(fn)(chain_keys)
