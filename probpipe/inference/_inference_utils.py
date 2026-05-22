"""Backend-agnostic inference utilities.

Functions for building target log-density callables and initial chain
states from a :class:`~probpipe.core.distribution.Distribution` plus
observed data. Lifted out of ``_tfp_mcmc.py`` so both the TFP and
BlackJAX MCMC paths consume the same source of truth, and so future
backends (VI, SMC, Laplace) get the same machinery.

The Record-shaped path (:func:`build_target_log_prob`) is the original
TFP-flavoured interface. The flat-vector path
(:func:`build_target_log_prob_flat`) is the BlackJAX entry point — it
wraps the Record-shaped target through the prior's
:meth:`~probpipe.core._numeric_record_distribution.NumericRecordDistribution.as_flat_distribution`
view so kernels that operate on flat parameter vectors can plug in
without per-backend flatten / unflatten plumbing.

Scope: private to ``probpipe.inference``. Symbols are package-private
utilities shared across the backend modules (``_tfp_mcmc.py``,
``_rwmh.py``, ``_blackjax_sgmcmc.py``, and the forthcoming
``_blackjax_mcmc.py``); they are not re-exported through
``probpipe.inference.__init__`` or the top-level package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xarray import DataTree

import jax
import jax.numpy as jnp
import numpy as np

from ..core.distribution import Distribution
from ..core.protocols import SupportsMean
from ..core.record import Record, RecordTemplate
from ..custom_types import Array, ArrayLike


__all__ = [
    "build_mcmc_datatree",
    "build_target_log_prob",
    "build_target_log_prob_flat",
    "get_init_state",
    "get_prior",
    "extract_record_template",
    "is_jax_traceable",
    "is_simple_model",
]


# ---------------------------------------------------------------------------
# JAX-traceability probe
# ---------------------------------------------------------------------------

def is_jax_traceable(fn: Callable, init_state: jnp.ndarray) -> bool:
    """Probe whether *fn* can be traced by JAX at *init_state*.

    Used by gradient-based MCMC `check()` methods to filter out targets
    that would fail at execute() time. The cost is ~one JAX trace,
    cached by JAX on subsequent calls.
    """
    try:
        jax.make_jaxpr(fn)(init_state)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Initial-state heuristics
# ---------------------------------------------------------------------------

def get_init_state(
    dist: Distribution, init: ArrayLike | None, observed: ArrayLike | None,
) -> jnp.ndarray:
    """Determine an initial chain state.

    Resolution order: explicit ``init`` > distribution mean (via the
    ``SupportsMean`` protocol) > zeros of ``dist.event_shape``.
    Inherits dtype from the target distribution when available so
    ``log_prob`` / ``sample`` stay self-consistent under JAX x64.

    ``observed`` is accepted for signature uniformity with the other
    backend-agnostic helpers but is intentionally not used as an init
    source: ``mean(observed)`` lives in the *data* space, while the
    chain state lives in the *parameter* space. The two coincide only
    for pure location models (e.g. ``y_i ~ N(theta, sigma**2)`` with
    parameter ``theta``); for anything else — regression coefficients,
    scale parameters, latent variables, non-Gaussian likelihoods — the
    heuristic puts the init on the wrong manifold and the chain has to
    burn warmup correcting it. Callers that genuinely need an init in
    the data space should pass ``init=`` explicitly.
    """
    del observed  # accepted for signature uniformity; see docstring

    target_dtype = getattr(dist, "dtype", None)
    if not isinstance(target_dtype, jnp.dtype):
        from .._dtype import _default_float_dtype
        target_dtype = _default_float_dtype()

    if init is not None:
        return jnp.atleast_1d(jnp.asarray(init, dtype=target_dtype))

    if isinstance(dist, SupportsMean):
        try:
            m = dist._mean()
            if isinstance(m, Record):
                from ..core._numeric_record import NumericRecord
                if not isinstance(m, NumericRecord):
                    m = NumericRecord.from_record(m)
                m = m.flatten()
            return jnp.atleast_1d(jnp.asarray(m, dtype=target_dtype))
        except Exception:
            pass

    if hasattr(dist, "event_shape"):
        return jnp.zeros(dist.event_shape, dtype=target_dtype)

    raise ValueError(
        "Cannot determine initial state: pass init= explicitly, or "
        "provide a distribution that implements SupportsMean or "
        "exposes event_shape."
    )


# ---------------------------------------------------------------------------
# SimpleModel detection and prior extraction
# ---------------------------------------------------------------------------

def is_simple_model(dist: Distribution) -> bool:
    """Check whether *dist* is a SimpleModel (lazy import for circularity)."""
    from ..modeling._simple import SimpleModel
    return isinstance(dist, SimpleModel)


def get_prior(dist: Distribution) -> Distribution:
    """Return the prior of a model, or *dist* itself for non-model targets."""
    return dist._prior if is_simple_model(dist) else dist


def extract_record_template(dist: Distribution) -> RecordTemplate | None:
    """Return *dist*'s prior's ``record_template``, or ``None``.

    Uses ``getattr`` so a non-``RecordDistribution`` prior (which has
    no ``record_template`` after the PR #200 hierarchy cleanup) yields
    ``None`` rather than ``AttributeError``. SimpleModel-rooted callers
    can rely on the prior being a ``RecordDistribution`` and read
    ``prior.record_template`` directly; this helper exists for the
    bare ``SupportsLogProb`` paths.
    """
    prior = get_prior(dist)
    return getattr(prior, "record_template", None)


# ---------------------------------------------------------------------------
# Target log-density construction
# ---------------------------------------------------------------------------

def build_target_log_prob(
    dist: Distribution, observed: ArrayLike | Record | None,
) -> Callable[[Any], Array]:
    """Build a ``target_log_prob_fn(params)`` from *dist* and *observed*.

    Three cases:

    1. **SimpleModel** (has prior + likelihood):
       ``prior._log_prob(params) + likelihood.log_likelihood(params, data)``.
    2. **Bare ``SupportsUnnormalizedLogProb`` with data**:
       ``dist._unnormalized_log_prob(params)`` (data is assumed already
       folded in, e.g., via closure).
    3. **Joint over (params, data)**:
       ``dist._unnormalized_log_prob((params, data))``.

    The unnormalized accessor is used for cases 2 and 3 because MCMC
    samplers do not require a normalized density. Distributions that
    only implement ``_log_prob`` are unaffected: the
    ``SupportsUnnormalizedLogProb`` protocol provides a default
    ``_unnormalized_log_prob`` that delegates to ``_log_prob``.

    Observed data is passed through to the likelihood as-is (may be a
    raw array, a ``Record`` object, or a dict — the likelihood handles
    its own input types).
    """
    if is_simple_model(dist):
        def target_log_prob_fn(params):
            lp = dist._prior._log_prob(params)
            if observed is not None:
                lp = lp + dist._likelihood.log_likelihood(
                    params=params, data=observed,
                )
            return lp

        return target_log_prob_fn

    if observed is not None:
        return lambda params: dist._unnormalized_log_prob((params, observed))

    return dist._unnormalized_log_prob


def build_target_log_prob_flat(
    dist: Distribution, observed: ArrayLike | Record | None,
    *,
    init: ArrayLike | None = None,
) -> tuple[Callable[[Array], Array], Array, RecordTemplate | None]:
    """Build a flat-vector target + initial state + (optional) record template.

    Returns ``(target_flat_fn, flat_init, record_template)``:

    - ``target_flat_fn(theta_flat) -> log_prob``: a callable that
      consumes a flat parameter vector.
    - ``flat_init``: the flat-vector initial chain state from
      :func:`get_init_state`.
    - ``record_template``: the prior's ``record_template`` when the
      target's parameter space is Record-shaped; ``None`` for bare
      array-shaped targets. Passes through to
      :func:`~probpipe.inference._approximate_distribution.make_posterior`
      so the posterior preserves the structured parameterisation
      when available.

    Two cases:

    1. **Record-shaped prior** (a :class:`~probpipe.core._numeric_record_distribution.NumericRecordDistribution`,
       which a ``SimpleModel`` prior is required to be under the PR
       #200 hierarchy cleanup). ``target_flat_fn`` composes
       :func:`build_target_log_prob` with the prior's
       :meth:`~probpipe.core._numeric_record_distribution.FlatNumericRecordDistribution.unflatten_sample`,
       and the record template is returned for downstream lift-back.
    2. **Bare ``SupportsLogProb`` target** with no Record-shaped prior
       (e.g., a hand-rolled ``Distribution`` subclass implementing
       ``_unnormalized_log_prob`` over a flat ``Array``). The target
       already takes a flat input; no flattening is needed.
       ``record_template`` is returned as ``None``.

    Intended for use by BlackJAX-flavoured MCMC / VI backends.
    """
    prior = get_prior(dist)
    target_record = build_target_log_prob(dist, observed)
    flat_init = get_init_state(prior, init, observed)

    flat_view = getattr(prior, "as_flat_distribution", None)
    record_template = getattr(prior, "record_template", None)
    if flat_view is not None and record_template is not None:
        flat_prior = flat_view()

        def target_flat(theta_flat: Array) -> Array:
            return target_record(flat_prior.unflatten_sample(theta_flat))

        return target_flat, flat_init, record_template

    # Bare array-shaped target: ``target_record`` already accepts a
    # flat array and no template is available to lift the chain.
    return target_record, flat_init, None


# ---------------------------------------------------------------------------
# ArviZ DataTree builder (backend-agnostic)
# ---------------------------------------------------------------------------


def build_mcmc_datatree(
    chains: list[Array],
    sample_stats: dict[str, np.ndarray] | None = None,
    warmup_chains: list[Array] | None = None,
) -> "DataTree":
    """Build an arviz-convention DataTree from MCMC chains + diagnostics.

    Groups: ``posterior``, ``sample_stats`` (if provided), ``warmup``
    (if provided). Backend-agnostic — consumed by both the TFP and
    BlackJAX MCMC paths.
    """
    import arviz as az
    import xarray as xr

    def _stack(chain_list):
        return np.stack([np.asarray(c) for c in chain_list], axis=0)

    posterior_dict = {"params": _stack(chains)}
    # arviz 1.x ``from_dict`` takes a positional groups dict; arviz 0.x
    # takes keyword arguments (``posterior=``, ``sample_stats=``, ...).
    import inspect
    sig = inspect.signature(az.from_dict)
    first_param = next(iter(sig.parameters))
    if first_param == "posterior":
        kw: dict = {"posterior": posterior_dict}
        if sample_stats:
            kw["sample_stats"] = sample_stats
        dt = az.from_dict(**kw)
    else:
        groups: dict = {"posterior": posterior_dict}
        if sample_stats:
            groups["sample_stats"] = sample_stats
        dt = az.from_dict(groups)

    if warmup_chains is not None and all(w is not None for w in warmup_chains):
        warmup_array = _stack(warmup_chains)
        n_chains, n_warmup = warmup_array.shape[:2]
        event_dims = [f"params_dim_{i}" for i in range(warmup_array.ndim - 2)]
        dims = ["chain", "draw"] + event_dims
        warmup_ds = xr.Dataset({
            "params": xr.DataArray(
                warmup_array, dims=dims,
                coords={"chain": np.arange(n_chains), "draw": np.arange(n_warmup)},
            ),
        })
        dt["warmup"] = xr.DataTree(dataset=warmup_ds)

    return dt
