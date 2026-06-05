"""Backend-agnostic inference utilities.

Functions for building target log-density callables and initial chain
states from a :class:`~probpipe.core.distribution.Distribution` plus
observed data. Shared across every inference backend in
``probpipe.inference`` so they consume the same source of truth.

Two target builders:

- :func:`build_target_log_prob` returns a Record-shaped target
  (the TFP-flavoured interface).
- :func:`build_target_log_prob_flat` returns a flat-vector target —
  the BlackJAX entry point. It wraps the Record-shaped target through
  the prior's
  :meth:`~probpipe.core._numeric_record_distribution.NumericRecordDistribution.as_flat_distribution`
  view so kernels that operate on flat parameter vectors plug in
  without per-backend flatten / unflatten plumbing.

Scope: private to ``probpipe.inference``. Symbols are package-private
utilities shared across the backend modules; not re-exported through
``probpipe.inference.__init__`` or the top-level package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xarray import DataTree

import logging

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

from ..core.distribution import Distribution
from ..core.protocols import SupportsSampling
from ..core.record import Record, RecordTemplate
from ..custom_types import Array, ArrayLike


__all__ = [
    "as_prng_key",
    "build_likelihood_flat",
    "build_mcmc_datatree",
    "build_target_log_prob",
    "build_target_log_prob_flat",
    "extract_chain_columns",
    "get_init_state",
    "get_prior",
    "extract_record_template",
    "is_jax_traceable",
    "is_simple_model",
    "parallel_chain_map",
    "run_chain_scan",
]


# ---------------------------------------------------------------------------
# Chain extraction
# ---------------------------------------------------------------------------

def extract_chain_columns(
    trace: Any, names: list[str], num_chains: int,
) -> list[Array]:
    """Per-chain flat sample matrices from an ArviZ-like trace.

    For each chain ``c`` and each variable in *names* (in that order),
    pulls ``trace.posterior[name].values[c]``, flattens trailing axes to
    2-D ``(draws, -1)``, and concatenates across variables. Columns are
    laid out in *names* order, to match the ``record_template`` field
    order.

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

    Pass the full target (a ``SimpleModel`` or a bare ``Distribution``);
    this helper calls :func:`get_prior` internally and works against
    the prior — that is the parameter-space distribution from which
    init candidates should be drawn.

    Resolution order:

    1. Explicit ``init`` — trusted, returned verbatim (cast to the
       prior's dtype).
    2. **Prior sample** — if the prior implements ``SupportsSampling``,
       draw a single sample with the supplied ``random_seed``. For a
       ``RecordDistribution`` the sample is flattened to a numeric
       vector via ``NumericRecord``.
    3. **Stan default** — if the prior has no sampling path but
       exposes ``event_shape``, return a coordinate-wise
       ``Uniform(-2, 2)`` draw, matching Stan's default init for
       unconstrained parameters. The gradient-based MCMC methods this
       helper feeds all assume an unconstrained parameter space, so
       the box is guaranteed to be inside the support.
    4. Raise — no init heuristic applies.

    Observed data is deliberately not consulted: a ``mean(observed)``
    heuristic would live in the *data* space while the chain state
    lives in the *parameter* space, and the two coincide only for
    pure location models. Callers that genuinely need a data-derived
    init should pass ``init=`` explicitly.
    """
    prior = get_prior(dist)

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
            if isinstance(s, Record):
                from ..core._numeric_record import NumericRecord
                if not isinstance(s, NumericRecord):
                    s = NumericRecord.from_record(s)
                s = s.flatten()
            return jnp.atleast_1d(jnp.asarray(s, dtype=target_dtype))
        except Exception:
            logger.debug(
                "get_init_state: prior._sample failed for %r; falling "
                "back to Uniform(-2, 2)", prior, exc_info=True,
            )

    if hasattr(prior, "event_shape"):
        return jax.random.uniform(
            key, shape=prior.event_shape,
            minval=-2.0, maxval=2.0, dtype=target_dtype,
        )

    raise ValueError(
        "Cannot determine initial state: pass init= explicitly, or "
        "provide a distribution whose prior implements "
        "SupportsSampling or exposes event_shape."
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

    Uses ``getattr`` to tolerate priors that aren't a
    ``RecordDistribution`` (e.g. bare ``SupportsLogProb`` targets);
    SimpleModel-rooted callers can rely on the prior being a
    ``RecordDistribution`` and read ``prior.record_template`` directly.
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

    Three cases, in the order the body dispatches them:

    1. **SimpleModel** (has prior + likelihood):
       ``prior._log_prob(params) + likelihood.log_likelihood(params, data)``.
    2. **Bare target with data**: joint over ``(params, data)``,
       evaluated as ``dist._unnormalized_log_prob((params, data))``.
    3. **Bare target without data**: ``dist._unnormalized_log_prob``
       returned directly (the caller is presumed to have already
       folded the data into the distribution, e.g. via closure).

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
    random_seed: int | Array = 0,
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

    1. **Record-shaped prior** (a :class:`~probpipe.core._numeric_record_distribution.NumericRecordDistribution`
       — every ``SimpleModel`` prior is one). ``target_flat_fn``
       composes :func:`build_target_log_prob` with the prior's
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
    flat_init = get_init_state(dist, init, random_seed=random_seed)

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


def build_likelihood_flat(
    prior: Distribution,
    likelihood: Any,
    data: ArrayLike | Record | None,
) -> Callable[[Array], Array]:
    """Build a flat-vector ``loglikelihood_fn(theta_flat)`` from a prior +
    likelihood + data.

    Unlike :func:`build_target_log_prob_flat` (which builds the *joint*
    prior + likelihood density), this returns the *likelihood alone* as
    a function of a flat parameter vector. Elliptical slice sampling
    folds the Gaussian prior into the proposal mechanism, so it needs
    the likelihood by itself.

    Two cases:

    - **Record-shaped prior** (any ``SimpleModel`` prior): the flat
      vector unflattens through the prior's
      :meth:`~probpipe.core._numeric_record_distribution.FlatNumericRecordDistribution.unflatten_sample`
      so the likelihood sees structured ``Record``-shaped params.
    - **Bare-array prior**: the likelihood already accepts a flat
      vector, so it is called directly.
    """
    flat_view = getattr(prior, "as_flat_distribution", None)
    record_template = getattr(prior, "record_template", None)
    if flat_view is not None and record_template is not None:
        flat_prior = flat_view()

        def loglikelihood_fn(theta_flat: Array) -> Array:
            params = flat_prior.unflatten_sample(theta_flat)
            return likelihood.log_likelihood(params=params, data=data)

        return loglikelihood_fn

    def loglikelihood_fn(theta_flat: Array) -> Array:
        return likelihood.log_likelihood(params=theta_flat, data=data)

    return loglikelihood_fn


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
