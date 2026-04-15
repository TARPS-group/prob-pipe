"""TFP-backed inference methods: NUTS and HMC."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from xarray import DataTree

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.mcmc as tfp_mcmc

from ..core._registry import MethodInfo
from ..core.distribution import Distribution
from ..core.protocols import SupportsLogProb, SupportsMean
from ..custom_types import Array, ArrayLike
from ..core.record import Record, RecordTemplate
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._registry import InferenceMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_jax_traceable(fn: Callable, init_state: jnp.ndarray) -> bool:
    """Probe whether *fn* can be traced by JAX."""
    try:
        jax.make_jaxpr(fn)(init_state)
        return True
    except Exception:
        return False


def _get_init_state(
    dist: Distribution, init: ArrayLike | None, data: ArrayLike | None,
) -> jnp.ndarray:
    """Determine an initial chain state from the distribution or data."""
    if init is not None:
        return jnp.atleast_1d(jnp.asarray(init, dtype=jnp.float32))

    if isinstance(dist, SupportsMean):
        try:
            m = dist._mean()
            if isinstance(m, Record):
                m = m.flatten()
            return jnp.atleast_1d(jnp.asarray(m, dtype=jnp.float32))
        except Exception:
            pass

    if data is not None:
        return jnp.atleast_1d(jnp.mean(jnp.asarray(data), axis=0))

    if hasattr(dist, "event_shape"):
        return jnp.zeros(dist.event_shape, dtype=jnp.float32)

    raise ValueError(
        "Cannot determine initial state: provide init=, "
        "a distribution with SupportsMean, or observed data."
    )


def _is_simple_model(dist: Distribution) -> bool:
    """Check if *dist* is a SimpleModel (lazy import to avoid circularity)."""
    from ..modeling._simple import SimpleModel
    return isinstance(dist, SimpleModel)


def _build_target_log_prob(dist: Distribution, observed: ArrayLike | Record | None) -> Callable[[jnp.ndarray], Array]:
    """Build a target_log_prob_fn(params) from *dist* and *observed*.

    Handles three cases:

    1. **SimpleModel** (has prior + likelihood):
       ``prior._log_prob(params) + likelihood.log_likelihood(params, data)``
    2. **Bare SupportsLogProb with data**: ``dist._log_prob(params)``
       (data is assumed already folded in, e.g., via closure).
    3. **Joint over (params, data)**: ``dist._log_prob((params, data))``

    Observed data is passed through to the likelihood as-is (may be a
    raw array, a ``Record`` object, or a dict — the likelihood handles
    its own input types).
    """
    if _is_simple_model(dist):
        data = observed
        # Pre-resolve Record fields to raw arrays so that JAX tracing
        # doesn't trigger Record's lazy _resolve (which causes tracer leaks).
        from ..core.record import Record
        if isinstance(data, Record):
            data = Record({f: jnp.asarray(data[f]) for f in data.fields})

        def target_log_prob_fn(params):
            lp = dist._prior._log_prob(params)
            if data is not None:
                lp = lp + dist._likelihood.log_likelihood(params=params, data=data)
            return lp

        return target_log_prob_fn

    if observed is not None:
        return lambda params: dist._log_prob((params, observed))

    return dist._log_prob


def _run_tfp_chains(
    target_log_prob_fn: Callable,
    init_state: jnp.ndarray,
    *,
    algorithm: str,
    num_results: int,
    num_warmup: int,
    num_chains: int,
    step_size: float,
    random_seed: int,
) -> tuple[list[Array], dict[str, np.ndarray]]:
    """Run TFP-backed MCMC chains.

    Returns (chains, sample_stats_dict) where sample_stats_dict contains
    arrays shaped (num_chains, num_results) for building DataTree.
    """
    if algorithm == "nuts":
        inner_kernel = tfp_mcmc.NoUTurnSampler(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
        )
    elif algorithm == "hmc":
        inner_kernel = tfp_mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=10,
        )
    else:
        raise ValueError(f"algorithm must be 'nuts' or 'hmc', got {algorithm!r}")

    num_adapt = int(0.8 * num_warmup) if num_warmup > 0 else 0
    if num_adapt > 0:
        kernel = tfp_mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=inner_kernel,
            num_adaptation_steps=num_adapt,
            target_accept_prob=0.75,
        )
    else:
        kernel = inner_kernel

    key = jax.random.PRNGKey(random_seed)
    chain_keys = jax.random.split(key, num_chains)

    def _run_one_chain(chain_key):
        return tfp_mcmc.sample_chain(
            num_results=num_results,
            current_state=init_state,
            kernel=kernel,
            num_burnin_steps=num_warmup,
            seed=chain_key,
            trace_fn=lambda _, kr: kr,
        )

    all_samples, all_traces = jax.vmap(_run_one_chain)(chain_keys)
    # all_samples: (num_chains, num_results, *event_shape)
    chains = [all_samples[c] for c in range(num_chains)]

    sample_stats = _extract_sample_stats(all_traces, num_chains)
    return chains, sample_stats


def _extract_sample_stats(traces: Any, num_chains: int) -> dict[str, np.ndarray]:
    """Extract sample stats arrays from TFP traces.

    Returns dict of numpy arrays shaped (num_chains, num_draws).
    """
    results = traces
    stats: dict[str, np.ndarray] = {}

    if hasattr(results, "new_step_size"):
        stats["step_size"] = np.asarray(results.new_step_size)
        results = results.inner_results
    elif hasattr(results, "step_size"):
        stats["step_size"] = np.asarray(results.step_size)

    log_ar = getattr(results, "log_accept_ratio", None)
    if log_ar is not None:
        ar = np.asarray(jnp.exp(jnp.minimum(log_ar, 0.0)))
        stats["acceptance_rate"] = ar

    is_accepted = getattr(results, "is_accepted", None)
    if is_accepted is not None:
        stats["is_accepted"] = np.asarray(is_accepted)

    return stats


# ---------------------------------------------------------------------------
# Auxiliary DataTree builder (MCMC-specific)
# ---------------------------------------------------------------------------


def _build_mcmc_datatree(
    chains: list[Array],
    sample_stats: dict[str, np.ndarray] | None = None,
    warmup_chains: list[Array] | None = None,
) -> DataTree:
    """Build an arviz-convention DataTree from MCMC chains + diagnostics.

    Groups: ``posterior``, ``sample_stats`` (if provided), ``warmup``
    (if provided).
    """
    import arviz as az
    import xarray as xr

    def _stack(chain_list):
        return np.stack([np.asarray(c) for c in chain_list], axis=0)

    posterior_dict = {"params": _stack(chains)}
    # arviz 1.x from_dict takes a positional groups dict;
    # arviz 0.x takes keyword arguments (posterior=, sample_stats=, ...).
    import inspect
    sig = inspect.signature(az.from_dict)
    first_param = next(iter(sig.parameters))
    if first_param == "posterior":
        # arviz 0.x keyword-style API
        kw: dict = {"posterior": posterior_dict}
        if sample_stats:
            kw["sample_stats"] = sample_stats
        dt = az.from_dict(**kw)
    else:
        # arviz 1.x groups-dict API
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
# Helpers for prior extraction
# ---------------------------------------------------------------------------

def _get_prior(dist: Distribution) -> Distribution:
    """Extract the prior from a model, or return dist itself."""
    return dist._prior if _is_simple_model(dist) else dist


def _extract_record_template(dist: Distribution) -> RecordTemplate | None:
    """Return the RecordTemplate from *dist*'s prior, or ``None``."""
    prior = _get_prior(dist)
    return prior.record_template


# ---------------------------------------------------------------------------
# Inference methods
# ---------------------------------------------------------------------------


class _TFPGradientMethod(InferenceMethod):
    """Base for TFP gradient-based MCMC methods (NUTS, HMC)."""

    def __init__(self, algorithm: str, method_name: str, method_priority: int):
        self._algorithm = algorithm
        self._method_name = method_name
        self._method_priority = method_priority

    @property
    def name(self) -> str:
        return self._method_name

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    @property
    def priority(self) -> int:
        return self._method_priority

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        # Intentionally probes JAX traceability (via jax.make_jaxpr) to avoid
        # selecting a gradient-based method that would fail at execute() time.
        # The cost is ~one JAX trace, cached by JAX on subsequent calls.
        if not isinstance(dist, SupportsLogProb):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires SupportsLogProb")
        try:
            target = _build_target_log_prob(dist, observed)
            init = _get_init_state(_get_prior(dist), kwargs.get("init"), observed)
            if not _is_jax_traceable(target, init):
                return MethodInfo(feasible=False, method_name=self.name,
                                  description="Log-prob is not JAX-traceable")
        except Exception as e:
            return MethodInfo(feasible=False, method_name=self.name,
                              description=str(e))
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        target = _build_target_log_prob(dist, observed)
        prior = _get_prior(dist)
        init = _get_init_state(prior, kwargs.get("init"), observed)
        record_template = _extract_record_template(dist)

        num_results = kwargs.get("num_results", 1000)
        num_warmup = kwargs.get("num_warmup", 500)
        num_chains = kwargs.get("num_chains", 1)

        chains, sample_stats = _run_tfp_chains(
            target, init,
            algorithm=self._algorithm,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=num_chains,
            step_size=kwargs.get("step_size", 0.1),
            random_seed=kwargs.get("random_seed", 0),
        )
        auxiliary = _build_mcmc_datatree(chains, sample_stats)
        return make_posterior(
            chains, parents=(prior,), algorithm=self._method_name,
            auxiliary=auxiliary, record_template=record_template,
            num_results=num_results, num_warmup=num_warmup, num_chains=num_chains,
        )


def TFPNutsMethod() -> _TFPGradientMethod:
    """TFP No-U-Turn Sampler (gradient-based MCMC)."""
    return _TFPGradientMethod("nuts", "tfp_nuts", 100)


def TFPHmcMethod() -> _TFPGradientMethod:
    """TFP Hamiltonian Monte Carlo."""
    return _TFPGradientMethod("hmc", "tfp_hmc", 90)
