"""ProbPipe-to-ProbPipe converter.

Handles conversions between ProbPipe distribution types:

- **Same-class**: returns the source unchanged (no copy, no provenance).
- **Cross-family**: moment-matches using analytical source moments when
  available. Known Monte Carlo moment implementations instead draw one
  conversion-owned batch and reuse it for every required moment.

Registered at priority 100 so it is always tried first for ProbPipe types.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from ..core._spec_base import NumericArraySpec, NumericSpec
from ..core.constraints import _supports_compatible
from ..core.provenance import Provenance
from ..core.tracked import TrackedTerm
from ..distributions._capabilities import SupportsMean
from ..distributions._distribution import Distribution, NumericDistribution
from ..distributions._empirical import EmpiricalDistribution, _batch_form, _coordinates
from ..families._backend import _allow_batched_tfp_init
from ..families._transformed import BijectorTransformedDistribution
from ..functions._result import SAMPLE_LEVEL
from ._registry import (
    _PROBPIPE_PROVIDER_ABI,
    ConversionInfo,
    ConversionMethod,
    Converter,
    _ConversionExecutionMode,
    _ConversionExecutionPlan,
    _sample_probpipe_conversion_source,
    _sampled_conversion_plan,
    _validate_conversion_sample_count,
)

# Default sample count for moment-matching conversions
DEFAULT_NUM_SAMPLES = 1024

_SAMPLED_MOMENT_BATCH_KWARG = "_sampled_moment_batch"
_EXECUTION_PLAN_KWARG = "_workflow_execution_plan"
_MOMENT_MATCH_TARGETS = frozenset(
    {
        "Normal",
        "Beta",
        "Gamma",
        "InverseGamma",
        "Exponential",
        "LogNormal",
        "StudentT",
        "Uniform",
        "Cauchy",
        "Laplace",
        "HalfNormal",
        "TruncatedNormal",
        "Bernoulli",
        "Binomial",
        "Poisson",
        "NegativeBinomial",
        "MultivariateNormal",
        "Dirichlet",
        "Multinomial",
        "VonMisesFisher",
    }
)
_TOTAL_COUNT_TARGETS = frozenset({"Binomial", "NegativeBinomial", "Multinomial"})


def _pairs_by_path(
    source: dict[str, Any], target: dict[str, Any]
) -> list[tuple[tuple[str, Any], tuple[str, Any]]] | None:
    """Pair each target leaf with the source leaf whose path holds it, or ``None``.

    A source leaf holds the target leaf of its own path and the target leaves
    under it, as a posterior's flat chunk holds a nested component's leaves. The
    pairing exists when the source leaves, in order, hold consecutive runs of
    the target leaves that together cover them all.
    """
    targets = list(target.items())
    pairs: list[tuple[tuple[str, Any], tuple[str, Any]]] = []
    i = 0
    for source_item in source.items():
        path = source_item[0]
        start = i
        while i < len(targets) and (targets[i][0] == path or targets[i][0].startswith(path + "/")):
            pairs.append((source_item, targets[i]))
            i += 1
        if i == start:
            return None
    return pairs if i == len(targets) else None


def _check_support_compatible(target: Distribution, source: Distribution) -> None:
    """Raise ``ValueError`` if *source*'s per-field supports are incompatible with *target*'s.

    Called post-construction by the converter, so both sides expose
    instance-level ``supports``, the views of a numeric law's declaration. For a single-field target (the common case),
    every source field's support is compared against the lone
    target support. For a multi-field target, supports pair up
    field-by-field in insertion order, or else a source field pairs
    with each target leaf under its path. Any other field-count
    mismatch raises ``ValueError`` rather than silently truncating
    via ``zip``.

    Sources that don't expose per-field supports (non-NRD endpoints
    like ``EmpiricalDistribution`` with object-dtype data) are
    treated as "unknown" and the check returns without complaint.
    """
    try:
        target_per_field = target.supports
        source_per_field = source.supports
    except AttributeError:
        return

    multi_leaf_source = len(source_per_field) > 1

    if len(target_per_field) == 1:
        target_support = next(iter(target_per_field.values()))
        for field_name, source_support in source_per_field.items():
            if _supports_compatible(source_support, target_support):
                continue
            field_part = f" field {field_name!r}" if multi_leaf_source else ""
            raise ValueError(
                f"Cannot convert {type(source).__name__}{field_part} "
                f"(support={source_support}) to {type(target).__name__} "
                f"(support={target_support}). "
                f"Pass check_support=False to override."
            )
        return

    # Multi-field target. Equal field counts pair positionally. Otherwise
    # a source leaf that holds a flattened group, as a posterior holds a
    # nested component, pairs with each target leaf under its path, and any
    # other mismatch raises, since ``zip`` would silently truncate.
    if len(source_per_field) == len(target_per_field):
        pairs = list(zip(source_per_field.items(), target_per_field.items()))
    else:
        pairs = _pairs_by_path(source_per_field, target_per_field)
    if pairs is None:
        raise ValueError(
            f"Cannot convert {type(source).__name__} "
            f"({len(source_per_field)} fields: "
            f"{tuple(source_per_field)}) to {type(target).__name__} "
            f"({len(target_per_field)} fields: "
            f"{tuple(target_per_field)}): field-count mismatch. "
            f"Pass check_support=False to override."
        )
    for (s_name, s_sup), (t_name, t_sup) in pairs:
        if _supports_compatible(s_sup, t_sup):
            continue
        raise ValueError(
            f"Cannot convert {type(source).__name__} field "
            f"{s_name!r} (support={s_sup}) to "
            f"{type(target).__name__} field {t_name!r} "
            f"(support={t_sup}). "
            f"Pass check_support=False to override."
        )


@dataclass(frozen=True)
class _SampledMomentBatch:
    """One shared Monte Carlo realization for a conversion's moments."""

    samples: Any

    def mean(self) -> Any:
        return jnp.mean(self.samples, axis=0)

    def variance(self) -> Any:
        return jnp.var(self.samples, axis=0)

    def covariance(self) -> Any:
        mean = self.mean()
        diff = self.samples - mean
        return jnp.einsum("ni,nj->ij", diff, diff) / self.samples.shape[0]


def _requires_sampled_moments(source: Any, target_name: str) -> bool:
    """Return whether a known ProbPipe moment implementation uses MC."""
    return (
        isinstance(source, BijectorTransformedDistribution)
        and not isinstance(source, SupportsMean)
        and target_name in _MOMENT_MATCH_TARGETS
    )


def _sampled_moment_batch(kw: dict[str, Any]) -> _SampledMomentBatch | None:
    """Return the conversion-owned batch, when moment matching sampled."""
    batch = kw.get(_SAMPLED_MOMENT_BATCH_KWARG)
    if batch is not None and not isinstance(batch, _SampledMomentBatch):
        raise TypeError("invalid private sampled-moment batch")
    return batch


def _source_mean(source: Any, kw: dict[str, Any]) -> Any:
    batch = _sampled_moment_batch(kw)
    return source._mean() if batch is None else batch.mean()


def _source_variance(source: Any, kw: dict[str, Any]) -> Any:
    batch = _sampled_moment_batch(kw)
    return source._variance() if batch is None else batch.variance()


def _source_covariance(source: Any, kw: dict[str, Any]) -> Any:
    batch = _sampled_moment_batch(kw)
    return source._cov().to_dense() if batch is None else batch.covariance()


def _conditional_conversion_plan(num_samples: Any) -> _ConversionExecutionPlan:
    """Build the conditional covariance-fallback plan."""
    count = _validate_conversion_sample_count(num_samples)
    return _ConversionExecutionPlan(
        execution_mode="conditional",
        sample_shape=(count,),
        provider_abi=_PROBPIPE_PROVIDER_ABI,
        automatic_key_certified=True,
    )


def _probpipe_sampled_plan(num_samples: Any) -> _ConversionExecutionPlan:
    """Build the standard ProbPipe sampled-conversion plan."""
    return _sampled_conversion_plan(
        num_samples,
        provider_abi=_PROBPIPE_PROVIDER_ABI,
    )


def _sampled_moment_plan(
    source: Any,
    target_name: str,
    kwargs: dict[str, Any],
) -> _ConversionExecutionPlan | None:
    """Plan one known Monte Carlo moment-matching conversion."""
    if not _requires_sampled_moments(source, target_name):
        return None
    if target_name in _TOTAL_COUNT_TARGETS and kwargs.get("total_count") is None:
        raise ValueError(
            f"total_count is required when converting to {target_name} "
            f"from a non-{target_name} source."
        )
    return _probpipe_sampled_plan(kwargs.get("num_samples", DEFAULT_NUM_SAMPLES))


def _sample_with_execution_plan(
    source: Any,
    key: Any | None,
    kwargs: dict[str, Any],
) -> Any:
    """Sample with the conversion plan prepared before execution."""
    kwargs.pop("num_samples", None)
    return _sample_probpipe_conversion_source(
        source,
        key,
        kwargs.pop(_EXECUTION_PLAN_KWARG),
    )


def _probpipe_nonrandom_plan(
    execution_mode: _ConversionExecutionMode = "analytic",
) -> _ConversionExecutionPlan:
    """Build a non-consuming ProbPipe conversion plan."""
    return _ConversionExecutionPlan(
        execution_mode=execution_mode,
        sample_shape=None,
        provider_abi=_PROBPIPE_PROVIDER_ABI,
        automatic_key_certified=True,
    )


# ---------------------------------------------------------------------------
# Moment-matching helpers using expectation()
# ---------------------------------------------------------------------------


def _mm_provenance(source):
    """Build provenance for a moment-matching conversion of *source*."""
    return Provenance.create("from_distribution", parents=[source], metadata={})


def _point_estimate(x):
    """The array a moment or a batch of draws holds when the law's event has one leaf.

    A record law returns its moments and its draws in raw form, a nested
    mapping of raw leaves, and a law over a one-field record returns a
    ``NumericRecord`` for its moments; each with one leaf is that leaf's array.
    Any other value is returned as it is.
    """
    from ..core._numeric_record import NumericRecord

    if isinstance(x, NumericRecord) and len(x.fields) == 1:
        return x[x.fields[0]]
    if isinstance(x, Mapping) and not isinstance(x, TrackedTerm):
        leaves = _mapping_leaves(x)
        if len(leaves) == 1:
            return leaves[0]
    return x


def _mapping_leaves(node: Mapping) -> list:
    """The leaves of a nested mapping, in its order."""
    leaves = []
    for value in node.values():
        if isinstance(value, Mapping) and not isinstance(value, TrackedTerm):
            leaves.extend(_mapping_leaves(value))
        else:
            leaves.append(value)
    return leaves


# ---------------------------------------------------------------------------
# Per-target conversion functions
#
# Each function has signature:
#   (source, key, **kwargs) -> Distribution
# ---------------------------------------------------------------------------


def _convert_to_normal(source, key, **kw):
    from ..families._continuous import Normal

    if isinstance(source, Normal):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = _source_mean(source, kw), _source_variance(source, kw)
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    r = Normal(loc=m, scale=jnp.sqrt(v), name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_beta(source, key, **kw):
    from ..families._continuous import Beta

    if isinstance(source, Beta):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = _source_mean(source, kw), _source_variance(source, kw)
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    common = m * (1.0 - m) / v - 1.0
    alpha = jnp.maximum(m * common, 0.01)
    beta = jnp.maximum((1.0 - m) * common, 0.01)
    r = Beta(alpha=alpha, beta=beta, name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_gamma(source, key, **kw):
    from ..families._continuous import Gamma

    if isinstance(source, Gamma):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = _source_mean(source, kw), _source_variance(source, kw)
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    r = Gamma(concentration=m**2 / v, rate=m / v, name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_inverse_gamma(source, key, **kw):
    from ..families._continuous import InverseGamma

    if isinstance(source, InverseGamma):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = _source_mean(source, kw), _source_variance(source, kw)
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    conc = m**2 / v + 2
    scale = m * (m**2 / v + 1)
    r = InverseGamma(concentration=conc, scale=scale, name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_exponential(source, key, **kw):
    from ..families._continuous import Exponential

    if isinstance(source, Exponential):
        return source
    kw.pop("num_samples", None)
    m_raw = _source_mean(source, kw)
    m = _point_estimate(m_raw)
    r = Exponential(rate=1.0 / m, name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_lognormal(source, key, **kw):
    from ..families._continuous import LogNormal

    if isinstance(source, LogNormal):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = _source_mean(source, kw), _source_variance(source, kw)
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    scale = jnp.sqrt(jnp.log(1.0 + v / (m**2)))
    loc = jnp.log(m) - scale**2 / 2.0
    r = LogNormal(loc=loc, scale=scale, name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_studentt(source, key, **kw):
    from ..families._continuous import StudentT

    if isinstance(source, StudentT):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = _source_mean(source, kw), _source_variance(source, kw)
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    # var = scale^2 * df/(df-2) for df>2, so scale = sqrt(var * (df-2)/df)
    df = 5.0
    r = StudentT(
        df=df, loc=m, scale=jnp.sqrt(v * (df - 2.0) / df), name=kw.get("name") or source.name
    )
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_uniform(source, key, **kw):
    from ..families._continuous import Uniform

    if isinstance(source, Uniform):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = _source_mean(source, kw), _source_variance(source, kw)
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    half = jnp.sqrt(3.0 * v)
    r = Uniform(low=m - half, high=m + half, name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_cauchy(source, key, **kw):
    from ..families._continuous import Cauchy

    if isinstance(source, Cauchy):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = _source_mean(source, kw), _source_variance(source, kw)
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    r = Cauchy(loc=m, scale=jnp.sqrt(v) / 2.0, name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_laplace(source, key, **kw):
    from ..families._continuous import Laplace

    if isinstance(source, Laplace):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = _source_mean(source, kw), _source_variance(source, kw)
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    r = Laplace(loc=m, scale=jnp.sqrt(v / 2.0), name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_halfnormal(source, key, **kw):
    from ..families._continuous import HalfNormal

    if isinstance(source, HalfNormal):
        return source
    kw.pop("num_samples", None)
    v_raw = _source_variance(source, kw)
    v = _point_estimate(v_raw)
    # var = scale^2 * (1 - 2/pi), so scale = sqrt(var / (1 - 2/pi))
    r = HalfNormal(scale=jnp.sqrt(v / (1.0 - 2.0 / jnp.pi)), name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_halfcauchy(source, key, **kw):
    from ..families._continuous import HalfCauchy

    if isinstance(source, HalfCauchy):
        return source
    samples = _point_estimate(_sample_with_execution_plan(source, key, kw))
    med = jnp.median(samples)
    r = HalfCauchy(loc=0.0, scale=jnp.maximum(med, 0.01), name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_pareto(source, key, **kw):
    from ..families._continuous import Pareto

    if isinstance(source, Pareto):
        return source
    samples = _point_estimate(_sample_with_execution_plan(source, key, kw))
    n = samples.shape[0]
    scale = jnp.maximum(jnp.min(samples), 1e-6)
    conc = jnp.maximum(n / jnp.sum(jnp.log(samples / scale)), 0.01)
    r = Pareto(concentration=conc, scale=scale, name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_truncatednormal(source, key, **kw):
    from ..families._continuous import TruncatedNormal

    if isinstance(source, TruncatedNormal):
        return source
    batch = _sampled_moment_batch(kw)
    m_raw, v_raw = _source_mean(source, kw), _source_variance(source, kw)
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    if batch is None:
        samples = _point_estimate(_sample_with_execution_plan(source, key, kw))
    else:
        kw.pop("num_samples", None)
        samples = batch.samples
    r = TruncatedNormal(
        loc=m,
        scale=jnp.sqrt(v),
        low=jnp.min(samples),
        high=jnp.max(samples),
        name=kw.get("name") or source.name,
    )
    r.with_provenance(_mm_provenance(source))
    return r


# -- discrete ---------------------------------------------------------------


def _convert_to_bernoulli(source, key, **kw):
    from ..families._discrete import Bernoulli

    if isinstance(source, Bernoulli):
        return source
    kw.pop("num_samples", None)
    r = Bernoulli(probs=_source_mean(source, kw), name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_binomial(source, key, **kw):
    from ..families._discrete import Binomial

    if isinstance(source, Binomial):
        return source
    total_count = kw.pop("total_count", None)
    if total_count is None:
        raise ValueError(
            "total_count is required when converting to Binomial from a non-Binomial source."
        )
    kw.pop("num_samples", None)
    probs = _source_mean(source, kw) / total_count
    r = Binomial(total_count=total_count, probs=probs, name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_poisson(source, key, **kw):
    from ..families._discrete import Poisson

    if isinstance(source, Poisson):
        return source
    kw.pop("num_samples", None)
    r = Poisson(rate=_source_mean(source, kw), name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_categorical(source, key, **kw):
    from ..families._discrete import Categorical

    if isinstance(source, Categorical):
        return source
    samples = _point_estimate(_sample_with_execution_plan(source, key, kw))
    n_cat = int(jnp.max(samples)) + 1
    counts = jnp.array([(samples == k).sum() for k in range(n_cat)])
    probs = counts / counts.sum()
    r = Categorical(probs=probs, name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_negativebinomial(source, key, **kw):
    from ..families._discrete import NegativeBinomial

    if isinstance(source, NegativeBinomial):
        return source
    total_count = kw.pop("total_count", None)
    if total_count is None:
        raise ValueError(
            "total_count is required when converting to NegativeBinomial from a non-NegativeBinomial source."
        )
    kw.pop("num_samples", None)
    m = _source_mean(source, kw)
    probs = total_count / (total_count + m)
    r = NegativeBinomial(total_count=total_count, probs=probs, name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


# -- multivariate -----------------------------------------------------------


def _convert_to_multivariatenormal(source, key, **kw):
    from ..families._multivariate import MultivariateNormal

    kw.pop("num_samples", None)
    name = kw.get("name") or source.name
    if isinstance(source, MultivariateNormal):
        return source
    if isinstance(source, EmpiricalDistribution):
        # The flat coordinates of the atoms give the moments of every numeric event.
        loc = source.weights @ _coordinates(source)
        r = MultivariateNormal(loc=loc, cov=source._cov().to_dense(), name=name)
        r.with_provenance(_mm_provenance(source))
        return r
    # General case: use _mean() and _cov() directly
    m_raw = _source_mean(source, kw)
    loc = _point_estimate(m_raw)
    try:
        cov_mat = _source_covariance(source, kw)
    except (NotImplementedError, AttributeError):
        # Fallback to sample-based covariance
        samples = _point_estimate(_sample_with_execution_plan(source, key, kw))
        diff = samples - loc
        cov_mat = jnp.einsum("ni,nj->ij", diff, diff) / samples.shape[0]
    cov_mat = 0.5 * (cov_mat + cov_mat.T)
    cov_mat = cov_mat + 1e-6 * jnp.eye(cov_mat.shape[0])
    r = MultivariateNormal(loc=loc, cov=cov_mat, name=name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_dirichlet(source, key, **kw):
    from ..families._multivariate import Dirichlet

    if isinstance(source, Dirichlet):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = _source_mean(source, kw), _source_variance(source, kw)
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    conc0 = m[0] * (1.0 - m[0]) / (v[0] + 1e-8) - 1.0
    conc0 = jnp.maximum(conc0, 0.01)
    conc = jnp.maximum(m * conc0, 0.01)
    r = Dirichlet(concentration=conc, name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_multinomial(source, key, **kw):
    from ..families._multivariate import Multinomial

    if isinstance(source, Multinomial):
        return source
    total_count = kw.pop("total_count", None)
    if total_count is None:
        raise ValueError(
            "total_count is required when converting to Multinomial from a non-Multinomial source."
        )
    kw.pop("num_samples", None)
    m_raw = _source_mean(source, kw)
    m = _point_estimate(m_raw)
    probs = m / total_count
    probs = probs / probs.sum()
    r = Multinomial(total_count=total_count, probs=probs, name=kw.get("name") or source.name)
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_wishart(source, key, **kw):
    from ..families._multivariate import Wishart

    if isinstance(source, Wishart):
        return source
    samples = _point_estimate(_sample_with_execution_plan(source, key, kw))
    mean_mat = jnp.mean(samples, axis=0)
    d = mean_mat.shape[-1]
    df = d + 2.0
    scale_mat = mean_mat / df
    scale_mat = 0.5 * (scale_mat + scale_mat.T)
    scale_mat = scale_mat + 1e-6 * jnp.eye(d)
    r = Wishart(
        df=df, scale_tril=jnp.linalg.cholesky(scale_mat), name=kw.get("name") or source.name
    )
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_vonmisesfisher(source, key, **kw):
    from ..families._multivariate import VonMisesFisher

    if isinstance(source, VonMisesFisher):
        return source
    kw.pop("num_samples", None)
    m_raw = _source_mean(source, kw)
    mean_vec = _point_estimate(m_raw)
    R = jnp.linalg.norm(mean_vec)
    mean_dir = mean_vec / jnp.maximum(R, 1e-8)
    d = mean_vec.shape[-1]
    R2 = R**2
    conc = jnp.maximum(R * (d - R2) / jnp.maximum(1.0 - R2, 1e-8), 0.0)
    r = VonMisesFisher(
        mean_direction=mean_dir, concentration=conc, name=kw.get("name") or source.name
    )
    r.with_provenance(_mm_provenance(source))
    return r


def _convert_to_empirical(source, key, **kw):
    """Convert any distribution to an EmpiricalDistribution of its draws.

    The draws are the atoms, on the level ``sample`` that drawing them mints,
    and keep the source's event declaration, so the empirical law's components
    and packaging are the source's.
    """
    if isinstance(source, EmpiricalDistribution):
        return source
    name = kw.get("name") or source.name
    samples = _sample_with_execution_plan(source, key, kw)
    atoms = _batch_form(name, samples, SAMPLE_LEVEL, source.event_spec.spec)
    r = EmpiricalDistribution(name, atoms, event_spec=source.event_spec)
    r.with_provenance(_mm_provenance(source))
    return r


def _smoothed_atoms(name, values, spec):
    """*values*, raw values of *spec* along one axis, as a KDE takes its atoms.

    An array event's values are an array; a record event's are a batch of
    records on one level.
    """
    if isinstance(spec, NumericArraySpec):
        return jnp.asarray(values)
    return _batch_form(name, values, SAMPLE_LEVEL, spec)


def _convert_to_kde(source, key, **kw):
    """Convert any distribution to a KDEDistribution.

    An empirical source's atoms and weights are the KDE's, so an inference
    result's posterior keeps its target's event declaration. Other sources are
    sampled, and the KDE smooths the draws under the source's declaration.

    Raises
    ------
    TypeError
        If *source*'s event is not numeric, since a KDE smooths numeric atoms.
    """
    from ..families._resampling import KDEDistribution

    if isinstance(source, KDEDistribution):
        return source

    bandwidth = kw.pop("bandwidth", None)
    name = kw.get("name") or source.name
    spec = source.event_spec.spec
    if not isinstance(spec, NumericSpec):
        raise TypeError(f"a KDE smooths numeric atoms, and {source.name!r} declares {spec!r}")

    if isinstance(source, EmpiricalDistribution):
        r = KDEDistribution(
            name,
            _smoothed_atoms(name, source._rows, source.event_spec.spec),
            bandwidth,
            source.weights,
            event_spec=source.event_spec,
        )
        r.with_provenance(_mm_provenance(source))
        return r

    samples = _sample_with_execution_plan(source, key, kw)
    atoms = _smoothed_atoms(name, samples, source.event_spec.spec)
    r = KDEDistribution(name, atoms, bandwidth, event_spec=source.event_spec)
    r.with_provenance(_mm_provenance(source))
    return r


def _check_atoms_in_support(target: Any, source: Any) -> None:
    """Refuse an empirical source whose atoms lie outside the target's support.

    An empirical law's array atoms declare no support, so the declared check
    has nothing to compare at such a leaf; its atoms are what the law is
    supported on, and each must lie in the support the target's leaves share.
    A target whose leaves differ in support, or a leaf whose support the source
    declares, is left to the declared check.

    Raises
    ------
    ValueError
        If an atom of a leaf without a declared support lies outside the
        target's support.
    """
    if not isinstance(source, EmpiricalDistribution):
        return
    if not isinstance(source.event_spec.spec, NumericSpec):
        return
    support = target.support
    if support is None:
        return
    declared = source.supports
    rows = source._rows
    columns = rows if isinstance(rows, dict) else dict.fromkeys(declared, rows)
    for path, column in columns.items():
        if declared.get(path) is not None:
            continue
        if not bool(jnp.all(support.check(jnp.asarray(column)))):
            raise ValueError(
                f"Cannot convert {type(source).__name__} {source.name!r} to "
                f"{type(target).__name__} (support={support}): atoms of {path!r} lie "
                f"outside that support. Pass check_support=False to override."
            )


# ---------------------------------------------------------------------------
# Dispatch table: target class name -> conversion function
# ---------------------------------------------------------------------------


def _build_dispatch_table() -> dict[str, callable]:
    """Build the dispatch table lazily to avoid circular imports."""
    return {
        "Normal": _convert_to_normal,
        "Beta": _convert_to_beta,
        "Gamma": _convert_to_gamma,
        "InverseGamma": _convert_to_inverse_gamma,
        "Exponential": _convert_to_exponential,
        "LogNormal": _convert_to_lognormal,
        "StudentT": _convert_to_studentt,
        "Uniform": _convert_to_uniform,
        "Cauchy": _convert_to_cauchy,
        "Laplace": _convert_to_laplace,
        "HalfNormal": _convert_to_halfnormal,
        "HalfCauchy": _convert_to_halfcauchy,
        "Pareto": _convert_to_pareto,
        "TruncatedNormal": _convert_to_truncatednormal,
        "Bernoulli": _convert_to_bernoulli,
        "Binomial": _convert_to_binomial,
        "Poisson": _convert_to_poisson,
        "Categorical": _convert_to_categorical,
        "NegativeBinomial": _convert_to_negativebinomial,
        "MultivariateNormal": _convert_to_multivariatenormal,
        "Dirichlet": _convert_to_dirichlet,
        "Multinomial": _convert_to_multinomial,
        "Wishart": _convert_to_wishart,
        "VonMisesFisher": _convert_to_vonmisesfisher,
        "EmpiricalDistribution": _convert_to_empirical,
        "KDEDistribution": _convert_to_kde,
    }


# ---------------------------------------------------------------------------
# The converter
# ---------------------------------------------------------------------------


class ProbPipeConverter(Converter):
    """Converter for ProbPipe-to-ProbPipe distribution conversions.

    Same-class conversions return the source unchanged.  Cross-family
    conversions moment-match using the source's ``mean()`` and
    ``variance()`` methods and enforce support constraints.
    """

    def __init__(self) -> None:
        self._dispatch: dict[str, callable] | None = None

    @property
    def _table(self) -> dict[str, callable]:
        if self._dispatch is None:
            self._dispatch = _build_dispatch_table()
        return self._dispatch

    def source_types(self) -> tuple[type, ...]:
        return (Distribution,)

    def target_types(self) -> tuple[type, ...]:
        return (Distribution,)

    def _is_source(self, source: Any) -> bool:
        return isinstance(source, Distribution)

    def _is_target(self, target_type: type) -> bool:
        return isinstance(target_type, type) and issubclass(target_type, Distribution)

    def check(self, source: Any, target_type: type) -> ConversionInfo:
        if not self._is_source(source):
            return ConversionInfo(feasible=False)
        if not self._is_target(target_type):
            return ConversionInfo(feasible=False)

        target_name = target_type.__name__
        if target_name not in self._table:
            return ConversionInfo(
                feasible=False, description=f"No converter for target {target_name}"
            )

        # Same class = exact copy
        if isinstance(source, target_type):
            return ConversionInfo(
                feasible=True,
                method=ConversionMethod.EXACT,
                estimated_time=0.0,
                source_type=type(source),
                target_type=target_type,
                description=f"Copy {type(source).__name__} parameters",
            )

        # Support compatibility is verified post-construction in
        # ``convert()`` (the target instance exposes per-field
        # ``supports``; there is no class-level pre-construction hint).
        return ConversionInfo(
            feasible=True,
            method=ConversionMethod.MOMENT_MATCH,
            estimated_time=0.1,
            source_type=type(source),
            target_type=target_type,
            description=f"Moment-match {type(source).__name__} -> {target_name}",
        )

    def _workflow_plan_conversion(
        self,
        source: Any,
        target_type: type,
        kwargs: dict[str, Any],
    ) -> _ConversionExecutionPlan:
        """Classify the selected path before converter execution."""
        if isinstance(source, target_type):
            return _probpipe_nonrandom_plan("exact")

        target_name = target_type.__name__
        sampled_moment_plan = _sampled_moment_plan(source, target_name, kwargs)
        if sampled_moment_plan is not None:
            return sampled_moment_plan

        sampled_targets = {
            "HalfCauchy",
            "Pareto",
            "TruncatedNormal",
            "Categorical",
            "Wishart",
            "EmpiricalDistribution",
        }
        if target_name in sampled_targets:
            return _probpipe_sampled_plan(kwargs.get("num_samples", DEFAULT_NUM_SAMPLES))

        if target_name == "KDEDistribution":
            if isinstance(source, EmpiricalDistribution):
                return _probpipe_nonrandom_plan()
            return _probpipe_sampled_plan(kwargs.get("num_samples", DEFAULT_NUM_SAMPLES))

        if target_name == "MultivariateNormal":
            if isinstance(source, EmpiricalDistribution):
                return _probpipe_nonrandom_plan()
            return _conditional_conversion_plan(kwargs.get("num_samples", DEFAULT_NUM_SAMPLES))

        return _probpipe_nonrandom_plan()

    def convert(
        self, source: Any, target_type: type, *, key: Any | None = None, **kwargs: Any
    ) -> Distribution:
        plan = self._workflow_plan_conversion(source, target_type, dict(kwargs))
        return self._workflow_execute_conversion(
            source,
            target_type,
            plan,
            key=key,
            **kwargs,
        )

    def _workflow_execute_conversion(
        self,
        source: Any,
        target_type: type,
        plan: _ConversionExecutionPlan,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Distribution:
        """Execute a conversion from its already validated private plan."""
        target_name = target_type.__name__
        fn = self._table.get(target_name)
        if fn is None:
            raise TypeError(f"ProbPipeConverter: no conversion for target {target_name}")

        check_support = kwargs.pop("check_support", True)
        if _requires_sampled_moments(source, target_name):
            samples = _sample_probpipe_conversion_source(source, key, plan)
            kwargs[_SAMPLED_MOMENT_BATCH_KWARG] = _SampledMomentBatch(
                jnp.asarray(_point_estimate(samples))
            )
        else:
            kwargs[_EXECUTION_PLAN_KWARG] = plan

        # Some converters (e.g., ``_convert_to_normal``) fabricate
        # TFP-backed scalars from a source's ``_mean`` / ``_variance``.
        # When the source's moments are ``(d,)``-shaped (a 1-d
        # empirical with one observation, or a Normal whose source
        # was tracked through a WF sweep), the implied ``batch_shape``
        # is non-empty — which ``TFPDistribution.__init__``'s rejection
        # would otherwise refuse. The converter is library-internal
        # infra, not user code, so we opt into the bypass for the
        # dispatch. A proper fix (route to ``DistributionArray`` for
        # batched moments) is tracked as a follow-up.
        with _allow_batched_tfp_init():
            result = fn(source, key, **kwargs)

        # Post-construction support check. Per-field ``supports`` is
        # instance state, so the check has to run after the target is
        # built. Only a numeric target declares supports; sources that
        # don't expose per-field ``supports`` raise ``AttributeError``,
        # which counts as "unknown".
        if check_support and isinstance(result, NumericDistribution):
            with contextlib.suppress(AttributeError):
                _check_support_compatible(result, source)
            _check_atoms_in_support(result, source)

        return result

    @property
    def priority(self) -> int:
        return 100
