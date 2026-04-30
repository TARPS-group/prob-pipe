"""ProbPipe-to-ProbPipe converter.

Handles conversions between ProbPipe distribution types:

- **Same-class**: returns the source unchanged (no copy, no provenance).
- **Cross-family**: moment-matches using the source's ``_mean()`` and
  ``_variance()`` methods, which delegate to TFP for analytical moments
  or fall back to Monte Carlo via the protocol defaults.
  When MC is used, the resulting ``BootstrapDistribution`` is stored
  in provenance metadata so users can inspect conversion error.

Registered at priority 100 so it is always tried first for ProbPipe types.
"""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp

from ..custom_types import PRNGKey
from .._utils import _auto_key
from ..core.constraints import _supports_compatible
from ..core.distribution import (
    RecordEmpiricalDistribution,
    Distribution,
)
from ..core.provenance import Provenance
from ._registry import ConversionInfo, ConversionMethod, Converter

# Default sample count for moment-matching conversions
DEFAULT_NUM_SAMPLES = 1024


# ---------------------------------------------------------------------------
# Moment-matching helpers using expectation()
# ---------------------------------------------------------------------------

def _mm_provenance(source, mean_result=None, var_result=None):
    """Build provenance for a moment-matching conversion.

    If *mean_result* or *var_result* are ``BootstrapDistribution``
    instances (from MC fallback), they are stored in the metadata so
    users can inspect conversion error.
    """
    from ..core.distribution import BootstrapDistribution
    metadata = {}
    if isinstance(mean_result, BootstrapDistribution):
        metadata["mean_bootstrap"] = mean_result
    if isinstance(var_result, BootstrapDistribution):
        metadata["var_bootstrap"] = var_result
    return Provenance("from_distribution", parents=(source,), metadata=metadata)


def _point_estimate(x):
    """Extract a plain array from a value that may be a BootstrapDistribution
    or a single-field NumericRecord (the auto-wrap form returned by the
    merged ``RecordEmpiricalDistribution._mean``)."""
    from ..core.distribution import BootstrapDistribution
    from ..core._numeric_record import NumericRecord
    if isinstance(x, BootstrapDistribution):
        x = x._mean()
    if isinstance(x, NumericRecord) and len(x.fields) == 1:
        return x[x.fields[0]]
    return x


# ---------------------------------------------------------------------------
# Per-target conversion functions
#
# Each function has signature:
#   (source, key, **kwargs) -> Distribution
# ---------------------------------------------------------------------------

def _convert_to_normal(source, key, **kw):
    from ..distributions.continuous import Normal
    if isinstance(source, Normal):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = source._mean(), source._variance()
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    r = Normal(loc=m, scale=jnp.sqrt(v), name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source, m_raw, v_raw))
    return r


def _convert_to_beta(source, key, **kw):
    from ..distributions.continuous import Beta
    if isinstance(source, Beta):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = source._mean(), source._variance()
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    common = m * (1.0 - m) / v - 1.0
    alpha = jnp.maximum(m * common, 0.01)
    beta = jnp.maximum((1.0 - m) * common, 0.01)
    r = Beta(alpha=alpha, beta=beta, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source, m_raw, v_raw))
    return r


def _convert_to_gamma(source, key, **kw):
    from ..distributions.continuous import Gamma
    if isinstance(source, Gamma):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = source._mean(), source._variance()
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    r = Gamma(concentration=m ** 2 / v, rate=m / v, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source, m_raw, v_raw))
    return r


def _convert_to_inverse_gamma(source, key, **kw):
    from ..distributions.continuous import InverseGamma
    if isinstance(source, InverseGamma):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = source._mean(), source._variance()
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    conc = m ** 2 / v + 2
    scale = m * (m ** 2 / v + 1)
    r = InverseGamma(concentration=conc, scale=scale, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source, m_raw, v_raw))
    return r


def _convert_to_exponential(source, key, **kw):
    from ..distributions.continuous import Exponential
    if isinstance(source, Exponential):
        return source
    kw.pop("num_samples", None)
    m_raw = source._mean()
    m = _point_estimate(m_raw)
    r = Exponential(rate=1.0 / m, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source, m_raw))
    return r


def _convert_to_lognormal(source, key, **kw):
    from ..distributions.continuous import LogNormal
    if isinstance(source, LogNormal):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = source._mean(), source._variance()
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    scale = jnp.sqrt(jnp.log(1.0 + v / (m ** 2)))
    loc = jnp.log(m) - scale ** 2 / 2.0
    r = LogNormal(loc=loc, scale=scale, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source, m_raw, v_raw))
    return r


def _convert_to_studentt(source, key, **kw):
    from ..distributions.continuous import StudentT
    if isinstance(source, StudentT):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = source._mean(), source._variance()
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    # var = scale^2 * df/(df-2) for df>2, so scale = sqrt(var * (df-2)/df)
    df = 5.0
    r = StudentT(df=df, loc=m, scale=jnp.sqrt(v * (df - 2.0) / df), name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source, m_raw, v_raw))
    return r


def _convert_to_uniform(source, key, **kw):
    from ..distributions.continuous import Uniform
    if isinstance(source, Uniform):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = source._mean(), source._variance()
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    half = jnp.sqrt(3.0 * v)
    r = Uniform(low=m - half, high=m + half, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source, m_raw, v_raw))
    return r


def _convert_to_cauchy(source, key, **kw):
    from ..distributions.continuous import Cauchy
    if isinstance(source, Cauchy):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = source._mean(), source._variance()
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    r = Cauchy(loc=m, scale=jnp.sqrt(v) / 2.0, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source, m_raw, v_raw))
    return r


def _convert_to_laplace(source, key, **kw):
    from ..distributions.continuous import Laplace
    if isinstance(source, Laplace):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = source._mean(), source._variance()
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    r = Laplace(loc=m, scale=jnp.sqrt(v / 2.0), name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source, m_raw, v_raw))
    return r


def _convert_to_halfnormal(source, key, **kw):
    from ..distributions.continuous import HalfNormal
    if isinstance(source, HalfNormal):
        return source
    kw.pop("num_samples", None)
    v_raw = source._variance()
    v = _point_estimate(v_raw)
    # var = scale^2 * (1 - 2/pi), so scale = sqrt(var / (1 - 2/pi))
    r = HalfNormal(scale=jnp.sqrt(v / (1.0 - 2.0 / jnp.pi)), name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source, var_result=v_raw))
    return r


def _convert_to_halfcauchy(source, key, **kw):
    from ..distributions.continuous import HalfCauchy
    if isinstance(source, HalfCauchy):
        return source
    num_samples = kw.pop("num_samples", DEFAULT_NUM_SAMPLES)
    if key is None:
        key = _auto_key()
    samples = source._sample(key, (num_samples,))
    med = jnp.median(samples)
    r = HalfCauchy(loc=0.0, scale=jnp.maximum(med, 0.01), name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source))
    return r


def _convert_to_pareto(source, key, **kw):
    from ..distributions.continuous import Pareto
    if isinstance(source, Pareto):
        return source
    num_samples = kw.pop("num_samples", DEFAULT_NUM_SAMPLES)
    if key is None:
        key = _auto_key()
    samples = source._sample(key, (num_samples,))
    n = samples.shape[0]
    scale = jnp.maximum(jnp.min(samples), 1e-6)
    conc = jnp.maximum(n / jnp.sum(jnp.log(samples / scale)), 0.01)
    r = Pareto(concentration=conc, scale=scale, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source))
    return r


def _convert_to_truncatednormal(source, key, **kw):
    from ..distributions.continuous import TruncatedNormal
    if isinstance(source, TruncatedNormal):
        return source
    num_samples = kw.pop("num_samples", DEFAULT_NUM_SAMPLES)
    if key is None:
        key = _auto_key()
    m_raw, v_raw = source._mean(), source._variance()
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    samples = source._sample(key, (num_samples,))
    r = TruncatedNormal(
        loc=m, scale=jnp.sqrt(v),
        low=jnp.min(samples), high=jnp.max(samples),
        name=kw.get("name") or source.name,
    )
    r.with_source(_mm_provenance(source, m_raw, v_raw))
    return r


# -- discrete ---------------------------------------------------------------

def _convert_to_bernoulli(source, key, **kw):
    from ..distributions.discrete import Bernoulli
    if isinstance(source, Bernoulli):
        return source
    kw.pop("num_samples", None)
    r = Bernoulli(probs=source._mean(), name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source))
    return r


def _convert_to_binomial(source, key, **kw):
    from ..distributions.discrete import Binomial
    if isinstance(source, Binomial):
        return source
    total_count = kw.pop("total_count", None)
    if total_count is None:
        raise ValueError("total_count is required when converting to Binomial from a non-Binomial source.")
    kw.pop("num_samples", None)
    probs = source._mean() / total_count
    r = Binomial(total_count=total_count, probs=probs, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source))
    return r


def _convert_to_poisson(source, key, **kw):
    from ..distributions.discrete import Poisson
    if isinstance(source, Poisson):
        return source
    kw.pop("num_samples", None)
    r = Poisson(rate=source._mean(), name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source))
    return r


def _convert_to_categorical(source, key, **kw):
    from ..distributions.discrete import Categorical
    if isinstance(source, Categorical):
        return source
    num_samples = kw.pop("num_samples", DEFAULT_NUM_SAMPLES)
    samples = source._sample(key, (num_samples,))
    n_cat = int(jnp.max(samples)) + 1
    counts = jnp.array([(samples == k).sum() for k in range(n_cat)])
    probs = counts / counts.sum()
    r = Categorical(probs=probs, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source))
    return r


def _convert_to_negativebinomial(source, key, **kw):
    from ..distributions.discrete import NegativeBinomial
    if isinstance(source, NegativeBinomial):
        return source
    total_count = kw.pop("total_count", None)
    if total_count is None:
        raise ValueError("total_count is required when converting to NegativeBinomial from a non-NegativeBinomial source.")
    kw.pop("num_samples", None)
    m = source._mean()
    probs = total_count / (total_count + m)
    r = NegativeBinomial(total_count=total_count, probs=probs, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source))
    return r


# -- multivariate -----------------------------------------------------------

def _convert_to_multivariatenormal(source, key, **kw):
    from ..distributions.multivariate import MultivariateNormal
    num_samples = kw.pop("num_samples", DEFAULT_NUM_SAMPLES)
    name = kw.get("name") or source.name
    if isinstance(source, MultivariateNormal):
        return source
    if isinstance(source, RecordEmpiricalDistribution):
        loc = source._mean()
        cov_mat = source._cov()
        r = MultivariateNormal(loc=loc, cov=cov_mat, name=name)
        r.with_source(_mm_provenance(source))
        return r
    # General case: use _mean() and _cov() directly
    m_raw = source._mean()
    loc = _point_estimate(m_raw)
    try:
        cov_mat = source._cov()
    except (NotImplementedError, AttributeError):
        # Fallback to sample-based covariance
        if key is None:
            key = _auto_key()
        samples = source._sample(key, (num_samples,))
        diff = samples - loc
        cov_mat = jnp.einsum("ni,nj->ij", diff, diff) / num_samples
    cov_mat = 0.5 * (cov_mat + cov_mat.T)
    cov_mat = cov_mat + 1e-6 * jnp.eye(cov_mat.shape[0])
    r = MultivariateNormal(loc=loc, cov=cov_mat, name=name)
    r.with_source(_mm_provenance(source, m_raw))
    return r


def _convert_to_dirichlet(source, key, **kw):
    from ..distributions.multivariate import Dirichlet
    if isinstance(source, Dirichlet):
        return source
    kw.pop("num_samples", None)
    m_raw, v_raw = source._mean(), source._variance()
    m, v = _point_estimate(m_raw), _point_estimate(v_raw)
    conc0 = m[0] * (1.0 - m[0]) / (v[0] + 1e-8) - 1.0
    conc0 = jnp.maximum(conc0, 0.01)
    conc = jnp.maximum(m * conc0, 0.01)
    r = Dirichlet(concentration=conc, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source, m_raw, v_raw))
    return r


def _convert_to_multinomial(source, key, **kw):
    from ..distributions.multivariate import Multinomial
    if isinstance(source, Multinomial):
        return source
    total_count = kw.pop("total_count", None)
    if total_count is None:
        raise ValueError("total_count is required when converting to Multinomial from a non-Multinomial source.")
    kw.pop("num_samples", None)
    m_raw = source._mean()
    m = _point_estimate(m_raw)
    probs = m / total_count
    probs = probs / probs.sum()
    r = Multinomial(total_count=total_count, probs=probs, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source, m_raw))
    return r


def _convert_to_wishart(source, key, **kw):
    from ..distributions.multivariate import Wishart
    if isinstance(source, Wishart):
        return source
    num_samples = kw.pop("num_samples", DEFAULT_NUM_SAMPLES)
    samples = source._sample(key, (num_samples,))
    mean_mat = jnp.mean(samples, axis=0)
    d = mean_mat.shape[-1]
    df = d + 2.0
    scale_mat = mean_mat / df
    scale_mat = 0.5 * (scale_mat + scale_mat.T)
    scale_mat = scale_mat + 1e-6 * jnp.eye(d)
    r = Wishart(df=df, scale_tril=jnp.linalg.cholesky(scale_mat), name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source))
    return r


def _convert_to_vonmisesfisher(source, key, **kw):
    from ..distributions.multivariate import VonMisesFisher
    if isinstance(source, VonMisesFisher):
        return source
    kw.pop("num_samples", None)
    m_raw = source._mean()
    mean_vec = _point_estimate(m_raw)
    R = jnp.linalg.norm(mean_vec)
    mean_dir = mean_vec / jnp.maximum(R, 1e-8)
    d = mean_vec.shape[-1]
    R2 = R ** 2
    conc = jnp.maximum(R * (d - R2) / jnp.maximum(1.0 - R2, 1e-8), 0.0)
    r = VonMisesFisher(mean_direction=mean_dir, concentration=conc, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source, m_raw))
    return r


def _convert_to_empirical(source, key, **kw):
    """Convert any distribution to RecordEmpiricalDistribution by sampling."""
    if isinstance(source, RecordEmpiricalDistribution):
        return source
    num_samples = kw.pop("num_samples", DEFAULT_NUM_SAMPLES)
    if key is None:
        key = _auto_key()
    samples = source._sample(key, (num_samples,))
    r = RecordEmpiricalDistribution(samples, name=kw.get("name") or source.name)
    r.with_source(_mm_provenance(source))
    return r


def _convert_to_kde(source, key, **kw):
    """Convert any distribution to a KDEDistribution.

    If the source is an ``RecordEmpiricalDistribution`` (or subclass),
    the stored samples and weights are reused directly.  Otherwise,
    samples are drawn from the source.
    """
    from ..distributions.kde import KDEDistribution

    if isinstance(source, KDEDistribution):
        return source

    bandwidth = kw.pop("bandwidth", None)
    name = kw.get("name") or source.name

    if isinstance(source, RecordEmpiricalDistribution):
        # Pull the underlying stacked array from the (single-field by
        # construction) Record. Multi-field empirical sources fall
        # through to the sample-based path below.
        if len(source.samples.fields) == 1:
            field = source.samples.fields[0]
            r = KDEDistribution(
                source.samples[field],
                weights=source._w, bandwidth=bandwidth, name=name,
            )
            r.with_source(_mm_provenance(source))
            return r

    num_samples = kw.pop("num_samples", DEFAULT_NUM_SAMPLES)
    if key is None:
        key = _auto_key()
    samples = source._sample(key, (num_samples,))
    r = KDEDistribution(samples, bandwidth=bandwidth, name=name)
    r.with_source(_mm_provenance(source))
    return r


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
        "RecordEmpiricalDistribution": _convert_to_empirical,
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
            return ConversionInfo(feasible=False, description=f"No converter for target {target_name}")

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

        # Check support compatibility — report in description but don't block
        support_note = ""
        try:
            target_support = target_type._default_support()
            source_support = source.support
            if not _supports_compatible(source_support, target_support):
                support_note = f" (support mismatch: {source_support} -> {target_support}; use check_support=False to override)"
        except (NotImplementedError, AttributeError):
            pass

        return ConversionInfo(
            feasible=True,
            method=ConversionMethod.MOMENT_MATCH,
            estimated_time=0.1,
            source_type=type(source),
            target_type=target_type,
            description=f"Moment-match {type(source).__name__} -> {target_name}{support_note}",
        )

    def convert(self, source: Any, target_type: type, *, key: Any | None = None, **kwargs: Any) -> Distribution:
        if key is None:
            key = _auto_key()
        target_name = target_type.__name__
        fn = self._table.get(target_name)
        if fn is None:
            raise TypeError(f"ProbPipeConverter: no conversion for target {target_name}")

        check_support = kwargs.pop("check_support", True)
        if check_support:
            try:
                target_support = target_type._default_support()
                source_support = source.support
                if not _supports_compatible(source_support, target_support):
                    raise ValueError(
                        f"Cannot convert {type(source).__name__} (support={source_support}) "
                        f"to {target_name} (support={target_support}). "
                        f"Pass check_support=False to override."
                    )
            except (NotImplementedError, AttributeError):
                pass

        result = fn(source, key, **kwargs)

        # Mark approximate if source is approximate or conversion used sampling
        if source.is_approximate or not isinstance(source, target_type):
            result._approximate = True

        return result

    @property
    def priority(self) -> int:
        return 100
