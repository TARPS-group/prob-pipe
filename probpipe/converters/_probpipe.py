"""ProbPipe-to-ProbPipe converter.

Absorbs the moment-matching logic formerly in each distribution's
``_from_distribution()`` classmethod.  Registered at priority 100
so it is always tried first for ProbPipe types.
"""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp

from ..custom_types import PRNGKey
from ..distributions.distribution import (
    Distribution,
    EmpiricalDistribution,
    Provenance,
    _auto_key,
    _supports_compatible,
)
from ._protocol import Converter
from ._registry import ConversionInfo, ConversionMethod


# ---------------------------------------------------------------------------
# Moment-matching helpers
# ---------------------------------------------------------------------------

def _sample_moments(other: Distribution, key: PRNGKey, num_samples: int):
    """Draw samples and return (samples, mean, variance)."""
    samples = other.sample(key, sample_shape=(num_samples,))
    return samples, jnp.mean(samples, axis=0), jnp.var(samples, axis=0)


# ---------------------------------------------------------------------------
# Per-target conversion functions
#
# Each function has signature:
#   (source, key, **kwargs) -> Distribution
# ---------------------------------------------------------------------------

def _convert_to_normal(source, key, **kw):
    from ..distributions.continuous import Normal
    if isinstance(source, Normal):
        r = Normal(loc=source._loc, scale=source._scale, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    _, m, v = _sample_moments(source, key, kw.pop("num_samples", 1024))
    r = Normal(loc=m, scale=jnp.sqrt(v), name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_beta(source, key, **kw):
    from ..distributions.continuous import Beta
    if isinstance(source, Beta):
        r = Beta(alpha=source._alpha, beta=source._beta, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    _, m, v = _sample_moments(source, key, kw.pop("num_samples", 1024))
    common = m * (1.0 - m) / v - 1.0
    alpha = jnp.maximum(m * common, 0.01)
    beta = jnp.maximum((1.0 - m) * common, 0.01)
    r = Beta(alpha=alpha, beta=beta, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_gamma(source, key, **kw):
    from ..distributions.continuous import Gamma
    if isinstance(source, Gamma):
        r = Gamma(concentration=source._concentration, rate=source._rate, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    _, m, v = _sample_moments(source, key, kw.pop("num_samples", 1024))
    r = Gamma(concentration=m ** 2 / v, rate=m / v, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_inverse_gamma(source, key, **kw):
    from ..distributions.continuous import InverseGamma
    if isinstance(source, InverseGamma):
        r = InverseGamma(concentration=source._concentration, scale=source._scale, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    _, m, v = _sample_moments(source, key, kw.pop("num_samples", 1024))
    conc = m ** 2 / v + 2
    scale = m * (m ** 2 / v + 1)
    r = InverseGamma(concentration=conc, scale=scale, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_exponential(source, key, **kw):
    from ..distributions.continuous import Exponential
    if isinstance(source, Exponential):
        r = Exponential(rate=source._rate, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    _, m, _ = _sample_moments(source, key, kw.pop("num_samples", 1024))
    r = Exponential(rate=1.0 / m, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_lognormal(source, key, **kw):
    from ..distributions.continuous import LogNormal
    if isinstance(source, LogNormal):
        r = LogNormal(loc=source._loc, scale=source._scale, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    _, m, v = _sample_moments(source, key, kw.pop("num_samples", 1024))
    scale = jnp.sqrt(jnp.log(1.0 + v / (m ** 2)))
    loc = jnp.log(m) - scale ** 2 / 2.0
    r = LogNormal(loc=loc, scale=scale, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_studentt(source, key, **kw):
    from ..distributions.continuous import StudentT
    if isinstance(source, StudentT):
        r = StudentT(df=source._df, loc=source._loc, scale=source._scale, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    _, m, v = _sample_moments(source, key, kw.pop("num_samples", 1024))
    r = StudentT(df=5.0, loc=m, scale=jnp.sqrt(v), name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_uniform(source, key, **kw):
    from ..distributions.continuous import Uniform
    if isinstance(source, Uniform):
        r = Uniform(low=source._low, high=source._high, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    _, m, v = _sample_moments(source, key, kw.pop("num_samples", 1024))
    half = jnp.sqrt(3.0 * v)
    r = Uniform(low=m - half, high=m + half, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_cauchy(source, key, **kw):
    from ..distributions.continuous import Cauchy
    if isinstance(source, Cauchy):
        r = Cauchy(loc=source._loc, scale=source._scale, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    _, m, v = _sample_moments(source, key, kw.pop("num_samples", 1024))
    r = Cauchy(loc=m, scale=jnp.sqrt(v) / 2.0, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_laplace(source, key, **kw):
    from ..distributions.continuous import Laplace
    if isinstance(source, Laplace):
        r = Laplace(loc=source._loc, scale=source._scale, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    _, m, v = _sample_moments(source, key, kw.pop("num_samples", 1024))
    r = Laplace(loc=m, scale=jnp.sqrt(v / 2.0), name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_halfnormal(source, key, **kw):
    from ..distributions.continuous import HalfNormal
    if isinstance(source, HalfNormal):
        r = HalfNormal(scale=source._scale, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    _, _, v = _sample_moments(source, key, kw.pop("num_samples", 1024))
    r = HalfNormal(scale=jnp.sqrt(2.0 * v / (1.0 - 2.0 / jnp.pi)), name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_halfcauchy(source, key, **kw):
    from ..distributions.continuous import HalfCauchy
    if isinstance(source, HalfCauchy):
        r = HalfCauchy(loc=source._loc, scale=source._scale, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    samples, _, _ = _sample_moments(source, key, kw.pop("num_samples", 1024))
    med = jnp.median(samples)
    r = HalfCauchy(loc=0.0, scale=jnp.maximum(med, 0.01), name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_pareto(source, key, **kw):
    from ..distributions.continuous import Pareto
    if isinstance(source, Pareto):
        r = Pareto(concentration=source._concentration, scale=source._scale, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    samples, _, _ = _sample_moments(source, key, kw.pop("num_samples", 1024))
    n = samples.shape[0]
    scale = jnp.maximum(jnp.min(samples), 1e-6)
    conc = jnp.maximum(n / jnp.sum(jnp.log(samples / scale)), 0.01)
    r = Pareto(concentration=conc, scale=scale, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_truncatednormal(source, key, **kw):
    from ..distributions.continuous import TruncatedNormal
    if isinstance(source, TruncatedNormal):
        r = TruncatedNormal(loc=source._loc, scale=source._scale, low=source._low, high=source._high, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    samples, m, v = _sample_moments(source, key, kw.pop("num_samples", 1024))
    r = TruncatedNormal(
        loc=m, scale=jnp.sqrt(v),
        low=jnp.min(samples), high=jnp.max(samples),
        name=kw.get("name") or source.name,
    )
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


# -- discrete ---------------------------------------------------------------

def _convert_to_bernoulli(source, key, **kw):
    from ..distributions.discrete import Bernoulli
    if isinstance(source, Bernoulli):
        r = Bernoulli(probs=source._probs, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    kw.pop("num_samples", None)
    r = Bernoulli(probs=source.mean(), name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_binomial(source, key, **kw):
    from ..distributions.discrete import Binomial
    if isinstance(source, Binomial):
        r = Binomial(total_count=source._total_count, probs=source._probs, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    total_count = kw.pop("total_count", None)
    if total_count is None:
        raise ValueError("total_count is required when converting to Binomial from a non-Binomial source.")
    kw.pop("num_samples", None)
    probs = source.mean() / total_count
    r = Binomial(total_count=total_count, probs=probs, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_poisson(source, key, **kw):
    from ..distributions.discrete import Poisson
    if isinstance(source, Poisson):
        r = Poisson(rate=source._rate, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    kw.pop("num_samples", None)
    r = Poisson(rate=source.mean(), name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_categorical(source, key, **kw):
    from ..distributions.discrete import Categorical
    if isinstance(source, Categorical):
        r = Categorical(probs=source._probs, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    num_samples = kw.pop("num_samples", 1024)
    samples = source.sample(key, sample_shape=(num_samples,))
    n_cat = int(jnp.max(samples)) + 1
    counts = jnp.array([(samples == k).sum() for k in range(n_cat)])
    probs = counts / counts.sum()
    r = Categorical(probs=probs, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_negativebinomial(source, key, **kw):
    from ..distributions.discrete import NegativeBinomial
    if isinstance(source, NegativeBinomial):
        r = NegativeBinomial(total_count=source._total_count, probs=source._probs, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    total_count = kw.pop("total_count", None)
    if total_count is None:
        raise ValueError("total_count is required when converting to NegativeBinomial from a non-NegativeBinomial source.")
    kw.pop("num_samples", None)
    m = source.mean()
    probs = total_count / (total_count + m)
    r = NegativeBinomial(total_count=total_count, probs=probs, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


# -- multivariate -----------------------------------------------------------

def _convert_to_multivariatenormal(source, key, **kw):
    from ..distributions.multivariate import MultivariateNormal
    num_samples = kw.pop("num_samples", 1024)
    name = kw.get("name") or source.name
    if isinstance(source, EmpiricalDistribution):
        loc = source.mean()
        cov = source.cov()
    elif isinstance(source, MultivariateNormal):
        r = MultivariateNormal(loc=source.loc, scale_tril=source._scale_tril, name=name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    else:
        samples = source.sample(key, sample_shape=(num_samples,))
        loc = jnp.mean(samples, axis=0)
        diff = samples - loc
        cov = jnp.einsum("ni,nj->ij", diff, diff) / num_samples
    cov = 0.5 * (cov + cov.T)
    cov = cov + 1e-6 * jnp.eye(cov.shape[0])
    r = MultivariateNormal(loc=loc, cov=cov, name=name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_dirichlet(source, key, **kw):
    from ..distributions.multivariate import Dirichlet
    if isinstance(source, Dirichlet):
        r = Dirichlet(concentration=source._concentration, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    samples, m, v = _sample_moments(source, key, kw.pop("num_samples", 1024))
    conc0 = m[0] * (1.0 - m[0]) / (v[0] + 1e-8) - 1.0
    conc0 = jnp.maximum(conc0, 0.01)
    conc = jnp.maximum(m * conc0, 0.01)
    r = Dirichlet(concentration=conc, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_multinomial(source, key, **kw):
    from ..distributions.multivariate import Multinomial
    if isinstance(source, Multinomial):
        r = Multinomial(total_count=source._total_count, probs=source._probs, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    total_count = kw.pop("total_count", None)
    if total_count is None:
        raise ValueError("total_count is required when converting to Multinomial from a non-Multinomial source.")
    num_samples = kw.pop("num_samples", 1024)
    samples = source.sample(key, sample_shape=(num_samples,))
    m = jnp.mean(samples, axis=0)
    probs = m / total_count
    probs = probs / probs.sum()
    r = Multinomial(total_count=total_count, probs=probs, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_wishart(source, key, **kw):
    from ..distributions.multivariate import Wishart
    if isinstance(source, Wishart):
        r = Wishart(df=source._df, scale_tril=source._scale_tril, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    num_samples = kw.pop("num_samples", 1024)
    samples = source.sample(key, sample_shape=(num_samples,))
    mean_mat = jnp.mean(samples, axis=0)
    d = mean_mat.shape[-1]
    df = d + 2.0
    scale_mat = mean_mat / df
    scale_mat = 0.5 * (scale_mat + scale_mat.T)
    scale_mat = scale_mat + 1e-6 * jnp.eye(d)
    r = Wishart(df=df, scale_tril=jnp.linalg.cholesky(scale_mat), name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_vonmisesfisher(source, key, **kw):
    from ..distributions.multivariate import VonMisesFisher
    if isinstance(source, VonMisesFisher):
        r = VonMisesFisher(mean_direction=source._mean_direction, concentration=source._concentration, name=kw.get("name") or source.name)
        r.with_source(Provenance("from_distribution", parents=(source,)))
        return r
    num_samples = kw.pop("num_samples", 1024)
    samples = source.sample(key, sample_shape=(num_samples,))
    mean_vec = jnp.mean(samples, axis=0)
    R = jnp.linalg.norm(mean_vec)
    mean_dir = mean_vec / jnp.maximum(R, 1e-8)
    d = mean_vec.shape[-1]
    R2 = R ** 2
    conc = jnp.maximum(R * (d - R2) / jnp.maximum(1.0 - R2, 1e-8), 0.0)
    r = VonMisesFisher(mean_direction=mean_dir, concentration=conc, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
    return r


def _convert_to_empirical(source, key, **kw):
    """Convert any distribution to EmpiricalDistribution by sampling."""
    if isinstance(source, EmpiricalDistribution):
        return source
    num_samples = kw.pop("num_samples", 1024)
    if key is None:
        key = _auto_key()
    samples = source.sample(key, sample_shape=(num_samples,))
    r = EmpiricalDistribution(samples, name=kw.get("name") or source.name)
    r.with_source(Provenance("from_distribution", parents=(source,)))
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
    }


# ---------------------------------------------------------------------------
# The converter
# ---------------------------------------------------------------------------

class ProbPipeConverter(Converter):
    """Handles conversion between ProbPipe distribution types.

    Absorbs the moment-matching logic formerly in each distribution's
    ``_from_distribution()`` classmethod.
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

    def check(self, source: Any, target_type: type) -> ConversionInfo:
        if not isinstance(source, Distribution):
            return ConversionInfo(feasible=False)
        if not (isinstance(target_type, type) and issubclass(target_type, Distribution)):
            return ConversionInfo(feasible=False)

        target_name = target_type.__name__
        if target_name not in self._table:
            return ConversionInfo(feasible=False, description=f"No converter for target {target_name}")

        # Same class = exact copy
        if isinstance(source, target_type):
            return ConversionInfo(
                feasible=True,
                method=ConversionMethod.EXACT,
                estimated_error=0.0,
                cost=0.0,
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
            estimated_error=0.1,
            cost=0.3,
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
