"""The shipped converters between the catalog's families and backend representations.

Each converter is exact or approximate by its declaration, and its result
carries the source's event declaration:

- ``tfp`` (exact) brings a TFP backend distribution into ProbPipe as its law:
  the family whose parameters it holds, or the bare backend adapter for a
  backend distribution the catalog has no family for;
- ``scipy`` (exact) brings a SciPy frozen distribution into ProbPipe as the
  family whose parameters it holds;
- ``moment_match`` (approximate) fits a parametric family to a law by matching
  its moments, or, for a family fit to draws, the statistics of the law's draws;
- ``empirical`` (approximate) represents a law by the empirical law of its draws;
- ``kde`` (approximate) smooths an empirical law's atoms, or another law's
  draws, into a kernel density estimate.

The approximate converters read a backend source through the law it enters
ProbPipe as, so a backend distribution converts as its law does. The draws a
conversion takes are workflow-owned random events, and the converters register
with the global converter registry at import.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

from .._dtype import _default_float_dtype
from ..core._spec_base import NumericArraySpec, NumericSpec
from ..core._specs import OutputSpec
from ..core.constraints import _supports_compatible
from ..distributions._capabilities import (
    _LAW_CAPABILITIES,
    SupportsCovariance,
    SupportsMean,
    SupportsQuantile,
    SupportsSampling,
    SupportsVariance,
)
from ..distributions._conversion import ConversionInfo, Converter, converter_registry
from ..distributions._distribution import Distribution, DistributionSpec
from ..distributions._empirical import EmpiricalDistribution, _batch_form
from ..functions import _broker
from ..functions._result import SAMPLE_LEVEL
from ..operations._operation import _workflow_draws
from ._backend import TFPDistribution
from ._continuous import (
    Beta,
    Cauchy,
    Exponential,
    Gamma,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    Laplace,
    LogNormal,
    Normal,
    Pareto,
    StudentT,
    TruncatedNormal,
    Uniform,
)
from ._discrete import Bernoulli, Binomial, Categorical, NegativeBinomial, Poisson
from ._multivariate import Dirichlet, Multinomial, MultivariateNormal, VonMisesFisher, Wishart
from ._resampling import KDEDistribution

try:
    import scipy.stats as _stats
    from scipy.stats._distn_infrastructure import rv_frozen as _rv_frozen

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is installed with the package's dependencies
    _HAS_SCIPY = False

__all__: list[str] = []

#: The number of draws a sampling conversion takes when the caller names none.
_DEFAULT_NUM_SAMPLES = 1024

#: The provider contract of the draws of a SciPy distribution, from a workflow-owned key.
_SCIPY_PROVIDER_ABI = "scipy.stats.seedsequence-pcg64/v1"

#: The moment capabilities an empirical law of a numeric event claims.
_NUMERIC_MOMENTS: tuple[type, ...] = (
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
    SupportsQuantile,
)

#: The backend distributions a converter reads besides ProbPipe laws.
_BACKEND_SOURCES: tuple[type, ...] = (tfd.Distribution, *((_rv_frozen,) if _HAS_SCIPY else ()))


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def _refuse_unread(converter: str, options: dict[str, Any], reads: tuple[str, ...]) -> None:
    """Raise ``TypeError`` for an option the converter *converter* does not read.

    Raises
    ------
    TypeError
        If *options* names an option outside *reads*, naming the converter and
        the options it reads.
    """
    unread = sorted(set(options).difference(reads))
    if unread:
        raise TypeError(
            f"the converter {converter!r} reads the options {list(reads)}, and {unread} are not "
            f"among them"
        )


def _sample_count(options: dict[str, Any]) -> int:
    """The ``num_samples`` option as a positive integer, ``1024`` when it is absent.

    Raises
    ------
    TypeError
        If the count is not an integer, a ``bool`` included.
    ValueError
        If the count is not positive.
    """
    value = options.get("num_samples", _DEFAULT_NUM_SAMPLES)
    if isinstance(value, bool):
        raise TypeError(f"num_samples must be an integer; got {value!r}")
    try:
        count = operator.index(value)
    except TypeError:
        raise TypeError(f"num_samples must be an integer; got {value!r}") from None
    if count <= 0:
        raise ValueError(f"num_samples must be positive; got {value!r}")
    return count


def _declared(source: Any, options: dict[str, Any]) -> OutputSpec | None:
    """The ``event_spec`` option, the declaration of a backend source.

    Raises
    ------
    TypeError
        If the option is not an ``OutputSpec``, or is given for a ProbPipe law,
        which carries its own declaration.
    """
    declared = options.get("event_spec")
    if declared is None:
        return None
    if isinstance(source, Distribution):
        raise TypeError(
            f"event_spec declares a backend distribution's event, and {source.label!r} is a "
            f"ProbPipe law that carries its own declaration"
        )
    if not isinstance(declared, OutputSpec):
        raise TypeError(f"event_spec must be an OutputSpec, got {type(declared).__name__}")
    return declared


def _claims(law: Any) -> tuple[type, ...]:
    """The capabilities *law* claims."""
    return tuple(capability for capability in _LAW_CAPABILITIES if isinstance(law, capability))


def _class_claims(cls: type) -> tuple[type, ...]:
    """The capabilities every instance of *cls* claims."""
    return tuple(capability for capability in _LAW_CAPABILITIES if issubclass(cls, capability))


# ---------------------------------------------------------------------------
# Backend distributions entering ProbPipe
# ---------------------------------------------------------------------------

#: The family a TFP backend distribution enters as, with the family's arguments it holds.
_TFP_FAMILIES: dict[type, tuple[type[TFPDistribution], Callable[[Any], dict[str, Any]]]] = {
    tfd.Normal: (Normal, lambda d: {"loc": d.loc, "scale": d.scale}),
    tfd.Beta: (Beta, lambda d: {"alpha": d.concentration1, "beta": d.concentration0}),
    tfd.Gamma: (Gamma, lambda d: {"concentration": d.concentration, "rate": d.rate}),
    tfd.InverseGamma: (
        InverseGamma,
        lambda d: {"concentration": d.concentration, "scale": d.scale},
    ),
    tfd.Exponential: (Exponential, lambda d: {"rate": d.rate}),
    tfd.LogNormal: (LogNormal, lambda d: {"loc": d.loc, "scale": d.scale}),
    tfd.StudentT: (StudentT, lambda d: {"df": d.df, "loc": d.loc, "scale": d.scale}),
    tfd.Uniform: (Uniform, lambda d: {"low": d.low, "high": d.high}),
    tfd.Cauchy: (Cauchy, lambda d: {"loc": d.loc, "scale": d.scale}),
    tfd.Laplace: (Laplace, lambda d: {"loc": d.loc, "scale": d.scale}),
    tfd.HalfNormal: (HalfNormal, lambda d: {"scale": d.scale}),
    tfd.HalfCauchy: (HalfCauchy, lambda d: {"loc": d.loc, "scale": d.scale}),
    tfd.Pareto: (Pareto, lambda d: {"concentration": d.concentration, "scale": d.scale}),
    tfd.Bernoulli: (Bernoulli, lambda d: {"probs": d.probs_parameter()}),
    tfd.Poisson: (Poisson, lambda d: {"rate": d.rate}),
    tfd.Categorical: (Categorical, lambda d: {"probs": d.probs_parameter()}),
    tfd.Dirichlet: (Dirichlet, lambda d: {"concentration": d.concentration}),
    tfd.MultivariateNormalTriL: (
        MultivariateNormal,
        lambda d: {"loc": d.loc, "scale_tril": d.scale_tril},
    ),
    tfd.MultivariateNormalDiag: (
        MultivariateNormal,
        lambda d: {"loc": d.loc, "cov": jnp.diag(d.scale.diag**2)},
    ),
}


def _tfp_law(source: tfd.Distribution, declared: OutputSpec | None) -> TFPDistribution:
    """The exact law the TFP backend distribution *source* enters ProbPipe as.

    A backend distribution with a family enters as that family at its
    parameters, and any other enters through the bare backend adapter. The law
    is labeled by the backend's name, and *declared*, when given, names its
    component.
    """
    name = source.name or type(source).__name__
    entry = _TFP_FAMILIES.get(type(source))
    if entry is None:
        return TFPDistribution(name, source, event_spec=declared)
    family, arguments = entry
    return family(name=name, **arguments(source), event_spec=declared)


def _scipy_arguments(source: Any) -> tuple[tuple[Any, ...], float, float]:
    """The shape arguments, location, and scale of a SciPy frozen distribution."""
    shapes, loc, scale = source.dist._parse_args(*source.args, **source.kwds)
    return shapes, float(loc), float(scale)


#: The family a SciPy frozen distribution enters as, by its distribution's class.
_SCIPY_FAMILIES: dict[type, tuple[type[TFPDistribution], Callable[[Any], dict[str, Any]]]] = {}
if _HAS_SCIPY:
    _SCIPY_FAMILIES = {
        type(_stats.norm): (
            Normal,
            lambda d: (lambda s, loc, scale: {"loc": loc, "scale": scale})(*_scipy_arguments(d)),
        ),
        type(_stats.beta): (
            Beta,
            lambda d: (lambda s, loc, scale: {"alpha": s[0], "beta": s[1]})(*_scipy_arguments(d)),
        ),
        type(_stats.gamma): (
            Gamma,
            lambda d: (lambda s, loc, scale: {"concentration": s[0], "rate": 1.0 / scale})(
                *_scipy_arguments(d)
            ),
        ),
        type(_stats.expon): (
            Exponential,
            lambda d: (lambda s, loc, scale: {"rate": 1.0 / scale})(*_scipy_arguments(d)),
        ),
        type(_stats.lognorm): (
            LogNormal,
            lambda d: (lambda s, loc, scale: {"loc": jnp.log(scale), "scale": s[0]})(
                *_scipy_arguments(d)
            ),
        ),
        type(_stats.uniform): (
            Uniform,
            lambda d: (lambda s, loc, scale: {"low": loc, "high": loc + scale})(
                *_scipy_arguments(d)
            ),
        ),
        type(_stats.cauchy): (
            Cauchy,
            lambda d: (lambda s, loc, scale: {"loc": loc, "scale": scale})(*_scipy_arguments(d)),
        ),
        type(_stats.laplace): (
            Laplace,
            lambda d: (lambda s, loc, scale: {"loc": loc, "scale": scale})(*_scipy_arguments(d)),
        ),
    }


def _scipy_law(source: Any, declared: OutputSpec | None) -> TFPDistribution | None:
    """The family the SciPy frozen distribution *source* enters as, or ``None`` when it has none."""
    entry = _SCIPY_FAMILIES.get(type(source.dist))
    if entry is None:
        return None
    family, arguments = entry
    return family(name=source.dist.name, **arguments(source), event_spec=declared)


def _entering_law(source: Any, declared: OutputSpec | None) -> Distribution | None:
    """*source* as a ProbPipe law: itself, the law a backend distribution enters as, or ``None``.

    ``None`` is a SciPy frozen distribution the catalog has no family for, whose
    draws a sampling converter takes through SciPy itself.
    """
    if isinstance(source, Distribution):
        return source
    if isinstance(source, tfd.Distribution):
        return _tfp_law(source, declared)
    return _scipy_law(source, declared)


def _scipy_declaration(source: Any, declared: OutputSpec | None) -> OutputSpec:
    """The declaration of one draw of a SciPy frozen distribution: a scalar under its name."""
    spec = NumericArraySpec((), jnp.float32)
    if declared is not None:
        return declared.with_spec(spec)
    return OutputSpec(**{source.dist.name: spec})


def _scipy_generator_from_key(key: Any) -> np.random.Generator:
    """Adapt a JAX key's words to the fixed SeedSequence and PCG64 provider."""
    words = np.asarray(jax.random.key_data(key), dtype=np.uint32).reshape(-1)
    seed_sequence = np.random.SeedSequence([int(word) for word in words])
    return np.random.Generator(np.random.PCG64(seed_sequence))


def _draws(source: Any, law: Distribution | None, count: int) -> Any:
    """*count* draws of *source*, the draw axis leading, from a workflow-owned key.

    A law draws through its ``_sample``. A SciPy frozen distribution with no
    family draws through SciPy, from a generator seeded by the derived key.
    """
    if law is not None:
        return _workflow_draws(law, (count,), operation_kind="convert", execution_mode="sampled")
    key = _broker._resolve_automatic_key(
        None,
        _broker._singleton_effect_plan(
            operation_kind="convert",
            execution_mode="sampled",
            sample_shape=(count,),
            provider_abi=_SCIPY_PROVIDER_ABI,
        ),
    )
    return jnp.asarray(source.rvs(size=count, random_state=_scipy_generator_from_key(key)))


# ---------------------------------------------------------------------------
# The exact converters: backend distributions entering ProbPipe
# ---------------------------------------------------------------------------


class _BackendConverter(Converter):
    """An exact converter that brings a backend distribution into ProbPipe as its law."""

    _reads = ("event_spec",)

    @property
    def exact(self) -> bool:
        return True

    @property
    def priority(self) -> int:
        return 10

    def _law(self, source: Any, declared: OutputSpec | None) -> Distribution | None:
        raise NotImplementedError

    def check(self, source: Any, target_type: type, **options: Any) -> ConversionInfo:
        """The law *source* enters ProbPipe as, when it is a *target_type*, read from its parameters."""
        law = self._law(source, _declared(source, options))
        if law is None:
            return ConversionInfo(
                False, description=f"the catalog has no family for {type(source).__name__}"
            )
        if not isinstance(law, target_type):
            return ConversionInfo(
                False,
                description=(
                    f"a {type(source).__name__} enters ProbPipe as a {type(law).__name__}, which "
                    f"is not a {getattr(target_type, '__name__', target_type)}"
                ),
            )
        return ConversionInfo(
            True,
            method_name=self.name,
            exact=True,
            target_spec=law.spec,
            target_class=type(law),
            capabilities=_claims(law),
        )

    def execute(self, source: Any, target_type: type, **options: Any) -> Distribution:
        """The law *source* enters ProbPipe as."""
        _refuse_unread(self.name, options, self._reads)
        law = self._law(source, _declared(source, options))
        if law is None:
            raise TypeError(f"the catalog has no family for {type(source).__name__}")
        return law


class _TFPConverter(_BackendConverter):
    """A TFP backend distribution enters ProbPipe as its family, or through the bare adapter."""

    @property
    def name(self) -> str:
        return "tfp"

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        families = tuple(dict.fromkeys(family for family, _ in _TFP_FAMILIES.values()))
        return ((tfd.Distribution,), (TFPDistribution, *families))

    def _law(self, source: Any, declared: OutputSpec | None) -> Distribution:
        return _tfp_law(source, declared)


class _ScipyConverter(_BackendConverter):
    """A SciPy frozen distribution enters ProbPipe as the family whose parameters it holds."""

    @property
    def name(self) -> str:
        return "scipy"

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        families = tuple(dict.fromkeys(family for family, _ in _SCIPY_FAMILIES.values()))
        return ((_rv_frozen,), families)

    def _law(self, source: Any, declared: OutputSpec | None) -> Distribution | None:
        return _scipy_law(source, declared)


# ---------------------------------------------------------------------------
# Moment matching
# ---------------------------------------------------------------------------


class _Statistics:
    """The statistics of a law that a moment-matched family reads.

    A moment the law claims is its closed form, and any other statistic is read
    from one shared batch of the law's draws, the draw axis leading.
    """

    def __init__(self, law: Distribution | None, draws: Any) -> None:
        self._law = law
        self._draws = None if draws is None else jnp.asarray(draws)

    def draws(self) -> Any:
        if self._draws is None:
            raise RuntimeError("the conversion read draws it did not plan to take")
        return self._draws

    def mean(self) -> Any:
        if isinstance(self._law, SupportsMean):
            return jnp.asarray(self._law._mean())
        return jnp.mean(self.draws(), axis=0)

    def variance(self) -> Any:
        if isinstance(self._law, SupportsVariance):
            return jnp.asarray(self._law._variance())
        return jnp.var(self.draws(), axis=0)

    def covariance(self) -> Any:
        """The covariance of the flattened draw, symmetrized and with ``1e-6`` added to its diagonal."""
        if isinstance(self._law, SupportsCovariance):
            covariance = jnp.asarray(self._law._cov().to_dense())
        else:
            draws = jnp.reshape(self.draws(), (self.draws().shape[0], -1))
            centered = draws - jnp.mean(draws, axis=0)
            covariance = jnp.einsum("ni,nj->ij", centered, centered) / draws.shape[0]
        covariance = 0.5 * (covariance + covariance.T)
        return covariance + 1e-6 * jnp.eye(covariance.shape[0])


def _total_count(options: dict[str, Any], family: type) -> Any:
    total_count = options.get("total_count")
    if total_count is None:
        raise ValueError(
            f"total_count is required when converting to {family.__name__} from a law of "
            f"another class"
        )
    return total_count


def _location_scale(scale: Callable[[Any, Any], Any]) -> Callable[..., dict[str, Any]]:
    """The fit of a location-scale family: the mean, and the scale *scale* gives from the moments."""

    def fit(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
        mean, variance = s.mean(), s.variance()
        return {"loc": mean, "scale": scale(mean, variance)}

    return fit


def _beta(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    mean, variance = s.mean(), s.variance()
    common = mean * (1.0 - mean) / variance - 1.0
    return {
        "alpha": jnp.maximum(mean * common, 0.01),
        "beta": jnp.maximum((1.0 - mean) * common, 0.01),
    }


def _gamma(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    mean, variance = s.mean(), s.variance()
    return {"concentration": mean**2 / variance, "rate": mean / variance}


def _inverse_gamma(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    mean, variance = s.mean(), s.variance()
    return {"concentration": mean**2 / variance + 2.0, "scale": mean * (mean**2 / variance + 1.0)}


def _log_normal(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    mean, variance = s.mean(), s.variance()
    scale = jnp.sqrt(jnp.log(1.0 + variance / mean**2))
    return {"loc": jnp.log(mean) - scale**2 / 2.0, "scale": scale}


#: The degrees of freedom of a Student-t fit; its scale matches the variance at them.
_STUDENT_T_DF = 5.0


def _student_t(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    mean, variance = s.mean(), s.variance()
    scale = jnp.sqrt(variance * (_STUDENT_T_DF - 2.0) / _STUDENT_T_DF)
    return {"df": _STUDENT_T_DF, "loc": mean, "scale": scale}


def _uniform(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    mean, half = s.mean(), jnp.sqrt(3.0 * s.variance())
    return {"low": mean - half, "high": mean + half}


def _half_normal(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    return {"scale": jnp.sqrt(s.variance() / (1.0 - 2.0 / jnp.pi))}


def _half_cauchy(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    """The scale of each coordinate is the median of its draws, at least ``0.01``."""
    scale = jnp.maximum(jnp.median(s.draws(), axis=0), 0.01)
    return {"loc": jnp.zeros_like(scale), "scale": scale}


def _pareto(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    """The maximum-likelihood fit of each coordinate's draws."""
    draws = s.draws()
    scale = jnp.maximum(jnp.min(draws, axis=0), 1e-6)
    concentration = draws.shape[0] / jnp.sum(jnp.log(draws / scale), axis=0)
    return {"concentration": jnp.maximum(concentration, 0.01), "scale": scale}


def _truncated_normal(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    """The moments, truncated to the range of each coordinate's draws."""
    draws = s.draws()
    return {
        "loc": s.mean(),
        "scale": jnp.sqrt(s.variance()),
        "low": jnp.min(draws, axis=0),
        "high": jnp.max(draws, axis=0),
    }


def _bernoulli(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    return {"probs": s.mean()}


def _binomial(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    total_count = _total_count(options, Binomial)
    return {"total_count": total_count, "probs": s.mean() / total_count}


def _poisson(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    return {"rate": s.mean()}


def _categorical(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    """The frequencies of the draws' categories, counting from zero to the largest drawn."""
    draws = s.draws()
    counts = jnp.bincount(jnp.asarray(draws, dtype=jnp.int32), length=int(jnp.max(draws)) + 1)
    return {"probs": counts / counts.sum()}


def _negative_binomial(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    total_count = _total_count(options, NegativeBinomial)
    return {"total_count": total_count, "probs": total_count / (total_count + s.mean())}


def _multivariate_normal(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    return {"loc": s.mean(), "cov": s.covariance()}


def _dirichlet(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    """The concentration whose mean is the law's and whose first coordinate's variance matches."""
    mean, variance = s.mean(), s.variance()
    total = jnp.maximum(mean[0] * (1.0 - mean[0]) / (variance[0] + 1e-8) - 1.0, 0.01)
    return {"concentration": jnp.maximum(mean * total, 0.01)}


def _multinomial(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    total_count = _total_count(options, Multinomial)
    probs = s.mean() / total_count
    return {"total_count": total_count, "probs": probs / probs.sum()}


def _wishart(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    """The scale matrix whose mean, at ``d + 2`` degrees of freedom, is the law's."""
    mean = s.mean()
    size = mean.shape[-1]
    df = size + 2.0
    scale = mean / df
    scale = 0.5 * (scale + scale.T) + 1e-6 * jnp.eye(size)
    return {"df": df, "scale_tril": jnp.linalg.cholesky(scale)}


def _von_mises_fisher(s: _Statistics, options: dict[str, Any]) -> dict[str, Any]:
    """The mean direction, and the concentration Banerjee's approximation gives from its length."""
    mean = s.mean()
    length = jnp.linalg.norm(mean)
    size = mean.shape[-1]
    concentration = length * (size - length**2) / jnp.maximum(1.0 - length**2, 1e-8)
    return {
        "mean_direction": mean / jnp.maximum(length, 1e-8),
        "concentration": jnp.maximum(concentration, 0.0),
    }


#: Each moment-matched family: the statistics its fit reads, the fit, and the rank of its
#: event, ``None`` for a family of independent coordinates of any shape.
_FITS: dict[
    type, tuple[frozenset[str], Callable[[_Statistics, dict[str, Any]], dict[str, Any]], int | None]
] = {
    Normal: (frozenset({"mean", "variance"}), _location_scale(lambda m, v: jnp.sqrt(v)), None),
    Beta: (frozenset({"mean", "variance"}), _beta, None),
    Gamma: (frozenset({"mean", "variance"}), _gamma, None),
    InverseGamma: (frozenset({"mean", "variance"}), _inverse_gamma, None),
    Exponential: (frozenset({"mean"}), lambda s, options: {"rate": 1.0 / s.mean()}, None),
    LogNormal: (frozenset({"mean", "variance"}), _log_normal, None),
    StudentT: (frozenset({"mean", "variance"}), _student_t, None),
    Uniform: (frozenset({"mean", "variance"}), _uniform, None),
    Cauchy: (
        frozenset({"mean", "variance"}),
        _location_scale(lambda m, v: jnp.sqrt(v) / 2.0),
        None,
    ),
    Laplace: (
        frozenset({"mean", "variance"}),
        _location_scale(lambda m, v: jnp.sqrt(v / 2.0)),
        None,
    ),
    HalfNormal: (frozenset({"variance"}), _half_normal, None),
    HalfCauchy: (frozenset({"draws"}), _half_cauchy, None),
    Pareto: (frozenset({"draws"}), _pareto, None),
    TruncatedNormal: (frozenset({"mean", "variance", "draws"}), _truncated_normal, None),
    Bernoulli: (frozenset({"mean"}), _bernoulli, None),
    Binomial: (frozenset({"mean"}), _binomial, None),
    Poisson: (frozenset({"mean"}), _poisson, None),
    Categorical: (frozenset({"draws"}), _categorical, 0),
    NegativeBinomial: (frozenset({"mean"}), _negative_binomial, None),
    MultivariateNormal: (frozenset({"mean", "covariance"}), _multivariate_normal, 1),
    Dirichlet: (frozenset({"mean", "variance"}), _dirichlet, 1),
    Multinomial: (frozenset({"mean"}), _multinomial, 1),
    Wishart: (frozenset({"mean"}), _wishart, 2),
    VonMisesFisher: (frozenset({"mean"}), _von_mises_fisher, 1),
}

#: The families whose fit needs a total count, which the option ``total_count`` gives.
_COUNTED = (Binomial, NegativeBinomial, Multinomial)

#: The fitted families whose draws are integers, with their dtype; every other family
#: draws the default floating dtype.
_INTEGER_DRAWS: dict[type, Any] = {Bernoulli: jnp.int32, Categorical: jnp.int32}


def _fit_dtype(family: type) -> np.dtype:
    """The dtype of a draw of the fitted *family*, which its promise declares."""
    return np.dtype(_INTEGER_DRAWS.get(family, _default_float_dtype()))


#: The capability that gives each statistic in closed form.
_CLOSED_FORM = {
    "mean": SupportsMean,
    "variance": SupportsVariance,
    "covariance": SupportsCovariance,
}


def _samples(law: Distribution | None, needs: frozenset[str]) -> bool:
    """Whether a fit reading *needs* takes draws: a statistic *law* has in no closed form.

    A SciPy frozen distribution without a family, whose *law* is ``None``, has
    every statistic from its draws.
    """
    return any(
        need not in _CLOSED_FORM or not isinstance(law, _CLOSED_FORM[need]) for need in needs
    )


def _array_event(declaration: OutputSpec, label: str, family: type, rank: int | None) -> str | None:
    """Why the event *declaration* cannot be a draw of *family*, or ``None`` when it can.

    A family draws one array, as a whole term, of the rank its event has.
    """
    spec = declaration.spec
    if declaration.exposes_record or not isinstance(spec, NumericArraySpec):
        return (
            f"{family.__name__} draws one array, and {label!r} declares "
            f"{'an exposed record' if declaration.exposes_record else type(spec).__name__}; "
            f"convert a field's law d[path] instead"
        )
    if spec.free_dims:
        return f"{label!r} has unbound dimensions {sorted(spec.free_dims)}"
    shape = spec.shape
    if rank == 2 and (len(shape) != 2 or shape[0] != shape[1]):
        return f"{family.__name__} draws a square matrix, and {label!r} draws shape {shape}"
    if rank is not None and rank != 2 and len(shape) != rank:
        return f"{family.__name__} draws an array of rank {rank}, and {label!r} draws shape {shape}"
    return None


def _cast_event(declaration: OutputSpec, label: str, family: type) -> str | None:
    """Why a draw of *family* does not cast to the dtype *declaration* sets, or ``None``.

    A conversion's result carries the source's declaration, so the family's
    dtype must cast to the source's by the same-kind rule, as a law's must to
    the ``DistributionSpec`` it matches.
    """
    declared = declaration.spec.dtype
    drawn = _fit_dtype(family)
    if declared is None or np.can_cast(drawn, declared, casting="same_kind"):
        return None
    return (
        f"{family.__name__} draws {drawn}, which does not cast to the dtype {declared} "
        f"that {label!r} declares"
    )


def _check_support(result: Distribution, law: Distribution) -> None:
    """Refuse a fit whose support does not contain the law's.

    A law whose support is undeclared has nothing to compare, unless it is an
    empirical law, whose atoms are what it is supported on and must each lie in
    the fit's support.

    Raises
    ------
    ValueError
        If the law's support is not contained in the fit's, or an atom lies
        outside it.
    """
    target = result.support
    if target is None:
        return
    source = getattr(law, "support", None)
    if source is not None:
        if not _supports_compatible(source, target):
            raise ValueError(
                f"Cannot convert {type(law).__name__} {law.label!r} (support={source}) to "
                f"{type(result).__name__} (support={target}). Pass check_support=False to "
                f"override."
            )
        return
    if isinstance(law, EmpiricalDistribution) and not bool(
        jnp.all(target.check(jnp.asarray(law._rows)))
    ):
        raise ValueError(
            f"Cannot convert {type(law).__name__} {law.label!r} to {type(result).__name__} "
            f"(support={target}): its atoms lie outside that support. Pass check_support=False "
            f"to override."
        )


@dataclass(frozen=True)
class _MomentFit:
    """The plan of one moment-matched fit, before it reads any statistic.

    ``law`` is ``None`` for a SciPy frozen distribution without a family, whose
    statistics all come from its draws.
    """

    law: Distribution | None
    declaration: OutputSpec
    label: str
    parameters: Callable[[_Statistics, dict[str, Any]], dict[str, Any]]
    samples: bool


class _MomentMatching(Converter):
    """A parametric family fit to a law, matching its moments or its draws' statistics.

    The family is the requested target, and the fit reads the statistics the
    family's parameters need: a moment the law claims in closed form, and any
    other statistic from one shared batch of ``num_samples`` draws. The fit
    keeps the law's label and component, and a family whose draws do not cast
    to the law's dtype is infeasible, as a ``Normal`` fit to a ``Bernoulli``
    law is. Unless ``check_support=False``, its support must contain the
    law's. The counted families need the option ``total_count``.
    """

    _reads = ("num_samples", "total_count", "check_support", "event_spec")

    @property
    def name(self) -> str:
        return "moment_match"

    @property
    def exact(self) -> bool:
        return False

    @property
    def priority(self) -> int:
        return 10

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return ((Distribution, *_BACKEND_SOURCES), tuple(_FITS))

    def _fit(self, source: Any, target_type: type, options: dict[str, Any]) -> _MomentFit | str:
        """The plan of the fit of *target_type* to *source*, or the reason it is infeasible.

        Raises
        ------
        ValueError
            If a counted family's ``total_count`` is absent, or ``num_samples``
            is not positive where the fit draws.
        TypeError
            If ``num_samples`` is not an integer where the fit draws.
        """
        fit = _FITS.get(target_type)
        if fit is None:
            return (
                f"moment matching fits the parametric family the target names, and "
                f"{getattr(target_type, '__name__', target_type)} names none"
            )
        declared = _declared(source, options)
        law = _entering_law(source, declared)
        if law is None:
            declaration, label = _scipy_declaration(source, declared), source.dist.name
        else:
            declaration, label = law.event_spec, law.label
        needs, parameters, rank = fit
        reason = _array_event(declaration, label, target_type, rank) or _cast_event(
            declaration, label, target_type
        )
        if reason is not None:
            return reason
        if target_type in _COUNTED:
            _total_count(options, target_type)
        samples = _samples(law, needs)
        if samples:
            if law is not None and not isinstance(law, SupportsSampling):
                return f"{label!r} has no closed-form moments for the fit and does not sample"
            _sample_count(options)
        return _MomentFit(law, declaration, label, parameters, samples)

    def check(self, source: Any, target_type: type, **options: Any) -> ConversionInfo:
        """Promise the family *target_type* over the source's declaration, without fitting it.

        The promised declaration is the source's shape with the family's dtype.
        A fit whose draws do not cast to the source's dtype by the same-kind
        rule is infeasible, so the registry tries the next converter. The
        support is left to the fit.

        Raises
        ------
        ValueError, TypeError
            As the plan of the fit raises them.
        """
        planned = self._fit(source, target_type, options)
        if isinstance(planned, str):
            return ConversionInfo(False, description=planned)
        declaration = planned.declaration
        promised = declaration._with_spec(
            NumericArraySpec(declaration.spec.shape, _fit_dtype(target_type))
        )
        return ConversionInfo(
            True,
            method_name=self.name,
            exact=False,
            target_spec=DistributionSpec(promised),
            target_class=target_type,
            capabilities=_class_claims(target_type),
            samples=planned.samples,
        )

    def execute(self, source: Any, target_type: type, **options: Any) -> Distribution:
        """The fit of the family *target_type* to the source, over its declaration.

        Raises
        ------
        TypeError
            If an option is not one the converter reads, or the fit is infeasible.
        ValueError
            As :meth:`check` raises it, or if the fit's support does not contain
            the law's.
        """
        _refuse_unread(self.name, options, self._reads)
        planned = self._fit(source, target_type, options)
        if isinstance(planned, str):
            raise TypeError(planned)
        law = planned.law
        draws = _draws(source, law, _sample_count(options)) if planned.samples else None
        (component,) = planned.declaration.components
        result = target_type(
            name=planned.label,
            **planned.parameters(_Statistics(law, draws), options),
            event_spec=OutputSpec(**{component: None}),
        )
        if options.get("check_support", True):
            _check_support(result, law)
        return result


# ---------------------------------------------------------------------------
# Sample-based representations
# ---------------------------------------------------------------------------


def _sampled_source(
    source: Any, options: dict[str, Any]
) -> tuple[Distribution | None, OutputSpec] | str:
    """The law a sampling converter draws from, with the declaration of a draw, or why it cannot.

    A SciPy frozen distribution without a family has no law, and draws through
    SciPy under the declaration of a scalar named by its distribution.
    """
    declared = _declared(source, options)
    law = _entering_law(source, declared)
    if law is None:
        return None, _scipy_declaration(source, declared)
    if not isinstance(law, SupportsSampling):
        return f"{law.label!r} does not sample"
    return law, law.event_spec


def _label(source: Any, law: Distribution | None, declaration: OutputSpec) -> str:
    """The label of a sampled representation: the law's, or a SciPy distribution's name."""
    return law.label if law is not None else source.dist.name


class _EmpiricalDraws(Converter):
    """The empirical law of ``num_samples`` draws of a law, its atoms on the level ``sample``.

    Besides the class, it declares the moment capabilities an empirical law of
    a numeric event claims, so it serves a request for a moment of a law that
    has none in closed form.
    """

    _reads = ("num_samples", "event_spec")

    @property
    def name(self) -> str:
        return "empirical"

    @property
    def exact(self) -> bool:
        return False

    @property
    def priority(self) -> int:
        return 20

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return ((Distribution, *_BACKEND_SOURCES), (EmpiricalDistribution, *_NUMERIC_MOMENTS))

    def check(self, source: Any, target_type: type, **options: Any) -> ConversionInfo:
        """Promise the empirical law of the source's draws, over its declaration.

        Raises
        ------
        TypeError, ValueError
            If ``num_samples`` is not a positive integer.
        """
        _sample_count(options)
        sampled = _sampled_source(source, options)
        if isinstance(sampled, str):
            return ConversionInfo(False, description=sampled)
        _, declaration = sampled
        numeric = isinstance(declaration.spec, NumericSpec)
        claims = _class_claims(EmpiricalDistribution) + (_NUMERIC_MOMENTS if numeric else ())
        return ConversionInfo(
            True,
            method_name=self.name,
            exact=False,
            target_spec=DistributionSpec(declaration),
            target_class=EmpiricalDistribution,
            capabilities=claims,
            samples=True,
        )

    def execute(self, source: Any, target_type: type, **options: Any) -> Distribution:
        """The empirical law of ``num_samples`` draws, keeping the source's declaration.

        Raises
        ------
        TypeError
            If an option is not one the converter reads, or the source does not
            sample.
        """
        _refuse_unread(self.name, options, self._reads)
        count = _sample_count(options)
        sampled = _sampled_source(source, options)
        if isinstance(sampled, str):
            raise TypeError(sampled)
        law, declaration = sampled
        name = _label(source, law, declaration)
        atoms = _batch_form(name, _draws(source, law, count), SAMPLE_LEVEL, declaration.spec)
        return EmpiricalDistribution(name, atoms, event_spec=declaration)


def _smoothed_atoms(name: str, values: Any, spec: Any) -> Any:
    """*values*, raw values of *spec* along one axis, as a KDE takes its atoms.

    An array event's values are an array, and a record event's are a batch of
    records on one level.
    """
    if isinstance(spec, NumericArraySpec):
        return jnp.asarray(values)
    return _batch_form(name, values, SAMPLE_LEVEL, spec)


class _KDESmoothing(Converter):
    """The kernel density estimate of a law: its atoms for an empirical law, else its draws.

    An empirical law's atoms and weights are the estimate's, and any other law
    is smoothed from ``num_samples`` of its draws. The option ``bandwidth`` is
    the estimate's, Scott's rule when it is absent.
    """

    _reads = ("bandwidth", "num_samples", "event_spec")

    @property
    def name(self) -> str:
        return "kde"

    @property
    def exact(self) -> bool:
        return False

    @property
    def priority(self) -> int:
        return 10

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return ((Distribution, *_BACKEND_SOURCES), (KDEDistribution,))

    def _planned(self, source: Any, options: dict[str, Any]) -> Any:
        sampled = _sampled_source(source, options)
        if isinstance(sampled, str) and not isinstance(source, EmpiricalDistribution):
            return sampled
        law, declaration = (source, source.event_spec) if isinstance(sampled, str) else sampled
        if not isinstance(declaration.spec, NumericSpec):
            return (
                f"a KDE smooths numeric atoms, and {_label(source, law, declaration)!r} declares "
                f"{type(declaration.spec).__name__}"
            )
        samples = not isinstance(law, EmpiricalDistribution)
        if samples:
            _sample_count(options)
        return law, declaration, samples

    def check(self, source: Any, target_type: type, **options: Any) -> ConversionInfo:
        """Promise the kernel density estimate of the source, over its declaration.

        Raises
        ------
        TypeError, ValueError
            If ``num_samples`` is not a positive integer where the estimate
            draws.
        """
        planned = self._planned(source, options)
        if isinstance(planned, str):
            return ConversionInfo(False, description=planned)
        _, declaration, samples = planned
        return ConversionInfo(
            True,
            method_name=self.name,
            exact=False,
            target_spec=DistributionSpec(declaration),
            target_class=KDEDistribution,
            capabilities=_class_claims(KDEDistribution),
            samples=samples,
        )

    def execute(self, source: Any, target_type: type, **options: Any) -> Distribution:
        """The kernel density estimate, keeping the source's declaration.

        Raises
        ------
        TypeError
            If an option is not one the converter reads, or the source's event
            is not numeric.
        """
        _refuse_unread(self.name, options, self._reads)
        planned = self._planned(source, options)
        if isinstance(planned, str):
            raise TypeError(planned)
        law, declaration, samples = planned
        name = _label(source, law, declaration)
        bandwidth = options.get("bandwidth")
        if not samples:
            return KDEDistribution(
                name,
                _smoothed_atoms(name, law._rows, declaration.spec),
                bandwidth,
                law.weights,
                event_spec=declaration,
            )
        draws = _draws(source, law, _sample_count(options))
        atoms = _smoothed_atoms(name, draws, declaration.spec)
        return KDEDistribution(name, atoms, bandwidth, event_spec=declaration)


converter_registry.register(_TFPConverter())
if _HAS_SCIPY:
    converter_registry.register(_ScipyConverter())
converter_registry.register(_MomentMatching())
converter_registry.register(_EmpiricalDraws())
converter_registry.register(_KDESmoothing())
