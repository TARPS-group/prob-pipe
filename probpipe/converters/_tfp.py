"""Raw TFP distribution converter.

Converts ``tfd.*`` instances to/from ProbPipe distribution types.
Always available since TFP is a hard dependency.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ..custom_types import PRNGKey
from .._utils import _auto_key
from ..core.distribution import (
    ArrayDistribution,
    ArrayEmpiricalDistribution,
)
from ..core.provenance import Provenance
from ._registry import Converter
from ._registry import ConversionInfo, ConversionMethod


# ---------------------------------------------------------------------------
# TFP → ProbPipe mapping: (ProbPipe class, parameter extractor)
# ---------------------------------------------------------------------------

def _build_tfp_to_probpipe() -> dict[type, tuple[str, callable]]:
    """Build mapping lazily to avoid circular imports."""
    from ..distributions.continuous import (
        Normal, Beta, Gamma, InverseGamma, Exponential, LogNormal,
        StudentT, Uniform, Cauchy, Laplace, HalfNormal, HalfCauchy,
        Pareto,
    )
    from ..distributions.discrete import (
        Bernoulli, Binomial, Poisson, Categorical, NegativeBinomial,
    )
    from ..distributions.multivariate import (
        MultivariateNormal, Dirichlet,
    )

    return {
        tfd.Normal: (Normal, lambda d: {"loc": d.loc, "scale": d.scale}),
        tfd.Beta: (Beta, lambda d: {"alpha": d.concentration1, "beta": d.concentration0}),
        tfd.Gamma: (Gamma, lambda d: {"concentration": d.concentration, "rate": d.rate}),
        tfd.InverseGamma: (InverseGamma, lambda d: {"concentration": d.concentration, "scale": d.scale}),
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
            lambda d: {"loc": d.loc, "cov": jnp.diag(d.scale.diag ** 2)},
        ),
    }


def _build_probpipe_to_tfp() -> dict[str, callable]:
    """Build ProbPipe → TFP mapping lazily."""
    return {
        "Normal": lambda d: tfd.Normal(loc=d._loc, scale=d._scale),
        "Beta": lambda d: tfd.Beta(concentration1=d._alpha, concentration0=d._beta),
        "Gamma": lambda d: tfd.Gamma(concentration=d._concentration, rate=d._rate),
        "Exponential": lambda d: tfd.Exponential(rate=d._rate),
        "LogNormal": lambda d: tfd.LogNormal(loc=d._loc, scale=d._scale),
        "StudentT": lambda d: tfd.StudentT(df=d._df, loc=d._loc, scale=d._scale),
        "Uniform": lambda d: tfd.Uniform(low=d._low, high=d._high),
        "Cauchy": lambda d: tfd.Cauchy(loc=d._loc, scale=d._scale),
        "Laplace": lambda d: tfd.Laplace(loc=d._loc, scale=d._scale),
        "HalfNormal": lambda d: tfd.HalfNormal(scale=d._scale),
        "HalfCauchy": lambda d: tfd.HalfCauchy(loc=d._loc, scale=d._scale),
        "Pareto": lambda d: tfd.Pareto(concentration=d._concentration, scale=d._scale),
        "InverseGamma": lambda d: tfd.InverseGamma(concentration=d._concentration, scale=d._scale),
        "Bernoulli": lambda d: tfd.Bernoulli(probs=d._probs),
        "Poisson": lambda d: tfd.Poisson(rate=d._rate),
        "Categorical": lambda d: tfd.Categorical(probs=d._probs),
        "Dirichlet": lambda d: tfd.Dirichlet(concentration=d._concentration),
        "MultivariateNormal": lambda d: tfd.MultivariateNormalTriL(loc=d.loc, scale_tril=d._scale_tril),
    }


class TFPConverter(Converter):
    """Bidirectional converter between raw TFP and ProbPipe distributions."""

    def __init__(self) -> None:
        self._to_probpipe: dict | None = None
        self._to_tfp: dict | None = None

    @property
    def _tfp_map(self):
        if self._to_probpipe is None:
            self._to_probpipe = _build_tfp_to_probpipe()
        return self._to_probpipe

    @property
    def _pp_map(self):
        if self._to_tfp is None:
            self._to_tfp = _build_probpipe_to_tfp()
        return self._to_tfp

    def source_types(self) -> tuple[type, ...]:
        return (tfd.Distribution, ArrayDistribution, ArrayEmpiricalDistribution)

    def target_types(self) -> tuple[type, ...]:
        return (ArrayDistribution, ArrayEmpiricalDistribution, tfd.Distribution)

    @staticmethod
    def _is_probpipe_target(target_type: type) -> bool:
        return isinstance(target_type, type) and (
            issubclass(target_type, ArrayDistribution)
            or issubclass(target_type, ArrayEmpiricalDistribution)
        )

    def check(self, source: Any, target_type: type) -> ConversionInfo:
        # Case 1: TFP -> ProbPipe
        if isinstance(source, tfd.Distribution) and not isinstance(source, ArrayDistribution):
            if self._is_probpipe_target(target_type):
                src_cls = type(source)
                if src_cls in self._tfp_map:
                    pp_cls, _ = self._tfp_map[src_cls]
                    if target_type is pp_cls or issubclass(pp_cls, target_type):
                        return ConversionInfo(
                            feasible=True, method=ConversionMethod.EXACT,
                            estimated_time=0.0,
                            source_type=src_cls, target_type=target_type,
                            description=f"Extract parameters from {src_cls.__name__}",
                        )
                    # Known TFP -> natural ProbPipe -> requested ProbPipe
                    return ConversionInfo(
                        feasible=True, method=ConversionMethod.MOMENT_MATCH,
                        estimated_time=0.1,
                        source_type=src_cls, target_type=target_type,
                        description=f"TFP {src_cls.__name__} -> ProbPipe -> {target_type.__name__}",
                    )
                # Unknown TFP type -> sample fallback
                if issubclass(target_type, (ArrayDistribution, ArrayEmpiricalDistribution)):
                    return ConversionInfo(
                        feasible=True, method=ConversionMethod.SAMPLE,
                        estimated_time=0.2,
                        source_type=src_cls, target_type=target_type,
                        description=f"Sample {src_cls.__name__} -> ArrayEmpiricalDistribution",
                    )

        # Case 2: ProbPipe -> TFP
        if isinstance(source, ArrayDistribution) and isinstance(target_type, type) and issubclass(target_type, tfd.Distribution):
            src_name = type(source).__name__
            if src_name in self._pp_map:
                return ConversionInfo(
                    feasible=True, method=ConversionMethod.EXACT,
                    estimated_time=0.0,
                    source_type=type(source), target_type=target_type,
                    description=f"Extract parameters from ProbPipe {src_name}",
                )

        return ConversionInfo(feasible=False)

    def convert(self, source: Any, target_type: type, *, key: Any | None = None, **kwargs: Any) -> Any:
        # Case 1: TFP -> ProbPipe
        if isinstance(source, tfd.Distribution) and not isinstance(source, ArrayDistribution):
            if self._is_probpipe_target(target_type):
                src_cls = type(source)
                if src_cls in self._tfp_map:
                    pp_cls, extractor = self._tfp_map[src_cls]
                    pp_dist = pp_cls(**extractor(source))
                    pp_dist.with_source(Provenance("convert_from_tfp", parents=()))
                    if isinstance(pp_dist, target_type):
                        return pp_dist
                    # Chain: TFP -> natural ProbPipe -> target ProbPipe
                    from ._registry import converter_registry
                    return converter_registry.convert(pp_dist, target_type, key=key, **kwargs)

                # Unknown TFP: sample -> ArrayEmpiricalDistribution
                if key is None:
                    key = _auto_key()
                n = kwargs.pop("num_samples", 1024)
                samples = source.sample(seed=key, sample_shape=(n,))
                emp = ArrayEmpiricalDistribution(samples)
                emp.with_source(Provenance("convert_from_tfp", parents=()))
                if issubclass(target_type, ArrayEmpiricalDistribution):
                    return emp
                from ._registry import converter_registry
                return converter_registry.convert(emp, target_type, key=key, **kwargs)

        # Case 2: ProbPipe -> TFP
        if isinstance(source, ArrayDistribution):
            src_name = type(source).__name__
            fn = self._pp_map.get(src_name)
            if fn is not None:
                return fn(source)
            raise TypeError(f"Cannot convert {src_name} to TFP distribution")

        raise TypeError(f"TFPConverter cannot handle {type(source).__name__} -> {target_type.__name__}")

    @property
    def priority(self) -> int:
        return 50
