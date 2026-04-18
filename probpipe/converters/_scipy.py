"""Scipy distribution converter (optional dependency).

Converts ``scipy.stats`` frozen distribution instances to/from ProbPipe
types.  Only registered when scipy is importable.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .._utils import _auto_key
from ..core.distribution import (
    NumericRecordDistribution,
    NumericEmpiricalDistribution,
)
from ..core.provenance import Provenance
from ._registry import ConversionInfo, ConversionMethod, Converter

try:
    import scipy.stats as _stats
    from scipy.stats._distn_infrastructure import rv_frozen as _rv_frozen
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    _rv_frozen = None


def _build_scipy_to_probpipe() -> dict[type, tuple[str, callable]]:
    """Build scipy.stats type → (ProbPipe class, kwargs extractor)."""
    from ..distributions.continuous import (
        Normal, Beta, Gamma, Exponential, LogNormal,
        Uniform, Cauchy, Laplace,
    )
    if not _HAS_SCIPY:
        return {}

    def _extract(d):
        """Extract (shape_args, loc, scale) using scipy's internal parser.

        Scipy frozen distributions store positional args in ``d.args``
        and keyword args in ``d.kwds``.  The positional args may include
        loc/scale depending on how the distribution was created.  We
        use the underlying distribution's ``_parse_args``  to reliably
        separate shape parameters from loc/scale.
        """
        shapes, loc, scale = d.dist._parse_args(*d.args, **d.kwds)
        return shapes, float(loc), float(scale)

    # Keys are distribution *classes* (norm_gen, beta_gen, etc.)
    return {
        type(_stats.norm): (Normal, lambda d: (
            lambda s, loc, scale: {"loc": loc, "scale": scale}
        )(*_extract(d))),
        type(_stats.beta): (Beta, lambda d: (
            lambda s, loc, scale: {"alpha": s[0], "beta": s[1]}
        )(*_extract(d))),
        type(_stats.gamma): (Gamma, lambda d: (
            lambda s, loc, scale: {"concentration": s[0], "rate": 1.0 / scale}
        )(*_extract(d))),
        type(_stats.expon): (Exponential, lambda d: (
            lambda s, loc, scale: {"rate": 1.0 / scale}
        )(*_extract(d))),
        type(_stats.lognorm): (LogNormal, lambda d: (
            lambda s, loc, scale: {"loc": jnp.log(scale), "scale": s[0]}
        )(*_extract(d))),
        type(_stats.uniform): (Uniform, lambda d: (
            lambda s, loc, scale: {"low": loc, "high": loc + scale}
        )(*_extract(d))),
        type(_stats.cauchy): (Cauchy, lambda d: (
            lambda s, loc, scale: {"loc": loc, "scale": scale}
        )(*_extract(d))),
        type(_stats.laplace): (Laplace, lambda d: (
            lambda s, loc, scale: {"loc": loc, "scale": scale}
        )(*_extract(d))),
    }


def _build_probpipe_to_scipy() -> dict[str, callable]:
    """Build ProbPipe class name → scipy frozen dist constructor."""
    if not _HAS_SCIPY:
        return {}
    return {
        "Normal": lambda d: _stats.norm(loc=float(d._loc), scale=float(d._scale)),
        "Beta": lambda d: _stats.beta(float(d._alpha), float(d._beta)),
        "Gamma": lambda d: _stats.gamma(float(d._concentration), scale=1.0 / float(d._rate)),
        "Exponential": lambda d: _stats.expon(scale=1.0 / float(d._rate)),
        "Uniform": lambda d: _stats.uniform(loc=float(d._low), scale=float(d._high) - float(d._low)),
        "Cauchy": lambda d: _stats.cauchy(loc=float(d._loc), scale=float(d._scale)),
        "Laplace": lambda d: _stats.laplace(loc=float(d._loc), scale=float(d._scale)),
    }


class ScipyConverter(Converter):
    """Bidirectional converter between scipy.stats and ProbPipe distributions."""

    def __init__(self) -> None:
        self._to_probpipe: dict | None = None
        self._to_scipy: dict | None = None

    @property
    def _scipy_map(self):
        if self._to_probpipe is None:
            self._to_probpipe = _build_scipy_to_probpipe()
        return self._to_probpipe

    @property
    def _pp_map(self):
        if self._to_scipy is None:
            self._to_scipy = _build_probpipe_to_scipy()
        return self._to_scipy

    def source_types(self) -> tuple[type, ...]:
        if not _HAS_SCIPY:
            return ()
        return (_rv_frozen, NumericRecordDistribution, NumericEmpiricalDistribution)

    def target_types(self) -> tuple[type, ...]:
        if not _HAS_SCIPY:
            return ()
        return (NumericRecordDistribution, NumericEmpiricalDistribution, _rv_frozen)

    @staticmethod
    def _is_probpipe_target(target_type: type) -> bool:
        return isinstance(target_type, type) and (
            issubclass(target_type, NumericRecordDistribution)
            or issubclass(target_type, NumericEmpiricalDistribution)
        )

    def check(self, source: Any, target_type: type) -> ConversionInfo:
        if not _HAS_SCIPY:
            return ConversionInfo(feasible=False)

        # Case 1: scipy -> ProbPipe
        if isinstance(source, _rv_frozen):
            if self._is_probpipe_target(target_type):
                # Find the underlying scipy distribution class
                dist_cls = type(source.dist)
                if dist_cls in self._scipy_map:
                    pp_cls, _ = self._scipy_map[dist_cls]
                    if target_type is pp_cls or issubclass(pp_cls, target_type):
                        return ConversionInfo(
                            feasible=True, method=ConversionMethod.EXACT,
                            estimated_time=0.0,
                            source_type=type(source), target_type=target_type,
                            description=f"Extract parameters from scipy {dist_cls.__name__}",
                        )
                    return ConversionInfo(
                        feasible=True, method=ConversionMethod.MOMENT_MATCH,
                        estimated_time=0.1,
                        source_type=type(source), target_type=target_type,
                    )
                # Unknown scipy dist -> sample
                return ConversionInfo(
                    feasible=True, method=ConversionMethod.SAMPLE,
                    estimated_time=0.2,
                    source_type=type(source), target_type=target_type,
                    description="Sample from scipy distribution",
                )

        # Case 2: ProbPipe -> scipy
        if isinstance(source, NumericRecordDistribution):
            if _HAS_SCIPY and isinstance(target_type, type) and issubclass(target_type, _rv_frozen):
                src_name = type(source).__name__
                if src_name in self._pp_map:
                    return ConversionInfo(
                        feasible=True, method=ConversionMethod.EXACT,
                        estimated_time=0.0,
                        source_type=type(source), target_type=target_type,
                    )

        return ConversionInfo(feasible=False)

    def convert(self, source: Any, target_type: type, *, key: Any | None = None, **kwargs: Any) -> Any:
        if not _HAS_SCIPY:
            raise TypeError("scipy is not installed")

        # Case 1: scipy -> ProbPipe
        if isinstance(source, _rv_frozen):
            dist_cls = type(source.dist)
            if dist_cls in self._scipy_map:
                pp_cls, extractor = self._scipy_map[dist_cls]
                params = extractor(source)
                params.setdefault("name", dist_cls.__name__)
                pp_dist = pp_cls(**params)
                pp_dist.with_source(Provenance("convert_from_scipy", parents=()))
                if isinstance(pp_dist, target_type):
                    return pp_dist
                from ._registry import converter_registry
                return converter_registry.convert(pp_dist, target_type, key=key, **kwargs)

            # Unknown scipy: sample -> NumericEmpiricalDistribution
            n = kwargs.pop("num_samples", 1024)
            samples = jnp.asarray(source.rvs(size=n))
            emp = NumericEmpiricalDistribution(samples)
            emp.with_source(Provenance("convert_from_scipy", parents=()))
            if issubclass(target_type, NumericEmpiricalDistribution):
                return emp
            from ._registry import converter_registry
            return converter_registry.convert(emp, target_type, key=key, **kwargs)

        # Case 2: ProbPipe -> scipy
        if isinstance(source, NumericRecordDistribution):
            src_name = type(source).__name__
            fn = self._pp_map.get(src_name)
            if fn is not None:
                return fn(source)
            raise TypeError(f"Cannot convert {src_name} to scipy distribution")

        raise TypeError(f"ScipyConverter cannot handle {type(source).__name__}")

    @property
    def priority(self) -> int:
        return 25
