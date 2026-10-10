"""Small stand-in laws built on Distribution, each claiming the capabilities a test reads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from probpipe import NumericArraySpec, Record, RecordSpec
from probpipe.core._specs import OutputSpec
from probpipe.core.constraints import boolean, real
from probpipe.distributions._capabilities import (
    SupportsApproximateConditioning,
    SupportsCovariance,
    SupportsExactConditioning,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMarginals,
    SupportsMean,
    SupportsQuantile,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
)
from probpipe.distributions._conditional import ConditionalDistribution
from probpipe.distributions._distribution import Distribution, DistributionSpec
from probpipe.linalg import DenseLinOp

REAL = NumericArraySpec((), jnp.float32, real)


def normal_draws(key: Any, loc: float, scale: float, shape: tuple[int, ...]) -> Any:
    return (loc + scale * jax.random.normal(key, shape)).astype(jnp.float32)


class Gaussian(
    Distribution,
    SupportsSampling,
    SupportsLogProb,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
    SupportsQuantile,
):
    """A scalar normal law with every closed form the moment tests read."""

    def __init__(self, label: str, loc: float = 0.0, scale: float = 1.0) -> None:
        super().__init__(label, OutputSpec(**{label: REAL}))
        self.loc, self.scale = float(loc), float(scale)

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return normal_draws(key, self.loc, self.scale, tuple(sample_shape))

    def _log_prob(self, value: Any) -> Any:
        return jax.scipy.stats.norm.logpdf(jnp.asarray(value), self.loc, self.scale)

    def _mean(self) -> Any:
        return jnp.float32(self.loc)

    def _variance(self) -> Any:
        return jnp.float32(self.scale**2)

    def _cov(self) -> Any:
        return DenseLinOp(jnp.array([[self.scale**2]], jnp.float32))

    def _quantile(self, q: Any) -> Any:
        return (self.loc + self.scale * jax.scipy.special.ndtri(jnp.asarray(q))).astype(jnp.float32)


class GuardedMean(Gaussian):
    """A normal law whose closed-form mean holds only when its answer allows it."""

    def __init__(self, label: str, answer: Any) -> None:
        super().__init__(label, 3.0, 1.0)
        self.answer = answer

    def _mean_guard(self) -> Any:
        """The stand-in's answer admits the closed form."""
        return self.answer


class Sampler(Distribution, SupportsSampling):
    """A scalar normal law that only samples, recording each sample shape it is asked for."""

    def __init__(self, label: str, loc: float = 0.0, scale: float = 1.0) -> None:
        super().__init__(label, OutputSpec(**{label: REAL}))
        self.loc, self.scale = float(loc), float(scale)
        self.shapes: list[tuple[int, ...]] = []

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        self.shapes.append(tuple(sample_shape))
        return normal_draws(key, self.loc, self.scale, tuple(sample_shape))


class Vector(Distribution, SupportsSampling):
    """A law on R² with independent coordinates of scales 1 and 2, which only samples."""

    def __init__(self, label: str) -> None:
        super().__init__(label, OutputSpec(**{label: NumericArraySpec((2,), jnp.float32, real)}))

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        draws = jax.random.normal(key, (*sample_shape, 2))
        return (draws * jnp.array([1.0, 2.0])).astype(jnp.float32)


class Pair(Distribution, SupportsSampling, SupportsMean):
    """A law drawing an exposed record of a scalar ``a`` and a vector ``b``."""

    def __init__(self, label: str) -> None:
        super().__init__(label, RecordSpec(a=REAL, b=NumericArraySpec((2,), jnp.float32, real)))

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        ka, kb = jax.random.split(key)
        shape = tuple(sample_shape)
        return Record(
            self.label,
            {
                "a": normal_draws(ka, 1.0, 1.0, shape),
                "b": normal_draws(kb, -1.0, 1.0, (*shape, 2)),
            },
        )

    def _mean(self) -> Any:
        return Record(self.label, {"a": jnp.float32(1.0), "b": -jnp.ones(2, jnp.float32)})


class OneField(Distribution, SupportsSampling, SupportsLogProb):
    """A law drawing a record with the one field ``x``, which is not an array-valued law."""

    def __init__(self, label: str) -> None:
        super().__init__(label, RecordSpec(x=REAL))

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return Record(self.label, {"x": normal_draws(key, 0.0, 1.0, tuple(sample_shape))})

    def _log_prob(self, value: Any) -> Any:
        return jax.scipy.stats.norm.logpdf(jnp.asarray(value["x"]))


class Coin(Distribution, SupportsSampling, SupportsMean, SupportsLogProb, SupportsExpectation):
    """A Bernoulli law on {0, 1}, drawn as int32, with an exact expectation over its atoms."""

    def __init__(self, label: str, p: float = 0.25) -> None:
        super().__init__(label, OutputSpec(**{label: NumericArraySpec((), jnp.int32, boolean)}))
        self.p = float(p)

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return jax.random.bernoulli(key, self.p, tuple(sample_shape)).astype(jnp.int32)

    def _mean(self) -> Any:
        return jnp.float32(self.p)

    def _log_prob(self, value: Any) -> Any:
        return jnp.where(jnp.asarray(value) == 1, jnp.log(self.p), jnp.log1p(-self.p))

    def _expectation(self, f: Any) -> Any:
        return jax.tree.map(
            lambda one, zero: self.p * one + (1.0 - self.p) * zero, f(jnp.int32(1)), f(jnp.int32(0))
        )


class Unnormalized(Distribution, SupportsUnnormalizedLogProb):
    """A law that knows its log-density only up to the constant ``log 2``."""

    def __init__(self, label: str) -> None:
        super().__init__(label, OutputSpec(**{label: REAL}))

    def _unnormalized_log_prob(self, value: Any) -> Any:
        return jax.scipy.stats.norm.logpdf(jnp.asarray(value)) + jnp.log(2.0)


class Polymorphic(Distribution, SupportsLogProb):
    """A standard normal law on Rⁿ whose length ``n`` is free."""

    def __init__(self, label: str) -> None:
        super().__init__(label, OutputSpec(**{label: NumericArraySpec(("n",), jnp.float32, real)}))

    def _log_prob(self, value: Any) -> Any:
        return jnp.sum(jax.scipy.stats.norm.logpdf(jnp.asarray(value)), axis=-1)


class CountedVector(Distribution, SupportsLogProb):
    """A standard normal law on R³ that records each value its density is called with."""

    def __init__(self, label: str) -> None:
        super().__init__(label, OutputSpec(**{label: NumericArraySpec((3,), jnp.float32, real)}))
        self.calls: list[Any] = []

    def _log_prob(self, value: Any) -> Any:
        self.calls.append(value)
        return jnp.sum(jax.scipy.stats.norm.logpdf(jnp.asarray(value)), axis=-1)


class Measure(Distribution, SupportsSampling):
    """A random measure whose draws are normal laws with standard-normal locations."""

    def __init__(self, label: str) -> None:
        super().__init__(label, OutputSpec(**{label: DistributionSpec(OutputSpec(x=REAL))}))

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        shape = tuple(sample_shape)
        if not shape:
            return Gaussian("x", float(jax.random.normal(key)))
        keys = jax.random.split(key, int(np.prod(shape)))
        laws = np.empty(len(keys), dtype=object)
        for index, subkey in enumerate(keys):
            laws[index] = Gaussian("x", float(jax.random.normal(subkey)))
        return laws.reshape(shape)


class RandomDensity(Distribution, SupportsRandomLogProb, SupportsRandomUnnormalizedLogProb):
    """A random measure whose random log-densities are stand-in laws."""

    def __init__(self, label: str) -> None:
        super().__init__(label, OutputSpec(**{label: DistributionSpec(OutputSpec(x=REAL))}))

    def _random_log_prob(self) -> Distribution:
        return Gaussian("log_density", -1.0)

    def _random_unnormalized_log_prob(self) -> Distribution:
        return Gaussian("unnormalized_log_density", -2.0)


class Marginalizing(Distribution, SupportsMarginals):
    """A record law over ``a`` and ``b`` whose marginal is exact only at ``a``."""

    def __init__(self, label: str) -> None:
        super().__init__(label, RecordSpec(a=REAL, b=REAL))
        self.paths: list[Any] = []

    def _marginal(self, path: Any) -> Distribution:
        self.paths.append(path)
        return Gaussian(path.rsplit("/", 1)[-1], 5.0)

    def _marginal_guard(self, path: Any) -> bool:
        """The marginal is exact at the field a."""
        return path == "a"


def _given_values(given: Any) -> dict[str, Any]:
    if isinstance(given, Record):
        return {name: given[name] for name in given.fields}
    if isinstance(given, Mapping):
        return dict(given)
    raise TypeError(f"a kernel's given is a Record or a mapping; got {type(given).__name__}")


class Kernel(ConditionalDistribution):
    """The kernel ``y | slots ~ Normal(offset + sum of the slots, 1)``, whose event is ``y``.

    The component is fixed at construction, so a curried kernel and the law it
    returns keep it whatever the kernel's label becomes.
    """

    def __init__(
        self,
        label: str = "y",
        slots: tuple[str, ...] = ("mu",),
        offset: float = 0.0,
        component: str | None = None,
    ) -> None:
        self.component = label if component is None else component
        super().__init__(
            label, {slot: REAL for slot in slots}, OutputSpec(**{self.component: REAL})
        )
        self.slots, self.offset = tuple(slots), float(offset)

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        values = _given_values(given)
        left = tuple(slot for slot in self.slots if slot not in values)
        offset = self.offset + sum(float(values[slot]) for slot in self.slots if slot in values)
        if left:
            return Kernel(self.label, left, offset, self.component)
        return Gaussian(self.component, offset, 1.0)


class ExactPosterior(Distribution, SupportsSampling, SupportsExactConditioning):
    """A law whose ``_condition_on`` returns the conditional law, recording each given."""

    def __init__(self, label: str) -> None:
        super().__init__(label, RecordSpec(theta=REAL, y=REAL))
        self.givens: list[Any] = []

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        kt, ky = jax.random.split(key)
        shape = tuple(sample_shape)
        return Record(
            self.label,
            {"theta": normal_draws(kt, 0.0, 1.0, shape), "y": normal_draws(ky, 0.0, 1.0, shape)},
        )

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        self.givens.append(given)
        return Gaussian("theta", 1.0, 0.5)


class Amortized(Distribution, SupportsApproximateConditioning):
    """A law whose ``_condition_on`` returns a stand-in for the conditional law."""

    def __init__(self, label: str) -> None:
        super().__init__(label, RecordSpec(theta=REAL, y=REAL))

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        return Gaussian("theta", 2.0, 1.0)


class Bare(Distribution):
    """A law that claims no capability."""

    def __init__(self, label: str) -> None:
        super().__init__(label, OutputSpec(**{label: REAL}))
