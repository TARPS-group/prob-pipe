# Multiple Dispatch in ProbPipe: Design Exploration

## Context

ProbPipe currently does **single dispatch** via:
- **Protocol-based ops** in `core/ops.py` — `isinstance(dist, SupportsLogProb)` → `dist._log_prob(x)`
- **MethodRegistry** in `core/_registry.py` — priority-ordered list of methods, each with `supported_types()` (one type pre-filter), `check()`, and `execute()`. Used for `condition_on`.
- **ConverterRegistry** in `converters/_registry.py` — same pattern for distribution type conversion.

The motivating question: **how do we implement KL divergence**, and what does that reveal about supporting operations that need to dispatch on the types of *two* arguments?

---

## The Problem: KL Divergence Needs Binary Dispatch

KL divergence `KL(p ‖ q) = E_p[log p(x) − log q(x)]` has multiple implementations with different tradeoffs:

| Situation | Best strategy |
|---|---|
| Both `p` and `q` are TFP distributions with a registered pair | **Closed-form** via `tfd.kl_divergence(p, q)` — exact, fast |
| `p` is an `EmpiricalDistribution` (samples already exist) | **Reuse stored samples** — no sampling cost |
| General case (`p` supports sampling + log_prob, `q` supports log_prob) | **Monte Carlo estimate** — universal fallback |

This is a genuine binary dispatch problem: the right strategy depends on the *joint* type of `(p, q)`.

---

## Approach 1: If/else inside the function (no dispatch framework)

The simplest approach — embed all the type logic directly in the op.

```python
# core/ops.py

@workflow_function
def kl_divergence(
    p: Distribution,
    q: Distribution,
    *,
    random_seed: int | None = None,
    n_samples: int = 10_000,
) -> Array:
    from ..distributions._tfp_base import TFPDistribution
    from ..distributions._empirical import EmpiricalDistribution

    # Case 1: closed-form via TFP
    if isinstance(p, TFPDistribution) and isinstance(q, TFPDistribution):
        import tensorflow_probability.substrates.jax.distributions as tfd
        try:
            return tfd.kl_divergence(p._tfp_dist, q._tfp_dist)
        except NotImplementedError:
            pass  # fall through to MC

    # Case 2: p already has samples stored
    if isinstance(p, EmpiricalDistribution):
        xs = p.samples
        return jnp.mean(p._log_prob(xs) - q._log_prob(xs))

    # Case 3: general MC fallback
    key = jax.random.PRNGKey(random_seed) if random_seed is not None else _auto_key()
    xs = p._sample(key, sample_shape=(n_samples,))
    return jnp.mean(p._log_prob(xs) - q._log_prob(xs))
```

**Pros:** Zero infrastructure. Easy to read and debug.

**Cons:** Every new distribution type pair requires editing `core/ops.py`. External code can't extend it. The `core/` → `distributions/` dependency is hard-coded (uses lazy imports but still tightly coupled). Logic sprawls as more cases are added.

---

## Approach 2: Protocol on `p` (`_kl_divergence(self, q)` method)

Add a `SupportsKLDivergence` protocol. Distributions that can compute their own KL implement it; others fall back to MC.

```python
# core/protocols.py

class SupportsKLDivergence(Protocol):
    @runtime_checkable
    def _kl_divergence(self, q: Any) -> Array: ...
```

```python
# distributions/_tfp_base.py

class TFPDistribution(ArrayDistribution):
    def _kl_divergence(self, q: Any) -> Array:
        import tensorflow_probability.substrates.jax.distributions as tfd
        if isinstance(q, TFPDistribution):
            try:
                return tfd.kl_divergence(self._tfp_dist, q._tfp_dist)
            except NotImplementedError:
                pass
        raise NotImplementedError
```

```python
# core/ops.py

@workflow_function
def kl_divergence(
    p: Distribution,
    q: Distribution,
    *,
    random_seed: int | None = None,
    n_samples: int = 10_000,
) -> Array:
    if isinstance(p, SupportsKLDivergence):
        try:
            return p._kl_divergence(q)
        except NotImplementedError:
            pass
    # MC fallback
    key = jax.random.PRNGKey(random_seed) if random_seed is not None else _auto_key()
    xs = p._sample(key, sample_shape=(n_samples,))
    return jnp.mean(p._log_prob(xs) - q._log_prob(xs))
```

**Pros:** Follows the existing protocol pattern exactly. Closed-form logic lives in the distribution class that owns it. No new infrastructure.

**Cons:** The dispatch on `q`'s type is hidden ad hoc inside `p._kl_divergence(q)`, not by the dispatch system. Third parties who own neither `p` nor `q` can't register a new `(MyDist, TheirDist)` pair without monkey-patching. Doesn't generalize — each binary op needs its own protocol and ad hoc `q`-dispatch.

---

## Approach 3: `multipledispatch` package

The [`multipledispatch`](https://github.com/mrocklin/multipledispatch) library provides true multiple dispatch via decorator syntax.

```python
# pip install multipledispatch

from multipledispatch import Dispatcher

_kl_dispatch = Dispatcher("kl_divergence")

@_kl_dispatch.register(TFPDistribution, TFPDistribution)
def _(p: TFPDistribution, q: TFPDistribution) -> Array:
    import tensorflow_probability.substrates.jax.distributions as tfd
    return tfd.kl_divergence(p._tfp_dist, q._tfp_dist)

@_kl_dispatch.register(EmpiricalDistribution, object)
def _(p: EmpiricalDistribution, q: Any) -> Array:
    xs = p.samples
    return jnp.mean(p._log_prob(xs) - q._log_prob(xs))

@_kl_dispatch.register(object, object)
def _(p: Any, q: Any, n_samples: int = 10_000) -> Array:
    key = _auto_key()
    xs = p._sample(key, sample_shape=(n_samples,))
    return jnp.mean(p._log_prob(xs) - q._log_prob(xs))

# Wrap for WorkflowFunction broadcasting/orchestration
@workflow_function
def kl_divergence(p: Distribution, q: Distribution, **kwargs) -> Array:
    return _kl_dispatch(p, q, **kwargs)
```

**Pros:** Very concise registration syntax. MRO-aware — most specific type pair wins automatically. External code adds overloads via `_kl_dispatch.register(MyType, OtherType)(fn)`.

**Cons:**
- `multipledispatch` has not been maintained since 2019 and has known Python 3.12 issues.
- No `method="kl_forward"` named override.
- No `check()` feasibility probe — failures only surface at call time.
- Silent registration-ordering problem: TFP overloads must be registered before the first call; if `distributions/` hasn't been imported yet, the user silently gets only the MC fallback.
- Introduces an external dependency against the project's current convention.

---

## Approach 4: `plum-dispatch` package

[`plum`](https://github.com/beartype/plum) is a modern, actively maintained alternative using standard Python type annotations.

```python
# pip install plum-dispatch

from plum import dispatch

@dispatch
def kl_divergence(p: TFPDistribution, q: TFPDistribution) -> Array:
    import tensorflow_probability.substrates.jax.distributions as tfd
    return tfd.kl_divergence(p._tfp_dist, q._tfp_dist)

@dispatch
def kl_divergence(p: EmpiricalDistribution, q: Distribution) -> Array:
    xs = p.samples
    return jnp.mean(p._log_prob(xs) - q._log_prob(xs))

@dispatch
def kl_divergence(p: Distribution, q: Distribution) -> Array:
    key = _auto_key()
    xs = p._sample(key, sample_shape=(10_000,))
    return jnp.mean(p._log_prob(xs) - q._log_prob(xs))

# Wrap for WorkflowFunction broadcasting/orchestration
kl_divergence = workflow_function(kl_divergence)
```

**Pros:** Clean annotation-based syntax. Actively maintained. Raises `NotFoundLookupError` on missing pairs (clear failure mode). Union types and generics supported.

**Cons:**
- Same silent registration-ordering issue as `multipledispatch`.
- No `method=` named override or `check()` probe without reimplementing them on top.
- `plum` uses `beartype` for runtime type checking, adding a small per-call overhead.
- Adds a new non-trivial dependency.

---

## Approach 5: New `DiscrepancyRegistry` following the existing `MethodRegistry` pattern (recommended)

Create a new `DiscrepancyRegistry` in `probpipe/core/` that mirrors `MethodRegistry` and `ConverterRegistry`: a priority-ordered list of named measures, each with a `check()` feasibility probe and `compute()` implementation. Type-specific methods (e.g., the TFP closed-form) are defined in `distributions/` and registered at import time — exactly how inference methods are registered today.

### The registry

```python
# core/_discrepancy.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..custom_types import Array
from .protocols import SupportsLogProb, SupportsSampling

__all__ = ["Discrepancy", "DiscrepancyResult", "DiscrepancyRegistry", "discrepancy_registry"]


@dataclass(frozen=True)
class DiscrepancyResult:
    value: Array
    method_name: str = ""
    description: str = ""

    def within(self, eps: float) -> bool:
        return float(self.value) < eps


class Discrepancy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def priority(self) -> int:
        return 0

    @abstractmethod
    def check(self, p: Any, q: Any) -> bool:
        """Cheap feasibility probe — no computation."""

    @abstractmethod
    def compute(self, p: Any, q: Any, **kwargs: Any) -> DiscrepancyResult:
        """Compute the discrepancy."""


class DiscrepancyRegistry:
    def __init__(self) -> None:
        self._measures: list[Discrepancy] = []
        self._name_index: dict[str, Discrepancy] = {}

    def register(self, measure: Discrepancy) -> None:
        if measure.name in self._name_index:
            raise ValueError(f"Discrepancy {measure.name!r} is already registered")
        self._measures.append(measure)
        self._name_index[measure.name] = measure
        self._measures.sort(key=lambda m: m.priority, reverse=True)

    def get(self, name: str) -> Discrepancy:
        try:
            return self._name_index[name]
        except KeyError:
            available = ", ".join(sorted(self._name_index)) or "(none)"
            raise KeyError(f"No discrepancy named {name!r}. Available: {available}") from None

    def list_measures(self) -> list[str]:
        return [m.name for m in self._measures]

    def compute(
        self, p: Any, q: Any, *, method: str | None = None, **kwargs: Any
    ) -> DiscrepancyResult:
        if method is not None:
            m = self.get(method)
            if not m.check(p, q):
                raise TypeError(
                    f"Discrepancy {method!r} is not applicable to "
                    f"{type(p).__name__} and {type(q).__name__}"
                )
            return m.compute(p, q, **kwargs)

        for m in self._measures:
            if m.check(p, q):
                return m.compute(p, q, **kwargs)

        raise TypeError(
            f"No discrepancy measure applicable to "
            f"{type(p).__name__} and {type(q).__name__}. "
            f"Available: {self.list_measures()}"
        )


discrepancy_registry = DiscrepancyRegistry()
```

### MC fallback — lives in `core/`, no TFP dependency

```python
# core/_discrepancy.py (continued)

class KLDivergence(Discrepancy):
    """Monte Carlo KL(p ‖ q) = E_p[log p(x) − log q(x)]."""

    def __init__(self, n_samples: int = 10_000) -> None:
        self._n_samples = n_samples

    @property
    def name(self) -> str:
        return "kl_forward"

    @property
    def priority(self) -> int:
        return 50

    def check(self, p: Any, q: Any) -> bool:
        return (
            isinstance(p, SupportsSampling)
            and isinstance(p, SupportsLogProb)
            and isinstance(q, SupportsLogProb)
        )

    def compute(self, p: Any, q: Any, *, random_seed: int | None = None,
                n_samples: int | None = None) -> DiscrepancyResult:
        import jax
        import jax.numpy as jnp
        from .._utils import _auto_key
        key = jax.random.PRNGKey(random_seed) if random_seed is not None else _auto_key()
        n = n_samples if n_samples is not None else self._n_samples
        xs = p._sample(key, sample_shape=(n,))
        value = jnp.mean(p._log_prob(xs) - q._log_prob(xs))
        return DiscrepancyResult(value=value, method_name=self.name,
                                 description=f"MC KL(p ‖ q) with n={n}")


discrepancy_registry.register(KLDivergence())
```

### Closed-form TFP method — lives in `distributions/`, registered at import time

```python
# distributions/_kl_tfp.py

from __future__ import annotations
from typing import Any
from ..core._discrepancy import Discrepancy, DiscrepancyResult, discrepancy_registry

__all__: list[str] = []


class TFPKLDivergence(Discrepancy):
    """Closed-form KL via TFP's internal registry (when the pair is registered)."""

    @property
    def name(self) -> str:
        return "kl_tfp"

    @property
    def priority(self) -> int:
        return 100  # tried before the MC fallback

    def check(self, p: Any, q: Any) -> bool:
        from ._tfp_base import TFPDistribution
        from tensorflow_probability.substrates.jax.distributions import kullback_leibler
        if not (isinstance(p, TFPDistribution) and isinstance(q, TFPDistribution)):
            return False
        # probe TFP's own registry without running the computation
        return kullback_leibler._registered_kl(
            type(p._tfp_dist), type(q._tfp_dist)
        ) is not None

    def compute(self, p: Any, q: Any, **_: Any) -> DiscrepancyResult:
        import tensorflow_probability.substrates.jax.distributions as tfd
        value = tfd.kl_divergence(p._tfp_dist, q._tfp_dist)
        return DiscrepancyResult(value=value, method_name=self.name,
                                 description="Closed-form TFP KL(p ‖ q)")


discrepancy_registry.register(TFPKLDivergence())
```

This file is imported in `distributions/__init__.py` (same pattern as how inference methods are registered), so the TFP method is in place by the time any user code runs.

### The public op

```python
# core/ops.py

@workflow_function
def kl_divergence(
    p: Distribution,
    q: Distribution,
    *,
    method: str | None = None,
    n_samples: int | None = None,
    random_seed: int | None = None,
) -> Array:
    """KL(p ‖ q) — closed-form when available, MC otherwise.

    Pass ``method='kl_forward'`` to force MC, ``method='kl_tfp'`` to
    require closed-form (raises if the pair has no registered formula).
    """
    from ._discrepancy import discrepancy_registry
    result = discrepancy_registry.compute(
        p, q, method=method, n_samples=n_samples, random_seed=random_seed
    )
    return result.value
```

### Usage

```python
from probpipe import Normal, kl_divergence

p = Normal(loc=0.0, scale=1.0)
q = Normal(loc=0.5, scale=1.0)

kl_divergence(p, q)                             # auto-selects kl_tfp (exact)
kl_divergence(p, q, method="kl_forward")        # force MC
kl_divergence(p, q, n_samples=50_000, random_seed=42)
```

### Priority ordering

| Priority | Name | Condition |
|---|---|---|
| 100 | `kl_tfp` | Both are `TFPDistribution` and TFP has a closed-form for the pair |
| 50 | `kl_forward` | `p` is `SupportsSampling ∩ SupportsLogProb`, `q` is `SupportsLogProb` |

**Pros:**
- Zero new dependencies.
- Fully pluggable — third parties call `discrepancy_registry.register(MyMethod())`.
- `method=` override and `check()` probe work identically to `inference_method_registry`.
- Import acyclicity preserved — `core/_discrepancy.py` never imports `distributions/`.
- `@workflow_function` wraps naturally — both `p` and `q` are `Distribution`-annotated, so no MC broadcasting is triggered by the op itself.
- `set_priorities()` can be added to reorder methods at runtime (same as `MethodRegistry`).

**Cons:**
- More boilerplate per new pair than `plum` or `multipledispatch`.
- No MRO-based implicit specificity — the developer sets explicit priorities.

---

## Comparison Summary

| | If/else | Protocol | `multipledispatch` | `plum` | DiscrepancyRegistry |
|---|---|---|---|---|---|
| Lines of new infra | 0 | ~20 | 0 (dep) | 0 (dep) | ~100 |
| New dependency | No | No | Yes (stale) | Yes | No |
| External extensibility | No | No | Yes (fragile) | Yes (fragile) | Yes (clean) |
| `method=` named override | No | No | No | No | Yes |
| `check()` feasibility probe | No | No | No | No | Yes |
| Fits existing pattern | Partially | Yes | No | No | Yes (exact match) |
| Import acyclicity | Fragile | Clean | Fragile | Fragile | Clean |

---

## Recommendation

**Approach 5 — a new `DiscrepancyRegistry` modeled on `MethodRegistry`.**

It fits ProbPipe's existing conventions exactly, adds no new dependencies, preserves the acyclic import structure, and is fully pluggable. Any future binary op (`mutual_information`, `wasserstein_distance`, a bilinear form `x^T A y`) can follow the same pattern: a new `XxxRegistry` with an `XxxMethod` ABC in `core/`, with type-specific implementations registered from `distributions/` or downstream packages.

---

## Relevant Files

| File | Role |
|---|---|
| `probpipe/core/_registry.py` | Existing `MethodRegistry` — the template to follow |
| `probpipe/core/ops.py` | Where `kl_divergence` op would live |
| `probpipe/core/protocols.py` | `SupportsSampling`, `SupportsLogProb` used in `check()` |
| `probpipe/distributions/_tfp_base.py` | Where `TFPKLDivergence` would be defined |
| `probpipe/inference/_registry.py` | Model for how methods are registered at import time |
