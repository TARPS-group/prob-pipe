# ProbPipe Style Guide

This document defines the coding conventions for the ProbPipe project.
It is intended for contributors and AI assistants working on the codebase.

> **Spelling:** Always write **ProbPipe** — not "probpipe", "prob-pipe",
> "Probpipe", or "PROBPIPE". The Python package name `probpipe` (lowercase,
> no hyphen) appears only in import statements and file paths.

---

## 1. Naming Conventions

### 1.1 Protocols

Protocol classes are named `Supports<Capability>` in CamelCase:

```python
SupportsSampling, SupportsLogProb, SupportsMean, SupportsConditioning
```

Protocol methods use a **single leading underscore** to distinguish the
primitive implementation from the public op:

```python
_sample, _log_prob, _mean, _variance, _cov, _condition_on, _expectation
```

### 1.2 Ops (public API)

Ops are the public entry points in `probpipe.core.ops`. They are
`snake_case` with **no** leading underscore:

```python
sample, log_prob, mean, variance, cov, condition_on, expectation
```

Users call ops, never the underscore methods directly:

```python
# correct
m = mean(dist)

# incorrect
m = dist._mean()
```

### 1.3 Implementation functions

Each op has a private implementation function named `_<op>_impl`:

```python
_sample_impl, _log_prob_impl, _mean_impl, _condition_on_impl
```

These functions contain the protocol-check-and-delegate logic and are
registered in `_OP_REGISTRY`. They are wrapped by `WorkflowFunction`
for broadcasting/orchestration and by `_make_op` for the positional-arg
public API.

### 1.4 Standalone workflow functions

Workflow functions that are **not** ops (i.e., not universal operations
on distributions) follow the pattern:

- Implementation function: `_<name>_impl`
- Public `WorkflowFunction` instance: `<name>`

```python
# probpipe/inference/_nutpie.py
def _condition_on_nutpie_impl(model, data=None, *, ...):
    ...

condition_on_nutpie = WorkflowFunction(
    func=_condition_on_nutpie_impl, name="condition_on_nutpie"
)
```

### 1.5 Classes

- **Distribution classes:** CamelCase, descriptive — `Normal`, `MultivariateNormal`,
  `EmpiricalDistribution`, `BootstrapReplicateDistribution`.
- **Base / mixin classes:** `Distribution`, `ArrayDistribution`,
  `TFPDistribution`, `ProbabilisticModel`.
- **Private helper classes:** Leading underscore — `_LinearMapGRF`, `_ShiftedGRF`.

### 1.6 Modules

- **Private implementation modules:** Leading underscore — `_simple.py`,
  `_stan.py`, `_rwmh.py`, `_nutpie.py`.
- **Public modules:** Descriptive names — `continuous.py`, `discrete.py`,
  `protocols.py`, `ops.py`.
- **Package `__init__.py`** files re-export the public API. Users should
  import from `probpipe` or from subpackage `__init__` modules, not from
  private modules.

### 1.7 Reserved parameter names

`WorkflowFunction` reserves the parameter names `seed`, `n_broadcast_samples`,
and `include_inputs` internally. Use `random_seed` instead when defining
functions that accept a PRNG seed.

### 1.8 The `.n` property convention

Finite-sample distribution classes expose an `.n` property (read-only
`int`) giving the number of stored samples, components, or observations.
The meaning depends on the class but always answers "how many items does
this distribution hold?"

| Class | `.n` meaning |
|-------|-------------|
| `EmpiricalDistribution` | Number of samples |
| `JointEmpiricalDistribution` | Number of joint samples |
| `BootstrapDistribution` | Number of function evaluations |
| `BootstrapReplicateDistribution` | Number of observations per bootstrap dataset |
| `BroadcastDistribution` | Number of input–output pairs |
| `_ArrayMarginal` | Number of output samples |
| `_MixtureMarginal` | Number of mixture components |
| `_ListMarginal` | Number of output items |

When adding a new class that wraps a finite collection of samples or
components, define `.n` as a `@property` returning `int`.

---

## 2. Module Granularity

### 2.1 General rule

Each file should contain **one independent concern**. Use judgment:

- **Thin wrappers** that share a common base and pattern (e.g.,
  TFP-backed distributions) belong together in a single file grouped
  by mathematical category (`continuous.py`, `discrete.py`,
  `multivariate.py`).
- **Substantial classes** with distinct logic or backends (e.g.,
  `SimpleModel` vs `StanModel`, `RWMH` vs nutpie) get their own file.
- **Small utility types** tightly coupled to one consumer belong in
  that consumer's file (e.g., `InferenceDiagnostics` in
  `_diagnostics.py`).

The test: *if two classes are always modified together or one only exists
to serve the other, they belong in the same file. If they can evolve
independently, they get separate files.*

### 2.2 Size threshold

Split a module when it exceeds roughly **1000 lines**. When splitting
a distribution module, group by support domain (e.g., real-line vs
positive vs bounded).

### 2.3 Private vs public modules

Within subpackages that contain multiple implementation files
(`modeling/`, `inference/`, `converters/`), implementation modules
use a leading underscore (`_simple.py`, `_rwmh.py`). The package
`__init__.py` re-exports the public API so users never import from
underscore modules directly.

Within `core/` and `distributions/`, modules are public (no underscore)
because they form the foundational layer and their names are part of the
internal vocabulary (e.g., `from probpipe.core.protocols import ...`).

---

## 3. Docstring Conventions

Use **NumPy-style** docstrings with `Parameters`, `Returns`, and `Raises`
sections as needed.

### 3.1 Module docstrings

Every module has a docstring explaining its purpose. Include a usage
example when helpful:

```python
"""Built-in operations for distribution computation.

Each public function (``sample``, ``mean``, ``log_prob``, ...) is a
lightweight positional-arg wrapper around an internal
:class:`~probpipe.core.node.WorkflowFunction`.

Usage::

    from probpipe import sample, mean, log_prob
    m = mean(dist)
"""
```

### 3.2 Class docstrings

Summary line, then `Parameters` section:

```python
class Normal(TFPDistribution):
    """Univariate normal (Gaussian) distribution.

    Parameters
    ----------
    loc : array-like
        Mean of the distribution.
    scale : array-like
        Standard deviation (> 0).
    name : str, optional
        Distribution name for provenance.
    """
```

### 3.3 Function/method docstrings

Summary line, then `Parameters`, `Returns`, `Raises` as needed:

```python
def _sample_impl(
    dist: SupportsSampling,
    *,
    key: PRNGKey | None = None,
    sample_shape: tuple[int, ...] = (),
) -> Any:
    """Draw samples from a distribution.

    Parameters
    ----------
    dist : SupportsSampling
        Distribution to sample from.
    key : PRNGKey or None
        JAX PRNG key. Auto-generated if not provided.
    sample_shape : tuple of int
        Shape prefix for independent draws.

    Returns
    -------
    Any
        Sampled value(s).
    """
```

### 3.4 Section separators

Use `# --` comment blocks to separate logical sections within a class
or module:

```python
# -- Distribution interface ------------------------------------------------

# -- Conditioning ----------------------------------------------------------
```

---

## 4. Import Conventions

### 4.1 Future annotations

Every module starts with:

```python
from __future__ import annotations
```

This enables PEP 604 union syntax (`X | Y`) and forward references in
all Python versions ProbPipe supports.

### 4.2 Import order

1. `from __future__ import annotations`
2. Standard library (`functools`, `logging`, `dataclasses`, `typing`, etc.)
3. Third-party (`jax`, `jax.numpy`, `tensorflow_probability`, etc.)
4. Internal (relative imports within `probpipe`)

Separate each group with a blank line.

### 4.3 Relative imports

Always use **relative imports** for internal references:

```python
from ..core.distribution import Distribution, Provenance
from ..core.protocols import SupportsSampling
from ..custom_types import Array, PRNGKey
```

### 4.4 Optional dependencies

Use try/except with a helpful error at the call site:

```python
def _ensure_nutpie():
    try:
        import nutpie
        return nutpie
    except ImportError:
        raise ImportError(
            "nutpie is required: pip install probpipe[nutpie]"
        ) from None
```

For test files, use `pytest.importorskip("nutpie")` or mock-based
approaches with `patch.dict(sys.modules)`.

### 4.5 TYPE_CHECKING guards

Use `TYPE_CHECKING` for imports needed only by type checkers:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._registry import Converter
```

---

## 5. Type Annotation Conventions

Use modern Python 3.12+ syntax everywhere:

| Use this             | Not this                |
|----------------------|-------------------------|
| `list[str]`          | `List[str]`             |
| `tuple[int, ...]`    | `Tuple[int, ...]`       |
| `dict[str, Any]`     | `Dict[str, Any]`        |
| `str \| None`        | `Optional[str]`         |
| `X \| Y`             | `Union[X, Y]`           |

Type aliases live in `probpipe/custom_types.py`:

```python
Array: TypeAlias = jnp.ndarray
ArrayLike: TypeAlias = jnp.ndarray | list | tuple | float | int
PRNGKey: TypeAlias = jax.Array
```

---

## 6. Subpackage Dependencies

The dependency graph must remain **acyclic**. Allowed import directions:

```
custom_types  (no internal deps — leaf)
     ↑
   core/      (imports custom_types only)
     ↑
distributions/ (imports core/, custom_types)
     ↑
linalg/       (imports core/, custom_types; no distribution imports)
     ↑
converters/   (imports core/, distributions/, custom_types)
modeling/     (imports core/, inference/, custom_types)
inference/    (imports core/, custom_types)
```

### Rules

1. **`core/`** must never import from `distributions/`, `modeling/`,
   `inference/`, `converters/`, or `linalg/`.
2. **`distributions/`** must never import from `modeling/`, `inference/`,
   or `converters/`.
3. **`inference/`** must never import from `modeling/` or `converters/`.
4. **`modeling/`** may import from `inference/` (for MCMC result types)
   but must never import from `converters/`.
5. **`converters/`** may import from `distributions/` and `core/` but
   must never import from `modeling/` or `inference/`.
6. **`linalg/`** is self-contained; it may import from `core/` and
   `custom_types` only.

> **Exception:** `core/node.py` imports from `distributions/joint.py` for
> `WorkflowFunction` broadcasting. This is the single allowed reverse edge
> and should not be extended.

---

## 7. Protocol Design

### 7.1 Defining protocols

All protocols are `@runtime_checkable` and inherit from `Protocol`:

```python
@runtime_checkable
class SupportsFoo(Protocol):
    """Distribution that supports foo."""

    def _foo(self, ...) -> Any: ...
```

### 7.2 Protocol hierarchy

- `SupportsLogProb` extends `SupportsUnnormalizedLogProb`.
- All other protocols (`SupportsSampling`, `SupportsMean`,
  `SupportsVariance`, `SupportsCovariance`, `SupportsExpectation`,
  `SupportsConditioning`, `SupportsNamedComponents`) are standalone.

### 7.3 Implementing protocols

Concrete classes inherit protocols and implement the underscore methods.
Protocol checks use `isinstance`, not `issubclass` (the latter does not
work with protocols that have non-method members like `ClassVar`).

```python
class Normal(TFPDistribution):
    # TFPDistribution provides _sample, _log_prob, _mean, etc.
    # by delegating to the internal tfd.Normal instance.
    ...
```

---

## 8. Testing Conventions

### 8.1 File naming

Test files mirror the source structure: `tests/test_<module>.py`.

### 8.2 Test classes

Group related tests in `Test*` classes. Use pytest fixtures for shared
setup:

```python
class TestSample:
    def test_sample_scalar(self, normal):
        s = sample(normal, key=jax.random.PRNGKey(0))
        assert s.shape == ()
```

### 8.3 Fixtures

Define reusable fixtures at module scope:

```python
@pytest.fixture
def normal():
    return Normal(loc=2.0, scale=0.5)
```

Use `@pytest.fixture(params=...)` for parametrized testing across
distribution families.

### 8.4 Optional dependencies

- `pytest.importorskip("pymc")` for tests requiring optional packages.
- `patch.dict(sys.modules, ...)` for mock-based isolation of optional
  backends.
- Stan/nutpie tests use `object.__new__()` + manual attribute setting to
  avoid requiring compiled models.

### 8.5 Test runner

Tests run in parallel via `pytest-xdist` (`-n auto --dist worksteal`).
Disable for debugging: `pytest -p no:xdist -o "addopts="`.

---

## 9. Miscellaneous

### 9.1 `__all__` exports

Every public module defines `__all__`. Package `__init__.py` files
aggregate exports from private submodules.

### 9.2 Immutability

Distribution objects are immutable. Parameters are fixed at construction;
operations return new distribution objects rather than mutating state.

### 9.3 Error messages

When a protocol check fails, raise `TypeError` with a message that names
the missing capability:

```python
if not isinstance(dist, SupportsMean):
    raise TypeError(
        f"{type(dist).__name__} does not support mean; "
        f"it must implement the SupportsMean protocol"
    )
```
