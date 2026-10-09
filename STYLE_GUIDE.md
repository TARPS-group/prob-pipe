# ProbPipe Style Guide

This document defines the coding and writing conventions for the ProbPipe project.
It is intended for contributors and AI assistants working on the codebase.

> **Spelling:** Always write **ProbPipe** — not "probpipe", "prob-pipe",
> "Probpipe", or "PROBPIPE". The lowercase forms `probpipe` and `probpipe-core`
> are reserved for the Python import package (in import statements and file
> paths) and the distribution names (in `pip install` commands).

---

## 1. Naming Conventions

### 1.1 Protocols

Protocol classes are named `Supports<Capability>` in CamelCase:

```python
SupportsSampling, SupportsLogProb, SupportsMean, SupportsExactConditioning,
SupportsArrayBackend
```

Protocol methods use a **single leading underscore** to distinguish the
primitive implementation from the public operation:

```python
_sample, _log_prob, _mean, _variance, _cov, _condition_on, _expectation,
_make_array_backend
```

Most `Supports*` protocols are instance-level capabilities (does
`isinstance(my_dist, SupportsSampling)` hold?). `SupportsArrayBackend`
is the exception: its method is a `@classmethod`, so the contract
lives at class scope. Always check
`isinstance(MyDistribution, SupportsArrayBackend)` (i.e. on the
class object). `isinstance(an_instance, SupportsArrayBackend)`
returns `True` too — instances inherit class attributes, and
`runtime_checkable` just looks for the named attribute — but the
result is misleading because the protocol's contract is the class
declaring `_make_array_backend`, not the instance.

`SupportsExactConditioning` and `SupportsApproximateConditioning` are the
other exception: they are abstract base classes, not protocols, so a class
claims one by inheriting it. Both declare the same `_condition_on`, and
whether that method returns the conditional law or a stand-in for it is a
claim about the result rather than a fact about the method, so a structural
check cannot tell them apart.

The corresponding *backend* interface that `_make_array_backend`
returns (`_DistributionArrayBackend`) is private to the library —
user code never imports or constructs it.

### 1.2 Operations (public API)

The operations are the public entry points, declared in `probpipe.operations`
and exported from `probpipe`. They are `snake_case` with **no** leading
underscore:

```python
sample, log_prob, mean, variance, cov, condition_on, expectation
```

Users call operations, never the underscore methods directly:

```python
# correct
m = mean(dist)

# incorrect
m = dist._mean()
```

### 1.3 Operation routes

Each operation is declared once, with the `@operation` decorator, in the module
of `probpipe/operations/` that matches its design section. The decorated
function's signature names the operands, and its body is its docstring. Each
implementation is a route registered on the operation, as
`probpipe/operations/_sample.py` registers the route that calls `_sample`:

```python
@operation(result=_sample_result)
def sample(d: Distribution, sample_shape: tuple[int, ...] = ()):
    """Draw from a distribution."""


sample.capability_route(
    "exact", operand="d", protocol=SupportsSampling, method="_sample", exact=True, execute=_draw
)
```

Design VI.0 owns the operation model, its route sources included.

### 1.4 Standalone Functions

A public function that is not an operation is a `Function`, defined with the
`@function` decorator in its implementation module:

```python
# probpipe/inference/_nutpie.py
@function
def condition_on_nutpie(model, data=None, *, num_results=1000, num_warmup=500, ...):
    ...
```

Treat a public `Function` as an immutable, first-class `TrackedTerm` / `Annotated`
computation term. Its construction-time `inspect.Signature` is the Python
calling contract; optional `input_spec: InputSpec` and `output_spec: OutputSpec` declarations
are authoritative schemas but never derive or replace that signature. Use
`apply(*args, **kwargs)` for one raw evaluation with binding and schema checks.
Use `__call__` for distribution lifting, array sweeps, orchestration, result
wrapping, and Function-first provenance. Results use `output_label`, which
defaults to the function's label at construction and survives `with_label`. A
raw implementation's own label survives `apply`, while a normal call labels its
independent result.

If an implementation returns an existing `Record`, `RecordBatch`, or
`Distribution`, `apply` preserves its identity. `__call__` instead creates a
shallow result copy that shares value data and specs, owns a separate
annotations container, and receives only the current call's provenance. Do not
restore identity-through behavior at the workflow boundary. Variadic Functions
without an input declaration are supported: the planner treats every `*args`
element and `**kwargs` entry as an independent slot while reconstructing the
original `BoundArguments` before invoking user code.

Exception: the learned-likelihood factories `learn_amortized_likelihood` and
`learn_amortized_ratio`, and the samplers `rwmh` and `elliptical_slice`, are
plain functions. A plain public function states its reason in a comment at its
definition site.

### 1.5 Classes

- **Distribution classes:** CamelCase, descriptive — `Normal`, `MultivariateNormal`,
  `EmpiricalDistribution`, `BootstrapReplicateDistribution`.
- **Base / mixin classes:** `Distribution`, `NumericDistribution`,
  `TFPDistribution`, `RandomFunction`.
- **View classes:** end in "View" — `FieldView` (the field view `d[path]`),
  `DiagnosticsView` (the accessor `Distribution.diagnostics`).
- **Private helper classes:** Leading underscore — `_LinearMapGRF`, `_ShiftedGRF`.

### 1.6 Modules

- **Private implementation modules:** Leading underscore — `_continuous.py`,
  `_programs.py`, `_blackjax_rwmh.py`, `_nutpie.py`.
- **Public modules:** Descriptive names — `constraints.py`, `named_tree.py`,
  `protocols.py`, `record.py`.
- **Package `__init__.py`** files re-export the public API. Users should
  import from `probpipe` or from subpackage `__init__` modules, not from
  private modules.

### 1.7 Registry method naming

Inference methods registered with the ``UnaryDispatchRegistry`` follow
``{backend}_{algorithm}`` naming in ``snake_case``:

```
blackjax_nuts, blackjax_hmc, blackjax_rwmh, blackjax_elliptical_slice,
nutpie_nuts, cmdstan_nuts, pymc_nuts, pymc_advi
```

Method classes are CamelCase: ``TFPNutsMethod``, ``CmdStanNutsMethod``,
``PyMCNutsMethod``. Factory functions that return parameterized instances
(e.g., ``TFPNutsMethod() -> _TFPGradientMethod``) use the same naming.

### 1.8 Workflow option namespace

`Function` keeps ProbPipe controls separate from wrapped-function
kwargs. Use `@function(...)` for definition-time controls
such as `dispatch` and `n_broadcast_samples`, and use
`workflow.with_options(...)` for a copy with revised controls, such as
`n_broadcast_samples` and `include_inputs`, which every call of the copy reads.

Ordinary workflow calls should treat keyword arguments as user-function
inputs. Wrapped functions may use names such as `seed`,
`n_broadcast_samples`, and `include_inputs` when those names are part of
their own domain API.

`seed` is not a `Function` or `with_options` control. Put reproducible lifted
executions inside `with workflow_run(seed=...):`; a wrapped function's own
parameter named `seed` remains an ordinary domain input.

### 1.9 The `num_atoms` / `replicate_size` property convention

Finite-sample distribution classes expose a read-only `int` property
naming the size of the finite collection they hold. The name reflects
*what is being counted*:

- **`num_atoms`** — items in an empirical *measure* (atoms / point
  masses in `\sum_i w_i \delta_{x_i}`). Use for any class whose
  ``_sample`` returns *one of N stored atoms*.
- **`replicate_size`** — items in a single bootstrap *replicate*. Use
  for any class whose ``_sample`` returns a *whole resampled dataset or
  measure* whose size is the named count. (`*_size` rather than `num_*`
  because a generative resampler holds no finite atom set — the count is a
  parameter of the resample, like `batch_size` / `event_size`.)

| Class | Property | Meaning |
|-------|----------|---------|
| `EmpiricalDistribution` | `num_atoms` | Stored atoms |
| `KDEDistribution` | `num_atoms` | Kernel centres |
| `BootstrapReplicateDistribution` | `replicate_size` | Draws in one replicate |
| `BootstrapDistribution` | `replicate_size` | Atoms of one drawn measure |

When adding a new class that wraps a finite collection, define the
property as a `@property` returning `int`. Pick the name based on
what each `_sample` call produces — a single atom (use `num_atoms`)
or a whole resampled replicate (use `replicate_size`).

### 1.10 Record field iteration and path access

The mapping protocol on `Record` and `RecordSpec` (`keys` / `values`
/ `items` / `__iter__` / `__len__` / `__contains__` / `__getitem__`) is
**leaf-keyed**: it enumerates every leaf by its full `/`-path, never
interior nodes. The
order is **canonical first-appearance order** — the order each field
was first introduced (keyword-argument or input-dict order at
construction). Code must not rely on an alphabetical or sorted order;
the constructor preserves what the caller wrote.

The `/` character is reserved as a nested-path separator. Field
names may not contain `/` (raises `ValueError` at construction).
Slash-delimited strings are accepted everywhere a field name is
accepted, as a short form of the tuple form:

```python
record["params/intercept"]   # same as record["params", "intercept"]
"params/intercept" in record # same as record["params", "intercept"] not raising
```

Because `[]` reaches only leaves, navigate to a leaf **or** an interior
subtree with `at_path` instead — `record.at_path("params")` returns the
sub-Record, whereas `record["params"]` raises when `params` is not a
leaf. `keys()` lists every leaf's path using the same `/` separator, so
those paths round-trip with `__getitem__`.

**Mappings are never leaves.** A `Mapping` value denotes tree
structure: a dict field value is always materialised into a nested
subtree, never stored as a single opaque leaf (construction recurses
into every `Mapping`, whether passed as a keyword value or nested inside
a positional mapping). There is no way to carry an opaque payload dict
inside a `Record`; use a non-mapping container if you need one leaf.

**Renaming fields.** `with_path_names(old=new, ...)` returns a
same-family tree with the given nodes (leaves or whole subtrees)
renamed. Each key is the exact path of a node, so a single name
addresses a top-level node and a nested node takes its full path. It
renames fields *within* the tree; relabeling the object itself is
`with_label`.

When adding new Record-based containers, follow these conventions:
preserve first-appearance order, reject `/` in field names, materialize
a mapping value into a nested subtree (never a leaf), key the mapping by
leaf path, and accept the slash-delimited form in any string-keyed lookup.

### 1.11 Distribution iteration

A `Distribution` represents a single random variable, not a
collection. Every concrete `Distribution` subclass —
`Normal`, `EmpiricalDistribution`, `BootstrapReplicateDistribution`,
factored joints — is **non-iterable**. An empirical law exposes its stored
atoms on `.atoms`, and the size property of §1.9 reports the count.
Parametric distributions have neither.

Iteration is reserved for the `Record` family — `Record` and
`NumericRecord` — which iterate field names dict-style
(`keys()` / `values()` / `items()`). A
`RecordBatch` is a collection, not a named tree: it iterates leading-axis
views like an array, and its fields are read from `event_template`.

`DistributionBatch` is positional and follows numpy/jax conventions:
`len(batch)` is the leading-axis size and `batch_size` is the total
count of laws; elements are accessed via `batch[i]`.
Its `event_spec` declares the term every law draws.
Iteration visits views of the laws along the leading axis.

A new `Distribution` subclass defines no `__iter__`.
The regression test in `tests/core/test_iteration_protocol.py` enforces
this rule across the user-constructible distribution classes, such as
`Normal`, `KDEDistribution`, and `MinibatchedDistribution`. A lifted call's
result is an
`EmpiricalDistribution`, which the test covers with the others.

### 1.12 Naming accuracy

Names describe what the object *is*, not its history or one of its
uses:

- **Semantic accuracy.** A class representing a joint model is not a
  `*Posterior`; a distribution is not a `*LogDensity`; a helper that
  imports and returns a module is not adequately described by
  `_ensure_*` alone. If documenting a class honestly requires saying
  what it is *not*, rename it.
- **Ecosystem alignment.** Where numpy/JAX have an established
  convention (`shape`, `__len__` over the leading axis, `size` as the
  total element count), follow it rather than inventing a parallel
  vocabulary.
- **Symmetry.** Paired APIs get symmetric names — e.g.,
  `NumericRecord.to_vector` / `NumericRecord.from_vector`.
- **Rename sweeps are complete.** A rename covers every analogous symbol,
  such as the `with_*` method of a renamed attribute. It also covers the test
  files named after the old symbol and the docs that mention it.

---

## 2. Module Granularity

### 2.1 General rule

Each file should contain **one independent concern**. Use judgment:

- **Thin wrappers** that share a common base and pattern (e.g.,
  TFP-backed distributions) belong together in a single file grouped
  by mathematical category (`families/_continuous.py`,
  `families/_discrete.py`, `families/_multivariate.py`).
- **Substantial classes** with distinct logic or backends get their own
  file, as the random-walk sampler in `inference/_blackjax_rwmh.py` and the
  nutpie sampler in `inference/_nutpie.py` do.
- **Small helpers** tightly coupled to one consumer belong in
  that consumer's file (e.g., `_function_draws` in
  `operations/_sample.py`).

The test: *if two classes are always modified together or one only exists
to serve the other, they belong in the same file. If they can evolve
independently, they get separate files.*

### 2.2 Size threshold

Split a module when it exceeds roughly **1000 lines**. When splitting
a distribution module, group by support domain (e.g., real-line vs
positive vs bounded).

### 2.3 Private vs public modules

Implementation modules use a leading underscore (`_continuous.py`,
`_blackjax_rwmh.py`, `_array_backend.py`); the package `__init__.py`
re-exports their public symbols so users never need to import from
underscore modules directly.

Modules **without** the underscore are reserved for the foundational
vocabulary that user code may reasonably import directly:

```python
from probpipe.core.protocols import SupportsArrayBackend
from probpipe.core.constraints import positive
from probpipe.core.record import Record
```

The public modules are these:

- `core/`: `config.py`, `constraints.py`, `named_tree.py`, `node.py`,
  `protocols.py`, `provenance.py`, `record.py`, `tracked.py`, and `transition.py`;
- `linalg/`: `linear_operator.py`, `operations.py`, and `utils.py`;
- `record/`: `design.py`;
- `diagnostics/`: `views.py`.

Every other module (the `_*.py` files) is an implementation detail
re-exported via its package `__init__.py`.

Test: *if a user can plausibly type the module path on a doc page or
in their own code, the module is public. If it only exists to
factor implementation off the foundational class, it is private.*

---

## 3. Docstring Conventions

Use **NumPy-style** docstrings with `Parameters`, `Returns`, and `Raises`
sections as needed. Their prose follows the writing rules of §10.

### 3.1 Module docstrings

Every module has a docstring explaining its purpose. Include a usage
example when helpful:

```python
"""The sample operation: one draw, or a batch of draws, of a distribution.

``sample(d)`` returns one draw at the kind the law's event declaration names,
and a non-empty ``sample_shape`` prepends batch axes on a level named
``sample``, returning the batch form of that kind.

Usage::

    from probpipe import Normal, sample
    draws = sample(Normal(loc=0.0, scale=1.0, label="x"), sample_shape=(100,))
"""
```

### 3.2 Class docstrings

Summary line, then a `Parameters` section for the constructor's parameters.
`__init__` has no docstring of its own:

```python
class Normal(TFPDistribution):
    """Univariate normal (Gaussian) distribution.

    Parameters
    ----------
    label : str
        Distribution label.
    loc : array-like
        Mean of the distribution.
    scale : array-like
        Standard deviation (> 0).
    """
```

### 3.3 Function/method docstrings

Summary line, then `Parameters`, `Returns`, and `Raises` as needed. A docstring
with any section documents every parameter in `Parameters`, in signature order,
and the returned value in `Returns`. Each entry states its type in words, such as
`int or tuple of int` or `array-like`:

```python
@operation(result=_sample_result)
def sample(d: Distribution, sample_shape: tuple[int, ...] = ()):
    """Draw from a distribution.

    Parameters
    ----------
    d : Distribution
        The law to draw from.
    sample_shape : int or tuple of int
        The batch axes to prepend; ``()`` draws once, and a bare integer is one
        axis.

    Returns
    -------
    TrackedTerm
        One draw at the kind the event declaration names; or, for a non-empty
        *sample_shape*, the batch form of that kind.

    Raises
    ------
    ApplicabilityError
        If *sample_shape* is malformed.
    ResolutionError
        If *d* does not sample.
    """
```

The `pydoclint` pre-commit hook checks the parameters and the returned value, and
the CI lint job runs it over `probpipe/` (CONTRIBUTING.md § Linting & pre-commit).

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

Every module of `probpipe/` other than a package's `__init__.py` starts with:

```python
from __future__ import annotations
```

The import postpones the evaluation of annotations, so an annotation can name a
class defined later in the module or imported only under `TYPE_CHECKING`.
Ruff enforces the rule through the `required-imports` setting in
`pyproject.toml`. `probpipe/linalg/operations.py` lacks the import, so its
per-file ignore stays until a change to its source adds it.

### 4.2 Import order

1. `from __future__ import annotations`
2. Standard library (`functools`, `logging`, `dataclasses`, `typing`, etc.)
3. Third-party (`jax`, `jax.numpy`, `tensorflow_probability`, etc.)
4. Internal (relative imports within `probpipe`)

Separate each group with a blank line.

### 4.3 Relative imports

Always use **relative imports** for internal references:

```python
from ..core.provenance import Provenance
from ..custom_types import Array, PRNGKey
from ..distributions._capabilities import SupportsSampling
from ..distributions._distribution import Distribution
```

### 4.4 Optional dependencies

Import an optional dependency where it is used, and raise a helpful error
there:

```python
try:
    import nutpie
except ImportError as e:
    raise ImportError(
        "nutpie is required for condition_on_nutpie. Install it with: pip install nutpie"
    ) from e
```

For test files, use `pytest.importorskip("nutpie")` or mock-based
approaches with `patch.dict(sys.modules)`; see §8.4 for the per-backend
testing patterns (e.g. the BridgeStan probe-compile fixture for Stan).

### 4.5 `Callable` and other ABCs

Import `Callable`, `Iterable`, `Iterator`, `Mapping`, `Sequence`,
`Hashable`, etc. from `collections.abc`, not from `typing`:

```python
# ✓ Preferred
from collections.abc import Callable, Iterable, Mapping, Sequence

# ✗ Soft-deprecated since Python 3.9
from typing import Callable, Iterable, Mapping, Sequence
```

Generic aliases of the same names in `typing` have been deprecated
since Python 3.9 (PEP 585). Use the `collections.abc` originals so the
codebase is consistent and future-proof. The only exceptions are
`typing.Optional` / `Union` style imports (don't use those — see §5
for the modern `X | None` form instead).

### 4.6 TYPE_CHECKING guards

Use `TYPE_CHECKING` for imports needed only by type checkers:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .distributions._distribution import Distribution
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

Type aliases are defined in `probpipe/custom_types.py`:

```python
type Array = jnp.ndarray
type ArrayLike = jnp.ndarray | list | tuple | float | int
type PRNGKey = jax.Array  # JAX PRNG key
```

These conventions are checked (advisorily) by pyright in CI — see
[CONTRIBUTING.md § Type checking](CONTRIBUTING.md#type-checking). The
check does not gate merges yet, but new code should type-check cleanly
under the project's `basic` pyright mode where practical.

---

## 6. Subpackage Dependencies

The dependency graph must remain **acyclic**. Each entry lists the packages
it imports at module level:

```
custom_types   (no internal deps — leaf)
     ↑
core/          (custom_types; the exceptions below)
     ↑
values/        (core/)
     ↑
linalg/        (core/, custom_types)
     ↑
distributions/ (core/, linalg/)
     ↑
functions/     (core/, values/, distributions/)
     ↑
operations/    (core/, values/, distributions/, functions/)
     ↑
families/      (core/, values/, linalg/, distributions/, functions/, operations/)
record/        (core/)
     ↑
inference/     (core/, values/, distributions/, functions/, operations/, families/)
validation/    (core/, distributions/, functions/, operations/)
diagnostics/   (distributions/, functions/, validation/)
```

The private helper modules (`_array_utils.py`, `_dtype.py`, `_weights.py`)
import only `custom_types` and each other, so every package may import them.

`values/_function_base.py` owns `Function` and `FunctionSpec`, and `functions/`
owns the engine, from binding through result wrapping. The base never imports the engine:
`install_call_engine` installs it when `functions/_function.py` is imported.
The pure Python binding helpers are in `values/_binding.py`, so raw evaluation
works without the engine. `functions/__init__.py` resolves its exports lazily.

> **Exceptions** (intentional reverse edges):
>
> - `core/` → `values/`: `core/_function_batch.py` and `core/_record_batch.py`
>   import from `values/_function_base.py`. The two batch modules belong in
>   `values/`, where `design/package-structure.md` places them.
> - `core/transition.py` → `values/`, `distributions/`, and `functions/`:
>   `iterate`, `with_conversion`, and `with_resampling` build `Function`s over
>   distributions, and their placement is an open point of
>   `design/package-structure.md`.
> - `core/` → `families/`: the function of `core/_fingerprint.py` that
>   fingerprints a `KDEDistribution` imports the class lazily.
> - `distributions/` → `diagnostics.views` (lazy import inside
>   `Distribution.diagnostics` to construct the read-only diagnostics accessor)
> - `diagnostics/` → `inference/`: four diagnostics modules import private
>   helpers of `inference/` lazily, such as the chain helpers of
>   `inference/_approximate_distribution.py`.
>
> A new reverse edge needs a maintainer's agreement first, and a lazy
> (in-function) import keeps it from creating a cycle at module load time.

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
- All other capability protocols (`SupportsSampling`, `SupportsMean`,
  `SupportsVariance`, `SupportsCovariance`, `SupportsExpectation`) are
  standalone, as are the two conditioning capabilities, which are abstract
  base classes rather than protocols.

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

Test files mirror the package tree, as `design/package-structure.md` § Principles
states: `tests/core/test_record.py` tests `probpipe/core/record.py`.

### 8.2 Test classes

Group related tests in `Test*` classes. Use pytest fixtures for shared
setup:

```python
class TestSample:
    def test_sample_scalar(self, normal):
        s = sample(normal)
        assert s.shape == ()
```

### 8.3 Fixtures

Define reusable fixtures at module scope:

```python
@pytest.fixture
def normal():
    return Normal(loc=2.0, scale=0.5, label="x")
```

Use `@pytest.fixture(params=...)` for parametrized testing across
distribution families.

### 8.4 Optional dependencies

- `pytest.importorskip("pymc")` for tests requiring optional packages.
- `patch.dict(sys.modules, ...)` for mock-based isolation of optional
  backends.
- nutpie tests replace nutpie's compiled model with a small stand-in class,
  so the helper and error-path tests need no compiled model.
- Stan tests compile real programs through BridgeStan, gated by a fixture
  that `importorskip`s `bridgestan` and probes the C++ toolchain, and run in
  the dedicated `stan` CI job (`--extra stan`). StanModel's backend-free tests
  (the parameter-name parser, the bridgestan-missing import guard, and the
  CmdStan import shim) need no compiled model and run in the main matrix.

### 8.5 Test runner

Tests run in parallel via `pytest-xdist` (`-n auto --dist worksteal`).
Disable for debugging: `pytest -p no:xdist -o "addopts="`.

### 8.6 Numerical correctness and tolerances

Tests of mathematical behavior (distributions, inference, learned
estimators, linear algebra) must validate against an **independent
baseline** — never the code path under test:

- an analytic result (conjugate posteriors, known moments);
- an exact reference computation run through the same pipeline (e.g.,
  judging a learned likelihood by comparing its NUTS posterior against
  NUTS run with the *true* likelihood on the same model — this isolates
  the component's error from sampler and prior effects);
- a known invariant (support membership, symmetry, positive
  definiteness, simplex sums, `log_prob`/`prob` consistency);
- central finite differences for gradients.

Checking `mean(d) == d.loc` only verifies passthrough; it is not a
correctness test. Where the claim is distributional, check **both
location and spread** (e.g., posterior mean *and* std against the
analytic values), not just point recovery.

**Tolerances are measured, not guessed.** For stochastic or trained
components, run the test's exact configuration across several seeds
(3–4 is typically enough), then set the bound to cover the observed
spread with roughly 2–3× margin. Seed everything that can be seeded —
training, simulation, sampling — so a given environment is exactly
reproducible; the margin exists only for cross-platform and
library-version numerical drift, not for RNG variation.

**Document the measured range in a comment next to the assertion**, so
the bound reads as calibrated rather than arbitrary, and anyone
tightening it later knows the baseline:

```python
# Observed across training seeds: mean err 0.05-0.10 post-std,
# std ratios 0.99-1.11.
assert mean_err < 0.3
assert 0.85 < ratio < 1.25
```

Use `np.testing.assert_allclose` (with explicit `atol`/`rtol`) for
numerical comparisons rather than bare `assert` with hand-rolled
tolerance arithmetic — its failure output shows the offending values.
Flag both failure directions when choosing bounds: too loose masks
bugs; too tight is flaky on other platforms.

---

## 9. Miscellaneous

### 9.1 `__all__` exports

Every public module defines `__all__`. Package `__init__.py` files
aggregate exports from private submodules. Private implementation
modules (`_*.py`) whose symbols are re-exported through the package
`__init__.py` are exempt.

### 9.2 Immutability

Design II.4 owns the immutability rule and its one writable store, the
append-only `annotations`. Every tracked term, a distribution included,
enforces the rule: assignment and deletion raise `AttributeError`, naming the
class, and an operation that changes a term returns a new one.

**A memo** holds what a term computes lazily. It is a `_memo` dictionary that
`transient_memo` in `probpipe/core/_immutable.py` creates on first use, and the
read that needs a value fills it in place, as `StanModel` does with its
BridgeStan model. Filling it leaves the term's own attributes as construction
set them, which is what the immutability guard sees. A class holding one
declares `_memo` in `_transient_state` so no copy inherits it, and whatever
reads it must tolerate its absence, since a copy or an unpickle arrives
without one.

### 9.3 Error and warning messages

A message is read by someone who has just made a mistake and does not know the
design. It says what went wrong in the caller's terms and how to fix it. These
rules govern every exception and warning message, which includes a
`Feasibility` description and the default message of an error class. The
writing rules of §10 govern docstrings and design prose, and they do not
govern messages, because a message built from §10 rules 1 and 2 and from the
glossary vocabulary explains the design to a reader who needs a fix.

1. **Lead with what failed.** Give the action that could not happen, then the
   reason with the offending value, then the fix. Write one or two short
   sentences, with no em-dash and no chain of semicolons.
2. **State the problem, not the design.** Leave out why the rule exists. State
   a rule only when the fix needs it, and state it as a requirement with
   *must* or *cannot*. A rule stated as a fact, such as "weights are
   nonnegative", reads as if it contradicts the input. Use *but* for a
   contrast, never *and*.
3. **Use the caller's words.** Name the function, the argument, and the value
   the caller passed. A term is fine when the public API or the user
   guide uses it, such as `ConditionalDistribution` or `sample_shape`. Describe
   any other term in plain words, which includes design terms of
   `design/glossary.md` such as packaging, whole term, and kernel.
4. **Show no private names.** A message names nothing with a leading
   underscore and no type the caller never sees, such as the class of a JAX
   array. A `NotImplementedError` says what is unsupported in public
   terms, such as "log_det_jacobian is not implemented yet", and never gives
   an internal method path.
5. **Show the offending value.** Print the value, shape, or type that failed
   whenever it is at hand. A lookup of a name that does not exist also
   lists the names that do, with `unknown_names` from `probpipe/_messages.py`.
   Keep `KeyError` where a `Mapping` or a docstring promises it.
6. **Give a fix only when it is certain.** Give the fix when it is short and
   holds for every way the check can fail. A check whose branches need
   different fixes raises a separate message from each branch.
7. **Word each check once.** A check raised from several places builds its
   message in one helper, so that one mistake reads the same everywhere.
8. **Follow the mechanics.** Start in lowercase unless the message starts with
   an identifier. End a message of one clause without a period. Write
   "got int" rather than putting an article before a type name, and pluralize
   a count with `count` from `probpipe/_messages.py`.
9. **Give each route's reason once.** A message that lists the routes or
   methods it tried already names each one, so a `Feasibility` description
   gives only the reason. A description of a failure the caller can fix, such
   as a field name the argument does not have, sets `actionable=True`, and the
   listing leads with it.

Each pair below shows a message that breaks these rules and its rewrite.

```text
# Explains the design (rules 1-3)
with_path_names() keeps the packaging, so the whole term's component 'mu' is renamed in place, not moved to 'population/mu'
cannot rename 'mu' to 'population/mu': with_path_names() can rename a single-component output but cannot move it into a group. Choose a name without '/'.

# Inverted lookup with no fix (rules 1 and 5)
not levels of this batch: ['test']; have ['quantile']
unknown level 'test'; available levels: ['quantile']

# Internal vocabulary (rule 3)
standalone replay does not support a parent with nested automatic workflow randomness
cannot replay this call: model draws random values through a nested Function call, which replay_run does not support. Replay the provenance of the inner call's result instead.

# Rule stated as a fact (rule 2)
X stacks the input points along its leading axis, so it has an axis
X must have a leading axis of input points, got a 0-d array; pass shape (n, ...)

# Private names and no value (rules 4 and 5)
_ensure_matrix: Required 3 columns. Got 2.
A must have 3 columns, got shape (3, 2)

# Repeated route name (rule 9)
curry (exact methods): route 'curry' declined: the conditioned object is not a kernel
curry (exact methods): Normal is not a ConditionalDistribution
```

A protocol check that fails raises `TypeError` and names the missing
capability:

```python
if not isinstance(dist, SupportsMean):
    raise TypeError(f"{type(dist).__name__} does not support mean; it must implement SupportsMean")
```

### 9.4 A scalar where a sequence is expected

An argument that takes a shape, level names, or one axis count per level also
takes a single item as a sequence of one:

- a shape takes a single int or str as one dimension, so `3` is `(3,)` and
  `"n"` is `("n",)`, by the rule of design II.1;
- level names take a single str as one name, so `"draw"` is `("draw",)`;
- axis counts take a single int as the count of one level.

Any other sequence, such as a tuple, a list, a `range`, or a 1-D array, is read
as one item per entry and stored as a tuple. An iterator such as a generator, a
set, `bytes`, a `memoryview`, and a mapping are refused, so an argument is never
used up or read in an arbitrary order. A `RecordSpec` field given as a shape
takes a tuple only, since a field's value may also be a spec. `probpipe/core/_shapes.py`
is the one place these arguments are read. A function that takes one calls the reader there
rather than calling `tuple()` on the argument, passes its own name and the
argument's for the error messages (§9.3 rule 7), and annotates the parameter
with the alias there, such as `ShapeLike` or `LevelNamesLike`.

---

## 10. Writing

These rules govern the prose of the repository: docstrings and comments, the
documentation and `design/`, and PR and issue text. Error and warning messages
follow §9.3 instead. `scripts/design/prose.py`
lists the places that may break the rules a script can detect, for a reader to
judge.

1. **One claim per sentence.** Write declarative sentences that each make one
   claim. Connect clauses with a word, such as *therefore*, *which*, or
   *because*, rather than with an appositive comma. A parenthetical is a short
   gloss or a citation. Use em-dashes sparingly, and never two in one sentence.
2. **State rules positively.** Say what a thing is. Cut a contrast that only
   says what a thing is not, and cut historical asides.
3. **Cut what adds nothing.** Cut each clause that adds nothing, and prefer the
   plain statement to a compressed parallel one.
4. **Justify truly.** A justification implies the rule it justifies.
5. **Format lists.** Where a list illustrates, give two or three examples.
   Three or more parallel items become a numbered or bulleted list with a colon
   gloss for each.
6. **Use plain words.** Prefer plain words to metaphor and jargon, and use none
   of these as a figure of speech:
   - words and phrases: "surface" for an API, "load-bearing", "machinery",
     "escape hatch", "door", "hook" unless it is a literal callback,
     "plumbing", "under the hood", "sugar", "elide", "knob", "dial", "seams",
     "fine print", "story", and "picture";
   - vague verbs: "reach", "touch", "hand back", "walk", "live", "ride",
     "land", and "sit".

   Use a generic word only in its literal sense, so "shape" is an array shape.
   Use a term that `design/glossary.md` defines only in that sense, so a "path"
   is a tree address.
7. **Name exactly.** Use no intensifier: "precisely", "deliberately",
   "genuinely", "of course", and "exactly" outside its mathematical sense. Name
   attributes, methods, and APIs exactly, and let each relational noun name its
   object.
8. **State each rule once.** State a rule in the section that owns it, and
   point to that section from every other place.
9. **Name things by behavior.** Code, comments, tests, and PR text name each
   thing by its behavior, and never by a development-plan label: a phase, a
   tier, a wave, or a stage letter.
10. **Keep PR and issue text self-contained.** Motivate each change from the
    repository's code, merged PRs, `design/`, and open issues.
