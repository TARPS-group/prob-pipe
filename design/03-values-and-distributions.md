# Part III — Values and Distributions

Part III introduces the value and distribution objects a user constructs and operates on, and the machinery specific to them. Each is built on the shared abstractions of Part II and introduced in dependency order. The final two sections cover the registries that act across these objects: cross-type conversion and constraint reparameterization.

## III.0 — Overview: the layer map

The sections build in the order below, each depending only on those above it and on the shared abstractions of Part II:

| §      | Layer                       | Contents                                                                                              | Role                                                                                                            |
| ------ | --------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| III.1  | Schema                      | `EventTemplate`                                                                                       | A `NamedTree` of type-specs — the type-level structure of one value. Pure structure, no data.                  |
| III.2  | Values                      | `FunctionBatch` / `OpaqueBatch`                                                                                       | Batches of function-valued and opaque values, giving every value spec a batch form.                             |
| III.3  | Values                      | `Record` / `NumericRecord`                                                                            | A `NamedTree` of values bound to an `EventTemplate` — the data-level counterpart.                              |
| III.4  | Values                      | `RecordBatch` / `NumericRecordBatch`                                                                  | A batch of records — what `sample` returns for many draws.                                                      |
| III.5  | Values                      | `LinOp`                                                                                               | A lazy structured linear map typed by numeric event templates, and the carrier of covariances.                          |
| III.6  | Distributions               | `Distribution`                                                                                        | A probability measure over one value type that carries an `event_template` for its draws.                           |
| III.7  | Distributions               | Distribution capabilities                                                                             | The `Supports*` protocols — sampling, density, moments, conditioning — a distribution implements.               |
| III.8  | Conditional Distributions   | `ConditionalDistribution`                                                                             | A probability kernel: a family of distributions indexed by a conditioning value, and a sibling of `Distribution`.   |
| III.9  | (Conditional) Distributions | `DistributionBatch` / `ConditionalDistributionBatch`                                                  | A batch of distributions (or conditional distributions): `N` separate laws, distinct from one joint distribution.                 |
| III.10  | (Conditional) Distributions | factored distributions (`SupportsFactors`, `FactoredDistribution`, `FactoredConditionalDistribution`) | A distribution built from named sub-distributions, with the factor and field access interfaces.                       |
| III.11  | (Conditional) Distributions | the `*` operator                                                                                      | Builds a joint from parts, with the result kind derived from the operands.                                        |
| III.12 | (Conditional) Distributions | `Distribution` hierarchy                                                                              | The catalog of kinds — basic, structured, and joint — assembled once composition exists.                        |
| III.13 | Registries | cross-type conversion (`converter_registry`) | Moving a distribution between representations, at a recorded fidelity. |
| III.14 | Registries | constraint reparameterization (`bijector_for`) | Mapping a `Constraint` to a bijector for unconstrained inference. |

## III.1 — `EventTemplate`

### Contract

An `EventTemplate` is a `NamedTree` that defines the *shape of one event* (a draw, a stored datum, ...). Hence, each leaf specifies the type of value it holds as a `ValueSpec` which must provide an `is_valid` method to check if a value satisfies that spec.

```python
class ValueSpec(ABC):
    @abstractmethod
    def is_valid(self, value: Any) -> bool: ...

class ArraySpec(ValueSpec):  # a numeric array leaf
    shape: tuple[int | str, ...]   # a str names a symbolic dimension
    dtype: DType
    support: Constraint

class OpaqueSpec(ValueSpec):  # the fallback spec; is_valid accepts any non-mapping value
    meta: Hashable

class FunctionSpec(ValueSpec):  # a leaf holding a callable
    input_template: EventTemplate | None   # None: that side's structure is unspecified
    output_template: EventTemplate | None

class Constraint(ABC):              # an array support, carried by ArraySpec
    @abstractmethod
    def check(self, value: ArrayLike) -> Array: ...   # elementwise membership
    # constraints compare and hash by value, so an instance can serve as a registry key
```

A `FunctionSpec` types a callable by its input and output structure. Either template is optional: `None` leaves that side's structure unspecified, so a bare `FunctionSpec()` describes any callable, and the templates are supplied only where the signature is known or required. Each specified side is an explicit `EventTemplate` — a single-field signature is written out, e.g. `FunctionSpec(EventTemplate(x=...), EventTemplate(out=...))` — so a function's field names are always caller-chosen and meaningful, matching `DistributionSpec` (which likewise takes an explicit template). The two templates are independent — a `FunctionSpec` places no relation between them, so a callable may map a space to itself or between two different spaces.

When every leaf is an `ArraySpec` then all values are numeric and construction auto-promotes to a `NumericEventTemplate`. The promotion is re-derived whenever a transform constructs a new template, so a replacement that removes the last non-numeric leaf promotes the result and one that introduces a non-numeric leaf demotes it: the numeric axis is an invariant of the current leaves, not of the object's history. Beyond the inherited `NamedTree` interface (with `L = ValueSpec`), `EventTemplate` adds construction, lossy template inference from a value, and projection to `NumericEventTemplate`:

```python
class EventTemplate(NamedTree[ValueSpec]):
    def __init__(self, field_specs: Mapping[str, Any] | None = None, /,
                 **fields: ValueSpec | EventTemplate | tuple[int, ...] | None) -> None: ...
    # sugar: a bare shape tuple means ArraySpec(shape) and None means OpaqueSpec();
    # the positional mapping form accepts "/"-path keys and names that collide with keywords

    @classmethod
    def infer_from(cls, value: Any) -> EventTemplate: ...   # best-effort, possibly lossy
    @property
    def is_numeric(self) -> bool: ...
    @property
    def is_concrete(self) -> bool: ...                      # False when any dimension is symbolic
    @property
    def free_dims(self) -> frozenset[str]: ...              # the unbound symbolic dimensions
    def numeric_subset(self) -> NumericEventTemplate: ...   # remove non-ArraySpec leaves
```

`NumericEventTemplate` further provides a flat (vectorized) layout of the leaves:

```python
class NumericEventTemplate(EventTemplate):
    @property
    def leaf_shapes(self) -> dict[str, tuple[int, ...]]: ...   # per-field array shapes, canonical order
    @property
    def vector_size(self) -> int: ...                          # total flat dimension; defined only when concrete
```

**Symbolic dimensions.** A shape entry may be a **named symbolic dimension** instead of an integer. `ArraySpec(shape=("obs", "features"))` fixes the rank and gives each dimension an identity while deferring its size, and within one template a name refers to one dimension: a template with fields `X: ("obs", "features")` and `coefficients: ("features",)` states that the second dimension of `X` and the length of `coefficients` are the same dimension, an equality no pair of concrete integers can express. A template with any symbolic entry is **polymorphic**, with `is_concrete` false and `free_dims` listing the unbound names. Templates carry no scope object beyond the names themselves, so they serialize as plain data.

A polymorphic template is checked by **unification** rather than per-leaf comparison. Validating values against it runs one pass over all fields: each occurrence of a name must resolve to a single size, a conflict raises, and a name, once bound, never rebinds. The per-leaf `is_valid` covers rank and dtype (an `ArraySpec`'s `support` is descriptive metadata, not checked by `is_valid`), and leaves size consistency to that one pass. Binding produces a new template, so refinement is monotone and nothing mutates. The flat layout of a `NumericEventTemplate` is defined only when the template is concrete, and anything that needs sizes raises with the free dimensions named.

### Rationale

As the *type layer*, an `EventTemplate` is the explicit structure that travels with a value and with the producers and consumers of values (`D5 – Explicit, carried structure`). It separates the structure of one event from the orthogonal axes of *multiplicity* and *identity*, keeping those distinctions explicit (`D1 – Mathematical fidelity`). A symbolic dimension carries a dimension's identity, which is mathematical structure, while deferring its size to the data that determines it, so cross-field equalities travel with the term and sizes bind when their producer appears (`D5 – Explicit, carried structure`, `C3 – Computational detail hidden by default, available on demand`).

---

## III.2 — `FunctionBatch` and `OpaqueBatch`

### Contract

Every value spec has a **batch form**. Since an `ArraySpec` value batches natively, as an array with the batch axes leading, no class is needed. Function-valued and opaque values have no native stacking, so two thin `Batch` specializations provide it. Each is `Batch` over its element type and carries the shared spec its elements satisfy, adding no other interface.

```python
class FunctionBatch(Batch[Callable]):
    @property
    def spec(self) -> FunctionSpec: ...   # the spec every element satisfies

class OpaqueBatch(Batch[Any]):
    @property
    def spec(self) -> OpaqueSpec: ...
```

### Rationale

The batch forms close the multiplicity axis over the value specs: `N` function draws are a *collection* of functions, never one function, the same `D1 – Mathematical fidelity` distinction every `Batch` enforces. Giving every value spec a batch form keeps batched operations total over event types (`D2 – Generality first`), so an operation that returns many draws can always stack them.

## III.3 — `Record` and `NumericRecord`

### Contract

A `Record` is a  `NamedTree` that is `Tracked` and `Annotated` with leaves that are *values*. Its structure conforms to an authoritative `EventTemplate`. Records provide a uniform representation for all types of values, including the data a function consumes and the draws a distribution produces. `NumericRecord` is the specialization in which every leaf is a numeric array and hence carries a `NumericEventTemplate`.

Since the structure of `Record` matches that of its template, the following invariants must hold:
1. *matching keys:* `record.keys() == record.event_template.keys()`.
2. *valid values:* for any valid key `p`, `record.event_template[p].is_valid(record[p])`.
3. *matching sub-templates:* for any valid non-key path `p`, `record.at_path(p).event_template == record.event_template.at_path(p)`.

Against a polymorphic template, the invariants are checked by one joint unification across all fields: with `X: ("obs", "features")` and `coefficients: ("features",)`, data of shapes `(100, 5)` and `(5,)` bind both dimensions consistently, while `(100, 5)` and `(7,)` raise. Construction binds the template, so a `Record` always carries the concrete, bound form and never an unbound dimension, and the data and its schema cannot disagree.

Two records are equal when they share a class, an `event_template`, and field-by-field equal data. Because the template is carried rather than re-inferred, an identity transform that threads it through compares equal to its input. A transform that instead rebuilds the template by inference matches only when that inference recovers the original, for instance when the original template was itself produced by `infer_from`.

```python
class Record(NamedTree[Any], Tracked, Annotated):
    def __init__(self, name: str, fields: Mapping[str, Any] | None = None, /, *,
                 event_template: EventTemplate | None = None,
                 name_is_auto: bool = False,
                 **kw_fields: Any) -> None: ...
        # name is the required first argument (semantic identity)
        # name_is_auto marks an operation-derived name (II.2); user constructions leave it False
        # a nested sub-record's name is its field key; a mapping-valued field is a subtree, never a leaf.
        # Binds to event_template if given (structural validation);
        # Otherwise, infers it once via EventTemplate.infer_from.

    @property
    def event_template(self) -> EventTemplate: ...
    def to_numeric(self) -> NumericRecord: ...  # requires every leaf to be an array

    @classmethod
    def from_field_values(cls, name: str, template: EventTemplate, values: Sequence[Any]) -> Record: ...
    # reconstruct from values in the template's canonical order; ValueError on count/shape mismatch

    def select(self, *fields: str, **mapping: str) -> dict[str, Any]: ...
    # fields into a plain dict for **-splatting into a computation call;
    # keywords remap: select(x="r") == {"x": self["r"]}
    def select_all(self) -> dict[str, Any]: ...   # every top-level field, ready to splat
```

`select` resolves each argument with `at_path`, so a key reaches a leaf and a partial path a subtree view, and returns a plain `dict` carrying no schema; its purpose is `**`-splatting a value's parts into a computation call (Part IV), with `select_all` the whole-record form over the top-level children.

When every leaf is numeric, a `Record` is a `NumericRecord`. Leaves are stored in native form — a bare array, an `xarray` / `pandas` container, or any registered array backend — and convert to `jax.Array` only at the compute boundary (the pytree flatten that `grad` / `vmap` / `jit` traverse, and `to_vector`), each leaf at most once. Because promotion changes no data, construction auto-promotes exactly when every leaf is numeric and no explicit non-numeric template vetoes it, and every transform re-derives the promotion from the current leaves — removing the last non-numeric leaf promotes, introducing one demotes — exactly as for `EventTemplate`. Flat vectorization reads its layout (`leaf_shapes`, `vector_size`, canonical order) from the template. At the boundary a `NumericRecord` presents a bare array pytree, so it passes through `grad` / `vmap` / `jit` unchanged and a JAX round-trip returns bare-array leaves; passing through means leaf transport only — a transform never promotes a `Record` to a `RecordBatch`. Flattening is deliberately numeric-only, which is why `NamedTree` itself has no `flatten`.

```python
class NumericRecord(Record):
    def to_vector(self) -> Array: ...
    @classmethod
    def from_vector(cls, name: str, template: NumericEventTemplate, vec: Array) -> NumericRecord: ...
```

### Rationale

A `Record` is the *values* half of `C1 – Uniform interface to distributions and values`: a distribution's draw is a `Record` (or a `RecordBatch` for many), and a function over named values consumes one. It is where `D5 – Explicit, carried structure` becomes concrete — a `Record` *carries* its template as authoritative structure, threaded forward from whoever produced it, rather than having it re-inferred from raw arrays downstream. Inheriting the `NamedTree` interface, a value's parts are reached by meaningful name (`C5 – Naming for unambiguous meaning`) and navigation yields views, not copies (`D7 – Single source of truth`).

### Notes

- *Pytrees.* `Record` and `NumericRecord` are registered as JAX pytrees for advanced use, and the native `NamedTree` methods are the supported interface. JAX traversal follows the pytree registration, which does not always agree with ProbPipe on what is a leaf, so users applying raw JAX functions are responsible for the documented behavior. Record equality is structural value equality, which is weaker than treedef equality. The registration's children are the field arrays (a `NumericRecord`'s native leaves convert at this boundary) and its static aux data is the template and the identity pair (`name`, `name_is_auto`) only, so provenance, annotations, and native container types do not survive a JAX transform boundary; lineage instead rides on the computation layer, which records the transform itself.

- *Single-field presentation.* A single-field `NumericRecord` forwards the explicit coercion entry points to its sole leaf — `float` / `int` / `bool`, `np.asarray` / `jnp.asarray`, and the `.shape` / `.dtype` / `.ndim` attributes — and a single-field `Record` holding a callable forwards `__call__`. The shim is deliberately narrow: no arithmetic, reductions, or slicing, and a multi-field record raises and points at explicit field access, since unwrapping one field of many would be ambiguous.
- *Construction validation.* Construction checks each leaf against its spec's `is_valid`, which validates structure only — for an `ArraySpec`, shape and dtype (dtype by `numpy.can_cast` same-kind: a widening promotion or a within-kind narrowing passes, a cross-kind conversion raises). An `ArraySpec`'s `support` is **not** part of `is_valid`: it is a data-dependent, element-wise check that reduces to a Python `bool` and so cannot run under `jax.jit` tracing, where construction also happens (pytree unflatten reconstructs a value inside the trace). `support` is therefore descriptive metadata, and invariant 2 (`is_valid`) covers shape and dtype. Leaf validation is skipped on the unflatten path, where a leaf's shape is transform-relative.

## III.4 — `RecordBatch` and `NumericRecordBatch`

### Contract

A `RecordBatch` is a batch of `Record`s that all conform to one shared `EventTemplate`. It is the batched value a computation produces and consumes, such as the many draws a `sample` yields. Being a `Batch`, it is `Tracked` but not `Annotated`, and it is a *collection* of records rather than itself a named tree. `NumericRecordBatch` is the all-array specialization. Indexing reaches both axes and stays unambiguous by dispatching on the key's type:

```python
class RecordBatch(Batch[Record]):
    @property
    def event_template(self) -> EventTemplate: ...

    def __getitem__(self, key: int | slice | tuple[int, ...] | str | tuple[str, ...]) -> Record | RecordBatch | Array | Batch: ...
    # int / slice (or a tuple of ints) -> an element Record or a sub-batch, indexing the batch axes
    # field path (str or tuple of strs) -> the field's column in its native batch form:
    #   an array for an array field, the matching element batch otherwise; a sub-RecordBatch if nested
```

Storage is columnar: per-field columns in each field's batch form. A field column is therefore a direct view, and an element `Record` is assembled on demand from the columns rather than stored a second time. A `RecordBatch` omits the field-keyed `Mapping` protocol (`keys()` / `values()` / `children`), so `len` and `iter` unambiguously range over the batch, and the field structure is read from `event_template`.

When every element is a `NumericRecord`, the batch is a `NumericRecordBatch`: a pytree of arrays whose leading dimensions are the `batch_shape`, bound to one shared `NumericEventTemplate`. Because it is a bare array pytree, it passes through `vmap` / `grad` / `jit` unchanged. It also adds batched flat vectorization, where `to_vector` stacks one flat vector per element into a `(*batch_shape, vector_size)` array:

```python
class NumericRecordBatch(RecordBatch):
    def to_vector(self) -> Array: ...
    @classmethod
    def from_vector(cls, name: str, template: NumericEventTemplate, vec: Array) -> NumericRecordBatch: ...
    # vec has shape (*batch_shape, vector_size): the last axis is the flat dimension
```

### Rationale

A `RecordBatch` makes `D1 – Mathematical fidelity` concrete on the value side: a batch of `N` records is a *collection* of `N` distinct records, never the same as a single record with `N` fields. This is why it claims only the batch axis and never the leaf-keyed `Mapping` contract.

## III.5 — `LinOp`

### Contract

A `LinOp` is a lazy linear map `A : ℝⁿ → ℝᵐ` between flat numeric spaces, and it is `Tracked`. It is how ProbPipe represents structured matrices, above all covariances, without materializing them. It carries an input and an output `NumericEventTemplate` (mirroring a `FunctionSpec`), so it maps numeric records and not just anonymous vectors; those templates name its domain and codomain, and a bare matrix is given names explicitly rather than defaulting to a single-field placeholder. The two sides coincide exactly when the operator maps a space to *itself* (an endomorphism such as a covariance or Hessian): then `input_template == output_template`, which the operator algebra reads as the structural fact that operands compose or act on the same space.

Its templates are always concrete, and construction from a template with unbound dimensions raises. A consumer whose sizes are not yet known holds the operator as a recipe, the operator class and its size-free parameters, and mints the instance once the sizes are bound. The base fixes the action and the square-only queries, and every query raises `LinAlgError` where it is undefined:

```python
class LinOp(Tracked, ABC):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]: ...    # (output_template.vector_size, input_template.vector_size)
    @property
    @abstractmethod
    def dtype(self) -> DType: ...
    @property
    @abstractmethod
    def input_template(self) -> NumericEventTemplate: ...
    @property
    @abstractmethod
    def output_template(self) -> NumericEventTemplate: ...
    @abstractmethod
    def to_dense(self) -> Array: ...

    def matvec(self, x: Array | NumericRecord) -> Array | NumericRecord: ...
    # A x; a NumericRecord flattens through input_template, and the result matches the argument's form
    def matmat(self, X: Array) -> Array: ...   # A X on a matrix; rmatvec / rmatmat apply the transpose

    # square-only queries
    def solve(self, b: Array) -> Array: ...
    def cholesky(self) -> LinOp: ...           # a triangular factor L with A = L Lᵀ
    def diag(self) -> Array: ...
    def logdet(self) -> Array: ...   # scalar Arrays rather than floats, keeping the queries differentiable
    def trace(self) -> Array: ...

    @property
    def flags(self) -> frozenset[str]: ...      # structure metadata, e.g. "symmetric", "positive_definite"
    def with_flag(self, flag: str) -> Self: ... # functional; construction otherwise fixes the flags
```

**The operator algebra.** `A @ B`, `A + B`, `c * A`, and `A.T` return lazy composite operators (`ProductLinOp`, `SumLinOp`, `ScaledLinOp`, and a transpose view) that defer to their parts. The scalar `*` coexists with distribution composition by operand type. The algebra checks and propagates the templates: `A @ B` requires `B`'s output template to equal `A`'s input template and carries `(B.input_template, A.output_template)`, `A + B` requires both pairs to match, and `A.T` swaps them. Composite operators are tracked terms like any other, with names auto-derived from their operands and marked `name_is_auto`.

**Structured subclasses.** `DenseLinOp`, `DiagonalLinOp`, `TriangularLinOp`, `CholeskyLinOp`, `RootLinOp`, and `DiagonalRootLinOp` each override the queries their structure accelerates, such as a triangular solve or a diagonal log-determinant.

**The batch form.** `LinOpBatch` is the element batch over operators, a thin `Batch[LinOp]` whose elements share both templates. It is what a batched `cov` returns. Application is elementwise: a single operator maps over a `RecordBatch`'s elements, and a `LinOpBatch` zips with a broadcast-compatible `RecordBatch`, element by element in both cases. The queries lift the same way, elementwise to batched results.

### Rationale

Operations mint linear operators, covariances above all, and every operation must return a tracked term, so a `LinOp` is `Tracked` (`D4 – Closed system of objects under operations`). The structured subclasses exploit their form automatically behind one interface (`C3 – Computational detail hidden by default, available on demand`), the algebra returns lazy views rather than materialized matrices (`D7 – Single source of truth`), array-backed operators keep every query differentiable (`D6 – Differentiability where possible`), and flags are functional rather than mutating (`C2 – Functional interface over immutable objects`). Typing both sides with numeric event templates is what makes closure concrete: the operator `cov` returns accepts the very draws its distribution produces (`D5 – Explicit, carried structure`).

### Open points

- *Structure-exploiting solves.* Exploiting structure in both operands of `A⁻¹B`, possibly through a dedicated `SolveLinOp`, is open.
- *Flag semantics.* Whether flags only describe structure or also steer which implementation a query selects is open.
- *Batched matrix action.* `matmat` against a batched operand, where a batch axis would meet the operator's matrix axis, and any richer `LinOpBatch` alignment are deferred until a concrete consumer exists.

## III.6 — `Distribution`

### Contract

A `Distribution[T]` is a single random law: a probability measure over values of type `T`. The type `T` is the natural raw form of a draw (an `Array` for a scalar law, or a `Record` for a multi-field one), which implementer code uses directly. It carries an `event_template` (the schema of one draw) and is `Tracked` and `Annotated`. It declares the operations it supports as **capabilities**, which are structural protocols it implements, so operational support is decoupled from the class. The draw a *user* sees is a `Record`: for a scalar law, a single-field `Record` keyed by the distribution's `name`. Fields can be renamed with `with_path_names`, which returns the same law under new field names. A distribution whose template is polymorphic is legal: operations that need sizes raise, naming the free dimensions, until a value binds them or `with_dims` does so explicitly.

A `NumericDistribution` is a `Distribution` whose draws are numeric, so it carries a `NumericEventTemplate` and can use the flat-vector machinery.

```python
class Distribution[T](Tracked, Annotated):
    def __init__(self, name: str, event_template: EventTemplate) -> None: ...

    @property
    def event_template(self) -> EventTemplate: ...   # fixed at construction
    @property
    def event_shape(self) -> tuple[int, ...]: ...    # defined only when a draw is a single array

    def with_path_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # rename event fields; keys resolve as for NamedTree.with_path_names, and the law is unchanged
    def with_dims(self, **sizes: int) -> Self: ...
    # bind named symbolic dimensions; monotone, and a conflict with an existing binding raises
    def __getitem__(self, path: str | tuple[str, ...]) -> FieldView: ...
    # the field view at a leaf or group path; available on every distribution

class NumericDistribution(Distribution): ...   # marker: numeric draws, carries a NumericEventTemplate
```

**Field views.** `d[path]` returns a `FieldView`: a `Distribution` over the field or field group at `path`, holding a reference to its parent rather than a detached law. Sibling views co-sample from one parent draw, so correlation between them is preserved. Every distribution carries an `event_template` with at least one named field, so views exist on every distribution, however it was constructed and whatever it supports. The capabilities a view offers are derived from its parent's, one by one, fixed with the capability protocols.

```python
class FieldView(Distribution):
    # constructed by Distribution.__getitem__, never by hand
    @property
    def parent(self) -> Distribution: ...
    @property
    def path(self) -> str: ...
    # event_template == parent.event_template.at_path(path); name == the field key,
    # marked name_is_auto; provenance records the view and its parent
```

**The distribution value specification.** The `DistributionSpec(event_template)` class extends the value spec to include a `Distribution` as a valid value (with the given `event_template`). Hence, it is possible to define random measures (distributions over distributions). A draw from a random measure is therefore a `Record` whose leaf value is a `Distribution`, wrapped like any other draw.

```python
DistributionSpec(event_template: EventTemplate)
```

### Rationale

Including a `Distribution` class is necessary to satisfy `C1 – Uniform interface to distributions and values`. It carries its draw schema rather than re-inferring it at each step to satisfy `D5 – Explicit, carried structure`, and its operations are pure to satisfy `C2 – Functional interface over immutable objects` and, when it is array-backed, differentiable end-to-end (`D6 – Differentiability where possible`). A field view is a reference rather than a copy (`D7 – Single source of truth`), and deriving its capabilities from its parent's keeps advertised support honest (`D3 – Capability-based operations`).

## III.7 — Distribution capabilities

### Contract

Each operation on a distribution is a **capability**: a distribution implements an underscore method (`_sample`, `_log_prob`, `_mean`, …) on the raw type `T` for each operation it supports, and the matching user-facing op (`sample`, `log_prob`, `mean`, …) wraps the result at the boundary when appropriate.

```python
@runtime_checkable
class SupportsSampling[T](Protocol):
    def _sample(self, key: Key, sample_shape: tuple[int, ...] = ()) -> T: ...
    # one draw for sample_shape=(); a non-empty shape prepends batch axes

@runtime_checkable
class SupportsUnnormalizedLogProb[T](Protocol):
    def _unnormalized_log_prob(self, value: T) -> Array: ...   # log-density up to an additive constant

@runtime_checkable
class SupportsLogProb[T](SupportsUnnormalizedLogProb[T], Protocol):
    def _log_prob(self, value: T) -> Array: ...                # the *normalized* log-density (refines the above)

@runtime_checkable
class SupportsRandomUnnormalizedLogProb(Protocol):
    def _random_unnormalized_log_prob(self) -> Distribution: ...
    # for a random measure M: the law of x ↦ log D̃(x) with D ~ M, itself a random function

@runtime_checkable
class SupportsRandomLogProb(Protocol):
    def _random_log_prob(self) -> Distribution: ...   # likewise, with the normalized log-density of a draw

@runtime_checkable
class SupportsMean[T](Protocol):
    def _mean(self) -> T: ...       # event-typed: a value shaped like a draw

@runtime_checkable
class SupportsVariance[T](Protocol):
    def _variance(self) -> T: ...   # event-typed, like _mean

@runtime_checkable
class SupportsCovariance(Protocol):
    def _cov(self) -> LinOp: ...    # a (d, d) operator over the flat numeric event

@runtime_checkable
class SupportsQuantile[T](Protocol):
    def _quantile(self, q: ArrayLike) -> Array: ...   # one value per level in q; per-coordinate for multivariate

@runtime_checkable
class SupportsExpectation[T](Protocol):
    def _expectation(self, f: Callable[[T], Array]) -> Array: ...   # exact E[f(X)] for arbitrary f

@runtime_checkable
class SupportsConditioning(Protocol):
    def _condition_on(self, given: Any, /, **kwargs: Any) -> Distribution: ...   # the conditional law given fixed values

@runtime_checkable
class SupportsMarginals(Protocol):
    def _marginal(self, path: str | tuple[str, ...]) -> Distribution: ...   # the detached marginal of a field or field group
```

Here `Key` is a PRNG key and `ArrayLike` an array-or-scalar input. `_cov` and `_quantile` require a numeric draw, with `_cov` ranging over the *flattened* draw, while `_mean` and `_variance` are event-typed and open to any event type that supports them. For a random measure, a draw's log-density is itself random, so the `SupportsRandom*LogProb` capabilities take no value and return the law of the log-density function, the random function `x ↦ log D(x)` for `D ~ M`. `_expectation` must integrate an *arbitrary* function exactly, which in practice means finite support: its argument is an opaque callable, so a per-call feasibility check has nothing to inspect, and a law that is exact only for special maps must not advertise the capability. Exact moments of structured maps are instead the business of `evaluate`, which dispatches on the map's type.

**View derivation.** A `FieldView` derives each capability from its parent's, so what a view supports is read off the parent. For a parent `d` and a view `v = d[p]`, with π the extraction of field `p` from an event:

| capability on `v` | derivation | available when |
|---|---|---|
| `_sample` | co-sample: draw `X ~ d` and return `π(X)` | parent `SupportsSampling` |
| `_mean` | projection: `mean(d)[p]`, since `E[πX] = π E[X]` | parent `SupportsMean` |
| `_variance` | restriction of `variance(d)` to the coordinates of `p` | parent `SupportsVariance` |
| `_cov` | the sub-block `P Σ Pᵀ`, with `P` the coordinate-selection `LinOp`, built lazily through the operator algebra | parent `SupportsCovariance`, numeric field |
| `_quantile` | restriction of the parent's per-coordinate quantiles to `p` | parent `SupportsQuantile`, numeric field |
| `_expectation` | composition: `d._expectation(f ∘ π)` | parent `SupportsExpectation` |
| `_log_prob` / `_unnormalized_log_prob` | via the detached marginal `d._marginal(p)` | parent `SupportsMarginals`, exact at `p`, and the marginal scores |
| `_marginal` at a sub-path `q` | path composition: `d._marginal(p/q)` | parent `SupportsMarginals`, exact at `p/q` |
| `_condition_on` a sub-field `s ⊂ p` | conditioning commutes with marginalization: `d.condition_on(s)[p ∖ s]`, both sides the law of `p ∖ s` given `s` | parent conditioning available for `s` |

The projection rows are exact whenever the parent's answer is, and the density rows are exact per path. Only sampling requires the parent to sample, so a view on a non-sampling parent still carries its projected moments.

### Rationale

Making each operation a *capability* rather than a base-class method follows `D3 – Capability-based operations`. Because support is structural (tested by `isinstance(dist, SupportsX)`, not subclassing), a distribution gains an operation just by implementing its method. A transform that preserves the event exposes exactly the capabilities of whatever it wraps, and a field view offers those its parent's capabilities can derive, so advertised support stays honest in both cases.

## III.8 — `ConditionalDistribution`

### Contract

A `ConditionalDistribution[S, T]` is a *probability kernel* `K : S → P(T)` — a family of distributions p(· | s) indexed by a *conditioning value* `s : S`. Supply a value for what it conditions on and it yields an ordinary `Distribution` over what it produces. A `ConditionalDistribution` is not a `Distribution` and does not inherit from one: it has no marginal law, so `sample` / `log_prob` / `mean` do not apply to it unconditionally. Rather, the two are siblings sharing a capability vocabulary.

A `ConditionalDistribution` carries two schemas: a `given_template` (the `EventTemplate` of the conditioning value `S`) and an `event_template` (the schema of one produced draw `T`). Unlike a function's domain and codomain, a kernel's given and event are distinct *roles* — the value conditioned on versus the law produced — so their field names stay disjoint even when the two spaces coincide. For example, a Markov kernel (where `S = T`) uses names like `state → next_state` rather than `state → state`, for the same reason we write `K(x, dy)` rather than `K(x, dx)`. Symbolic dimensions are scoped over the two templates jointly, so a name shared between given and event fields is one dimension, and the fused conditional paths bind dimensions from the given value at call time. `with_path_names` renames fields across both templates, returning the same kernel under new field names, and `with_dims` binds symbolic dimensions across both.

Users never call a method on the `ConditionalDistribution`. Instead, they use the existing ops: `condition_on(K, s)` binds the given fields, evaluating it to a `Distribution` exactly and with no inference, and `sample(K, given=s)` / `log_prob(K, y, given=s)` / `mean(K, given=s)` are the fused conditional paths, with the invariant `op(K, given=s) == op(condition_on(K, s))`, bitwise under a shared PRNG key in the exact cases and in law when inference is involved. Conditioning on only a subset of given fields *curries* to a smaller `ConditionalDistribution` view. A value supplied for a given field is always bound, whatever its type; the predictive mixture `∫ K(s, ·) μ(ds)` over a mixing distribution is obtained through the separate `predictive` operation, not by conditioning on a distribution.

```python
class ConditionalDistribution[S, T](Tracked, Annotated):
    def __init__(self, name: str, given_template: EventTemplate, event_template: EventTemplate) -> None: ...
        # given before event, as in FunctionSpec
    @property
    def given_template(self) -> EventTemplate: ...
    @property
    def event_template(self) -> EventTemplate: ...
    def _condition_on(self, given: S, /, **kwargs) -> Distribution[T] | ConditionalDistribution: ...
    # the required primitive: the law K(given, ·), or a curried kernel for a partial given

@runtime_checkable
class SupportsConditionalSampling[S, T](Protocol):
    def _conditional_sample(self, given: S, key: Key, sample_shape: tuple[int, ...] = ()) -> T: ...
@runtime_checkable
class SupportsConditionalLogProb[S, T](Protocol):
    def _conditional_log_prob(self, given: S, value: T) -> Array: ...
@runtime_checkable
class SupportsConditionalMean[S, T](Protocol):
    def _conditional_mean(self, given: S) -> T: ...
# … and likewise SupportsConditionalVariance (_conditional_variance(given) -> T),
#   SupportsConditionalCovariance (_conditional_cov(given) -> LinOp),
#   SupportsConditionalExpectation (_conditional_expectation(given, f, …) -> Array),
#   SupportsConditionalMarginals (_conditional_marginal(given, path) -> Distribution).
```

The conditional vocabulary is closed by one rule: every unconditional capability has a conditional counterpart whose method prepends the given to the unconditional signature. The two vocabularies stay mirrored by construction, and a capability added on the unconditional side names its conditional twin automatically.

**The numeric special cases.** A `ConditionalDistribution` has *two* templates, and either can be numeric, so the single `Numeric` prefix becomes positional: `Numeric` before `Conditional` marks the **given** side numeric, `Numeric` before `Distribution` marks the **event** side numeric, and `FullyNumeric*` marks both. Each is a marker only, adding no operations of its own (mirroring `NumericDistribution`).

```python
class ConditionalNumericDistribution(ConditionalDistribution): ...   # event numeric: every K(s, ·) is a NumericDistribution
class NumericConditionalDistribution(ConditionalDistribution): ...   # given numeric: the conditioning value is a numeric vector
class FullyNumericConditionalDistribution(
        NumericConditionalDistribution, ConditionalNumericDistribution): ...   # both sides numeric
```

| class | given | event |
|---|---|---|
| `NumericConditionalDistribution` | numeric | any |
| `ConditionalNumericDistribution` | any | numeric |
| `FullyNumericConditionalDistribution` | numeric | numeric |

**The conditional distribution value specification.** The `ConditionalDistributionSpec(given_template, event_template)` class extends the value spec to include a `ConditionalDistribution` as a valid value, analogously to `DistributionSpec`:

```python
ConditionalDistributionSpec(given_template: EventTemplate, event_template: EventTemplate)
```

### Rationale

Applying a `ConditionalDistribution` to a conditioning value returns a `Distribution`, which ensures `D4 – Closed system of objects under operations` is satisfied. A `ConditionalDistribution`'s capabilities are the `Distribution` capabilities shifted by one conditioning argument (`D3 – Capability-based operations`), so a single operation vocabulary applies to conditional distributions too, under the rule that *`Distribution` and `ConditionalDistribution` behave as similarly as possible*. As with `Distribution`, a concrete `ConditionalDistribution` family derives both templates from its parameters and passes them up, and the base only requires they are fixed at construction (`D5 – Explicit, carried structure`, `D7 – Single source of truth`). The capabilities use distinct `_conditional_*` method names because a `@runtime_checkable` check matches on method name alone, so reusing `_sample` / `_log_prob` would corrupt the unconditional capability checks. `_condition_on` is the deliberate exception: fixing given fields means the same thing on both types, so a `ConditionalDistribution` satisfying `SupportsConditioning` is intended rather than a collision, while the names stay distinct exactly where the meanings differ.

## III.9 — `DistributionBatch` and `ConditionalDistributionBatch`

### Contract

A `DistributionBatch` is a `Batch` of `Distribution`s: `N` separate distributions sharing one `event_template`, indexed along a batch axis. A `ConditionalDistributionBatch` is the same construction over `ConditionalDistribution`s: `N` separate conditional distributions sharing one `given_template` and one `event_template`. They are grouped because they are the identical multiplicity wrapper over the two distribution-like base types, and the conditional one merely adds the `given_template`. They are also the native batch forms of `DistributionSpec`- and `ConditionalDistributionSpec`-valued draws.

```python
class DistributionBatch(Batch[Distribution]):
    @property
    def event_template(self) -> EventTemplate: ...
    def __getitem__(self, index: int | slice) -> Distribution | DistributionBatch: ...

class ConditionalDistributionBatch(Batch[ConditionalDistribution]):
    @property
    def given_template(self) -> EventTemplate: ...
    @property
    def event_template(self) -> EventTemplate: ...
    def __getitem__(self, index: int | slice) -> ConditionalDistribution | ConditionalDistributionBatch: ...
```

### Rationale

This is `D1 – Mathematical fidelity` on the distribution layer: a `DistributionBatch` of `N` laws is a *collection of separate measures*, kept firmly distinct from one *joint* law over a product space, exactly as a `RecordBatch` of `N` draws is distinct from one `Record` of `N` fields. It is the natural result of a vectorized operation that yields many distributions: sweeping a parameter batch through a `ConditionalDistribution` produces a `DistributionBatch` of conditioned laws. Like every `Batch`, it is `Tracked` but not `Annotated`, and indexing or iterating yields a *view* (`D7 – Single source of truth`).

## III.10 — Factored distributions

### Contract

A *factored distribution* is a distribution **built from named sub-distributions**. Beyond being an ordinary distribution, it carries an explicit factorization into its parts. The capability that marks a factored distribution is `SupportsFactors`. The `FactoredDistribution` and `FactoredConditionalDistribution` classes generically implement `SupportsFactors` for distributions and conditional distributions. As another example, the `FactoredMultivariateGaussian` is a factored distribution in which the factors are jointly Gaussian, so conditioning and marginalization are exact.

The generic factored (conditional) distributions `FactoredDistribution` and `FactoredConditionalDistribution`  carry an ordered list of factors, each a `Distribution` or a `ConditionalDistribution`. The dependence graph is *derived* by matching each factor's given fields against the fields produced by earlier factors, rather than stored. The joint distribution's `event_template` is the structural, disjoint union of the factors' produced templates: each factor's top-level fields become top-level fields of the joint, with their internal structure preserved and no additional nesting introduced. Factor names are unique across the list, and a duplicate is an error.
In the case of the `FactoredConditionalDistribution`, conditioning values for all given fields results in a `FactoredDistribution`. Sampling and the log-prob capabilities follow the intersection of the factors': if all factors implement `SupportsLogProb` then so does the factored distribution. The moment capabilities are decided at construction, present exactly when the joint's structure makes the moment derivable, so a capability check stays honest. An edge-free joint derives its moments componentwise, so it has a given moment exactly when every factor does, the same intersection rule sampling and log-prob follow. Jointly Gaussian factors compose into a `FactoredMultivariateGaussian`, whose moments are exact. Any other dependent joint carries no moment capability, and the Monte Carlo fallback answers, since factor-wise conditional moments do not compose into a closed form. For example, with `x ~ Normal(0, 1)` and `y | x ~ Normal(exp(x), 1)`, `E[y] = e^{1/2}` is not reachable from the factors' means. As with other generic distributions, there are also numeric specializations.

```python
@runtime_checkable
class SupportsFactors(Protocol):
    @property
    def factors(self) -> tuple[Distribution | ConditionalDistribution, ...]: ...   # ordered; the graph is derived, not stored

class FactoredDistribution(Distribution, SupportsFactors): ...
class FactoredConditionalDistribution(ConditionalDistribution, SupportsFactors): ...

# numeric markers, by which of the (given, event) templates is numeric. Numeric is positional,
# as for the ConditionalDistribution markers: before Distribution marks the event numeric, before Conditional the given.
class FactoredNumericDistribution(FactoredDistribution): ...                                # unconditional joint, event numeric
class FactoredConditionalNumericDistribution(FactoredConditionalDistribution): ...          # conditional joint, event numeric
class FactoredNumericConditionalDistribution(FactoredConditionalDistribution): ...          # conditional joint, given numeric
class FactoredFullyNumericConditionalDistribution(
        FactoredNumericConditionalDistribution, FactoredConditionalNumericDistribution): ... # conditional joint, both numeric
```

**Field versus factor.** A **field** is a named part of a draw, that is, a path in the `event_template`. A **factor** is a constituent distribution the joint was built from. The two coincide only for an independent joint of single-field factors and differ in general. A correlated `MultivariateNormal` presented as `{intercept, slope}` is one factor with two fields. Conversely, the same draw `{x, y}` can arise from a single bivariate normal (no factors), from two independent factors (no edges), or from a chain p(y | x) · p(x) (two factors, one edge). The fields are identical but the factorization differs.

**The two access interfaces.** A joint exposes up to two clearly separated interfaces, never through the same operator.
- The **field interface** is available on every distribution. `d["intercept"]` returns a **view**: the field's marginal carrying a reference to its parent, so that sibling views co-sample from one parent draw and preserve correlation under broadcast. A view carries each capability its parent's capabilities can derive, its density routes through the parent's `SupportsMarginals`, and `marginal(d, "intercept")` returns that same marginal **detached** from the parent.
- The **factor interface** is available only with `SupportsFactors`. `factor(d, "coeffs")` returns a building-block factor, keyed by factor name, which is a `Distribution` or, for a dependent edge, a `ConditionalDistribution`. There need be no factor for a given field, and no field for a given factor.

**Marginals of a joint.** Whether a marginal is exactly available depends on the factors' own marginal support and on where the target sits in the dependence graph, so the factored classes resolve `_marginal` per path rather than wholesale. The graph reduction is always exact: the target's ancestor closure yields a sub-joint of whole factors, and everything outside it integrates out for free. What remains is integrating the extra ancestor fields back out, which is exact in three cases: there are none (the target is ancestrally closed, as for a root factor or an edge-free group), the reduction lies within a single factor and delegates to that factor's own `SupportsMarginals`, or the affected factors admit closed-form integration, as when they are jointly Gaussian. On any other path `_marginal` raises, and the `marginal` operation falls back to its Monte Carlo route.

### Rationale

Factorization is an *optional capability*, `SupportsFactors`, rather than a base class. This is `D2 – Generality first`: a joint is an ordinary distribution that gains factor access by carrying the capability, instead of sitting in a parallel class tower. Keeping the field interface (part of a draw) and the factor interface (part of the construction) separate serves `D1 – Mathematical fidelity`, since the two are genuinely different in the mathematics. Independence is likewise a property of the derived graph rather than a class, so dropping any dedicated product class loses no behavior: an edge-free joint still samples its factors in parallel and conditions exactly on a field. For sampling and densities a joint's capabilities are the intersection of its factors': ancestral `sample` requires every factor to sample, and a summed `log_prob` requires every factor to score. Deciding moment presence at construction keeps `D3 – Capability-based operations` honest, since a capability is advertised exactly when the object can answer it.

### Notes

- *Group views.* The field interface also accepts an interior path, which names a group of fields rather than a single field. For example, when the `event_template` nests `coeffs/intercept` and `coeffs/slope` under `coeffs`, `d["coeffs"]` returns the marginal over the whole group. Like a single-field view, it is a view onto the parent joint, not a detached distribution, so co-sampling through the parent preserves correlation.

## III.11 — Composition

### Contract

Composition builds a factored distribution from parts, written as an *expression*: a single binary operator `*` combines `Distribution`s and `ConditionalDistribution`s into one joint. The *kind* of the result is **derived** from the operands and never chosen by hand, and every result is itself a `Distribution` or a `ConditionalDistribution`. The operator is defined after the factored-distribution classes it returns. The base objects carry no composition logic. Each merely exposes `*` as a thin `__mul__` that delegates to the operator, which keeps the dependency one-directional.

**The `*` operator.** `A * B` composes two operands into a joint. It is **conditional-first**: the left operand may condition on the right, so `lik * prior` reads as the density p(y | β) · p(β), while the reverse (a consumer before its producer) is an error. Characterize each operand by its **produced fields** `F` and its **unmet given fields** `G`, with `G = ∅` for a `Distribution` and `G` the given fields for a `ConditionalDistribution`. The composition is then fixed by the field sets alone:

```
bound  = G_A ∩ F_B            # dependency edges: left A conditions on a field that right B produces
unmet  = (G_A − F_B) ∪ G_B    # residual exogenous givens — met by no factor
require  F_A ∩ F_B = ∅         # each field is produced exactly once
     and G_B ∩ F_A = ∅         # B must not consume a field A produces — else reorder (producer on the right)
law:  p(F_A, F_B | unmet) = p_A(F_A | bound ∪ (G_A − F_B)) · p_B(F_B | G_B)      # reads left → right
```

The result has two mathematical degrees of freedom, *conditional?* (`unmet ≠ ∅`) and *dependent?* (`bound ≠ ∅`), but only the first is a **class** distinction:

| `unmet` | result |
|---|---|
| `∅` | `FactoredDistribution` — a joint `Distribution` |
| `≠ ∅` | `FactoredConditionalDistribution` — a joint `ConditionalDistribution`, its `given_template` exactly `unmet` |

`*` returns the **most specific** class, recomputed from the *flattened* factor graph at each step. `A * B * C` builds one flat N-factor joint, with independent factors commuting and dependent ones kept in conditional-first order. Symbolic dimensions never unify by name across operands, since two factors may both call something `"obs"` and mean different dimensions. Each operand's dimensions instead enter the joint under a deterministic factor-qualified renaming, shared fields contribute the only identifications, and the joint stores the factors so refined. The renaming is canonical, so derived names and fingerprints stay deterministic.

**Naming the result.** A joint is *derived*, not created by the user, so `*` **auto-derives** its name deterministically from its factors and marks it `name_is_auto`. An auto-named operand is flattened into a larger joint, while an operand pinned with `with_name` is kept as a single named factor.

```python
def __mul__(self, other: Distribution | ConditionalDistribution) -> FactoredDistribution | FactoredConditionalDistribution: ...
```

### Rationale

Reifying both degrees of freedom would force a 2×2 of joint classes. By `D2 – Generality first`, an independent product (`bound = ∅`) is just an edge-free joint, so *dependent?* is a runtime property of the derived graph and only *conditional?* names a class, giving two classes rather than four. The conditional-first order is already a valid topological listing, so composition is associative and acyclicity is automatic, with no separate graph inference. Associativity rests on the `G_B ∩ F_A = ∅` requirement, under which the validity of `(A * B) * C` and `A * (B * C)` coincide.

### Notes

- *Operator coexistence.* `*` also denotes scalar scaling on some objects, such as a random function or a linear operator. The two coexist by operand-type dispatch: `Distribution` and `ConditionalDistribution` operands compose, while scalar operands scale.
- *The realigning `joint` form.* `joint(A, B, **align)` is `*` plus field renaming, for factors whose names do not line up: it is `A * B.with_path_names(**align)`.

## III.12 — The `Distribution` hierarchy

### Contract

With the base `Distribution`, the `ConditionalDistribution`, and the factored distributions and their composition in place, the distribution *kinds* can be cataloged. Every class named here is defined earlier, and this section only organizes them. Two **independent** questions classify any distribution:
1. **The type axis** — what does a draw look like, and is a factorization exposed? This fixes *which interfaces apply*.
2. **The family axis** — how is the law realized? This fixes *which capabilities* the object implements, and how.

The axes are orthogonal, so they combine freely: a `Normal` is *atomic* (type) and *parametric* (family); a posterior over `{μ, σ}` reconstructed from samples is *atomic-structured* and *empirical*; a joint's factors may themselves come from any family.

**The type axis — atomic vs. joint.** The structural classification:

| type | a draw is | factorization? | interface beyond the capabilities |
|---|---|---|---|
| **atomic, array-valued** | one `Array` (scalar or vector) | none | — (a single-field draw) |
| **atomic, structured** | a multi-field `Record` | none | the *field* interface — `d["x"]`, `marginal` |
| **joint** | a `Record` | `SupportsFactors` | the *field* interface **and** the *factor* interface — `factor` |

The line between the last two is **factorization, not field count**: a multi-field empirical distribution or an amortized posterior has fields but no factors, so it is *atomic-structured*, and only a distribution built from named sub-distributions is *joint*. Hence there is **no `RecordDistribution`**: draw structure lives in `event_template` and factor access in `SupportsFactors`.

**The family axis — how the law is realized.** Each family is an ordinary `Distribution` (a `NumericDistribution` when its event is numeric), differing only in which capabilities it implements and how. This is refinement by capability, not a parallel class tower:
- **Parametric (closed-form).** The standard families: continuous (`Normal`, `Gamma`, `Beta`, `Exponential`), discrete (`Bernoulli`, `Categorical`, `Poisson`), and multivariate (`MultivariateNormal`, `Dirichlet`). They are backed by a tensor library, with exact `sample` / `log_prob` / moments and a constrained-support `event_template`, and they make up the bulk of the atomic, array-valued row. The multivariate families implement `SupportsMarginals` exactly.
- **Empirical.** A finite, possibly weighted, sample set: `sample` resamples, moments are sample estimates, and marginals are again empirical. It carries no density. Scalar or structured.
- **Transformed (pushforward).** A base distribution pushed through a map with recognizable structure. An invertible map keeps an exact `log_prob` by change of variables, and a linear map keeps exact first and second moments. A general map's pushforward proceeds by sampling instead, so its result lands in the empirical family.
- **Mixture.** A convex combination of finitely many component distributions. It is the form a dependent joint's detached `marginal` takes when the mixing parent is finite. Over a continuous parent the true marginal is a continuous mixture, which no finite-component family can represent, so the `marginal` operation returns its Monte Carlo fallback, an `EmpiricalDistribution` of projected draws, unless an exact route applies, as when the factors are jointly Gaussian.
- **Random function.** A distribution over functions, whose event is a `FunctionSpec` leaf: a draw is a callable, and `mean` returns the mean function. A Gaussian process is the canonical case.
- **Random measure.** A distribution *over distributions*: a draw is itself a `Distribution` (a `DistributionSpec` leaf), and `mean` returns the marginalized law.

Any family can arise as the approximation of a target: what makes a result approximate (the target, the method, and the fit) is recorded in its `provenance`.

**Conditional distributions and batches stratify identically.** A `ConditionalDistribution` repeats both axes (atomic / structured / the `FactoredConditionalDistribution` joint, crossed with parametric / amortized / empirical / …), and a `DistributionBatch` is `N` of any of these. The catalog is one classification, reused across the conditional and multiplicity layers.

### Rationale

The hierarchy embodies `D2 – Generality first`: one base refined by *optional capabilities* (`SupportsFactors`, the numeric marker, each `SupportsX`) rather than a rigid class tower, so a new family slots in by implementing the capabilities it supports rather than by widening the base. The atomic-versus-joint split is a `D1 – Mathematical fidelity` distinction, since a joint genuinely offers its factors as distributions while an atomic-structured law does not, and it is carried by a capability rather than by the draw's type. Because composition is closed (`D4 – Closed system of objects under operations`), a joint re-enters the catalog as an ordinary `Distribution` for the next operation.

## III.13 — Cross-type conversion

### Contract

A distribution may have more than one representation, and an operation or backend sometimes needs a different one than the user holds. **Conversion** moves a distribution from its current type to a requested target type, resolved by the **converter registry** whose methods are *converters*. The registry is an ordinary binary dispatch registry keyed on `(type(source), target)`: the target is already a type, so the second key is the argument itself rather than its type, supplied through the key extraction the dispatch base leaves to its subclasses.

A converter declares the source types it converts *from* and the target types it converts *to* in the binary method's two slots, a cheap `check` that reports feasibility without converting, and the conversion as its `execute`. A conversion is rarely unique, so each carries a **fidelity**: `exact` for an equivalent representation, `moment_match` when only low-order moments are preserved, and `sample` for a Monte Carlo stand-in. The fidelities are totally ordered, `EXACT > MOMENT_MATCH > SAMPLE`. Priority remains the registry's sole selection order, with converter priorities assigned from the fidelity tiers, `EXACT` in the exact tier and the inexact fidelities in descending bands below it, so higher fidelity is preferred through the existing mechanism rather than a second ordering. A caller can name a converter or set `min_fidelity`, a feasibility floor `check` enforces.

```python
class ConversionMethod(Enum):
    EXACT = "exact"
    MOMENT_MATCH = "moment_match"
    SAMPLE = "sample"

@dataclass(frozen=True)
class ConversionInfo(MethodInfo):   # the feasibility probe's result, extended with fidelity
    method: ConversionMethod | None = None

class Converter(BinaryDispatchMethod):
    name: str                            # unique within the registry; convert(..., method=name) selects it
    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]: ...   # (source, target) types
    def check(self, source, target_type: type, *,
              min_fidelity: ConversionMethod | None = None) -> ConversionInfo: ...
    def execute(self, source, target_type: type) -> Distribution: ...             # the conversion itself

class ConverterRegistry(BinaryDispatchRegistry[Converter]):
    # keyed on (type(source), target): the second key is the target type itself
    def convert(self, source, target_type: type,
                method: str | None = None, min_fidelity: ConversionMethod | None = None) -> Distribution: ...

converter_registry: ConverterRegistry   # the global instance
```

### Rationale

Conversion makes `C3 – Computational detail hidden by default, available on demand` concrete on the distribution layer: a representation is a computational choice, so the library converts as needed and the user rarely converts by hand. Recording each conversion's fidelity keeps the approximation honest, which is `D1 – Mathematical fidelity`, since an `exact` conversion loses nothing while a `moment_match` or `sample` conversion is a stated approximation the caller can see and control. New representations interoperate by registering converters, so the set of convertible pairs grows without changing the distributions themselves (`D2 – Generality first`). Realizing the registry as a subclass of the shared dispatch machinery gives conversion registration, feasibility probing, prioritized selection, and cataloging without duplicating any of them (`D7 – Single source of truth`).

## III.14 — Constraint reparameterization

### Contract

Gradient-based and Hamiltonian inference work in an unconstrained space, so a constrained support must be reparameterized. The **constraint-to-bijector factory** maps a `Constraint` (the support an `ArraySpec` carries) to a bijector that takes `ℝⁿ` onto that support. `bijector_for(constraint)` returns the canonical bijector, and `register_bijector` plugs in a factory for a constraint type or for a specific constraint instance, with instance registrations taking precedence over type registrations.

```python
class Bijector(ABC):                    # an invertible map with a tractable Jacobian
    @abstractmethod
    def forward(self, x: Array) -> Array: ...
    @abstractmethod
    def inverse(self, y: Array) -> Array: ...
    @abstractmethod
    def forward_log_det_jacobian(self, x: Array) -> Array: ...

def bijector_for(constraint: Constraint) -> Bijector: ...    # the canonical map ℝⁿ → support(constraint)
def register_bijector(
    key: type[Constraint] | Constraint,
    factory: Callable[[Constraint], Bijector],
) -> None: ...
```

The factory keys on constraint instances and types rather than dispatching on argument types alone, so it is not a dispatch registry; it still satisfies `SupportsRegistryCataloging` and appears in the registry catalog alongside the dispatch registries.

### Rationale

A bijector for every constraint lets inference run in an unconstrained space while a model stays stated in its natural, constrained one, which serves `D6 – Differentiability where possible`. Keeping the map open through `register_bijector` is `D2 – Generality first`: a new constrained support becomes inference-ready by registering its reparameterization, without touching the distributions that use it.

### Open points

- *Round-trip fidelity.* The forward map (bijector to support) and this inverse map (support to bijector) are not strict inverses for every constraint, so a reparameterized support can drift to a coarser one. Whether to unify the two is unsettled.
