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
    shape: tuple[int, ...]
    dtype: DType
    support: Constraint

class OpaqueSpec(ValueSpec):  # a non-array Python object
    meta: Hashable

class FunctionSpec(ValueSpec):  # a leaf holding a callable
    input_template: EventTemplate
    output_template: EventTemplate
```

A `FunctionSpec` accepts a bare `ValueSpec` on either side and wraps it in a single-field template, so `FunctionSpec(ArraySpec(...), ArraySpec(...))` types a scalar function by the same convention that presents a scalar draw as a single-field value.

When every leaf is an `ArraySpec` then all values are numeric and construction auto-promotes to a `NumericEventTemplate`. Beyond the inherited `NamedTree` interface (with `L = ValueSpec`), `EventTemplate` adds construction, lossy template inference from a value, and projection to `NumericEventTemplate`:

```python
class EventTemplate(NamedTree[ValueSpec]):
    def __init__(self, **fields: ValueSpec | EventTemplate) -> None: ...

    @classmethod
    def infer_from(cls, value: Any) -> EventTemplate: ...   # best-effort, possibly lossy
    @property
    def is_numeric(self) -> bool: ...
    def numeric_subset(self) -> NumericEventTemplate: ...   # remove non-ArraySpec leaves
```

`NumericEventTemplate` further provides a flat (vectorized) layout of the leaves:

```python
class NumericEventTemplate(EventTemplate):
    @property
    def leaf_shapes(self) -> dict[str, tuple[int, ...]]: ...   # per-field array shapes, canonical order
    @property
    def vector_size(self) -> int: ...                          # total flat dimension
```

### Rationale

As the *type layer*, an `EventTemplate` is the explicit structure that travels with a value and with the producers and consumers of values (`D5 – Explicit, carried structure`). It separates the structure of one event from the orthogonal axes of *multiplicity* and *identity*, keeping those distinctions explicit (`D1 – Mathematical fidelity`).

### Open points

- *Validation scope.* Whether construction performs structural validation only, with opt-in deep (shape/dtype) checks, is open.

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

Two records are equal when they share a class, an `event_template`, and field-by-field equal data. Because the template is carried rather than re-inferred, an identity transform that threads it through compares equal to its input. A transform that instead rebuilds the template by inference matches only when that inference recovers the original, for instance when the original template was itself produced by `infer_from`.

```python
class Record(NamedTree[Any], Tracked, Annotated):
    def __init__(self, name: str, fields: Mapping[str, Any] | None = None, /, *,
                 event_template: EventTemplate | None = None,
                 **kw_fields: Any) -> None: ...
        # name is the required first argument (semantic identity)
        # a nested sub-record's name is its field key.
        # Binds to event_template if given (structural validation);
        # Otherwise, infers it once via EventTemplate.infer_from.

    @property
    def event_template(self) -> EventTemplate: ...
    def to_numeric(self) -> NumericRecord: ...  # requires every leaf to be an array

    @classmethod
    def from_field_values(cls, name: str, template: EventTemplate, values: Sequence[Any]) -> Record: ...
    # reconstruct from values in the template's canonical order; ValueError on count/shape mismatch
```

When every leaf is a numeric array, a `Record` is a `NumericRecord`, which is itself a pytree of arrays (its children the field values, its `event_template` static metadata). It adds flat vectorization, reading the necessary information (`leaf_shapes`, `vector_size`, and canonical order) from the template. Because a `NumericRecord` is a bare array pytree, it passes through `grad` / `vmap` / `jit` unchanged. 

```python
class NumericRecord(Record):
    def to_vector(self) -> Array: ...
    @classmethod
    def from_vector(cls, name: str, template: NumericEventTemplate, vec: Array) -> NumericRecord: ...
```

### Rationale

A `Record` is the *values* half of `C1 – Uniform interface to distributions and values`: a distribution's draw is a `Record` (or a `RecordBatch` for many), and a function over named values consumes one. It is where `D5 – Explicit, carried structure` becomes concrete — a `Record` *carries* its template as authoritative structure, threaded forward from whoever produced it, rather than having it re-inferred from raw arrays downstream. Inheriting the `NamedTree` interface, a value's parts are reached by meaningful name (`C5 – Naming for unambiguous meaning`) and navigation yields views, not copies (`D7 – Single source of truth`).

### Open points

- *Single-value coercion.* How a single-field `Record` presents is unresolved. For example, in the array case, it could present as a bare scalar/array (coercion via `float` / `jnp.asarray`, a `.shape` shim, or a dedicated single-value wrapper). Or, in the function case, it could support `__call__`. 

## III.4 — `RecordBatch` and `NumericRecordBatch`

### Contract

A `RecordBatch` is a batch of `Record`s that all conform to one shared `EventTemplate`. It is the batched value a workflow produces and consumes, such as the many draws a `sample` yields. Being a `Batch`, it is `Tracked` but not `Annotated`, and it is a *collection* of records rather than itself a named tree. `NumericRecordBatch` is the all-array specialization. Indexing reaches both axes and stays unambiguous by dispatching on the key's type:

```python
class RecordBatch(Batch[Record]):
    @property
    def event_template(self) -> EventTemplate: ...

    def __getitem__(self, key: int | slice | str | tuple[str, ...]) -> Record | RecordBatch | Array: ...
    # int / slice -> an element Record or a sub-batch
    # field path  -> that field's column, or a sub-RecordBatch if nested
```

Both forms return views. A `RecordBatch` omits the field-keyed `Mapping` protocol (`keys()` / `values()` / `children`), so `len` and `iter` unambiguously range over the batch, and the field structure is read from `event_template`.

When every element is a `NumericRecord`, the batch is a `NumericRecordBatch`: a pytree of arrays whose leading dimensions are the `batch_shape`, bound to one shared `NumericEventTemplate`. Because it is a bare array pytree, it passes through `vmap` / `grad` / `jit` unchanged. It also adds batched flat vectorization, where `to_vector` stacks one flat vector per element into a `(*batch_shape, vector_size)` array:

```python
class NumericRecordBatch(RecordBatch):
    def to_vector(self) -> Array: ...
    @classmethod
    def from_vector(cls, name: str, template: NumericEventTemplate, vec: Array) -> NumericRecordBatch: ...
```

### Rationale

A `RecordBatch` makes `D1 – Mathematical fidelity` concrete on the value side: a batch of `N` records is a *collection* of `N` distinct records, never the same as a single record with `N` fields. This is why it claims only the batch axis and never the leaf-keyed `Mapping` contract.

## III.5 — `LinOp`

### Contract

A `LinOp` is a lazy linear map `A : ℝⁿ → ℝᵐ` between flat numeric spaces, and it is `Tracked`. It is how ProbPipe represents structured matrices, above all covariances, without materializing them. It carries an input and an output `NumericEventTemplate` (mirroring a `FunctionSpec`), so it maps numeric records and not just anonymous vectors. A `LinOp` built from a bare matrix defaults to single-field templates, by the same convention that presents a scalar draw as a single-field `Record`. The base fixes the action and the square-only queries, and every query raises `LinAlgError` where it is undefined:

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
    def logdet(self) -> float: ...
    def trace(self) -> float: ...

    @property
    def flags(self) -> frozenset[str]: ...      # structure metadata, e.g. "symmetric", "positive_definite"
    def with_flag(self, flag: str) -> Self: ... # functional; construction otherwise fixes the flags
```

**The operator algebra.** `A @ B`, `A + B`, `c * A`, and `A.T` return lazy composite operators (`ProductLinOp`, `SumLinOp`, `ScaledLinOp`, and a transpose view) that defer to their parts. The scalar `*` coexists with distribution composition by operand type. The algebra checks and propagates the templates: `A @ B` requires `B`'s output template to equal `A`'s input template and carries `(B.input_template, A.output_template)`, `A + B` requires both pairs to match, and `A.T` swaps them.

**Structured subclasses.** `DenseLinOp`, `DiagonalLinOp`, `TriangularLinOp`, `CholeskyLinOp`, `RootLinOp`, and `DiagonalRootLinOp` each override the queries their structure accelerates, such as a triangular solve or a diagonal log-determinant.

**The batch form.** `LinOpBatch` is the element batch over operators, a thin `Batch[LinOp]` whose elements share both templates. It is what a batched `cov` returns.

### Rationale

Operations mint linear operators, covariances above all, and every operation must return a tracked term, so a `LinOp` is `Tracked` (`D4 – Closed system of objects under operations`). The structured subclasses exploit their form automatically behind one interface (`C3 – Computational detail hidden by default, available on demand`), the algebra returns lazy views rather than materialized matrices (`D7 – Single source of truth`), array-backed operators keep every query differentiable (`D6 – Differentiability where possible`), and flags are functional rather than mutating (`C2 – Functional interface over immutable objects`). Typing both sides with numeric event templates is what makes closure concrete: the operator `cov` returns accepts the very draws its distribution produces (`D5 – Explicit, carried structure`).

### Open points

- *Structure-exploiting solves.* Exploiting structure in both operands of `A⁻¹B`, possibly through a dedicated `SolveLinOp`, is open.
- *Flag semantics.* Whether flags only describe structure or also steer which implementation a query selects is open.

## III.6 — `Distribution`

### Contract

A `Distribution[T]` is a single random law: a probability measure over values of type `T`. The type `T` is the natural raw form of a draw (an `Array` for a scalar law, or a `Record` for a multi-field one), which implementer code uses directly. It carries an `event_template` (the schema of one draw) and is `Tracked` and `Annotated`. It declares the operations it supports as **capabilities**, which are structural protocols it implements, so operational support is decoupled from the class. The draw a *user* sees is a `Record`: for a scalar law, a single-field `Record` keyed by the distribution's `name`. Fields can be renamed with `with_names`, which returns the same law under new field names. 

A `NumericDistribution` is a `Distribution` whose draws are numeric, so it carries a `NumericEventTemplate` and can use the flat-vector machinery. 

```python
class Distribution[T](Tracked, Annotated):
    def __init__(self, name: str, event_template: EventTemplate) -> None: ...

    @property
    def event_template(self) -> EventTemplate: ...   # fixed at construction
    @property
    def event_shape(self) -> tuple[int, ...]: ...    # defined only when a draw is a single array

    def with_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # rename event fields; keys resolve as for NamedTree.with_names, and the law is unchanged

class NumericDistribution(Distribution): ...   # marker: numeric draws, carries a NumericEventTemplate
```

**The distribution value specification.** The `DistributionSpec(event_template)` class extends the value spec to include a `Distribution` as a valid value (with the given `event_template`). Hence, it is possible to define random measures (distributions over distributions). A draw from a random measure is therefore a `Record` whose leaf value is a `Distribution`, wrapped like any other draw. 

```python
DistributionSpec(event_template: EventTemplate)
```

### Rationale

Including a `Distribution` class is necessary to satisfy `C1 – Uniform interface to distributions and values`. It carries its draw schema rather than re-inferring it at each step to satisfy `D5 – Explicit, carried structure`, and its operations are pure to satisfy `C2 – Functional interface over immutable objects` and, when it is array-backed, differentiable end-to-end (`D6 – Differentiability where possible`).

### Notes

- *Field views.* `d["x"]` returns a view of the marginal of field `x`. Sibling views co-sample from one parent draw, so correlation between them is preserved. Every distribution carries an `event_template` with at least one named field, so the field interface is available on every distribution, however it was constructed.

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
    def _expectation(self, f: Callable[[T], Array]) -> Array: ...   # closed-form E[f(X)]

@runtime_checkable
class SupportsConditioning(Protocol):
    def _condition_on(self, given: Any, /, **kwargs: Any) -> Distribution: ...   # the conditional law given fixed values

@runtime_checkable
class SupportsMarginals(Protocol):
    def _marginal(self, path: str | tuple[str, ...]) -> Distribution: ...   # the detached marginal of a field or field group
```

Here `Key` is a PRNG key and `ArrayLike` an array-or-scalar input. `_cov` and `_quantile` require a numeric draw, with `_cov` ranging over the *flattened* draw, while `_mean` and `_variance` are event-typed and open to any event type that supports them. For a random measure, a draw's log-density is itself random, so the `SupportsRandom*LogProb` capabilities take no value and return the law of the log-density function, the random function `x ↦ log D(x)` for `D ~ M`.

### Rationale

Making each operation a *capability* rather than a base-class method follows `D3 – Capability-based operations`. Because support is structural (tested by `isinstance(dist, SupportsX)`, not subclassing), a distribution gains an operation just by implementing its method, and a wrapper (a view, a transform) exposes exactly the capabilities of whatever it wraps.

## III.8 — `ConditionalDistribution`

### Contract

A `ConditionalDistribution[S, T]` is a *probability kernel* `K : S → P(T)` — a family of distributions p(· | s) indexed by a *conditioning value* `s : S`. Supply a value for what it conditions on and it yields an ordinary `Distribution` over what it produces. A `ConditionalDistribution` is not a `Distribution` and does not inherit from one: it has no marginal law, so `sample` / `log_prob` / `mean` do not apply to it unconditionally. Rather, the two are siblings sharing a capability vocabulary.

A `ConditionalDistribution` carries two schemas: a `given_template` (the `EventTemplate` of the conditioning value `S`) and an `event_template` (the schema of one produced draw `T`). They lift a `FunctionSpec`'s input/output from values to distributions, and their field names must be disjoint. For example, a Markov kernel (where `S = T`) uses names like `state → next_state` rather than `state → state`, for the same reason we write `K(x, dy)` rather than `K(x, dx)`. `with_names` renames fields across both templates, returning the same kernel under new field names.

Users never call a method on the `ConditionalDistribution`. Instead, they use the existing ops: `condition_on(K, s)` evaluates it (exact, no inference) to a `Distribution`, and `sample(K, given=s)` / `log_prob(K, y, given=s)` / `mean(K, given=s)` are the fused conditional paths, with the invariant `op(K, given=s) == op(condition_on(K, s))`. Conditioning on only a subset of given fields *curries* to a smaller `ConditionalDistribution` view, while conditioning on a *distribution* over the given yields the predictive mixture `∫ K(s, ·) μ(ds)`.

```python
class ConditionalDistribution[S, T](Tracked, Annotated):
    def __init__(self, name: str, given_template: EventTemplate, event_template: EventTemplate) -> None: ...
        # given before event, as in FunctionSpec
    @property
    def given_template(self) -> EventTemplate: ...
    @property
    def event_template(self) -> EventTemplate: ...
    def _condition_on(self, given: S, /, **kwargs) -> Distribution[T]: ...   # the required primitive: the law K(given, ·)

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

**The numeric special cases.** A `ConditionalDistribution` has *two* templates, and either can be numeric, so the single `Numeric` prefix becomes positional: `Numeric` before `Conditional` marks the **given** side numeric, `Numeric` before `Distribution` marks the **event** side numeric, and `FullyNumeric*` marks both. Each is a marker only, adding no operations of its own (mirroring `NumericDistribution`).

```python
class ConditionalNumericDistribution(ConditionalDistribution): ...   # event numeric: every K(s, ·) is a NumericDistribution
class NumericConditionalDistribution(ConditionalDistribution): ...   # given numeric: the conditioning value is a numeric vector
class FullyNumericConditionalDistribution(
        NumericConditionalDistribution, ConditionalNumericDistribution): ...   # both sides numeric
```

**The conditional distribution value specification.** The `ConditionalDistributionSpec(given_template, event_template)` class extends the value spec to include a `ConditionalDistribution` as a valid value, analogously to `DistributionSpec`:

```python
ConditionalDistributionSpec(given_template: EventTemplate, event_template: EventTemplate)
```

### Rationale

Applying a `ConditionalDistribution` to a conditioning value returns a `Distribution`, which ensures `D4 – Closed system of objects under operations` is satisfied. A `ConditionalDistribution`'s capabilities are the `Distribution` capabilities shifted by one conditioning argument (`D3 – Capability-based operations`), so a single operation vocabulary applies to conditional distributions too, under the rule that *`Distribution` and `ConditionalDistribution` behave as similarly as possible*. As with `Distribution`, a concrete `ConditionalDistribution` family derives both templates from its parameters and passes them up, and the base only requires they are fixed at construction (`D5 – Explicit, carried structure`, `D7 – Single source of truth`). The capabilities use distinct `_conditional_*` method names because a `@runtime_checkable` check matches on method name alone, so reusing `_sample` / `_log_prob` would corrupt the unconditional capability checks.

## III.9 — `DistributionBatch` and `ConditionalDistributionBatch`

### Contract

A `DistributionBatch` is a `Batch` of `Distribution`s: `N` separate distributions sharing one `event_template`, indexed along a batch axis. A `ConditionalDistributionBatch` is the same construction over `ConditionalDistribution`s: `N` separate conditional distributions sharing one `given_template` and one `event_template`. They are grouped because they are the identical multiplicity wrapper over the two distribution-like base types, and the conditional one merely adds the `given_template`.

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

The generic factored (conditional) distributions `FactoredDistribution` and `FactoredConditionalDistribution`  carry an ordered list of factors, each a `Distribution` or a `ConditionalDistribution`. The dependence graph is *derived* by matching each factor's given fields against the fields produced by earlier factors, rather than stored. The joint distribution's `event_template` is the flat union of the fields its factors produce. 
In the case of the `FactoredConditionalDistribution`, conditioning values for all given fields results in a `FactoredDistribution`. Both provide the `Supports*` capabilities dynamically, based on what their factors support. For example, if all factors implement `SupportsLogProb` then so does the factored distribution. As with other generic distributions, there are also numeric specializations. 

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
- The **field interface** is available on every distribution that can sample. `d["intercept"]` returns a **view**: the field's marginal carrying a reference to its parent, so that sibling views co-sample from one parent draw and preserve correlation under broadcast. A view's detached law, and any capability beyond sampling, comes from the parent's `SupportsMarginals`, and `marginal(d, "intercept")` returns that same marginal **detached** from the parent.
- The **factor interface** is available only with `SupportsFactors`. `factor(d, "coeffs")` returns a building-block factor, keyed by factor name, which is a `Distribution` or, for a dependent edge, a `ConditionalDistribution`. There need be no factor for a given field, and no field for a given factor.

**Marginals of a joint.** Whether a marginal is exactly available depends on the factors' own marginal support and on where the target sits in the dependence graph, so the factored classes resolve `_marginal` per path rather than wholesale. The graph reduction is always exact: the target's ancestor closure yields a sub-joint of whole factors, and everything outside it integrates out for free. What remains is integrating the extra ancestor fields back out, which is exact in three cases: there are none (the target is ancestrally closed, as for a root factor or an edge-free group), the reduction lies within a single factor and delegates to that factor's own `SupportsMarginals`, or the affected factors admit closed-form integration, as when they are jointly Gaussian. On any other path `_marginal` raises, and the `marginal` operation falls back to its Monte Carlo route.

### Rationale

Factorization is an *optional capability*, `SupportsFactors`, rather than a base class. This is `D2 – Generality first`: a joint is an ordinary distribution that gains factor access by carrying the capability, instead of sitting in a parallel class tower. Keeping the field interface (part of a draw) and the factor interface (part of the construction) separate serves `D1 – Mathematical fidelity`, since the two are genuinely different in the mathematics. Independence is likewise a property of the derived graph rather than a class, so dropping any dedicated product class loses no behavior: an edge-free joint still samples its factors in parallel and conditions exactly on a field. A joint's capabilities are the intersection of its factors' capabilities: it supports an operation exactly when every factor does, so ancestral `sample` requires every factor to sample and a summed `log_prob` requires every factor to score.

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
law:  p(F_A, F_B | unmet) = p_A(F_A | F_B) · p_B(F_B | G_B)      # reads left → right
```

The result has two mathematical degrees of freedom, *conditional?* (`unmet ≠ ∅`) and *dependent?* (`bound ≠ ∅`), but only the first is a **class** distinction:

| `unmet` | result |
|---|---|
| `∅` | `FactoredDistribution` — a joint `Distribution` |
| `≠ ∅` | `FactoredConditionalDistribution` — a joint `ConditionalDistribution`, its `given_template` exactly `unmet` |

`*` returns the **most specific** class, recomputed from the *flattened* factor graph at each step. `A * B * C` builds one flat N-factor joint, with independent factors commuting and dependent ones kept in conditional-first order.

**Naming the result.** A joint is *derived*, not created by the user, so `*` **auto-derives** its name deterministically from its factors and marks it `name_is_auto`. An auto-named operand is flattened into a larger joint, while an operand pinned with `with_name` is kept as a single named factor.

```python
def __mul__(self, other: Distribution | ConditionalDistribution) -> FactoredDistribution | FactoredConditionalDistribution: ...
```

### Rationale

Reifying both degrees of freedom would force a 2×2 of joint classes. By `D2 – Generality first`, an independent product (`bound = ∅`) is just an edge-free joint, so *dependent?* is a runtime property of the derived graph and only *conditional?* names a class, giving two classes rather than four. The conditional-first order is already a valid topological listing, so composition is associative and acyclicity is automatic, with no separate graph inference.

### Notes

- *Operator coexistence.* `*` also denotes scalar scaling on some objects, such as a random function or a linear operator. The two coexist by operand-type dispatch: `Distribution` and `ConditionalDistribution` operands compose, while scalar operands scale.
- *The realigning `joint` form.* `joint(A, B, **align)` is `*` plus field renaming, for factors whose names do not line up: it is `A * B.with_names(**align)`.

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
- **Empirical.** A finite, possibly weighted, sample set: `sample` resamples, `log_prob` reads the (weighted) empirical measure, moments are sample estimates, and marginals are again empirical. Scalar or structured.
- **Transformed (pushforward).** A base distribution pushed through a function, which is what lifting a function over a distribution-valued argument produces. An invertible map keeps an exact `log_prob` by change of variables, while a general map keeps `sample` and Monte-Carlo-estimates the rest.
- **Mixture.** A convex combination of component distributions. It is also the form a dependent joint's detached `marginal` generally takes.
- **Random function.** A distribution over functions, whose event is a `FunctionSpec` leaf: a draw is a callable, and `mean` returns the mean function. A Gaussian process is the canonical case.
- **Random measure.** A distribution *over distributions*: a draw is itself a `Distribution` (a `DistributionSpec` leaf), and `mean` returns the marginalized law.

Any family can arise as the approximation of a target: what makes a result approximate (the target, the method, and the fit) is recorded in its `provenance`.

**Conditional distributions and batches stratify identically.** A `ConditionalDistribution` repeats both axes (atomic / structured / the `FactoredConditionalDistribution` joint, crossed with parametric / amortized / empirical / …), and a `DistributionBatch` is `N` of any of these. The catalog is one classification, reused across the conditional and multiplicity layers.

### Rationale

The hierarchy embodies `D2 – Generality first`: one base refined by *optional capabilities* (`SupportsFactors`, the numeric marker, each `SupportsX`) rather than a rigid class tower, so a new family slots in by implementing the capabilities it supports rather than by widening the base. The atomic-versus-joint split is a `D1 – Mathematical fidelity` distinction, since a joint genuinely offers its factors as distributions while an atomic-structured law does not, and it is carried by a capability rather than by the draw's type. Because composition is closed (`D4 – Closed system of objects under operations`), a joint re-enters the catalog as an ordinary `Distribution` for the next operation.

## III.13 — Cross-type conversion

### Contract

A distribution may have more than one representation, and an operation or backend sometimes needs a different one than the user holds. **Conversion** moves a distribution from its current type to a requested target type, resolved by the **converter registry** whose methods are *converters*.

A converter declares the source types it converts *from* and the target types it converts *to*, a cheap `check` that reports feasibility without converting, and the conversion itself. A conversion is rarely unique, so each carries a **fidelity**: `exact` for an equivalent representation, `moment-match` when only low-order moments are preserved, and `sample` for a Monte Carlo stand-in. The registry prefers higher fidelity, and a caller can name a converter or set a minimum fidelity.

```python
class ConversionMethod(Enum):
    EXACT = "exact"
    MOMENT_MATCH = "moment_match"
    SAMPLE = "sample"

@dataclass(frozen=True)
class ConversionInfo:               # the result of a converter's feasibility probe
    feasible: bool
    method: ConversionMethod | None = None

class Converter(ABC):
    def source_types(self) -> tuple[type, ...]: ...
    def target_types(self) -> tuple[type, ...]: ...
    def check(self, source, target_type: type) -> ConversionInfo: ...
    def convert(self, source, target_type: type) -> Distribution: ...

class ConverterRegistry:
    def register(self, converter: Converter) -> None: ...
    def check(self, source, target_type: type) -> ConversionInfo: ...
    def convert(self, source, target_type: type,
                method: str | None = None, min_fidelity: ConversionMethod | None = None) -> Distribution: ...

converter_registry: ConverterRegistry   # the global instance, keyed on the (source, target) type pair
```

### Rationale

Conversion makes `C3 – Computational detail hidden by default, available on demand` concrete on the distribution layer: a representation is a computational choice, so the library converts as needed and the user rarely converts by hand. Recording each conversion's fidelity keeps the approximation honest, which is `D1 – Mathematical fidelity`, since an `exact` conversion loses nothing while a `moment-match` or `sample` conversion is a stated approximation the caller can see and control. New representations interoperate by registering converters, so the set of convertible pairs grows without changing the distributions themselves (`D2 – Generality first`).

## III.14 — Constraint reparameterization

### Contract

Gradient-based and Hamiltonian inference work in an unconstrained space, so a constrained support must be reparameterized. The **constraint-to-bijector factory** maps a `Constraint` (the support an `ArraySpec` carries) to a bijector that takes `ℝⁿ` onto that support. `bijector_for(constraint)` returns the canonical bijector, and `register_bijector` plugs in a factory for a constraint type or for a specific constraint instance, with instance registrations taking precedence over type registrations.

```python
def bijector_for(constraint: Constraint) -> Bijector: ...    # the canonical map ℝⁿ → support(constraint)
def register_bijector(
    key: type[Constraint] | Constraint,
    factory: Callable[[Constraint], Bijector],
) -> None: ...
```

### Rationale

A bijector for every constraint lets inference run in an unconstrained space while a model stays stated in its natural, constrained one, which serves `D6 – Differentiability where possible`. Keeping the map open through `register_bijector` is `D2 – Generality first`: a new constrained support becomes inference-ready by registering its reparameterization, without touching the distributions that use it.

### Open points

- *Round-trip fidelity.* The forward map (bijector to support) and this inverse map (support to bijector) are not strict inverses for every constraint, so a reparameterized support can drift to a coarser one. Whether to unify the two is unsettled.
