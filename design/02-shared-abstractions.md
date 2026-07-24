# Part II — Shared Abstractions

Part II introduces the shared abstractions the rest of the library is built on: generic, type-agnostic machinery, one piece at a time, in dependency order.

## II.0 — Overview: the shared abstractions

The shared abstractions, in dependency order. Each is generic and type-agnostic, defined once and reused throughout the library:

| § | Layer | Abstraction | Role |
|---|---|---|---|
| II.1 | Structure | `NamedTree` | The named, ordered tree addressed by path that every structured object is built on, owning the leaf-keyed mapping contract and navigation. |
| II.2 | Type | `ValueSpec` / `TermSpec` / `EventTemplate` | The value specifications every field and declaration is typed by, the event templates built from them, and the storage rule: a tracked term stores its type as its spec. |
| II.3 | Identity | `TrackedTerm` / `Annotated` / `Provenance` | The name, type (spec), lineage, and annotations an object carries beyond its raw representation. |
| II.4 | Multiplicity | `Batch` | The generic multiplicity axis: an indexed collection of *separate* objects, distinct from one object over a structured space. |
| II.5 | Dispatch | dispatch & registries | Registry-based multiple dispatch that selects an implementation by the types involved. The shared mechanism behind converters, inference selection, and bijector factories. |

## II.1 — `NamedTree`

### Contract

Values in ProbPipe are represented as named, ordered trees.  ProbPipe uses the following standardized **terminology** to refer to components of these trees:
- A **field** is one named leaf — a single object in the collection.
- A **path** is a `/`-joined sequence which can address either a field or an interior node.
- A **key** is a path that specifically addresses a *field*. Every key is a path but a path for an interior node is not a key.
- A **child** of a node is an entry directly under that node.
- The **canonical order** of a tree is a depth-first walk visiting children in insertion order.

Naming, addressing, traversal, and structure-preserving transforms are defined once in the `NamedTree` class. Specifically, a `NamedTree` implements the `Mapping` interface, with keys exactly its leaf paths, in canonical order, so `keys()`, `values()`, `items()`, `len`, `in`, and `[]` all agree on that one key set, as for a plain `dict`. A leaf path may be equivalently written as a string (`"a/b"`) or a tuple (`("a", "b")`). Since interior nodes are *not* keys, `[]` raises on an interior path. Interior nodes are instead accessed via either the one-level `children` view or `at_path`, which can access any leaf or subtree. So, for example, the invariants  `x.children["a"].children["b"] == x.at_path("a", "b") == x.at_path("a/b")` hold. Sibling names are distinct, so every path identifies at most one node. Distinct subtrees may reuse a name, as in `a/c` and `b/c`, and a bare name is then ambiguous on its own.

```python
class NamedTree[L]:
    # mapping interface — keyed by FIELD path
    def __getitem__(self, key: str | tuple[str, ...]) -> L: ...
    def __contains__(self, key: str | tuple[str, ...]) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...  # field paths
    def __len__(self) -> int: ...             # field count
    def keys(self) -> Iterable[str]: ...      # field paths, canonical order
    def values(self) -> Iterable[L]: ...
    def items(self) -> Iterable[tuple[str, L]]: ...

    # tree navigation — ranges over ALL paths
    def at_path(self, *path: str) -> L | Self: ...    # at_path("a", "b") == at_path("a/b")
    @property
    def children(self) -> Mapping[str, L | Self]: ...
    def is_field(self, path: str | tuple[str, ...]) -> bool: ...
    @property
    def is_multi_field(self) -> bool: ...   # True when the tree holds more than one field

    # structure-preserving transforms — return the same family
    def with_path_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # rename nodes, old -> new; keys are paths, or bare names when unambiguous
    def map(self, f: Callable[[L], L], /, *args, **kwargs) -> Self: ...
    def map_with_keys(self, f: Callable[[str, L], L], /, *args, **kwargs) -> Self: ...
    def replace(self, path: str | tuple[str, ...], leaf: L | Self) -> Self: ...
    def merge(self, other: Self) -> Self: ...              # a key collision raises
    def without(self, path: str | tuple[str, ...]) -> Self: ...

    # serialization (export; the constructor reads a nested mapping back)
    def to_nested_dict(self) -> dict[str, Any]: ...

    @classmethod
    def _node_type(cls) -> type[Self]: ...         # the family's own node type
    @classmethod
    def _leaf_type(cls) -> type | UnionType: ...   # the family's declared leaf type
```

### Rationale

Using named paths is necessary to satisfy `C5 – Naming for unambiguous meaning`. Housing the collection contract in one shared class ensures `D7 – Single source of truth` is satisfied — the type- and value-level structures built on it cannot drift apart on how a field is named or a path is resolved.
### Notes

- *Same-family closure.* Every navigator and transform returns the same family, enforced by `_node_type()`.
- *Leaf type versus node type.* The parameter `L` declares the leaf type, the only axis on which tree families differ: it is what `values()`, `[]`, and `map` traffic in. The node type is not a second parameter, since interior nodes are always instances of the family's own class, which is what `_node_type()` reports. The runtime partition therefore uses the node type alone: a field value is an interior node when it is an instance of `_node_type()`, and a leaf otherwise. Validation is handled once in `NamedTree` rather than by each family: construction checks every leaf against the family's declared leaf type, reported by `_leaf_type()`, so a malformed tree fails at construction rather than at first navigation. A family whose leaves are arbitrary values declares `object`, making the check vacuous.
- *Navigation yields views.* `children`, `at_path`, and `[]` return a subtree or leaf that is a *view* into the same underlying store, derived on demand rather than copied out.
- *Mappings are never leaves.* Construction materializes a mapping-valued field into a subtree, so a nested mapping and its `to_nested_dict` export round-trip faithfully through the constructor — there is no separate nested-dict reader.
- *Mapping protocol, not `Mapping` ABC.* `NamedTree` implements the `Mapping` interface but is deliberately **not** a `collections.abc.Mapping` instance: construction detects tree structure with `isinstance(value, Mapping)`, which must capture a mapping-valued leaf to materialize while leaving a nested tree to nest as a child.
- *Abstract substrate.* `NamedTree` provides no constructor logic and is not directly instantiable; each concrete family's constructor owns storage and validation policy.
- *Bare-name reference.* `with_path_names` renames by `old="new"` pairs, where a key may be a bare name instead of a full path: a bare name resolves to the unique node so named and raises when the tree contains it more than once. Keyword pairs therefore cover the common case, while the positional mapping form addresses any path, as in `with_path_names({"group1/mu": "loc"})`. The `Mapping` interface itself is untouched, with `[]` and `keys()` keyed by full path.

## II.2 — Value specifications & event templates: `ValueSpec`, `TermSpec`, `EventTemplate`

### Contract

An `EventTemplate` is a `NamedTree` that defines the *shape of one event* — a draw, a stored datum, and so on. Each leaf is a `ValueSpec`: it says what the single value at that path looks like, and it answers `is_valid` on a candidate value.

Value specs come in two families. A **raw-value spec** types a leaf holding a plain value that names no ProbPipe kind — a numeric array (`ArraySpec`) or an opaque Python object (`OpaqueSpec`). A **term spec** types a leaf holding a tracked ProbPipe term: a `Record`, `Distribution`, `ConditionalDistribution`, or `Function`, one per kind. A tracked term is, at bottom, still one value that can occupy a leaf, so a term spec *is* a value spec: `TermSpec` subclasses `ValueSpec` as a marker, and the concrete class — `RecordSpec`, `DistributionSpec`, and so on — is the kind. Two consequences follow. `is_valid` stays declared once, on `ValueSpec`; and anywhere a leaf is accepted, a term spec is accepted unchanged.

A term spec plays one of two roles, fixed by its position.

- **As data.** At a named leaf of a template, a term spec types a field that holds a term. A `Distribution` stored inside a record is a leaf value.
- **As result.** As an output declaration, a term spec states that the result *is* a term of that kind. A fitted mapping declares that it returns a `Function` this way.

A `Distribution`'s event declaration is an output declaration: it types what `sample` returns. The two roles therefore settle every draw kind. `Distribution(name, DistributionSpec(t))` declares a random measure; its draws are `Distribution`s. `Distribution(name, EventTemplate(x=DistributionSpec(t)))` declares a record law; its draws are `Record`s with a distribution-valued field `x`. The two spaces are isomorphic, but the draw kinds differ, and the declaration alone decides which — never the runtime type of `_sample`'s return. A result of the wrong kind raises the kind error; a schema mismatch raises its own.

An `EventTemplate` is not itself a spec. It is the *schema*, the structure that indexes a kind, and it is never one of its own leaves. A declaration is *stored* as a spec: a bare `EventTemplate` is accepted wherever a record declaration is meant, as a convenience, and construction wraps it as `RecordSpec(template)`. The two forms denote the same space; after construction only the spec remains, so the declared kind is simply the stored spec's class.

The same storage rule holds for the tracked types themselves. Each carries the spec of its kind — a `Record` its `RecordSpec`, a `Distribution` its `DistributionSpec`, a `ConditionalDistribution` its `ConditionalDistributionSpec`, a `Function` its `FunctionSpec` — as the single stored source of its type. The slot itself is `TrackedTerm.spec` (II.3), declared once on the tracked base, and each kind narrows it to its own spec class. Each type exposes convenience accessors for its own spec's properties — a `Record` its `event_template`, a `Function` its `input_template` and `output_spec`, a `Distribution` its `event_spec`, a `ConditionalDistribution` its `given_template` and `event_spec` — views on the one stored object, so they cannot disagree with it or with each other.

```python
class ValueSpec(ABC):               # a leaf value; is_valid declared once, here
    @abstractmethod
    def is_valid(self, value: Any) -> bool: ...

# --- raw-value specs: a leaf holding a plain value, no kind ---
class ArraySpec(ValueSpec):  # a numeric array leaf
    shape: tuple[int | str, ...]   # a str names a symbolic dimension
    dtype: DType
    support: Constraint

class OpaqueSpec(ValueSpec):  # the fallback spec; is_valid accepts any non-mapping value
    meta: Hashable

# --- term specs: a leaf holding a tracked term; the concrete class is the kind ---
class TermSpec(ValueSpec): ...      # marker; adds nothing, is_valid inherited

class RecordSpec(TermSpec):  # a Record; is_valid accepts a matching Record
    event_template: EventTemplate

class FunctionSpec(TermSpec):  # a callable; is_valid accepts any callable
    input_template: EventTemplate | None   # None: that side's structure unspecified
    output_spec: TermSpec | None           # the output declaration, stored as a spec;
                                           #   construction wraps an EventTemplate as RecordSpec(template)
# DistributionSpec (III.6) and ConditionalDistributionSpec (III.8) are the other two term specs.

class Constraint(ABC):              # an array support, carried by ArraySpec
    @abstractmethod
    def check(self, value: ArrayLike) -> Array: ...   # elementwise membership
    # constraints compare and hash by value, so an instance can serve as a registry key
```

**Placement.** `TermSpec` lives beside `ValueSpec` in `core/_specs.py`: it is a marker over the spec hierarchy, tied to no one kind, and the tracked base that houses the spec slot (II.3) lives in `core/`, which cannot depend on the value layer. `EventTemplate` and `Constraint` sit beside them in `core/_event_template.py` and `core/_constraints.py`. Each concrete term spec still lives with its kind's base type: `RecordSpec` with `Record` and `NumericRecord` in `values/_record.py`, `FunctionSpec` with `Function` in `values/_function_base.py`, `DistributionSpec` in `distributions/_distribution.py`, and `ConditionalDistributionSpec` in `distributions/_conditional.py`.

`RecordSpec(τ)` and the template `τ` denote the same space; the tag, not the denotation, fixes the kind and the operations. Two rules then govern record-valued positions. **Raw mappings are never leaves**: a raw `dict` flattens to nested tree structure. **A tracked term as a field value stays a term-valued leaf**, at every kind, so its identity — name, provenance, capabilities — is never dropped implicitly. Both follow the wrap boundary (V.0).

A `FunctionSpec` types a callable by its input and output structure, either side optional: `None` leaves it unspecified, so a bare `FunctionSpec()` describes any callable. The input side is an explicit `EventTemplate`, a single-field signature written out as `FunctionSpec(EventTemplate(x=...), EventTemplate(out=...))`, so a function's field names are caller-chosen and meaningful, matching `DistributionSpec`. The output side, `output_spec`, accepts any `TermSpec`, so a function may declare a term-valued result: a `Function` returning a `Distribution`, or a fitted mapping declared `Fun`/`Cond`, with a record output the common case; an `EventTemplate` output wraps to `RecordSpec` at construction, so the stored side is always a spec. Validity is callability alone — the value-layer specs stay callable-generic, admitting any callable as a leaf value — and it is the spec's identity as a `FunctionSpec`, not `is_valid`, that tells the wrap boundary to coproject a raw callable *result* into a `Function`. The two sides are otherwise independent, so a callable may map a space to itself or between two different spaces.

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

## II.3 — Identity, type & metadata: `TrackedTerm`, `Annotated`, `Provenance`

### Contract

Identity, type & metadata is the cross-cutting layer that lets any object carry, alongside its raw representation, four things: a **name** (what the object is called), a **spec** (the declaration of its type, II.2), a **provenance** (how it was produced), and free-form **annotations** (auxiliary information supplied by the user or an algorithm). The structure is provided by two mixins: `TrackedTerm` (name, spec, and provenance) and `Annotated` (annotations). Every first-class object, the kind an operation consumes and produces, must be a `TrackedTerm`, while structural helpers such as templates and specs are not. We call any such object a **tracked term**: a value, distribution, conditional distribution, linear operator, or batch that carries a name, its spec, and provenance.

Annotation metadata is a free-form mapping:

```python
class Annotated:
    annotations: Mapping[str, Any] | None
```

A tracked term's name must be provided by the user when constructed explicitly (as the required first argument to the constructor). When an operation produces an object, it must provide a meaningful, deterministic name derived from its inputs. The `name_is_auto` flag records which, because the two behave differently: a structure-changing transform re-derives an auto-derived name from its result, and composition into a larger object may rename it again, while a user-given name is preserved. A *nested* object (i.e., a sub-object of a `NamedTree`) takes its name from the field key it sits under. For example, a sub-object reached at `parameters` is itself named `parameters`. `with_name` renames the object itself, unlike `NamedTree.with_path_names`, which renames the fields within it. A name is a human label rather than an identity: nothing resolves an object by name, derived names need no escaping scheme, and two objects may share a name wherever field uniqueness does not force distinctness. The provenance of a tracked term stores pointers to descriptors of the parent objects that created it, along with the operation. Optionally, it can also provide references to the parents themselves.

The `spec` slot is the term's type, stored once (II.2). Each kind narrows it to its own spec class — a `Record` to `RecordSpec`, a `Distribution` to `DistributionSpec`, and so on — and exposes convenience accessors for its properties (Part III).

```python
class TrackedTerm:
    name:         str
    name_is_auto: bool
    spec:         TermSpec                       # the single stored source of the term's type (II.2)
    provenance:   Provenance | None              # write-once via with_provenance(...)
    def with_name(self, name: str) -> Self: ...  # shallow copy with name_is_auto = False
    def with_provenance(self, p: Provenance) -> Self: ...

class Provenance:
    operation: str                       # the operation that produced the object
    parents:   tuple[ParentInfo, ...]    # descriptors of the tracked inputs
    controls:  Mapping[str, Any]         # the resolved controls: PRNG key, sample count, selected method, ...
    inputs:    Mapping[str, ParentInfo]  # plain (untracked) arguments, keyed by parameter name

class ParentInfo:
    type_name:   str
    name:        str
    fingerprint: str            # best-effort content hash
    fingerprint_is_weak: bool   # True when the fingerprint is only object identity
    parent:      Any | None     # optional reference to original parent
```

A provenance records everything its operation resolved: the tracked parents, the plain arguments, and the **controls** the run actually used, the PRNG key, sample counts, and any selected method among them, so a result can be reproduced from its record. Fingerprints are best-effort, in tiers: a content hash for arrays and records, the qualified name and code hash for a closure-free callable, and object identity otherwise, with `fingerprint_is_weak` marking that last tier.

### Rationale

`TrackedTerm` serves the two non-mathematical principles: `C5 – Naming for unambiguous meaning` and `C6 – Traceable and reproducible workflows`. Housing the spec on the tracked base is `D7 – Single source of truth` for a term's type: one slot, declared once, that every kind's accessors are views on. The guarantee behind `C6 – Traceable and reproducible workflows` is a single rule: **every object a ProbPipe operation natively returns is a tracked term** (whether or not it is also `Annotated`), so the provenance chain is never broken. Recording the resolved controls, not just the parents, is what turns traceability into reproducibility: re-running the recorded operation on the recorded inputs with the recorded controls reproduces the result. Auto-derived names keep every intermediate object identifiable without forcing the user to label it (`C5 – Naming for unambiguous meaning`). Because identity and metadata are orthogonal to *what* an object is mathematically, they are defined uniformly across classes.

## II.4 — `Batch`

### Contract

A `Batch` is the generic `TrackedTerm` nd array of shape `batch_shape` that holds objects of a common element type. `batch_shape` is nonempty: a batch has at least one batch axis. It could also be `Annotated` if applications for it arise. A concrete batch implementation must specify how to store the elements. Since a batch is a *collection* of its elements, `len` / `iter` / `batch_shape` / `batch_size` operate only on the batch axes. The `batch_*` names are kept deliberately rather than a numpy-style `.shape` / `size`, which could ambiguously cover both the batch axes and the per-element content. A concrete batch implementation adds whatever its element type affords in that element's own section — including, where useful, indexing into the elements' fields, since `[]` dispatches unambiguously on the key type.

```python
class Batch[E](TrackedTerm):
    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    @property
    def batch_size(self) -> int: ...                    # total element count, prod(batch_shape)
    @property
    def axis_groups(self) -> tuple[tuple[int, ...], ...]: ...   # batch_shape tiled into levels, outermost first
    @property
    def level_names(self) -> tuple[str, ...]: ...       # one name per level, aligned with axis_groups
    def with_level_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # shallow copy renaming levels old -> new; shapes and elements unchanged, as with_path_names is for fields
    def __len__(self) -> int: ...                       # leading-axis size, batch_shape[0]
    def __iter__(self) -> Iterator[E | Self]: ...       # over the leading batch axis
    def __getitem__(self, index: Any) -> E | Self: ...  # returns a view of either an element or a sub-batch
    def select_at(self, **levels: int | slice | None | tuple[int | slice | None, ...]) -> E | Self: ...
    # index by named level (a view); unnamed levels kept whole, None means the whole axis (:)
```

**Axis groups.** A batch's axes are partitioned into ordered **levels**. `axis_groups` tiles `batch_shape` into contiguous groups, outermost level first, and `batch_shape` stays their flat concatenation, so anything stated over `batch_shape`, flat vectorization above all, applies to a multi-level batch unchanged. A single-level batch has one group holding all its axes. `len`, `iter`, and indexing operate on the outermost level, and an element of a multi-level batch is the inner-level batch, as a view. Nesting needs no dedicated classes: a batch is itself a tracked term, so a batch whose elements are batches is already admitted, and grouped storage presents the levels as views into one store.

**Level names.** Each level carries a name, listed in order by `level_names`. A name is auto-derived by the operation that produces the level, and `with_level_names` repins it: a shallow copy changing only the names. Names are unique within a batch: an operation that would mint a duplicate appends the smallest free integer suffix, so nested sampling yields `draw`, then `draw2`, and a rename onto an existing name raises. Operations align batched operands by level name, and two levels meant to correspond under different names are lined up by renaming one. Level names are independent of the field names within each element.

**Element identity.** An element view of a batch derives its name as `name[i]` from the batch's own name, marked `name_is_auto`, with provenance recording the indexing, and the elements of nested levels compose the scheme, as in `name[i][j]`. A batch whose elements are bare values yields bare elements, which carry no identity to derive.

**Selecting by level.** `select_at(**levels)` indexes a batch along its named levels and returns a view, the by-name counterpart of positional `[]`. The name is `select_at` rather than `select`, which is the field-splatting selector on `Record` (III.3): a batch of records can sensibly carry both, so the two keep distinct names. Each indexer is an integer, a slice, `None`, or a tuple of these addressing the level's axes in order, where an integer drops its axis and a slice or `None` keeps it, `None` meaning the whole axis as `:` does. A level spanning several axes takes one indexer per axis, and a shorter indexer fills the leading axes and leaves the rest whole, so a scalar `draw=i` on a two-axis `draw` level means `draw=(i, None)`. Selecting an entire single-axis level by an integer removes it, yielding the inner batch or element just as positional indexing and iteration do, while a level left unnamed is kept whole. This parallels xarray's `isel`, with `level_names` in the role of xarray dimension names; there is no label-based counterpart, since batch levels carry no coordinate labels.

### Rationale

`Batch` is necessary to satisfy `D1 – Mathematical fidelity` by ensuring how many objects there are stays separate from what one object contains. The level structure extends the same fidelity to collections of collections: how many objects sit at each level is a mathematical distinction, so `N` laws with `S` draws each are `(N,)` of `(S,)` rather than one anonymous `(N, S)`. Naming the levels is `C5 – Naming for unambiguous meaning` on that axis, letting a user say which multiplicity is which and letting operations align batches by meaning rather than by position. An operation broadcasts across a batch by mapping over its elements, so a batch supports an operation exactly when its elements do (`D3 – Capability-based operations`). When those elements are array-backed and claim differentiation, the mapping is vectorized and differentiable end-to-end (`D6 – Differentiability as a capability`). To satisfy `D7 – Single source of truth`, indexing or iterating yields a *view*, the levels of a multi-level batch included.

## II.5 — Dispatch and registries

### Contract

Some operations have many possible implementations, and which one applies depends on the *types* of the objects involved rather than on an object's own class. A **dispatch registry** holds those implementations as named methods and selects one for a given call.

Each **dispatch method** declares a unique `name`, the types it applies to via `supported_types`, a `check` function that probes feasibility without doing significant computation, an `execute` function that performs it, and a `priority` that orders auto-selection. Dispatch is by argument type: a `UnaryDispatchRegistry` keys on the first argument's type, and a `BinaryDispatchRegistry` on the first two. The registry takes the matching methods in priority order and runs the first whose `check` reports feasible. Within one priority, the method whose declared types sit closest to the argument's class in method-resolution order wins, and any remaining tie falls to registration order. A caller can bypass auto-selection and name a method with `method="..."`. New methods are added by registration.

```python
class BaseDispatchMethod(ABC):
    name: str
    priority: int   # defaults to opt-in-only (0)

    @abstractmethod
    def check(self, *args, **kwargs) -> MethodInfo: ...
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any: ...

class UnaryDispatchMethod(BaseDispatchMethod):    # still abstract
    @abstractmethod
    def supported_types(self) -> tuple[type, ...]: ...
class BinaryDispatchMethod(BaseDispatchMethod):   # still abstract
    @abstractmethod
    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]: ...   # (left, right) types

class MethodInfo:
    feasible:    bool
    method_name: str
    description: str

class BaseDispatchRegistry[M: BaseDispatchMethod](ABC):
    # the public interface is concrete; arity subclasses supply key extraction and matching
    def register(self, method: M) -> None: ...
    def execute(self, *args, method: str | None = None, **kwargs) -> Any: ...   # auto-select, or run the named method
    def check(self, *args, method: str | None = None, **kwargs) -> MethodInfo: ...
    def list_methods(self) -> list[str]: ...                           # names, in selection order

class UnaryDispatchRegistry[M: UnaryDispatchMethod](BaseDispatchRegistry[M]): ...    # keys on one argument's type
class BinaryDispatchRegistry[M: BinaryDispatchMethod](BaseDispatchRegistry[M]): ...  # keys on the first two
```

A single **catalog** makes every registry discoverable. It provides a list of registries, their entries with their priorities, and a one-line description, so a user can see which entries exist and how a given call will resolve. An **entry** is one registered item within a registry; the catalog uses this generic term rather than *method* because it spans registries whose items are not all type-dispatched methods. A registry can be added to the catalog if it implements `SupportsRegistryCataloging`. Satisfying the protocol is structural, while membership requires an explicit `register`.

```python
@dataclass(frozen=True)
class EntrySummary:
    name: str
    priority: int | None
    supported_types: tuple[Any, ...] = ()
    description: str = ""
    module_path: str = ""
    @property
    def is_opt_in_only(self) -> bool: ...

@dataclass(frozen=True)
class RegistryInfo:            # the catalog's per-registry record
    name: str                  # unique within the catalog
    description: str           # one line
    kind: str                  # e.g., "dispatch", "factory", "converter"
    entry_count: int

@runtime_checkable
class SupportsRegistryCataloging(Protocol):
    name: str
    description: str
    kind: str
    def entry_summaries(self) -> list[EntrySummary]: ...
    def describe_entry(self, name: str) -> EntrySummary: ...

class RegistryCatalog:
    def register(self, registry: SupportsRegistryCataloging) -> None: ...   # empty or duplicate names raise
    def names(self) -> list[str]: ...
    def list(self) -> list[RegistryInfo]: ...
    def __getitem__(self, name: str) -> SupportsRegistryCataloging: ...     # get the registry
    def __contains__(self, name: object) -> bool: ...                       # check if a registry exists
    def describe(self, name: str) -> str: ...                               # a readable summary of one registry
registry_catalog = RegistryCatalog()  # the global instance
```

### Rationale

A registry is how `C3 – Computational detail hidden by default, available on demand` and `D3 – Capability-based operations` reach operations whose implementation cannot be chosen from a single object alone. The `check` probe keeps auto-selection safe, while `method="..."` leaves the choice in the user's hands. New implementations join by registering a method, so the supported set grows without touching the call sites that use it, which is `D2 – Generality first`. Gathering every registry under one catalog serves `D7 – Single source of truth`: there is one place to see which implementations exist and how a call resolves.
