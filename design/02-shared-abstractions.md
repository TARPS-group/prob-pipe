# Part II — Shared Abstractions

Part II introduces the shared abstractions the rest of the library is built on.

| § | Layer | Abstraction | Role |
|---|---|---|---|
| II.1 | Structure | `NamedTree` | The named, ordered tree addressed by path that every structured object is built on, owning the leaf-keyed mapping contract and navigation. |
| II.2 | Type | `TermSpec` | The term-specification base every field and declaration is typed by, with the symbolic-dimension protocol. Each kind's concrete spec is defined beside the kind it describes. |
| II.3 | Declarations | `InputSpec` / `OutputSpec` | The input and output declarations of the map-like kinds: named slots in, one named produced term out. |
| II.4 | Numeric value | `Numeric` / `NumericSpec` / `Constraint` | The flat-vector interface the numeric kinds share, its spec-side mixin, and the elementwise support constraint. |
| II.5 | Identity | `TrackedTerm` / `Provenance` | The name, type (spec), lineage, and annotations an object carries beyond its raw representation, and its `raw()` door back to that representation. |
| II.6 | Multiplicity | `Batch` | The generic multiplicity axis: an indexed collection of *separate* objects, distinct from one object over a structured space, with its `BatchSpec`. |
| II.7 | Dispatch | dispatch & registries | Registry-based multiple dispatch that selects an implementation by the types involved. The shared mechanism behind converters, inference selection, and bijector factories. |

## II.1 — `NamedTree`

### Contract

Structured values in ProbPipe are represented as named, ordered trees. The following standardized terminology is used to refer to components of these trees:
- A **field** is one named leaf — a single object in the collection.
- A **path** is a `/`-joined sequence which can address either a field or an interior node.
- A **key** is a path that specifically addresses a *field*. Every key is a path but a path for an interior node is not a key.
- A **child** of a node is an entry directly under that node.
- The **canonical order** of a tree is a depth-first walk visiting children in insertion order.

Naming, addressing, traversal, and structure-preserving transforms are defined once in the `NamedTree` class. Specifically, a `NamedTree` implements the `Mapping` interface, with keys exactly its leaf paths. A leaf path may be equivalently written as a string (`"a/b"`) or a tuple (`("a", "b")`). Since interior nodes are *not* keys, `[]` raises on an interior path. Interior nodes are instead accessed via either the one-level `children` view or `at_path`, which can access any leaf or subtree. So, for example, the invariants `x.children["a"].children["b"] == x.at_path("a", "b") == x.at_path("a/b")` hold, and navigation returns *views* into the same underlying store, derived on demand rather than copied out. Sibling names are distinct, so every path identifies at most one node; distinct subtrees may reuse a name, as in `a/c` and `b/c`.

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
    # rename or move nodes, old -> new; keys are paths, or bare names when unambiguous;
    # a new name may itself be a path, which moves the node there
    def map(self, f: Callable[[L], L], /, *args, **kwargs) -> Self: ...
    def map_with_keys(self, f: Callable[[str, L], L], /, *args, **kwargs) -> Self: ...
    def replace(self, path: str | tuple[str, ...], leaf: L | Self) -> Self: ...
    def merge(self, other: Self) -> Self: ...              # a key collision raises
    def without(self, path: str | tuple[str, ...]) -> Self: ...

    # serialization (export; the constructor reads a nested mapping back)
    def to_nested_dict(self) -> dict[str, Any]: ...

    @classmethod
    def _node_type(cls) -> type[Self]: ...         # the family's own node type: every navigator and transform returns the same family
    @classmethod
    def _leaf_type(cls) -> type | UnionType: ...   # the family's declared leaf type
```

The parameter `L` declares the leaf type — the one axis on which families differ, and what `values()`, `[]`, and `map` traffic in — while interior nodes are always the family's own class. Implementations should check leaves against the declared leaf type at construction.

The `with_path_names` method renames or moves nodes by `old="new"` pairs. A key may be a bare name, resolving to the unique node so named and raising on ambiguity — keyword pairs cover the common case, while the positional mapping form addresses any path. A target may itself be a path: `with_path_names({"group/mu": "mu"})` promotes the field to the top level and `{"mu": "group/mu"}` demotes it under `group`, creating intermediate nodes as needed and dissolving an interior node a move empties, since a tree holds no empty subtrees. All substitutions apply simultaneously, so sources resolve against the original tree; this way, swaps and simultaneous ancestor–descendant moves are well-defined. Every target is checked against the result, where a collision raises as a rename onto an existing sibling does; a move into the moved node's own subtree raises, as do two targets where one is a prefix of the other. Ordering stays deterministic: an in-place rename keeps its position, and a moved node appends at the end of its new parent's children.

### Rationale

Using named paths is necessary to satisfy `C5 – Naming for unambiguous meaning`. Housing the collection contract in one shared class ensures the type- and value-level structures built on it cannot drift apart on how a field is named or a path is resolved, to satisfy `C1 — Uniform interface to functions, distributions, and values`.

### Notes

- *Mappings are never leaves.* Construction should materialize a mapping-valued field into a subtree. The need for this mapping check is why `NamedTree` implements the `Mapping` interface without registering as a `collections.abc.Mapping`, ensuring a nested mapping and its `to_nested_dict` export round-trip through the constructor.

## II.2 — Term specifications: `TermSpec`

### Contract

A **term specification** (``term spec'') describes the available typing information for a mathematical term and validates whether an object satisfies the term's type constraints.

The specs partition into the **base kinds** and the **batch kinds**. Every base kind has exactly one term spec, one **tracked class**, and one **batch form**. Since a batch of batches is a batch, `BatchSpec` is the one spec whose batch spec is the same.

**Symbolic dimensions.** A dimension size for a numeric value may be an integer or a **named symbolic dimension**: a name that fixes a dimension's identity while deferring its size. A spec with any unbound name is said to be **polymorphic**, and one with none is said to be **concrete**. Three operations manage dimensions, and every spec carries its own: it *reports* the names still unbound, it *substitutes* explicit sizes for names, and it *binds* names by unification against a value, reading the sizes off the data. Substitution and binding return a new spec, so refinement is monotone. Until every name is bound an operation that needs sizes raises an error.

Validation and the dimension protocol together are the base API:

```python
class TermSpec(ABC):
    @abstractmethod
    def is_valid(self, value: Any) -> bool: ...      # structural validity: kind, rank, dtype

    # the symbolic-dimension protocol: report, substitute, bind
    @property
    def free_dims(self) -> frozenset[str]: ...       # report: the unbound symbolic dimensions
    @property
    def is_concrete(self) -> bool: ...               # True when free_dims is empty
    def with_dims(self, **sizes: int) -> Self: ...   # substitute explicit sizes; a new spec
    def bind_dims_from_value(self, value: Any) -> Self: ...   # bind by unification against a value
```
s
A spec accepts a value in **either presentation** — the raw representation or the tracked term of the spec's kind — with construction normalizing to the stored form. The rule also holds in the opposite direction: wherever a raw value is accepted by a function, its tracked form is accepted too; however, an implementer method's *arguments* arrive already normalized to the implementer type, so an implementation only needs to handle one presentation. An implementer may returns either presentation: the boundary keeps the kind and mints the result's identity afresh either way.

### Rationale

One `is_valid` contract across the kinds keeps validation uniform (`C1 – Uniform interface to functions, distributions, and values`), and defining each concrete spec beside the kind it describes keeps this layer generic and type-agnostic (`D2 – Generality first`). Naming the base for the terms it types is `C5 – Naming for unambiguous meaning` applied to the library's own vocabulary: every spec types a tracked term, and *value* stays reserved for the mathematical kind. The kind rule enforces `D2 – Generality first`: every result can be tracked and every collection of draws stacked, so nothing an operation produces falls outside the system. A symbolic dimension carries a dimension's identity, which is mathematical structure, while deferring its size to the data that determines it, so cross-field equalities travel with the term and sizes bind when their producer appears (`D5 – Explicit, carried structure`, `C3 – Computational detail hidden by default, available on demand`).

## II.3 — Input and output declarations: `InputSpec`, `OutputSpec`

### Contract

An `InputSpec` is a flat mapping from names to term specs: the independently bindable slots a map-like term takes in. An `OutputSpec` is a term spec plus a required name for the produced term. Every output declaration is named because every tracked term is named. Symbolic dimensions are scoped jointly across an `InputSpec`'s slots and the output declaration beside it, so a name shared between an input and the output is one dimension:

```python
class InputSpec(Mapping[str, TermSpec]): ...   # named slots; keys are Python identifiers

class OutputSpec:
    name: str        # an identifier naming the produced term
    spec: TermSpec
```

Neither declaration is a `TermSpec` since they do not specify a term, so the kind rule of II.2 covers the term specs alone.

### Rationale

Requiring a name on every output declaration serves `C5 – Naming for unambiguous meaning` and `C6 – Traceable and reproducible workflows` together: the produced term's name is fixed where the producer is declared.

## II.4 — Numeric values: `Numeric`, `NumericSpec`, `Constraint`

### Contract

**The `Numeric` interface.** The numeric kinds — the array kind, the numeric record specialization, and their batch forms — share one flat-vector interface. The `to_vector` method lays a value out as one flat vector in canonical order, the `vector_size` property is the flat vector's length, and the `from_vector` method rebuilds a value from the flat representation. The coordinate hooks expose the same layout to foreign libraries, so `np.*` and `jnp.*` functions can be applied to the numeric kinds at the coordinates and return bare arrays. ProbPipe operators and elementwise `map` preserve structure and return tracked terms. Anything typed over flat numeric values types against `Numeric` once. If a bare array is passed where a `Numeric` value is expected, the array is promoted to the appropriate numeric type without any value copying.

`Numeric` is an **abstract base class rather than a structural protocol**, and the distinction is worth stating once. A *capability* is open: any object may claim `SupportsMean` by implementing it, third-party families included, so capabilities are structural (III.8). A *kind interface* is closed: the numeric kinds are the ones ProbPipe defines, and a bare array reaching a `Numeric` position is promoted rather than duck-typed, so nothing outside the library ever satisfies it. Being a base rather than a shape also lets it carry what every numeric kind shares — the coordinate hooks are `to_vector` under other names, so the base defines them once and each kind implements only `to_vector` and `vector_size`.

```python
class Numeric(ABC):                         # the flat-vector interface of the numeric kinds
    @property
    @abstractmethod
    def vector_size(self) -> int: ...       # total flat dimension; defined only when concrete
    @abstractmethod
    def to_vector(self) -> Array: ...       # the coordinates: one flat vector, canonical order

    # supplied once, so no kind restates them: numpy and JAX see to_vector
    def __array__(self) -> np.ndarray: ...  # and therefore return bare arrays
    def __jax_array__(self) -> Array: ...
```

**The `NumericSpec` mixin.** The spec-side counterpart of `Numeric` marks the specs whose values implement it. A `NumericSpec` carries the flat dimension and fixes the canonical flat layout that its values' `Numeric` interface obeys; construction from coordinates stays with the value types, since a spec describes and never builds. The mixin is abstract and doesn't specify a kind of its own.

```python
class NumericSpec(TermSpec, ABC):   # mixin: the specs whose values implement Numeric
    @property
    def vector_size(self) -> int: ...   # total flat dimension; defined only when concrete
    # fixes the canonical flat layout that its values' to_vector / from_vector obey
```

A numeric kind may also specify the **support** of its values with a `Constraint`, which
compares and hashes by value, so an instance can serve as a registry key.

```python
class Constraint(ABC):
    @abstractmethod
    def check(self, value: ArrayLike) -> Array: ...   # elementwise membership
```

### Rationale

One flat-vector interface over the numeric kinds is `D2 – Generality first`: everything that consumes flat numeric values types against it once, and the coordinate protocols keep foreign array functions usable with no ProbPipe-specific code (`C3 – Computational detail hidden by default, available on demand`). The spec-side mixin is the same generality at the type level: everything that requires a numeric declaration types against `NumericSpec` once, whether the event is one array or a named tree of them. Both are abstract bases rather than protocols, which keeps the pair symmetric and states the rule the library follows elsewhere: an interface a closed set of ProbPipe kinds implements is a base, while an open claim any object may make is a structural protocol (`D6 – Single source of truth`, since the base holds the shared coordinate hooks instead of four copies). A constraint is data, not behavior: comparing and hashing by value lets a support key a registry, so the bijector factories select by the mathematics rather than by class identity (`D3 – Capability-based operations`).

## II.5 — Identity, type & metadata: `TrackedTerm`, `Provenance`

### Contract

Identity, type & metadata is the workflow layer's hold on a semantic object: `TrackedTerm` is the one mixin that carries all of it, so every tracked term carries the same four things and no kind opts in or out. They are a **name** (what the object is called), a **spec** (the declaration of its type, II.2), a **provenance** (how it was produced), and free-form **annotations** (auxiliary information supplied by the user or an algorithm). Every first-class object — the kind an operation consumes and produces — must be a `TrackedTerm`, while structural helpers such as specs are not. We call any such object a **tracked term**: values, functions, distributions, conditional distributions, linear operators, and batches.

A tracked term's name must be provided by the user when constructed explicitly (as the required first argument to the constructor). When an operation produces an object, it must provide a meaningful, deterministic name derived from its inputs. The `name_is_auto` flag records which, because the two behave differently: a structure-changing transform re-derives an auto-derived name from its result, and composition into a larger object may rename it again, while a user-given name is preserved. A *nested* object (i.e., a sub-object of a `NamedTree`) takes its name from the field key it sits under. For example, a sub-object reached at `parameters` is itself named `parameters`. `with_name` renames the object itself, unlike `NamedTree.with_path_names`, which renames the fields within it. A name is a human label rather than an identity: nothing resolves an object by name, derived names need no escaping scheme, and two objects may share a name wherever field uniqueness does not force distinctness. The provenance of a tracked term stores pointers to descriptors of the parent objects that created it, along with the operation. Optionally, it can also provide references to the parents themselves.

The `spec` slot is the term's type, stored once. Each kind narrows it to its own spec class and exposes convenience accessors for its properties.

Every tracked term exposes `raw()`, the **one door to the representation layer**: it returns the term **detached** from the workflow, and it is the only such door, so the escape hatch is one rule rather than a per-class convention. *Detached* means precisely: **without provenance, without annotations, and without any reference to a container or parent it was viewed from**. What detachment does *not* strip is what the object needs to be itself — a detached value keeps its spec, and keeps its name and `name_is_auto` where it has them, so detaching is a lightening rather than a stripping. What detachment yields depends on whether the kind has a representation of its own to hand back. A kind with a raw host returns it: the backing array, the nested mapping of raw leaves, the wrapped callable, the wrapped object, an operator's stored parameterization, and, for a batch, the storage view. A kind whose representation *is* a ProbPipe object — a distribution, a kernel — returns the object itself, detached. Either way the rule a user learns is one sentence: `raw()` gives you the thing, cut loose from the workflow.

Access follows from this, uniformly. Reaching into a container — a record field, a batch element, a distribution's field — returns a **view**: a tracked term whose name derives from the accessor, marked `name_is_auto`, whose provenance records the container it came from and the source term where one was supplied. Views are how the workflow stays connected; `raw()` is how it is cut. No kind reverses the two.

**A tracked term is immutable, and that is a property of being one**: assignment and deletion raise, naming the class the caller touched, and every transformation returns a new term. `TrackedTerm` therefore carries the immutability itself rather than each kind opting in — one guard, so a subclass cannot report a different rule than its base. Immutability obliges a second thing, since `pickle` and `copy` restore an object by assigning its state back: a term reconstructs by allocating its resolved class and restoring the state it actually holds, rather than by rebuilding through its constructor. So a reconstruction cannot re-derive a schema an explicit declaration had pinned, cannot re-decide a class from arguments the state no longer carries, and cannot omit a field — including one written after construction, which no constructor argument names. A term declares any *memo* it holds as transient, keeping a cache out of the round-trip, and any *store written in place* as decoupled, so a copy takes its own container rather than sharing one. Annotations are the one such store (below). Identity is **boundary-attached** under compiled execution: inside a `jit` or `vmap` trace a term presents as its raw representation with only its spec as static data — name, provenance, and annotations never enter a trace, so a name can never affect compilation-cache identity — and the tracked result is minted at the enclosing call boundary.

```python
class TrackedTerm:
    name:         str
    name_is_auto: bool
    spec:         TermSpec                       # the single stored source of the term's type (II.2)
    provenance:   Provenance | None              # write-once via with_provenance(...)
    annotations:  Mapping[str, Any] | None       # free-form; the one store written after construction
    def with_name(self, name: str) -> Self: ...  # shallow copy with name_is_auto = False
    def with_provenance(self, p: Provenance) -> Self: ...
    def raw(self) -> Any: ...                    # the representation, detached from the workflow
    # immutable: __setattr__ / __delattr__ raise; state round-trips through the
    # attributes the term holds, so copy and pickle need nothing from the class

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

The **annotations** store is the one exception to immutability: it can be written after construction, so a diagnostic can attach its result to the object it examined. By convention it is append-only, and it is decoupled on copy — a copy takes its own container rather than sharing one — as the reconstruction contract above requires of any store written in place. Annotations are inert everywhere else: no operation reads them to decide behavior, they do not propagate to results, since lineage rides on `provenance`, and they never enter a compiled trace.

A provenance records everything its operation resolved: the tracked parents, the plain arguments, and the **controls** the run actually used, the PRNG key, sample counts, and any selected method among them, so a result can be reproduced from its record. Fingerprints are best-effort, in tiers — a content hash, the qualified name and code hash for a closure-free callable, or object identity, with `fingerprint_is_weak` marking the weakest — and **the provenance mode, not the operation, sets the tier**: the default lightweight mode records identity-tier descriptors for every operation's parents, with no content hashing; the full mode retains parent references and fingerprints content-verifiably, computable lazily at export since the objects are held; off records nothing. Every operation behaves identically under a given mode, so provenance cost is one user-visible dial rather than a property of what ran.

### Rationale

`TrackedTerm` serves the two non-mathematical principles: `C5 – Naming for unambiguous meaning` and `C6 – Traceable and reproducible workflows`. Housing the spec on the tracked base is `D6 – Single source of truth` for a term's type: one slot, declared once, that every kind's accessors are views on. The guarantee behind `C6 – Traceable and reproducible workflows` is a single rule: **every object a ProbPipe operation natively returns is a tracked term**, so the provenance chain is never broken. Recording the resolved controls, not just the parents, is what turns traceability into reproducibility: re-running the recorded operation on the recorded inputs with the recorded controls reproduces the result. Auto-derived names keep every intermediate object identifiable without forcing the user to label it (`C5 – Naming for unambiguous meaning`). Boundary attachment keeps names semantically inert in computation: nothing resolves an object by name, so a name may never decide what gets compiled. Immutability itself is `C2 – Functional interface over immutable objects` embodied: every transformation returns a new term, so nothing downstream ever observes a change; confining the one writable store to a container nothing load-bearing reads keeps that contract intact in substance. Carrying annotations on the same base, rather than on a second mixin some kinds elect, is what makes *every* tracked term annotatable — a diagnostic can label a batch of draws exactly as it labels a record. Because identity and metadata are orthogonal to *what* an object is mathematically, they are defined uniformly across classes.

## II.6 — `Batch`

### Contract

A `Batch` is the generic `TrackedTerm` nd array of shape `batch_shape` that holds objects of a common element type. `batch_shape` is nonempty: a batch has at least one batch axis. A concrete batch implementation must specify how to store the elements. Since a batch is a *collection* of its elements, `len` / `iter` / `batch_shape` / `batch_size` operate only on the batch axes. The `batch_*` names are kept deliberately rather than a numpy-style `.shape` / `size`, which could ambiguously cover both the batch axes and the per-element content. A concrete batch adds whatever its element type affords in that element's own section, indexing into the elements' fields included. `BatchSpec` is the batch kind's term spec (II.2), carrying the spec the elements satisfy together with the named multiplicity:

```python
class BatchSpec(TermSpec):         # the batch kind's spec; is_valid accepts a matching batch
    element_spec: TermSpec                          # the spec every element satisfies
    axis_groups: tuple[tuple[int | str, ...], ...]  # the multiplicity, tiled into levels below;
                                                    #   a str names a symbolic dimension (II.2)
    level_names: tuple[str, ...]
```

**`[]` dispatch behavior.** A key is either a **position** or a **name**, and the two namespaces never collide: an axis has no name, and a field no position. A *position* (an integer, a slice, or a tuple of those) addresses the batch axes, which `Batch` itself answers. A *name* (a string, or a tuple of strings for a path) addresses a field within every element, which applies only to a batch whose elements have fields. A tuple mixing the two is invalid.

```python
class Batch[E](TrackedTerm):
    @property
    def spec(self) -> BatchSpec: ...                     # the single stored source of the type
    @property
    def element_spec(self) -> TermSpec: ...              # view on spec: what every element satisfies
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
    def __repr__(self) -> str: ...                      # the class, the name, and each level with its sizes
    def __getitem__(self, key: Any) -> Any: ...         # a position indexes the axes, a name the elements' fields
    def at_levels(self, /, **levels: int | slice | None | tuple[int | slice | None, ...]) -> E | Self: ...
    # index by named level (a view); unnamed levels kept whole, None means the whole axis (:)
    def raw(self) -> Any: ...
    # the storage view: the elements' raw values in their native stacked layout
```

**Axis groups.** A batch's axes are partitioned into ordered **levels**. `axis_groups` tiles `batch_shape` into contiguous groups, outermost level first, and `batch_shape` stays their flat concatenation, so anything stated over `batch_shape`, flat vectorization above all, applies to a multi-level batch unchanged. A single-level batch has one group holding all its axes. `len`, `iter`, and positional `[]` address the leading **axis**, `batch_shape[0]`, rather than the leading level. The two coincide when the outermost level holds one axis: iterating `(N,)` of `(S,)` then walks the `N`, yielding each inner batch of `S` as a view. When the outermost level spans several axes, the leading axis is only the first of them, so indexing drops that axis and leaves the level in place, one axis shorter. Nesting needs no dedicated classes: a batch is itself a tracked term, so a batch whose elements are batches is already admitted, and grouped storage presents the levels as views into one store.

**The batch's own type.** A batch stores its `BatchSpec` (above), which carries the element spec and the named multiplicity together; `element_spec`, `axis_groups`, and `level_names` are views on it. This keeps the storage rule reading the same way for a batch as for any other term — `spec` is the batch's *own* type, not its element's — which is what keeps `OpaqueBatch` well typed even though its elements carry no structure of their own. It also mirrors the model, where a batch inhabits the *family* kind over its element's kind rather than the element kind itself.

**The raw view.** `raw()` returns the storage view: the elements' raw values in their native stacked layout — one array for array-valued elements, an object array for callable- or opaque-valued ones, and the nested mapping of raw columns for record-valued ones. It is the batch counterpart of the `raw` opt-out on operations: the tracked batch is the default, and the raw view is the same data with no identity attached.

**A polymorphic multiplicity.** An axis size may be a symbolic dimension name instead of an integer, exactly as a `NumericArraySpec` shape entry may, so a *declaration* can fix the number of levels while deferring how many elements each holds — "returns a batch of `S` draws" before `S` is known. The names share one scope with the element's schema, so a batch of `("n",)` over arrays of shape `("n",)` is square by declaration. A declaration may be polymorphic; a live `Batch` may not, since it holds elements at positions, so construction refuses a spec with free dimensions and `batch_size` is undefined until they are bound.

**Level names.** Each level carries a name, listed in order by `level_names`, and names are unique within a batch. An operation that mints a level takes the name to give it; a name already in use raises, as does a rename onto one, so the caller supplies another. A suffixing rule was considered and rejected: `draw2` would state the order the levels were added in, a fact about the computation path rather than about meaning, and levels are named precisely so that operands align by meaning. `with_level_names` repins names on a shallow copy, shapes and elements unchanged, re-reading its own derived name under the new ones so that it and any view taken from it agree. Renaming a *view* is refused where the new name belongs to a level the view derives its name from but no longer carries, since the derived name would then be ambiguous; `with_name` re-roots it instead. Names must be identifiers, since `at_levels` addresses a level by keyword. Operations align batched operands by level name — two levels meant to correspond under different names are lined up by renaming one — and level names are independent of the field names within an element.

**View identity.** A view of a batch — an element or a sub-batch — derives its name from the batch it was taken from and the positions it selects, naming the level each selection addresses. Take a batch named `posterior`, with a `chain` level of `(4,)` over a `draw` level of `(1000,)`:

```python
posterior.at_levels(chain=0).name           # "posterior[chain=0]"          — a sub-batch of draws
posterior.at_levels(chain=0, draw=7).name   # "posterior[chain=0, draw=7]"  — an element
posterior.at_levels(draw=slice(1, 3)).name  # "posterior[draw=1:3]"         — both levels kept
```

The derived name is marked `name_is_auto`, and it states what was selected rather than how it was reached: levels selected whole are left out, so selecting all of a batch derives the batch's own name, and the levels that do appear are listed in the batch's own order rather than the order they were indexed in. Two routes to one selection therefore read alike, and two different selections never do. Lineage needs no node of its own, since selecting computes nothing — a view carries the lineage of the batch it came out of, and which position it was is already stated by the name. Storage changes none of this. Whether a batch *materializes* an element, as a row of columnar storage that does not exist until it is built, or *stores* it outright, indexing returns a **view** under the derived name, its provenance recording the batch and, where a tracked element was supplied, that element as the source (II.5). The stored term itself is never handed back and never renamed: a view is new, an immutable term is not relabelled in place, and the source stays reachable through the view's provenance. A *sub-batch* is a view in the same way, being the batch's own selection rather than an object a caller put there.

**Selecting by level.** `at_levels(**levels)` indexes a batch along its named levels and returns a view, the by-name counterpart of positional `[]`. It is the level analogue of `NamedTree.at_path` (II.1), and shares its shape: a path addresses a position and returns a leaf or a subtree, while named level indexers address positions and return an element or a sub-batch. The name is neither `select`, the field-splatting selector on `Record` that a batch of records also carries, nor a name promising elements, which it returns only when the selection indexes down to one. Each indexer is an integer, a slice, `None`, or a tuple of these addressing the level's axes in order, where an integer drops its axis and a slice or `None` keeps it, `None` meaning the whole axis as `:` does. It means that here and only here: a keyword cannot take a `:` literal, so `None` is the spelling this form needs, while positional `[]` writes `:` and refuses `None`, which an unset argument would otherwise leave standing for *all of it*. A level spanning several axes takes one indexer per axis, and a shorter indexer fills the leading axes and leaves the rest whole, so a scalar `draw=i` on a two-axis `draw` level means `draw=(i, None)`. A level whose axes are all dropped is removed, so selecting a single-axis level by an integer — or every axis of a multi-axis level — yields the inner batch or element just as positional indexing and iteration do, while a level left unnamed is kept whole. This parallels xarray's `isel` in indexing by name and by position, with `level_names` in the role of xarray dimension names, and there is no label-based counterpart, since batch levels carry no coordinate labels. The tuple form departs from it: a tuple here addresses a level's axes in order, where `isel` reads one as several positions of a single dimension.

### Rationale

`Batch` is necessary to satisfy `D1 – Mathematical fidelity` by ensuring how many objects there are stays separate from what one object contains. The level structure extends the same fidelity to collections of collections: how many objects sit at each level is a mathematical distinction, so `N` laws with `S` draws each are `(N,)` of `(S,)` rather than one anonymous `(N, S)`. Naming the levels is `C5 – Naming for unambiguous meaning` on that axis, letting a user say which multiplicity is which and letting operations align batches by meaning rather than by position. An operation broadcasts across a batch by mapping over its elements, so a batch supports an operation exactly when its elements do (`D3 – Capability-based operations`). When those elements are array-backed and claim differentiation, the mapping is vectorized and differentiable end-to-end, the same capability rule read at the gradient. To satisfy `D6 – Single source of truth`, indexing or iterating yields a *view*, the levels of a multi-level batch included. The obligation lands on the concrete batch, since only it knows its storage: the hook that presents a sub-batch shares the store rather than copying out of it, wherever the storage affords sharing. That is also why selecting all of a batch needs no special case — it reaches the same hook as any other selection, and a view costs nothing there.

## II.7 — Dispatch and registries

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

A registry is how `C3 – Computational detail hidden by default, available on demand` and `D3 – Capability-based operations` reach operations whose implementation cannot be chosen from a single object alone. The `check` probe keeps auto-selection safe, while `method="..."` leaves the choice in the user's hands. New implementations join by registering a method, so the supported set grows without touching the call sites that use it, which is `D2 – Generality first`. Gathering every registry under one catalog serves `D6 – Single source of truth`: there is one place to see which implementations exist and how a call resolves.
