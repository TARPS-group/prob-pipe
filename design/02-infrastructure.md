# Part II — Infrastructure

Part II introduces the infrastructure the rest of the library is built on: generic, type-agnostic machinery, one piece at a time, in dependency order.


## II.0 — Overview: the infrastructure

The infrastructure layers, in dependency order. Each is generic and type-agnostic, defined once and reused throughout the library:

| § | Layer | Abstraction | Role |
|---|---|---|---|
| II.1 | Structure | `NamedTree` | The named, ordered tree addressed by path that every structured object is built on, owning the leaf-keyed mapping contract and navigation. |
| II.2 | Identity | `Tracked` / `Annotated` / `Provenance` | The name, lineage, and annotations an object carries beyond its mathematical content. |
| II.3 | Multiplicity | `Batch` | The generic multiplicity axis: an indexed collection of *separate* objects, distinct from one object over a structured space. |
| II.4 | Dispatch | dispatch & registries | Registry-based multiple dispatch that selects an implementation by the types involved. The shared mechanism behind converters, inference selection, and bijector factories. |

## II.1 — `NamedTree`

### Contract

Values in ProbPipe are represented as named, ordered trees.  ProbPipe uses the following standardized **terminology** to refer to components of these trees:
- A **field** is one named leaf — a single object in the collection.
- A **path** is a `/`-joined sequence which can address either a field or an interior node.
- A **key** is a path that specifically addresses a *field*. Every key is a path but a path for an interior node is not a key.
- A **child** of a node is an entry directly under that node. 
- The **canonical order** of a tree is a depth-first walk visiting children in insertion order. 

Naming, addressing, traversal, and structure-preserving transforms are defined once in the `NamedTree` class. Specially, a `NamedTree` is a `Mapping` whose keys are exactly its leaf paths, in canonical order, so `keys()`, `values()`, `items()`, `len`, `in`, and `[]` all agree on that one key set, as for a plain `dict`. A leaf path may be equivalently written as a string (`"a/b"`) or a tuple (`("a", "b")`). Since interior nodes are *not* keys, they are instead accessed via either the one-level `children` view or `at_path`, which can access any leaf or subtree. So, for example, the invariants  `x.children["a"].children["b"] == x.at_path("a", "b") == x.at_path("a/b")` hold. 

```python
class NamedTree[L]:
    # mapping interface — keyed by FIELD path
    def __getitem__(self, key: str | tuple[str, ...]) -> L: ...          # leaf only
    def __contains__(self, key: str | tuple[str, ...]) -> bool: ...      # leaf only
    def __iter__(self) -> Iterator[str]: ...                             # field paths
    def __len__(self) -> int: ...                                        # field count
    def keys(self) -> Iterable[str]: ...                                 # field paths, canonical order
    def values(self) -> Iterable[L]: ...
    def items(self) -> Iterable[tuple[str, L]]: ...

    # tree navigation — ranges over ALL paths
    def at_path(self, *path: str) -> L | Self: ...                       # at_path("a", "b") == at_path("a/b")
    @property
    def children(self) -> Mapping[str, L | Self]: ...                    # one level
    def is_field(self, path: str | tuple[str, ...]) -> bool: ...
    @property
    def is_multi_field(self) -> bool: ...

    # structure-preserving transforms — return the same family
    def map(self, f: Callable[[L], L], /, *args, **kwargs) -> Self: ...
    def map_with_keys(self, f: Callable[[str, L], L], /, *args, **kwargs) -> Self: ...
    def replace(self, path: str | tuple[str, ...], leaf: L | Self) -> Self: ...
    def merge(self, other: Self) -> Self: ...
    def without(self, path: str | tuple[str, ...]) -> Self: ...

    # (de)serialization
    def to_nested_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_nested_dict(cls, d: Mapping[str, Any]) -> Self: ...

    @classmethod
    def _node_type(cls) -> type[Self]: ...                               # the family's own node type
    @classmethod
    def _leaf_type(cls) -> type | UnionType: ...                         # the family's declared leaf type; leaves are validated against it at construction
```

### Rationale

Using named paths is necessary to satisfy `C5 – Naming for unambiguous meaning`. Housing the collection contract in one substrate ensures`D7 – Single source of truth` is satisfied — the type- and value-level structures built on it cannot drift apart on how a field is named or a path is resolved.
### Notes

- *Same-family closure.* Every navigator and transform returns the same family, enforced by `_node_type()`.
- *Leaf type versus node type.* The parameter `L` declares the leaf type, the only axis on which tree families differ: it is what `values()`, `[]`, and `map` traffic in. The node type is not a second parameter, since interior nodes are always instances of the family's own class, which is what `_node_type()` reports. The runtime partition therefore uses the node type alone: a field value is an interior node when it is an instance of `_node_type()`, and a leaf otherwise. Validation is the substrate's job rather than each family's: construction checks every leaf against the family's declared leaf type, reported by `_leaf_type()`, so a malformed tree fails at construction rather than at first navigation. A family whose leaves are arbitrary values declares `object`, making the check vacuous.
- *Navigation yields views.* `children`, `at_path`, and `[]` return a subtree or leaf that is a *view* into the same underlying store, derived on demand rather than copied out.

## II.2 — Identity & metadata: `Tracked`, `Annotated`, `Provenance`

### Contract

Identity & metadata is the cross-cutting layer that lets any object carry, alongside its mathematical content, three things: a **name** (what the object is called), a **provenance** (how it was produced), and free-form **annotations** (auxiliary information supplied by the user or an algorithm). The structure for identity and metadata is provided by two mixins: `Tracked` (name + provenance) and `Annotated` (annotations). All ProbPipe objects must be `Tracked`. We call any such object a **tracked term**: a value, distribution, conditional distribution, or batch that carries a name and provenance, and the kind of object ProbPipe operations consume and produce.  

Annotation metadata is a free-form mapping: 

```python
class Annotated:                           
    annotations: Mapping[str, Any] | None
```

A tracked term's name must be provided by the user when constructed explicitly (as the required first argument to the constructor). When an operation produces an object, it must provide a meaningful, deterministic name derived from its inputs. The `name_is_auto` flag records which, because the two behave differently: an auto-derived name may need to be updated when the object is combined into a larger one, while a user-given name is preserved. A *nested* object (i.e., a sub-object of a `NamedTree`) takes its name from the field key it sits under. For example, a sub-object reached at `parameters` is itself named `parameters`.  The provenance of a tracked term stores pointers to descriptors of the parent objects that created it, along with the operation. Optionally, it can also provide references to the parents themselves. 

```python
class Tracked:                              # name + provenance
    name:         str                       # semantic identity — user-given or auto-derived
    name_is_auto: bool                      # True if an op minted the name (regenerable); False once user-given
    provenance:   Provenance | None         # how it was produced — write-once
    def with_name(self, name: str) -> Self: ...            # functional: a new object, user-given name (name_is_auto -> False)
    def with_provenance(self, p: Provenance) -> Self: ...   # set once

class Provenance:
    operation: str                          # the operation that produced the object
    parents:   tuple[ParentInfo, ...]       # descriptors of the inputs — not live object references

class ParentInfo:                           # a lightweight stand-in for a parent object
    type_name:   str
    name:        str
    fingerprint: str                        # stable content digest (cache / identity checks)
    obj:         Any | None                 # optional reference to original parent
```

### Rationale

`Tracked` serves the two principles that reach beyond the mathematics: `C5 – Naming for unambiguous meaning` (an object and its parts are legible) and `C6 – Traceable and reproducible workflows` (provenance is the recorded lineage that lets a result be audited and re-derived). The guarantee behind `C6 – Traceable and reproducible workflows` is a single rule: **every object a ProbPipe operation natively returns is a tracked term** (whether or not it is also `Annotated`), so the provenance chain is never broken. Auto-derived names keep every intermediate object identifiable without forcing the user to label it (`C5 – Naming for unambiguous meaning`). Because identity and metadata are orthogonal to *what* an object is mathematically, they are defined uniformly across classes. 

## II.3 — `Batch`

### Contract

A `Batch` is the generic multiplicity axis: an indexed collection of `N` *separate* objects of a common element type, arranged in an nd `batch_shape`. It is itself a value and hence `Tracked` (but not `Annotated` unless applications for it arise). A concrete batch implementation must specify how to store the elements. 

```python
class Batch[E](Tracked):                            # an indexed collection of N separate objects of type E
    @property
    def batch_shape(self) -> tuple[int, ...]: ...   # nd-shape of the collection — excludes per-element content
    @property
    def batch_size(self) -> int: ...                # prod(batch_shape): total element count
    def __len__(self) -> int: ...                   # leading-axis size, batch_shape[0]
    def __iter__(self) -> Iterator[E | Self]: ...   # over the leading batch axis
    def __getitem__(self, index: Any) -> E | Self: ...  # index/slice the batch axes -> a view: an element or a sub-batch
```

Since a batch is a *collection* of its elements, `len` / `iter` / `batch_shape` / `batch_size` operate only on the batch axes. The `batch_*` names are kept deliberately rather than a numpy-style `.shape` / `size`, which could ambiguously cover both the batch axes and the per-element content. A concrete batch implementation adds whatever its element type affords in that element's own section — including, where useful, indexing into the elements' fields, since `[]` dispatches unambiguously on the key type.
### Rationale

`Batch` is necessary for `D1 – Mathematical fidelity`: *how many objects there are* stays separate from *what one object contains*. A batch of `N` objects is a collection of `N` distinct things, keeping multiplicity and structure separate. An operation broadcasts across a batch by mapping over its elements, so a batch supports an operation exactly when its elements do (`D3 – Capability-based operations`), and when those elements are array-backed the mapping should be vectorized and differentiable (`D6 – Differentiability where possible`). Indexing or iterating yields a *view*, not a copy — the batch stays the single authoritative store, and an element or sub-batch is derived on demand (`D7 – Single source of truth`).

## II.4 — Dispatch and registries

### Contract

Some operations have several interchangeable implementations, and which one applies depends on the *types* of the objects involved rather than on a single object's own class. A **dispatch registry** holds those implementations as named methods and selects one for a given call.

Each **method** declares a unique `name`, the types it applies to (`supported_types`, a fast `issubclass` pre-filter), a cheap `check` that probes feasibility without doing the work, an `execute` that performs it, and a `priority` that orders auto-selection. Dispatch is by argument type: a `UnaryDispatchRegistry` keys on the first argument's type, and a `BinaryDispatchRegistry` on the first two. The registry takes the matching methods in priority order and runs the first whose `check` reports feasible. A caller can bypass auto-selection and name a method with `method="..."`. New methods join by registration, so an implementation is added without changing any call site.

```python
class BaseDispatchMethod(ABC):
    name: str                                    # unique within the registry
    priority: int                                # orders auto-selection
    def supported_types(self) -> ...: ...        # issubclass pre-filter; shape set by arity
    def check(self, *args) -> MethodInfo: ...    # cheap feasibility probe
    def execute(self, *args) -> Any: ...         # the actual computation

class BaseDispatchRegistry[M: BaseDispatchMethod]:
    def register(self, method: M) -> None: ...
    def execute(self, *args, method: str | None = None) -> Any: ...      # auto-select, or run the named method
    def check(self, *args, method: str | None = None) -> MethodInfo: ...
    def list_methods(self) -> list[str]: ...                             # names, in selection order

class UnaryDispatchRegistry[M](BaseDispatchRegistry[M]): ...    # keys on one argument's type
class BinaryDispatchRegistry[M](BaseDispatchRegistry[M]): ...   # keys on the first two
```

A single **catalog** makes every registry in the process discoverable. The global `registry_catalog` lists each registry, its methods with their priorities, and a one-line description, so a user can see which methods exist and how a given call will resolve. A registry joins the catalog by implementing `SupportsRegistryCataloging`, a protocol broad enough to admit registries that do not share the dispatch base.

```python
@runtime_checkable
class SupportsRegistryCataloging(Protocol):
    def catalog_entry(self) -> CatalogEntry: ...   # the registry's name and kind, and its methods with priorities

registry_catalog: RegistryCatalog                  # global; lists every registry and resolves describe(name)
```

### Rationale

A registry is how `C3 – Computational detail hidden by default, available on demand` and `D3 – Capability-based operations` reach operations whose implementation cannot be chosen from a single object alone. The `check` probe keeps auto-selection safe, while `method="..."` leaves the choice in the user's hands. New implementations join by registering a method, so the supported set grows without touching the call sites that use it, which is `D2 – Generality first`. Gathering every registry under one catalog serves `D7 – Single source of truth`: there is one place to see which implementations exist and how a call resolves.
