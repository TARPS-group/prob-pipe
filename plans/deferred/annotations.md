# Plan: when an annotation follows a term

> **AI-generated.** An AI assistant drafted this plan, and no maintainer has reviewed or approved it yet.

> **Status: proposed; not approved.**
> Written on 2026-10-28 against `main` at `97891b7d`.
> Review it, and answer its questions, before implementing anything from it.
> `plans/deferred.md` lists it under "Annotations".
> Once approved, its rules move into `design/` (II.4, V.4, VII.7) and its steps into the plans of the work after the workshop.

**Decisions this plan asks for, all open** (each explained where it arises, and gathered in § 8):

1. The propagation an entry takes when its writer declares none (§ 5.4).
2. Whether an inference result's ArviZ draws are stored and restricted, or derived from its atoms (§ 6.1).
3. Whether `annotations` is an empty mapping, rather than `None`, on a term with no entries (§ 5.5).
4. Whether the inference `method` leaves the annotations, since provenance records it (§ 6.1).
5. Whether `annotate` refuses a key that is already present (§ 5.6).

---

## 1. Why this plan exists

An annotation is information attached to a term that is not part of its mathematics, such as the ArviZ data of an MCMC run or the R-hat of each parameter.
The design says where annotations are stored, and it says that they do not propagate.
The code propagates them through some operations and drops them in others, and one operation can do both, depending on which route it takes.

A posterior from MCMC shows the problem:

```python
lik = conditional_distribution(
    lambda mu, sigma: Normal("y", mu * jnp.ones(3), sigma),
    given_spec={"mu": NumericArraySpec(()), "sigma": NumericArraySpec(())},
)
post = condition_on(lik & (Normal("mu", 0.0, 1.0) & HalfNormal("sigma", 1.0)), {"y": y})

post.annotations["arviz"]["posterior"]   # variables mu, sigma over (chain, draw)
post.annotations["arviz"]["sample_stats"]  # diverging, energy, step_size, ...
post["mu"].annotations                   # None
```

The marginal `post["mu"]` holds the same chains of `mu` as `post`, from the same run.
Its draws of `mu` and the run's sampler statistics are still true of it, and a user expects to find them there.

The goal is one rule that decides, for every operation and every entry, whether the entry is carried to the result and in what form.
The rule is stated once in `design/`, implemented once on the base class `TrackedTerm`, and tested once over every kind.

## 2. The design today

### 2.1 What `design/` states

1. **Every term carries annotations** (II.4). They are the fourth part of a tracked term, after its label, spec, and provenance: "free-form auxiliary information supplied by the user or an algorithm". `TrackedTerm` declares `annotations: Mapping[str, Any] | None`, and the rationale says that carrying them on the base "makes every tracked term annotatable, including a batch of draws".
2. **The one mutable store** (II.4). Annotations "can be written after construction, so a diagnostic can attach its result to the object it examined", and by convention the store is append-only.
3. **Inert** (II.4). "No operation reads them to decide behavior, they do not propagate to results because lineage is recorded in `provenance`, and they never enter a compiled trace."
4. **Detached by `raw`** (II.4). Detachment removes provenance and annotations.
5. **Copied, not shared** (II.4, Notes). `copy` and `pickle` restore annotations, and a copy takes its own container.
6. **Host metadata** (V.4, step 3a). A registered array host's "remaining metadata, such as coordinates and attributes, is carried as annotations".
7. **Inference results** (VII.7). A result built from a run's draws "records in its annotations the attribute `method`". A fidelity summary "must derive from provenance rather than add an independent truth in `annotations`", and an approximation error is stored in annotations, "keyed by the producing method".

### 2.2 What the design leaves open

1. **Identity transforms.** "They do not propagate to results" leaves open whether `with_label`, `copy`, and `rename` are results in this sense. A relabeled posterior that lost its ArviZ data would surprise every user.
2. **Parts.** A marginal, a field of a record, and an element of a batch are parts of the term they come from. The design does not say whether an entry about the whole is true of a part.
3. **Writing.** The design gives no API for a write, so it does not say how a writer adds an entry or what happens when the key is present.
4. **Reading.** "No operation reads them" conflicts with diagnostics, which read the ArviZ data that a run recorded (§ 3.3, D7).

## 3. The code today

### 3.1 Where annotations are stored and written

The store is the attribute `_annotations` of the mixin `Annotated` in `core/tracked.py`, separate from `TrackedTerm`.

| Writer | What it writes | How |
|---|---|---|
| `_record_run` (`inference/_approximate_distribution.py`), for every inference method | a `DataTree` whose root attribute is `method` and whose subtree `arviz/` holds the run's groups | `_init_annotations` at construction |
| `_add_group` (`diagnostics/_datatree_store.py`), for `add_rhat`, `add_ess`, `add_mcse`, `add_loo`, `add_ppc` | groups under `diagnostics/` and `arviz/log_likelihood` | rebuilds the tree and replaces `_annotations` with `object.__setattr__` |
| `_record_check_in_annotations` (`validation/_predictive_check.py`), for `predictive_check` | groups `predictive_check/check_N` | mutates the tree in place, and creates one with `object.__setattr__` |
| `Function.__init__` | an empty `dict` | `object.__setattr__` |
| a user | anything | `object.__setattr__`, or the `dict` of a `Function` |

Four propagation sites exist, each written by hand:

| Site | Effect |
|---|---|
| `_shallow_copy`, which `with_label` and the in-place `rename` use | carries a decoupled copy |
| `__getstate__` and `__setstate__` | carry a decoupled copy |
| `_as_declared` and `_empirical_of` (`operations/_condition.py`) | carry the normalized law's annotations to the posterior |
| `CoupledMarginal`, `_RenamedDistribution`, `_RenamedConditionalDistribution` | set `None` |

### 3.2 Where the code agrees with the design

- Annotations are excluded from fingerprints (`core/_fingerprint.py`, `_SKIP`), so two terms that differ only in annotations compute the same thing.
- `raw` detaches them: `post.raw.annotations` is `None` (`_detached_term`).
- A copy, a pickle round-trip, and `with_label` give the result its own container, so a later write on one does not show on the other.
- A pytree rebuilt inside a trace carries only its spec, so annotations never enter a compiled trace.
- Every inference method writes through one function, `_record_run`.
- MCMC diagnostics compute R-hat, ESS, and MCSE from the atoms (`to_named_posterior_dataset`), not from the stored ArviZ copy of them.

### 3.3 Where the code departs from the design

**D1. Batches carry no annotations.**
`Annotated` is mixed into `Record`, `NumericArray`, `Opaque`, `Function`, `Distribution`, and `ConditionalDistribution`, and into no batch.
Its docstring says annotations are "not required of batches", while II.4 says that every term carries them:

```python
sample(post, sample_shape=4).annotations
# AttributeError: 'NumericRecordBatch' object has no attribute 'annotations'
```

**D2. Each writer writes in its own way, and the store has no one type.**
No public method writes an entry, so every writer sets the private attribute.
The empty store is `None` on five kinds and a mutable `dict` on `Function`, so one kind accepts a write through the public property and the others refuse it:

```python
f = Function(lambda x: x, label="f")
f.annotations["note"] = "fit on 2026 data"   # accepted
Normal("x", 0.0, 1.0).annotations           # None, so there is nothing to write into
```

The convention that the store is append-only is not kept: `_add_group` replaces a group of the same name, and `add_loo(force=True)` overwrites the stored LOO.

**D3. One operation gives different results by route.**
The same request carries the entries on one route and drops them on another:

| Operation | Result's annotations |
|---|---|
| `post.with_label("p")` | carried |
| `Normal("x", 0, 1).rename(x="z")` | carried, through `_shallow_copy` |
| `post.rename(mu="m")` | dropped, through `EmpiricalDistribution._renamed_in_family` |
| a rename that `_RenamedDistribution` holds | dropped |
| `post["mu"]`, `post[["mu", "sigma"]]` | dropped (`CoupledMarginal`) |
| `marginal(post, "mu")` | dropped (a new `EmpiricalDistribution`) |
| `record.with_label("r")` | carried |
| `record.rename(a="c")`, `record["a"]` | dropped |

A rename does not change a law, so whether its annotations survive depends only on the law's family.

**D4. The code propagates where the design says annotations do not.**
`with_label`, `copy`, and `condition_on`'s re-declaration of the normalized law (`_as_declared`) carry annotations, which II.4 forbids as written.
The code is right in these cases, since each result is the same law, so the design text is what changes.

**D5. The inference record repeats facts held elsewhere.**
Two facts are stored twice, against `D6 – Single source of truth`:

```python
post.annotations.attrs["method"]          # 'blackjax_nuts'
post.provenance.metadata["method"]        # 'blackjax_nuts'

post.annotations["arviz"]["posterior"]["mu"]   # (chain, draw) array
post.atoms["mu"]                               # the same draws on the levels chain and draw
```

VII.7 asks a fidelity summary to derive from provenance rather than annotations, and the `method` attribute is such a summary.

**D6. `predictive_check` writes into its input, under its own layout.**
`predictive_check(law, ...)` returns its result and also adds `predictive_check/check_N` groups to `law`'s annotations.
A function that does not announce a write mutates its argument, and `add_ppc` records the same kind of result under a different path, `diagnostics/runs/ppc`.
Without xarray installed, the write is skipped with no message.

**D7. Diagnostics read annotations to decide what to do.**
`add_loo` reads `arviz/log_likelihood` and computes it only when absent, and it skips the whole computation when `diagnostics/runs/loo` exists.
The divergence count reads `arviz/sample_stats/diverging`.
These reads are reasonable, since a run's sampler statistics exist only in its record, but they contradict "no operation reads them".

**D8. Host metadata is kept in the host, not in annotations.**
V.4 says an xarray container's coordinates and attributes become annotations.
The code keeps the container itself as the raw host, so they are part of the representation:

```python
da = xr.DataArray(np.ones(3), dims=["k"], coords={"k": ["a", "b", "c"]}, attrs={"units": "m"})
x = NumericArray(da, label="x")
type(x.raw)       # xarray.DataArray, with its coords and attrs
x.annotations     # None
```

The code is right: the coordinates are part of what the value is, and they enter its fingerprint (`_array_backend._metadata_of`).
V.4's sentence is what changes.

## 4. How xarray decides which attributes a result keeps

xarray's `attrs` are its analog of annotations, and its rules have changed toward keeping more:

1. **Attributes belong to the object they describe.** `ds["a"]` returns the variable `a` with `a`'s attributes, and the dataset's attributes stay with the dataset. `ds[["a"]]` is a dataset, so it keeps the dataset's attributes.
2. **Most operations keep attributes.** Since version 2025.11 the option `keep_attrs` defaults to `True`, so selection, renaming, reductions, ufuncs, and arithmetic with a scalar keep the attributes of their input.
3. **Binary operations combine.** `a + b` keeps the attributes on which `a` and `b` agree and drops those on which they conflict (`"drop_conflicts"`).
4. **A caller can override.** `keep_attrs=` on an operation, or `xr.set_options(keep_attrs=...)`, decides for that call.

Checked against xarray 2026.4.0:

```python
a = xr.DataArray(..., attrs={"units": "m", "src": "a"})
b = xr.DataArray(..., attrs={"units": "m", "src": "b"})
a.mean().attrs     # {'units': 'm', 'src': 'a'}
(a + b).attrs      # {'units': 'm'}
```

Rule 1 suits ProbPipe, and this plan adopts it: an entry describes a term, and a part of the term inherits the parts of the entry that describe it.
Rule 2 does not suit ProbPipe.
xarray's attributes are mostly descriptions, such as units, which stay plausible after a reduction.
ProbPipe's entries are mostly records of a computation on one law, such as an R-hat or a LOO estimate, which are false of a different law:
the R-hat of `post` is not a property of `mean(post)`.
Appendix A gives the reasons for not adopting rules 2 to 4.

## 5. The proposed design

### 5.1 Principles

1. **An entry is a statement about its term.** A result carries an entry exactly when the statement is still true of the result, possibly in a restricted or renamed form.
2. **The operation states its relation.** Each operation declares how its result relates to its input, once, and that relation is the operation's whole contribution to the rule.
3. **The writer states how an entry follows.** The writer of an entry declares how the entry follows its term, since only the writer knows what the entry says.
4. **One implementation.** One function carries annotations from a source to a result, and every operation calls it.
5. **Lineage stays in provenance.** A result that is a new law carries no entries of its input, and its provenance names the input. In the provenance mode `full`, the parent reference holds the input with its annotations.

### 5.2 What an annotation is

> **Definition (annotation).** An *annotation* of a term `t` is an entry of the mapping `t.annotations`: a string key and a value, which states a fact about `t` that is not part of `t`'s mathematical content.
> An entry has a *propagation*, which says how the entry follows `t` into a term derived from it (§ 5.4).
> No operation that computes a value, a law, or a function reads an annotation.

The last sentence replaces "no operation reads them".
A diagnostic, such as `add_loo`, reads the record that a run wrote, because that record holds data, such as sampler statistics, that the law's content does not determine.
It computes a diagnostic of the law and never a law, so the law's mathematics never depends on an annotation.

### 5.3 How a result relates to its source

> **Definition (relation).** Every operation that returns a term derived from one source term declares one *relation* of its result to that source:
> 1. **same**: the result is the source under another label, container, or declaration. Its paths, levels, and law are those of the source.
> 2. **renamed**: the result is the source with its paths, dimensions, or levels renamed by a map `old → new`. Its law is the source's law under the new names.
> 3. **part**: the result is the part of the source at a set of paths or at a set of batch positions.
> 4. **new**: every other result. It is a different object, related to the source only through the computation that provenance records.

| Relation | Operations |
|---|---|
| same | `with_label`, `copy`, `pickle`, `condition_on`'s re-declaration of a normalized law under the source's event spec, `_with_declaration` |
| renamed | `rename`, `with_dim_names`, `with_level_names`, and `with_path_names` where it exists |
| part | `d[path]`, `d[[paths]]`, `marginal(d, path)`, `record[path]`, `batch[path]`, `batch[i]`, `at_levels`, and `without` where it exists |
| new | `mean`, `sample`, `pushforward`, `condition_on`, `&`, `mixture`, a function call, and every other operation |

`raw` is not in the table, since detachment removes every entry by II.4.
An operation with two operands, such as `&`, has the relation `new` to each.
A joint keeps its operands as factors, and each factor keeps its own annotations.

### 5.4 How an entry follows its term

> **Definition (propagation).** A *propagation* says what an entry becomes under each relation. Three are built in:
> 1. **term**: a fact about the whole term, such as a LOO estimate or a predictive check. It is kept under *same* and *renamed*, and dropped under *part*.
> 2. **run**: a fact about the process that produced the term, such as the sampler's statistics or the observed data. It is kept under *same*, *renamed*, and *part*, since every part of the term comes from the same run.
> 3. **by path**: a mapping from the term's paths to facts about each, such as units per field or R-hat per parameter. Under *same* it is kept; under *renamed* its keys are renamed; under *part* it is restricted to the part's paths.
>
> Every propagation drops its entry under *new*.
> A writer whose entry mixes these, such as ArviZ's tree of groups, supplies a propagation of its own (§ 5.5).

| | same | renamed | part | new |
|---|---|---|---|---|
| term | kept | kept | dropped | dropped |
| run | kept | kept | kept | dropped |
| by path | kept | keys renamed | restricted | dropped |

Restriction keeps the result's own names for its paths.
A marginal keeps its output name (III.7), so the entry `{"mu": ..., "sigma": ...}` restricted to `post["mu"]` is `{"mu": ...}`.
A positional part of a batch, `batch[i]`, has the batch's paths, so a *by path* entry is kept whole.

**Decision 1: the default propagation.**
An entry written with no declared propagation takes *term*.
A fact about the whole is never claimed of a part, and the identity transforms keep it, as xarray's default does.
The alternative, *run*, also carries an entry to every part, which is wrong for a LOO estimate and every other fact about the joint law.

### 5.5 The one implementation

```python
class Propagation(Protocol):
    """How an annotation entry follows its term into a renamed term or a part."""
    def renamed(self, value: Any, renames: Mapping[str, str]) -> Any: ...   # DROP to drop
    def restricted(self, value: Any, part: TermSpec, paths: tuple[str, ...]) -> Any: ...

TERM: Propagation      # renamed: value; restricted: DROP
RUN: Propagation       # renamed: value; restricted: value
BY_PATH: Propagation   # a Mapping keyed by paths: renamed keys; restricted keys

class TrackedTerm:
    @property
    def annotations(self) -> Mapping[str, Any]: ...      # a read-only view; empty when none
    def annotate(self, key: str, value: Any, *, propagation: Propagation = TERM,
                 replace: bool = False) -> None: ...      # the one writer

def _carry_annotations(source: TrackedTerm, result: TrackedTerm, relation: Relation,
                       *, renames: Mapping[str, str] = {}, paths: tuple[str, ...] = ()) -> None:
    """Give *result* the entries of *source* that *relation* keeps, in their new forms."""
```

1. The store moves from `Annotated` onto `TrackedTerm`, as II.4 states, so every batch has one (D1). `Annotated` is deleted.
2. The store holds each entry with its propagation. `annotations` presents the values alone, so `post.annotations["arviz"]` returns a `DataTree` as it does today.
3. Every operation of a *same*, *renamed*, or *part* relation calls `_carry_annotations` once, which replaces the four hand-written sites of § 3.1 and removes D3.
4. One table-driven test builds a term of each kind with a *term*, a *run*, and a *by path* entry, applies each operation of § 5.3, and compares the result's entries with the table of § 5.4.

**Decision 3: an empty mapping, not `None`.**
`annotations` returns an empty read-only mapping on a term with no entries.
Every reader in `diagnostics/` and `validation/` checks for `None` before it reads, and those checks are removed.
A user's `if d.annotations is None` changes meaning, so the change is listed in the CHANGELOG under "Changed".

### 5.6 Writing an entry

`annotate` is the only writer.
It adds `key` with its value and propagation, and it is the one write after construction that II.4 allows.

```python
post.annotate("note", "refit after the data fix")                   # term
d.annotate("units", {"mu": "m", "sigma": "m"}, propagation=BY_PATH)
d["mu"].annotations["units"]                                        # {'mu': 'm'}
```

**Decision 5: append-only, enforced.**
`annotate` raises when `key` is present, unless `replace=True`.
A diagnostic recomputed with `force=True` passes `replace=True`, so overwriting is explicit at each call site.
A writer adds only under its own key, so ProbPipe's writers own the keys `arviz` and `diagnostics`, and a user's key never collides with them unless the user chooses one of them.

## 6. Case by case

### 6.1 Inference results and their marginals

`_record_run` writes two entries, each with a propagation the inference module defines:

| Key | Content | Propagation |
|---|---|---|
| `arviz` | the run's ArviZ groups | *ArviZ*, below |
| `diagnostics` | written later by § 6.2 | *diagnostics*, § 6.2 |

The *ArviZ* propagation follows each group by what it describes:

| Group | Describes | Under *renamed* | Under *part* |
|---|---|---|---|
| `posterior`, `warmup_posterior` | the term's draws, one variable per leaf path | variables renamed | restricted to the part's leaf paths |
| `sample_stats`, `warmup_sample_stats`, `observed_data`, `constant_data` | the run | kept | kept |
| `log_likelihood`, `posterior_predictive`, `predictions` | the joint law | kept | dropped |

With it, the case of § 1 gives what a user expects:

```python
m = post["mu"]
m.annotations["arviz"]["posterior"]       # one variable, mu
m.annotations["arviz"]["sample_stats"]    # the run's statistics, unchanged
"log_likelihood" in m.annotations["arviz"]  # False: the pointwise likelihood is of the joint
az.plot_trace(m.annotations["arviz"])
```

`post.rename(mu="m")` renames the variable `mu` to `m`, and `marginal(post, "mu")` gives the same entries as `post["mu"]`.

**Decision 2: store and restrict, or derive.**
The `posterior` group repeats the atoms (D5).

- (a) **Store and restrict**, as above. It needs no change to what a run stores, and the restriction matches variables to leaf paths by the names `_leaf_variables` gives them. A run whose backend names its variables itself, as CmdStan and PyMC do, has its draws matched by those names, and a variable that matches no leaf path, such as a Stan transformed parameter, is kept under *renamed* and dropped under *part*.
- (b) **Derive.** The `arviz` entry stores only the run groups, and the `posterior` group is built from the term's atoms when it is read. The draws are then stored once, and a marginal or a renamed law has the right group with no restriction rule. The cost is that `annotations` stops being a plain store, and a marginal must expose its draws on the levels `chain` and `draw`, which `CoupledMarginal` does not do today.

This plan recommends (a), and records (b) as later work in `plans/deferred.md`.
(a) fixes every case of D3 with no change to the stored layout, and a test asserts that the `posterior` group equals the atoms.

**Decision 4: `method` leaves the annotations.**
The provenance records the method (D5), and VII.7 asks a summary of how a result was made to derive from provenance.
The root attribute `method` is removed, and the tests that read `annotations.attrs["method"]` read `provenance.metadata["method"]`.

### 6.2 Diagnostic records

`add_rhat`, `add_ess`, `add_mcse`, `add_ppc`, and `add_loo` write under the key `diagnostics`, whose propagation follows its subtrees:

| Subtree | Describes | Under *part* |
|---|---|---|
| `diagnostics/mcmc` | one value per parameter, on the coordinate `param` | restricted to the part's paths, with the attribute `n_divergences` kept |
| `diagnostics/runs` | checks of the whole law, such as LOO and PPC | dropped |

`predictive_check` stops writing into its argument (D6).
It returns its result, and `add_ppc` remains the function that records a check, under `diagnostics/runs/ppc`.
A user who wants the check recorded calls `add_ppc`, whose name says it writes.

### 6.3 Records and arrays

A record or an array carries user entries, such as a source or units.
A field of a record is a *part*, so `record["a"]` keeps *run* entries and the `a` entry of *by path* entries.
`record.rename(a="c")` is *renamed* on every route, so the result of D3 becomes the same for records and for laws.
Host metadata, such as an xarray container's coordinates, stays in the raw host (D8), and V.4's sentence is changed to say so.

### 6.4 Batches

A batch has annotations (D1).
A sweep's result and a sample are *new*, so `sample(post, 4)` starts with no entries.
`batch["mu"]` is a part at a path, and `batch[i]` and `at_levels` are parts at positions, whose paths are the batch's.

The atoms of an empirical law, `post.atoms`, are the law's representation rather than a term derived from it, so they carry no entries.
A user reads the run's record from the law.

### 6.5 Functions and kernels

A `Function` and a `ConditionalDistribution` carry entries as every term does, and their store is empty until `annotate` writes it (D2).
`rename` of a kernel's inputs or outputs is *renamed*, and binding a kernel's inputs, `k(given)`, is *new*.

### 6.6 Computations

Every operation outside the first three rows of § 5.3 is *new*, so its result starts with no entries.
`mean(post)`, `condition_on(post & lik, data)`, and `pushforward(post, f)` carry none of `post`'s entries.
Their provenance names `post`, and in the provenance mode `full` the parent reference holds `post` with its annotations.

## 7. Steps

| Step | Work | Depends on |
|---|---|---|
| A1 | The design text: II.4's annotation paragraph rewritten from §§ 5.2 to 5.6, with the relation table; V.4 step 3a for host metadata; VII.7 for `method` | the decisions |
| A2 | The store on `TrackedTerm`, `annotate`, `Propagation` with its three built-ins, `_carry_annotations`, and the table-driven test of § 5.5 | A1 |
| A3 | Every *same*, *renamed*, and *part* operation calls `_carry_annotations`: `_shallow_copy`, `_with_declaration`, `_as_declared`, `_empirical_of`, the three routes of `rename`, `with_dim_names`, `CoupledMarginal`, `marginal`, and the record and batch selections | A2 |
| A4 | The writers: `_record_run` with the ArviZ propagation and without `method`; `_add_group` and the MCMC writers through `annotate`; `predictive_check` without its write | A2 |
| A5 | The docs: the user guide's paragraph on annotations, `docs/api/provenance.md`, and the CHANGELOG entries for decisions 3 and 4 | A3, A4 |

A2 changes no behaviour visible to a user except decision 3.
A3 is the step that fixes the marginal of § 1.

## 8. Decisions

| # | Question | Recommendation | Section |
|---|---|---|---|
| 1 | The propagation of an entry whose writer declares none | *term* | § 5.4 |
| 2 | ArviZ draws stored and restricted, or derived from the atoms | stored and restricted; derivation as later work | § 6.1 |
| 3 | `annotations` on a term with no entries | an empty read-only mapping | § 5.5 |
| 4 | The inference `method` in annotations | removed; read from provenance | § 6.1 |
| 5 | `annotate` on a key that is present | raises unless `replace=True` | § 5.6 |

## Appendix A. Alternatives not adopted

1. **Keep every entry by default, as xarray does.** A result of a computation would carry records of a different law, such as the R-hat of `post` on `mean(post)`. Provenance already links a result to its input, so keeping the entries adds no information and adds false statements.
2. **Combine the entries of two operands, as `drop_conflicts` does.** A joint `post & lik` is a new law, and agreement between two entries does not make either true of it. Its factors keep their own entries.
3. **A per-call `keep_annotations=` control.** It would let a call carry an entry into a result where the entry is false, and it adds a control to every operation. A user who wants an entry on a new result writes it with `annotate`.
4. **Annotations stored in provenance.** Provenance is write-once and records how a term was made. Diagnostics are written later, about the term as it is, so they need a store of their own.
