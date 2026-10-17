# Declarations: slots, components, and operations on `InputSpec` and `OutputSpec`

> **Status: imported source, not yet reconciled.**
> This file holds the text of [#571](https://github.com/TARPS-group/prob-pipe/issues/571) as it stood on 2026-10-17: the issue body (Part 1) and jhuggins's reply (Part 2), copied without edits.
> The reply changes parts of the body, and [`addressing.md`](addressing.md) (grammar B) changes parts of both.
> [`gaps-and-conflicts.md`](gaps-and-conflicts.md) lists every place where the three texts disagree.
> Once those items are settled, this file and `addressing.md` are replaced by one reconciled text, and #571 points to it.

## Part 1: the issue body

### Summary

A term that consumes what another term produces declares the names it consumes, as a kernel declares its given slots with `given_spec`. The declarations are sound as types. The interface that composition matches has no single public form and no operations, and the vocabulary that describes it is defined only in part. This issue proposes definitions of the declarations and their terms, and then an API that follows from those definitions.

Code references are to `dev/overhaul` at `c02b91ed`.

### Problems with the current design

1. **The interface has three forms with three APIs.**
   - `OutputSpec.components` is a `mappingproxy` with no operations.
   - `InputSpec` is a flat `Mapping` with `without` and `with_optional`.
   - `RecordSpec` has the `NamedTree` operations (`without`, `merge`, `replace`, `with_path_names`), and its `Mapping` interface iterates leaf paths.

   A user converts between the three by hand, and the right conversion depends on whether a declaration names its term whole or offers the term's parts. For example, take a law `prior` over `K`, `r`, `phi`, and `sigma`, and a kernel `shocks_given_sigma` that produces shocks `e` given `sigma`. A likelihood that conditions on `K`, `r`, `phi`, and `e` declares its given slots as follows:

   ```python
   shock_spec = RecordSpec(shocks_given_sigma.event_spec.components)
   likelihood_given_spec = prior.event_spec.spec.without("sigma").merge(shock_spec)
   ```

   `prior.event_spec.spec` is a record because the prior offers the record's parts. The kernel names its event `e` whole, so its `.spec` is a `NumericArraySpec`, and its components are copied into a `RecordSpec` first. The tutorials select components with a dict comprehension in five places, such as `{name: prior.event_spec.components[name] for name in ["K", "r", "phi"]}`, and they combine them with `{**inputs, ...}` in three places.

2. **A record read as slots is read by its leaf paths.** `conditional_distribution` reads `given_spec` with `.items()` (`probpipe/distributions/_conditional.py:1392`), and a `RecordSpec` yields its leaf paths. After the prior is regrouped with `prior.with_path_names({"K": "growth/K", "r": "growth/r"})`, passing `prior.event_spec.spec.without("sigma")` as `given_spec` raises `TypeError: ... unknown parameters ['growth/K', 'growth/r']`. The code in problem 1 works only while the prior is flat. `InputSpec(record_spec)` raises `TypeError` for a `RecordSpec` that `conditional_distribution` accepts.

3. **The public attributes do not state whether a declaration names its term whole.** For a law `d` whose event is `mu`, `OutputSpec(mu=d.spec)` and `OutputSpec(d.spec)` have the same component names, the same `exposes_record`, and the same `.spec`. They differ in the spec of `mu`, which is the law's `DistributionSpec` in the first and the event's `NumericArraySpec` in the second. Telling them apart from public attributes therefore means comparing a component's spec with the term spec, and no attribute states the result. Code outside `_specs.py` reads the private `OutputSpec._component_name` 20 times instead, as `probpipe/functions/_broadcast.py:356` does to detect an exposed law or batch.

4. **A law whose draws are laws can offer the parts of each draw, and its joints fail.** `distribution(sample=..., event_spec=OutputSpec(Normal("mu", 0.0, 1.0).spec))` builds a law whose draws are laws and whose component is `mu`, typed as the `NumericArraySpec` of a draw of the inner law. When the law is composed with `Normal("x", 0.0, 1.0)`, the joint declares the field `mu` as that `NumericArraySpec`, and `sample` of the joint raises `ResultSchemaError: ... output/mu does not conform to NumericArraySpec(...): got Normal`. `RandomMeasure` declares its draws whole and has no such failure.

5. **The library uses private helpers for the operations users lack.**
   - `_components_record` (`probpipe/core/_specs.py:627`) is the conversion of problem 1, with 7 call sites in 6 modules.
   - `_view_declaration` (`probpipe/distributions/_views.py:159`) computes the declaration of a selection and has no public counterpart.
   - The joint's event declaration is assembled inline from its factors' components (`probpipe/distributions/_factored.py:447`).

6. **`event_spec.spec` reads as a spec of a spec.** `OutputSpec.spec` is the type of the produced term, stored as `_term_spec`, and the name does not say so. `.event_spec.spec` occurs on 88 lines of `probpipe/` and 197 lines of `tests/`.

7. **The shape shorthand is accepted in one place.** `RecordSpec(K=())` declares a scalar field. `InputSpec(K=())`, `given_spec={"K": ()}`, and `OutputSpec(K=())` raise `TypeError`, so test modules define their own scalar spec, such as `SCALAR = NumericArraySpec(())`.

8. **A batch of draws drops the names of its event.** II.5 specifies that a batch an operation produces carries the producer's declaration. The code stores a bare `TermSpec` as `BatchSpec.element_spec` (`probpipe/core/_batch.py:197`). So `sample(Normal("mu", 0.0, 1.0), sample_shape=(4,))` has the spec `BatchSpec(NumericArraySpec(...), sample=4)` and no `components`, and `batch["mu"]` raises `TypeError`.

9. **The vocabulary is defined in part.** `design/glossary.md` defines *component*, *given slot*, *optional slot*, and *packaging*. It has no entry for *slot* or *map-like kind*, which II.2 uses to define `InputSpec`. The reference also calls components "produced slots" (III.7, IV.2, and VI.9) and "output slot" (III.7). The user documentation defines none of these terms, since `docs/api/declarations.md` holds one introductory paragraph and the generated API entries.

### Proposed design

#### Definitions

These definitions replace the corresponding text of II.2 and the glossary. Each later part of the design follows from them. The words *leaf*, *field*, *path*, and *child* keep their II.6 meanings, so a field is a leaf and a child is an entry directly under a node.

**Term spec.** A term spec is the type of one term. It is an instance of a `TermSpec` subclass, such as `NumericArraySpec`, `RecordSpec`, or `DistributionSpec` (II.1).

**Producer and consumer.** A producer is a term that declares what it produces, in the sense of II.2's completion rule:

1. **A `Function`** declares its return value as `output_spec`.
2. **A `Distribution` or a `ConditionalDistribution`** declares one draw as `event_spec`.

A consumer is a term that declares what it consumes:

1. **A `Function`** declares its parameters as `input_spec`.
2. **A `ConditionalDistribution`** declares what it conditions on as `given_spec`.

A term may be both, and a `Distribution` is a producer that consumes nothing. *Consumer* replaces *map-like kind*.

**Slot.** A slot is a named place where a consumer receives one value. Its name is a Python identifier that is not a keyword, and its term spec types the values it admits. A slot is **required** or **optional**: a binding may omit an optional slot, and the consumer then uses its default. A `Function`'s slots are its parameters, and a `ConditionalDistribution`'s slots are its **given slots**. A slot that holds a record is one slot, so one value binds to the whole slot.

**`InputSpec`.** An `InputSpec` declares a consumer's slots. It is an ordered mapping from slot names to term specs, together with the set of optional slots. Two `InputSpec`s are equal when they have the same slots, the same specs, and the same optional slots, in any order.

**Component.** A component is a name that a producer offers, together with a term spec, or `None` while a type hole is pending. The term spec types one value at the component, and what the declaration exposes determines which value that is. A component name is a non-empty string without `/`. A component meets a slot only when its name is also a valid slot name.

**`OutputSpec`.** An `OutputSpec` declares one produced term. It states two things:

1. the **term spec** of the produced term, which is `None` only while a type hole is pending;
2. **what it exposes**, which determines the components.

**What an `OutputSpec` exposes.** An `OutputSpec` exposes either its whole term `t`, under one component, or the parts of `t` as its components: a record's children, a law's event components, or a batch element's components. The property `exposes` names the case, and there are four:

| `exposes` | Constructed as | Term spec | Components | The value that a component's spec types |
|---|---|---|---|---|
| `"whole"` | `OutputSpec(c=spec)` | any, or `None` | `{c: spec}` | `t` itself |
| `"record"` | `OutputSpec(record_spec)` | a `RecordSpec` | the record's children | the child `c` of `t` |
| `"event"` | `OutputSpec(law_spec)` | a `DistributionSpec` or a `ConditionalDistributionSpec` | the components of the event declaration | the value at `c` of one draw of `t`, or of one draw of a law that the kernel `t` yields |
| `"element"` | `OutputSpec(batch_spec)` | a `BatchSpec` whose element declaration has components | the components of the element declaration | the value at `c` of one element of `t` |

The keyword form constructs `"whole"`, and the positional form exposes the parts that the kind of its spec determines. The `"event"` and `"element"` cases read the components of a nested declaration, so they apply recursively. Only `"whole"` admits a type hole, so an exposed term always has a complete spec.

**The event of a law.** The `event_spec` of a `Distribution` or a `ConditionalDistribution` exposes `"whole"` or `"record"`. Composition stores the value at each component of a draw as a field of the joint's draw, and only these two cases give that value the component's spec. A law whose draws are laws or batches therefore declares them whole, as `RandomMeasure` does. This rule removes problem 4. The `"event"` and `"element"` cases declare a law or a batch that a `Function` returns (VI.0).

**Paths.** The paths of a declaration start with a component (II.2):

1. **`"whole"`:** the component `c`, followed by the paths within the term, as `c/beta`.
2. **`"record"`:** the paths of the record.
3. **`"event"` and `"element"`:** the paths of the event declaration or of the element declaration.

**Reading a component.** A consumer reads the term at component `c` of a produced term `t` by what the declaration exposes:

1. **`"whole"`:** `t` itself.
2. **`"record"`:** `t.at_path(c)`, which is `t[c]` when the child is a field.
3. **`"event"`, for a law `t`:** `t[c]`, which is the field view of `t` at `c` (III.7). It is `t` itself when the event of `t` is a whole term under `c`.
4. **`"event"`, for a kernel `t`:** the term at `c` of each law that `t` yields once its given slots are bound, since a `ConditionalDistribution` has no indexing.
5. **`"element"`:** the term at `c` of each element of `t`, read by what the element's declaration exposes. For a batch of records this is the column `t[c]` (II.5), and for a batch whose element is a whole term under `c` it is `t` itself. A batch of laws has no indexing by component, so its term at `c` is read element by element.

In cases 3 and 5 the term read is a law over values of the component's spec or a collection of such laws or values. So a consumer that expects a value receives it by lifting (C4).

**Matching.** Composition meets a slot with the component of the same name, and the component's spec must unify with the slot's spec (IV.2). The two roles differ in one rule:

1. **A component is produced once.** Two operands that produce one name raise `ValueError`.
2. **A slot may be shared.** Same-named unmet slots of several operands become one slot, which carries the unification of their specs.

#### `OutputSpec` states its term and what it exposes

`OutputSpec` gains `exposes` and renames `spec` to `term_spec`. `components` and the two constructor forms are unchanged.

```python
class OutputSpec:
    @property
    def term_spec(self) -> TermSpec | None: ...   # the produced term's type; None while a hole is pending
    @property
    def exposes(self) -> Literal["whole", "record", "event", "element"]: ...
    @property
    def components(self) -> Mapping[str, TermSpec | None]: ...   # as today
```

- `exposes` replaces `exposes_record`, which is `exposes == "record"`.
- The component of a `"whole"` declaration is the one key of `components`, which replaces each read of `OutputSpec._component_name`.
- `term_spec` names the attribute by the glossary's term. `prior.event_spec.term_spec` is the type of one draw, and a tracked term's own `spec` is unchanged, so `prior.spec` is still the law's `DistributionSpec`.
- Construction refuses an `"event"` or `"element"` declaration as the `event_spec` of a law or a kernel with `TypeError`, by the rule for the event of a law. No test under `tests/distributions/`, `tests/operations/`, `tests/families/`, or `tests/functions/` constructs such a law today.

#### One reading of slots

Every place that takes a consumer's slots reads them through the `InputSpec` constructor, which accepts four sources:

1. **An `InputSpec`**, as it is.
2. **A mapping** from slot names to term specs.
3. **A `RecordSpec`**, read by its children, which is the rule that the positional form of `OutputSpec` follows.
4. **An `OutputSpec`**, read by its components. A component name that is not a valid slot name raises `ValueError`, and a pending type raises `TypeError`.

Every slot of a mapping, a `RecordSpec`, or an `OutputSpec` source is required. `conditional_distribution`'s `given_spec`, `Function`'s `input_spec`, and `ConditionalDistributionSpec`'s `given_spec` take every source that the constructor takes. A nested record therefore declares one slot per child, which removes problem 2. `conditional_distribution` takes the slots' specs from `given_spec` and keeps its rule for optional slots, so the parameters with defaults are the optional slots. An `InputSpec` source that marks a slot optional whose parameter has no default raises `ValueError`, since the kernel has no default to use.

Wherever a declaration takes a term spec, a tuple of sizes is the `NumericArraySpec` of that shape, as it is for a `RecordSpec` field. This applies to an `InputSpec` slot, to `given_spec`, and to the keyword form of `OutputSpec`. II.1's rule for a `RecordSpec` field applies unchanged, so the shorthand is a tuple only.

#### Operations on declarations

```python
class OutputSpec:
    def project(self, key: str | tuple[str, ...]) -> OutputSpec: ...
    def without(self, *paths: str) -> OutputSpec: ...
    def merge(self, *others: OutputSpec) -> OutputSpec: ...

class InputSpec:
    def merge(self, *others: InputSpec | Mapping | RecordSpec | OutputSpec) -> InputSpec: ...
```

1. **`OutputSpec.project(key)`** is the declaration of the field view `d[key]` (III.7), and its key takes the forms that `d[key]` takes:
   - On a `"whole"` or `"record"` declaration, a path `str` gives its node whole, under a component named by the path's final segment. A tuple of paths, including a tuple of one, gives a `"record"` declaration of their nodes in the order given. For every law `d`, `d[key].event_spec == d.event_spec.project(key)`.
   - On an `"event"` or `"element"` declaration, the projection applies to the nested declaration and exposes the same kind of term, which declares the law or the batch at the key.
   - An empty tuple and two paths with the same final segment raise `ValueError`, and a path that is not a path of the declaration raises `KeyError`.
   - The name differs from `Record.select`, which returns a mapping keyed by full path for splatting into a call (III.5).
2. **`OutputSpec.without(*paths)`** removes the nodes at `paths` from the declaration and exposes the same kind of term, as `RecordSpec.without` removes nodes from a record:
   - A `"whole"` declaration removes nodes within its term, and the result declares a term of the reduced type. Removing its component raises `ValueError`.
   - An `"event"` or `"element"` declaration removes the nodes from its nested declaration.
   - A path that is not a path of the declaration raises `KeyError`, and a removal that leaves no component raises `ValueError`, as `RecordSpec.without` does.
3. **`OutputSpec.merge(*others)`** is the `"record"` declaration whose children are the components of each operand, in operand order:
   - For laws `a` and `b` whose dimensions are concrete, `(a * b).event_spec == a.event_spec.merge(b.event_spec)`, which is the joint's declaration (IV.2).
   - A name that two operands produce raises `ValueError`, by the first matching rule.
   - A pending type and an `"event"` or `"element"` operand raise `TypeError`, since a record child needs the complete spec of the value it holds.
4. **`InputSpec.merge(*others)`** combines slots by the second matching rule, reading each operand as the constructor does:
   - The slots appear in the order of their first occurrence.
   - Same-named slots carry the unification of their specs, which admits only the values every operand admits. Specs that do not unify raise `ValueError`.
   - The operands share one dimension scope, so the sizes that unification binds are substituted in every slot (II.1, IV.2).
   - A slot is optional when every operand that names it holds it optional, which is IV.2's rule for the unmet slots of a joint.
5. **`InputSpec.without(*names)`** raises `KeyError` for a name that is not a slot, as `RecordSpec.without` does for a path. Today it ignores such a name. A removal that leaves no slot stays valid, since a consumer may take no input.

With these operations, the slots of problem 1 are declared without a conversion:

```python
likelihood = conditional_distribution(
    likelihood,
    given_spec=prior.event_spec.without("sigma").merge(shocks_given_sigma.event_spec),
    label="likelihood",
)
```

The shorthand states the same slot names and shapes with weaker types:

```python
given_spec={"K": (), "r": (), "phi": (), "e": (n_years,)}
```

The kernel's slots then set no dtype and no support. Composition still accepts them, since a spec that sets no dtype or support unifies with any dtype and support.

The tutorial idioms map as follows:

| Today | Proposed |
|---|---|
| `{"sigma": prior.event_spec.components["sigma"]}` | `prior.event_spec.project("sigma")` |
| `{name: prior.event_spec.components[name] for name in ["K", "r", "phi"]}` | `prior.event_spec.project(("K", "r", "phi"))` |
| `{**inputs, "e": shocks_given_sigma.event_spec.components["e"]}` | `inputs.merge(shocks_given_sigma.event_spec)`, with `inputs` the projection above |
| `prior.event_spec.spec.without("sigma")` | `prior.event_spec.without("sigma")` |

In the third row, `OutputSpec.merge` raises `ValueError` when `inputs` already holds `e`, where the dict merge replaces the entry.

#### A batch keeps its element's declaration

A batch that an operation produces stores the producer's declaration as its element declaration, as II.5 specifies. The batch of draws of `Normal("mu", 0.0, 1.0)` then has `components == {"mu": NumericArraySpec(...)}`, and `batch["mu"]` returns the batch. A batch built from raw elements keeps the bare element spec that II.5 states. Its components are those of that spec, as they are today: a record's children, a law's event components, and none for an array, an opaque value, or a function. A batch with no components cannot be exposed, so `OutputSpec(BatchSpec(NumericArraySpec(()), sample=3))` raises `TypeError`, as it does today. `Batch` gains the `components` property that II.5 lists.

#### Vocabulary

1. **"What a declaration exposes" replaces "packaging".** The glossary defines *packaging* as "whether a declaration exposes a record's fields as its components or names one whole term", and the code uses `exposes_record` and `_exposed_kind`, so the prose and the API use one verb. The verb also avoids the sense of Python packaging.
2. **"Component" replaces "produced slot" and "output slot".** A slot is a place where a consumer receives a value.
3. **A grouped sub-joint replaces a packaged sub-joint (IV.1), and IV.2's "packaged joint".** The rename that gathers components under one node creates it. IV.2's paragraph "Packing the joint" becomes "The joint's event".
4. **The glossary gains entries for *slot*, *producer*, *consumer*, and the verb *exposes*, and it removes *packaging*.** The entry for *component* takes the definition above and keeps its second sense, the component laws of a `MixtureDistribution` (VII.3).

#### Alternatives considered

1. **`components` returns a `RecordSpec`.** A `RecordSpec` indexes leaf paths, so `components["growth"]` would raise for a component that is a record, and every reader of `components` would change.
2. **`RecordSpec` replaces `InputSpec`.** A record has no optional fields and no rule for identifier names, and its `merge` raises on a shared name, where slots unify.
3. **A given slot's type stays pending until composition fills it.** `conditional_distribution` evaluates the function at stand-ins of each slot's type to read its event declaration and its claims (IV.4), and III.7 requires a complete declaration. A kernel with a pending slot would therefore have no event declaration.
4. **`given_spec` holds specs for more names than the function has parameters.** This removes the need for `without`, and it accepts a misspelled key without an error.
5. **A law's event exposes laws, and each operation keeps what the declaration exposes.** A field of the joint's draw would then hold a law, and the joint's declaration would type it by the inner law's event, which is problem 4.

#### Open questions

1. Are `"whole"`, `"record"`, `"event"`, and `"element"` the clearest values of `exposes`?
2. Should the projection be `OutputSpec.__getitem__`, so that `d[key].event_spec == d.event_spec[key]` mirrors law indexing? Today `components[c]` returns a term spec, and `event_spec[c]` would return an `OutputSpec`.
3. Does `with_spec` become `with_term_spec`, so the attribute and the method that completes it share one name?
4. Are `spec` and `exposes_record` removed outright, or kept for one release as deprecated aliases?
5. Are *component* and *slot* the clearest names for the two roles, or should they be *produced slot* and *consumed slot*? The alternative names the API `OutputSpec.produced_slots` in place of `components`, and `consumed_slots` for the slots that an `InputSpec` declares.
   - The paired names state that composition matches a produced slot with the consumed slot of the same name, and the reference already says "produced slots" in III.7, IV.2, and VI.9.
   - The current names give each role its own word, so a slot is always a place where a consumer receives a value. They also keep `components`, which `event_spec.components` alone reads on about 380 lines of `probpipe/`, `tests/`, and `docs/`.
   - The answer decides item 2 of the vocabulary section, the definitions of *slot* and *component*, and the wording of problem 9.

### Required updates to the code

Each item names the change and the condition that completes it.

- [ ] **The design reference.**
  - II.2 owns the definitions, the table of what an `OutputSpec` exposes, the rule for the event of a law, the properties `exposes` and `term_spec`, the sources of slots, and the operations.
  - II.1 states the tuple shorthand once for every declaration.
  - II.5 states that an operation's batch stores the producer's declaration, and which components a batch of raw elements has.
  - II.6 states `without(*paths)` as the code implements it.
  - III.3, III.9, and IV.4 point to II.2 for the sources that `input_spec` and `given_spec` take.
  - III.7 defines a field view's declaration as `event_spec.project(key)` and points to II.2 for the rule for the event of a law.
  - IV.1 and IV.2 use the new vocabulary and define the joint's event with `merge`.
  - The glossary changes as listed above.
  - Each use of "packaging", "packaged", "produced slot", "output slot", and "map-like kind" in parts II through VII changes, in about 30 places, including the II.2 row of Part II's summary table. "Input slot" becomes "slot".

  *(Done when `tests/docs/` and the four `test_design_conformance.py` files pass, and each candidate that `scripts/design/prose.py` reports is resolved under `STYLE_GUIDE.md` §10.)*
- [ ] **`OutputSpec.exposes` and `term_spec`.**
  - Add both to `probpipe/core/_specs.py`, with `_exposed_kind` as the source of `exposes`.
  - Replace the reads of `exposes_record`, on 32 lines of `probpipe/` and 30 of `tests/`.
  - Replace the reads of `OutputSpec._component_name`, on 19 lines of `probpipe/` outside `_specs.py`.
  - Rename `spec` to `term_spec` at each read. These include `.event_spec.spec` on 88 lines of `probpipe/` and 197 of `tests/`, and `.output_spec.spec` on 4 and 24.
  - Refuse an `"event"` or `"element"` event declaration in the constructors of `Distribution`, `ConditionalDistribution`, `DistributionSpec`, and `ConditionalDistributionSpec`.
  - Update the docstrings and messages in `probpipe/` that say "packaging", "packaged", or "map-like kind", such as the `InputSpec` docstring.

  *(Done when no module reads `OutputSpec._component_name` outside `_specs.py`, the law of problem 4 raises `TypeError` at construction, and the full suite passes.)*
- [ ] **One reading of slots.**
  - The `InputSpec` constructor accepts the four sources and the tuple shorthand.
  - `_slots_of` and `_complete_given_spec` in `probpipe/distributions/_conditional.py`, and the `input_spec` reading at `probpipe/values/_function_base.py:1093`, construct through it.
  - The keyword form of `OutputSpec` accepts the tuple shorthand.

  *(Done when tests show that a nested `RecordSpec` and an `OutputSpec` each declare a kernel's slots, that the regrouped prior of problem 2 builds a kernel, and that an `OutputSpec` source with a non-identifier component or a type hole raises.)*
- [ ] **The operations.**
  - Add `OutputSpec.project`, `without`, and `merge`, and `InputSpec.merge`.
  - `InputSpec.without` raises on an unknown name. Its callers at `probpipe/distributions/_factored.py:2463`, `probpipe/distributions/_views.py:2954`, and `probpipe/distributions/_conditional.py:1156` pass computed names, so each is checked to pass only slots.
  - `_view_declaration` becomes `OutputSpec.project`, and the joint's event declaration in `probpipe/distributions/_factored.py` uses `merge`.
  - Callers of `_components_record` that need a record of the components use `RecordSpec(declaration.components)`.

  *(Done when tests cover each raise listed above and check the identities for `project` and `merge`.)*
- [ ] **The batch's element declaration.**
  - `BatchSpec` stores the producer's `OutputSpec` for a batch that an operation produces: draws (VI.3) and sweeps (V.6). It keeps the bare spec for raw elements.
  - `Batch` gains `components`.
  - A column of a batch of records, such as `rb["a"]`, carries the declaration `OutputSpec(a=...)`.
  - `BatchSpec` hashes its element declaration without its component names, as a `FunctionSpec` hashes its output declaration (`probpipe/core/_fingerprint.py:436-439`), so a rename of components keeps replay keys. The new stored form invalidates existing cached results once.
  - Two batches whose elements differ only in component names stop being equal.

  *(Done when the batch of draws of `Normal("mu", 0.0, 1.0)` has `components == {"mu": NumericArraySpec(...)}` and `batch["mu"]` returns the batch.)*
- [ ] **The documentation.**
  - `docs/api/declarations.md` opens with the definitions of slot and component and of what a declaration exposes.
  - The tutorials use the operations of this issue in place of the dict comprehensions.
  - `CHANGELOG.md` records the renames and the event rule as breaking changes.

  *(Done when `uv run mkdocs build --strict` and the notebooks job pass.)*

## Part 2: jhuggins's reply (2026-10-15)

Source: https://github.com/TARPS-group/prob-pipe/issues/571#issuecomment-6193839296

This is a clear improvement. The definitions, the single reading of slots, the rule for a law's event, and the batch's element declaration should go in as written. Four changes would make the design uniform across kinds.

### 1. `OutputSpec` is a mapping of its components

Composition matches slots with components by name, so both declarations map names to term specs. `OutputSpec` becomes a `Mapping`, as `InputSpec` is, and `components` goes.

```python
prior = joint(Normal("mu", 0.0, 5.0), HalfCauchy("tau", 0.0, 5.0))
list(prior.event_spec)                        # ['mu', 'tau']
prior.event_spec["tau"]                       # the term spec of tau
prior.event_spec.project("tau")               # OutputSpec(tau=...), the declaration of prior["tau"]
InputSpec(prior.event_spec)                   # the slots mu and tau
likelihood.given_spec.keys() & prior.event_spec.keys()   # the slots the prior meets
```

- Indexing reads a component and `project` builds a declaration, so `d[key].event_spec == d.event_spec.project(key)`. This answers open question 2.
- Equality compares `term_spec` and whether the declaration exposes parts. `OutputSpec(parameters=R)` and `OutputSpec(RecordSpec(parameters=R))` have the same components and produce different terms.
- Equality stays order-sensitive, because component order is the layout that `to_vector` reads. `InputSpec` equality ignores order, because binding reads slots by name.
- `components`, `spec`, and `exposes_record` are removed, with no deprecated aliases.

### 2. One rule for exposure

A spec exposes parts when it carries a declaration of what its term produces. `OutputSpec(T)` exposes that declaration's components, and component `c` of a term is the term that yields its output at `c`.

| Kind | Produced declaration | Component `c` of a term `t` |
|---|---|---|
| `RecordSpec` | its children | the child `t.at_path(c)` |
| `DistributionSpec` | `event_spec` | the field view `t[c]` |
| `ConditionalDistributionSpec` | `event_spec` | the kernel whose law at each given value is the field view at `c` |
| `FunctionSpec` | `output_spec` | the function `x ↦ t(x)[c]` |
| `BatchSpec` | the element declaration | each element's part at `c` |
| `NumericArraySpec`, `OpaqueSpec` | none | named whole only |

Each kind implements three methods, and `project`, `without`, path reading, and matching use only these:

```python
class TermSpec:
    def produced_declaration(self) -> OutputSpec | None:   # None for a kind without parts
        return None

    def with_produced_declaration(self, declaration: OutputSpec) -> Self:
        raise TypeError(f"a {type(self).__name__} has no parts")


class TrackedTerm:
    def _component(self, component: str) -> TrackedTerm:    # the term at component
        raise TypeError(f"a {type(self).__name__} has no components")


class FunctionSpec(TermSpec):
    def produced_declaration(self) -> OutputSpec | None:
        return self.output_spec

    def with_produced_declaration(self, declaration: OutputSpec) -> FunctionSpec:
        return FunctionSpec(self.input_spec, declaration)


class OutputSpec(Mapping[str, TermSpec | None]):
    def project(self, key: str | tuple[str, ...]) -> OutputSpec:
        if not self.exposes_parts or isinstance(self.term_spec, RecordSpec):
            return _project_named(self, key)                 # the rule this issue states
        inner = self.term_spec.produced_declaration().project(key)
        return OutputSpec(self.term_spec.with_produced_declaration(inner))
```

A function that returns a function then exposes the returned function's outputs:

```python
fit_emulator = Function(train_gp, output_spec=OutputSpec(FunctionSpec(
    InputSpec(theta=theta_spec), OutputSpec(RecordSpec(mean=("n_days",), var=("n_days",))))))
list(fit_emulator.output_spec)                # ['mean', 'var']
emulator = fit_emulator(design=design, runs=runs)
emulator["mean"]                              # the Function theta ↦ emulator(theta)["mean"]
```

- `exposes` has two values, so it becomes the boolean `exposes_parts`, and the kind of the parts is `type(term_spec)`. This answers open question 1.
- The rule for a law's event reads: a declaration whose components become fields of a record exposes the parts of a record only. A law's event is such a declaration, since the joint's draw stores each component as a field.
- A consumer that expects one value at `c` receives the term at `c` through a lifting rule, which C4 gives for laws and batches. A function's parts need a rule that lifts by postcomposition. Without it, they meet only slots that declare functions.

### 3. `&` merges every kind

`*` is arithmetic on `NumericArray`, `NumericRecord`, and `LinOp`, and a law's draw is a record. So `d1 * d2` composes while `sample(d1) * sample(d2)` multiplies. `&` reads as "both", and it replaces `*` for composition and both `merge` methods.

`a & b` holds, produces, and requires what `a` and `b` do:

1. A name both operands produce raises `ValueError`.
2. A name both operands consume is one slot. Its specs unify, and it is optional only when optional in both.
3. For laws and kernels, a name the left operand consumes and the right produces is matched (IV.2). Functions pair and never wire, so for them such a name raises.
4. The operands share one dimension scope.

| Operands | `a & b` |
|---|---|
| `Record`, `RecordSpec`, `OutputSpec` | both sets of fields or components |
| `InputSpec` | both sets of slots |
| `Distribution`, `ConditionalDistribution`, and their specs | the joint, and the joint's spec |
| `Function`, `FunctionSpec` | the function that returns both outputs |
| `Batch`, `BatchSpec` | element by element, over equal levels |
| `NumericArray` | NumPy's elementwise `&`, unchanged |

For every kind, `(a & b).spec == a.spec & b.spec`. For two `Distribution`s, `(d1 & d2).event_spec == d1.event_spec & d2.event_spec`, with symbolic dimensions too. For kernels the given side is `A.given_spec.without(*(A.given_spec.keys() & B.event_spec.keys())) & B.given_spec`.

```python
# two likelihoods that share their parameters
data_model = lik_nee & lik_lai
list(data_model.given_spec)                   # ['theta']: one shared slot
list(data_model.event_spec)                   # ['nee', 'lai']
data_model.given_spec == lik_nee.given_spec & lik_lai.given_spec
model = data_model & prior                    # the prior meets theta

# two summaries of the same data, as one Function
center = Function(lambda y: jnp.mean(y), output_spec=OutputSpec(center=None))
spread = Function(lambda y: jnp.std(y), output_spec=OutputSpec(spread=None))
summaries = center & spread                   # the slot y, the outputs center and spread

# posterior draws beside the replicates simulated from them
draws = sample(posterior, sample_shape=(1000,))    # elements {mu, tau}
y_rep = simulate(draws)                            # elements {y}
discrepancy(draws & y_rep)                         # elements {mu, tau, y}
```

`pushforward(d, f & g)` is the joint law of `f(X)` and `g(X)`, which share `X`. `pushforward(d, f) & pushforward(d, g)` is their independent product.

### 4. Two more operations

- **`InputSpec.with_path_names`** renames slots, as indexing a transition kernel by time does.
- **`&` on `DistributionSpec`, `ConditionalDistributionSpec`, and `FunctionSpec`** lets a function that builds a model declare its output before it runs:

```python
step = observation.spec & transition.spec     # y and x, given x_prev and theta
renames = lambda t: {"x": f"x{t}", "y": f"y{t}", "x_prev": f"x{t - 1}"}
chain = functools.reduce(operator.and_, [step.with_path_names(renames(t)) for t in (3, 2, 1)])
build_model = Function(make_chain, output_spec=OutputSpec(model=chain & initial.spec & prior.spec))
```

### Open questions

1. `exposes` becomes the boolean `exposes_parts`.
2. Indexing reads a component, and `project` builds a declaration.
3. Yes, `with_term_spec`.
4. No deprecated aliases.
5. Keep *component* and *slot*. A slot meets the component with its key, which the mapping form shows directly.

---
🤖 AI-generated by Claude (Anthropic) at Jonathan Huggins's request. Verify before relying on it.
