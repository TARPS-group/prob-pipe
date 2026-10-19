# Outline of the rewritten declarations design

This outlines the rewrite of [`declarations-and-addressing-io.md`](../declarations-and-addressing-io.md) at commit `8af5bd11`, the source.
The design is held fixed; an ambiguity found while writing goes to Appendix A.2 with the reading the text adopts.
[`sample.md`](sample.md) shows the front matter and §§ 3, 5, and 14 in final form.

## Rules for the whole document

1. **Succinct.** One clear sentence over two; no introductions that announce, no restating.
2. **Words.** Part I speaks of *objects* (this law, this record) and their *kinds*: base value, record, law, kernel, function, batch. It never says "term". Part II introduces the code: an object is represented by a *tracked term*, its type by a *term spec*. Both parts say *input*, *output*, *input spec*, *output spec*, *event spec*, *path*, *part*, *merge*, and *feeds*.
3. **Generic names** in every example: `a`, `b`, `c`, `x`, `y`, `z`, `mu`, `sigma`, `tau`, `theta`, `data`, `params`, `g` for a group. The one exception is `lik & prior` in the motivating example of § 5, where the names state the roles.
4. **Boxes.** `> [!IMPORTANT]` holds a definition, starting `**Definition (…).**`. `> [!NOTE]` holds consequences or properties, several in one box as a numbered list; a single item gets its own box only when it carries weight. `> [!TIP]` holds intuition and `> [!WARNING]` a sharp edge.
5. **Technical notes** are `<details><summary>Technical note: …</summary>` blocks, with blank lines around the Markdown inside.
6. **Figures** are `![alt](../figures/<name>.svg)`, each followed by an italic caption line `*Figure: …*`. An invalid merge has a red boundary and cross, and its caption gives the reason in one line.
7. **Math** sparingly: names in code font, set notation (∩, ∪, ∅, −) where it says what prose cannot, and `$…$` or `$$…$$` for the few formulas left.
8. **No history in the body.** Steps, transition rules, today's code, and earlier plans go to Appendices B to D.

Figures the generator does not draw yet (`held-field`, `select-one`, `law-of-laws`, `function-outputs`, `batch-columns`) are added with their sections.

## Front matter

- Title "Inputs, outputs, paths, and merge"; a summary in plain words (what the design settles, who it is for, what each part holds); the contents.
- **Source:** § 1.1, without the grammar label or the lemma names.

## Part I: The conceptual model

No code objects. Notation is defined where introduced: records as `{a: 1.0, g: {b: 2.0}}`, signatures as `a(y | mu)`, selections as `x[p]` and `x[[p, q]]`, merges as `a & b`.

### 1. Values and the places inside them

| Subsection | Defines | States | Figures | Examples | Notes | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 1.1 Base values and records | *base value* (an array or an opaque value); *record*; *name* | — | — | `2.5`; `"forest"`; `{a: 1.0, g: {b: 2.0, c: 3.0}}` | — | § 2.1, § 2.3; III.1, III.2, III.5 |
| 1.2 Paths, nodes, fields, and groups | *node*, *root*, *path*, *child*, *canonical order*, *field*, *group* | every node has one path | `record-tree` | the paths, fields, and group of `{a, g: {b, c}}` | — | § 2.1, § 2.4 |
| 1.3 The parts of a value | *part* (a node below the root) | a base value has no parts and no paths | — | four parts, and none | "Why an array's entries are not parts" | § 2.3, § 4.1 |

### 2. Objects, kinds, and types

| Subsection | Defines | States | Figures | Examples | Notes | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 2.1 Objects and their kinds | *object*; *kind*, with the six kinds; *label* (a display name no operation reads) | — | — | one per kind | — | § 2.1 |
| 2.2 What an object produces | *produced object* (a value itself; a law a draw; a kernel a draw at each input value; a function its return value; a batch itself); *draw*, *event* | statements about a law's produced object are about the law of a draw | — | a draw `{mu, sigma}` | — | § 2.5 |
| 2.3 The type of an object | *type*; *base type*; *record type*; *event type*; the types of kernels, functions, batches; *dimension name*, *dimension scope* | every law has an event type, sampleable or not | — | the type of a record; `("n", "m")` with `("m",)` | "Dimension scopes in full" | § 2.3, § 2.8 (without `held`), § 2.14; II.1 |
| 2.4 The paths of every object | *paths of a type*, by kind; *paths of an object*; *column* | paths continue through a field holding a law, kernel, function, or batch; a function with an unknown return type has no known paths | `held-field` | `{x: 3.0, g: d}` for a law `d` over `{mu, sigma}` | "The paths of a type, case by case" (§ 2.10 without `spec_at`) | § 2.4, § 2.5, § 2.10 |
| 2.5 When one type admits another | *conforms*; *unify*, *unification* | one box: a type conforms to the unification exactly when it conforms to both; unification is symmetric and associative; a size-only type conforms to any type of its shape | — | `float32` with `float64`; `(3,)` against `("n",)` | "Conformance by kind"; "Unification by kind" | § 2.14 |

### 3. Inputs and outputs (in [`sample.md`](sample.md))

| Subsection | Defines | States | Figures | Examples | Notes | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 3.1 Inputs | *input*; *optional input*; *binding* | one box: which kinds have inputs; binding a kernel's last required input gives a law; a record input is one input | — | `a(y \| mu, sigma)` with `sigma` optional | "Binding part of a record input" | § 2.6; III.9, IV.4 |
| 3.2 Outputs | *output*; *value at an output* | — | — | — | — | § 2.11 |
| 3.3 How outputs are named | *the naming rule*; *output at the root*; *wrap* | one box: one output at the root or one-name paths, never both; the root name is not a path; a lone base value has no outputs | `output-naming` | seven one-line cases | "Why a function into records offers outputs and a law does not"; "Names of outputs and inputs" | § 1.1, § 2.7, § 2.9, § 2.11; Q2(ii) |
| 3.4 Input specs and output specs | *input spec*; *output spec*, *event spec*; *type of an output*; *type hole*; *record of outputs* | one box: the type decides whether there is a root name; each output's type is read from the output spec; the record of outputs gives back the produced object | — | the specs of `a(y \| mu)` and of a law over `{mu, sigma}` | "Equality and order" | § 2.6, § 2.7, § 2.11, § 2.12 (without `rec`/`dec`), § 4.14 |
| 3.5 Signatures and their figures | *signature* | — | `legend-objects`, `signature`, `record-parts` | `a(y \| mu, sigma)` | — | § 2.1 (notation) |
| 3.6 Inputs and outputs compared | — | the comparison table | — | — | — | § 2.23 (rows not about merge or code) |

### 4. Selecting, removing, and renaming parts

| Subsection | Defines | States | Figures | Examples | Notes | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 4.1 Selecting one part | *selection* `x[p]` (a value's node; for other kinds, the object of the same kind whose produced object is the node); *marginal law* | one box: selecting commutes with producing (with sampling, for a law); paths chain; inputs are kept; positions and paths commute on a batch | `select-one` | a law over `{a, g: {b, c}}` at `g` and `g/b` | — | § 4.2; I2, I4, I5, I13, I14 |
| 4.2 Selecting several parts | *selection at a list of paths* | one box: one path gives the node, a list of one path a record holding it; same-named parameters in two groups are kept | — | `x[[a, g/b]]`; `x[[s1/theta, s2/theta]]` | "Through a held object"; "The root name in a list" | § 4.3, § 4.4 rule 3; I9; O6 |
| 4.3 Removing parts | *removal* | selection and removal partition the leaf paths | — | — | — | § 4.5; I10 |
| 4.4 Selecting from a type | *selection from an output spec* | one box: it gives the output spec of the selection; it adds no name; dimension names are kept | — | three cases | "Case by case" (§ 2.10, § 4.4, without `held`, `restrict`, `remove` as notation); "Dimension names" | § 2.10, § 4.4, § 4.5, § 4.10; I6 |
| 4.5 Renaming and moving parts | *rename* | one box: no key is both a name and a path; a root name cannot become a nested path | — | three renames | "How renames apply" (II.6) | § 4.7, § 4.8 |
| 4.6 The object at an output | *the object at an output* | its type is the output's type | — | — | — | § 4.11; I12; Q5 |

### 5. Merging objects by name (in [`sample.md`](sample.md))

| Subsection | Defines | States | Figures | Examples | Notes | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 5.1 Why merge | — | — | — | `lik & prior` and its density | — | § 2.15 |
| 5.2 The definition | *family*; *feeds*; *shared input*; *merge*; *valid merge* | — | `legend-merge`, `merge-general`, `merge-kernel-law`, `merge-shared-input` | the three figures | "Factors and the inputs a law's factors still take"; "Merging types"; "Objects of two families" | § 2.13, § 2.15 to § 2.19 |
| 5.3 Optional inputs | — | an optional input stays open unless fed; a later merge can feed one left at its default | `merge-optional-open`, `merge-optional-fed` | — | "A shared optional input that a binding omits"; "A merge leaving only optional inputs" | § 2.19, § 4.16 |
| 5.4 The value of a merge | *the law of a merge* | records merge by top-level name | `merge-records` | records; functions; batches | "Batches, element by element"; "A deep merge by hand" | § 2.16, § 2.20, § 5.7, § 5.8 |
| 5.5 Refused merges | — | the table of conditions and reasons | `merge-invalid-output`, `merge-invalid-order`, `merge-invalid-conform`, `merge-invalid-unify`, `merge-invalid-law` | the five figures | "Why outputs feed right to left"; "A merge never lifts"; "Two selections of one law" | § 2.17, § 2.21 |
| 5.6 Properties | — | one box with eight properties | — | paths kept | "When the validity of two groupings differs"; "Renaming"; "Optional inputs across merges"; "Dimension scopes across merges" | § 4.15 to § 4.17; I15 to I22 |
| 5.7 What merge is not | — | not composition, conditioning, or renaming | — | — | — | § 2.22 |

### 6. The kinds, one by one

Each subsection: what it is; what it produces; its parts; its inputs; its outputs; what merge means for it; a figure; an example. It points to §§ 1 to 5 for the general rules.

| Subsection | Defines | States | Figures | Notes | Source |
| --- | --- | --- | --- | --- | --- |
| 6.1 Base values and records | — | one box | `record-parts` | — | § 5.2 to § 5.4 |
| 6.2 Laws | *conditioning*; *summary* | one box: selection commutes with sampling, from the event type; conditioning removes what it fixes; summaries keep paths; a law of laws offers nothing; families wrap their draws | `law-of-laws` | "Conditioning at nested paths"; "Scoring a summary" | § 5.5, § 2.11; I2, I3, I7, I8, I11 |
| 6.3 Kernels | — | one box | `signature` | "A kernel and the function returning its laws" | § 5.6, § 2.9 |
| 6.4 Functions | — | one box: outputs of the return value; returning a law offers nothing; functions merge as point-mass kernels | `function-outputs` | "An unknown return type" | § 5.7 |
| 6.5 Batches | *element type*; *equal levels* | one box | `batch-columns` | "Why names stay out of a batch's equality" | § 5.8, § 2.9, § 6.21 |

### 7. Invariants

| Subsection | Defines | States | Notes | Source |
| --- | --- | --- | --- | --- |
| 7.1 The invariants | *equal* | I1 to I22 in one table: each with the definitions it follows from, and whether it follows or is chosen | "What each proof rests on" | § 3, without "Delivered by" |
| 7.2 Exceptions on purpose | — | one collapsible table | — | § 6.3 (conceptual rows) |

## Part II: The design in code

Each class appears by its API, with definitions in docstrings, then short standalone examples in generic names.

| Section | Holds | Source |
| --- | --- | --- |
| 8. Reading Part II | *tracked term* (the code form of an object) and *term spec* (the code form of a type); one table of the names this document writes against the names the code uses until the rename (slot, component, the declaration words, `components`, `component(c)`, `component=`); the full `legend` | § 1.4, § 8.2; Q1 |
| 9. Term specs | `TermSpec` and its classes; the size shorthand | § 2.3, § 2.8, § 4.9, § 5.12 |
| 10. `InputSpec` | API: four sources, `[]`, `in`, `get`, `without`, `&`, `rename`, equality; the four messages | § 5.10, § 6.1, § 2.6 |
| 11. `OutputSpec` | API: two forms, `term_spec`, `outputs`, `[]` (four rules), `in`, no iteration, `without`, `&`, `with_term_spec`, `rename`, `default`, equality; type holes case by case | § 5.11, § 4.4, § 4.12, § 4.14, § 6.1 |
| 12. Indexing in code | the path reader, its refusals and messages; `[]` per class; `in`; operations taking paths; `select` | § 4.1, § 4.2, § 4.6 to § 4.8 |
| 13. Merging in code | `__and__` per class; error type and message per condition | § 2.16, § 2.17, § 6.1 |
| 14. Records (in [`sample.md`](sample.md)) | `RecordSpec` and `Record` API; examples; errors | § 5.4, § 5.9, § 4.3, § 4.5, § 4.6, § 6.1 |
| 15. Laws | `Distribution`, `MarginalView`, `marginal`, `condition_on`, `log_prob`, summaries, families | § 5.5, § 6.1, § 6.2.1 to § 6.2.3 |
| 16. Kernels | `ConditionalDistribution`, `conditional_distribution`; inputs declared from output specs | § 5.6, § 6.2.4 (in generic names) |
| 17. Functions | `Function`; `OutputSpec.default` by kind; a bare `output_spec`; lifting | § 5.7, § 4.12, § 4.13, § 6.2.5, § 6.2.6 |
| 18. Batches | `Batch`, `BatchSpec`, `element_spec`; names out of equality and JAX | § 5.8, § 6.1, § 6.21 |
| 19. Errors and messages | every error, by operation; code-level exceptions | § 4.1, § 5.10, § 2.17, § 6.3 |
| 20. Symbolic dimensions in code | `with_dim_sizes`, `with_dim_names`, `free_dims`; stored scopes | § 4.10, § 4.17 |

## Appendices

### A. Open questions and issues

**A.1 Open questions:** "feeds" or "binds to" ("bind" already means supplying a value in a call or `condition_on`); the questions deferred to after the workshop (Q4(b) option 3, Q4(d), Q6); the pending items of `decisions.md` that touch this design, one line each: N5, N12, N15, N16, N17, N18, N19, N20, N21, N29, N31, N32.

**A.2 Open questions and issues found while writing**, each with the reading the text adopts:
1. **When the library stops building the named-whole form.** Source § 6.8 says G1b; T19, § 5.5, and the probe allowance say G3, and the source marks T19 "Inconsistent". Reading: G3.
2. **"Has parts" without parts.** The source calls every law and batch an object "with parts", including a law over one array and a batch of arrays, which have no paths. Reading: a part is a node below the root, and the rule reads "only a base value is named from outside"; behaviour is unchanged.
3. **`in` on a function with an unknown return type.** `f[s]` succeeds and checks at the call (§ 5.7), while its type has no paths (§ 2.10); I18 ties `in` to `[]`. Reading: none; both rules are stated.
4. **`without` on batches of functions or kernels** is not defined (§ 4.5). Reading: not defined.
5. **`in` on an opaque value** is not stated. Reading: `False`, by I18.
6. **A record node inside a held law** is not clearly a field or a group (§ 2.4). Reading: neither; `without` removes a held field whose leaf paths it covers.
7. **The word for IV.2's "unmet".** The source has S0 write "unfed" into `design/`; this text says "remaining inputs". Reading: Appendix B keeps the source's instruction, pending this question.
8. **Whether a law's type and a kernel's type merge.** Source § 2.16 says "two of one of these". Reading: they merge, as their objects do.

Recorded in Appendix B, not as a question: property 3 gives `a & b` and `b & a` equal input specs once a shared input carries the unification, while T29 keeps the first operand's type until after the workshop.

### B. Implementation plan and sequencing

B.1 The steps S0 to G5 and the rename: delivers, done-condition, files, tests, demonstration, invariants delivered. B.2 Rules for every step, the invariant tests, the interim-form probe. B.3 Transition rules T1 to T31, each with its ending step, and the interim whole form. B.4 The words before the rename; changes to `design/`, `STYLE_GUIDE.md`, `docs/`, and the glossary. B.5 What the merge gates; the cost of the naming rule.
**Source:** § 6.4, § 6.4.1, § 6.5, § 6.7 to § 6.17, § 8.

### C. Differences from today's code

C.1 `*`, `merge`, `with_path_names`, `FieldView` and their replacements. C.2 The renames table. C.3 Behaviour changes, from every "Change from today" and "Today" block. C.4 The problems that motivate the design (§ 1.2). C.5 The CHANGELOG list.
**Source:** § 1.2, § 6.1 (table), § 6.4, § 6.6.

### D. Relation to previous plans

Grammars A, B, B′; #571 and jhuggins's reply; the addressing plan (#573); the inputs and outputs proposal and its proofs; `declarations-and-addressing.md` and the decisions reversed; O1 to O16, R1 to R6, Q1 to Q6, and `element_spec`.
**Source:** § 7, § 6.18 to § 6.21.

### E. Alternatives considered

**Source:** § 6.22.

### F. Status and history

Status, sources and precedence, commits checked, scope, relation to the execution plan.
**Source:** header, § 1.3, § 1.4.

## Dependency order

Each section uses only what earlier sections define. Changes from the source's order:
1. **Legend.** `legend-objects` appears in § 3.5, after inputs and outputs; `legend-merge` in § 5.2, after merge; § 1.2 draws a bare tree.
2. **Conformance and unification** move to § 2.5, before inputs and merge use them.
3. **Columns** are defined in § 2.4 as the node of a batch at a path, before selection.
4. **Type hole** (§ 3.4) precedes condition 2 of § 5.2; **factor** is defined in a note of § 5.2; **rename** (§ 4.5) precedes the grouped sub-joint; **marginal law** (§ 4.1) precedes conditioning (§ 6.2); **equal** is defined in § 7.1.
