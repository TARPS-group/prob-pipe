# Inputs and outputs: what terms produce and receive, and merging by name

> **Status: proposal, revised 2026-10-18 after three rounds of adversarial review**, for Andrew and jhuggins to read and decide on.
> It is one design, synthesized from three independent proposals (A, B, and C) that agreed on the core; Appendix D records what each chose differently and why this text does not.
> It changes `plans/declarations-and-addressing.md`, here called **the plan**. Appendix C lists every section of the plan whose text changes and gives the replacement text of each step whose plan done-condition becomes false; every other part of the plan stands.
> § 3 lists every decision it reverses, with its date and source, and the open question of § 14 that asks it.
> It gates step S0 and the declaration steps after it, not `plans/` itself (step PL) or the execution steps (§ 13).
> **Merged 2026-10-18** into [`declarations-and-addressing-io.md`](declarations-and-addressing-io.md), with the recommended answer to each question of § 14, provisionally. At the merge, `element_output_spec` was renamed `element_spec` (decisions, 2026-10-18): today's `BatchSpec.element_spec`, the element's term spec, becomes `element_spec.term_spec`, and `element_spec` is the output spec of one element as a term, `held` of the batch's spec (§ 5.6, § 5.7). This text was revised to match; the merged plan's § 21.4 explains the choice.
> Claims about today's code cite `file:line` at `6f8859fc` and were checked by running scripts with `uv run python` from the worktree root; Appendix E lists them.
> Before the workshop it keeps today's behaviour wherever a corner is hard, states that behaviour's consequences (§ 8.10, § 8.11), and leaves the long-term choice to an open question.

## Contents

1. Summary
2. How this answers the maintainer's questions
3. Decisions this proposal reverses
4. Foundations
5. Definitions
6. Every kind placed
7. Indexing: selecting and removing
8. Merge
9. Invariants
10. API sketch
11. Worked examples
12. Differences from `declarations-and-addressing.md`
13. Effect on the push
14. Open questions

Appendices: A. Proofs of the merge properties. B. Deliberate exceptions. C. Edits to the plan and to `design/`. D. Alternatives considered. E. Facts checked against the code.

## 1. Summary

### 1.1 The design in one page

This page is an overview; § 5 defines each term before any rule uses it.

**Every term produces a term.**
A law produces a draw, a random value.
A kernel produces a draw of the law it gives at each value of its inputs.
A function produces its return value at each value of its inputs.
An array, an opaque value, a record, or a batch produces itself.
This one rule (§ 5.4) lets every later rule be stated once.

**Two plain nouns, one per side.**
- An **input** is a named place where a term receives one value. `lik`'s input is `mu`.
- An **output** is a name under which a term offers a value it produces. `prior`'s outputs are `mu` and `tau`.

An **input spec**, the class `InputSpec`, lists a term's inputs.
An **output spec**, the class `OutputSpec`, states the term spec of the term's produced term and, when that term spec is a base spec or pending, the one name it is offered under; until step G2 an interim form also names a structured term (§ 5.5).
These replace "slot", "component", "input declaration", and "output declaration"; § 14, Q1 asks when the code and the design text change words.

**The spec of an output is the spec of the produced term's value at that output.**
A record offers its children, a batch of values its columns, and a function that returns records one function per field.
A produced term that is a law or a kernel offers no outputs, because a joint law is not the record of its marginals (D1).

**`[]` means one thing on each kind.**
On a record, `r[s]` is the node at `s` itself.
On a law, a kernel, a function, or a batch, `t[s]` is the term of the same kind whose produced term is the node at `s` of `t`'s produced term.
`OutputSpec[s]` gives the output spec of `t[s]`, so `d[s].event_spec == d.event_spec[s]` holds as written, and `project` is gone.
`without(*paths)` removes nodes, on both sides.

**One merge.**
`a & b` puts the outputs of both operands side by side.
An input that both operands have becomes one input.
An output of `b` **feeds** the input of `a` with the same name, and that input leaves the result.
Eight numbered conditions guard it (§ 8.3), of which three carry the design:
1. no name is an output of both operands, since one value needs one source;
2. the right operand takes no input that the left one outputs, which is IV.2's order rule, omitted from the plan's § 14.2;
3. a fed output's spec conforms to its input's, and the specs of a shared input unify, in one dimension scope.

One definition gives the joint density for laws and kernels, the union of fields for records, both return values for functions, and, for batches, the merge of elements position by position, each element read as one factor.
Functions feed, because a function is a kernel whose laws are point masses.

**Two corners keep today's behaviour before the workshop.**
A merge of laws and kernels reads its operands as their factors, as IV.2's flattening does, so an optional input that one merge left at its default can be fed, or shared, by a later merge (§ 8.10).
A merged law or kernel keeps its dimension scope, while a merged input spec or function does not (§ 8.11).
Both have consequences that § 8.10 and § 8.11 state in full, and § 14, Q4(b) and Q6, ask what to do after the workshop.

### 1.2 Main differences from the plan

§ 3 lists the decisions reversed, and § 12 every other difference.

1. **Vocabulary.** Input and output, input spec and output spec, produced term, and feeds. "Slot", "component" (apart from a mixture's component laws), "producer", "consumer", "address", "whole declaration", "exposed declaration", "component spec", and "packaging" are not terms. `t.output(c)` replaces the decided `t.component(c)`, and `element_spec`, an output spec, replaces O15's `element_declaration`; `bt.element_spec.outputs` replaces II.5's `Batch.components`.
2. **A produced term that is a law or a kernel offers no outputs.** The law's event rule and the wrap of a view at a held law go, so I2, I5, I6, and I9 hold without exception once G3 has merged.
3. **`OutputSpec[s]` replaces `project`.** `OutputSpec` is not a `Mapping`; its attribute `outputs` replaces `components`.
4. **Merge has one definition**, with IV.2's order rule restated and its flattening stated as today's behaviour. Functions feed, and merge only with complete specs. Batches merge by their elements.
5. **`&` on records is shallow**, as on laws, from S2b.
6. **`InputSpec` gains list selection**, and `&` on input specs takes two input specs.
7. **The word "unify" keeps one sense**, IV.2's symmetric one, and S0 writes "conforms" for the directional check and "completes" for II.2's completion (§ 12.3).

**Effect on the push.**
No step is added, and the order and the cut stand, but a decision PR on this proposal now precedes S0, at the head of a never-cut chain (§ 13 states the schedule risk).
S2b keeps the plan's content, minus the event rule, plus three small items (§ 13.1).
S4 also types a batch's outputs by their columns; G1b, G2, and G3 shrink; G4 gains paths into a record input.
Under § 14, Q1's recommended option the vocabulary changes in one PR after the workshop, so S0 and S2a keep today's words.
This proposal gates S0, and PL can merge without waiting for it (§ 13).

## 2. How this answers the maintainer's questions

1. **"The plan never gives a general, precise definition of merge."**
   § 8 defines `a & b` once, from each operand's input spec and output spec (§ 5.7) and its produced term (§ 5.4).
   It has eight numbered conditions, one rule for the result's inputs and outputs, and one value rule per kind of operand, all written in the notation of § 5.10.
   Every word it uses (input, output, produced term, feeds) is defined for every kind before § 8, and § 8.10 states where today's flattening makes a merge read more than its operands' specs.
2. **"`project` has no complete contract, and its name does not describe what it does."**
   `project` becomes `[]` on an output spec.
   § 7.3 gives its whole contract: one path, a list of paths, the output at the root, and paths through held laws, functions, and batches, with order and errors; § 7.8 gives symbolic dimensions.
   No new name is needed.
3. **"Are records producers, consumers, both, or neither?"**
   A record produces itself, its outputs are its children, and it has no inputs.
   In a merge a record only offers values, so `r1 & r2` puts both sets of fields side by side and nothing is fed.
   A `RecordSpec` has one reading, as outputs: `InputSpec(R)` equals `InputSpec(OutputSpec(R))`, the inputs those outputs would feed.
4. **"What does merging non-distribution terms mean?"**
   For records, the fields of both.
   For functions, the function that runs the right operand, feeds its outputs to the left operand's inputs of the same names, runs the left operand, and returns both return values.
   For batches, the merge of their elements, position by position, each element read as one factor.
   For specs, the spec of the merged term.
   Functions feed because a function is a kernel whose laws are point masses (§ 8.7); a merge differs from the composition `f ∘ g` because it returns both operands' outputs.
5. **"Terminology must be consistent across the two sides."**
   Each side has one noun (input, output), one spec class (`InputSpec`, `OutputSpec`), and the same operations (`[]` with one name or a list, `without`, and `&`).
   One word, *feeds*, names the relation between them, and *conforms* and *unify* name the two checks on specs.
   § 5.13 lists every remaining asymmetry with its reason.
6. **"The word 'declaration', and the name `project`."**
   "Declaration" leaves this area: the objects are named after their classes, *input spec* and *output spec*, and "declaration" keeps its broad sense of a static claim (II.7), as `plans/execution-layer.md` uses it.
   `project` is removed in favour of `[]`.

## 3. Decisions this proposal reverses

README § When to stop and ask requires the maintainer's answer before any of these changes.
Each row names the question of § 14 that asks it.

| # | Decision | Date and source | This proposal | Asked in |
| --- | --- | --- | --- | --- |
| 1 | The words "slot" and "component", "input declaration" and "output declaration", and the glossary terms producer, consumer, exposes, whole declaration, exposed declaration, address, and component spec | the plan's § 4.3, § 4.4, and § 17, accepted 2026-10-18 (`decisions.md`, "accepted every recommendation of `maintainer-review.md`") | input, output, input spec, output spec; none of the glossary terms (Appendix C.3) | Q1 |
| 2 | The accessor `t.component(c)` | `decisions.md`, 2026-10-17 | `t.output(c)` on terms; no accessor on output specs (§ 7.7) | Q1, Q5 |
| 3 | `BatchSpec.element_declaration` (O15), and `Batch.components` (II.5; the plan's § 15.2 and S4) | the plan's O15 and S4, accepted 2026-10-18; II.5 | `element_spec`, the output spec of one element as a term, whose `term_spec` is today's `element_spec` (§ 5.7) | Q1 |
| 4 | The outputs of a produced term that is a law: its event's components, typed by the views | the plan's § 4.4 and § 5.3, accepted 2026-10-18 | none (§ 5.9) | Q2(ii) |
| 5 | The law's event rule: "#571's rule that a law's event is whole or an exposed record stands unchanged" | `decisions.md`, 2026-10-17; the plan's § 5.4 | removed | Q2(i) |
| 6 | "A view at or below a field that holds a law or a function wraps, keeping the field's name" | `decisions.md`, 2026-10-17; the plan's § 5.3, R3, and O5 | no view wraps (§ 7.2) | Q2(i) |
| 7 | Two defaults, an output default and a private event default that wraps (the plan's alternative 17 rejects one default) | the plan's § 5.5 and § 20, accepted 2026-10-18 | one `OutputSpec.default`; the private event default keeps today's rule until G2 and then goes | Q2(i) |
| 8 | `OutputSpec` is a `Mapping` of component names, with `project`; `components` is removed | the plan's § 7.3 and § 7.5, and README's S2a, accepted 2026-10-18 | indexed by paths; the attribute `outputs` replaces `components` (§ 7.3) | Q3(i) |
| 9 | Either operand of `InputSpec &` may be any of the four sources | the plan's § 6.4, accepted 2026-10-18 | both operands are `InputSpec`s (§ 8.6) | Q3(ii) |
| 10 | `&` on records keeps `NamedTree.merge`'s deep rule | the plan's R2, accepted 2026-10-18 | shallow (§ 8.8) | Q4(a) |
| 11 | `&` on records arrives in the mechanical S3, and `&` on batches after the workshop | the plan's § 19.5 and § 14.4, accepted 2026-10-18 | S2b makes `merge` shallow on records and on `RecordBatch`, and S3 renames it `&` | Q4(a) |
| 12 | Functions "pair and never wire" | the plan's § 14.2, rule 3, accepted 2026-10-18 | functions feed, after the workshop (§ 8.7) | Q4(c) |
| 13 | Two batches of arrays raise or combine bitwise under `&` | the plan's § 16, accepted 2026-10-18 | they merge into a batch of records, element by element, after the workshop (§ 8.2) | Q4(a) |

**Kept, with a restated reason.**
The lifted law-returning function keeps the plan's G1b done-condition: its draws are `{law_of_y: law}`, under the callable's output name (§ 11.6).
The reason is that the lift builds its own law and names its own record (C5), as `include_inputs=True` already does today.
The decision of 2026-10-17, "a function with a wrapped output returns the record itself … ProbPipe never wraps a returned value", is about a function body's return value, and it stands.

**Kept, unchanged.**
O9 (accepted 2026-10-18): `Function` reads optional inputs by the kernel's rule.
The plan's § 6.2: a source that marks optional an input whose parameter has no default raises `ValueError`.
III.7 and III.9: a law has no inputs, and a kernel has at least one; binding a kernel's last required input gives a law whose optional inputs take their defaults (IV.4).
II.1: dimension names bind in one scope, and two different names at one axis raise.
IV.2's flattening: a merge of laws and kernels reads its operands' factors (§ 8.10 states its consequences).
Every other accepted item of `decisions.md` and of the plan's § 21 stands, apart from the rows above and the items Appendix C.1 marks as lapsing (O1, O5, O14, O15, R2, R3).

## 4. Foundations

### 4.1 Concepts taken from `design/`

This proposal builds on these concepts, in the sense their sections give them.

| Concept | Meaning, briefly | Source |
| --- | --- | --- |
| **term** | Any object ProbPipe tracks: a value, a law, a kernel, a function, or a batch | II.4 |
| **kind** | A sort of term, identified by the class of its term spec | II.1 |
| **term spec** | The type of one term, a `TermSpec`, such as `NumericArraySpec((3,))`, `RecordSpec(a=())`, or `DistributionSpec(...)` | II.1 |
| **symbolic dimension**, **dimension scope** | A dimension size given by a name, such as `"n_obs"`; specs in one scope bind each name to one size, and a later occurrence that disagrees raises | II.1 |
| **name** | A non-empty `str` without `/` | II.2, II.6 |
| **identifier** | A name that is a Python identifier and not a keyword | III.3; the plan's § 4.3 |
| **path** | The `/`-joined names from the top of a structured value down to one position in it; always a `str` | II.6; decided 2026-10-15 |
| **child**, **canonical order** | An entry directly under a node; the depth-first order that visits children in insertion order | II.6 |
| **record**, `RecordSpec` | A named tree of values, the class `Record`; its spec, a tree of term specs | III.5 |
| **law**, **draw** | A probability distribution, the class `Distribution`, which has no inputs; one value sampled from it | III.7 |
| **kernel** | A conditional distribution `K : S → P(T)`, the class `ConditionalDistribution`, with at least one input: a law for each value of what it conditions on | III.9 |
| **function** | A tracked callable, the class `Function`, whose `FunctionSpec` holds an `input_spec` and an `output_spec`, either of which may be `None` | III.3 |
| **batch**, **element**, **level**, **position** | Many separate terms of one kind, the class `Batch`; one of them; a named group of batch axes; an integer, a slice, or a tuple of them | II.5 |
| **label** | A display name for a term, which no operation reads | C5, II.4 |
| **factor**, **factored joint** | One of the laws or kernels a joint is built from; a joint that keeps its factors, `FactoredDistribution` or `FactoredConditionalDistribution` | IV.1 |
| **grouped sub-joint** | The one factor that a rename gathering components under one node makes of their factors, whose event is that node as one whole output (IV.1 calls it a packaged sub-joint; the plan's § 17 renames it) | IV.1 |
| **view**, **marginal** | `d[s]`, the law of part of `d`'s draws, which refers to `d`; `marginal(d, s)`, the same law detached | III.7, VI.8 |
| **lifting** | Applying a function to a law or a batch where it expects a value | C4 |
| **distribution family** | A class of laws with its own constructor, such as `Normal` or `RandomMeasure` | VII |

The principles cited are C1 (uniform interface), C3 (detail hidden by default), C4 (lifting), C5 (names), D1 (mathematical fidelity), D2 (generality first), D5 (carried structure), and D6 (single source of truth), from `design/01-design-principles.md`.

**The path reader.**
The plan's § 8 reader is kept, and every `[]`, `marginal`, and `without` reads its argument through it:
- a `str` is one path;
- a list or tuple of `str` is a list of paths;
- an integer, a slice, or a tuple of them is a position, on a batch only.

Its errors are kept too.
An empty path or segment, an empty list, a path given twice, and a path inside another raise `ValueError`.
A path that does not exist raises `KeyError`.
A path and a position in one `[]`, a list of positions, a set, and an iterator raise `TypeError`.
Appendix C.1 lists the two sentences of the plan's § 8 that change, since they name `project` and say an output spec is indexed by name.

### 4.2 The running examples

These are the plan's § 2 examples, written in the proposed API.
"Today" lines use today's `*` and `with_path_names`, and show what `6f8859fc` returns.

```python
prior = Normal("mu", 0.0, 5.0) & Normal("tau", 1.0, 1.0)          # draws {mu, tau}
grouped = (Normal("a", 0.0, 1.0) & Normal("c", 0.0, 1.0) & Normal("e", 0.0, 1.0)) \
    .rename({"c": "g/c", "e": "g/e"})                              # draws {a, g: {c, e}}
hier = (Normal("t1", 0.0, 1.0) & Normal("t2", 0.0, 1.0) & Normal("mu", 0.0, 1.0)) \
    .rename({"t1": "s1/theta", "t2": "s2/theta"})                  # draws {s1: {theta}, s2: {theta}, mu}
lik = conditional_distribution(lambda mu: Normal("y", mu, 1.0), given_spec={"mu": ()}, label="lik")
model = lik & prior                                                # draws {y, mu, tau}
post = condition_on(model, {"y": y_observed})                      # draws {mu, tau}

def lik_opt_fn(mu, s=1.0):
    return Normal("y", mu, s)
lik_opt = conditional_distribution(lik_opt_fn, given_spec={"mu": (), "s": ()}, label="lik_opt")
s_prior = Normal("s", 100.0, 0.001)                                # a law over s, near 100

growth_prior = (LogNormal("K", jnp.log(350.0), 0.4) & LogNormal("r", jnp.log(0.4), 0.4)
                & Beta("phi", 2.0, 2.0) & HalfNormal("sigma", 0.2))   # draws {K, r, phi, sigma}
shocks_given_sigma = conditional_distribution(
    lambda sigma: Normal("e", jnp.zeros(n_years), sigma),
    given_spec=growth_prior.event_spec["sigma"])                   # draws e, given sigma
```

`lik_opt` is a kernel with the required input `mu` and the optional input `s`, whose default is `1.0`.

Other terms, whose construction does not matter to the examples:
- `y_observed` and `data`: observed arrays; `n_years`: a number of years;
- `emulator`: a function from `theta` to a record `{mean, var}`, returned by a function `fit_emulator`, which a callable `train_gp` implements; `theta`: a value of its input, and `theta_spec` its spec;
- `rm`: a random measure, a law whose draws are records `{G: law}`, where each drawn law has the spec `law_spec` and draws `{mu, tau}`;
- `lik_g`: a kernel `lik_g(y | g)` whose input `g` takes a record `{c, e}`; `lik_G`: a kernel whose input `G` takes a law of spec `law_spec`;
- `lik_nee` and `lik_lai`: kernels that produce `nee` and `lai` given the same input `theta`;
- `likelihood`: a kernel that produces `y` given `K`, `r`, `phi`, and `e`, built in § 11.4;
- `run_inference`: a plain Python function that returns a law like `post`;
- `draws`: `sample(post, sample_shape=1000)`, a batch of records `{mu, tau}`; `simulate`: a function from `mu` and `tau` to a record `{y}`, so `simulate(draws)` is a batch of records `{y}` over the same level.

## 5. Definitions

Each definition uses § 4 and the definitions before it, and ends with an example; where one needs a later operation, it names the section that defines it.
Some definitions recurse on the nesting of specs; the recursion ends because specs nest finitely.

### 5.1 Base specs and specs with parts

A **base spec** is a `NumericArraySpec` or an `OpaqueSpec`: the type of a term with no named parts.
A term whose spec is a base spec is a **base value**: an array or an opaque value.

A **spec with parts** is one of five:
1. a `RecordSpec` with at least one child;
2. a `DistributionSpec`;
3. a `ConditionalDistributionSpec`;
4. a `FunctionSpec`;
5. a `BatchSpec`.

An empty `RecordSpec` is neither, and an output spec refuses it (the plan's O10, kept).

```python
NumericArraySpec(())                          # a base spec
RecordSpec(a=(), g=RecordSpec(c=(), e=()))    # a spec with parts
post.spec                                     # a spec with parts: a DistributionSpec
```

### 5.2 Nodes, fields, and groups

These keep the plan's § 4.2 meanings, apart from the root.
- A **node** is the position that a path names. The **root** is the term itself; it has no path.
- A **field** is a node of a record, named through records only, that is not itself a record. A field may hold a law, a kernel, a function, or a batch.
- A **group** is a node of a record, other than the root, that is itself a record.
- A **leaf path** is a path with no path below it.
- `last(p)` is the last name of the path `p`.

```python
# in a draw of grouped, {a, g: {c, e}}:
# nodes a, g, g/c, g/e; fields a, g/c, g/e; one group, g; leaf paths a, g/c, g/e; last("g/c") == "c"
```

### 5.3 Inputs, input specs, and binding

An **input** is a named place where a term receives one value.
It has:
- a name, which is an identifier;
- a term spec, which types the values it admits;
- a flag: **required**, or **optional**.

**Binding** supplies values to a term's inputs by name: a call of a function, or `condition_on(k, given)` and the fused calls `sample(k, given=...)`, `log_prob(k, y, given=...)`, and `mean(k, given=...)` of a kernel (III.9).
A binding may omit an optional input, and the term then uses its own default.
Binding a kernel's last required input gives a law, whose unbound optional inputs take their defaults (III.9, IV.4): `condition_on(lik_opt, {"mu": 0.3})` is a `Normal` with `s = 1.0` (Appendix E, fact 24).
An input whose spec is a `RecordSpec` is one input, so one record fills it whole, until step G4 adds partial binding of a record input (III.9).

An **input spec** is an `InputSpec`: an ordered mapping from input names to term specs, together with the set of optional inputs.
Two input specs are equal when they have the same inputs, specs, and optional inputs, in any order, since binding reads inputs by name.
An input spec may have no inputs, and `InputSpec(RecordSpec())` is `InputSpec()`.
An `InputSpec` stores no defaults; the term that owns an optional input holds its default.

**Which inputs are optional** (the plan's § 6.2 and O9, kept).
`conditional_distribution` and `Function` mark as optional each input whose parameter has a default (`_slots_of`, `distributions/_conditional.py:1366`, for kernels; O9 extends the rule to functions).
A source that marks optional an input whose parameter has no default raises `ValueError`, since the term has no default to use.
A kernel's inputs may all be optional, since III.9 requires only a non-empty `given_spec`; such a kernel is still a kernel, and `sample` asks for its inputs (Appendix E, facts 33 and 37).

```python
InputSpec(mu=(), s=()).with_optional("s")     # two inputs, s optional (the size shorthand is § 7.4)
lik.given_spec                                # InputSpec(mu=NumericArraySpec(())): one required input
lik_opt.given_spec                            # InputSpec(mu=..., s=...).with_optional("s")
condition_on(lik_opt, {"mu": 0.3})            # a law: Normal("y", 0.3, 1.0), with s at its default
```

### 5.4 The produced term, and columns

The **produced term** of a term is what one evaluation of it yields:

| Term | Its produced term |
| --- | --- |
| an array, an opaque value, or a record | the term itself |
| a law `d` | a random draw `X ~ d` |
| a kernel `k` | at each value `w` of its inputs, a random draw `X ~ k(w)` |
| a function `f` | at each value `w` of its inputs, its return value `f(w)` |
| a batch | the batch itself |

Here `w` is an **input record**: a record with one field for each input it binds.
The produced term of a law or a kernel is random, so a statement about it is a statement about the law of `X`.
Its type is the term spec its output spec states (§ 5.5).
A law's produced term is a draw, not the law, and a kernel's is a draw, not the law it gives; a joint law is a law over a joint draw, so this is the reading a merge needs.

The **column** of a batch at a path `p` is the batch, over the same levels, of each element's `[p]` (§ 7.1): the node at `p` for an element that is a value, and the view, kernel, or function at `p` for an element that is a law, a kernel, or a function.
A batch's paths are its elements' paths (§ 5.8), and the node of a batch at `p` is its column at `p`.

```python
sample(prior)                        # one draw of prior's produced term: Record {mu, tau}
sample(lik, given={"mu": 0.3})       # one draw of lik's produced term at mu = 0.3: an array
emulator(theta)                      # emulator's produced term at theta: Record {mean, var}
draws["mu"]                          # the column of draws at mu: a batch of arrays over the level sample
```

### 5.5 Output specs and grammar B

An **output spec** is an `OutputSpec`.
It states two things about a term's produced term:
1. its **term spec**, read as `term_spec`;
2. a **name**, when that term spec is a base spec, when it is pending in a type hole (below), and, until step G2, in an interim whole form (below).

When the output spec is in the keyword form over a base spec, the name is the term's one output, and that output is **at the root**: the whole produced term is offered under it.
The constructor has two forms:
1. **the keyword form**, `OutputSpec(x=T)`, for a base spec `T`;
2. **the positional form**, `OutputSpec(T)`, for a spec with parts `T`.

From step G2 the term spec alone decides the form.
Today the positional form refuses a `FunctionSpec` and a `BatchSpec` of base elements (`core/_specs.py:355`); step G1b accepts them, after S4 for batches (the plan's § 7.2).

**Grammar B.**
A name is stated outside a value only when the value is a base value, which has no paths for the name to disagree with.
A record carries its names inside it, as its children.
A law, a kernel, a function, or a batch carries no name of its own.
To give such a term one name, the code that returns it **wraps** it: it returns the one-field record `{name: term}`, declared `OutputSpec(RecordSpec(name=T))`.
ProbPipe never wraps a function body's return value itself (decided 2026-10-17).
This is the plan's grammar B (§ 5.2), restated; B, not B′, stays decided.

**The interim whole form, until step G2.**
Until G2, the keyword form also accepts a spec with parts, as today (the plan's § 5.2); this proposal calls `OutputSpec(n=T)`, for a spec with parts `T`, an **interim whole form**.
Today's random measures, random functions, Gaussian processes, bootstrap laws, empirical laws of laws, and minibatched targets declare their events this way, and so does the grouped sub-joint factor of a regrouped law, as `grouped`'s factor `OutputSpec(g=NumericRecordSpec(c=..., e=...))` (Appendix E, fact 23).
Its rules are today's component-rooted reading (the plan's § 5.2), stated here in full:
1. **outputs:** one output, `n`, at the root, whose spec is `T`;
2. **paths:** `n`, and `n/q` for each path `q` of `T` (§ 5.8);
3. **selection:** `O[n]` is `O`; `O["n/q"]` is the output spec at `q` of `T`, built as § 5.8's `E_p` is, from `N = spec_at(T, q)`, privately, so the positional form's public refusal of a `FunctionSpec` until G1b does not apply; a list `ps` of these paths gives `OutputSpec(restrict(RecordSpec(n=T), ps))`, with `restrict` as § 7.3 defines it;
4. **removal:** `O.without(n)` raises `ValueError`, since nothing would be left; for paths `n/q`, `O.without(*ps)` is `OutputSpec(n=remove(T, qs))`, with `remove` as § 7.5 defines it and `qs` the paths below `n`, so the name stays;
5. **the record of outputs** (§ 5.10) is `{n: v}`, so a merge stores the produced term whole, as the field `n` (§ 8.3, M2).

Grammar B's guarantee (§ 5.9) and I17 do not hold for an interim whole form, which § 9 lists among the deliberate exceptions.
The grouped sub-joint's interim form ends with G1a, which wraps it (the plan's § 5.8); the G2 families' ends with G2, which moves them to wraps and then refuses the form.

A **type hole** is the keyword form with `None` in place of the term spec, as `OutputSpec(mean=None)`: its name is stated, and its term spec is pending.
The function or law that owns it fills it with `with_term_spec` once it knows the type:
- filled with a base spec, it stays in the keyword form;
- filled with a batch built from raw values, whose elements a batch constructor names `element` by default (the plan's § 15.2), it names the elements by the hole's name (the plan's O2, kept);
- filled with any other spec with parts, it raises `TypeError` from G2, and gives an interim whole form until then.

```python
OutputSpec(x=NumericArraySpec(()))                             # keyword form: one output x, at the root
OutputSpec(RecordSpec(beta=(), sigma=()))                      # positional form over a record
OutputSpec(RecordSpec(params=RecordSpec(beta=(), sigma=())))   # the record, wrapped under params
OutputSpec(post.spec)                                          # positional form over a law: no name
P = OutputSpec(params=RecordSpec(beta=(), sigma=()))           # an interim whole form until G2; TypeError from G2
P["params"] is P; P["params/beta"]                             # P itself; OutputSpec(beta=...)
OutputSpec(mean=None)                                          # a type hole
```

### 5.6 The output spec that a spec holds

Four specs with parts type a term that produces something, and each holds the output spec of what that term produces.
This proposal writes `held(T)` for the output spec that `T` holds, and `with_held(T, E)` for `T` with `held(T)` replaced by the output spec `E` and everything else kept:

| `T` | `held(T)` | It describes | `with_held` keeps |
| --- | --- | --- | --- |
| `DistributionSpec` | `T.event_spec` | one draw | — |
| `ConditionalDistributionSpec` | `T.event_spec` | one draw, at one value of the inputs | `given_spec` |
| `FunctionSpec` | `T.output_spec`, which may be `None` | one return value | `input_spec` |
| `BatchSpec` | `T.element_spec`, the output spec of one element as a term (§ 5.7) | one element | the levels |

For a `BatchSpec`, `with_held(T, E)` stores `E` as `element_spec`, which keeps a base element's name.
A `RecordSpec` holds no output spec, since its children are its structure, and a base spec holds none.
`held` and `with_held` are notation in this document; in code they are private helpers of `core/_specs.py`, which jhuggins's reply calls `produced_declaration` and `with_produced_declaration` (the plan's O11 keeps them private).

```python
held(post.spec)                                            # OutputSpec(RecordSpec(mu=..., tau=...))
with_held(post.spec, OutputSpec(mu=NumericArraySpec(())))  # the spec of a law whose draws are the array mu
```

### 5.7 The input spec and the output spec of each term

| Term `t` | Its input spec | Its output spec |
| --- | --- | --- |
| an array or an opaque value | none | none: a base value carries no name, and a label is not one (C5) |
| a record `r` | none | `OutputSpec(r.spec)`, derived and never stored (D6); none for an empty record |
| a law `d` | none (III.7) | `d.event_spec` |
| a kernel `k` | `k.given_spec`, with at least one input (III.9) | `k.event_spec` |
| a function `f` | `f.input_spec`, or `None` when undeclared | `f.output_spec`, or `None` when undeclared |
| a batch `bt` | its elements' input spec, read from the element spec: `given_spec` for a batch of kernels (III.10), `input_spec` for a batch of functions, and none for a batch of values or laws | `OutputSpec(bt.spec)`, derived, whose outputs are its columns (§ 5.9) |

In formulas, `input_spec(t)` is `t`'s input spec, or the empty `InputSpec()` when `t` has none; § 8.1's `names(I)` gives the set of an input spec's names.
`output_spec(t)` is `t`'s output spec.

**The elements' own output spec.**
`bt.element_spec` is the output spec of one element as a term, `held(bt.spec)`, and `bt.element_spec.term_spec` is the element's term spec, which today's `BatchSpec.element_spec` holds:
- for an element that is a base value, which has no output spec of its own, the keyword form under the name its maker gave or `element` (the plan's § 15, O2, O3), the one thing `element_spec` stores beyond the term spec;
- for an element that is a record, the record's own output spec, `OutputSpec(element_spec.term_spec)`;
- for a law, a kernel, or a function, the positional form over its spec.

So `bt.element_spec == output_spec(bt[j])` at every position `j` of a batch of values, the batch supplying a base element's name. For a batch of laws, kernels, or functions, each element's own output spec is the one its term spec holds: `element_spec.term_spec.event_spec`, III.10's `DistributionBatch.event_spec` and `ConditionalDistributionBatch.event_spec`, or `element_spec.term_spec.output_spec`.
The batch's own output spec, `OutputSpec(bt.spec)`, describes the batch as a produced term and offers its columns; `[]`, `t.output(c)`, and `InputSpec(o)` read it, while a merge of batches reads the elements' own specs (§ 8.2).

**The attribute names.**
Each side has two attribute names, and each name says the role of what it describes:
- `event_spec` is the output spec of a law or a kernel, and describes a draw. A kernel and the function that returns its laws are one map `S → P(T)`, but their produced terms differ, a draw and a law, and the two names keep that difference visible.
- `output_spec` is the output spec of a function, and describes its return value.
- `element_spec` is the output spec of one element, above, as `event_spec` is of one draw.
- `given_spec` is the input spec of a kernel, named for what it is conditioned on, the word its own calls use: `condition_on(k, given)` and `sample(k, given=...)`.
- `input_spec` is the input spec of a function, and describes its arguments.

III.9 distinguishes conditioning from application, and D1 asks that a real distinction stay real in the interface, so the names stay.
Each is still an input spec or an output spec, and every rule below reads them as such.

**Disjoint sides.**
A kernel never has a name that is both an input and an output: `_check_disjoint_sides` refuses one at construction (`distributions/_conditional.py:131`).
A function does not check this today: `Function(lambda x: x, input_spec={"x": ...}, output_spec=OutputSpec(x=None))` builds.
Merge refuses such an operand (§ 8.3, M5).

```python
lik.given_spec          # its input spec: the input mu
lik.event_spec          # its output spec: OutputSpec(y=NumericArraySpec(())), one output y at the root
prior.event_spec        # OutputSpec(RecordSpec(mu=..., tau=...))
draws.spec.element_spec             # OutputSpec(RecordSpec(mu=..., tau=...)): one element
```

### 5.8 Paths, and the spec at a path

For a term spec `T` and a path `p`, the **spec at `p`**, written `spec_at(T, p)`, is a term spec.
Let `c` be the first name of `p`, and `rest` what follows `c/` in `p`, which is empty when `p` is `c`:
1. **`T` is a `RecordSpec`.** Let `child` be `T`'s child `c`; when `T` has no child `c`, `p` is not a path of `T`. When `rest` is empty, `spec_at(T, p)` is `child`. Otherwise `spec_at(T, p)` is `spec_at(child, rest)`.
2. **`T` holds an output spec `E` in the positional form.** Let `N = spec_at(E.term_spec, p)`. Then `spec_at(T, p)` is `with_held(T, E_p)`, where `E_p`, **the output spec of `E` at `p`**, is `OutputSpec(**{last(p): N})` when `N` is a base spec and `OutputSpec(N)` otherwise.
3. **`T` holds an interim whole form `E = OutputSpec(n=U)`, until G2.** `spec_at(T, n)` is `T` itself, and `spec_at(T, n + "/" + q)` is `with_held(T, OutputSpec(U)[q])`, for each path `q` of `U` (§ 5.5, rule 3).
4. **In every other case**, `p` is not a path of `T`: a base spec has no paths; a held output spec in the keyword form over a base spec names its value at the root, with nothing below the name; and a `FunctionSpec` that declares no output has no paths known before it is called (the plan's § 9.6).

The **paths** of `T` are the paths at which `spec_at` is defined.
The paths of an output spec `O` in the positional form are the paths of `O.term_spec`; an output spec in the keyword form over a base spec has none, since the name of an output at the root is not a path; an interim whole form has the paths of § 5.5, rule 2.
The paths of a term are the paths of its output spec.

Case 2 is how a path goes through every kind (decided 2026-10-16).
Below a held law, function, or batch, the spec at a path is the spec of a term of the same kind whose produced term is the node at that path.
So a law's paths are its draws' paths, a function's are its return values', and a batch's are its elements', since `held` of a `BatchSpec` describes one element as a term.

```python
R = RecordSpec(a=(), g=RecordSpec(c=(), e=()))
spec_at(R, "g")                                    # RecordSpec(c=(), e=())
spec_at(R, "g/c")                                  # NumericArraySpec(())
spec_at(post.spec, "mu")                           # DistributionSpec(OutputSpec(mu=...)): the spec of post["mu"]
spec_at(RecordSpec(n=(), G=post.spec), "G/mu")     # the same DistributionSpec, through the held law
spec_at(draws.spec, "mu")                          # the BatchSpec of the column mu, over the level sample
spec_at(Normal("x", 0.0, 1.0).spec, "x")           # undefined: x names the draw at the root, it is not a path
```

### 5.9 Outputs

The **value at an output** `c` of a produced term `v` is:
- `v` itself when `c` is at the root;
- the child `v[c]` when `v` is a record;
- the column at `c` when `v` is a batch;
- the function `x ↦` the value at `c` of `v(x)`, when `v` is a function.

The **outputs** of an output spec `O` are names, and `O.outputs` maps each to its spec, in order:

| `O` | The outputs of `O` | The spec of the output `c`, `O.outputs[c]` |
| --- | --- | --- |
| keyword form, or a type hole, named `n` | the one name `n`, at the root | `O.term_spec`, or `None` in a type hole |
| an interim whole form `OutputSpec(n=T)` | the one name `n`, at the root | `T` |
| positional over a `RecordSpec` `R` | its children's names, in `R`'s order | the child's spec |
| positional over a `BatchSpec` or a `FunctionSpec` `T` that holds `E` | the outputs of `E`; none when a `FunctionSpec` declares no output | `with_held(T, E_c)`, where `E_c` is `E` when `c` is `E`'s output at the root, and the output spec of `E` at the path `c` (§ 5.8) otherwise |
| positional over a `DistributionSpec` or a `ConditionalDistributionSpec` | none | — |

`outputs(t)` is the set of the outputs of `output_spec(t)`, empty when `t` has no output spec.

**The principle: the spec of an output is the spec of the produced term's value at that output.**
- For a record, the value at `c` is the child, so the spec of `c` is the child's spec.
- For a batch, the value at `c` is the column, so the spec of `c` is the column's `BatchSpec`.
- For a function as a produced term, the value at `c` is the output function, so the spec of `c` is its `FunctionSpec`.

**Why a function into records offers outputs, while a law over records does not.**
A function `f` that returns records `{mean, var}` is determined by its output functions: `f(x)` is `{mean: f_mean(x), var: f_var(x)}`, where `f_mean` is `x ↦ f(x)["mean"]`.
Splitting `f` into outputs therefore loses nothing, and a batch of records is likewise the record of its columns.
A law over records `{mu, tau}` is not determined by its marginals, since many joint laws share them.
So a produced term that is a law or a kernel offers no outputs (D1), and § 14, Q2 states what that rule buys and costs.
A batch of laws offers none either, since `held` of its spec describes an element as a law, positionally.

**The output at the root.**
`O` has an output at the root when it is in the keyword form, is a type hole, is an interim whole form, or is positional over a `BatchSpec` or a `FunctionSpec` that holds an `E` with an output at the root, under the same name.
Every other output of `O` is a path of `O` of one name.
**Grammar B's guarantee:** from G2, an output spec offers one output at the root and no paths, or outputs that are all paths of one name, never both. Until G2, an interim whole form offers an output at the root and has paths below it.

```python
prior.event_spec.outputs                  # {mu: NumericArraySpec(()), tau: NumericArraySpec(())}
Normal("x", 0.0, 1.0).event_spec.outputs  # {x: NumericArraySpec(())}: one output, at the root
grouped.event_spec.outputs                # {a: NumericArraySpec(()), g: RecordSpec(c=(), e=())}
OutputSpec(post.spec).outputs             # {}: a produced term that is a law offers no outputs
emulator.output_spec.outputs              # {mean: ..., var: ...}
OutputSpec(draws.spec).outputs            # {mu: <the BatchSpec of the column mu>, tau: ...}: the columns
fit_emulator.output_spec.outputs          # {mean: FunctionSpec(InputSpec(theta=...), OutputSpec(mean=...)), var: ...}
```

### 5.10 The record of outputs, and record notation

This notation states a merge's value (§ 8.5).
Let `O` be an output spec in the keyword form, an interim whole form, or positional over a `RecordSpec`, and `v` a value of its term spec:
- `rec_O(v)` is the record with one field per output of `O`, holding the value at that output (§ 5.9): `v` itself for a record, and `{n: v}` for an output at the root named `n`;
- `dec_O(z)` is the inverse of `rec_O`: the field `n` of `z` for an output at the root named `n`, and `z` itself for a record.

For records `r` and `q` and a set of names `N`:
- `r↾N` is the record of the top-level children of `r` whose names are in `N`, in `r`'s order;
- `r ⊎ q` is the record of `r`'s top-level children followed by `q`'s, defined when no top-level name is in both.

```python
rec_O(0.3)            # for O = lik.event_spec, named y: Record {y: 0.3}
dec_O({"y": 0.3})     # 0.3
rec_O(sample(prior))  # for O = prior.event_spec: the draw {mu, tau} itself
```

### 5.11 Feeds

An output `c` of a term `b` **feeds** the input `c` of a term `a`, or the unfed input `c` of one of `a`'s factors (§ 8.1), when they share the name `c` and `a & b` is formed (§ 8): the value at `b`'s output `c` becomes the value received at `c`.
In `a & b` only the right operand's outputs feed, and only the left operand's inputs are fed (§ 8.8 says why).
"Feeds" is the one word for this relation; "meets", "is matched", "wires", "connects", and IV.2's "bound" are retired.
"Binds" keeps its sense of supplying a value to an input in a call or `condition_on` (§ 5.3); for a dimension, this proposal says that its size is bound, as II.1 does.

```python
inputs(lik) ∩ outputs(prior)        # {"mu"}: prior's output mu feeds lik's input mu in lik & prior
```

### 5.12 Dimension scopes, conformance, and unification

This proposal uses one word per relation between specs: **conforms** for the directional check that a value of one spec is admitted by another, and **unify** for the symmetric relation whose result is the spec of a shared input. § 12.3 lists the passages of `design/` that S0 rewords to match.

**Dimension scope (II.1, unchanged).**
In one scope, a dimension name's size is bound by its first occurrence against a size, and a later occurrence that disagrees raises.
At one axis, two equal sizes agree, the same name agrees with itself, a name and a size bind the name, and two different sizes, or two different free names, raise `ValueError` (`_unify_array_shape`, `core/_spec_base.py:550`).
The scope's **substitution**, written `β`, replaces each bound name by its size, as `with_dim_sizes` does.
A merged law or kernel keeps its scope's bindings, as today's joint does (`_FactorGraph.scope`, `distributions/_factored.py:241`), and a later merge starts from the union of its operands' stored scopes, raising when they bind one name to two sizes (`_flattened`, `distributions/_factored.py:269`).
A merged input spec, output spec, record spec, or function stores no scope: `β` is applied to its specs and the bindings are not kept. § 8.11 states the consequences of both rules.

**Conformance (directional).**
A term spec `A` **conforms** to a term spec `B` when every value of type `A` is one that `B` admits, in one scope:
1. **arrays:** equal ranks, each axis agreeing by the scope rule above, and, when both set a dtype, `A`'s casting into `B`'s by NumPy's same-kind rule; supports are not compared, since a support is descriptive (III.5), as today;
2. **opaque values:** `B`'s type is `None` or a superclass of `A`'s, and `B`'s `meta` is `None` or equal to `A`'s (III.2);
3. **records:** the same child names, and each child of `A` conforms to the child of `B`;
4. **output specs:** the same form and name, and `A`'s term spec conforms to `B`'s; a type hole in `B` admits any term spec;
5. **laws:** `A`'s event spec conforms to `B`'s, so laws are covariant in what they produce;
6. **kernels and functions** (III.9's variance): the same input names; each input spec of `B` conforms to `A`'s input spec of the same name, and an input optional in `B` is optional in `A`, so inputs are contravariant; `A`'s output spec conforms to `B`'s, so outputs are covariant; a side that `B` leaves `None` admits any side of `A`, and a side that only `A` leaves `None` does not conform;
7. **batches:** the same levels, and `A`'s element term spec, `element_spec.term_spec`, conforms to `B`'s; a base element's name is not compared, as the plan's § 15.3 keeps names out of a batch's equality.

II.1's "binding by unification against a spec" is this relation, and today `_unify_specs(B, A, ...)` checks its array case (`core/_spec_base.py:525`); the joint calls it for each fed input with the output's spec as `A` (`distributions/_factored.py:437`).

**Unification (symmetric; IV.2's sense).**
Two term specs **unify** when their unification `U(A, B)` below exists, in one scope.
`U(A, B)` is the spec of a shared input, which "admits only the values every consuming factor admits" (IV.2), and it is defined by kind:
1. **arrays:** equal ranks and each axis agreeing by the scope rule; a dtype that only one sets, and, where both set one, the dtype of the spec that conforms to the other, the narrower by NumPy's safe cast when each conforms to the other (`float32` with `float64` gives `float32`), with no unification when neither conforms; a support only when both set the same one, and none otherwise;
2. **opaque values:** III.2's rule: the types are equal or one is `None`, and so are the `meta`s; `U` takes the known type and the set `meta`;
3. **records:** the same child names, child by child;
4. **output specs:** the same form and name, and `U` of the term specs; a type hole takes the other's term spec;
5. **laws:** `U` of the event specs;
6. **kernels and functions:** the same input names, with equal input specs and optional flags, and `U` of the output specs; a side that one leaves `None` takes the other's;
7. **batches:** the same levels, and `U` of the element term specs.

Any other pair, such as an array spec with a record spec, does not unify.
`U` is symmetric, and associative on the specs it accepts.
A spec `C` conforms to `U(A, B)` exactly when `U(A, B)` exists and `C` conforms to both `A` and `B`, provided no two different free dimension names occur at one axis among the three (§ 8.9 uses this; a probe over the standard NumPy dtypes checks the dtype case, Appendix E, fact 38).
Today's joint keeps the first operand's spec for a shared input instead of `U` (Appendix E, fact 8); since it checks a value bound to a shared input against every factor that consumes it, this changes only the reported `given_spec`, not which merges are valid (fact 39). Appendix C.4 schedules the fix, and § 14, Q4(d) asks whether to make it now.

```python
NumericArraySpec((3,), dtype="float32")    # conforms to NumericArraySpec(("p",)), binding p to 3
NumericArraySpec((), dtype="int32")        # conforms to NumericArraySpec((), dtype="float32"): same-kind cast
NumericArraySpec((), dtype="float32")      # does not conform to NumericArraySpec((), dtype="int32")
NumericArraySpec((4,))                     # does not conform to NumericArraySpec((3,)): two different sizes
NumericArraySpec(("m",))                   # does not conform to NumericArraySpec(("n",)): two different names (II.1)
# U(NumericArraySpec((), dtype="float32"), NumericArraySpec((), dtype="float64")) == NumericArraySpec((), dtype="float32")
# U(NumericArraySpec((), dtype="float32"), NumericArraySpec((), dtype="int32")) == NumericArraySpec((), dtype="int32")
# U(OpaqueSpec(type=str), OpaqueSpec()) == OpaqueSpec(type=str)
```

### 5.13 The two sides compared

Each concept on one side has its counterpart on the other, with the same structure, apart from the differences this table states.

| # | Input side | Output side | Why they differ |
| --- | --- | --- | --- |
| 1 | an input | an output | — |
| 2 | `InputSpec` | `OutputSpec` | — |
| 3 | several operands of a merge may share an input, which becomes one input | no two operands of a merge have one output | one value can serve many receivers, while one value needs one source |
| 4 | the left operand's inputs are fed | the right operand's outputs feed | a density is written `p(y \| mu) · p(mu)`, and a fixed direction makes the written order a topological order of the factors (§ 8.8) |
| 5 | names are identifiers | names are any names | an input binds a Python parameter, while an output becomes a record field; an output feeds an input only when its name is also an identifier, which `InputSpec(o)` checks |
| 6 | every input is named | only a base value is named from outside, at the root | an argument is always passed by name, while a produced base value carries no name (grammar B) |
| 7 | one form | two constructor forms, keyword and positional | follows from row 6 |
| 8 | flat names; paths into a record input come in G4 | an output is at the root or at a top-level path, with paths below it | a path is a position inside a produced term; until G4 an input receives one value whole (III.9) |
| 9 | `I[name]` is a term spec, and `InputSpec` is a `Mapping` | `O[path]` is an output spec, `O.outputs[name]` is a term spec, and `OutputSpec` is not a `Mapping` | the node at a path of a produced term is itself described by an output spec, which needs a name when it is a base value; an input's value already has its input's name, and the input names are the keys |
| 10 | an input may be optional, and its default belongs to the term that owns it | an output is always present | a produced term has a value at every output; an `InputSpec` stores no default |
| 11 | a shared optional input that a binding omits takes each operand's own default | — | row 10: the defaults belong to the operands and cannot be compared; today's `*` does the same (Appendix E, fact 26) |
| 12 | no type holes | type holes in the keyword form | a function or a law fills its output's type once it runs, while an input's type must be known to check a binding |
| 13 | equality ignores order | equality keeps order | binding reads inputs by name, while output order is the layout `to_vector` reads |
| 14 | an input spec may have no inputs | an output spec's term spec is never an empty record, and `without` never removes every node | a term with no inputs is a law or a value, while an empty record describes nothing produced |
| 15 | a merge of laws and kernels also reads the unfed inputs of its operands' factors, which for a law are not inputs of the law (§ 8.1) | — | today's flattening (IV.2), kept before the workshop; § 8.10 states its consequences, and § 14, Q4(b) asks what to do after |
| 16 | a shared input's specs unify, symmetrically | a fed output's spec conforms to the input's, directionally | one value fills both inputs, so neither side is privileged; a fed value is given, and the input must admit it |
| 17 | a batch's inputs are its elements' inputs | a batch's outputs are its columns | one binding fills the same input of every element, while a produced batch is offered column by column; a merge of batches reads the elements on both sides (§ 8.2) |
| 18 | `InputSpec(o)` reads an output spec as the inputs it would feed | nothing reads an input spec as outputs | an output spec states a produced term, from which inputs follow; an input spec states no produced term, so no output spec follows from it |
| 19 | `InputSpec.rename` comes after the workshop | `OutputSpec.rename` comes in S3 | `rename` replaces `with_path_names`, which `OutputSpec` has today and `InputSpec` does not; a kernel's own `rename` renames its inputs already (III.9) |
| 20 | `given_spec`, `input_spec` | `event_spec`, `output_spec`, `element_spec` | each attribute names its kind's role (§ 5.7) |

The two sides share everything else: selection with one name or a list (§ 7.3, § 7.4), `without` (§ 7.5), `&` (§ 8.6), membership (§ 7.6), the size shorthand, and one dimension scope.

## 6. Every kind placed

| Kind | Produced term (§ 5.4) | Inputs (§ 5.7) | Outputs (§ 5.9) | `[]` takes (§ 7) | `t[s]` is | `without` | `&` (§ 8) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| array, `NumericArray` | itself | none | none | NumPy's positions into its entries | its entries, unchanged | none | NumPy's elementwise `&`, which is not a merge (the plan's O7) |
| opaque value, `Opaque` | itself | none | none | nothing | — | none | none |
| record, `Record` | itself | none | its children | paths, lists of paths | the node itself, or the record of the chosen nodes | yes | merge: both sets of fields |
| law, `Distribution` | a random draw | none | those of `event_spec` | paths, lists of paths | the view, a `MarginalView` | step G4 | merge, read as its factors (§ 8.1); feeds the left operand |
| kernel, `ConditionalDistribution` | a random draw at each value of its inputs | `given_spec` | those of `event_spec` | paths, lists of paths (step G5) | the kernel whose law at each input value is the view there | after the workshop | merge, read as its factors; feeds and is fed |
| function, `Function` | its return value at each value of its inputs | `input_spec` | those of `output_spec` | paths, lists of paths (step G3) | the function `w ↦ f(w)[s]` | after the workshop | merge; feeds and is fed (after the workshop) |
| batch, `Batch` | itself | its elements' | its columns | positions, or paths of its elements | by position, a sub-batch or an element; by path, the batch of each element's node | for batches of records | merge of the elements, position by position, each read as one factor (`RecordBatch` now, other batches after the workshop) |

A spec is not a term: it produces and receives nothing.
It describes the terms it types, so each operation on a spec gives what the spec says about the same operation on those terms.

| Spec | What it describes | `[]` | `without` | `&` |
| --- | --- | --- | --- | --- |
| `NumericArraySpec`, `OpaqueSpec` | a base value | nothing | nothing | nothing |
| `RecordSpec` | the type of a record, read as outputs by `OutputSpec(R)` and as the inputs they feed by `InputSpec(R)` | a path gives the spec at that path; a list gives the cut-down `RecordSpec` (step S1) | paths | the union of the children; a top-level name in both raises |
| `DistributionSpec`, `ConditionalDistributionSpec`, `FunctionSpec`, `BatchSpec` | a law, a kernel, a function, or a batch, through the output spec each holds (§ 5.6) | none of their own: index the held spec, as `post.spec.event_spec["mu"]` | none | the spec of the merged term, after the workshop |
| `InputSpec` | a term's inputs | a name gives its term spec; a list of names gives an `InputSpec` (§ 7.4) | names | the inputs of both, a shared input unified (§ 8.6) |
| `OutputSpec` | a term's produced term and its outputs | a path or a list of paths gives an `OutputSpec` (§ 7.3) | paths | the outputs of both (§ 8.6) |

**Records.**
A record produces itself, its outputs are its children, and it has no inputs.
In a merge a record only offers values: `r1 & r2` puts both sets of fields side by side, and nothing is fed.
A record merges only with another record (§ 8.2).

**A `RecordSpec` has one reading.**
`OutputSpec(R)` reads its children as outputs, and `InputSpec(R)` equals `InputSpec(OutputSpec(R))`: the inputs that those outputs would feed (§ 7.4).

```python
R = RecordSpec(growth=RecordSpec(K=(), r=()), phi=())
OutputSpec(R).outputs         # {growth: RecordSpec(K=(), r=()), phi: NumericArraySpec(())}
InputSpec(R)                  # two inputs, growth (a record) and phi: the same as InputSpec(OutputSpec(R))
```

## 7. Indexing: selecting and removing

### 7.1 `[]` on terms

`[]` reads a path or a list of paths `s` through the path reader (§ 4.1), and its result depends on the kind:

| `t` | `t[s]` | Identity (I6) |
| --- | --- | --- |
| a record `r` | the node at `s` itself: a field's value or held term, or a group's record; for a list, the record of the chosen nodes | `r[s].spec == r.spec[s]` |
| a law `d` | the view, of class `MarginalView` (today `FieldView`, renamed in step S3): the law of `X`'s node at `s`, for `X ~ d`, which refers to `d`; `marginal(d, s)` is the same law detached | `d[s].event_spec == d.event_spec[s]` |
| a kernel `k` (step G5) | the kernel whose law at each input record `w` is the law of `X`'s node at `s`, for `X ~ k(w)` | `k[s].event_spec == k.event_spec[s]` and `k[s].given_spec == k.given_spec` |
| a function `f` (step G3) | the function `w ↦ f(w)[s]` | `f[s].output_spec == f.output_spec[s]` and `f[s].input_spec == f.input_spec` |
| a batch `bt` | the batch, over the same levels, of each element's node at `s`; for one path, the column | `bt[s].spec.element_spec == bt.spec.element_spec[s]` |

So on a law, a kernel, a function, or a batch, `t[s]` is the term of the same kind whose produced term is the node at `s` of `t`'s produced term.
A record is its own produced term, and its nodes are its values and held terms, so `r[s]` is the node itself: `r["a"]` is an array labeled `a`, and `record_with_law["G"]` is the law held there.
A record's identity is stated on its term spec, since a base child has no output spec; the name `a` is in `OutputSpec(r.spec)["a"]`, not in the array.
Indexing never changes a term's inputs, and the definition makes the plan's I2, I4, and I13 hold by construction.

**One path or a list.**
One path gives the node; a list gives a record, or a term whose produced term is a record, that keeps each chosen node at its own path, in canonical order whatever order the list gives (decided 2026-10-15).
A path in a list that goes through a field holding a law, a function, or a batch cuts that held term down to the rest of the path, in the held term's own kind (`restrict`, § 7.3).

```python
draw = sample(grouped)          # Record {a, g: {c, e}}
draw["g"]                       # Record {c, e}                   (today: KeyError, "use at_path()")
draw[["g"]]                     # Record {g: {c, e}}
draw["g/e", "a"]                # Record {a, g: {e}}: grouped's order
grouped["g"]                    # the law of {c, e}; outputs c and e   (step G1a)
grouped[["g"]]                  # the law of {g: {c, e}}; one output, g
emulator["mean"]                # the function theta -> emulator(theta)["mean"]   (step G3)
record_with_law = Record({"n": 3.0, "G": post})
record_with_law["G"]            # post itself
record_with_law["G/mu"]         # the view post["mu"]             (step G3; today: KeyError)
record_with_law["n", "G/mu"]    # Record {n, G}, whose G is post[["mu"]]   (step G3)
```

### 7.2 Views at fields that hold a law, a kernel, a function, or a batch

From step G3, a view at or below such a field follows § 7.1 like any other node, and no wrap is added.
So `rm["G"]` is a law whose draws are the drawn laws: its event spec is `OutputSpec(law_spec)`, which offers no outputs (§ 5.9).
To keep the field's name, index with a list: `rm[["G"]]` is the law over records `{G: law}`, which offers `G` and merges.

```python
rm["G"]                    # a law whose draws are laws; event_spec OutputSpec(law_spec); outputs {}   (G3)
sample(rm["G"])            # a law, distributed as sample(rm)["G"]            (I2, with no exception)
rm["G"]["mu"] == rm["G/mu"]                  # I5, with no exception
rm[["G"]]                  # a law over {G: law}; one output, G
lik_G & rm[["G"]]          # rm[["G"]]'s output G feeds lik_G's input G
lik_G & rm["G"]            # TypeError (M2): rm["G"] produces a law; index with [["G"]] to keep the name
```

**Until G3**, such a view keeps today's whole form, an interim whole form named by the field: `d[p].event_spec == OutputSpec(**{last(p): d.event_spec[p].term_spec})`, which is the plan's interim I6 at these fields (its § 7.5).
S2b's switch of `_view_declaration` to `O[s]` therefore leaves out one path to a field that holds a law, a kernel, a function, or a batch, as it leaves out one path to a group until G1a (Appendix C.2).

The plan wrapped these views under the field's name from G3 (its § 5.3, R3, O5), because its law's event rule forbade a law whose event spec is positional over a law.
Here a produced term that is a law offers no outputs, and M2 refuses it in a merge, so neither the event rule nor the wrap is needed (§ 14, Q2).

### 7.3 `[]` on an output spec

For an output spec `O` and a path or a list of paths `s`, `O[s]` is defined by four rules, read through the path reader.
It replaces the plan's `project` and today's private `_view_declaration` (`distributions/_views.py:160`).

1. **One path `p`** of `O` gives the output spec of `O` at `p` (§ 5.8). So:
   - a path to a base node gives that node under its last name: `grouped.event_spec["g/c"]` is `OutputSpec(c=...)`;
   - a path to a group gives the group's record positionally, whose outputs are the group's children;
   - a path to a field that holds a law, a kernel, a function, or a batch gives that held term's spec positionally;
   - a path below such a field gives the spec of a term of the held term's kind, whose produced term is the node there.
2. **A list of paths `ps`** gives `OutputSpec(restrict(O.term_spec, ps))`, where `restrict` keeps each node at its own path:
   - for a `RecordSpec` `T`: the children of `T` that a path of `ps` names or enters, in `T`'s order; a child that `ps` names is kept whole, and a child that `ps` enters through the remaining paths `qs` is kept as `restrict(child, qs)`;
   - for a `T` that holds `E`: `with_held(T, E[ps])`.
3. **A list may also name the output at the root, at the top level only.** A list is read against the record of outputs (§ 5.10), which for an output at the root named `n` is `{n: v}`. So `O[[n]]` is `OutputSpec(RecordSpec(n=O.term_spec))` for an output spec in the keyword form, and `OutputSpec(with_held(T, E[[n]]))` for one positional over a batch or function spec `T` holding `E`. This is an extra rule beside rules 1 and 2, kept from the plan's O6: `Normal("x", 0.0, 1.0)[["x"]]` is the law over `{x}`. Inside `restrict`, a path must be a path, so for a field `G` that holds `Normal("x", 0.0, 1.0)`, `O[["G/x"]]` raises `KeyError`.
4. **The name of the output at the root, alone.** Until step G3, `O[n]` is `O` itself. This mirrors `Distribution.__getitem__`, which returns the law for that name today (`distributions/_distribution.py:1015`), so I6 holds for laws at every step; G3 removes the reading from both at once, so it is not bridging code. From G3, `O[n]` raises `KeyError`, and `t.output(n)` reads the term (§ 7.7).

An interim whole form follows § 5.5's rules 3 and 4 until G2.
`O[s]` adds no name that `O` does not already hold: rule 3 uses the name `O` states, and rules 1 and 2 add none.
The errors are the path reader's (§ 4.1), and a `str` that is neither a path of `O` nor, until G3, the name of its output at the root raises `KeyError`.
Two differences from today's `_view_declaration` were already adopted by the plan: a list keeps paths instead of flattening to last names (S1), and one path to a group gives the group's outputs (G1a).

**The property these rules give.**
For a law, a kernel, a function, or a batch `t` whose output spec is `O`, `O[s]` is the output spec of `t[s]` (I6), with three interim exceptions that Appendix B lists: one path to a group of a law until G1a, one path to a held field until G3, and a batch's output at the root until G3.
For a record, the property is stated on term specs, `r[s].spec == r.spec[s]`, since a base child has no output spec.

```python
O = grouped.event_spec                       # positional over RecordSpec {a, g: {c, e}}
O["a"]                                       # OutputSpec(a=...)
O["g/c"]                                     # OutputSpec(c=...)
O["g"]                                       # OutputSpec(RecordSpec(c=..., e=...)): outputs c and e
O[["g"]]                                     # OutputSpec(RecordSpec(g=RecordSpec(c=..., e=...))): output g
O[["g/e", "a", "g/c"]]                       # OutputSpec(RecordSpec(a=..., g=RecordSpec(c=..., e=...)))
hier.event_spec[["s1/theta", "s2/theta"]]    # OutputSpec(RecordSpec(s1=RecordSpec(theta=...), s2=RecordSpec(theta=...)))
OutputSpec(post.spec)["mu"]                  # OutputSpec(DistributionSpec(OutputSpec(mu=...))): the spec of a law of mu
OutputSpec(RecordSpec(n=(), G=post.spec))[["n", "G/mu"]]
# OutputSpec(RecordSpec(n=..., G=DistributionSpec(OutputSpec(RecordSpec(mu=...))))): G holds post[["mu"]]'s spec
OutputSpec(x=NumericArraySpec(()))[["x"]]    # OutputSpec(RecordSpec(x=...)): rule 3
OutputSpec(x=NumericArraySpec(()))["x"]      # the spec itself until G3; KeyError from G3
O["x"]                                       # KeyError: x is not a path of grouped's draws
O[["g", "g/c"]]                              # ValueError: g/c is inside g
```

### 7.4 `[]` on an input spec, its four sources, and the size shorthand

`InputSpec` stays a `Mapping` from input names to term specs: `keys()`, iteration, and `len` read the inputs.
Its `[]` reads the path reader's argument against the inputs:

| Argument | Result | Step |
| --- | --- | --- |
| an input's name | that input's term spec | unchanged |
| a list of input names | the input spec of those inputs, in this spec's order, each optional input staying optional | S2b |
| a path into a record input, such as `"g/c"` | the term spec at that path | G4, after the workshop |
| a list with a path into a record input | the input spec whose input holds the cut-down record; that input becomes required, since a default is a whole value | G4, after the workshop |

Paths into a record input come with partial binding of a record input (III.9) in step G4, so they are design for after the workshop.
A name that is not an input raises `KeyError`, and the path reader's errors apply.

**The four sources.**
The `InputSpec` constructor reads one source:
1. an `InputSpec`, as it is;
2. a mapping from input names to term specs;
3. a `RecordSpec`, by its children;
4. an `OutputSpec`, by its outputs, each input typed by the spec of its output.

The constructor tests for a `RecordSpec` and an `OutputSpec` before a plain mapping, since a `RecordSpec` is not a `Mapping` and today's constructor refuses it (`core/_specs.py:120`), while an `OutputSpec` must be read by its outputs (the plan's § 6.2).
`InputSpec(R)` equals `InputSpec(OutputSpec(R))` for every non-empty `RecordSpec` `R`, and `InputSpec(RecordSpec())` is `InputSpec()`.
Every input read from a mapping, a record spec, or an output spec is required; `conditional_distribution` and `Function` then mark optional the inputs whose parameters have defaults, and refuse an optional input without one (§ 5.3).
An output name that is not an identifier raises `ValueError`, and a type hole raises `TypeError`.
`conditional_distribution`'s `given_spec`, `Function`'s `input_spec`, and `ConditionalDistributionSpec`'s `given_spec` read their argument through this constructor (the plan's § 6.2, kept).
Today `InputSpec(RecordSpec(K=(), r=()))` raises `TypeError` (`core/_specs.py:120`).

**The size shorthand.**
Wherever an input or the keyword form of `OutputSpec` takes a term spec, a tuple of sizes is the `NumericArraySpec` of that shape (the plan's § 6.3, kept): `InputSpec(K=(), e=("n_years",))`.

```python
lik.given_spec["mu"]                          # NumericArraySpec(()): a term spec
likelihood.given_spec[["K", "r"]]             # InputSpec(K=..., r=...)
InputSpec(growth_prior.event_spec)            # inputs K, r, phi, sigma
InputSpec(OutputSpec(post.spec))              # InputSpec(): a produced term that is a law offers no outputs
lik_g.given_spec["g/c"]                       # NumericArraySpec(())   (step G4)
```

### 7.5 Removing: `without`

**On a term.**
`t.without(*paths)` returns `t` in its own kind with the nodes at `paths` removed and every other node at its path, and its inputs are unchanged.
It exists on records, record specs, output specs, batches of records, and laws (step G4, as a view).
On functions, kernels, and batches of laws, and on a record at a path through a held term, it comes after the workshop.

**On an output spec.**
`O.without(*ps)` is `OutputSpec(remove(O.term_spec, ps))`, where `remove` is defined on a spec with parts `T`:
1. **`T` is a `RecordSpec`:** each child that `ps` names is removed; a child all of whose leaf paths `ps` covers is removed, whether a group or a field holding a term; each other child that `ps` enters, through the remaining paths `qs`, is replaced by `remove(child, qs)`.
2. **`T` holds `E`:** `remove(T, ps)` is `with_held(T, E.without(*ps))`.

A record stays a record even when one node is left, so a list of paths and `without` are complements (I10).
A path that is not a path of `O` raises `KeyError`, so every path of an output spec in the keyword form over a base spec raises, since its name is not a path.
A removal that would leave `O` itself with nothing raises `ValueError`.
An interim whole form follows § 5.5's rule 4 until G2.
The path reader's errors apply.

**On an input spec.**
`I.without(*names)` returns the other inputs, each optional input staying optional.
A name that is not an input raises `KeyError`; today it is ignored (`core/_specs.py:191`).
A removal may leave no input.
From step G4, a path into a record input leaves that input holding the rest of the record, and the input becomes required.

```python
draw.without("g/e")                           # Record {a, g: {c}}
post.without("tau")                           # the view of post over {mu}          (step G4)
growth_prior.event_spec.without("sigma")      # OutputSpec(RecordSpec(K=..., r=..., phi=...))
grouped.event_spec.without("g/c", "g/e")      # OutputSpec(RecordSpec(a=...)): the empty group goes
OutputSpec(RecordSpec(n=(), G=post.spec)).without("G/mu", "G/tau")   # OutputSpec(RecordSpec(n=...)): G goes
OutputSpec(x=NumericArraySpec(())).without("x")   # KeyError: x is not a path
lik.given_spec.without("mu")                  # InputSpec(): valid
lik.given_spec.without("zzz")                 # KeyError                            (today: ignored)
```

### 7.6 Membership

`key in t` is true when `key` is a `str` and `t[key]` succeeds, and false otherwise.
This holds on every term other than an array, and on `RecordSpec`, `OutputSpec`, and `InputSpec`.
A record's `keys()` still lists its fields only, and an input spec's `keys()` lists its inputs (the plan's § 11 and § 16, kept); a `NumericArray` keeps its own `in`, which tests entries.

```python
"g/c" in grouped                   # True        (today: TypeError)
"g/c" in grouped.event_spec        # True        (the plan: False)
"x" in Normal("x", 0.0, 1.0)       # True until step G3, then False: x is not a path
"mu" in lik.given_spec             # True
```

### 7.7 The term at an output

`t.output(c)` is the term at the output `c`: `t` itself when `c` is at the root, and `t[c]` otherwise.
It arrives in step G3 and replaces the decided `t.component(c)`.
Its use is the output at the root, which is not a path, so `[]` cannot read it from G3; away from the root it is `t[c]`.
On an output spec there is no counterpart: `O.outputs[c]` gives the spec of an output, and `O[c]` the output spec at a path (§ 14, Q5).
A name that is not an output raises `KeyError`, so a term whose produced term is a law offers nothing to `output`.

| Term `t` | `t.output(c)` |
| --- | --- |
| a record | `t[c]`, the child itself |
| a law or a kernel | the view `t[c]`, or `t` itself when its event's output `c` is at the root |
| a function | the function `w ↦ t(w)[c]`, or `t` itself when its output `c` is at the root |
| a batch | the column `t[c]`, or `t` itself when its elements' output `c` is at the root |

For every term `t` with outputs, `output_spec(t).outputs[c]` is the term spec of the produced term of `t.output(c)` (I12).

```python
grouped.output("g")                                         # grouped["g"]
Normal("x", 0.0, 1.0).output("x")                           # the law itself
sample(Normal("x", 0.0, 1.0), sample_shape=4).output("x")   # the batch itself
fit = Function(run_inference, output_spec=post.spec)
fit.output("mu")                                            # KeyError: fit produces a law, which offers no outputs
```

A mixture is a law, so `mix.output("x")` reads its event's output `x`; the word "component" keeps only its sense of a mixture's component laws (VII.3), which settles the plan's O14.

### 7.8 Symbolic dimensions

Selection and removal keep every dimension name as it is, bound or free, and bind no size.
The kept nodes stay in the dimension scope of the term they came from: a name in `O[s]` is the same dimension as in `O`.
The free dimensions of `O[s]` are the free dimensions of `O` that occur in the kept nodes.
Substituting a size commutes with selection: `O.with_dim_sizes(n=3)[s] == O[s].with_dim_sizes(n=3)` when `n` is free in `O[s]`.
A view applies a dimension substitution to its parent (III.7).

```python
O = OutputSpec(RecordSpec(X=("obs", "features"), beta=("features",)))
O["beta"]                       # OutputSpec(beta=NumericArraySpec(("features",)))
O["beta"].term_spec.free_dims   # frozenset({"features"})
O["X"] & O["beta"]              # OutputSpec(RecordSpec(X=..., beta=...)): one dimension features again
```

## 8. Merge

### 8.1 Notation, factors, and the unfed inputs of factors

`a & b` combines two terms, or two specs, by the names of their inputs and outputs.
It is written right to left, as a density is: `lik & prior` is `p(y | mu) · p(mu, tau)`.

**Factors** (IV.1, IV.2).
A merge of laws and kernels reads each operand as its factors, as IV.2's flattening does today (`_flattened`, `distributions/_factored.py:269`).
The factors of a law or a kernel `t`, written `factors(t)`, are the factors of a factored joint, apart from a grouped sub-joint, which stays one factor (IV.2); every other law or kernel is one factor.

**The unfed inputs of the factors.**
`unfed(t)` is the input spec of the inputs that `t`'s factors take and no factor of `t` feeds, optional ones included, in order of first appearance.
It is a description that the merge rules read, not an attribute and not an input of `t`:
- for a kernel, `unfed(t)` is its `given_spec`;
- for a law built by a distribution family, it is empty;
- for a law that is a joint, it holds the optional inputs its factors left at their defaults: `unfed(lik_opt & prior)` holds `s`, although the law has no inputs (III.7);
- for a grouped sub-joint used as one factor, it is that factor's own `given_spec`, so the optional inputs inside it are not read (Appendix E, fact 43).

For a record, a function, or a spec operand, `unfed(t)` is its input spec: none for a record, `f.input_spec` for a function, and `I` itself for an input spec.
A spec, and an element of a batch in a batch merge, is read as one factor, so `unfed` of a `DistributionSpec` is empty.
For a spec operand, `outputs(O)` is the set of an output spec `O`'s outputs, `outputs(R)` the set of a record spec `R`'s top-level names, and `outputs(I)` is empty.

In `a & b`, writing `names(I)` for the names of an input spec `I`:
- `fed = names(unfed(a)) ∩ outputs(b)` is the set of names that the right operand feeds (§ 5.11);
- `shared = (names(unfed(a)) − fed) ∩ names(unfed(b))` is the set of names both operands' factors take and neither feeds;
- `β` is the substitution of the merge's dimension scope (§ 5.12), which starts from both operands' stored scopes, if any, and adds every binding that M6 and M7 make.

### 8.2 Which operands merge

`&` merges two operands of one **merge category**, a grouping used only by this section's rules:

| Merge category | Operands | Result | Step |
| --- | --- | --- | --- |
| records | `Record`, `NumericRecord` | a record | S2b makes `merge` shallow; S3 renames it `&` |
| laws and kernels | `Distribution`, `ConditionalDistribution` | a law, or a kernel when a required unfed input is left | S3 (today's `*`) |
| functions | `Function` | a function | after the workshop |
| batches | two batches over equal levels whose elements merge | a batch over the same levels | `RecordBatch`: S2b rule, S3 operator; other batches after the workshop |
| input specs | two `InputSpec`s | an `InputSpec` | S2b |
| output specs | two `OutputSpec`s | an `OutputSpec` | S2b |
| record specs | two `RecordSpec`s | a `RecordSpec` | S2b rule, S3 operator |
| specs of laws, kernels, functions, and batches | two of one of these | the spec of the merged term | after the workshop |

Operands of two merge categories raise `TypeError`, so `InputSpec & OutputSpec` raises; to read an output spec as inputs, write `InputSpec(o)`.
An array and an opaque value belong to no merge category: they offer no output, and `&` on two `NumericArray`s keeps NumPy's elementwise meaning (the plan's O7, kept).

**Batches merge by their elements.**
Two batches have **equal levels** when they have the same level names, in the same order, with the same axis sizes, as `RecordBatch.merge` requires today (`core/_record_batch.py:783`).
For such batches, `a & b` is the batch, over the same levels, whose element at each position `j` is `ã_j & b̃_j`, where an element read as an operand `x̃_j` is:
- for an element that is a base value, the record `rec_E(x_j)` under the batch's `element_spec` `E` (§ 5.10), so it merges as a record;
- for a record, the record itself;
- for a law, a kernel, or a function, the element as one factor, with the input spec and output spec of its term spec `element_spec.term_spec` (§ 5.7).

The merge category, the conditions M2 to M8, `fed`, `shared`, and the result's inputs and outputs are those of the element merge, read once on those specs.
A batch's own outputs, its columns, serve `[]`, `t.output(c)`, and `InputSpec(o)`, and take no part in a merge.
Elements of two merge categories, such as records with laws, raise `TypeError`.
Today `&` on two batches of arrays raises `TypeError` (Appendix E, fact 44), and the plan's § 16 keeps it raising, so merging them into records is a difference (§ 12.1).

**Merges across categories are design for after the workshop.**
A record is the law that gives one draw, itself, with probability one, and a function is the kernel whose law at each input is a point mass at its return value.
By that reading, `lik & Record({"mu": 0.3})` and `f & d` have a meaning.
A law with a point-mass factor has no joint density, and IV.1's factors admit only laws and kernels, so these merges raise `TypeError` now, and step S0 records the point-mass reading in `deferred.md`.

### 8.3 The conditions

`a & b` raises unless all of these hold:

| # | Condition | Error | Why |
| --- | --- | --- | --- |
| M1 | `a` and `b` are of one merge category (§ 8.2); two batches have equal levels and elements of one category | `TypeError`; `ValueError` for unequal levels | Each category has its own value rule (§ 8.5) |
| M2 | **Complete specs, and produced terms that a record can hold.** Each law, kernel, or function operand states an output spec with no type hole, and a function also states its input spec. Each output spec is in the keyword form over a base spec or positional over a `RecordSpec`, so its produced term is a base value or a record. Until G2, an interim whole form also passes, and the merge stores its produced term whole, as the field of its name (§ 5.5). A record passes. A spec operand has no type hole | `TypeError`, which names the type hole or the missing spec, or says to wrap the output under a name | The result stores the value at each output as a field and must rebuild each operand's produced term from it (§ 8.8) |
| M3 | No name is an output of both: `outputs(a) ∩ outputs(b) = ∅` | `ValueError`, naming the output and both operands | One value needs one source |
| M4 | **The order rule:** the right operand's factors take no unfed input, required or optional, that the left one outputs: `names(unfed(b)) ∩ outputs(a) = ∅` | `ValueError`, which says to put the operand that outputs the name on the right, as today (`distributions/_factored.py:431`) | It keeps the result's sides disjoint and the written order a topological order of the factors (§ 8.8) |
| M5 | Each operand's unfed inputs and outputs are disjoint: `names(unfed(t)) ∩ outputs(t) = ∅` | `ValueError`, which says to rename one | Whether a name is fed or shared is then decided by names alone; III.9 already requires it of kernels |
| M6 | For each `c` in `fed`, the spec of `b`'s output `c` conforms to the spec at `c` in `unfed(a)` (§ 5.12) | `ValueError`, naming both specs | The input must admit every value it is fed |
| M7 | For each `s` in `shared`, `U` of the specs at `s` in `unfed(a)` and `unfed(b)` exists (§ 5.12) | `ValueError`, naming both specs | One value fills both inputs, so both must admit it |
| M8 | All specs of both operands form one dimension scope, with any stored scopes: no dimension name is bound to two sizes, and no two different free names occur at one axis (II.1) | `ValueError`, which names the dimension and both sizes; when a spec shows the name, it says to rename one apart with `with_dim_names`; when only an operand's stored scope holds it, it says so, names that operand, and suggests merging its parts in another order, since `with_dim_names` cannot rename a stored name (§ 8.11, item 2) | A dimension name is one quantity (II.1) |

For a merge of batches, M2 to M8 read the elements' specs (§ 8.2).
M2's interim clause keeps today's merges of the G2 families working until G2: `Normal("z", 0.0, 1.0) & random_function` and `rm_whole & Normal("x", 0.0, 1.0)`, for a random measure `rm_whole` declared `OutputSpec(G=law_spec)`, merge as they do today, each storing the whole term under its name (Appendix E, fact 27).

### 8.4 The result

| # | Part | Value |
| --- | --- | --- |
| R1 | Outputs | `output_spec(a & b) = β(output_spec(a) & output_spec(b))` (§ 8.6): `a`'s outputs, then `b`'s; an operand with no output spec, such as an empty record, contributes none; for batches, this holds of the element merge (§ 8.2), and the result's element output spec is the element merge's |
| R2 | Unfed inputs | `unfed(a & b) = β(unfed(a).without(*fed) & unfed(b))` (§ 8.6): every unfed input of the factors that no output feeds, optional ones included, `a`'s first; a shared one carries `U` of its specs and is optional only when optional in every operand that has it; a fed one leaves, whether it was required or optional; for batches, this holds of the element merge (§ 8.2) |
| R3 | Kind and inputs | records give a record; laws and kernels give a kernel whose `given_spec` is `unfed(a & b)` when a required unfed input is left, and otherwise a law, which has no inputs and whose factors take their optional inputs at their defaults when it is sampled or scored; functions give a function whose `input_spec` is `unfed(a & b)`; batches give a batch over the same levels whose elements are the element merges |
| R4 | Factors and scope | for laws and kernels, `factors(a)` followed by `factors(b)`, and the scope of § 8.1, which the result stores |
| R5 | Value | by category (§ 8.5) |
| R6 | Label | the operands' labels joined with `·` (IV.2; the plan's O12); no merge reads a label |

R3 follows IV.2's table, so a kernel whose inputs are all optional is a kernel alone, while a merge that leaves only its optional inputs is a law: `k_opt & Normal("z", 0.0, 1.0)` samples at `s = 1.0` although `sample(k_opt)` raises (Appendix E, fact 37). This is today's behaviour, and Appendix B lists it.

**A shared optional input that a binding omits.**
One value fills a shared input when a binding supplies it.
When a binding omits a shared optional input, each factor takes its own default, since a default belongs to the term that owns it and an `InputSpec` stores none (§ 5.13, row 11).
This is today's behaviour (Appendix E, fact 26), and Appendix B lists it.

```python
model = lik & prior
# fed = {mu}; outputs y, mu, tau; no unfed input left, so a law
data_model = lik_nee & lik_lai
# fed = {}; shared = {theta}: one input theta, U of its specs; outputs nee, lai: a kernel
lik_opt & prior                  # mu is fed; s stays unfed and optional: a law, with no inputs, sampled at s = 1.0
(lik_opt & prior) & s_prior      # s is fed now: y's spread follows s near 100, as today (§ 8.10)
prior & lik                      # ValueError (M4): lik takes mu, which prior outputs; write lik & prior
s_prior & (lik_opt & prior)      # ValueError (M4): lik_opt's unfed input s is an output of s_prior, on the left
Normal("x", 0.0, 1.0) & Normal("x", 2.0, 1.0)    # ValueError (M3): x is an output of both
lik & Normal("mu", jnp.zeros(3), 1.0)            # ValueError (M6): the output mu, of shape (3,), does not conform to lik's input mu
lik & Record({"mu": 0.3})                        # TypeError (M1): a kernel and a record are of two merge categories
echo = Function(lambda x: x, input_spec={"x": ()}, output_spec=OutputSpec(x=()))
twice = Function(lambda y: 2 * y, input_spec={"y": ()}, output_spec=OutputSpec(w=()))
echo & twice                                     # ValueError (M5): echo takes and outputs x   (after the workshop)
```

### 8.5 The value of a merge

Let `u` be an input record of `a & b`: a record with one field for each input of `a & b` that the binding supplies.
For a law, a kernel, or a function `t` and an input record `w`, `t(w)` is the law `t` gives at `w`, or the value it returns, where each factor fills each unfed input that `w` does not supply with its own default.
`x ~ L` says that `x` is a draw of the law `L`, and `p_t(x ; w)` is the density of `t(w)` at `x`.
`rec_a` and `dec_a` are `rec` and `dec` of `output_spec(a)` (§ 5.10), and likewise for `b`; `u↾N` and `⊎` are § 5.10's record notation, for a set of names `N`.

**Records.**
Records have no inputs, so nothing is fed: `a & b = a ⊎ b`.

```python
Record({"a": 1.0}) & Record({"b": 2.0, "g": {"c": 3.0}})   # Record {a, b, g: {c}}
Record({"g": {"c": 1.0}}) & Record({"g": {"e": 2.0}})      # ValueError (M3): g is an output of both
```

**Laws and kernels.**
A draw `z` of `a & b` at `u` is a record with the fields `outputs(a)` followed by `outputs(b)`.
Its law is defined by sampling the right operand first, since nothing feeds it:

```text
z_b ~ b( u↾names(unfed(b)) )
z_a ~ a( u↾(names(unfed(a)) − fed)  ⊎  rec_b(z_b)↾fed )
z   = rec_a(z_a) ⊎ rec_b(z_b)
```

Where both operands have densities, the same law is:

```text
p_{a&b}(z ; u)  =  p_a( dec_a(z↾outputs(a)) ; u↾(names(unfed(a)) − fed) ⊎ z↾fed )  ·  p_b( dec_b(z↾outputs(b)) ; u↾names(unfed(b)) )
```

IV.2 writes this as `p(F_A, F_B | unmet) = p_A(F_A | bound ∪ (G_A − F_B)) · p_B(F_B | G_B)`, where `F` and `G` are an operand's outputs and its factors' unfed inputs, `bound` is `fed`, and `unmet` is the result's unfed inputs.
Today's joint computes this on its sampling and scoring routes, over the factors of `_factor_graph` (`distributions/_factored.py:351`), with `_components_of` and `_event_of` (`:544`, `:557`).
Its exact conditioning route does not: `condition_on(lik_opt & s_prior, {"mu": 0.0})` draws `y` with `s` at its default though `s_prior` feeds `s` (Appendix E, fact 34). That is a bug under this definition and under today's IV.2; Appendix C.4 records it, and S3's mechanical rename carries it unchanged.

```python
model = lik & prior
# z_prior ~ prior: {mu, tau};  z_lik ~ lik(mu = z_prior["mu"]): an array y
# a draw of model: {y, mu, tau};  p(y, mu, tau) = p_lik(y ; mu) · p_prior(mu, tau)
```

**Functions.**
A function is the kernel whose every law is a point mass, so the law rule reads, for functions:

```text
w_b = rec_b( b(u↾names(unfed(b))) )
w_a = rec_a( a(u↾(names(unfed(a)) − fed) ⊎ w_b↾fed) )
(a & b)(u) = w_a ⊎ w_b
```

```python
center = Function(lambda y: jnp.mean(y), input_spec={"y": ("n",)}, output_spec=OutputSpec(center=()))
spread = Function(lambda y: jnp.std(y), input_spec={"y": ("n",)}, output_spec=OutputSpec(spread=()))
summaries = center & spread                        # input y, shared; outputs center and spread
standardize = Function(lambda y, center, spread: (y - center) / spread,
                       input_spec={"y": ("n",), "center": (), "spread": ()}, output_spec=OutputSpec(z=("n",)))
pipeline = standardize & summaries                 # summaries feeds center and spread; y is shared
pipeline(y=data)                                   # Record {z, center, spread}
pushforward(d, center & spread)                    # the joint law of both summaries of one draw of a law d over y
pushforward(d, center) & pushforward(d, spread)    # their independent product
```

**Batches.**
`(a & b)[j] = ã_j & b̃_j` at every position `j` (§ 8.2).
`RecordBatch.merge` does this today for batches of records, with equal levels (`core/_record_batch.py:754`, `:783`).
VI.11 exempts the operands of `*` from batch lifting (`design/06-operations.md:392`); S0 amends it to say that two batches merge element by element, and aligning other levels by name, with broadcasting, stays after the workshop.

```python
x = sample(Normal("x", 0.0, 1.0), sample_shape=3)
y = sample(Normal("y", 0.0, 1.0), sample_shape=3)
x & y                        # a batch of three records {x, y}: each element is {x: x_j} ⊎ {y: y_j}   (after the workshop)
draws & simulate(draws)      # elements {mu, tau} and {y} give elements {mu, tau, y}   (today: draws.merge(...))
```

### 8.6 Merge on specs

**`O & P` on output specs** is the output spec of two side-by-side terms, with nothing fed, since an output spec has no inputs:

```text
O & P  =  OutputSpec(RecordSpec(O.outputs followed by P.outputs))
```

It applies M2 (each output spec in the keyword form over a base spec, an interim whole form, or positional over a `RecordSpec`, with no type hole), M3, and M8.
An output at the root, named `n`, becomes the child `n`, so its value moves from the root to the path `n`.
A record contributes its children unchanged, so every path of an operand is a path of the result with the same spec (the plan's `graft_identity_B`).

**`I & J` on input specs** is the input spec of two terms that share their inputs, with nothing fed, since an input spec has no outputs.
The inputs of either appear in the order of first appearance; a shared input carries `U` of its specs (M7) and is optional only when optional in every operand that has it; the operands share one dimension scope (M8), whose bindings the result does not keep (§ 8.11).

**`R & S` on record specs** is the union of the children; a top-level name in both raises (M3).
**`&` on `DistributionSpec`, `ConditionalDistributionSpec`, `FunctionSpec`, and `BatchSpec`** is the spec of the merged term, after the workshop (the plan's § 14.4).
So `(a & b).spec == a.spec & b.spec` holds for records from S3, and, for the other categories once their spec merges exist, under § 8.10's condition.

For laws and kernels, R1 and R2 read:

```python
(a & b).event_spec == β(a.event_spec & b.event_spec)
unfed(a & b) == β(unfed(a).without(*fed) & unfed(b))   # the given_spec, when the result is a kernel
```

**The spec of a shared input, before and after the fix.**
Today the joint keeps the first operand's spec for a shared input (`distributions/_factored.py:424`; Appendix E, fact 8), not § 5.12's `U`.
S2b's `InputSpec &`, through which the plan's S2b builds the joint's inputs, keeps today's choice, so S2b changes no joint's behaviour; § 14, Q4(d) asks whether to switch to `U` in S2b, which one test asserts against (Appendix E, fact 35).

```python
OutputSpec(x=NumericArraySpec(())) & grouped.event_spec
# OutputSpec(RecordSpec(x=..., a=..., g=RecordSpec(c=..., e=...)))
grouped.event_spec & OutputSpec(a=NumericArraySpec(()))     # ValueError (M3): a is an output of both
OutputSpec(post.spec) & prior.event_spec                    # TypeError (M2): the first produces a law; wrap it
InputSpec(theta=("p",), x=()) & InputSpec(theta=(3,), z=())
# InputSpec(theta=NumericArraySpec((3,)), x=..., z=...): theta unifies, and p is bound to 3 in both
InputSpec(a=("n",)) & InputSpec(a=(2, 2))                   # ValueError (M7): the ranks differ
InputSpec(o) & InputSpec(p)                                  # how to merge the inputs two output specs o and p feed
```

### 8.7 Functions feed

A function is the kernel whose law at each input is a point mass at its return value (D2: a special case refines the general contract).
So the merge of two functions is the merge of two kernels, read deterministically, and § 8.5 gives its value; a function's optional inputs are always inputs of the function, as a kernel's are.
The plan's "functions pair and never wire" (its § 14.2, rule 3) gave no reason, and this proposal finds none.

**Complete specs.**
A function merges only with a declared input spec and a complete output spec (M2).
A function whose output spec is a type hole, such as `OutputSpec(center=None)`, does not merge: the result's output spec is a record of the operands' outputs, and a `RecordSpec` cannot hold a pending child (Appendix E, fact 28).
`&` raises `TypeError`, naming the hole and saying to give its spec.
A function with no output spec or no input spec, which `Function(f)` gives today (Appendix E, fact 20), raises `TypeError` too, saying which spec to pass.

**A function that returns a law** fails M2, since a law offers no outputs: `fit & center` raises, and `fit_wrapped`, whose body returns `{"posterior": law}`, merges (§ 11.6).

### 8.8 Why each rule

- **M2.** The merged produced term holds, at each output, the value there, and each operand's produced term must be rebuilt from those values, since a joint scores each factor's density on that factor's own draw (IV.2). A base value and a record are rebuilt without loss, by `dec`. A law, a kernel, a function, or a batch held under no name is not a record, so the merge would have to split it into its outputs or invent a name, and the caller wraps it instead, keeping it whole under a name of its choosing. M2's reason is this rebuilding; it does not rest on the rule that a law offers no outputs (§ 14, Q2).
- **M3, and shared inputs.** An output offers a value, so two operands cannot offer one name: the result would have two values for it. An input asks for a value, and one value can fill the input of every operand that has it.
- **M4.** Feeding runs only from right to left. If `b`'s factors took an unfed input that `a` outputs, `a & b` would leave that input unfed while `a` outputs its name, so the result would have one name on both sides (I15). With M4, in a chain `a & b & c` each operand is fed only from its right, so the written order is a topological order of the factors, and the density reads left to right. Since feeding is one-way, two operands could never feed each other anyway, so preventing such a cycle is not M4's job. M4 also does not let merge build every acyclic graph module by module: with `m = c & b`, for `c(z | y)`, `b = Normal("x")`, and `a(y | x)`, both `a & m` and `m & a` raise, while `c & a & b` builds the same graph from its factors (Appendix E, fact 29). IV.2 states the rule as `G_B ∩ F_A = ∅`, the code enforces it, and the plan's § 14.2 omits it. Associativity does not rest on it (Appendix A), so S0 amends IV.2's rationale, as the plan's § 14.5 already says.
- **M5.** With M5, a name of `a` is fed exactly when `b` outputs it, shared exactly when `b`'s factors take it, and the two never overlap.
- **M6 is directional.** The output's values are given, and the input must admit each of them, so the output's spec conforms to the input's.
- **M7 is symmetric.** One value fills a shared input, so neither operand's spec is privileged.
- **M8.** A dimension name means one size wherever specs share one scope (II.1).
- **A merge never lifts.** An output feeds an input only when its spec conforms. Lifting the receiving term over a law-valued output would ignore that law's dependence on everything else the receiver gets, which only the merge itself knows. A lifting rule inside a merge stays after the workshop, as in the plan (its § 14.4, item 4).
- **Records merge shallowly.** A group is a value too. If each operand produced part of a group `g`, the value at `g` would have two sources, which M3 forbids at `g`. A merge of laws is already shallow: two laws that both output a group `g` raise today (`distributions/_factored.py:402`), while `NamedTree.merge` and `Record.merge` merge groups deeply (`core/named_tree.py:835`, `core/record.py:931`). Keeping both rules would let `sample(a) & sample(b)` succeed where `a & b` raises (§ 14, Q4(a)).
- **Two views of one parent merge as independent factors.** `hier[["s1"]] & hier[["s2"]]` is the product of the two marginals, as `hier["s1"] * hier["s2"]` is today, and the joint law of both groups is `hier["s1", "s2"]`. A merge reads names and factors, and a view is one factor, so it cannot tell two views of one parent from two independent laws. Appendix B lists this, and Appendix C.4 records refusing such a merge as a possible later rule.

### 8.9 Properties

Appendix A proves each property and checks it against the reviewers' examples; here are the statements, each with its condition.

1. **Associative, when both groupings are valid.** For laws, kernels, records, record specs, output specs, and batches of records or of base values, when `(a & b) & c` and `a & (b & c)` are both valid they have the same factors, outputs in the same order, unfed inputs with the same specs and optional flags, stored scope, and value. Their validity can differ, and only where the groupings check different pairs of specs: at a name that `a` and `b` both take and `c` outputs, whose specs in `a` and `b` do not unify while `c`'s conforms to each; and where two different free dimension names occur at one axis in one grouping but are bound first in the other. For input specs and functions, which store no scope, both groupings also agree only when no dimension name that one grouping binds occurs free in the operand it is merged with later (§ 8.11). For batches of laws or kernels it can fail, since a batch merge reads each element as one factor (§ 8.2) and so drops the element's factors and stored scope (Appendix B).
2. **Merged operands keep their sides disjoint:** `names(unfed(a & b)) ∩ outputs(a & b) = ∅`.
3. **Commutative up to order** when neither operand feeds the other: `a & b` and `b & a` are valid together and have the same law or value, with outputs in a different order; their input specs are equal once a shared input carries `U` (§ 8.6). Their output specs are unequal, since output order is the layout `to_vector` reads.
4. **Paths are kept.** Every path `p` of `output_spec(a)` is a path of `output_spec(a & b)`, with `output_spec(a & b)[p] == β(output_spec(a)[p])`, and an output of `a` at the root, named `n`, moves to the path `n`; likewise for `b`. Interim whole forms are the exception until G2 (Appendix B).
5. **Sampling commutes with merge** for laws `a` and `b` that do not feed each other and produce records, when `a & b` is valid: `sample(a & b)` is distributed as `sample(a) & sample(b)`.
6. **Renaming commutes with merge.** `(a & b).rename(m) == a.rename(m_a) & b.rename(m_b)`, where `m` maps names to names injectively over the names of `a` and `b`, `m_a` and `m_b` are its restrictions to each operand's names, it renames inputs and outputs alike, and it renames no name that is an unfed input of a law operand's factors, since a law's `rename` does not rename it while the merged kernel's `rename` may (III.9): for `m = lik_opt & Normal("mu", 0.0, 0.001)` and `kw(v | w)`, `(m & kw).rename({"s": "s2"})` renames `s`, while `m.rename({}) & kw.rename({})` keeps it. It applies to laws, kernels, and records; for functions, once `InputSpec.rename` exists. A move of a node to a path does not commute: `grouped` is built by a move, and moving `c` and `e` under `g` before the merge raises M3 at `g`.
7. **Merge reads factors, not only the law.** Two equal laws can merge differently when their factors differ (§ 8.10).
8. **Labels are never read.** Relabeling an operand changes only the result's label.

### 8.10 Reading factors: today's behaviour and its consequences

Before the workshop, a merge of laws and kernels reads its operands' factors, as IV.2's flattening does today, and this proposal keeps that unchanged.
Every consequence is stated here; § 14, Q4(b), asks which rule to adopt for the long term.

1. **A later merge feeds an unfed optional input.** `(lik_opt & prior) & s_prior` feeds `s`, so `y`'s spread is near 100, the same law as `lik_opt & (prior & s_prior)` (Appendix E, fact 25). This keeps associativity (property 1).
2. **A later merge shares it.** For `m = lik_opt & Normal("mu", 0.0, 0.001)`, a law with spread of `y` near 1, and `k2 = conditional_distribution(lambda s: Normal("w", 0.0, s), ...)`, which neither feeds nor is fed by `m`, `m & k2` is a kernel in `s`, required, and binding `s = 30` gives `y` a spread near 30 (fact 40). M7 also checks the specs of such a shared input, so a merge can fail on an input that no spec of `m` shows.
   - **It also appears with no sharing.** For an unrelated kernel `kw(v | w)`, `m & kw` is a kernel whose `given_spec` is `InputSpec(s, w).with_optional("s")`, by R3, so a law that has no inputs (III.7) gains a bindable input `s` by a merge with a term that does not mention it; binding `w = 0` and `s = 30` gives `y` a spread near 30 (fact 46).
3. **Equal laws merge differently.** `lik_opt & prior` and `lik & prior` are equal laws, since `lik` is `lik_opt` at `s = 1.0`, but after `& s_prior` the first gives `y` a spread near 100 and the second near 5 (property 7).
4. **Which results keep the factor depends on the operation and its route**, and not on any rule of `design/`. A result keeps an unfed optional input exactly when it keeps, unchanged, the factor that takes it (fact 41):

   | Law built from `M = lik_opt & Normal("mu", 0.0, 0.001)` | Keeps `lik_opt` as a factor | `& s_prior` gives `y` a spread near |
   | --- | --- | --- |
   | `M[["y", "mu"]]`, a view | no: a view is one factor | 1 |
   | `marginal(M, ["y", "mu"])` | yes: a factored joint | 100 |
   | `M.rename({"y": "w/y"})` | yes | 100 |
   | `condition_on(lik_opt & Normal("mu") & Normal("tau"), {"tau": 1.0})` | yes, on its exact route | 100 |
   | `condition_on(M, {"y": y_observed})` | no: an `EmpiricalDistribution` | — |
   | `condition_on(lik_opt & Gamma("z", 2.0, 1.0), {"mu": 0.0})` | no: `lik_opt` is bound | 1 |
   | a law regrouped so that `lik_opt` is inside a grouped sub-joint | no: the sub-joint is one factor | 1 |

   So a view and a marginal of one joint, which I7 calls one law, can merge differently. Appendix B lists this against I7.
5. **A spec merge can be valid where the term merge raises.** `s_prior.spec & (lik_opt & prior).spec` reads a `DistributionSpec`, which states no unfed input, so it feeds nothing and is valid, while `s_prior & (lik_opt & prior)` raises M4. So `(a & b).spec == a.spec & b.spec` (I16) holds only when neither operand's factors take an unfed input that its input spec does not state.
6. **A batch merge reads each element as one factor** (§ 8.2), so an element's hidden unfed inputs are not fed in a batch merge.

Today's code behaves in every item as stated (facts 25, 40, 41, 43, 46); `condition_on`'s exact route on a joint kernel, which ignores a fed optional input, is the one bug (§ 8.5).

### 8.11 Dimension scopes across merges: today's behaviour and its consequences

A merged law or kernel keeps its scope, as today's joint does, while a merged input spec, output spec, record spec, or function applies `β` and keeps nothing (§ 5.12).
This proposal adds no new storage before the workshop; § 14, Q6, asks what to do after.

1. **Spec and function merges are associative only under a condition.** For `I1 = InputSpec(u=("n",))`, `I2 = InputSpec(u=(3,), q=("n",))`, and `I3 = InputSpec(w=("n",))`, `(I1 & I2) & I3` binds `n = 3` in the first merge and forgets it, so the result keeps `w: ("n",)`, while `I1 & (I2 & I3)` binds `n = 3` across all three and gives `w: (3,)`. Both groupings are valid, and they differ. Property 1 and I19 therefore state their condition for input specs and functions.
2. **A stored scope can refuse a merge over a name no spec shows.** `m1 = a & b3`, for `a(ya: ("n",) | u: ("n",))` and a law `b3` over `u: (3,)`, has the concrete event `{ya: (3,), u: (3,)}` and stores `n = 3`; `m2 = k & lv`, for `k(w: ("n",) | v: ("n",))` and a law `lv` over `v: (4,)`, stores `n = 4`. `m1 & m2` raises M8, and `m1.with_dim_names(n="m") & m2` raises the same, since neither event spec contains `n` (fact 42). `sample(m1) & sample(m2)` is a valid record merge, so property 5 holds only when `a & b` is valid. Appendix B lists this.

Storing the scope is what makes laws and kernels associative (Appendix E, fact 36); forgetting it is what keeps spec merges simple.

### 8.12 What merge is not

- **Not composition.** `f ∘ g` passes `g`'s whole result to `f` by position and returns only `f`'s result, and the product of linear operators `A B` keeps the word "composition" too. A merge matches by name and returns both operands' outputs.
- **Not conditioning.** `condition_on(lik, {"mu": 0.5})` binds `mu` to a fixed value, and the result does not produce `mu`. In `lik & prior`, `mu` is fed by a law and stays an output.
- **Not renaming.** `joint(A, B, **renames)` stays `A & B.rename(**renames)`.

## 9. Invariants

**Symbols.**
`d` is a law, `t` a term other than an array, `f` a function, `k` a kernel, `bt` a batch, and `r` a record.
`a`, `b`, and `c` are operands of one merge category.
`s` is a path or a list of paths, `p` and `q` are paths, `ps` is a list of paths that is not empty, has no path inside another, and leaves something out, and `j` is a position.
`I` is an input spec, `ns` a list of its input names, and `name` an output name.
`w` is an input record (§ 5.4), and `X` a draw (§ 5.4).
`given` is a mapping from paths of `d`'s event to values, and `fixed` is the set of those paths.
`m` is a rename, `key` a `str`, and `fed`, `unfed`, `names`, and `β` are as § 8.1 defines them.
"Equal" has the plan's § 3 meaning: equal contents for values, and for laws and functions equal output specs and the same distribution or map; merge reads factors, which this equality does not compare (§ 8.10).
The third column says whether the definitions imply the invariant ("follows") or this design chooses it ("rule"); the fourth names the step that delivers it.
Appendix B lists the cases where an invariant does not hold on purpose.

| # | Invariant | Follows or rule | Delivered by |
| --- | --- | --- | --- |
| I1 | **One set of paths.** A path or a list of paths valid on `d` is valid on `sample(d)`, on each element of `sample(d, sample_shape=n)`, on `mean(d)`, and on `d.event_spec` | follows from § 5.8: a term's paths are its output spec's | S1 for records and batches, S2b for output specs, G2 for the draws of the G2 families, G3 across kinds |
| I2 | **Indexing commutes with sampling.** `sample(d[s])` is distributed as `sample(d)[s]`, at every path, held fields included | follows: it is § 7.1's definition of `d[s]` | S1 for lists, G1a for groups, G3 at held terms |
| I3 | **Indexing commutes with summaries.** `mean(d)[s]` and `mean(d[s])` estimate one quantity, equal on exact routes; likewise the other per-field summaries | follows from I2 and the summaries rule of #576 | #576 with S1 |
| I4 | **Positions and paths commute on a batch.** `bt[j][s] == bt[s][j]` | follows from § 5.4 and § 7.1 | S1 |
| I5 | **Paths chain, across kinds.** `t[p][q] == t[p + "/" + q]` | follows from § 5.8, case 2 | S1 for records and batches, G1a for laws, G3 across kinds |
| I6 | **Output specs agree.** For a law, a kernel, a function, or a batch, `output_spec(t[s]) == output_spec(t)[s]`: `d[s].event_spec == d.event_spec[s] == marginal(d, s).event_spec`, `f[s].output_spec == f.output_spec[s]`, `k[s].event_spec == k.event_spec[s]`, and `bt[s].spec.element_spec == bt.spec.element_spec[s]`; for a record, `r[s].spec == r.spec[s]` | follows from § 7.1 and § 7.3 | S2b for the paths S1 supports, G1a for groups, G3 at held fields, S4 for batches with the output at the root from G3, G5 for kernels |
| I7 | **A view and a marginal are one law.** `marginal(d, s)` equals `d[s]`, detached | rule (VI.8) | S1 for lists, G1a for groups |
| I8 | **Draws of a list of paths condition back.** `condition_on(d, sample(d[ps]))` fixes the nodes at `ps` and nothing else | rule (decided 2026-10-15) | G4 |
| I9 | **One path gives the node; a list gives a record that holds it.** `t["g"]` is the node, and `t[["g"]]` is the record `{g: ...}`, or the term whose produced term is that record; for the output at the root, named `n`, `t[[n]]` is likewise the record `{n: ...}`, or the term that produces it | rule (decided 2026-10-15; the plan's O6) | S1 |
| I10 | **A list and `without` partition.** Every leaf path of `t` is a leaf path of exactly one of `t[ps]` and `t.without(*ps)`, on records (for paths that do not go through a held term, until after the workshop), record specs, batches of records, output specs, and laws; every input of `I` is in exactly one of `I[ns]` and `I.without(*ns)` | follows from § 7.3 to § 7.5 | S1 for records, record specs, and batches; S2b for output and input specs; G4 for laws |
| I11 | **Conditioning removes what it fixes.** `condition_on(d, given).event_spec == d.event_spec.without(*fixed)`, whichever route computes it | rule (decided 2026-10-15) | G4 |
| I12 | **The term at an output.** For every term `t` with outputs, `t.output(name)` is `t[name]`, or `t` for the output at the root, and `output_spec(t).outputs[name]` is the term spec of `t.output(name)`'s produced term | follows from § 5.9 and § 7.7 | G3; S4 stores what it reads for batches |
| I13 | **Indexing commutes with producing.** `f[s](w) == f(w)[s]`; the law of `k[s]` at `w` is the law of `X`'s node at `s`, for `X ~ k(w)` | follows: it is § 7.1's definition | G3 for functions, G5 for kernels |
| I14 | **Indexing keeps inputs** (new). `f[s].input_spec == f.input_spec`; `k[s].given_spec == k.given_spec` | follows from § 7.1 | G3, G5 |
| I15 | **Sides are disjoint** (new). `names(unfed(t)) ∩ outputs(t) = ∅` for every kernel, law, and merge | rule for kernels (III.9, `distributions/_conditional.py:131`); follows for merges from M4 and M5 (§ 8.9, property 2) | today for kernels; S3 for merges |
| I16 | **The spec of a merge is the merge of the specs, after substitution** (new). For records, laws, kernels, and functions, `output_spec(a & b) == β(output_spec(a) & output_spec(b))` and `unfed(a & b) == β(unfed(a).without(*fed) & unfed(b))`; for batches, the same holds of the element merge; `(a & b).spec == a.spec & b.spec` when neither operand's factors take an unfed input that its input spec does not state (§ 8.10, item 5) | follows from R1 and R2 | S2b for the specs, through `_factor_graph`; S3 for the operator; spec merges of laws after the workshop |
| I17 | **A merge keeps paths** (new). Every path `p` of `output_spec(a)` is a path of `output_spec(a & b)`, with `output_spec(a & b)[p] == β(output_spec(a)[p])`, and likewise for `b`; an output at the root moves to the path of its name | follows from § 8.6 (§ 8.9, property 4) | S2b for specs, S3 for laws; for interim whole forms, from G1a or G2 |
| I18 | **Membership agrees with `[]`** (new). `key in t` if and only if `t[key]` succeeds, on every term other than an array, and on `RecordSpec`, `OutputSpec`, and `InputSpec`; `rename` takes the same paths, and the name of the output at the root | rule | S1 for terms and record specs, S2b for output and input specs, S3 for `rename` |
| I19 | **Merge is associative when both groupings are valid** (new), under § 8.9, property 1's conditions: for input specs and functions, also when no dimension name one grouping binds occurs free in the operand merged with later; for batches, only of records or of base values | follows (§ 8.9, property 1) | holds today for `*`, apart from the shared-input specs of § 8.6; S3 |
| I20 | **Sampling commutes with merge** (new). `sample(a & b)` is distributed as `sample(a) & sample(b)` for laws that do not feed each other and produce records, when `a & b` is valid | follows from § 8.5 and the shallow record rule (§ 8.9, property 5) | S3, with S2b's shallow rule |
| I21 | **Merge commutes with renaming** (new). `(a & b).rename(m) == a.rename(m_a) & b.rename(m_b)` for laws, kernels, and records, under § 8.9, property 6's conditions on `m` | follows (§ 8.9, property 6) | S3; for functions, after the workshop |
| I22 | **Merge reads no label** (new). Relabeling an operand changes only the result's label | follows from § 8.3 and § 8.4, which read only names and factors | today |

I2, I4, I5, I6, and I13 follow from the definitions, so their tests check the implementation against the definition; once G3 has merged, I2, I5, I6, and I9 hold with no exception at held fields.
The plan's I11 is kept as it was; binding a kernel keeps today's rule (§ 5.3).

## 10. API sketch

Only the public members this proposal adds or changes are shown; the rest is the plan's.
Each comment is the contract a user reads.
Names follow this proposal's vocabulary; under § 14, Q1's option 2 they arrive in the rename after the workshop, and until then `outputs` is spelled `components`, `output_name=` is `component=`, and `t.output(c)` is `t.component(c)`; `element_spec` is a final name, which S4 introduces directly.

### 10.1 `OutputSpec`

```python
class OutputSpec:
    """The term spec of what a term produces, and the name it is offered under when it is a base value."""

    @overload
    def __init__(self, term_spec: TermSpec, /) -> None: ...
    # The positional form, over a spec with parts (§ 5.1): a non-empty RecordSpec, a DistributionSpec,
    # a ConditionalDistributionSpec, a FunctionSpec, or a BatchSpec (a FunctionSpec, and a BatchSpec of
    # base elements, from step G1b). TypeError for a base spec, which needs a name; ValueError for an
    # empty RecordSpec or a child name that is not a name.
    @overload
    def __init__(self, **named_output: TermSpec | tuple | None) -> None: ...
    # The keyword form: one keyword, the name of the output at the root, over a base spec, a size
    # tuple, or None for a type hole. Until step G2 a spec with parts gives an interim whole form;
    # from G2 it raises TypeError: "params would name a record; give OutputSpec(RecordSpec(...)) to
    # offer its fields, or wrap it: OutputSpec(RecordSpec(params=...))".

    @property
    def term_spec(self) -> TermSpec | None: ...
    # The produced term's type; None only while a type hole is pending.
    @property
    def outputs(self) -> Mapping[str, TermSpec | None]: ...
    # Each output's name and spec, in order (§ 5.9). The children when the term spec is a RecordSpec;
    # the one name at the root in the keyword form; the columns when it is a BatchSpec of values; the
    # output functions when it is a FunctionSpec; none when it is the spec of a law or a kernel.
    def __getitem__(self, paths: str | Sequence[str], /) -> OutputSpec: ...
    # The output spec at paths, by § 7.3's four rules. The path reader's errors; KeyError for a str that
    # is not a path (until G3, the name of the output at the root gives self).
    def __contains__(self, path: object, /) -> bool: ...
    # Whether path is a str and self[path] succeeds.
    def without(self, *paths: str) -> OutputSpec: ...
    # The other nodes, each at its path (§ 7.5). KeyError for a path that is not a path of self, every
    # path of a keyword form over a base spec included; ValueError when nothing would be left.
    def __and__(self, other_output_spec: OutputSpec, /) -> OutputSpec: ...
    # The outputs of both, side by side, self's first (§ 8.6). TypeError for an operand that is not an
    # OutputSpec, a positional form over a law, kernel, function, or batch spec, or a type hole;
    # ValueError for a name in both, or a dimension bound to two sizes.
    def with_term_spec(self, term_spec: TermSpec, /) -> OutputSpec: ...
    # Fill a type hole, or complete the stated term spec with the produced one, as today's with_spec (II.2).
    def rename(self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str) -> OutputSpec: ...
    # Today's with_path_names (step S3): rename or move nodes by their paths, or rename the output at the root.
    @classmethod
    def default(cls, term_spec: TermSpec, *, output_name: str) -> OutputSpec: ...
    # The output spec a function uses for its return value when given none: the keyword form under
    # output_name for a base spec; for a batch whose elements carry only the default name element, the
    # batch whose elements are named output_name (the plan's O2); the positional form for every other
    # spec with parts (step G1b). From G2 the distribution families' events use it too.
    # unchanged: is_concrete, with_dim_sizes, with_dim_names; free dimensions are term_spec.free_dims
    # equality: the same term spec and, for an output at the root, the same name; order-sensitive;
    # an OutputSpec equals only another OutputSpec, and the hash agrees (the plan's § 7.8)
```

`OutputSpec` is not a `Mapping`, not iterable, and has no `output` method: `O.outputs[c]` gives the spec of an output, and `O[c]` the output spec at a path.
`spec`, `with_spec`, `with_path_names`, `exposes_record`, and `components` go, with no aliases.

### 10.2 `InputSpec`

```python
class InputSpec(Mapping[str, TermSpec]):
    """A term's inputs: each input's name and term spec, in order, with the optional ones."""

    def __init__(self, inputs: InputSpec | Mapping[str, TermSpec | tuple] | RecordSpec | OutputSpec | None = None,
                 /, **inputs_by_name: TermSpec | tuple) -> None: ...
    # Reads one source (§ 7.4): an InputSpec as it is; a mapping; a RecordSpec by its children; an
    # OutputSpec by its outputs, each typed by its output's spec. InputSpec(R) == InputSpec(OutputSpec(R)).
    # Every input of a mapping, record spec, or output spec is required. ValueError for a name that is
    # not an identifier; TypeError for a type hole, any other source, or both forms at once.
    @property
    def required(self) -> tuple[str, ...]: ...                  # unchanged
    @property
    def optional(self) -> frozenset[str]: ...                   # unchanged
    def with_optional(self, *names: str) -> InputSpec: ...      # unchanged
    @overload
    def __getitem__(self, name: str, /) -> TermSpec: ...
    # That input's term spec; from G4, a path into a record input gives the spec there. KeyError for a
    # name that is not an input.
    @overload
    def __getitem__(self, names: Sequence[str], /) -> InputSpec: ...
    # The input spec of those inputs, in this spec's order, optional ones staying optional (S2b). The
    # path reader's errors; KeyError for a name that is not an input.
    def without(self, *names: str) -> InputSpec: ...
    # The other inputs, optional ones staying optional; may leave none. KeyError for a name that is not
    # an input. From G4, a path into a record input leaves the rest of that record, and the input
    # becomes required.
    def __and__(self, other_input_spec: InputSpec, /) -> InputSpec: ...
    # The inputs of both, in order of first appearance; a shared input carries the unification of its
    # specs (§ 5.12; the first operand's spec until § 14, Q4(d)'s fix) and is optional only when
    # optional in every operand that has it; one dimension scope, not kept. TypeError for an operand
    # that is not an InputSpec; ValueError when two specs do not unify or a dimension is bound to two sizes.
    def rename(self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str) -> InputSpec: ...
    # After the workshop (the plan's § 6.6).
    # equality: the same inputs, specs, and optional inputs, in any order (unchanged)
```

### 10.3 Batches

```python
class BatchSpec(TermSpec):
    def __init__(self, element: TermSpec | OutputSpec, levels=None, /, **level_shapes) -> None: ...
    # element is the elements' type, or an OutputSpec that names base elements (keyword form) or gives
    # a record element (positional form) (the plan's O8). A bare base spec gets the name "element".
    @property
    def element_spec(self) -> OutputSpec: ...
    # The output spec of one element as a term (§ 5.7): the keyword form under a base element's name;
    # otherwise the positional form over the element's term spec (step S4). Its term_spec is what
    # today's element_spec returns.

class Batch:
    @property
    def element_spec(self) -> OutputSpec: ...          # a view on spec; its outputs replace II.5's Batch.components
    def __and__(self, other_batch: Batch, /) -> Batch: ...
    # The merge of the elements, position by position, each read as one factor (§ 8.2). ValueError unless both have equal levels;
    # TypeError for elements of different kinds. RecordBatch: today's merge, with the shallow rule (S2b),
    # under its new name (S3); other batches after the workshop.
```

A batch's own output spec is `OutputSpec(bt.spec)`, derived, whose outputs are its columns (S4); a `DistributionBatch`'s and a `ConditionalDistributionBatch`'s `event_spec` is `element_spec.term_spec.event_spec` (III.10).

### 10.4 Terms

```python
class Record:
    def __and__(self, other_record: Record, /) -> Record: ...
    # The fields of both, self's first. TypeError for an operand that is not a record; ValueError for
    # a top-level name in both, groups included. Replaces merge (shallow from S2b, renamed in S3).
# RecordSpec.__and__ and NamedTree.__and__: the same rule.

class Distribution:
    def __getitem__(self, paths: str | Sequence[str]) -> Distribution: ...
    # The view at paths (§ 7.1), whose event_spec is self.event_spec[paths]; until G3 the name of the
    # output at the root gives self, and a view at a held field keeps today's whole form.
    def output(self, output_name: str, /) -> Distribution: ...
    # self[output_name], or self for the output at the root (G3). KeyError for a name that is not an output.
    def without(self, *paths: str) -> Distribution: ...      # step G4
    def __and__(self, other: Distribution | ConditionalDistribution, /) -> Distribution | ConditionalDistribution: ...
    # The merge of § 8, read over both operands' factors (§ 8.10): both sets of outputs side by side,
    # other's outputs feeding the inputs of this law's factors that no factor feeds. Raises as § 8.3
    # states; returns NotImplemented for an operand that is neither a law nor a kernel. Replaces __mul__ (S3).
# ConditionalDistribution: the same members; __getitem__ and output from step G5.

class Function:
    def __init__(self, fn, /, *, input_spec=None, output_spec: OutputSpec | TermSpec | None = None, **rest) -> None: ...
    # A bare TermSpec as output_spec reads as OutputSpec.default(term_spec, output_name=<the callable's
    # output name>) (the plan's § 5.6), from G1b for specs with parts. A parameter with a default is an
    # optional input, and input_spec marking optional a parameter without one raises ValueError (O9).
    def __getitem__(self, paths: str | Sequence[str]) -> Function: ...   # w -> self(w)[paths] (G3)
    def output(self, output_name: str, /) -> Function: ...                # G3
    def __and__(self, other_function: Function, /) -> Function: ...
    # § 8.5 and § 8.7, after the workshop. TypeError for an operand with no input spec, no output spec,
    # or a type hole, naming it.
```

`factor(d, output_name)` takes the name that its parameter `component_name` takes today, and each distribution family takes the name of its output at the root as `output_name`, its first parameter, as `Normal("x", 0.0, 1.0)` does.
`conditional_distribution(fn, given_spec=...)`, `ConditionalDistributionSpec(given_spec, ...)`, and `Function(fn, input_spec=...)` read their argument through the `InputSpec` constructor (the plan's § 6.2).

## 11. Worked examples

Each example states what users write and what it returns once every step has merged; "Today" lines show `6f8859fc`.
§ 7 and § 8 already show `grouped`, `hier`, the records, and the batches; this section gives the running examples that span several rules.

### 11.1 `prior`, `lik`, `model`, and `post`

```python
prior.event_spec                     # OutputSpec(RecordSpec(mu=..., tau=...))
prior.event_spec.outputs             # {mu: NumericArraySpec(()), tau: NumericArraySpec(())}
prior.event_spec["mu"]               # OutputSpec(mu=NumericArraySpec(()))
InputSpec(prior.event_spec)          # InputSpec(mu=..., tau=...): the inputs prior's outputs could feed

model = lik & prior                  # Today: lik * prior
# fed = {mu}; outputs y, mu, tau; no unfed input left, so model is a law
model.event_spec == lik.event_spec & prior.event_spec        # I16: OutputSpec(RecordSpec(y=..., mu=..., tau=...))
prior & lik                          # ValueError (M4): lik takes mu, which prior outputs; put prior on the right

post = condition_on(model, {"y": y_observed})
post.event_spec                      # OutputSpec(RecordSpec(mu=..., tau=...)), on every route (I11, step G4)
post["mu"].event_spec == post.event_spec["mu"]               # I6: OutputSpec(mu=NumericArraySpec(()))
condition_on(lik, mean(post[["mu"]]))                        # a plug-in estimate: the record {mu} binds lik's input
hier["s1"] & hier["s2"]              # ValueError (M3): both output theta     (Today: they merge as whole terms s1, s2)
```

### 11.2 `grouped`, and a kernel that takes a group

```python
grouped["g"].event_spec == grouped.event_spec["g"]           # OutputSpec(RecordSpec(c=..., e=...))   (G1a)
lik_g & grouped[["g"]]               # grouped[["g"]]'s output g feeds lik_g's input g
lik_g & grouped["g"]                 # nothing is fed: grouped["g"] outputs c and e, so lik_g keeps its input g
Normal("x", 0.0, 1.0) & grouped      # a law over {x, a, g: {c, e}}; every path of grouped is unchanged (I17)
```

### 11.3 Optional inputs across merges

```python
lik_opt.given_spec                   # InputSpec(mu=..., s=...).with_optional("s")
m1 = lik_opt & prior                 # a law: mu is fed, and s stays unfed at its default 1.0
m1 & s_prior                         # s is fed: y's spread is near 100, as today (§ 8.10, item 1)
lik_opt & (prior & s_prior)          # the same law (I19)
condition_on(lik_opt, {"mu": 0.3})   # a law, with s at its default: binding keeps today's rule (§ 5.3)
```

### 11.4 `growth_prior`, `shocks_given_sigma`, and the likelihood

A likelihood conditions on `K`, `r`, `phi`, and the shocks `e`, which `shocks_given_sigma` produces given `sigma`.
`likelihood_fn(K, r, phi, e)` returns the law of the data `y`.

```python
# Today (the plan's § 6.1, and the workshop tutorials 3 and 4)
shock_spec = RecordSpec(shocks_given_sigma.event_spec.components)
likelihood_given_spec = growth_prior.event_spec.spec.without("sigma").merge(shock_spec)

# Proposed
likelihood = conditional_distribution(
    likelihood_fn,
    given_spec=growth_prior.event_spec.without("sigma") & shocks_given_sigma.event_spec,
    label="likelihood",
)
# the output at the root e becomes the child e; given_spec reads the four outputs as inputs K, r, phi, e
list(likelihood.given_spec)          # ["K", "r", "phi", "e"]

model = likelihood & shocks_given_sigma & growth_prior
# likelihood & shocks_given_sigma: e is fed; unfed inputs K, r, phi, sigma; outputs y, e
# ... & growth_prior: K, r, phi, and sigma are fed; no unfed input left, so model is a law
list(model.event_spec.outputs)       # ["y", "e", "K", "r", "phi", "sigma"]
shocks_given_sigma & likelihood      # ValueError (M4): likelihood takes e, which shocks_given_sigma outputs
```

The tutorial idioms map one to one; the tutorials keep theirs until their rewrite (README directive 6).
In the table, `kept = growth_prior.event_spec[["K", "r", "phi"]]`, an output spec.

| Today | Proposed |
| --- | --- |
| `given_spec=prior.event_spec.components` | `given_spec=prior.event_spec` |
| `{"sigma": growth_prior.event_spec.components["sigma"]}` | `growth_prior.event_spec["sigma"]` |
| `{name: growth_prior.event_spec.components[name] for name in ["K", "r", "phi"]}` | `kept`, that is `growth_prior.event_spec[["K", "r", "phi"]]` |
| `{**inputs, "e": shocks_given_sigma.event_spec.components["e"]}` | `kept & shocks_given_sigma.event_spec`, a merge of two output specs, which raises where `kept` already offers `e` |
| `growth_prior.event_spec.spec.without("sigma")` | `growth_prior.event_spec.without("sigma")` |
| `list(model.event_spec.components)` | `list(model.event_spec.outputs)` |
| `posterior[tuple(prior.event_spec.components)]` | `posterior[tuple(prior.event_spec.outputs)]`, since a tuple of `str` is a list of paths |

### 11.5 `emulator` and `fit_emulator`

```python
emulator["mean"]                     # the function theta -> emulator(theta)["mean"]   (step G3)
emulator["mean"](theta) == emulator(theta)["mean"]           # I13
emulator.output("mean") == emulator["mean"]                  # I12

fit_emulator = Function(train_gp, output_spec=OutputSpec(FunctionSpec(
    InputSpec(theta=theta_spec), OutputSpec(RecordSpec(mean=("n_days",), var=("n_days",))))))
list(fit_emulator.output_spec.outputs)       # ["mean", "var"]: the returned function's outputs (step G1b)
fit_emulator.output_spec["mean"]
# OutputSpec(FunctionSpec(InputSpec(theta=...), OutputSpec(mean=...))): the spec of the function emulator["mean"]
fit_emulator & center                # TypeError (M2): fit_emulator produces a function; wrap it to merge it
```

A function that returns a function offers outputs, while one that returns a law does not: a function into records is the record of its output functions, and a law over records is not the record of its marginals (§ 5.9).

### 11.6 A function that returns a law, and lifting it

```python
fit = Function(run_inference, output_spec=post.spec)     # a bare law spec reads positionally (step G1b)
fit.output_spec.outputs              # {}: a law, with no outputs
fit["mu"]                            # the function y -> run_inference(y)["mu"]   (step G3)
fit.output_spec["mu"] == fit["mu"].output_spec               # OutputSpec(DistributionSpec(OutputSpec(mu=...)))

def run_inference_wrapped(y_observed):
    return {"posterior": run_inference(y_observed)}           # the body wraps the law under a name
fit_wrapped = Function(run_inference_wrapped, output_spec=OutputSpec(RecordSpec(posterior=post.spec)))
fit_wrapped.output_spec["posterior"] # OutputSpec(post.spec), as fit_wrapped(y)["posterior"] is the law
fit & center                         # TypeError (M2): fit produces a law; merge fit_wrapped instead
```

Today `Function(run_inference, output_spec=post.spec).output_spec` is `OutputSpec(run_inference=DistributionSpec(...))`, and `OutputSpec(post.spec).components` is `{mu: NumericArraySpec(()), tau: NumericArraySpec(())}`, the specs of a draw's fields rather than of anything the function returns.

**Lifting a function whose produced term is not a value.**
A lift builds its own law, an `EmpiricalDistribution` of the evaluations, and names that law's draws itself.
When the returned term is a base value or a record, the lifted law's event spec is the function's output spec.
When it is a law, a kernel, a function, or a batch, which a record could not hold under no name (M2's test), the event is the record that holds the returned term whole under the callable's output name (`_callable_component`, `values/_function_base.py:230`), as the plan's § 5.3 and G1b state.
With `include_inputs=True`, the record also holds each lifted argument under its parameter's name, as today (V.5).
The reason is C5 applied to a law the lift constructs, as the G2 families name theirs.

```python
def law_of_y(x):
    return Normal("y", x, 1.0)
Function(law_of_y)(Normal("x", 0.0, 1.0)).event_spec
# OutputSpec(RecordSpec(law_of_y=DistributionSpec(...))): draws {law_of_y: law}   (step G1b; today: OutputSpec(law_of_y=...))
Function(law_of_y).using(include_inputs=True)(x=Normal("x", 0.0, 1.0)).event_spec
# OutputSpec(RecordSpec(x=..., law_of_y=DistributionSpec(...))): draws {x, law_of_y: law}, as today
```

## 12. Differences from `declarations-and-addressing.md`

§ 3 lists the decisions this proposal reverses.
This section lists every other difference: § 12.1 the rules it adds, restates, or makes precise, § 12.2 the renames, and § 12.3 the change to the word "unify" in `design/`; Appendix C.1 lists the plan's text that must change.

### 12.1 Rules added, restated, or made precise

| # | Concept or rule | The plan says | This proposal says | Why |
| --- | --- | --- | --- | --- |
| 1 | What a term produces | not defined | the produced term (§ 5.4), random for a law or a kernel | Lets `[]` and merge be stated once |
| 2 | Records | neither producers nor consumers (§ 4.3), yet merged (§ 14.2) | outputs, its children; no inputs (§ 6) | A record offers values in a merge |
| 3 | The spec of a batch's output | the spec of the declared value at the address (§ 4.4; § 9.7, line 1329), the column | the same, the column's `BatchSpec` (§ 5.9); today's code types it by the element (`core/_specs.py:302`) | Agrees with the plan, departs from the code; S4 changes `_exposed_components` for batches, and a probe of the whole suite found no test that depends on today's typing (Appendix E, fact 30) |
| 4 | `[]` on a record | a record gives a record or a value (§ 9) | the node itself, stated per kind (§ 7.1) | A held law is a node, not something a record produces |
| 5 | The order rule | omitted from § 14.2; § 14.5 keeps its purpose | M4, over the unfed inputs of the factors, optional ones included | IV.2 requires it, and the code enforces it, optional inputs included (`distributions/_factored.py:431`; Appendix E, fact 33) |
| 6 | Merge, defined | four rules and a table by kind (§ 14.2) | one definition, conditions M1 to M8, and a value rule per merge category (§ 8) | The maintainer's first criticism |
| 7 | Flattening | IV.2: "`A * B * C` builds one flat N-factor joint", not restated in § 14 | stated as today's behaviour, with every consequence (§ 8.10) | Associativity with optional inputs rests on it; § 14, Q4(b) asks the long-term rule |
| 8 | Conformance of a fed input | "must unify" (§ 4.4) | the output's spec conforms to the input's (M6, § 5.12), with III.9's variance for law and function specs | The input must admit the values it is fed; the code already checks one direction (`distributions/_factored.py:437`) |
| 9 | The spec of a shared input | "carries the unification" (§ 4.4, § 6.4), not defined | `U`, total, in IV.2's sense (§ 5.12); today's joint keeps the first operand's spec until § 14, Q4(d)'s fix | States the rule; keeps S2b small unless Q4(d) chooses otherwise |
| 10 | Dimension scope of a merged term | not stated | a merged law or kernel stores its scope; spec and function merges do not (§ 5.12, § 8.11) | Today's `_FactorGraph.scope`; § 14, Q6 asks the long-term rule |
| 11 | A shared optional input that a binding omits | not stated | each factor takes its own default (§ 8.4) | An `InputSpec` stores no defaults; today's behaviour |
| 12 | Type holes in a merge | `OutputSpec &` raises on one (§ 7.7), while § 14.4 merges `center & spread`, which hold type holes | refused everywhere (M2) | A `RecordSpec` cannot hold a pending child |
| 13 | A function with no input spec in a merge | not addressed | refused (M2) | A merge reads input names |
| 14 | An operand with one name on both sides | not addressed | refused (M5) | Keeps I15 under merge |
| 15 | Merges across categories | not addressed | `TypeError`; the point-mass reading goes to `deferred.md` (§ 8.2) | A point-mass factor has no density |
| 16 | Batches | element by element, after the workshop (§ 14.4); two batches of arrays raise or are bitwise (§ 16) | defined by the element merge, with base elements read as records, so two batches of arrays merge into a batch of records (§ 8.2); VI.11's exemption amended | Makes the plan's sentence precise; the plan's § 16 clause on batches of arrays is replaced |
| 17 | Interim whole forms | kept until G2 in `project`, `without`, `[]`, `rename`, and `&` (§ 5.2); `&` refuses only an exposed law, function, or batch (§ 7.7) | the same, with each rule written out (§ 5.5), M2's interim clause (§ 8.3), and the grouped sub-joint's form ending at G1a | Directive 8: today's merges of the G2 families keep working |
| 18 | Views at held fields until G3 | today's whole form, with I6's interim form (§ 5.3, § 7.5) | the same; S2b's `_view_declaration` switch leaves them out (§ 7.2) | Keeps I6 until G3 |
| 19 | The lifted function whose result is not a value | wraps its draws under the function's name (§ 5.3, § 5.5, G1b) | the same, with the reason stated (§ 11.6) | The lift builds its own law (C5) |
| 20 | Symbolic dimensions in selection | not mentioned | kept, never bound; selection commutes with substitution (§ 7.8) | A gap |
| 21 | A list of paths, and `without`, through a held term | implied (§ 10) | `restrict` and `remove`; an emptied held field goes, as an emptied group does (§ 7.3, § 7.5) | A gap; keeps I10 |
| 22 | The list form on the name of the output at the root | accepted (O6) | kept, as an extra rule at the top level, read against the record of outputs (§ 7.3, rule 3), with its exceptions (Appendix B) | States where the rule applies |
| 23 | I6, I9, I12 | with `project`, the wrap, and `component` | restated per kind (§ 9) | Row 4 and § 3, rows 2 and 6 |
| 24 | New invariants | — | I14 to I22, each with its stated condition (§ 9) | Each is a property the definitions give and tests can check |
| 25 | Deliberate exceptions | include the wrap at held laws (§ 16) | that row removed, the rows of § 8.10 and § 8.11 added (Appendix B) | § 3, row 6, and the consequences above |
| 26 | Two attribute names per side | not explained | kept, explained by role (§ 5.7) | The roles differ |
| 27 | IV.2's rationale | S0 rewrites it to say associativity holds with or without the order rule (§ 14.5) | the same, with § 8.9's statement of when validity can differ | jhuggins's model does not model specs, dimension names, or optional inputs |
| 28 | The word "unify" in `design/` | four senses: conforming, unifying, completing, and naming the code | one per relation (§ 12.3) | Directive 11 |
| 29 | IV.2's "unmet" | an operand's "unmet given slots" | "unfed", for every option of § 14, Q1 (§ 12.3) | One word for one concept (directive 11) |

### 12.2 Renames

All of these follow § 14, Q1, which decides when they happen.

| # | The plan | This proposal | Why |
| --- | --- | --- | --- |
| 1 | input declaration, output declaration | input spec, output spec | The class names |
| 2 | slot, given slot, optional slot | input, a kernel's input, optional input | Plain and parallel |
| 3 | component | output; "component" keeps only a mixture's component laws | Plain and parallel; settles O14 |
| 4 | component spec | the spec of an output | A plain phrase |
| 5 | address | the output at the root, or a top-level path | Two places, two plain phrases |
| 6 | whole declaration, exposed declaration, exposes | the keyword and positional forms | The term spec decides the form, from G2 |
| 7 | meets, is matched, wire, bound, connections | feeds | One word for one relation |
| 8 | `components` (removed in the plan), `decl[c]` | `O.outputs`, `O.outputs[c]` | § 3, row 8 |
| 9 | `t.component(c)` | `t.output(c)` | § 3, row 2 |
| 10 | `BatchSpec.element_declaration`, `Batch.components` | `element_spec`, and its `outputs` | § 3, row 3 |
| 11 | the distribution families' keyword `component=`, `factor(d, component_name)`, `OutputSpec.default(component=)` | `output_name` | § 14, Q1 |
| 12 | the private names `_exposed_components`, `_component_name`, `_components_of`, `_callable_component`, and the page `docs/api/declarations.md` | `_exposed_outputs`, `_root_output_name`, `_outputs_of`, `_callable_output_name`, and `docs/api/input_and_output_specs.md` | Directive 5, rule 5, in the step that renames the vocabulary |

### 12.3 The word "unify" in `design/`

S0 gives each relation one word, as § 5.12 states, and rewords these passages; the last but one row also replaces IV.2's "unmet", one concept that this proposal calls "unfed":

| Sense | Passages in `design/` | S0 writes |
| --- | --- | --- |
| a directional check that a value or a spec is admitted, binding dimension names as it goes | II.1's binding (`02-shared-abstractions.md:23`, and the comments at `:39` and `:40`); III.7's leaf check (`03-term-kinds.md:461`); IV.4's abstract draw and explicit event spec (`04-distributions.md:162`, `:179`); V.6's planning (`05-functions.md:14`, `:42`, `:149`, `:204`); VI.0's slot (`06-operations.md:160`); VI.6's `given` (`:286`) | "conforms", and for several specs at once "conform in one dimension scope", II.1's words |
| the symmetric spec of values that both admit | IV.2's same-named givens (`04-distributions.md:86`, `:87`); III.2's opaque specs (`03-term-kinds.md:64`, the sentence that begins "Two opaque specs unify") | "unify", with § 5.12's `U` |
| both relations in one sentence | IV.2's composition errors (`04-distributions.md:83`, "matched specs that do not unify") | "an output that does not conform to the input it feeds, or shared inputs that do not unify" |
| completing a declared spec with a produced one, keeping the declared dtype and support | II.2's `with_spec` comment and completion (`02-shared-abstractions.md:87`, `:113`); III.2's clause on a declared `meta` (`03-term-kinds.md:64`); VII.2's declared type (`07-distribution-catalog.md:60`) | "completes", II.2's own heading, with "conforms" for its check |
| an input of a joint's factors that no factor feeds | IV.2's "unmet" (`04-distributions.md:18`, `:66`, `:70`, `:73`, `:76`, `:78`, `:81`, `:85`); VI.6's and VI.9's "unmet givens" (`06-operations.md:294`, `:368`) | "unfed", with § 8.1's `unfed(t)`; the variable `unmet` in IV.2's formulas becomes `unfed` |
| names of today's code | `package-structure.md:29`, `:34`, `:142` ("dimension unification", "unification (III.5)") | unchanged, since they describe `_unify_specs` and `_unify_array_shape` |

The code's `_unify_specs` keeps its name until a step changes it (directive 5, rule 5).

## 13. Effect on the push

**What this proposal gates.**
It changes only the declarations plan, so it gates S0 and the declaration steps after it (S1 through G5), not PL or the execution steps E0 and E1, which depend on PL and P0 and not on this proposal.
**Recommendation:** merge PL without waiting for this proposal, with a banner on `plans/declarations-and-addressing.md` naming it as pending, and merge this proposal, with the maintainer's answers to § 14 and Appendix C's edits, as a follow-up decision PR before S0 starts. README's dependency table then makes S0 depend on that PR, and README's PL row, whose dependency is "every open point of `declarations-and-addressing.md`, all settled on 2026-10-18" (`plans/README.md:121`), is edited to leave out the points this proposal reopens.

**The schedule risk.**
No step is added, but a PR is, and it comes at the head of the never-cut chain PL → S0 → S1 → S2b → S3.
S0 cannot start until the decision PR has six answers and Appendix C's edits, so every day it waits after PL merges delays S0, and with it S1, S2b, S3, S4, and E4, by a day.
With the checkpoint on 2026-10-20 and the freeze at the end of 2026-10-25, the decision PR needs to merge by the checkpoint for the chain to keep its slack; if it has not, the checkpoint of 2026-10-22 should weigh starting S0 from the plan as accepted, with this proposal's changes after the workshop.

`plans/README.md` § Steps owns the order, the dependencies, and the cut; apart from S0's added dependency, none of them changes, and no step is added or removed.
README's rows for S0, S2a, S2b, S3, S4, G1b, G3, and G4, which describe step content, change as Appendix C.2 says.

| Step | Change from the plan | Cost |
| --- | --- | --- |
| S0 | Writes this design into `design/`: II.2, II.5, III.3, III.5, III.7, IV.2, IV.4, VI.11, and the passages of § 12.3; the glossary as Appendix C.3; `deferred.md` as Appendix C.4. Under § 14, Q1's recommended option 2, in today's words "slot" and "component" | About the plan's 40 lines of `design/`, plus the 29 lines of § 12.3; under Q1's option 1, also 66 lines with "slot" and 149 with "component" |
| S1 | None | None |
| S2a | The plan's renames, without rewriting `components` into a mapping; under Q1's option 1, the vocabulary rename too | Smaller than the plan's S2a under option 2; larger under option 1 (§ 14, Q1) |
| S2b | § 13.1 | The plan's content, minus the event rule, plus three items |
| S3 | `&` replaces `*` for laws and kernels and `merge` on `NamedTree`, `Record`, `RecordSpec`, and `RecordBatch`; `rename`; `MarginalView` | One more method, `RecordBatch.merge` |
| S4 | `element_spec` as an output spec, and a batch's outputs typed by their columns | One function more, `_exposed_components`; no test changes (Appendix E, fact 30) |
| G1a | The done-condition reads `[]`, not `project`; the grouped sub-joint stays one factor after its wrap | One line: `_flattened`'s test for a grouped sub-joint (`_whole_term_component(...) is None`, `distributions/_factored.py:300`) stops identifying it once G1a makes its event positional, so G1a changes that test to keep recognizing the grouped sub-joint |
| G1b | A function returning a law offers no outputs, so no view specs are computed | Smaller |
| G2 | The G2 families wrap their draws under the name their constructor takes (C5), then the keyword form refuses a spec with parts; the private event default goes | Smaller |
| G3 | Views at held fields become positional, with no wrap; `t.output(c)`; `[]` stops reading the name of the output at the root | Smaller: the plan's O5 and R3 lapse |
| G4 | Paths into a record input | Small |
| G5 | None | — |
| After the workshop | Under Q1's option 2, one announced PR renames the vocabulary in `design/`, the glossary, and the code together; `&` on functions and other batches; spec merges for laws, kernels, functions, and batches; Appendix C.4's items; the answers to § 14, Q4(b) and Q6 | As the plan, plus feeding for functions and the rename |

**Directives 7 and 8.**
Each renamed or removed name changes everywhere in its step, and no step adds an alias, a shim, or a special case.
The interim whole form, the whole form of views at held fields, the private event default, `[]`'s reading of the name of the output at the root, the joint's first-operand spec for a shared input, reading factors (§ 8.10), and stored scopes (§ 8.11) are today's behaviour kept until their steps.
`OutputSpec[n]` returning `O` until G3 is new code, but it mirrors `Distribution.__getitem__`'s present reading and leaves with it in G3, as the plan's `project` did (its § 7.5).

### 13.1 The size of S2b

S2b is on both critical chains (S0 → S1 → S2b → S3, and S2b → S4 and S2b → E4), so this proposal keeps it to the plan's content, minus one item and plus three:

| Item | Plan | This proposal | Files | Call sites and tests |
| --- | --- | --- | --- | --- |
| `OutputSpec.__getitem__`, `__contains__`, `without`, `__and__` | `project`, `without`, `&` | the same logic under `[]`, plus `in` | `core/_specs.py`, `distributions/_views.py` | `_view_declaration`'s 2 call sites (`distributions/_views.py:1057`, `:1063`) move to `O[s]`, apart from a group (G1a) and a held field (G3) |
| `InputSpec` sources, size shorthand, `&`, strict `without`, O9 | yes | the same; `&` keeps today's first-operand spec for a shared input unless § 14, Q4(d) chooses `U` | `core/_specs.py`, `values/_function_base.py`, `distributions/_conditional.py` | 3 readers (`_slots_of`, `_complete_given_spec`, `Function._initialize`); 3 callers of `InputSpec.without`; tests `tests/core/test_specs.py:366` and `:1151` change, as in the plan |
| `InputSpec` list selection | no | yes | `core/_specs.py` | none |
| The joint's event and inputs through `&` | yes | the same | `distributions/_factored.py` | `_factor_graph` (1 function) |
| The law's event rule | yes, in four constructors | not done | — | — |
| A law-valued or kernel-valued produced term offers no outputs | no | yes, replacing the event rule | `core/_specs.py` (`_exposed_components`, 3 call sites) | 3 tests change (Appendix E, fact 21) |
| Shallow record merge | no (S3, deep) | yes | `core/named_tree.py`, `core/record.py`, `core/_record_batch.py` | 3 library call sites; 2 tests and 1 message test change (Appendix E, fact 5) |
| `docs/api/declarations.md` | yes | the same, in the words § 14, Q1 settles | `docs/api/declarations.md` | — |

So S2b edits eight Python modules and the docs page, against the plan's six and the docs page, and changes 6 tests beyond the plan's.
With S2b's two new behaviours patched in, every User Guide and Get Started notebook that runs here passed, and neither behaviour was exercised (Appendix E, fact 45).
Everything else this proposal adds waits: the column typing of a batch's outputs goes to S4, the `condition_on` bug to `deferred.md`, and functions' merges stay after the workshop.

## 14. Open questions

Six questions need the maintainer, in order of importance; § 3 maps every reversal to one of them.

| Question | Options | Recommendation |
| --- | --- | --- |
| Q1. The words "input" and "output", and when they change | 1 all in S2a; 2 all after the workshop, in one PR; 3 keep "slot" and "component" | 2 |
| Q2. (i) The event rule and the view wrap; (ii) the outputs of a law-valued produced term | (i) drop or keep; (ii) none or marginals | (i) drop; (ii) none |
| Q3. (i) `OutputSpec` indexing; (ii) `InputSpec &`'s operands | (i) by paths with `outputs`, the plan's mapping with `project`, or a mapping with `project` renamed; (ii) two input specs, or any source | (i) by paths; (ii) two input specs |
| Q4. Merge rules: (a) shallow records; (b) optional inputs across merges; (c) functions feed; (d) the spec of a shared input | (a) shallow, mixed, or deep; (b) today's factors, IV.2 strictly, or visible optional inputs; (c) feed or refuse; (d) `U` after the workshop or in S2b | (a) shallow; (b) today's factors for the push, visible optional inputs after; (c) feed; (d) after |
| Q5. The output at the root | 1 `[]` paths only from G3, with `t.output(c)`; 2 `[]` reads the name permanently; 3 no accessor, `t[[c]]` | 1 |
| Q6. Dimension scopes of merge results, after the workshop | store a scope on every merge result, or substitute and forget everywhere | decide after the workshop; store, leaning |

### Q1. The words "input" and "output", and when they change

**Background.**
The plan uses "slot" and "component", which are not parallel; "slot" is a technical word, and "component" also names a mixture's component laws.
"Input" and "output" are plain and parallel, and already spelled in `InputSpec`, `OutputSpec`, `input_spec`, and `output_spec`.
Choosing them also renames three decided names: `t.component(c)` becomes `t.output(c)` (2026-10-17), O15's `element_declaration` becomes `element_spec` (whose old meaning becomes `element_spec.term_spec`), and II.5's `Batch.components` becomes `element_spec.outputs` (2026-10-18).
The design does not depend on the words; the question is when they change, since directive 7 requires a PR that changes a name or a meaning to change every use, and directive 11 requires the code to use the glossary's words.

What the rename covers, counted at `6f8859fc`:
1. **public names:** `components` to `outputs` (151 lines of `probpipe/` and 347 of `tests/` say `.components`, some a mixture's or an expression signature's, which stay); `component=` to `output_name=` (493 uses in 81 files of `probpipe/`, `tests/`, and `docs/`, not counting keywords such as `output_component=`), with `factor`'s `component_name` and `OutputSpec.default(component=)`; the private names of § 12.2, row 12;
2. **prose:** "component" on 1,653 lines of `probpipe/`, about 110 of them a mixture's component laws; "slot" on 806 lines, 41 of them `__slots__`; "output declaration", "input declaration", and "event declaration" on 184 lines; and in `tests/`, "component" on 1,403 lines and "slot" on 452. Each line is a judgement, since some use the words in another sense, and directive 5's rule 5 extends the rename to local names such as `for component in ...`;
3. **design text:** 66 lines with "slot" and 149 with "component" in `design/`;
4. **notebooks:** 5 User Guide notebooks read `.components` (14 uses) and 4 pass `component=` (15 uses); all six workshop tutorials read `.components`, which they do under the plan's S2a too, and none passes `component=`;
5. **the execution chain:** README makes E3b depend on S2a, so a rename in S2a delays E3b → S4 and E3b → E4, which are never cut.

**Options.**
1. **The full rename in S2a, before the workshop:** every use in the code, messages, docstrings, `design/`, and the glossary. Every later step writes the final words, and the workshop's code matches the glossary. S2a, on both chains, carries item 2's line-by-line judgement, days before the freeze.
2. **Defer the words everywhere to one announced PR right after the workshop**, `design/` and the glossary included. Before then, S0 writes this proposal's rules in today's words, with this mapping, so `dev/overhaul` is coherent at every merge:

   | This proposal | Before the rename |
   | --- | --- |
   | input, optional input | slot, optional slot |
   | output; the output at the root | component; the component at the root |
   | input spec, output spec | input declaration, output declaration (`InputSpec`, `OutputSpec`) |
   | `O.outputs` | `O.components`, with this proposal's meaning (§ 5.9) |
   | `t.output(c)` | `t.component(c)` (G3) |
   | `output_name=` | `component=` |
   | `element_spec` | `element_spec`, a final name that S4 introduces |
   | feeds, produced term, conforms, unify | the same words, which name no renamed thing |

   S2a is the plan's S2a without the mapping rewrite of `components`. The cost is one more breaking rename after the workshop, of `components`, `component=`, and the accessor, with a CHANGELOG entry.
3. **Keep "slot" and "component" permanently**, with the mapping of option 2 as the final words.

**Recommendation.** Option 2.
It satisfies directives 7 and 11 at every merge, keeps the critical path free of line-by-line judgement, and still adopts the plain words; option 1 is right only if PL and this proposal merge in time for S2a to absorb about 4,500 lines of judgement by the checkpoint of 2026-10-20.

### Q2. Law-valued produced terms: two decisions

**Background.**
A function can return a law, as `fit` does (§ 11.6), and a law can draw laws, as a random measure does.
The plan offers such a produced term's marginals as outputs, typed by its views (its § 4.4), and adds two rules, decided 2026-10-17: construction refuses a law whose event offers a law's parts (its § 5.4), and a view at a field that holds a law wraps under the field's name (its § 5.3).
M2 refuses, in a merge, every operand whose produced term is a law, whatever outputs it offers (§ 8.8), so neither the outputs rule nor the event rule protects merges.

**(i) The event rule and the view wrap.**
Options: drop both, or keep both.
Dropping them makes a law that draws laws legal, makes a view at a held law positional, so I2, I5, I6, and I9 hold at held fields with no exception from G3, lets the plan's O1, O5, and R3 lapse, and leaves one `OutputSpec.default` from G2.
Its cost: a law of laws can be named only by a list, as `rm[["G"]]`, or by the code that builds it wrapping its draws.
**Recommendation:** drop both.

**(ii) The outputs of a law-valued produced term.**
Options: none, or the marginals, typed by the views.
Offering none keeps `InputSpec(o)` from typing inputs as marginal laws and `o.outputs` from listing marginals that cannot be passed on separately (D1).
Its cost: `fit.output("mu")` and `fit.output_spec.outputs["mu"]` are gone, while `fit["mu"]` still reads the function returning the view; and three tests that assert those outputs change (Appendix E, fact 21).
**Recommendation:** none.

```python
# The plan                                       # This proposal
list(fit.output_spec)         # [mu, tau]         fit.output_spec.outputs      # {}
rm["G"].event_spec            # {G: law}          rm["G"].event_spec           # OutputSpec(law_spec): a law of laws
sample(rm["G"])               # Record {G: law}   sample(rm["G"])              # a law, distributed as sample(rm)["G"]
rm[["G"]].event_spec          # {G: law}          rm[["G"]].event_spec         # {G: law}
```

### Q3. The operations on the two specs: two decisions

**(i) How an output spec is indexed.**
The plan makes `OutputSpec` a `Mapping` of component names, so `decl[c]` reads a component's spec, and adds `project(s)` for the output spec at paths (its § 7.3 and § 7.5, and README's S2a, accepted 2026-10-18).
An output spec is then the one object whose `[]` does not take paths: `d[s].event_spec == d.event_spec.project(s)`, and `"g/c" in grouped.event_spec` is false while `"g/c" in grouped` is true (its § 11).
The maintainer dislikes the name `project`.
Options: (1) `O[s]` takes paths, as on every term, and an attribute maps names to specs (`outputs`, or `components` until the rename under Q1's option 2); (2) the plan's `Mapping` with `project`; (3) a `Mapping` with `project` renamed, such as `O.part(s)`.
**Recommendation:** option 1: `[]` keeps one meaning, I6 reads as written, `in` agrees with `[]`, and no new method name is needed.

```python
# Option 1                                    # Option 2 (the plan)
grouped.event_spec["g/c"]                     grouped.event_spec.project("g/c")
grouped.event_spec.outputs["g"]               grouped.event_spec["g"]
"g/c" in grouped.event_spec   # True          "g/c" in grouped.event_spec   # False
```

**(ii) The operands of `InputSpec &`.**
The plan lets either operand be any of the four sources (its § 6.4), so `o & i` merges an output spec, read as inputs, with an input spec.
Options: (1) both operands are `InputSpec`s, and `InputSpec(o) & i` states the conversion; (2) the plan's rule.
**Recommendation:** option 1: each side's `&` stays within its side, and `o & p` keeps one meaning, the merge of two output specs.

### Q4. Merge rules that change or state settled behaviour

**(a) Shallow `&` on records, and its timing.**
Today `merge` on records is deep: `{g: {c}}` and `{g: {e}}` merge into `{g: {c, e}}` (Appendix E, fact 5), while a merge of laws is shallow (fact 6).
Options: (1) shallow, delivered on `NamedTree.merge`, `Record.merge`, and `RecordBatch.merge` in S2b, so that S3 stays a rename; (2) deep for records and record specs, shallow for laws and output specs, so `&` means two things and `OutputSpec(R1) & OutputSpec(R2)` differs from `OutputSpec(R1 & R2)`; (3) deep for both, which changes what a factor graph is.
A deep merge is still written by hand: `r1.without("g") & r2.without("g") & Record({"g": r1["g"] & r2["g"]})`.
The same question covers batches of base values, which merge as records (§ 8.2), so two batches of arrays give a batch of records, where the plan's § 16 has them raise or combine bitwise (§ 3, row 13).
**Recommendation:** option 1, with batches of arrays merging as records. The library calls the deep merge only inside its own implementations (`core/record.py:948`, `core/_record_batch.py:792`), and 21 test lines call `merge`, of which 2 assert the deep rule.

**(b) Optional inputs across merges.**
`lik_opt & prior` is a law, so it has no inputs (III.7), yet its factor `lik_opt` still reads `s` at its default.
Options:
1. **Today's factors** (§ 8.10): a merge reads its operands' factors, and the unfed inputs of the factors, optional ones included, can be fed or shared by a later merge. Associativity holds, and III.7, III.9, IV.4, and `condition_on` stay as they are. Its costs are § 8.10's: a law changes when merged with an operand that shares its hidden input; equal laws merge differently; a view and a marginal of one joint merge differently; whether a result keeps the input depends on the operation's route; and a spec merge can be valid where the term merge raises.
2. **IV.2 read strictly:** an unfed optional input takes its default and leaves, so `(lik_opt & prior) & s_prior` feeds nothing. Merge then reads only the law, with no hidden state, but associativity fails for optional inputs: `lik_opt & (prior & s_prior)` feeds `s` while `(lik_opt & prior) & s_prior` does not, and today's code does the first in both groupings.
3. **Visible optional inputs:** a merge that leaves optional inputs is a term that shows them, as a kernel all of whose inputs are optional, with `sample` and `log_prob` at their defaults. There is no hidden state, and associativity holds. Its cost: kernel views (G5) and inference on such terms must be defined, and every model with an optional parameter changes kind, so `sample(lik_opt & prior)`, which works today, needs the kernel to sample at its defaults.

**Recommendation:** option 1 for the push, since it is today's behaviour and needs no code; option 3 as the long-term design, to decide after the workshop.

**(c) Functions feed.**
The plan's § 14.2 has functions "pair and never wire", with no reason given.
Options: (1) functions feed, as point-mass kernels, after the workshop (§ 8.7); (2) a fed name raises for functions.
**Recommendation:** option 1, by D2: the function rule is then the kernel rule, read deterministically.

**(d) The spec of a shared input.**
Today the joint keeps the first operand's spec for a shared input; validity is unaffected, since every consuming factor checks a bound value (§ 5.12).
Options: (1) keep today's choice in S2b's new `InputSpec &`, and switch to `U` after the workshop; (2) implement `U` in S2b, which changes one test (Appendix E, fact 35) and makes `x: float64` shared with `x: float32` give `float32` in either order.
**Recommendation:** option 1, to keep S2b to the plan's content; option 2 costs little if S2b has room, and avoids shipping a new operation whose result changes later.

### Q5. The output at the root

**Background.**
From step G3, the plan has `[]` take paths only, so `Normal("x", 0.0, 1.0)["x"]`, which returns the law today, raises, and an accessor reads the term at an output (here `t.output(c)`).
Under grammar B the name of an output at the root never collides with a path (jhuggins's `no_ambiguous_key_B`).

**Options.**
1. **`[]` takes paths only from G3; `t.output(c)` on terms reads the term at an output**, as this proposal; output specs get no accessor, since `O.outputs[c]` and `O[c]` cover them. Its cost: away from the root, `t.output(c)` and `t[c]` are two spellings of one term.
2. **`[]` takes paths and the name of the output at the root, permanently** (proposal A). It needs no accessor, G3 breaks nothing, `Normal("x", 0.0, 1.0)["x"]` keeps working, and `draws["x"]` returns the batch from S4, as II.5 states. Its cost: for a record `{n, G}` whose field `G` holds `Normal("x", 0.0, 1.0)`, `t["G"]["x"]` works while `t["G/x"]` raises, so I5 gains an exception; `in` mixes names with paths; and `[]` means two things.
3. **No accessor at all; generic code writes `t[[c]]`** (review r5). One reading serves every output. Its cost: `t[[c]]` gives the term whose produced term is the record `{c: ...}`, not the term at `c`, so code that wants the law `Normal("x", 0.0, 1.0)` back from its output `x` has no way to ask for it.

**Recommendation.** Option 1: `[]` keeps one meaning and I5 keeps no exception; the price is one plain method on terms and the G3 break the plan already schedules.

### Q6. Dimension scopes of merge results, after the workshop

**Background.**
Today a merged law or kernel stores its dimension scope, and the proposal keeps that; a merged input spec or function forgets its bindings (§ 8.11).
Storing is what makes laws and kernels associative: without it, `a(ya: ("n",) | u: ("n",))`, `b(u: (3,) | q: ("n",))`, and a law over `q: (4,)` raise in one grouping and merge in the other (review r4; Appendix E, fact 36).
Storing also lets a merge fail on a name no spec shows (§ 8.11, item 2), and `with_dim_names` cannot rename the stored name today.

**Options.**
1. **Store a scope on every merge result**, input specs and functions included, as private state that equality ignores, and let `with_dim_names` rename stored names too. Associativity then holds for every category with property 1's two cases, and a hidden name can be renamed away.
2. **Substitute and forget everywhere**, laws and kernels included. A merge result shows every constraint in its specs, but laws and kernels lose associativity on r4's example, which today's code merges consistently.

**Recommendation.** Decide after the workshop, leaning to option 1, with `with_dim_names` renaming stored names too; nothing before the workshop needs either.

## Appendix A. Proofs of the merge properties

In the proofs, `F_t` is `outputs(t)`, `G_t` is `names(unfed(t))`, and `t_s` is the spec at `s` in `unfed(t)`, or of the output `s` of `t`.
Each proof covers laws, kernels, records, and the specs; a batch merge is its element merge (§ 8.2), so each property holds element by element for batches of records or of base values, whose elements are records. For batches of laws or kernels, property 1 fails, since each element is read as one factor (Appendix B).

**1. Associative, when both groupings are valid.**
*Proof.*
Both groupings have the factors `factors(a), factors(b), factors(c)` in that order, and the outputs `F_a, F_b, F_c`.
Their unfed inputs are `(G_a − F_b − F_c) ∪ (G_b − F_c) ∪ G_c` in both, by R2.
`(a & b) & c` requires, by M4, `G_b ∩ F_a = ∅` and `G_c ∩ (F_a ∪ F_b) = ∅`; `a & (b & c)` requires `G_c ∩ F_b = ∅` and `((G_b − F_c) ∪ G_c) ∩ F_a = ∅`. These agree, because M3 makes `F_a ∩ F_c` empty, so `(G_b − F_c) ∩ F_a = G_b ∩ F_a`.
M3 is the same pairwise disjointness in both, and M5 holds for every merged operand by property 2.
Each fed input is fed by the same operand in both groupings, so M6 compares the same pairs, except at a name `s` that `a` and `b` both take and `c` outputs: there the left grouping checks that `U(a_s, b_s)` exists and that `c_s` conforms to it, and the right one checks that `c_s` conforms to `b_s` and then to `a_s`. By § 5.12's lemma these agree when `U(a_s, b_s)` exists; when it does not, the left grouping raises while the right one may not.
Every shared input is unified among the same operands, and `U` is symmetric and associative on the specs it accepts.
For laws and kernels, both groupings start from the union of all three stored scopes and add the same bindings, since each binding comes from the same pair of specs; a size bound twice raises in both. Only the order in which two free names come to one axis can differ.
For input specs and functions, the first merge of each grouping applies its `β` and keeps no binding, so a name bound in it and free in the operand merged later stays free in the result of one grouping and is bound in the other (§ 8.11, item 1); under the stated condition no such name exists, and the bindings agree.
Each factor reads the same fed values and, at an omitted optional input, its own default, so the value is the same.

*Checks.*
- For `a` with input `x: OpaqueSpec(type=object)`, `b` with `x: OpaqueSpec(type=str)`, and `c` with output `x: OpaqueSpec(type=str)`, `(a & b) & c` raises M7, since III.2's rule unifies two opaque types only when they are equal or one is `None`, while `a & (b & c)` is valid, since `str` conforms to both. Over the standard NumPy dtypes no such case exists: when neither of two dtypes casts to the other by the same-kind rule, no dtype casts to both (Appendix E, fact 38).
- For `a` with input `s: ("n",)`, `b` with `s: ("m",)`, and `c` with output `s: (3,)`, `(a & b) & c` raises M7, since two different free names meet at one axis, while `a & (b & c)` binds `m` and then `n` to 3 and is valid.
- For `a(ya | u: ("n",))`, `b2(u: ("k",) | q: ("k",))`, and `c2(q: (3,) | v: ("n",))`, `(a & b2) & c2` raises and `a & (b2 & c2)` gives `v: (3,)`, as today (fact 36).
- For `a` with input `x: float32`, `b` with input `x: int32`, and `c` with output `x: float64` (review r1), both groupings are invalid: `U = int32`, to which `float64` does not conform, and `b & c` fails for the same reason. Today's joint raises in both too, since each consuming factor checks the value (fact 39).
- For `a(ya: ("n",) | u: ("n",))`, `b(u: (3,) | q: ("n",))`, and a law `c` over `q: (4,)` (review r4), both groupings raise, since each scope binds `n` to 3 and to 4 (fact 36).

**2. Merged operands keep their sides disjoint.**
*Proof.* `(G_a − F_b) ∩ F_a` is empty by M5 on `a`, `(G_a − F_b) ∩ F_b` is empty by construction, `G_b ∩ F_a` is empty by M4, and `G_b ∩ F_b` is empty by M5 on `b`.

**3. Commutative up to order.**
*Proof.* M3, M5, and M7 are symmetric, and M4 holds both ways by the hypothesis `G_a ∩ F_b = G_b ∩ F_a = ∅`. `U` and the rule for optional flags are symmetric, and an `InputSpec`'s equality ignores order. With nothing fed, the density of § 8.5 is a product of two factors that share only the input record `u`.

**4. Paths are kept.**
*Proof.* § 8.6 builds the record of both operands' outputs, and a record operand contributes its children unchanged, so `spec_at` follows the same children below them; R1 then applies `β`.

**5. Sampling commutes with merge.**
*Proof.* With nothing fed, `z_b` and `z_a` are independent draws of `b` and `a`, their unfed optional inputs at their own defaults, and `z = z_a ⊎ z_b`, which is the record merge. The condition that `a & b` is valid matters: `sample(m1) & sample(m2)` is a valid record merge while `m1 & m2` raises over a stored scope (§ 8.11, item 2). For a law whose output is at the root, a draw is a base value with no name, and `&` on two base values is NumPy's (Appendix B).

**6. Renaming commutes with merge.**
*Proof.* § 8.3 and § 8.4 read only names, their equalities, and their intersections, which an injective rename of names to names preserves, and a rename changes no spec (jhuggins's `dmerge_rename`). A law's `rename` renames no unfed input of its factors, since IV.1 renames a factor's input only with the output that feeds it, while a kernel result's `rename` renames every input (III.9), so a rename of such a name would rename it on one side only; the condition excludes it.

**7. Merge reads factors.**
*Proof by example.* `lik_opt & prior` and `lik & prior` are equal laws, since `lik` is `lik_opt` at `s = 1.0`, but `(lik_opt & prior) & s_prior` feeds `s`, giving `y` a spread near 100, while `(lik & prior) & s_prior` feeds nothing, near 5 (fact 25).

## Appendix B. Deliberate exceptions

The plan's § 16 stays, with these changes: its row "Views at a field that holds a law or a function: I2, I5, and I9 hold only up to the wrap" is removed, since no view wraps (§ 7.2); its row "`&` on records and on arrays" adds that two records merge by top-level name; its clause on two batches of arrays is replaced by § 8.2; its row "`without` on the other kinds" keeps "a batch of laws"; and the rows from "Interim whole forms" on are new.

| Case | What happens | Why |
| --- | --- | --- |
| Covariance | `cov(d)` is one base value, so `cov(d)[s]` raises | It is not one value per field; the block for `s` is `cov(d[s])` |
| Monte Carlo summaries | `mean(d)[s]` and `mean(d[s])` estimate one quantity but are not equal | Each call takes its own draws |
| Summaries and conditioning at a field that holds a law or a function | `mean(rm)` is undefined, and `condition_on` cannot fix a drawn law | I1, I3, and I8 hold for paths that end at values |
| `without` on other kinds | `Function.without`, `ConditionalDistribution.without`, `without` on a batch of laws, and `Record.without` at a path through a held term do not exist yet | After the workshop (`deferred.md`) |
| `&` on base values | `sample(Normal("x", 0.0, 1.0)) & sample(Normal("y", 0.0, 1.0))` raises `TypeError`, as NumPy's `&` refuses floats, while two integer or boolean arrays combine bitwise; `(x & y)[j]` on two records is a field, while `x[j] & y[j]` on two arrays is a bitwise element | A base value carries no name, so I20 needs record outputs; the plan's O7 keeps NumPy's `&` |
| `&` on records and on arrays | `&` on two records merges their fields by top-level name, while `&` on two `NumericArray`s is elementwise | A record is a tree of named values (the plan's O7) |
| A batch of arrays through raw JAX | `jax.jit(lambda b: b)(draws)` comes back offering `element` | Whole-element names stay out of the pytree (decided 2026-10-17) |
| Labels | `sample(grouped["a", "g/c"])` and `sample(grouped)["a", "g/c"]` differ in label | No operation reads a label |
| Kernels until step G5 | `lik["y"]` raises `TypeError` | Kernel views come in G5 |
| Python's mapping rule | `"g" in draw` is true while `"g"` is not in `draw.keys()`; from G4, `"g/c" in lik_g.given_spec` is true while `"g/c"` is not in its `keys()` | `keys()` lists fields or inputs, and `in` agrees with `[]` |
| Interim whole forms, until G2 | an interim whole form has an output at the root and paths below it, so grammar B's guarantee fails; in `Normal("a", 0.0, 1.0) & F`, for `grouped`'s factor `F` with event `OutputSpec(g=...)`, `F.event_spec["g"]` is `F.event_spec` while the merge's `["g"]` is the group, so I17 and property 4 fail; the grouped sub-joint's form ends at G1a | Today's reading, kept until G2 (§ 5.5) |
| Views at held fields, until G3 | `d["G"].event_spec` is the interim whole form `OutputSpec(G=...)`, so I6 reads `d[p].event_spec == OutputSpec(**{last(p): d.event_spec[p].term_spec})` there | Today's behaviour, kept until G3 (§ 7.2) |
| A batch's output at the root, until G3 | `x_draws["x"]` raises while `x_draws.spec.element_spec["x"]` is that spec | § 7.3, rule 4, mirrors laws, and `[]` on a batch refuses the name today |
| The list form at the root | `Normal("x", 0.0, 1.0)[["x"]]` succeeds while `sample(Normal("x", 0.0, 1.0))` is an array that NumPy indexes, and `OutputSpec(x=...).without("x")` raises | § 7.3's rule 3 is an extra rule at the top level; I1 and I10 cover paths |
| A shared optional input that a binding omits | each factor takes its own default, so one input holds two values, and the law at the defaults is not the kernel at any single value of that input | Defaults belong to the factors (§ 8.4) |
| A kernel whose inputs are all optional | `k_opt & Normal("z", 0.0, 1.0)` is a law, sampled at `s = 1.0`, while `sample(k_opt)` raises | IV.2's table decides the kind by the required inputs left (§ 8.4) |
| Two views of one parent | `hier[["s1"]] & hier[["s2"]]` is the product of the marginals | A merge reads names and factors, and a view is one factor (§ 8.8) |
| A view and a marginal of one joint (I7) | `M[["y", "mu"]] & s_prior` and `marginal(M, ["y", "mu"]) & s_prior` differ when `M` holds a factor with an unfed optional input `s` | Merge reads factors, and a view is one factor (§ 8.10, item 4) |
| A hidden unfed optional input, fed or shared later | `(lik_opt & prior) & s_prior` feeds `s`, `m & k2` shares it, and `m & kw` makes it a bindable input of the kernel result though `kw` does not mention it; equal laws merge differently | Today's flattening (§ 8.10, items 1 to 3) |
| Spec merges and term merges (I16) | `s_prior.spec & (lik_opt & prior).spec` is valid while `s_prior & (lik_opt & prior)` raises | A spec states no unfed input of its factors (§ 8.10, item 5) |
| Associativity of validity (I19) | in property 1's two cases, one grouping raises and the other merges | M7 and II.1's rule check different pairs in the two groupings |
| Batches of laws or kernels (I19) | for `A`, `B`, and `C` batches of `lik_opt`, `prior`, and `s_prior` over equal levels, `(A & B) & C` leaves `s` at its default, so `y`'s spread is near 1, while `A & (B & C)` feeds `s`, near 100; both are valid | A batch merge reads each element as one factor, with the input spec the batch states, so the elements' factors and stored scopes are not read (§ 8.2, § 8.10, item 6) |
| Associativity of input specs and functions (I19) | `(I1 & I2) & I3` and `I1 & (I2 & I3)` are both valid and differ in `w`'s size | They store no dimension scope (§ 8.11, item 1) |
| A stored scope | `m1 & m2` raises over a dimension name that neither event spec shows, and `with_dim_names` cannot rename it away | Today's stored scope (§ 8.11, item 2); § 14, Q6 |

## Appendix C. Edits to the plan and to `design/`

### C.1 The plan's text that changes

The plan's other sections stand.
Under § 14, Q1's option 2, the plan keeps "slot" and "component" in its prose, and these edits change only the rules; under option 1 they also rename the words as § 12.2 lists.

| Section | Change |
| --- | --- |
| § 1 Scope | under Q1's option 1, "producers, consumers, slots, components, declarations" becomes "inputs, outputs, input specs, output specs" |
| § 2 | the row "merge" uses feeds |
| § 3 | I1 cites `d.event_spec[s]`; I6, I9, and I12 as § 9; the paragraph on the Rocq model drops the wrap; I14 to I22 join |
| § 4 | replaced by § 5 |
| § 5.2 | "whole declaration" becomes the interim whole form, with § 5.5's rules; under Q1's option 1, "the `component` accessor" becomes `output` |
| § 5.3 | a returned law offers no outputs, and its marginals are not offered or lifted; a view at a held field has no wrap from G3 |
| § 5.4 | removed |
| § 5.5 | one `OutputSpec.default`; the private event default keeps today's rule until G2 and then goes; the lift's wrap stays, with § 11.6's reason |
| § 5.6 | a bare law spec reads positionally and offers no outputs |
| § 6, § 7 | replaced by § 5.3, § 7, and § 8; O9 and § 6.2's `ValueError` move to § 5.3 unchanged |
| § 8 | line 1149: "`[]`, `marginal`, and `project`" becomes "`[]`, `marginal`, and `OutputSpec[]`"; line 1161: an output spec is indexed by path |
| § 9.3 | `decl["g"]` becomes `O.components["g"]` (`O.outputs["g"]` after the rename), and `decl.project(...)` becomes `O[...]` |
| § 9.5 | item 3: from G3 a view at a held field is positional, not wrapped |
| § 9.7, § 11, § 14 | replaced by § 7.7, § 7.6, and § 8 |
| § 12 | line 1421: `decl.project(...)` becomes `O[...]`; line 1425: "meets each component with the slot of the same name" becomes "feeds each slot from the component of the same name"; lines 1430–1431: "an unmet slot" becomes "a slot of a factor that no component feeds" |
| § 15.2 | `BatchSpec.element` becomes `element_spec`, an output spec whose `term_spec` is today's `element_spec`, and `Batch.components` is not added |
| § 16 | as Appendix B |
| § 17 | replaced by Appendix C.3 |
| § 18.1 | rows II.2 (indexing by path, the attribute of outputs, no event rule), II.5 (the elements' output spec), III.7 (no wrap), IV.2 (merge, flattening kept, conformance and unification, "unfed" for "unmet"), V.5 and V.6 (the lift's reason), VI.11 (two batches merge by elements), VII (the G2 families wrap their draws under their constructor's name), and § 12.3's passages |
| § 18.4 | the S2a row says `OutputSpec` is indexed by path; the S2b rows drop the event rule and add the shallow record merge and a law's empty outputs; the S4 row adds the column-typed outputs; the G1b row drops "typed by their views" |
| § 19 | S2a, S2b, S3, S4, G1a, G1b, G2, G3, and G4 as Appendix C.2; § 19.9's rows "The law's event rule (S2b)", "`OutputSpec` as a mapping (S2a)", "The operations (S2b)", and "The batch's element declaration (S4)" as Appendix C.2 |
| § 20 | alternatives 3, 10, 11, 15, and 17 are marked superseded by this proposal, with Appendix D's reasons |
| § 21 | O1 and O5 lapse; O14 is settled; O15 is replaced; R2 and R3 are replaced; O9 stands |

### C.2 Replacement text for the steps that change

Each block replaces the plan's text for that step in its § 19; the parts not named stay.
The blocks use the names of § 14, Q1's recommended option 2: `*` until S3, `components` for the attribute of outputs, `t.component(c)`, and `component=`, with `element_spec` final; under option 1 each takes § 12.2's name from S2a.

**S2a (§ 19.4).**
*Delivers:* `spec` becomes `term_spec`, `with_spec` becomes `with_term_spec`, `exposes_record` goes, and the reads of `_component_name` outside `core/_specs.py` move to one private helper. `components` keeps its name and is not rewritten into a mapping. Under Q1's option 1, also the full rename of § 12.2.
*Files:* `core/_specs.py`, `core/_fingerprint.py`, and every module that reads the renamed names: `.event_spec.spec` on 99 lines of `probpipe/` and 199 of `tests/`, `exposes_record` on 33 and 31, and `._component_name` on 19 lines of 6 modules; under option 1, also the counts of § 14, Q1.
*Done when:* a search finds no `._component_name` outside `core/_specs.py`, no `exposes_record`, and no `.spec` and no `with_spec` on an `OutputSpec`; under option 1, also no `.components` on an `OutputSpec` and no `component=` keyword, with a list by sense of every remaining occurrence of the old words; the full suite passes.

**S2b (§ 19.4).**
*Delivers:* § 13.1's items.
*Invariants:* I6 for the paths S1 supports, I10 on output and input specs, I1 for output specs.
*Files:* § 13.1's list.
*Done when:*
- `grouped.event_spec["g/c"] == OutputSpec(c=...)`, `grouped.event_spec[["g"]]` has the one output `g`, `"g/c" in grouped.event_spec`, and `grouped.event_spec.without("g/e")` keeps `{a, g: {c}}`;
- `d[s].event_spec == d.event_spec[s]` and `(a * b).event_spec == a.event_spec & b.event_spec` over the running examples, leaving out one path to a group and one path to a held field;
- `OutputSpec(post_spec).components == {}`, and `fit_wrapped.output_spec["posterior"] == OutputSpec(post_spec)`;
- `Normal("z", 0.0, 1.0) * random_function` still merges, storing the random function under its name;
- the `InputSpec` constructor's four sources and the size shorthand work, `InputSpec(a=NumericArraySpec(())).without("zzz")` raises `KeyError`, `likelihood.given_spec[["K", "r"]]` is an `InputSpec`, and an optional input whose parameter has no default raises `ValueError`, for kernels and functions (O9);
- `Record({"g": {"c": 1.0}}).merge(Record({"g": {"e": 2.0}}))` raises `ValueError`, and so do `RecordSpec.merge` and `RecordBatch.merge` for a shared top-level name;
- the full suite passes.

**S3 (§ 19.5).**
*Delivers and done when:* as the plan, with `RecordBatch.merge` and `NamedTree.merge` among the renamed methods: a search finds no `.merge(` on a `NamedTree`, a `Record`, a `RecordSpec`, or a `RecordBatch`, and no `*` between two laws or kernels in `probpipe/`, `tests/`, or `docs/`.

**S4 (§ 19.6).**
*Delivers:* `BatchSpec.element_spec` and `Batch.element_spec` become the output spec of one element (§ 5.7), and every use of the old meaning becomes `element_spec.term_spec`; draws, sweeps, and stacked batches of base values store the name their maker gave; a raw batch gets the default `element`; a batch's outputs are its columns, each typed by its column's `BatchSpec` (`_exposed_components` for batches); the name-free canonical spec of the plan's § 15.3.
*Files:* the plan's S4 files, and `core/_specs.py` for `_exposed_components`.
*Done when:*
- `sample(Normal("mu", 0.0, 1.0), sample_shape=4).spec.element_spec == OutputSpec(mu=...)`;
- `NumericArrayBatch(values, "draw").spec.element_spec` offers `element`;
- `grouped_draws["a"].spec.element_spec == OutputSpec(a=...)`, and `grouped_draws["g"]`'s is `OutputSpec(RecordSpec(c=..., e=...))`;
- `OutputSpec(grouped_draws.spec).components["a"]` is the `BatchSpec` of the column `a`;
- a `DistributionBatch`'s `event_spec == spec.element_spec.term_spec.event_spec`;
- the plan's equality, hashing, pytree, and fingerprint conditions hold.

**G1a (§ 19.7).**
*Done when:* `grouped["g"]["c"]` is the view of `g/c`, and `grouped["g"].event_spec == grouped.event_spec["g"] == marginal(grouped, "g").event_spec`, offering `c` and `e`; the grouped sub-joint factor's event is positional over its record; the rest as the plan.

**G1b (§ 19.7).**
*Delivers:* a `FunctionSpec`, and a `BatchSpec` of base elements, are accepted positionally; a function that returns a law declares a positional law output with no outputs, and one that returns a function offers the returned function's outputs; `OutputSpec.default` follows § 10.1, and `_complete_output_spec` reads a bare `output_spec` by it; `_lift_result` wraps a returned law, kernel, function, or batch under the callable's output name; the private event default keeps today's rule.
*Files:* `core/_specs.py` (`_exposed_kind`, `default`), `values/_function_base.py` (`_complete_output_spec`), `_output_atoms` and `_lift_result` in `functions/_broadcast.py`.
*Done when:*
- `Function(run_inference, output_spec=post_spec).output_spec == OutputSpec(post_spec)`, with no outputs, and a `fit_emulator` declaration offers `mean` and `var`;
- lifting `def law_of_y(x)` over a law gives draws `{law_of_y: law}`, and with `include_inputs=True` draws `{x, law_of_y: law}`;
- `OutputSpec.default(BatchSpec(NumericArraySpec(()), sample=3), component="f")` offers `f`;
- the plan's declaration probe finds nothing outside its G1b allowance.

**G2 (§ 19.8).**
*Delivers:* as the plan, with each G2 family wrapping its draws under the name its constructor takes, so its event is positional over `RecordSpec(<name>=...)`; then the keyword form refuses a spec with parts, and the private event default goes.
*Done when:* as the plan, plus `rm.event_spec == OutputSpec(RecordSpec(G=law_spec))` for a random measure `rm` constructed under the name `G`, and `sample(rm)` is the record `{G: law}`; `OutputSpec(G=law_spec)` raises `TypeError` with § 10.1's message.

**G3 (§ 19.8).**
*Delivers:* paths through every kind; `Function[]`; `t.component(c)`; `[]` and `OutputSpec[]` stop reading the name of the output at the root; a view at or below a field that holds a law, a kernel, a function, or a batch is positional, with no wrap.
*Done when:* property tests of I1, I5, I12, and I13; for a law `L` declared `OutputSpec(RecordSpec(G=law_spec))`, built in the test from a sampler, `L["G"].event_spec == OutputSpec(law_spec)`, `sample(L["G"])` is a law, and `L["G"]["mu"] == L["G/mu"]`; `Normal("x", 0.0, 1.0)["x"]` raises `KeyError` and `Normal("x", 0.0, 1.0).component("x")` is the law. The test law does not depend on G2's families.

**G4 (§ 19.8).**
*Delivers:* as the plan, plus paths into a record input in `InputSpec.__getitem__` and `InputSpec.without`.
*Done when:* as the plan, plus `lik_g.given_spec["g/c"]` is `NumericArraySpec(())` and `lik_g.given_spec.without("g/c")` leaves `g` holding `{e}`, required.

### C.3 The glossary changes of step S0

- **New entries, in any option:** base spec, spec with parts, node, group, column, produced term, feeds, merge, conforms, unify (II.1 and IV.2, as § 5.12 defines them), and *unfed*, said of an input of a joint's factors that no factor feeds (§ 8.1), replacing IV.2's "unmet".
- **New entries by § 14, Q1:** under option 1, input (replacing *given slot* and *optional slot*), input spec, output, and output spec, in S0; under option 2, *slot* is added and *component* redefined as the plan's § 17 proposes, with this proposal's meanings, and the rename PR after the workshop replaces them with *input* and *output*; under option 3, the same as option 2, permanently.
- **Redefined:** *component*, after a rename, keeps only a mixture's component laws (VII.3); *declaration* keeps only its broad sense, and its sentence on `InputSpec` and `OutputSpec` points to their entries; *conversion* and *event* say "event spec" for "event declaration"; *field* and *path* take § 5.2's meanings; *name* lists the names of inputs, outputs, fields, and levels; *type hole* names `with_term_spec`; *signature* says "a law's outputs, a kernel's outputs with `|` and its inputs, or a function's parameters", in the words Q1 settles.
- **Removed:** *packaging*; *key* in its sense of a field's path; *selection*, apart from *selection order* (II.7).
- **Not added**, although the plan proposed them: producer, consumer, exposes, whole declaration, exposed declaration, address, component spec, leaf path, wrap (which stays a plain verb), position, and list of paths (plain phrases, as decided 2026-10-18 for "subscript"). "Merge category" and "interim whole form" are words of this document's rules, not glossary terms.
- **Canonical names:** "a distribution's event spec, an `OutputSpec`: `event_spec`" replaces the row that says "event declaration"; the row for the name of a family's output at the root reads `component` until the rename and `output_name` after it; the row for `element_spec` says it is the output spec of one element, whose `term_spec` is the element's type.

### C.4 The items for `deferred.md`

1. The point-mass reading of merges across categories (§ 8.2).
2. Refusing a merge of two views of one parent, or returning their joint view (§ 8.8).
3. A merge that sorts the flattened factors topologically (Appendix D, item 18).
4. The joint's spec for a shared input becomes `U` (§ 5.12, § 8.6), unless § 14, Q4(d) chooses S2b; one test asserts today's choice (Appendix E, fact 35).
5. `condition_on` of a joint kernel ignores an optional input that a merge feeds (§ 8.5; fact 34), carried unchanged by S3's rename.
6. Two different dimension names at one axis become one dimension (Appendix D, item 14).
7. The long-term rule for optional inputs across merges (§ 14, Q4(b)), recommended: a merge that leaves optional inputs shows them.
8. The dimension scopes of merge results (§ 14, Q6).

## Appendix D. Alternatives considered

Each alternative was chosen by one of the three source designs (A, B, C), by the plan, or by an earlier draft of this proposal.

1. **Keep "slot" and "component"** (A). § 14, Q1, option 3 presents it.
2. **No selection on the input side** (A). Not taken: selecting inputs describes a spec, and declaring a kernel's inputs from another term's is a common idiom (§ 11.4).
3. **`[]` keeps reading the name of the output at the root permanently** (A). § 14, Q5 presents it.
4. **The noun "root name" and a public `root_name` property** (A). Not taken: "the output at the root" says it in plain words.
5. **Keep the attribute `components` permanently** (A, C). § 14, Q1 decides; under option 2 it stays until the rename.
6. **Keep the wrap of views at held laws** (B, and the plan). § 14, Q2(i) presents it.
7. **Define merges across categories now, through point masses** (B). Not taken: a point-mass factor has no density, and nothing before the workshop needs them.
8. **`o.names`, `i.names`, and `O.output(c)`** (B, and the first two drafts). Not taken: the keys of the attribute of outputs and of the input spec list the names, and `O.output(c)` duplicates `O[c]` and `O.outputs[c]`.
9. **`BatchSpec.output_spec` for the elements' output spec** (B). Not taken: on a `BatchSpec`, `output_spec` would read as the batch's own.
10. **Merge functions with type holes, checking the output specs at the first call** (C, and the first draft). Not taken: the merged output spec would be a record of pending children, which `RecordSpec` cannot hold (Appendix E, fact 28).
11. **`InputSpec &` takes any of the four sources** (C, and the plan). § 14, Q3(ii) presents it.
12. **Keep the decided name `t.component(c)`** (C). § 14, Q1 decides when it becomes `t.output(c)`.
13. **Unification that keeps the left operand's spec** (the first draft, and today's joint). Not taken as the rule: the reported spec would depend on the operands' order; today's joint keeps it until § 14, Q4(d)'s fix.
14. **Two different dimension names at one axis become one dimension** (the second draft). Not taken now: it changes II.1 and `_unify_specs`, which about twenty call sites use, and the merged term would have to store alias classes; it would remove property 1's second case. `deferred.md` records it.
15. **A law that states the unfed optional inputs of its factors as inputs, through `Distribution.given_spec`** (the second draft). Not taken: it contradicts III.7, III.9, IV.4, and today's `condition_on`. § 14, Q4(b), option 3 presents the variant that makes such a merge a kernel instead, without changing what a law is.
16. **A merge of laws drops its unfed optional inputs**, IV.2 read strictly. § 14, Q4(b), option 2 presents it.
17. **Refuse two views of one parent in a merge**, or return their joint view. Not taken now: a merge reads names and factors; `deferred.md` records it.
18. **Merge as a topological sort of the flattened factors**, so a kernel can be added between two merged operands (review r2, finding 12; fact 29). Not taken now: it changes IV.2's conditional-first rule; `deferred.md` records it.
19. **`OutputSpec` as a mapping of names, with `project`** (the plan). § 14, Q3(i) presents it.
20. **`O[k]` returns a term spec**, as `RecordSpec[k]` does (A's alternative). Not taken: I6 would need a conversion on every kind.
21. **One attribute pair, `input_spec` and `output_spec`, on every kind** (A's alternative; B's open question). Not taken: it renames `event_spec` on 885 lines and `given_spec` on 210 lines of `probpipe/`, and drops a distinction III.9 states.
22. **Deep merge on every kind** (C's option). § 14, Q4(a), option 3; not recommended, since a factor would then produce part of an output, which IV.1 forbids.
23. **A lifted law-returning function with a positional event** (the first draft). Not taken: the lift could not name its own law, and the plan's G1b done-condition was accepted.
24. **Stored dimension scopes on every merge result now** (review r6). Not taken before the workshop: nothing before then needs it; § 14, Q6.

## Appendix E. Facts checked against the code

Each fact was run at `6f8859fc` from the worktree root, with the scripts `verify_synth.py` to `verify_synth4.py`, `plugdir/no_law_plugin.py`, and `plugdir2/m2_positional_plugin.py` beside this file, and the reviewers' scripts in `review/`, all outside the repository.
Spreads drawn by sampling are rounded; each used 4,000 draws or more.

1. `InputSpec(RecordSpec(K=(), r=()))` raises `TypeError` (`core/_specs.py:120`).
2. `InputSpec(a=NumericArraySpec(())).without("zzz")` returns the input spec unchanged (`core/_specs.py:191`).
3. `InputSpec & InputSpec`, `Record & Record`, and `Function & Function` raise `TypeError`: none defines `&`.
4. `InputSpec(K=())` and `OutputSpec(K=())` raise `TypeError`: the size shorthand exists only for `RecordSpec` fields; `tests/core/test_specs.py:366` and `:1151` assert the refusal for inputs.
5. `Record({"g": {"c": 1.0}}).merge(Record({"g": {"e": 2.0}}))` is `{g: {c, e}}`, and `RecordSpec` merges the same way (`core/named_tree.py:835`, `core/record.py:931`, recursing at `:948`). Two tests assert this rule (`tests/core/test_named_collection.py:462`, `:474`). `RecordBatch.merge` requires equal levels (`core/_record_batch.py:783`), merges its element specs with `RecordSpec.merge` (`:792`), and its message "needs batches with distinct fields" is tested at `tests/core/test_record_batch.py:981`. `probpipe/` calls `merge` only inside these implementations, and 21 lines of `tests/` call it.
6. Two laws that both output a group `g` raise `ValueError` (`distributions/_factored.py:402`).
7. `Normal("mu", 0.0, 1.0) * lik` raises `ValueError`, "put 'Normal' to the right of 'lik'" (`distributions/_factored.py:431`).
8. For kernels with inputs `x: NumericArraySpec(())` and `x: NumericArraySpec((), dtype="float32")`, the joint keeps the first operand's spec; for `x: float64` and `x: float32`, `(k1 * k2).given_spec` has `float64` and `(k2 * k1).given_spec` has `float32` (`distributions/_factored.py:424`).
9. For `lik_opt`, `lik_opt * prior` is a `FactoredDistribution` with no `given_spec` (`distributions/_factored.py:460`).
10. A kernel whose input and output share the name `mu` raises `ValueError` (`distributions/_conditional.py:131`), while `Function(lambda x: x, input_spec={"x": ...}, output_spec=OutputSpec(x=None))` builds.
11. `OutputSpec(prior.spec).components` is `{mu: NumericArraySpec(()), tau: NumericArraySpec(())}`, the specs of a draw's fields (`core/_specs.py:302`), and `OutputSpec(BatchSpec(RecordSpec(a=()), sample=3)).components` is `{a: NumericArraySpec(())}`, the element's.
12. `OutputSpec(FunctionSpec(...))` raises `TypeError` (`core/_specs.py:355`).
13. `list(prior.event_spec)` and `"mu" in prior.event_spec` raise `TypeError`: `OutputSpec` is not iterable.
14. `Normal("x", 0.0, 1.0)["x"]` returns the law (`distributions/_distribution.py:1015`), and `Normal("x", 0.0, 1.0)[("x",)].event_spec` is `OutputSpec(NumericRecordSpec(x=...))`.
15. `grouped["g"].event_spec` is `OutputSpec(g=NumericRecordSpec(c=..., e=...))`, a whole term.
16. `hier["s1/theta", "s2/theta"]` raises `ValueError`: "more than one path ends in 'theta'".
17. `"g/c" in grouped` raises `TypeError`.
18. For `draws` of `Normal("x", 0.0, 1.0)`, `draws["x"]` raises `TypeError` (`core/_batch.py:785`).
19. `Record({"n": 3.0, "G": prior})["G/mu"]` raises `KeyError`.
20. `Function(lambda y: y).output_spec` and `.input_spec` are `None`.
21. **Probe: a produced term that is a law offers no outputs.** A pytest plugin made `_exposed_components` return nothing for a law or a kernel spec. Over `tests/core`, `tests/distributions`, `tests/functions`, and `tests/operations`: 8,720 passed, 63 skipped, 19 xfailed, and 3 failed. The three failures, `tests/core/test_specs.py:87`, `tests/operations/test_condition.py:567`, and `tests/operations/test_marginal.py:46`, each assert those outputs directly. The rest of the suite: 3,502 passed, 104 skipped, 10 xfailed, and none failed.
22. **Counts.** In `probpipe/`: `event_spec` on 885 lines, `given_spec` on 210, `.components` on 151 (347 in `tests/`), `exposes_record` on 33 (31 in `tests/`), `.event_spec.spec` on 99 (199 in `tests/`), `._component_name` on 19 lines of 6 modules outside `core/_specs.py`, "component" on 1,653 lines, "slot" on 806 (41 of them `__slots__`), and "output declaration", "input declaration", or "event declaration" on 184; in `tests/`, "component" on 1,403 lines and "slot" on 452. The keyword `component=` has 493 uses in 81 files of `probpipe/`, `tests/`, and `docs/`. In `design/`, "slot" is on 66 lines and "component" on 149. In the notebooks, `.components` is on 14 lines of 5 User Guide notebooks and 29 lines of all six tutorials, and `component=` on 15 lines of 4 User Guide notebooks and none of the tutorials. `_view_declaration` has 2 call sites (`distributions/_views.py:1057`, `:1063`), `_exposed_components` 3, and `InputSpec.without` 3 callers.
23. `grouped`'s second factor declares the interim whole form `OutputSpec(g=NumericRecordSpec(c=..., e=...))`.
24. A `Distribution` has no `given_spec`; `condition_on(lik_opt, {"mu": 0.0})` returns a `Normal` with no `given_spec`, `s` at its default (review `r5/probe_r5a.py`); and `condition_on(lik_opt * prior, {"s": 5.0})` raises `ResolutionError`, "unknown field 's'".
25. `(lik_opt * prior) * s_prior` and `lik_opt * (prior * s_prior)` both give `y` a spread near 100: flattening feeds the unfed optional input `s` of the factor `lik_opt` (review `probe_r2.py`).
26. For `la(ya | mu, s=1.0)` and `lb(yb | mu, s=50.0)`, `la * lb` has one optional input `s`, and `condition_on(la * lb, {"mu": 0.0})` gives spreads near 1 and near 50: each factor takes its own default (review `probe_r2b.py`).
27. **Probe: the revised M2.** A pytest plugin made `_factor_graph` refuse a factor whose event spec is positional over a law, a kernel, a function, or a batch, and let the keyword form pass. Over the whole suite: 12,227 passed, 167 skipped, 29 xfailed, and none failed. Today `whole_laws * Normal("x", 0.0, 1.0)` builds `{G: law, x}` (review `probes/probe_m2_sanity.py`); the reviewers' stricter probe, which also refused the keyword form, failed 3 tests that merge the G2 families (review `m2_probe_run.txt`).
28. `RecordSpec(center=None)` and `OutputSpec(RecordSpec(center=None))` raise `TypeError`: "None is not a field spec".
29. For `c(z | y)`, `b = Normal("x")`, and `a(y | x)`, with `m = c * b`, both `a * m` and `m * a` raise `ValueError`, while `c * a * b` builds `{z, y, x}` (review `probe_r2b.py`).
30. **Probe: a batch's outputs typed by their columns.** A pytest plugin made `_exposed_components` type each output of a batch by its column's `BatchSpec`; the whole suite gave 12,227 passed, 167 skipped, 29 xfailed, and none failed (review `m2plug/batch_columns_plugin.py`).
31. Today `Function(law_of_y)(Normal("x", 0.0, 1.0)).event_spec` is `OutputSpec(law_of_y=DistributionSpec(...))`, and with `include_inputs=True` it is `OutputSpec(RecordSpec(x=..., law_of_y=DistributionSpec(...)))` (review `probes/probe_lift.py`).
32. Today's `_unify_specs` does not compare supports, and raises when `("m",)` is checked against `("n",)`: "symbolic dimension 'm' where 'n' is expected" (`verify_synth3.py`).
33. A kernel whose every input is optional builds as a `ConditionalDistribution` (`_FunctionKernel`) with `given_spec` `InputSpec(s=...).with_optional("s")`; and `Normal("s") * (lik_opt * prior)` raises the order-rule error, so the order rule counts optional inputs today (review `r5/probe_r5a.py`).
34. `condition_on(lik_opt * s_prior, {"mu": 0.0})` gives `y` a spread near 1, though `s_prior` feeds `s` near 100; with `s` required, the same construction gives near 100, and `lik_opt * prior * s_prior` sampled directly gives near 100 (review `r5/probe_r5b.py`).
35. **Probe: a symmetric rule for a shared input's dtype.** With `_unify_either_way` patched to refuse two different dtypes or supports, the whole suite gave 12,226 passed, 167 skipped, 29 xfailed, and 1 failed: `tests/distributions/test_composition.py::TestSameNamedGivens::test_same_named_givens_unify_whichever_dtype_is_listed_first`, which asserts today's first-operand choice (review `r5/symunify_run.txt`).
36. Today's joint stores its scope: for `a(ya: ("n",) | u: ("n",))`, `b(u: (3,) | q: ("n",))`, and a law over `q: (4,)`, both groupings raise; for `a`, `b2(u: ("k",) | q: ("k",))`, and `c2(q: (3,) | v: ("n",))`, `(a * b2) * c2` raises and `a * (b2 * c2)` gives `v: (3,)`; for `a`, a law over `u: (3,)`, and `c3(w: ("n",) | v: ("n",))`, both groupings give `v: (3,)` (review `r4/probe_dims.py`).
37. `condition_on(lik_opt * prior, {"y": y_obs})` is an `EmpiricalDistribution`; `k_opt = conditional_distribution(lambda s=1.0: ..., ...)` is a `_FunctionKernel` whose `sample` without `given` raises `ApplicabilityError`; and `k_opt * Normal("z", 0.0, 1.0)` is a `FactoredDistribution` with no `given_spec`, which samples (review `r4/probe_optional.py`, `r6/probe_r6a.py`).
38. **Probe: dtypes under § 5.12.** Over the 14 standard NumPy dtypes from `bool` to `complex128`: the same-kind cast is transitive; whenever two dtypes each cast to the other by the same-kind rule, one casts to the other safely, so `U` is defined; `U` is associative on every triple; the lemma holds on every triple where `U` exists; and when neither of two dtypes casts to the other, no dtype casts to both (review `r6/probe_dtypes.py`, and `verify_synth4.py`).
39. For kernels `a` with input `x: NumericArraySpec(())`, `b` with `x: NumericArraySpec((), dtype="int32")`, and a law `c` over `x: float32`, `(a * b).given_spec` reports `a`'s spec, and both `(a * b) * c` and `a * (b * c)` raise: "the given 'x' has dtype float32, which cannot be cast to the declared int32" (review `r6/probe_firstop.py`).
40. For `m = lik_opt * Normal("mu", 0.0, 0.001)`, a `FactoredDistribution` with no `given_spec` whose `y` has a spread near 1, and `k2 = conditional_distribution(lambda s: Normal("w", 0.0, s), ...)`, `m * k2` is a `FactoredConditionalDistribution` with `given_spec` `InputSpec(s=...)`, and binding `s = 30` gives `y` and `w` spreads near 30 (review `r6/probe_capture.py`).
41. For `M = lik_opt * Normal("mu", 0.0, 0.001)`, with a linear-Gaussian, a nonlinear, and a `Gamma` neighbour in turn: `M[("y", "mu")]` is a `FieldView`, and `* s_prior` gives `y` a spread near 1; `marginal(M, ("y", "mu"))` and `M.with_path_names({"y": "w/y"})` are `FactoredDistribution`s, and `* s_prior` gives near 100; `condition_on(lik_opt * Normal("z"), {"mu": 0.0}) * s_prior` gives near 1. `condition_on(lik_opt * Normal("mu") * Normal("tau"), {"tau": 1.0})` keeps `lik_opt` as a factor, and `* s_prior` gives near 100 (review `r7/probe_open2.py`, `r6/probe_route.py`). A list of paths, `M[["y", "mu"]]`, raises `TypeError` today; S1 adds it.
42. For `m1 = a * b3`, with `a(ya: ("n",) | u: ("n",))` and a law `b3` over `u: (3,)`, `m1.event_spec` is `OutputSpec(NumericRecordSpec(ya=(3,), u=(3,)))`; for `m2 = k * lv`, with `k(w: ("n",) | v: ("n",))` and a law `lv` over `v: (4,)`, `m1 * m2` raises "the dimension 'n' is bound to 3 and to 4", and so does `m1.with_dim_names(n="m") * m2`; an `OutputSpec` has no `free_dims` (review `r6/probe_hidden_dims.py`).
43. Regrouping `lik_opt * prior` by `with_path_names({"y": "g/y", "mu": "g/mu"})` gives a law whose first factor is a grouped sub-joint; `* s_prior` keeps that factor whole, and `g/y` keeps a spread near 1. `(lik_opt * prior)["y"] * s_prior` gives `y` a spread near 1, while `(lik_opt * prior) * s_prior` gives near 100 (review `r6/probe_r6b.py`, `r6/probe_r6a.py`).
44. `&` on two `NumericArrayBatch`es of floats raises `TypeError`: "unsupported operand type(s) for &" (review `r7/probe_batch_and.py`).
45. **Probe: S2b's two behaviours on the notebooks.** With a law- or kernel-valued output offering no outputs and record merges shallow, `docs/get_started/quickstart.ipynb` and the User Guide notebooks 01 to 09 ran in an IPython shell: nine passed, and `06_conditioning_and_inference.ipynb` stopped at cell 2 for the missing optional backend `pymc` and, in a second run, at cell 29 for a limit of the shell; neither patched path fired in any notebook (review `r7/run_nb_s2b.py`, `nb_s2b_run.txt`, `nb_s2b_06.txt`).
46. For `m = lik_opt * Normal("mu", 0.0, 0.001)` and an unrelated `kw = conditional_distribution(lambda w: Normal("v", w, 1.0), ...)`, `m * kw` is a `FactoredConditionalDistribution` with `given_spec` `InputSpec(s=..., w=...).with_optional("s")`, and `condition_on(m * kw, {"w": 0.0, "s": 30.0})` gives `y` a spread near 30 (review `r8/probe_surface.py`).
