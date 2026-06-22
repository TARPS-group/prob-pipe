# Contract Discipline — Value-Model Refactor (issue #235)

**Audience:** every session (Claude or human) implementing a phase of the #235 value-model plan —
`EventTemplate`, the `Record` / `Distribution` value containers, the Batch types
(`*Array` → `*Batch`), `WorkflowFunction`, and naming/provenance. **Read this before you start, and
follow it for every phase.** It is meant to outlive any single PR or session — do not assume the
plan's author is available to restate these rules.

## Why this exists
The #235 plan defines the *contracts* — APIs, shapes, return types, error cases, invariants,
variable names — for ProbPipe's core value abstractions. Those contracts must be **explicit,
documented, and obeyed**: for correctness and consistency now, and because this contract
documentation is the **source material for the user guide** (a later PR). Write every docstring as
if it were user-guide reference text.

## Standing directives (apply to every phase)

1. **Contract-first — clarify before you implement.** Before writing code, make the contract of
   every major abstraction you touch *crystal clear* in your own understanding (see the table
   below for where each lives). If #235 (Chapter 1, the naming contract, the relevant chapters) or
   the existing docstrings leave **any** part ambiguous — shape conventions, single-vs-batched
   behavior, canonical orderings, error/raise cases, return types, invariants — **resolve the
   ambiguity before coding**: ask the maintainer, or pin it explicitly, and record the decision in
   the PR. Do not guess and do not let an unclear contract reach the code.

2. **Document the contract fully, where it lives.** When you implement an abstraction, document its
   contract *completely* in the **docstring** (NumPy style: Parameters / Returns / Raises). The
   docstring must state the *precise* contract, not a vague summary:
   - exact return type and **shape**, distinguishing single vs batched
     (e.g. `(vector_size,)` vs `(*batch_shape, vector_size)`);
   - canonical orderings (e.g. the leaf-traversal order for vectorization);
   - **every** error/raise condition;
   - invariants and round-trip guarantees (e.g. `from_vector(to_vector(v)) == v`).
   Update the index in this file when you add or change a cross-cutting contract.

3. **Analyze clarity; make the code obey the contract.** As part of every PR:
   - explicitly assess whether the contracts you touched are unambiguous, and call out any that
     are not;
   - verify the **code obeys the documented contract** — there must be no drift between docstring
     and behavior — and add tests that *assert the contract* (shapes, orderings, and error cases,
     not just happy-path values);
   - use **consistent variable names** for the same concept across the codebase (see the naming
     table). Renaming for consistency within the files you touch is in scope; flag larger
     inconsistencies you cannot fix within scope.

4. **Stay in scope, but never ship an undocumented or contradicted contract.** Work to your phase's
   PR brief. But if implementing your slice reveals a contract in this file or in #235 that is
   wrong, missing, or unclear, fix the documentation (or escalate) rather than silently coding
   around it.

## Major abstractions & where their contract lives
| Abstraction | Canonical contract location |
|---|---|
| `EventTemplate` / `NumericEventTemplate` / `ArraySpec`·`OpaqueSpec`·`DistributionSpec`·`FunctionSpec` | docstrings in `probpipe/core/record.py`; #235 Chapter 1 |
| `Record` / `NumericRecord` | docstrings in `probpipe/core/record.py`, `_numeric_record.py`; #235 Chapter 2 |
| Batch types (`RecordArray`/`NumericRecordArray`/`DistributionArray` → `*Batch`) | docstrings in `_record_array.py`, `_distribution_array.py`; #235 Chapter 2 |
| `WorkflowFunction` & ops (`sample`, `log_prob`, …) | docstrings in `core/node.py`, `core/ops.py`, `_workflow_result.py`; #235 Chapter 3 |
| Naming / provenance / annotations (`Tracked` / `Annotated` mixins) | the naming contract in #235 Chapter 5; mixin docstrings once they land |

## Canonical variable names (use these; don't invent synonyms)
| Concept | Name |
|---|---|
| a draw / value of type T | `value` |
| a 1-D numeric serialization | `vec` |
| values for leaves dropped by `numeric_subset`, supplied when reconstructing a full value | `non_numeric` |
| batch dimensions | `batch_shape` |
| independent-draw shape prefix for `sample` | `sample_shape` |
| a distribution's structural schema | `event_template` |
| PRNG key | `key` |
| a field name / key | `name` |
*(Extend this table whenever a new contract introduces a recurring parameter.)*

## Per-PR checklist (copy into the PR description)
- [ ] Contract of every touched abstraction was clear before coding (ambiguities resolved & noted).
- [ ] New/changed contracts fully documented in docstrings (types, shapes, orderings, raises, invariants).
- [ ] Code verified to obey the documented contract; tests assert it (shapes + orderings + error cases).
- [ ] Variable names consistent with the canonical table above.
- [ ] This `CONTRACTS.md` index updated if a cross-cutting contract changed.
- [ ] `ruff format` and `ruff check` are clean (both are **blocking** in CI).
