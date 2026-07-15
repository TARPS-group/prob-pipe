# ADR Format

Use this profile only after the ADR decision gate in `SKILL.md` passes. Follow the repository's existing ADR/RFC location, naming, and status conventions. Do not create an ADR directory or impose numbered filenames merely because no convention exists; obtain user approval for a new convention first.

Common conventions include `docs/adr/0001-slug.md`, `docs/architecture/decisions/`, RFCs, or repository-specific design records. Prefer the established one.

## Template

```md
---
status: proposed
---

# Represent probability kernels with ConditionalDistribution

ProbPipe represents a probability kernel as a `ConditionalDistribution` carrying separate given and event templates, rather than as a `Distribution` with missing inputs. This keeps unconditional laws distinct from families of laws and lets the ordinary operation layer expose conditioning, sampling, and log-probability without inventing backend-specific model methods.
```

That's it. An ADR can be a single paragraph. The value is in recording *that* a decision was made and *why* — not in filling out sections.

## Optional sections

Only include these when they add genuine value. Most ADRs will not need all of them.

- **Status** (`proposed | accepted | deprecated | superseded`) — required when the repository distinguishes tentative from accepted decisions; follow its syntax
- **Considered Options** — only when the rejected alternatives are worth remembering
- **Consequences** — only when non-obvious downstream effects need to be called out

## Numbering

If the repository uses sequential numbering, scan its established ADR location for the highest existing number and increment it. Otherwise follow the repository's naming convention.

## When to offer an ADR

All three of these must be true:

1. **Hard to reverse** — the cost of changing your mind later is meaningful
2. **Surprising without context** — a future reader will look at the code and wonder "why on earth did they do it this way?"
3. **The result of a real trade-off** — there were genuine alternatives and you picked one for specific reasons

If a decision is easy to reverse, skip it — you'll just reverse it. If it's not surprising, nobody will wonder why. If there was no real alternative, there's nothing to record beyond "we did the obvious thing."

Do not mark an ADR `accepted` merely because the user accepted a branch during an exploratory session. Keep major changes `proposed` when they still require team, meeting, or maintainer approval.

### What qualifies

- **Architectural shape.** "ConditionalDistribution is a sibling of Distribution, not a subclass." "Batch axes are represented separately from one event's structure."
- **Integration patterns between layers.** "Estimator adapters return ordinary Function or ConditionalDistribution terms instead of exposing estimator-shaped prediction methods."
- **Technology choices that carry lock-in.** "The differentiable numeric boundary is JAX-native, while non-JAX execution is isolated behind dispatch." Record only choices whose replacement would reshape public contracts or a substantial part of the implementation.
- **Boundary and scope decisions.** "EventTemplate owns event schema; concrete distributions must not keep a second authoritative copy." The explicit no-s are as valuable as the yes-s.
- **Deliberate deviations from the obvious path.** "A RandomMeasure is preserved by default and marginalized only through an explicit operation." Record decisions a future contributor might otherwise simplify back into an incorrect automatic behavior.
- **Constraints not visible in one code path.** "The core package cannot import estimator or inference subpackages because the dependency graph must remain acyclic." "Related stochastic views must co-sample without using a global draw cache."
- **Rejected alternatives when the rejection is non-obvious.** If the team considered treating a probability kernel as a partially initialized Distribution and rejected it to preserve mathematical meaning and capability checks, record that rationale so the same conflation is not proposed again without new evidence.
