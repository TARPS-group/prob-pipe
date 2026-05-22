# BlackJAX MCMC migration — plan

Status: **draft for review.** No implementation yet.

This plan covers the next BlackJAX-integration phase under the parent
[Bayesian inference extensions plan][parent] and follows the plan-first
convention in [CONTRIBUTING.md § PR Workflow](./CONTRIBUTING.md).

[parent]: ~/.claude/plans/bayesian_inference_extensions_plan.md

## 1. Status

Already landed:

| Parent §9 phase | Done in |
|---|---|
| Phase 0 — plan | merged |
| Phase 1a — `Distribution.as_record_distribution(template=…)` | PR #187 |
| Phase 1b + 3 — `MinibatchedDistribution` + `blackjax_sgld` / `blackjax_sghmc` | PR #198 |
| Priority-rework (issue #189) | PR #190 |
| Distribution / Record hierarchy cleanup | PR #200 (in review) |

Open from the parent §9 table:

| Parent §9 phase | Status |
|---|---|
| 2a — register BlackJAX `nuts` / `hmc` at lower priority | this plan |
| 2b — flip MCMC default to BlackJAX | this plan |
| 4 — VI + Pathfinder + `PathfinderDistribution` | follow-up PR (§ 7) |
| 5 — BlackJAX SMC | deferred |
| 6a / 6b — native Laplace | deferred |

Parent §9 also lists Phase 2c (delete `_tfp_mcmc.py`). That step is
**not happening** under this plan — see § 2 below.

## 2. Scope

A **single PR** consolidating what the parent plan staged as Phase 2a + Phase 2b. The parent plan's multi-PR staging (lower-priority registration → benchmarks → priority flip → deprecation window → deletion) was written against the pre-#189 priority scheme and assumed a real downstream user base needing a migration window. Both assumptions have changed:

- ProbPipe is in alpha. There is no long-tail of pinned external users to support; a one-PR cutover is fine.
- Issue #189 introduced the `priority == 0` opt-in-only sentinel. Demoting `tfp_nuts` and `tfp_hmc` to `0` keeps them reachable via `method="tfp_nuts"` for bit-pattern regression checks, without the multi-release deprecation dance the parent plan described.

What this PR does **not** do — and what the parent §9 Phase 2c step
becomes obsolete for — is delete `_tfp_mcmc.py`. The opt-in `tfp_nuts`
and `tfp_hmc` methods stay registered at priority 0, so the file
remains the backing implementation for callers that pin `method="tfp_nuts"`
explicitly. The two-backend coexistence is the steady-state target, not
a deprecation window. The architectural follow-ups that the parent plan
tied to the deletion (e.g. consolidating sample-stats packing) are
re-scoped under § 8 below.

The PR also bundles a broader **inference-method priority audit**
triggered by the question "which methods should be on the
auto-dispatch path?" The audit's principle: any method whose `check()`
is identical to a higher-priority sibling is structurally unreachable
in auto-dispatch — it should be at `priority=0` (opt-in only). That
catches `blackjax_hmc` (same `check()` as `blackjax_nuts`) and
`blackjax_sghmc` (same `check()` as `blackjax_sgld`); `pymc_advi` is
also demoted on a different principle (VI is an explicit-tradeoff
choice). The four NUTS backends are re-anchored within tier 81–90 so
`blackjax_nuts` (the auto-dispatch winner for the canonical model
class) sits at 85, `nutpie_nuts` at 88 (top of tier), and
`cmdstan_nuts` / `pymc_nuts` both at 82 (disjoint model classes).
See § 4.3 for the full final-state table.

## 3. Design choices that have changed since the parent plan

### 3.1 Priority numbers

Parent §6.1.1 reads "register `blackjax_nuts` (priority 95) and `blackjax_hmc` (priority 85). Auto-selection still picks TFP." That was written against the old ad-hoc scheme (`tfp_nuts = 100`). Under #189's re-anchoring (`tfp_nuts = 75`, `tfp_hmc = 65`), registering BlackJAX at 95 / 85 would *immediately* outrank TFP, defeating the staging intent.

This plan replaces the staged-with-benchmarks ramp with a one-step swap:

| Method | Before | After | Tier (per #189 / `extending.md`) |
|---|---|---|---|
| `nutpie_nuts` | 85 | **88** | 81–90 (Rust-backed; fastest registered NUTS backend) |
| `blackjax_nuts` | — | **85** | 81–90 (optimised JAX-native; auto-dispatch winner for the canonical ProbPipe model class) |
| `cmdstan_nuts` | 82 | **82** | 81–90 (compiled Stan gradients) |
| `pymc_nuts` | 81 | **82** | 81–90 (PyTensor-backed; ties with cmdstan on disjoint model class) |
| `blackjax_hmc` | — | **0** | opt-in only — same `check()` as `blackjax_nuts`, structurally unreachable in auto-dispatch |
| `blackjax_sghmc` | 42 | **0** | opt-in only — same `check()` as `blackjax_sgld`; SGLD is the simpler default |
| `pymc_advi` | 25 | **0** | opt-in only — VI is an explicit-tradeoff choice |
| `tfp_nuts` | 75 | **0** | opt-in only — reachable via `method="tfp_nuts"` for bit-pattern regression |
| `tfp_hmc` | 65 | **0** | opt-in only — reachable via `method="tfp_hmc"` |
| `tfp_rwmh` | 55 | **55** | unchanged (gradient-free; migration deferred to a follow-up PR) |

`MethodRegistry.set_priorities` warns on crossings to / from `0`. The CHANGELOG entry covers the warning text and the migration recipe.

### 3.2 Builds on PR #200

PR #200 reshapes the inference-side surface in ways the BlackJAX path will rely on:

1. **`SimpleModel` runtime guard requires a `RecordDistribution` prior.** Every `SimpleModel`-rooted target therefore has a guaranteed `prior.record_template`. No defensive `getattr` needed inside the new BlackJAX module.
2. **`as_flat_distribution()` / `as_record_distribution(template=…)`** live on `NumericRecordDistribution` and are inherited by `Product`, `NumericJointEmpirical`, `SequentialJoint`. BlackJAX kernels consume flat parameter vectors; this round-trip is what turns a `Record`-shaped prior + likelihood into a `(flat_log_prob_fn, flat_init)` pair the kernel can run on.
3. **Distribution metaclass enforces `self._name` set in `__init__`** and **`RecordDistribution` metaclass enforces `record_template` set**. Any new Distribution class added in the follow-up Pathfinder PR (§ 7) needs to satisfy these; this PR adds no new Distribution classes.
4. **`RecordTemplate.event_shapes` / `field_event_shape(name)`** are public template-side accessors. The flat ↔ Record round-trip uses these uniformly.

This PR's first commit is therefore *strictly sequenced after* #200 lands.

### 3.3 `sample_stats` dict shape

BlackJAX's `info` is a `NamedTuple` (`acceptance_probability`, `step_size`, `num_integration_steps`, `is_divergent` for NUTS). The ArviZ converter currently consumes a TFP-shaped dict with specific keys. To keep the ArviZ path working without a converter rewrite, the BlackJAX path packs `info` into the **same dict shape** TFP produces.

A backend-neutral `SampleStats` dataclass (replacing both paths' ad-hoc dict construction) remains a sensible follow-up, but it's no longer tied to deleting `_tfp_mcmc.py` — both backends are permanent. Tracked under § 8.

### 3.4 `MinibatchedDistribution.record_template` access

[`probpipe/inference/_blackjax_sgmcmc.py:180`][sgmcmc-getattr] (post-#200) reads `record_template = getattr(prior, "record_template", None)`. SGMCMC requires a `SimpleModel` (which now requires a `RecordDistribution` prior), so the defensive `getattr` is redundant. Tightening to a direct attribute read is a three-line follow-up bundled into this PR's first commit (utility lift).

[sgmcmc-getattr]: https://github.com/TARPS-group/prob-pipe/blob/dev/distribution-record-hierarchy-cleanup/probpipe/inference/_blackjax_sgmcmc.py

## 4. PR scope and decomposition

Single PR, three commits for reviewability.

### 4.1 Commit 1 — utility lift + small post-#200 tightenings

`_build_target_log_prob` and `_get_init_state` move from `_tfp_mcmc.py` to a new `_inference_utils.py`. Pure refactor with one functional addition:

```python
# probpipe/inference/_inference_utils.py

def build_target_log_prob_flat(
    dist: SupportsLogProb,
    observed: Any,
) -> tuple[Callable[[Array], LogProb], Array, RecordTemplate]:
    """Return (flat_target_fn, flat_init, prior_record_template).

    The flat path: each call site gets a function from a flat
    parameter vector to log-density, plus a flat initial position
    derived from the prior, plus the record template needed to lift
    chains back into a structured posterior via
    ``as_record_distribution(template=...)``.
    """
    target_record = _build_target_log_prob(dist, observed)
    prior = _get_prior(dist)
    flat_prior = prior.as_flat_distribution()

    def target_flat(theta_flat):
        return target_record(flat_prior.unflatten_sample(theta_flat))

    init_record = _get_init_state(prior, None, observed)
    flat_init = flat_prior.flatten_value(init_record)
    return target_flat, flat_init, prior.record_template
```

Other changes in this commit:

- `_tfp_mcmc.py`'s `_build_target_log_prob` / `_get_init_state` / `_extract_record_template` / `_get_prior` import from `_inference_utils` rather than defining locally.
- `_blackjax_sgmcmc.py` line 180 tightens to `record_template = prior.record_template` (PR #200 makes the `RecordDistribution` prior an invariant for any model SGMCMC accepts).
- The defensive `getattr` in `_extract_record_template` itself stays — `_extract_record_template` is called from raw `SupportsLogProb` paths (not just SimpleModel), and those don't carry the guarantee.

No public API changes. Existing inference tests pass unchanged.

### 4.2 Commit 2 — register BlackJAX NUTS + HMC

New module `probpipe/inference/_blackjax_mcmc.py`:

```python
class _BlackJAXMCMCMethod(InferenceMethod):
    """Base class for BlackJAX gradient MCMC methods."""

    def __init__(
        self,
        algorithm: str,            # "nuts" | "hmc"
        method_name: str,          # "blackjax_nuts" | "blackjax_hmc"
        method_priority: int,
    ):
        ...

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    def check(self, dist, observed, **kwargs) -> MethodInfo:
        # Mirror TFPGradientMethod: require SupportsUnnormalizedLogProb +
        # JAX-traceable log-prob (probe via jax.make_jaxpr).
        ...

    def execute(self, dist, observed, **kwargs) -> ApproximateDistribution:
        target_flat, flat_init, record_template = build_target_log_prob_flat(
            dist, observed,
        )
        num_results = kwargs.get("num_results", 1000)
        num_warmup = kwargs.get("num_warmup", 500)
        num_chains = kwargs.get("num_chains", 1)
        step_size = kwargs.get("step_size", 0.1)
        inv_mass_matrix = kwargs.get(
            "inverse_mass_matrix", jnp.ones_like(flat_init),
        )
        seed = kwargs.get("random_seed", 0)
        chains, info_dict = _run_blackjax_chains(
            target_flat, flat_init,
            algorithm=self._algorithm,
            num_results=num_results, num_warmup=num_warmup,
            num_chains=num_chains, step_size=step_size,
            inverse_mass_matrix=inv_mass_matrix,
            random_seed=seed,
        )
        return make_posterior(
            chains, parents=(_get_prior(dist),),
            algorithm=self._method_name,
            auxiliary={"sample_stats": info_dict},
            record_template=record_template,
            num_results=num_results, num_warmup=num_warmup,
            num_chains=num_chains,
        )


def BlackJAXNutsMethod() -> _BlackJAXMCMCMethod:
    """BlackJAX No-U-Turn Sampler.

    Tier 71-80 (self-tuning gradient MCMC, broadly applicable).
    """
    return _BlackJAXMCMCMethod("nuts", "blackjax_nuts", 75)


def BlackJAXHmcMethod() -> _BlackJAXMCMCMethod:
    """BlackJAX Hamiltonian Monte Carlo.

    Tier 61-70 (well-understood gradient MCMC, requires hand-tuned
    step size).
    """
    return _BlackJAXMCMCMethod("hmc", "blackjax_hmc", 65)
```

`_run_blackjax_chains` is the kernel orchestration. It handles:

- Warmup via `blackjax.window_adaptation` (default 500 steps).
- Multi-chain sampling via `jax.vmap` over independent kernel states.
- Packing per-step `info` into a TFP-shaped dict for ArviZ.

Registration in `probpipe/inference/__init__.py`:

```python
from ._blackjax_mcmc import BlackJAXNutsMethod, BlackJAXHmcMethod

inference_method_registry.register(BlackJAXNutsMethod())
inference_method_registry.register(BlackJAXHmcMethod())
```

At this point, registry order is:

```
blackjax_nuts          priority=75     # NEW — auto-dispatch winner
tfp_nuts               priority=75
nutpie_nuts            priority=85
...
```

Two methods at `priority=75` — the registry's stable sort keeps `blackjax_nuts` after `tfp_nuts` (registration order). The next commit resolves this by demoting TFP to 0.

### 4.3 Commit 3 — demote TFP gradient methods to priority 0

`probpipe/inference/_tfp_mcmc.py`:

- `TFPNutsMethod` priority `75` → `0` (opt-in only).
- `TFPHmcMethod` priority `65` → `0` (opt-in only).
- `TFPRWMHMethod` unchanged (`55`) — gradient-free; no BlackJAX replacement in this PR.

The `Method.priority` change goes through the existing `_TFPGradientMethod(method_priority=...)` constructor argument; one-line change per factory.

After this commit and the priority audit bundled into the same PR:

```
nutpie_nuts            priority=88     # bumped from 85; Rust-backed, fastest backend
blackjax_nuts          priority=85     # NEW default for any SupportsLogProb + JAX-traceable
cmdstan_nuts           priority=82
pymc_nuts              priority=82     # bumped from 81; ties cmdstan on disjoint model class
tfp_rwmh               priority=55     # unchanged (gradient-free; BlackJAX migration deferred)
blackjax_sgld          priority=45
sbijax_smcabc          priority=5
blackjax_hmc           priority=0      # opt-in only (same check() as NUTS)
blackjax_sghmc         priority=0      # opt-in only (same check() as SGLD)
pymc_advi              priority=0      # opt-in only (VI is an explicit tradeoff)
tfp_nuts               priority=0      # opt-in only (bit-pattern regression)
tfp_hmc                priority=0      # opt-in only
```

Updates:

- `docs/api/inference.md` priority table.
- `docs/api/extending.md` example list (currently shows `tfp_nuts` as the canonical broadly-applicable adaptive method; switch the example to `blackjax_nuts`).
- `CHANGELOG.md` under `[Unreleased] / Changed (breaking)` — entry text in § 5.

The breaking-change entry names the migration recipe explicitly: `condition_on(model, data, method="tfp_nuts")` continues to work; callers relying on auto-dispatch picking TFP need either the explicit `method=` or `set_priorities(tfp_nuts=75)`.

## 5. CHANGELOG entry (draft)

```
### Changed (breaking)

- **`condition_on` MCMC default is now BlackJAX NUTS** (replaces TFP NUTS).
  The new method `blackjax_nuts` (priority 75) auto-dispatches for any
  ``SupportsLogProb`` + JAX-traceable target. `blackjax_hmc` (priority 65)
  joins as the no-adaptation gradient-MCMC fallback. `tfp_nuts` and
  `tfp_hmc` are demoted to the opt-in-only sentinel (`priority=0`) — they
  remain registered and reachable via `method="tfp_nuts"` / `method="tfp_hmc"`,
  but no longer participate in auto-dispatch.

  Migration: existing `condition_on(model, data)` calls that previously ran
  TFP NUTS now run BlackJAX NUTS. The numerical posterior is asymptotically
  identical but the per-seed bit pattern differs. Pin `method="tfp_nuts"`
  for bit-pattern regression. `tfp_rwmh` (gradient-free RWMH) is unchanged
  — no BlackJAX replacement in this drop.
```

## 6. Acceptance criteria

Two gates:

1. **Functional parity** — every existing test that previously ran a TFP NUTS / HMC path under auto-dispatch passes unchanged under BlackJAX. Notably:
   - `tests/test_inference.py` (end-to-end `condition_on` smoke tests).
   - `tests/test_inference_registry.py::TestInferenceMethodRegistry::test_auto_select_nuts` — auto-selection now picks `blackjax_nuts`; assertion updates accordingly.
   - `tests/test_inference_registry.py::TestBuiltInPriorityAnchors` — re-anchor table to include the new BlackJAX entries and the TFP demotions; the existing exact-above-inexact invariant continues to hold.
2. **Closed-form correctness on a 2-D Gaussian target** (per parent §6.1.2, slightly relaxed since we're not staging):
   - Posterior mean within `0.5 σ_MC` of the analytic mean.
   - Posterior covariance Frobenius distance within `5%` of the analytic covariance.
   - `arviz.ess` finite and > 100; `arviz.rhat` < 1.05.

No throughput / cold-start benchmark in this PR. The parent plan's §6.1.2 numeric throughput thresholds were load-bearing for the *staged* migration's priority flip — we're in alpha, the flip happens in one commit, and shared-CI hardware would make any wall-clock assertion flaky. If a real performance regression surfaces during use, it triggers its own follow-up.

## 7. Follow-up PR — VI + Pathfinder (parent Phase 4)

Strictly out of scope for this plan; sketching the contract so reviewers know what's coming next:

- `_blackjax_vi.py` with `BlackJAXMeanfieldVIMethod`, `BlackJAXFullrankVIMethod`, `BlackJAXPathfinderMethod`. Variational tier priorities (parent §11):
  - `blackjax_meanfield_vi` → priority 25 (tier 21–30, parametric posterior approximation).
  - `blackjax_fullrank_vi` → priority 28 (same tier, marginally higher because the full-rank family is strictly more expressive than mean-field).
  - `blackjax_pathfinder` → priority 30 (still tier 21–30 — Pathfinder is a parametric mixture, not a refinement-based method).
- New `PathfinderDistribution` class in `probpipe/distributions/_pathfinder.py` — a `NumericRecordDistribution` carrying `(locs, scale_trils, log_weights, record_template)` with analytic `_log_prob` / `mean_` / `variance_` / `_sample`. Must satisfy PR #200's metaclass invariants (sets `self._name` and `record_template` in `__init__`).
- Uses the same `build_target_log_prob_flat` utility introduced in this PR's commit 1, plus `as_record_distribution(template=…)` for the structured posterior wrap.

The Pathfinder PR is independent of any benchmark-gate decision on the MCMC migration — it only adds methods at non-overlapping priority slots.

## 8. Out of scope / deferred

- **`_tfp_mcmc.py` stays.** The opt-in `tfp_nuts` / `tfp_hmc` methods
  remain registered at priority 0 and `_tfp_mcmc.py` is their backing
  implementation. The parent §9 Phase 2c step (delete `_tfp_mcmc.py`)
  is obsolete under this plan: there's no migration window because the
  two backends coexist indefinitely.
- **Shared `_GradientMCMCMethod` base class for TFP + BlackJAX.** The
  two method classes now mirror each other's structure (`__init__`,
  `name`, `priority`, `supported_types`, `check`, `execute`) and will
  continue to. The original deferral rationale ("wait for the TFP path
  to go away") no longer applies. A small follow-up PR extracting the
  shared scaffolding is now justified on its own terms — *but only if*
  the per-backend warmup / kernel-construction details stay close
  enough to share. Defer until both paths have settled.
- **Backend-neutral `SampleStats` dataclass** replacing the ad-hoc
  TFP-shaped dict (parent §6.2). Same status: useful on its own,
  pairs naturally with the shared-base-class refactor above.
- **SMC (parent Phase 5)** and **Laplace (parent Phase 6a / 6b)** —
  separate plans, separate PRs.
- **Gradient-free MCMC family migration (RWMH + elliptical slice).**
  The current `tfp_rwmh` (a hand-rolled Python loop, not actually
  TFP-backed) is a natural candidate to retire in favour of
  `blackjax.normal_random_walk`, and `blackjax.elliptical_slice`
  would be a high-value addition for Gaussian-prior models. Split
  out into its own plan
  (`~/.claude/plans/bie-rwmh-blackjax-migration.md`) — the design
  (eager-fallback routing for non-traceable log-probs, DIY adaptive
  warmup) is large enough to deserve a dedicated PR. `tfp_rwmh`
  stays in place at priority 55 until that PR lands.

## 9. Risk and rollback

- **Bit-pattern regression** for users with a pinned seed under `condition_on(model, data)` auto-dispatch. Mitigated by: explicit migration recipe in the CHANGELOG, both methods still registered (only the dispatch order changed). Rollback is a one-line `set_priorities(tfp_nuts=75, tfp_hmc=65, blackjax_nuts=0, blackjax_hmc=0)`.
- **JAX-traceability surprises.** A `SupportsLogProb` model whose log-density compiles under TFP but not BlackJAX would silently fall through to `blackjax_hmc` (priority 65) or further. Mitigation: the `check()` probe in `_BlackJAXMCMCMethod.check` runs the same `jax.make_jaxpr` probe TFP uses; if either NUTS or HMC fails, both fail. There's no asymmetric failure mode.
- **Real-world performance regression.** Not benchmarked in-PR. If post-merge usage uncovers a meaningful slowdown vs. TFP NUTS, options are (a) tighten BlackJAX kernel parameters via the registered method's defaults; (b) one-line rollback above. The decision is reactive, not gating.

## 10. Decision asked of reviewer

Sign-off on:

- The **one-PR cutover** (no separate priority-flip PR), justified by alpha status.
- The **TFP gradient methods → priority 0** demotion (still registered, not deleted in this PR).
- `blackjax_nuts = 75`, `blackjax_hmc = 65`, `tfp_rwmh` unchanged.
- The **`sample_stats` dict shape** kept TFP-compatible (BlackJAX path packs `info` into the same dict keys).
- The **`_inference_utils.py` lift** (commit 1) as the dependency on PR #200 + a small tightening of `_blackjax_sgmcmc.py:180`.
- **Pathfinder + VI as a separate PR** after this one lands.

Once approved, the work is one PR with three commits per § 4.
