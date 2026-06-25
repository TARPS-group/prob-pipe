# Changelog

All notable changes to ProbPipe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`probpipe.validation` posterior-vs-reference comparison metrics.** A
  dependency-light scoring layer for validating inference methods against a
  trusted reference: `Reference` (a container for analytic / long-NUTS /
  sandwich references — high-precision `(mean, cov)`, `draws`, and/or a target
  `score_fn`), `standardized_mean_error` (Mahalanobis mean error
  `‖Σ_ref^{-1/2}(μ̂ − μ_ref)‖₂`), `relative_cov_error` (operator-norm whitened
  covariance error `‖I − Σ_ref^{-1/2} Σ̂ Σ_ref^{-1/2}‖₂`), `std_ratios`,
  `sliced_wasserstein`, `mmd` (unbiased
  RBF), `ksd` (IMQ kernel Stein discrepancy), and the `score_posterior`
  aggregator.

- **`quantile` op and `SupportsQuantile` protocol.** `quantile(dist, q)` returns
  per-field quantile(s) at probability level(s) `q`, parallel to
  `mean`/`variance`/`cov`. `RecordEmpiricalDistribution` implements it
  weight-aware via the midpoint-CDF (Hazen, type-5) quantile, used for both
  uniform and non-uniform weights so the estimator is continuous in the
  weights.

- **`ProvenanceMode` enum and `provenance_config` singleton for lineage-tracking
  control.** Three modes are available: `FULL` retains live references to every
  parent distribution (good for interactive debugging); `LIGHTWEIGHT` (the new
  default) stores only `ParentInfo` descriptors — type name, distribution name,
  and the parent's own provenance chain — so parent data arrays are free to be
  garbage-collected once a workflow step completes; `OFF` skips provenance
  entirely for minimum overhead.  The mode is set once at startup:
  ```python
  import probpipe
  from probpipe import ProvenanceMode
  probpipe.provenance_config.mode = ProvenanceMode.FULL  # for debugging
  ```

- **`ParentInfo` descriptor** (new public export).  A frozen dataclass carrying
  `type_name`, `name`, `source` (the parent's own `Provenance`, kept in all
  non-OFF modes so the ancestry DAG remains traversable), `fingerprint`
  (reserved for a future caching layer), and `obj` (the live parent object,
  set only in FULL mode).

- **`Provenance.create()` factory classmethod.**  Centralises mode-checking:
  reads `provenance_config.mode`, wraps each parent in a `ParentInfo`, and
  returns `None` in OFF mode.  All ~15 provenance assembly sites in the
  codebase now route through this single entry point, so mode behavior is
  uniform everywhere.

### Changed

- **`EventTemplate` moved to its own module and `Record` now carries an
  authoritative `EventTemplate` (breaking changes to the value-model surface).**
  `EventTemplate` / `NumericEventTemplate` and the leaf specs now live in
  `probpipe.core.event_template` (the public `probpipe.*` exports are unchanged).
  A `Record` stores its `EventTemplate` rather than re-deriving it on access.
  Several methods moved or were removed:
  - **Removed** `EventTemplate.pack`, `numeric_fields`, `non_numeric_fields`,
    `event_shapes`, and `field_event_shape`; **removed** `Record.flatten` /
    `Record.unflatten` (use `jax.tree_util.tree_flatten` for the JAX-pytree path).
  - **Renamed** `EventTemplate.from_record` → `EventTemplate.infer_from`
    (best-effort, lossy inference). The value upcast is consolidated to
    `Record.to_numeric()`.
  - **Moved** `to_vector` / `from_vector` and `leaf_shapes` onto
    `NumericEventTemplate`; `from_vector`'s `non_numeric` argument is dropped.
    `numeric_leaf_shapes` is consolidated into `leaf_shapes`.
  - **Added** `EventTemplate.to_leaf_list` / `from_leaf_list` and
    `Record.to_leaf_list` — the general, template-granularity leaf
    (de)composition (each leaf kept whole, in canonical `leaf_paths` order).

- **User Guide notebooks moved from the former examples section.** The docs nav
  and grouped overview now list all 11 User Guide notebooks under
  `/user_guide/.../`, including the Prefect scalability guide.

- **Adopt `ruff format` for code formatting.** Formatting is now owned by
  `ruff format` (Black-style) rather than the previous manual horizontal-packing
  conventions: the source tree was reformatted in one mechanical sweep (recorded
  in `.git-blame-ignore-revs`), a `ruff-format` pre-commit hook reformats on
  commit, and `ruff format --check` is a **blocking** CI step. Notebooks are
  excluded so the docs' tutorial cells keep their hand layout; string quotes
  normalize to double. See
  [CONTRIBUTING.md § Code formatting](CONTRIBUTING.md#code-formatting).

- **`provenance_ancestors()` now returns `ParentInfo` descriptors, not live
  Distribution objects (breaking change).**  Under the previous always-live
  model, every element of the returned list was a `Distribution` or `Record`
  that could be sampled, inspected, etc.  Under the new LIGHTWEIGHT default,
  elements are `ParentInfo` instances:
  ```python
  # Before
  ancestor = provenance_ancestors(result)[0]   # Distribution
  ancestor.sample(key, (10,))                  # worked

  # After (LIGHTWEIGHT default)
  ancestor = provenance_ancestors(result)[0]   # ParentInfo
  ancestor.name                                # "prior"
  ancestor.obj                                 # None — parent may be GC'd

  # To restore live-object access, opt in to FULL mode
  probpipe.provenance_config.mode = ProvenanceMode.FULL
  ancestor = provenance_ancestors(result)[0]
  ancestor.obj                                 # live Distribution
  ancestor.obj.sample(key, (10,))              # works
  ```
  Code that checks `x in provenance_ancestors(result)` or accesses
  `.samples` / `.log_prob` on ancestors needs to be updated — either
  switch to FULL mode, or use `ancestor.name` / `ancestor.type_name` for
  identity checks.
- **Two-distribution packaging: `probpipe-core` (minimal) and `probpipe`
  (core + all backends) (#237).** The root distribution is renamed `probpipe-core` (minimal JAX base —
  every inference backend is an optional extra), and a new code-less `probpipe`
  metapackage (`packaging/probpipe/`) pins `probpipe-core` and bundles the
  backends the docs exercise — PyMC, nutpie, and BayesFlow (marker-guarded
  `python_version < "3.14"`) — so `pip install probpipe` runs every example and
  tutorial (on Python 3.12–3.13; 3.14 omits BayesFlow until upstream lifts its
  cap). The `probpipe` **import** name is unchanged in both. Extras not
  already bundled (`prefect`, `viz`, `stan`) are re-exported on the metapackage,
  so `pip install "probpipe[<extra>]"` works alongside
  `pip install "probpipe-core[<extra>]"`. The package `authors` metadata is set
  to the ProbPipe Development Team, with the full contributor list in `AUTHORS`.
  Existing from-source installs should reinstall to pick up the renamed
  distribution. (The `pyabc` SMC-ABC backend joins `probpipe` once it lands; CI
  to build and publish both distributions follows in a separate PR.)

- **Nested `ProductDistribution` support in the record layer (#262).**
  `RecordArray` accepts slash-delimited paths in string indexing
  (`arr["outer/a"]`) and integer-indexes a nested array into a nested record
  element; `flatten` / `unflatten` recurse into nested record fields in
  depth-first leaf order; and a batched draw from a nested `ProductDistribution`
  is a canonical, flattenable nested record array.

- **Citation metadata and a "Cite" / "Help" docs section.** A
  `CITATION.cff` enables GitHub's "Cite this repository" button; the
  README gains a "Citing ProbPipe" section with a BibTeX entry; and the
  docs site adds a [Cite](https://tarps-group.github.io/prob-pipe/cite/)
  page (software citation + how to cite the inference backends) and a
  [Help](https://tarps-group.github.io/prob-pipe/help/) page (where to
  ask questions / file issues). The Zenodo DOI is minted from the first
  tagged release and is dropped into the BibTeX/CFF once available.

- **"Open in Colab" badges on the tutorials.** Both tutorial notebooks
  (Getting Started, Flexible Inference) gain an "Open in Colab" badge and a
  guarded setup cell that, *only when run on Colab*, installs ProbPipe with
  the extras the notebook uses (and, for Getting Started, fetches the
  dataset). The cell is a no-op in local Jupyter, the docs build, and CI, so
  notebook execution elsewhere is unaffected.

- **Keyword form for the `log_prob`-family ops (#228).** `log_prob`, `prob`,
  `unnormalized_log_prob`, `unnormalized_prob`, and the `random_*_log_prob`
  ops accept named field arguments —
  `log_prob(model, intercept=0.0, slope=0.5, X=X_obs, y=y_obs)` — built into a
  single draw via `Distribution._pack_value` (single-field → the bare field
  value; multi-field → a `Record`), whose general field validation and
  `Record` building is the new public `RecordTemplate.pack`. The ops stay plain
  `WorkflowFunction`s that resolve this in their body — the same shape as
  `condition_on`'s named data kwargs — so the positional form (including
  `value=`) is unchanged and still broadcasts, and per-call controls use
  `with_options` (`log_prob.with_options(seed=0)(dist, value)`). The keyword
  form is purely additive; the one case it cannot express is a distribution
  whose field name collides with the op's own `value`/`dist` parameter — for a
  multi-field distribution pass a positional `Record`, for a single-field one
  the bare positional value (mirroring `condition_on`'s `observed`).

- **`StanModel` participates in the `log_prob` keyword form with one field per
  Stan parameter (#228).** `StanModel` and its unconstrained view gain a
  `record_template` exposing one field per Stan parameter *block* — e.g.
  `theta` for `vector[3] theta`, `L` for `matrix[2, 2] L` — with each block's
  full multidimensional shape reconstructed from BridgeStan's flattened
  parameter names. The keyword form assembles the flat parameter vector
  BridgeStan consumes — `log_prob(model, mu=0.0, theta=theta_vec, L=chol)` —
  placing each scalar by its parsed index so matrices pack in BridgeStan's
  column-major order; a flat array may still be passed positionally.

### Changed

- **`RecordTemplate` → `EventTemplate` rename + leaf-spec representation
  (#235, Phase 1a).** `RecordTemplate` is now `EventTemplate`,
  `NumericRecordTemplate` is `NumericEventTemplate`, and
  `Distribution.record_template` is `event_template` (hard rename, **no
  deprecation alias** — pre-stable). Template leaves are now a closed sum of
  frozen, hashable specs (`ArraySpec` / `OpaqueSpec` / `DistributionSpec` /
  `FunctionSpec`) instead of `tuple[int, ...] | None`; construction-time sugar
  is preserved (`EventTemplate(x=(3,), label=None, sub=…)` still works) and
  `__getitem__` now returns the spec object (shape access stays on
  `leaf_shapes` / `event_shapes` / `field_event_shape`). Behavior-preserving
  otherwise.

- **`SupportsLogProb` / `SupportsUnnormalizedLogProb` are now generic in the
  sample type (#228)** — `SupportsLogProb[T]`. Annotation-level only; runtime
  behavior is unchanged.

- **`StanModel.fields` now returns one name per Stan parameter block (#228)**
  rather than one per scalar (`vector[3] theta` is the single field `theta`,
  not `theta.1` / `theta.2` / `theta.3`). BridgeStan's flat, per-scalar names
  remain available unchanged via `parameter_names`.

- **Amortized SBI learners accept nested priors (#262).**
  `learn_amortized_posterior` / `learn_amortized_likelihood` /
  `learn_amortized_ratio` now train on nested `ProductDistribution` priors,
  iterating the prior's numeric leaves; for NPE, per-leaf bijectors run at each
  leaf's native event shape, and posterior draws come back under their nested
  leaf names. Previously a nested prior was rejected up front.

- **Install docs: a "New to Python?" two-route fork.** The README and docs
  landing now split installation into a newcomer path (uv manages Python and
  the environment — no prior Python needed) and an experienced-user pip path,
  and note that ProbPipe installs from source (not yet on PyPI). The optional-
  extras list also gains the previously-missing `bayesflow` extra.

### Fixed

- **Linear-algebra and Gaussian-conditioning edge cases on the algebra bug-fix
  branch.** `RootLinOp.diag()` now squares diagonal roots; `CholeskyLinOp`
  keeps lower-root (`L @ L.T`) and upper-root (`U.T @ U`) representations
  consistent across `cholesky`, `to_cholesky_representation`, `matvec`,
  `rmatvec`, `matmat`, `rmatmat`, `diag`, `to_dense`, and `solve`;
  `JointGaussian.condition_on` uses linear solves instead of forming explicit
  covariance inverses; and `SumLinOp.matmat` / `rmatmat` preserve the `(n, 1)`
  matrix shape for single-column inputs.

- **Invalid log-space weights are rejected before normalization.**
  `Weights(log_weights=...)` now rejects `NaN` entries and zero-total-mass
  inputs such as all `-inf`, avoiding downstream `nan` normalized weights while
  still allowing individual `-inf` entries for zero-weight atoms.

- **`StanModel` now works against a real BridgeStan backend.** Two bugs at the
  BridgeStan boundary were hidden by the mocked tests: construction passed a
  `data=` keyword that `bridgestan.StanModel.from_stan_file` does not accept,
  and JAX arrays were handed to a ctypes interface that requires `float64`
  NumPy arrays. Construction now goes through BridgeStan's supported
  constructor — which takes the `.stan` path directly and serializes the data
  dict — and every value crossing into `param_constrain` / `param_unconstrain`
  / `log_density` is coerced to a `float64` ndarray, so `StanModel(stan_file)`
  and `log_prob(stan_model, ...)` succeed end to end. The `stan` extra now pins
  `bridgestan>=2.7` (the first release with that constructor), and a
  compile-gated integration test guards this boundary against future drift.

- **nutpie sampling of a `StanModel` keeps its construction-time data.** The
  nutpie path rebuilt the BridgeStan model from the conditioning data alone,
  dropping any data passed to `StanModel(file, data=...)` — so a model carrying
  fixed data (sizes, covariates) failed on the missing variables when sampled
  via nutpie, while the CmdStan path worked. The conditioning data is now merged
  on top of the construction-time data (conditioning values override), matching
  the CmdStan method.

- **`condition_on` no longer silently ignores a case-mismatched data kwarg
  (#228).** Passing `condition_on(model, x=...)` when the field is `X` used to
  route `x` to the inference parameters, where it was silently dropped (e.g. by
  NUTS) — a wrong result with no error. A kwarg that matches a field only up to
  case now raises a `TypeError` with the correct casing (`did you mean X=...?`);
  unknown kwargs that are *not* a case-variant of any field remain inference
  parameters.

- **Codecov no longer misreports coverage on targeted PRs (#261).**
  On a PR that ran only the changed-files test path, the main test job
  skipped its Codecov upload while the BayesFlow job still uploaded, so
  Codecov computed project/patch from the BayesFlow report alone —
  yielding spuriously low numbers and a "HEAD has 1 upload less than
  BASE" warning even though every Actions job passed. Now: the main
  test job uploads coverage on the targeted path too (tagged `unit`),
  so **patch** coverage is accurate and stays an enforced PR gate;
  Codecov **project** is `informational` on PRs (the real 88% floor is
  enforced in-CI on the full-suite run via `--cov-fail-under`); the
  BayesFlow leg is gated to run only on BayesFlow-relevant changes; and
  per-flag `carryforward` keeps the project number sane when a flag
  isn't uploaded.

- **Package license metadata corrected to Apache-2.0 (was MIT).**
  `pyproject.toml` declared `license = { text = "MIT" }` while the
  repository's `LICENSE` is Apache License 2.0 — and the metadata field is
  what PyPI displays. The field is now a PEP 639 SPDX expression
  (`license = "Apache-2.0"` with `license-files = ["LICENSE", "AUTHORS"]`),
  so built distributions carry `License-Expression: Apache-2.0` (core
  metadata 2.4). The setuptools build floor rises from 61 to 77.0.3 — PEP
  639 support landed in 77.0.0, which also deprecated the old
  `license = { text = ... }` table form, and 77.0.3 relaxed the new
  `license-files` validation from errors to warnings — and the redundant
  `wheel` build requirement is dropped (`bdist_wheel` ships inside
  setuptools since 70.1). Build-time changes only; runtime dependencies
  are unchanged.

### Added

- **Contributor conventions for comments, naming, tests, and PR hygiene.**
  CONTRIBUTING.md gains "Code comments & docstrings" (no process narration,
  no negative documentation, public docstrings describe behavior) and "Test
  quality" (tightest reliable tolerances, structured cases, dispatch-path
  equivalence) sections, a description-equals-final-state PR rule, and a
  docs-ship-with-the-change rule. STYLE_GUIDE.md gains §1.12 "Naming
  accuracy" (semantic accuracy, ecosystem alignment, symmetry, complete
  rename sweeps). The `review-pr` skill now checks all of these and reads
  the convention docs from the PR's base ref.

- **BayesFlow amortized-SBI backend (`[bayesflow]` extra).** New
  `learn_amortized_posterior(prior, simulator, method="npe"|"fmpe"|"cmpe",
  ...)` trains a jax-native (keras-on-JAX) amortized neural posterior
  estimator — NPE (coupling flow), FMPE (flow matching), or CMPE
  (consistency model) — and returns a `BayesFlowModel` bundling the joint
  model (prior + simulator, exposed as properties) with the trained
  estimator: `condition_on(model, observed)` draws from `p(theta | observed)`
  in a single network forward pass (no MCMC). This restores the amortized
  half of the SBI layer dropped with sbijax.
  - Training simulates `(theta, y)` offline (`prior` drawn via the `sample`
    op, `simulator.generate_data` for the data); the prior is used only to
    draw `theta` and needs no TFP translation. The trained estimator is
    amortized — the same instance conditions on any observation with no
    retraining — and its draws are named via the prior's `record_template`.
    The simulator receives the prior's native structured per-draw sample (named
    fields), matching the `GenerativeLikelihood` contract, and keras training is
    seeded for reproducibility.
  - Continuous priors with constrained supports — including matrix- and
    simplex-valued ones (positive, an interval, Dirichlet's simplex, Wishart's
    positive-definite matrices, …) — are handled by per-field `bijector_for`
    reparameterization applied at each field's native event shape: training runs
    in the unconstrained space and draws are mapped back to the support (identity
    for real-valued fields). NPE's coupling-flow minimum is counted in
    unconstrained dimensions. Discrete priors have no smooth bijector and are
    rejected with a clear error.
  - Training seeds keras for reproducibility but snapshots and restores the
    caller's global NumPy / Python RNG state, so a call does not perturb
    unrelated random streams.
  - The `[bayesflow]` extra is **Python 3.12–3.13 only** (BayesFlow 2.x caps
    `<3.14`); keras runs on the JAX backend (`KERAS_BACKEND=jax`) — no
    TensorFlow or PyTorch. The backend is imported lazily, so `import
    probpipe` does not load keras.

- **jax-native NLE and NRE (`[bayesflow]` extra).** New
  `learn_amortized_likelihood(prior, simulator, ...)` (neural likelihood
  estimation: a conditional coupling flow for `p(y | theta)`) and
  `learn_amortized_ratio(...)` (neural ratio estimation: an NRE-C classifier
  for the likelihood-to-evidence ratio) return `BayesFlowLikelihood` /
  `BayesFlowRatio` — `ConditionallyIndependentLikelihood` components whose
  `log_likelihood` is **jax.grad-transparent**, so
  `SimpleModel(prior, learned)` + `condition_on` samples the posterior with
  the existing BlackJAX/TFP NUTS machinery. No PyTorch: this replaces the
  planned sbi-torch default path (verified by the Step-6a spike — gradients
  finite-difference-exact and NUTS recovering analytic posteriors, including
  discrete-observation + constrained-parameter cases for NRE).
  - Per-row scores sum under conditional independence, so datasets of any
    size work natively (NPE's conditioning shape is fixed at training time),
    and `per_datum_log_likelihood` comes for free.
  - The networks take raw constrained `theta` as *input* (no bijector
    reparameterization needed on that side); discrete-valued parameter
    fields are accepted. NLE's default coupling flow needs observations with
    >= 2 dimensions and a reverse-differentiable density (adaptive-ODE
    networks such as `FlowMatching` integrate `log_prob` with a dynamic-bound
    `while_loop`, which JAX cannot reverse-differentiate); NRE's MLP
    classifier has neither restriction and handles discrete observations.
  - `learn_amortized_likelihood(dequantize=True)` supports integer-valued
    observations via uniform dequantization (Theis et al. 2016; Ho et al.
    2019, Flow++): training adds `U[0,1)` jitter to the simulated `y` and the
    wrapper scores integer data at the unit-cell midpoint `y + 1/2`. Without
    it, the continuous fit measurably overdisperses the posterior as
    observations concentrate on few atoms.
  - `BayesFlowRatio` values are log-ratios — valid for conditioning (the
    evidence constant cancels) but not for absolute-likelihood uses (model
    comparison, LOO/WAIC); the caveat is documented on the class.

- **`ProductDistribution.supports`** — per-field support constraints (each
  component's `support`), implementing the canonical `RecordDistribution`
  accessor that previously raised `NotImplementedError`.

- **Python 3.14 to the CI test matrix.** The matrix is now
  `[3.12, 3.13, 3.14]`. `requires-python = ">=3.12"` is unchanged.
- **Coverage floor enforced at 88%** on the full-suite CI run
  (`--cov-fail-under=88`). The changed-files-only PR path and local
  single-file runs are exempt (`--cov-fail-under=0`), since a global floor
  is only meaningful when the whole suite executes. Current measured
  coverage on `main` is ~91%; the floor is set conservatively within the
  beta plan's ≥85–90% commitment to leave headroom for normal fluctuation.
- **Concurrency cancellation on CI for PR pushes.** A new push to a PR
  branch cancels the prior in-progress CI run. Pushes to `main` are
  unaffected (no cancellation — the merge-history gate stays solid).
  Same pattern added to the docs build (PR builds cancel; pages deploys
  still serialize via the original `pages` group).
- **PR auto-labeling.** `.github/workflows/labeler.yml` +
  `.github/labeler.yml` apply `area:*` labels to PRs based on changed
  file paths. `kind:*` and `status:*` labels are still applied by
  humans.
- **Dependabot for GitHub Actions.** `.github/dependabot.yml` opens
  weekly PRs that bump pinned action versions (`actions/checkout`,
  `astral-sh/setup-uv`, `codecov/codecov-action`, `actions/labeler`).
  Auto-labeled `area:infrastructure`. Pip/uv dependency bumps are NOT
  enabled — the JAX/TFP resolver interaction means lockfile updates
  must be intentional.

### Changed

- **Pyright type checking (advisory).** A `typecheck (advisory)` CI job
  runs [pyright](https://microsoft.github.io/pyright/) over the `probpipe`
  package and reports type issues; it is **advisory for now** (does not
  gate merges) while the type-debt baseline — largely JAX/TFP
  untyped-attribute noise — is burned down. Config lives in
  `pyrightconfig.json` (`basic` mode, `reportMissingTypeStubs` off). Run
  locally with `uv run --with 'pyright[nodejs]' pyright`. To enforce
  later: drop the job's `continue-on-error` and tighten
  `typeCheckingMode`. See [CONTRIBUTING.md](CONTRIBUTING.md#type-checking).

- **Dev tooling: migrated to [uv](https://docs.astral.sh/uv/) for
  environment + dependency management.** `uv.lock` is committed and used
  by CI (`uv sync --frozen`), making the install reproducible and lifting
  the duplicated inline jax/jaxlib/tfp-nightly pins out of the workflow
  files. Local dev: `uv sync --extra dev --extra nutpie [--extra pymc]`,
  then `uv run pytest`. The pip path (`pip install -e ".[dev]"`) still
  works for contributors with an existing pip setup. See
  [CONTRIBUTING.md](CONTRIBUTING.md#installation).

- **Ruff linting + pre-commit hooks.** The `lint & format` CI job runs
  `ruff check` over the whole tree as a **blocking** gate. A
  `.pre-commit-config.yaml` (install with `uvx pre-commit install`) runs
  ruff (lint + format) plus file-hygiene hooks on staged files. The lint
  config (`[tool.ruff.lint]`) selects the `E`/`W`/`F`/`I`/`UP`/`B`/`SIM`/`RUF`
  families, ignores the ambiguous-unicode rules (`RUF001/2/3` — false
  positives on mathematical notation), and excludes notebooks (executed in
  CI instead). See [CONTRIBUTING.md](CONTRIBUTING.md#linting--pre-commit).

- **`pymc_nuts` reclaims multi-core sampling.** The method previously
  forced `cores=1` to avoid an `os.fork()` deadlock against JAX's worker
  threads. It now samples one worker per chain (capped at the CPU count,
  overridable via a `cores=` kwarg) using the **`spawn`** multiprocessing
  start method — clean worker processes with no inherited threads, so it
  is deadlock-free on every platform (POSIX `fork`, the deadlock-prone
  default on Linux, is never used). Empirically `cores=2` spawn is no
  slower than the old single-core path; a new test exercises the
  multi-core path after spinning up JAX's threads to reproduce the
  hazard.

- **Ecosystem cutover to arviz 1.x and pymc 6 (breaking).** The core
  `arviz` pin moves `>=0.13,<1.0` → `>=1.1,<2.0`, **dropping arviz 0.x
  entirely**, and the `[pymc]` extra moves `pymc>=5.28` → `pymc>=6`
  (pymc 5.x hard-caps `arviz<1.0`; pymc 6.0 is the first
  arviz-1.x-compatible release). ProbPipe now binds the arviz 1.x split
  packages **by name** — `arviz_base.from_dict` (`build_mcmc_datatree`),
  `arviz_base.from_cmdstanpy` (the CmdStan method), and `arviz_stats.*`
  — never bare `import arviz`; the runtime 0.x/1.x version probes are
  removed and the auxiliary is an arviz 1.x `xarray.DataTree`
  throughout (`ApproximateDistribution.warmup_samples` reads
  `aux.children`). `[pymc]` additionally requires `matplotlib` (the
  pymc 6 sampler progress bar imports it). Internal pymc-6 fix:
  `pm.sample_prior_predictive(samples=)` → `draws=` (the `samples=`
  kwarg was dropped in pymc 6). The
  ArviZ 1.0 defaults — credible interval 0.94 → 0.89 and HDI → ETI —
  are adopted as-is: no affected statistic is called in ProbPipe
  source, so no `rcParams` pin is needed, and a frozen-fixture suite
  (`tests/inference/test_arviz_regression.py`) locks the 0.89/ETI
  defaults plus golden `arviz_stats` values as a tripwire against
  future silent default drift.

- **Core jax / jaxlib floor raised to `>=0.9`; blackjax to `>=1.4`.**
  With sbijax gone (see *Removed*), the `<0.9` jax / jaxlib cap that
  existed solely to keep sbijax's hard `jax==0.8.1` pin satisfiable is
  lifted, and the floor moves up to `>=0.9` — the verified stack
  resolves to jax 0.10.1. `blackjax` moves `>=1.3` → `>=1.4` (held at
  `1.3` only to spare sbijax's jax 0.8.1 environment). Users pinned to
  jax 0.8.x must upgrade to `>=0.9`. Isolated as its own PR for
  bisectability given the broad RNG / numerics blast radius across
  every JAX backend. The arviz `<1.0` ceiling and the `pymc>=6` bump
  are intentionally *not* changed here — each lands in its own isolated
  PR (the arviz-1.x ceiling lift and the pymc 6 upgrade).

### Removed

- **sbijax dropped (breaking).** The `sbijax`-backed simulation-based
  inference (SBI) layer is removed in full, ahead of the PyMC 6 /
  ArviZ 1.0 ecosystem upgrade — `sbijax` constrains the jax / jaxlib
  floor and blocks the rest of the stack from moving forward. No
  replacement ships in this release; the SBI capability is being
  re-platformed onto **pyabc** (SMC-ABC), **BayesFlow** (amortized
  NPE / FMPE / CMPE), and **sbi** (NLE / NRE) in subsequent releases.
  Removed surface:
  - The **`[sbi]` extra** (`pip install probpipe[sbi]`) and its
    `sbijax>=0.3.6` dependency.
  - The public workflow functions **`sbi_learn_conditional`** and
    **`sbi_learn_likelihood`** (exported from both `probpipe` and
    `probpipe.inference`), the **`DirectSamplerSBIModel`** they
    returned (exported from `probpipe.inference`), their `method=`
    selectors (`npe` / `fmpe` / `cmpe` for the direct sampler,
    `nle` / `nre` for the emulated-likelihood path), and the
    `network_factory=` hook. `from probpipe import
    sbi_learn_conditional` now raises `ImportError` rather than
    returning an install-prompt stub.
  - The **`sbijax_smcabc`** inference method (`SbiSMCABCMethod`,
    priority 5) and its registration; `condition_on(generative_model,
    data, method="sbijax_smcabc", ...)` no longer resolves.
  - The internal `probpipe/inference/_sbijax.py` module, the `sbi`
    pytest marker, the `tests/inference/test_sbijax.py` suite, and the
    CI `--no-deps sbijax` install shims. The contract invariants those
    tests covered — posterior recovery, amortization, and SMC-ABC
    dispatch — are re-homed per backend as the replacements land,
    rather than in this removal.

  The jax / jaxlib `<0.9` and arviz `<1.0` version caps that `sbijax`
  forced are *retained* here and lifted in their own isolated PRs (the
  jax-0.10 floor bump and the arviz-1.x ceiling lift); this PR changes
  no runtime version pins. The `docs/tutorials/flexible_inference.ipynb`
  tutorial's SBI sections are flagged out of date until a replacement
  backend ships — its `condition_on` dispatch and NUTS material remain
  accurate.

### Added

- **BlackJAX-backed gradient-free MCMC.** Two new inference methods
  bundled with the BlackJAX MCMC migration:
  - **`blackjax_rwmh`** (priority 55) replaces the hand-rolled
    Python-loop RWMH. Two execution paths share the same BlackJAX
    kernel: a fast path (`jax.lax.scan` + `jax.vmap` across chains)
    when the target log-density is JAX-traceable, and an eager
    Python-loop fallback when it isn't (BridgeStan / scipy /
    external-simulator likelihoods — the case the hand-rolled loop
    existed to support). The default warmup is a Stan-style window
    adaptation: ``n_windows`` (default 4) geometrically-growing
    windows, each sampling with the current proposal Cholesky and
    accumulating Welford statistics on positions, refreshing the
    proposal at window boundaries. Production sigma is
    ``chol(Sigma_hat) * 2.38 / sqrt(d)`` per Roberts-Gelman-Gilks.
    Short warmups (``< 50`` steps) collapse to a single phase
    automatically. ``adapt=False`` falls back to the legacy
    ``step_size * I`` for parity with the prior behavior.
  - **`blackjax_elliptical_slice`** (priority 75, tier 71-80
    self-tuning) is new — restricted to `SimpleModel` targets with a
    Gaussian prior and a JAX-traceable likelihood. Recognises
    `Normal`, `MultivariateNormal`, `JointGaussian` (named multi-field
    Gaussian with cross-covariance), and `ProductDistribution`
    compositions via the new `_gaussian_prior_params` helper.
- New workflow function `probpipe.elliptical_slice(model, data, ...)`.

### Changed

- **`blackjax_hmc` randomizes its trajectory length.** Production now
  draws the number of leapfrog steps from a low-discrepancy Halton
  sequence (`blackjax.dynamic_hmc`) with mean `num_integration_steps`
  (default 10, unchanged), instead of a fixed count. A fixed trajectory
  length can resonate on near-Gaussian targets — the proposal returns
  near its start, giving high acceptance and zero divergences yet poor
  mixing and up to ~30% posterior-variance under-estimation. Jittering
  `L` around the same mean breaks the resonance (Neal 2011, sec. 4.2).
  Window adaptation tunes step size + mass matrix against the *same*
  randomized-`L` kernel, so dual-averaging's acceptance target is
  calibrated to the kernel that actually runs; `num_integration_steps`
  is now the *mean*. The drawn count is floored at 1 leapfrog step (the
  Halton range includes 0, a no-op trajectory). NUTS is unaffected.
- **Multi-chain BlackJAX MCMC dispatch picks `jax.pmap` when devices
  permit.** When `jax.local_device_count() >= num_chains` the per-chain
  runner is mapped with `pmap` (each chain on its own device,
  bit-identical to a single-chain sequential run at the same seed);
  otherwise the prior `vmap` path is used. Single-chain calls
  (``num_chains == 1``) short-circuit both, applying the runner
  directly. Default single-CPU-device behaviour is unchanged for
  ``num_chains == 1``; users with
  ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` get
  per-device parallelism and bit-identical-to-sequential draws
  (notably for NUTS) without code changes. The three BlackJAX
  modules (`_blackjax_mcmc.py`, `_blackjax_rwmh.py`, `_blackjax_ess.py`)
  route their multi-chain dispatch through the new
  `parallel_chain_map` helper in `_inference_utils.py`.

### Changed (breaking)

- **`WorkflowFunction` controls now live outside user call kwargs.**
  `@workflow_function(...)` configures definition-time controls, and
  `workflow.with_options(...)(...)` is the call-time override API for
  `seed`, `n_broadcast_samples`, and `include_inputs`. Wrapped
  functions may now declare and receive those names as ordinary
  parameters. Passing those names as call kwargs no longer configures
  ProbPipe controls; use `workflow.with_options(...)(...)` instead.
- **`WorkflowFunction.workflow_kind` and `Module.workflow_kind` now require
  `WorkflowKind` enum members.** String aliases such as `"task"` / `"flow"`
  and `None` are no longer accepted and now raise `TypeError`; use
  `WorkflowKind.TASK`, `WorkflowKind.FLOW`, or `WorkflowKind.OFF` explicitly.
  The old `parallel=` / `vectorize=` keyword guard on `WorkflowFunction` was
  also removed, so those names are no longer specially reserved by the
  constructor.
- **`tfp_rwmh` removed.** The hand-rolled Python-loop RWMH that sat
  behind ``method="tfp_rwmh"`` is gone; ``blackjax_rwmh`` is the only
  RWMH backend. Callers must rename ``method="tfp_rwmh"`` →
  ``method="blackjax_rwmh"``.
- **Sample-count / observation-count terminology unified
  across the codebase.** Several adjacent concepts had drifted into
  different naming styles (`.n`, `num_draws`, `n_samples`, `n_iter`,
  `n_simulations`, `n_replications`, `num_steps`). Audited and
  consolidated under three canonical names per concept:

  *Finite-sample distribution size.* `.n` is gone. Use
  **`num_atoms`** for any empirical-measure size (one atom = one
  stored realisation): `EmpiricalDistribution.num_atoms`,
  `RecordEmpiricalDistribution.num_atoms`,
  `JointEmpirical.num_atoms`, `BootstrapDistribution.num_atoms`,
  `KDEDistribution.num_atoms`, `BroadcastDistribution` family +
  marginals — all expose `num_atoms`. `ApproximateDistribution`
  inherits `num_atoms` (total chain×draw count) and additionally
  exposes `num_draws` (draws *per chain*).

  *Bootstrap replicate size.* Use **`replicate_size`** for the number
  of items in each bootstrap replicate:
  `BootstrapReplicateDistribution.replicate_size`,
  `RecordBootstrapReplicateDistribution.replicate_size`. The
  constructor kwarg changes from ``n=`` to ``replicate_size=``; the
  related ``source_n`` property becomes ``source_size``. Callers that
  previously wrote ``BootstrapReplicateDistribution(data, n=N)`` will
  now get a ``TypeError`` and must rename to ``replicate_size=N``.
  (`replicate_size`, not `num_observations`: the resampled items come
  from an arbitrary source — parameter samples, function values, etc. —
  so "observations" would overclaim.)

  *Generative-likelihood observation count.*
  ``generate_data(params, n_samples, ...)`` is now
  ``generate_data(params, num_observations, ...)`` across the
  `GenerativeLikelihood` protocol, `GLMLikelihood`,
  `SimpleGenerativeModel`, and `predictive_check` (the latter's
  `n_replications` kwarg also becomes `num_replications`).

- **Inference-method count kwargs unified under `num_*`.** Several
  inference methods exposed `n_*`-style kwargs out of sync with the
  rest of the registry (which uniformly used `num_results` /
  `num_warmup` / `num_chains`). Renamed:
  - `blackjax_sgld` / `blackjax_sghmc`: `num_steps=` → `num_results=`
    (SGMCMC produces one chain draw per step; the kwarg matches
    every other MCMC backend now).
  - `sbi_learn_conditional` / `sbi_learn_likelihood`: `n_iter=` →
    `num_iterations=`, `n_simulations=` → `num_simulations=`.
  - `sbi_learn_conditional` posterior-sampling default
    `n_samples=` → `num_results=`; `DirectSamplerSBIModel.__init__`
    and `condition_on(direct_sampler_model, ...,
    n_samples=...)` likewise.

  Internal `sbijax.simulate_data(..., n_simulations=...)` /
  `sbijax.fit(..., n_iter=...)` / `sbijax.sample_posterior(...,
  n_samples=...)` calls keep their native sbijax kwarg names —
  only the probpipe-facing surface changes.

  Bug fix bundled with the rename: `tests/test_sbijax.py` was
  calling `condition_on(nle_model, obs, method="tfp_nuts",
  n_samples=500, n_warmup=500, n_chains=2, ...)` — the MCMC backend
  silently ignored those kwargs (it expects `num_results=` /
  `num_warmup=` / `num_chains=`) and the test passed by accident.
  Fixed.

- **`condition_on` MCMC default switched from TFP to BlackJAX NUTS,
  plus inference-method priority re-anchoring.** Several entangled
  changes consolidated into a single migration:

  *Auto-dispatch winner switches to BlackJAX NUTS.* `blackjax_nuts`
  (priority 85, tier 81–90) wins auto-dispatch for any
  `SupportsLogProb` + JAX-traceable target — the canonical ProbPipe
  model class. `tfp_nuts` / `tfp_hmc` are demoted to the opt-in-only
  sentinel (`priority=0`); they stay registered and reachable via
  `method="tfp_nuts"` / `method="tfp_hmc"` for bit-pattern regression
  checks or side-by-side comparisons.

  *Structurally-unreachable methods demoted to `priority=0`.* Methods
  whose `check()` is identical to a higher-priority sibling can never
  win auto-dispatch — they're opt-in in effect. Made that explicit:
  `blackjax_hmc` (same `check()` as `blackjax_nuts`) and
  `blackjax_sghmc` (same `check()` as `blackjax_sgld`, which is also
  the simpler default — fewer tuning dials) are now opt-in only.

  *VI demoted to opt-in.* `pymc_advi` (was priority 25) is now
  `priority=0`. VI is a deliberate bias-for-speed tradeoff that users
  should pick explicitly via `method="pymc_advi"`; silently dispatching
  into it when (e.g.) `pymc_nuts` happens to fail would surface VI in
  MCMC's place.

  *NUTS-tier numbers retuned.* `nutpie_nuts` 85 → 88 (top of the
  optimised-backend tier — Rust gradients are the fastest of every
  registered NUTS backend); `pymc_nuts` 81 → 82 (ties with
  `cmdstan_nuts` at 82; the two apply to disjoint model classes so
  the tie is documentary).

  `tfp_rwmh` (gradient-free RWMH) is unchanged at priority 55 — the
  gradient-free-MCMC migration to BlackJAX is queued separately
  (`~/.claude/plans/bie-rwmh-blackjax-migration.md`).

  Migration: an existing `condition_on(model, data)` call that
  previously ran TFP NUTS now runs BlackJAX NUTS. The numerical
  posterior is asymptotically identical but the per-seed bit pattern
  differs. Pin `method="tfp_nuts"` for bit-pattern regression. The
  closed-form correctness gate (mean within ~3 σ_MC, variance within
  10% on a known 2-D Gaussian target) is tested under
  `tests/test_blackjax_mcmc.py`. Existing `condition_on(...,
  method="pymc_advi")` / `method="blackjax_hmc"` /
  `method="blackjax_sghmc"` calls continue to work — only the
  auto-dispatch path changes.

- **Distribution & Record hierarchy cleanup (#200).** Implements the
  integrated cleanup plan as six self-contained commits. The public-
  facing changes are:
  - **`Distribution.validation_results` is removed.**
    `predictive_check` now writes its per-invocation payload to
    `dist.auxiliary["predictive_check/check_N"]` (a wrapped
    `xarray.Dataset` under a numbered group). Future validation
    functions (LOO, WAIC, …) land under their own named groups in
    the same `DataTree`. Code that read `dist.validation_results`
    should read `dist.auxiliary["predictive_check"]` instead.
  - **`flatten_value` / `unflatten_value` are now `@staticmethod` with
    explicit kwargs.** Callers pass `event_shape=` /
    `template=` explicitly:
    `dist.flatten_value(value, event_shape=dist.event_shape)` and
    `dist.unflatten_value(flat, template=dist.record_template)`.
    The previous instance-method form (no kwargs) raises at runtime.
  - **`_default_support` classmethods are removed** from every
    concrete distribution (`Normal`, `Gamma`, `Poisson`, …; 24 in
    total). Support compatibility is now checked post-construction
    via `NumericRecordDistribution._check_support_compatible(source)`;
    downstream code that reached for the classmethod should use the
    instance `support` / `supports` properties.
  - **`SimpleModel.__init__` requires a `RecordDistribution` prior**
    (in addition to the pre-existing `SupportsLogProb` check). Priors
    that satisfy `SupportsLogProb` but aren't `RecordDistribution`
    raise `TypeError`. The type system can't express the intersection
    statically, so the runtime guard is the backstop.
  - **Default model names change from `None` to the class name.**
    `SimpleModel()`, `SimpleGenerativeModel()`, `PyMCModel()`,
    `StanModel()`, and `DirectSamplerSBIModel()` now default to
    `"SimpleModel"` / `"SimpleGenerativeModel"` / `"PyMCModel"` /
    `"StanModel"` / `"DirectSamplerSBIModel(<alg>)"` when no name is
    supplied. The metaclass invariant requires every `Distribution`
    instance to have a non-empty name.
  - **`NumericRecordDistribution.event_shape` is abstract** —
    raises `NotImplementedError` on the base. Single-leaf subclasses
    must override directly; multi-leaf subclasses (joints) set
    `_record_template` explicitly and never trigger the auto-build.
    Previously the default tried to derive from `event_shapes`,
    which looped back through `record_template`.
  - **`ProductDistribution` and `SequentialJointDistribution`
    conditionally mix in `NumericRecordDistribution`** based on
    their resolved leaves. Both stay rooted at the general
    `RecordDistribution` (their content is well-defined for
    non-numeric leaves too — sampling produces a `Record` keyed by
    component name, conditioning and named-component access always
    work). When *every* leaf is itself a `NumericRecordDistribution`,
    the dynamic class factory adds `NumericRecordDistribution` to the
    bases, so the joint also exposes the numeric API (`event_size`,
    `flatten_value` / `unflatten_value`, `as_flat_distribution`,
    `dtypes`, `supports`). For mixed or non-numeric leaves those
    methods are simply absent on the instance. Leaf type constraint
    relaxed from `NumericRecordDistribution` to `Distribution`.
    **Caller-visible consequence:**
    `isinstance(joint, NumericRecordDistribution)` is no longer
    guaranteed for `ProductDistribution` / `SequentialJointDistribution`
    instances — it returns `True` only when every resolved leaf is
    itself an NRD (the common case). Downstream code that branched on
    `isinstance(..., NumericRecordDistribution)` for these joints
    should verify the new dispatch matches its expectations, or
    switch to checking for the specific capability (e.g.,
    `hasattr(joint, "event_size")`).
  - **`NumericJointEmpirical` adds `NumericRecordDistribution` as a
    mixin** (previously implicit via `JointEmpirical` only). The
    sibling `JointEmpirical` stays on `RecordDistribution` and now
    builds a structural template from the stored samples (object-
    dtype leaves use `None` specs) to satisfy the metaclass
    invariant.

### Added

- **`RecordTemplate.event_shapes` and `RecordTemplate.field_event_shape(name)`**
  expose per-top-level-field event shapes (nested sub-templates and
  opaque leaves collapse to `()`). The previous helper
  `RecordDistribution._field_event_shape` is removed in favor of these
  template methods.

- **Metaclass-enforced invariants.** Every `Distribution` instance
  has a non-empty `name`; every `RecordDistribution` instance has a
  non-`None` `record_template`. The checks fire post-`__init__` via
  the `_DistributionMeta` / `_RecordDistributionMeta` metaclasses
  (derived from `typing._ProtocolMeta` to compose with
  `@runtime_checkable` protocols). Subclasses that forget either
  invariant raise `TypeError` at construction with a clear pointer.

### Changed

- **`GLMLikelihood` fits an intercept by default** (``fit_intercept=True``).
  The covariate matrix ``X`` carries only the covariates — no leading
  column of 1s. ``params`` flattens to ``(intercept, *slopes)`` and the
  likelihood computes ``eta = intercept + X @ slopes``. Pass
  ``fit_intercept=False`` for the classical "model matrix" convention
  where the user prepends the constant column to ``X`` themselves.
  Avoids the axis-position ambiguity of stacking the intercept slot
  into ``X``; matches the pattern in sklearn / statsmodels GLM APIs.

- **Dispatch-registry hierarchy split: `BaseDispatchRegistry`,
  `UnaryDispatchRegistry`, `BinaryDispatchRegistry`.** The
  arity-independent logic (registration, priority management, opt-in
  filtering, `check`/`execute` loop) lives on the new
  `BaseDispatchRegistry` abstract base. `UnaryDispatchRegistry` is the
  single-argument concrete subclass that replaces the previous
  `MethodRegistry` and now backs the inference method registry.
  `BinaryDispatchRegistry` adds two-argument dispatch on the joint type
  of the first two positional args. `BaseDispatchMethod` /
  `UnaryDispatchMethod` / `BinaryDispatchMethod` mirror the registry
  split on the method side. The previous `Method` / `MethodRegistry`
  aliases are removed — inference-method subclasses should subclass
  `UnaryDispatchMethod` (or the `InferenceMethod` re-export from
  `probpipe.inference`).

- **Inference-method registry priorities re-anchored with a semantic
  convention (issue #189).**
  - `priority > 50` marks *exact* methods; `0 < priority <= 50` marks
    *inexact* methods; `priority == 0` is the opt-in-only sentinel
    (selectable by name via `method="..."` but skipped during
    auto-dispatch). This is the new default value inherited from
    `Method.priority`.
  - Built-in priorities re-anchored: `nutpie_nuts` 80→85,
    `cmdstan_nuts` 70→82, `pymc_nuts` 60→81, `tfp_nuts` 100→75,
    `tfp_hmc` 90→65, `tfp_rwmh` 50→55, `blackjax_sgld` 30→45,
    `blackjax_sghmc` 25→42, `pymc_advi` 35→25, `sbijax_smcabc` 40→5.
    The relative ordering among exact methods is corrected so that
    optimised backends (`nutpie_nuts`, `cmdstan_nuts`, `pymc_nuts`)
    sit above the general-purpose `tfp_nuts`.
  - `MethodRegistry._find_methods()` now skips priority-0 methods
    during auto-dispatch. `MethodRegistry.set_priorities()` emits a
    `UserWarning` when an override crosses into or out of `0`;
    crossings of the documentary `50` break do not warn.
  - The `OPT_IN_ONLY_PRIORITY` sentinel is exported from
    `probpipe.core._registry` for use in `Method` subclasses that
    want to opt out of auto-dispatch by name.
  - The contributor-facing selection criteria and tier ranges for
    setting a new method's priority are documented under
    [Extending ProbPipe → Setting priority for a new method](docs/api/extending.md#setting-priority-for-a-new-method).
  - Migration: a `Method` subclass that previously relied on
    inheriting `priority = 0` from the base class while expecting
    auto-dispatch must now set a positive priority explicitly.
    `set_priorities` calls that stay within positive priorities are
    unaffected; calls that move a method to or from `0` emit a
    warning explaining the auto-dispatch participation change.

- **PyMC-backed posteriors now carry RV-keyed Record structure.**
  ``PyMCModel`` exposes a ``record_template`` property that pairs each
  free RV with its event shape (scalar RVs → ``()``; shape-`k` RVs →
  ``(k,)``). The PyMC NUTS, PyMC ADVI, and nutpie inference paths all
  thread this through to ``make_posterior``, so ``mean(post)`` returns
  a ``NumericRecord`` keyed by RV name and ``draws()`` returns a
  ``NumericRecordArray``. Previously, PyMC posteriors had no field
  structure and ``draws()`` returned a flat ``(n_draws, n_params)``
  array. Models declared with multiple scalar RVs (e.g. separate
  ``intercept`` and ``slope`` ``pm.Normal`` calls) now produce a
  field-per-RV posterior matching the ``ProductDistribution``-prior
  workflow.

  Free RVs whose ``type.shape`` contains a ``None`` dimension are
  rejected with ``ValueError`` — silently dropping unknown dims would
  produce an under-shaped template.

- **`GLMLikelihood` no longer accepts stacked ``(X, y)`` arrays.** Both
  ``log_likelihood`` and ``per_datum_log_likelihood`` now require either
  ``data = Record(X=..., y=...)`` (canonical) or, for ``log_likelihood``
  only, a bare response array when ``X`` was supplied at construction
  time. Passing a single matrix whose last column was interpreted as
  the response is intentionally rejected — ProbPipe uses named Records
  to avoid axis-position ambiguity. Existing call sites that used a
  ``Record`` are unaffected.

- **`SimpleModel.prior` / `SimpleGenerativeModel.prior` type
  annotations tightened** from ``Distribution[P]`` /
  ``SupportsSampling[P]`` to the specific capability protocol
  (``SupportsLogProb[P]`` / ``SupportsSampling[P]``). Static type
  checkers now catch wrong-type priors at the call site; the
  construction-time ``isinstance`` check stays as a backstop. The two
  model wrappers are now parallel in both the input typing and the
  ``.prior`` property return type.

### Added

- **BlackJAX-backed SGMCMC methods** registered with
  ``inference_method_registry``:
  - ``blackjax_sgld`` — Stochastic Gradient Langevin Dynamics. Priority 45.
  - ``blackjax_sghmc`` — Stochastic Gradient Hamiltonian Monte Carlo. Priority 42.

  Both consume a `SimpleModel` whose `likelihood` satisfies
  `ConditionallyIndependentLikelihood`, plus a required `batch_size=`
  kwarg. Internally they wrap the model+data in a
  `MinibatchedDistribution` and feed BlackJAX's gradient estimator
  via the per-step random-measure draw — the kernel stays oblivious
  to the minibatching convention.

  ```python
  posterior = condition_on(
      model, data,
      method="blackjax_sgld",
      batch_size=64, num_results=2000, num_warmup=500, step_size=1e-3,
  )
  ```

  Priorities sit in the refinement-based MC tier (1–50), below every
  exact full-batch gradient method (`tfp_nuts=75`, `tfp_hmc=65`,
  `tfp_rwmh=55`). SGMCMC's `check()` also requires `batch_size=`, so
  it does not fire on a routine `condition_on(model, observed)` call —
  the user opts in by passing `batch_size=` (and typically the
  matching `method=`).

- **`MinibatchedDistribution`** (`probpipe.MinibatchedDistribution`)
  — a `RandomMeasure[Record]` over fixed-minibatch stochastic
  surrogates of the full-data unnormalized log-posterior. A draw is a
  `Distribution[Record]` with unnormalized log-density
  `log p(theta) + (N/b) * sum_{d in B} log p(d|theta)`, an unbiased
  stochastic surrogate (in expectation over the minibatch `B`) of the
  full-data target; the `N/b` rescaling makes the gradient an unbiased
  estimator.

  The constructor takes a prior and a conditionally-independent
  likelihood directly, mirroring `SimpleModel(prior, likelihood)` on
  the first two args. Consume the measure via
  `SupportsRandomUnnormalizedLogProb` to get the per-minibatch
  log-density callable that SGMCMC kernels feed `jax.grad`:

  ```python
  from probpipe import MinibatchedDistribution, Record, random_unnormalized_log_prob

  m = MinibatchedDistribution(prior, likelihood, Record(X=X, y=y), batch_size=64)

  rf = random_unnormalized_log_prob(m)
  target = rf._sample(k)                     # callable: theta -> log~D_B(theta)
  grad = jax.grad(target)(theta)             # unbiased gradient estimate
  ```

  This is the path stochastic-gradient MCMC kernels use under the
  hood; the BlackJAX SGLD / SGHMC dispatch builds a `MinibatchedDistribution`
  internally and threads `target` into the BlackJAX gradient
  estimator. Tempered SMC (future work) is expected to consume the
  same surface.

- **`ConditionallyIndependentLikelihood`** (`probpipe.ConditionallyIndependentLikelihood`)
  — a `Likelihood` subclass / Protocol whose observations factorise as
  `log p(D | theta) = sum_i log p(d_i | theta)`. Adds a
  `per_datum_log_likelihood(params, datum)` method on top of the base
  `Likelihood`'s `log_likelihood(params, data)`. Required by
  stochastic-gradient inference (the upcoming `MinibatchedDistribution`)
  and independently useful for held-out predictive log-likelihoods,
  leave-one-out cross-validation, and PSIS-LOO. The existing concrete
  likelihoods (`GLMLikelihood`, `_NLELikelihood`, `_NRELikelihood`) all
  satisfy the Protocol — `GLMLikelihood` via a direct family
  `log_prob` evaluation that skips the per-batch tile, the two
  sbijax-backed classes via a length-1-batch fallback.

  A standalone helper `_default_per_datum_log_likelihood(likelihood,
  params, datum)` provides the length-1-batch implementation for
  subclasses that want a default rather than an efficient override.

- **`SimpleModel.prior` / `SimpleModel.likelihood`** and
  **`SimpleGenerativeModel.prior` / `SimpleGenerativeModel.likelihood`**
  — public read-only properties that expose the underlying components
  without poking at private state. The two model wrappers stay
  symmetric: `SimpleModel.likelihood` is typed `Likelihood`,
  `SimpleGenerativeModel.likelihood` is typed `GenerativeLikelihood`.

- **`FlatNumericRecordDistribution`** (`probpipe.FlatNumericRecordDistribution`)
  — a `NumericRecordDistribution` subclass that enforces the flat
  contract (single field, `event_shape == (N,)`). Algorithms that
  operate on a flat parameter vector (MCMC kernels, optimisers,
  Hessian / curvature builders, variational families, Pathfinder /
  Laplace surrogates) can require this type rather than runtime
  shape probes. Carries the `flat_size: int` shortcut (=
  `event_shape[0]`) and the `as_record_distribution(template=...)`
  method.

  The natively-multivariate parametrics
  (`MultivariateNormal`, `Dirichlet`, `Multinomial`, `VonMisesFisher`)
  now inherit from `FlatNumericRecordDistribution` in addition to
  `TFPDistribution`. `FlattenedDistributionView` also implements the
  contract by construction. Scalar parametrics (`Normal`, `Beta`,
  `Bernoulli`, …) have `event_shape == ()` and do not satisfy the
  contract directly; call `.as_flat_distribution()` to obtain a
  `FlattenedDistributionView` with `event_shape == (1,)`.

- **`FlatNumericRecordDistribution.as_record_distribution(template=...)`**
  — inverse of `as_flat_distribution()`. Lifts a flat distribution to
  a Record-keyed view under a user-supplied `NumericRecordTemplate`.
  Sampling, log-prob, and moments delegate to the source and reshape
  via the template; capability protocols (`SupportsX`) match the
  source via dynamic isinstance dispatch. The view is a thin wrapper —
  no value copying.

  ```python
  from probpipe import MultivariateNormal, NumericRecordTemplate

  mvn = MultivariateNormal(                     # already a FlatNRD
      loc=jnp.array([1.0, 2.0, 3.0, 4.0]),
      cov=jnp.diag(jnp.array([0.5, 1.0, 1.5, 2.0])),
      name="theta",
  )
  template = NumericRecordTemplate(intercept=(), slope=(3,))
  posterior = mvn.as_record_distribution(template=template)
  draw = sample(posterior, key=k)         # NumericRecord(intercept, slope)
  mean(posterior)["slope"]                # vector mean of the slope block
  ```

### Changed

- **`FlattenedView` renamed to `FlattenedDistributionView`** and now
  inherits from `FlatNumericRecordDistribution` (formerly
  `NumericRecordDistribution`). The view's flat contract was always
  satisfied structurally; the new base class makes it explicit and
  enables receiver-type-driven dispatch for `as_record_distribution`.
- **`_RecordLiftedView` renamed to `NumericRecordDistributionView`** and
  made public. Constructed via
  `FlatNumericRecordDistribution.as_record_distribution(template=...)`.

### Migration

- Code that imports `FlattenedView`: rename to `FlattenedDistributionView`.
- Code that imports `_RecordLiftedView`: rename to
  `NumericRecordDistributionView`.
- Code that calls `as_record_distribution` on a non-flat distribution
  (e.g., a scalar `Normal`): chain via `.as_flat_distribution()` first.
  Calling it directly on a non-flat `NumericRecordDistribution` now
  raises `TypeError` instead of `ValueError` and points at the
  `as_flat_distribution()` chain.

- **`SupportsArrayBackend` capability protocol** (`probpipe.SupportsArrayBackend`)
  declares that a `Distribution` subclass can produce a fused storage
  backend for `DistributionArray`. Implemented by every TFP-backed
  concrete class (`Normal`, `Beta`, `Gamma`, `MultivariateNormal`,
  `Dirichlet`, …) via inheritance from `TFPDistribution`. Distribution
  classes that don't implement the protocol still work in a
  `DistributionArray` via the literal-array fallback path.

- **`DistributionArray.from_batched_params(dist_cls, *, name, batch_shape=None, **batched_params)`**
  factory + ergonomic per-class alias **`Distribution.from_batched_params(*, name, batch_shape=None, **batched_params)`**.
  Constructs a `DistributionArray` of homogeneous components,
  dispatching on `SupportsArrayBackend`: TFP-backed classes get a
  fused TFP-batched backend; other classes fall back to one
  `dist_cls` instance per cell. Per-cell names auto-suffix
  `f"{name}_{flat_index}"`. `batch_shape` is inferred from broadcast
  of array-valued params; classes with heterogeneous per-param event
  ranks (`MultivariateNormal`, `Dirichlet`) require explicit
  `batch_shape=...`.

  ```python
  # Recommended ergonomic form
  da = Normal.from_batched_params(loc=jnp.zeros(5), scale=1.0, name="x")
  da.batch_shape       # (5,)
  da[2].name           # "x_2"

  # Equivalent universal form
  da = DistributionArray.from_batched_params(
      Normal, loc=jnp.zeros(5), scale=1.0, name="x",
  )
  ```

### Changed (breaking)

- **Prefect orchestration is now opt-in** (#182). The shipped global
  default for `prefect_config.workflow_kind` is `WorkflowKind.OFF`
  instead of the prior `WorkflowKind.DEFAULT` (which auto-promoted to
  `TASK` whenever Prefect was importable). The old behaviour silently
  enabled Prefect for any environment with Prefect on `sys.path` —
  including environments where Prefect was pulled in as a transitive
  dependency — and produced a confusing `httpx.ConnectError` when no
  Prefect server was running. The new default produces no surprise
  network traffic; users who want orchestration opt in once per
  session or deployment:

  ```python
  import probpipe
  probpipe.prefect_config.workflow_kind = probpipe.WorkflowKind.TASK
  ```

  Or via the new `PROBPIPE_WORKFLOW_KIND` environment variable
  (`off` / `task` / `flow` / `default`, case-insensitive), which is
  read once at import time. Per-workflow overrides remain available via
  `@workflow_function(workflow_kind=probpipe.WorkflowKind.TASK)`; string
  aliases are no longer accepted in this release (see the
  `workflow_kind` breaking entry above).
  Migration: production callers that relied on the implicit
  "Prefect importable → tasks enabled" path must add the one-line
  assignment or env var above.

- **`NumericRecordDistribution.dtypes` is canonical; subclasses must
  override.** The base accessor previously returned
  ``{name: default_float_dtype()}`` for every field of the
  ``record_template`` (a silent lie for every integer-valued TFP
  distribution — ``Bernoulli`` / ``Categorical`` reported
  ``float32``). The base now raises ``NotImplementedError`` so the
  truth direction is unambiguous; concrete subclasses declare
  ``dtypes`` directly via the new
  ``_spread_to_fields(value)`` helper:

  ```python
  >>> from probpipe import Bernoulli
  >>> Bernoulli(probs=0.5, name="x").dtype
  jnp.int32   # was float32 (the lie)
  >>> Categorical(probs=jnp.array([0.5, 0.5]), name="x").dtype
  jnp.int32   # was float32
  ```

  Migration for custom subclasses: implement
  ``dtypes`` returning ``{field: dtype}`` aligned with
  ``record_template.fields``. The single-leaf shortcut for
  uniform-dtype subclasses is
  ``return self._spread_to_fields(my_dtype)``. The convenience
  ``dtype`` accessor derives automatically.

  Related cleanups landing in the same PR:

  - ``supports`` is also canonical now (raises if not overridden);
    ``support`` is a convenience that derives via
    ``_single_field_name``. Existing single-field ``support``
    overrides on concrete TFP-backed classes continue to work.
  - ``record_template`` auto-build (single-field
    ``RecordTemplate(**{name: event_shape})``) moved from
    ``TFPDistribution`` to the base, so any concrete subclass
    with a ``name=`` and ``event_shape`` gets a template
    automatically.
  - ``treedef`` derives from ``record_template`` (leaf for
    single-leaf, ``NumericRecord`` skeleton for multi-leaf) and
    is cached on first read.
  - ``flat_event_shapes`` tree-walks ``event_shapes`` rather than
    hardcoding ``[event_shape]``.
  - ``_check_support_compatible`` reads canonical ``supports``
    (per-field check on multi-leaf source, single-leaf message
    preserved).

- **`Distribution.batch_shape` removed.** The property is gone
  from `Distribution` and every subclass; reads now raise
  `AttributeError`. Collections of distributions live in
  `DistributionArray`, which retains its own `batch_shape` (the
  outer array shape).

  ```python
  >>> from probpipe import Normal
  >>> hasattr(Normal(loc=0.0, scale=1.0, name="x"), "batch_shape")
  False
  ```

  Migration: drop the read — once batched parameters were rejected,
  it was always `()`. `GaussianRandomFunction.predict` (and every
  `ArrayRandomFunction` subclass) now returns a `DistributionArray`
  rather than a single batched `Normal` / `MultivariateNormal`;
  per-cell `event_shape` is unchanged. Fully-joint predictions with
  no extra batch axes return a 0-d `DistributionArray`; ops
  (`sample`, `mean`, `log_prob`, …) auto-unwrap a 0-d DA to its
  single cell, so call sites stay unchanged.

- **`DistributionArray` container surface aligned with numpy / jax**
  (#178). `iter(da)` now walks the leading axis: a 1-D array yields
  its scalar cells (unchanged); a multi-d array yields
  ``DistributionArray`` slices of shape ``batch_shape[1:]``,
  mirroring ``iter(np.zeros((2, 3)))``. Use ``da.components`` for
  flat row-major access over every cell (the pre-#178 default).
  Adds ``DistributionArray.size`` returning ``prod(batch_shape)``,
  matching ``np.ndarray.size`` / ``jax.Array.size``.

- **`RecordDistribution.n` and `DistributionArray.n` removed.**
  STYLE_GUIDE §1.9 reserves `.n` for finite-sample distribution
  classes that hold a finite collection of samples / observations
  / components (`EmpiricalDistribution`, `BootstrapDistribution`,
  `BroadcastDistribution`, …). The two cases removed here did
  not fit the contract: parametric `Normal(0, 1)` does not "hold"
  any items, and `DistributionArray` is a positional collection of
  independent cells, not a finite-sample distribution. Migration:
  for `DistributionArray`, use `len(da)` (leading-axis size) or
  `prod(da.batch_shape)` (total cell count) — `__repr__` now shows
  `batch_shape=...`. For parametric distributions, drop the
  call — it always returned `1`. Finite-sample distributions
  retain `.n` (see STYLE_GUIDE §1.9 for the full table).

- **TFP-backed distribution constructors reject batched parameters.**
  `Normal(loc=jnp.zeros(5), scale=1.0, name="x")` (and the same
  pattern for every other TFP-backed class — `Beta`, `Gamma`,
  `MultivariateNormal`, `Pareto`, `TruncatedNormal`, `Binomial`, …)
  now raises `ValueError` whenever the parameters imply a non-empty
  TFP `batch_shape`. The framework hierarchy rule "one random
  variable per `Distribution`" (CONTRIBUTING.md) is enforced at
  construction time.

  ```text
  ValueError: Normal parameters imply batch_shape=(5,); wrap multiple
  distributions in a DistributionArray instead. See
  DistributionArray.from_batched_params(Normal, ...) (or the alias
  Normal.from_batched_params(...)) for the factory.
  ```

  Migration: route through the
  `DistributionArray.from_batched_params` factory (or its per-class
  alias) added in the previous release. The factory is
  performance-equivalent to the legacy form because the fused
  `_TFPArrayBackend` wraps the same TFP-batched distribution under
  the hood.

  ```python
  # Before (rejected)
  n = Normal(loc=jnp.zeros(5), scale=1.0, name="x")

  # After (recommended ergonomic form)
  da = Normal.from_batched_params(loc=jnp.zeros(5), scale=1.0, name="x")

  # After (universal entry point)
  da = DistributionArray.from_batched_params(
      Normal, loc=jnp.zeros(5), scale=1.0, name="x",
  )
  ```

  Removed associated tests that exercised the legacy form's
  per-element support checks: ``test_uniform_support_array_bounds``,
  ``test_half_cauchy_support_array_bounds``,
  ``test_pareto_support_array_bounds``,
  ``test_truncated_normal_support_array_bounds``,
  ``test_binomial_support_array_total_count``,
  ``test_repr_with_batch_shape``. Per-element support checks belong
  on `Constraint` directly; batched constructions migrate to
  `DistributionArray.from_batched_params`.

  Internal infrastructure that legitimately needs the batched form
  (the `_TFPArrayBackend` fused-storage backend, the
  `ProbPipeConverter` dispatch, sequential-joint sampling /
  log_prob, `GaussianRandomFunction.predict`) opts into a private
  bypass; user code is unaffected by the bypass and always sees the
  rejection.

- **Empirical / Bootstrap / Marginal class consolidation.** The
  generic-vs-numeric pair is collapsed into a generic ``[T]`` base
  plus a single Record-based specialisation:

  | Removed | Replacement |
  |---|---|
  | ``NumericEmpiricalDistribution`` | ``RecordEmpiricalDistribution`` |
  | ``ArrayBootstrapReplicateDistribution`` | ``RecordBootstrapReplicateDistribution`` |
  | ``_ArrayMarginal`` (private) | ``_RecordMarginal`` (private) |
  | ``_RecordEmpiricalDistribution`` (private) | ``RecordEmpiricalDistribution`` |
  | ``_RecordBootstrapReplicateDistribution`` (private) | ``RecordBootstrapReplicateDistribution`` |
  | ``_RecordArrayMarginal`` (private) | ``_RecordMarginal`` (private) |

  Migration: a numeric array auto-wraps as a single-field ``Record``
  keyed by the (now mandatory) ``name=`` kwarg.

  ```python
  # Before
  emp = EmpiricalDistribution(arr)                    # Worked
  emp = NumericEmpiricalDistribution(arr)             # Worked
  emp = ArrayBootstrapReplicateDistribution(arr)      # Worked

  # After
  emp = EmpiricalDistribution(arr, name="theta")       # ✓
  emp = EmpiricalDistribution(arr)                    # ValueError: name= required
  ```

  The ``name=`` becomes the field name of the auto-wrapped
  ``Record``; downstream code that does ``emp.samples["theta"]`` /
  ``emp["theta"]`` then has a meaningful key. If you want to keep the
  old call-site shape, wrap explicitly:
  ``EmpiricalDistribution(Record(theta=arr))``.
- **`BootstrapReplicateDistribution[T]` accepts a `SupportsSampling`
  source.** Each replicate is ``n`` i.i.d. draws from
  ``source._sample``. **``n`` is mandatory** when ``source`` is a
  ``SupportsSampling`` distribution (no canonical observation count);
  it remains optional for ``Record`` / numeric-array / ``Empirical``
  sources, where it defaults to the source's row count.
  ``BootstrapReplicateDistribution(Normal(0, 1, name="x"), n=50)``.
- **`NumericJointEmpirical` no longer claims `SupportsLogProb`.** The
  Gaussian-approximation log-density is gone — empirical distributions
  do not advertise a density. Migration:
  ``from_distribution(emp, KDEDistribution, ...)`` for a non-parametric
  density, or fit a parametric distribution and call ``log_prob`` on
  that.
- **Distributions are non-iterable.** Codified in STYLE_GUIDE §1.11
  with a regression test
  (``tests/test_iteration_protocol.py``). Finite-sample subclasses
  (see §1.9) expose stored samples via ``.samples`` / ``.draws()``
  and ``.n``; parametric distributions do not have ``.n``.

- **`Record` field ordering is now insertion-order**, not alphabetical.
  ``Record(z=1, a=2)`` now iterates ``("z", "a")``. Same change applies
  to ``RecordTemplate``, ``RecordArray``, and every Record-based
  distribution that derives ``fields`` from the underlying store.
  Previous alphabetical ordering was an accident of
  ``OrderedDict(sorted(...))``.
- **`/` is reserved in `Record` and `RecordTemplate` field names.**
  Construction-time ``ValueError``. Used as the slash-delimited path
  separator in ``record["params/intercept"]`` style access.
- **`Record.to_datatree()` / `Record.from_datatree(...)` removed.**
  Use ``record.to_numeric().to_native()`` for a metadata-preserving
  round-trip via the aux registry, or ``xr.DataTree`` directly if you
  specifically want a DataTree.
- **`NumericRecord(...)` (and `Record.to_numeric()`) raise `TypeError`
  on non-coercible leaves** (strings, opaque objects). Today's
  implicit failure inside ``NumericRecord(...)`` becomes an explicit,
  well-messaged error at construction time.
- **`RecordTemplate.leaf_shapes` keys for nested templates use `/`**
  instead of ``.`` (e.g. ``"physics/force"`` instead of
  ``"physics.force"``) for consistency with ``Record["a/b"]`` path
  access.

### Added

- **Framework abstraction hierarchy** documented in CONTRIBUTING.md.
  Three rules: one random variable per ``Distribution``; two
  implementations per concept (generic + Record-based); iteration is
  a Record-family convention.

- **`RecordEmpiricalDistribution.flat_samples`** — flat ``(n, dim)``
  matrix view across all fields, where
  ``dim = sum(prod(event_shape_f) for f in fields)``. Field order is
  the dist's insertion order; multi-dim event shapes flatten
  row-major. Use ``.samples`` for the structured ``NumericRecord``
  view (per-field access via ``.samples[name]``) and ``.flat_samples``
  for stacked-matrix idioms — ``post.flat_samples.mean(axis=0)``,
  per-parameter posterior summaries, etc. Replaces hand-rolled
  ``np.column_stack([post.samples[f] for f in post.fields])``.

- **`Record.to_numeric()` / `NumericRecord.to_native()`** — explicit
  conversion to / from ProbPipe's native JAX-array form, with metadata
  round-trip via the aux registry.
- **`probpipe.AuxHooks` / `register_aux(...)` / `aux_for(...)` /
  `aux_registry`** in :mod:`probpipe.core._array_backend` — a registry
  of ``(capture, restore)`` hooks for round-tripping backend-specific
  metadata across the ``Record`` ↔ ``NumericRecord`` boundary.
  Built-in registrations (gated on import) cover
  ``xarray.DataArray`` (dims / coords / attrs / name),
  ``pandas.Series`` (index / name / dtype), and ``pandas.DataFrame``
  (index / columns / dtypes).
- **`NumericRecord.aux`** property — captured backend metadata, keyed
  by field name. ``None`` when no field had a registered hook.
- **Slash-delimited path access** on nested ``Record``s:
  ``record["params/intercept"]`` is sugar for
  ``record["params", "intercept"]``. ``"a/b/c" in record`` works the
  same way.

### Changed

- **dtype handling** now follows JAX's rules. Distributions, weights, and
  empirical classes preserve user-supplied dtypes and honor
  ``jax.config.update("jax_enable_x64", True)`` end-to-end. Previously every
  TFP-backed constructor silently downcast its parameters to ``float32``,
  causing ``log_prob`` / ``sample`` / ``mean`` to raise ``TypeError`` under
  x64. Multi-parameter constructors now promote inputs to a common float
  dtype via ``jnp.result_type`` (integer inputs are promoted to JAX's
  default float, so ``Normal(loc=0, scale=1)`` still works). Internal
  helpers ``_default_float_dtype()`` and ``_promote_floats()`` live in
  ``probpipe/_dtype.py``. The float64-truncation warning filter previously
  in ``probpipe/__init__.py`` is removed.

### Added

- **`_RecordArrayView`** (`RecordArray.view(field)`) — single-field view of a
  ``RecordArray`` column that carries its parent as shared-identity metadata.
  The ``WorkflowFunction`` sweep layer groups sibling views from one parent
  into a single zip axis; views from different parents product.
- **Uniform `select_all()`** on ``Record`` / ``RecordArray`` /
  ``RecordDistribution``. Splatting the result into a
  ``@workflow_function`` preserves correlation on the two batched variants
  and plain splats fields on scalar ``Record``.
- **Public `.parent` / `.field`** properties on both
  ``_RecordArrayView`` and ``_RecordDistributionView``.
- **Single-field `.shape` / `.ndim` shims** on ``RecordDistribution`` and
  ``_RecordDistributionView`` (mirror the existing shims on
  ``NumericRecord`` / ``NumericRecordArray``). Multi-field distributions
  raise ``TypeError``.

### Changed (breaking)

- **`len(RecordArray)`** now returns the **field count** (matching
  ``len(Record)``) instead of ``prod(batch_shape)``. For the flat batch
  size, use ``prod(ra.batch_shape)``.
- **`event_shapes`** now always returns ``dict[str, tuple[int, ...]]``.
  Untemplated (legacy) distributions return ``{}``; use the singular
  ``.event_shape`` for the whole-sample shape.
- **`component_names` → `fields`** on every Record-based distribution and
  model (``RecordDistribution``, ``ProductDistribution``, ``JointGaussian``,
  ``JointEmpirical``, ``SequentialJointDistribution``,
  ``BroadcastDistribution``, ``ProbabilisticModel``, ``SimpleModel``,
  ``SimpleGenerativeModel``, ``PyMCModel``, ``StanModel``). No backward
  alias.

## [0.1.0] - 2025-03-21

Initial release with TensorFlow Probability / JAX backend.

### Added

- **Distribution framework**: `Distribution` ABC with TFP shape semantics (`sample_shape + batch_shape + event_shape`), `TFPDistribution` mixin, `EmpiricalDistribution` (direct JAX).
- **23 distribution wrappers**: 14 continuous (Normal, Beta, Gamma, etc.), 5 discrete (Bernoulli, Binomial, Poisson, etc.), 4 multivariate (MultivariateNormal, Dirichlet, Multinomial, Wishart, VonMisesFisher).
- **Constraints**: `Constraint` base class with partial-order compatibility checking. Built-in singletons (`real`, `positive`, `unit_interval`, `simplex`, etc.) and factories (`interval()`, `greater_than()`).
- **Transformed distributions**: `TransformedDistribution` with TFP bijectors (Exp, Sigmoid, Softplus, Shift, Scale, Chain). Automatic support derivation.
- **Joint distributions**: `ProductDistribution` (independent), `SequentialJointDistribution` (autoregressive), `JointEmpirical` (weighted joint samples), `JointGaussian` (exact analytical conditioning). `DistributionView` for component access, `ConditionedComponent` for conditioning.
- **Workflows and broadcasting**: `WorkflowFunction` with automatic uncertainty propagation. Multi-backend broadcasting (`jax` vectorization, `loop`, `prefect` orchestration). Auto-detection of JAX traceability. Empirical enumeration with budget-aware cartesian product.
- **Bayesian inference**: `SimpleModel` with NUTS/HMC via TFP + auto-fallback to gradient-free RWMH. `Likelihood` base class. `IterativeForecaster` for sequential Bayesian updating. `ApproximateDistribution` with chain structure and diagnostics.
- **Provenance tracking**: Automatic lineage on all creation paths (transforms, broadcasting, conditioning, inference). `Provenance.to_dict()` / `from_dict()` serialization. `provenance_ancestors()` and `provenance_dag()` utilities.
- **Documentation**: MkDocs Material site with API reference, getting started guide, and 6 tutorial notebooks (distributions, transformations, joint distributions, broadcasting, autodiff, modular forecasting).
- **CI/CD**: GitHub Actions CI with pytest + coverage, Codecov integration, automated docs deployment to GitHub Pages.
