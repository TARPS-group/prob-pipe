# ProbPipe – Claude Code Instructions

## Workflow Preferences

### Git & Branch Management
- Create feature branches from `origin/main` using naming conventions: `feat/`, `dev/`, `fix/`
- Always fetch and pull main before creating a new branch
- After PR merge: pull main in parent repo, detach HEAD in worktree, delete local and remote branch, prune remote refs
- Commit messages: concise, imperative, focused on "why" not "what". Use HEREDOCs for multi-line
- Never amend commits; always create new ones
- Push immediately after committing unless told otherwise
- When creating PRs, use `gh pr create` with `--head` flag (worktree may not track remote correctly)

### Development Flow
- Work is done in a git worktree at `.claude/worktrees/suspicious-lalande/`
- The virtualenv is at the parent repo: `.venv/` (use full path to `.venv/bin/python`)
- Run tests with: `.venv/bin/python -m pytest tests/ -x -q`
- Run notebooks with: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace <path>`
- After modifying code, always run relevant tests before committing
- After modifying notebooks, re-run them to regenerate clean outputs (notebooks store outputs in JSON)
- The user prefers to review plans before implementation for non-trivial changes
- The user expects code to be validated (run, tested) before committing – don't commit untested code
- When told "PR merged, cleanup" → pull main, delete branch locally+remotely, prune refs

### Code Style
- Use en-dashes (–) not double-dashes (--) in prose/markdown
- Use "ProbPipe" (CamelCase) in prose; "prob-pipe" only for URLs, paths, and package references
- Don't add unnecessary wrapper functions – call methods directly when the wrapper adds no value
- Use `probpipe._utils.prod()` for shape-product computations (handles empty tuples, returns Python int)
- Docstrings should reflect current state of code, not historical development
- Bold key terms on first use in documentation

### Communication
- Be direct; don't preface with "I'll continue from where we left off"
- When the user says "fix all" or "proceed", just do it without asking for confirmation
- When asked to "audit" something, provide a prioritized list (high/medium/low)
- Propose changes before implementing when asked to "propose" or "suggest"
- When the user gives feedback mid-implementation, incorporate it immediately

## ProbPipe Architecture

### Core Design Philosophy
ProbPipe is a Python framework for building probabilistic pipelines with automated uncertainty quantification. The organizing principle is **distributions in, distributions out**: every node can consume and emit probability distributions.

Five design requirements:
1. Reusable inferential components (modular, swappable statistical units)
2. Interoperability with the Python ecosystem (scipy, TFP, sklearn)
3. End-to-end uncertainty propagation (broadcasting)
4. Seamless scalability (JAX vectorization + Prefect orchestration)
5. Provenance and reproducibility (lineage tracking)

### Technology Stack
- **JAX** as the sole array backend (no numpy/scipy dependency in core code)
- **TensorFlow Probability (TFP)** JAX substrate for distribution implementations
- **TFP Nightly** (`tfp-nightly`) is required – stable TFP doesn't support recent JAX versions
- JAX defaults to float32; do NOT enable `jax_enable_x64` globally (conflicts with TFP)
- Tests may use numpy for assertions (`np.testing.assert_allclose`) but library code must not

### Package Structure
- `probpipe/core/distribution.py` – Base distribution hierarchy (Distribution, ArrayDistribution, PyTreeArrayDistribution, EmpiricalDistribution, Provenance, constraints)
- `probpipe/core/node.py` – `WorkflowFunction` and broadcasting logic
- `probpipe/core/modeling.py` – `MCMCSampler`, `RWMH`, `Likelihood`, `IterativeForecaster`
- `probpipe/core/provenance.py` – DAG visualization and ancestor traversal
- `probpipe/distributions/` – Concrete distribution implementations (continuous.py, discrete.py, multivariate.py, joint.py, transformed.py, random_function.py, gaussian_random_function.py)
- `probpipe/converters/` – Automatic type conversion registry (_registry.py, _probpipe.py, _tfp.py, _scipy.py)
- `probpipe/linalg/` – Linear operator algebra (all JAX, no scipy)
- `probpipe/_array_utils.py` – Array canonicalization utilities (internal)
- `probpipe/_utils.py` – Shared utilities (`prod()` for shape products, internal)
- `docs/examples/` – Jupyter notebooks (01–09)

### Key Technical Concepts

#### Generalized Distribution Hierarchy
The distribution system uses a layered generic hierarchy:
- `Distribution[T]` – root base class, generic over return type T. Provides `_sample()`, optional `log_prob()`, `expectation()`, `from_distribution()` (delegates to converter registry)
- `PyTreeArrayDistribution[T]` – distributions over pytrees of arrays with shared `batch_shape`, per-leaf `event_shapes`, `event_size`, `flatten_value()`/`unflatten_value()`, `as_flat_distribution()`
- `ArrayDistribution` – single-array specialization with full TFP shape semantics (formerly just `Distribution`)
- `FlattenedView` – wraps a `PyTreeArrayDistribution` as a flat `ArrayDistribution`
- `BootstrapDistribution` – MC error tracking (not an `ArrayDistribution`)

#### TFP Shape Semantics
ArrayDistributions follow: `sample_shape + batch_shape + event_shape`

#### Broadcasting
When a `WorkflowFunction` node receives a `Distribution` where it expects a concrete value, it automatically draws samples and evaluates the function for each, returning an `EmpiricalDistribution`. The `"auto"` backend probes JAX traceability and picks the fastest strategy.

#### Provenance
`Provenance` dataclass with `operation`, `parents`, `metadata`. Attached via `with_source()` (write-once). Supports `to_dict(recurse=True)` serialization and `provenance_dag()`/`provenance_ancestors()` visualization.

#### Expectations and Error Tracking
- `Distribution.expectation(f)` computes E[f(X)]
- For finite-support distributions (Bernoulli, Categorical): exact computation, returns `Array`
- For infinite-support: Monte Carlo, returns `BootstrapDistribution` by default (captures MC error)
- `@monte_carlo` decorator wraps `mean()`, `variance()`, `cov()` with automatic MC fallback
- `BootstrapDistribution` represents a distribution over sample means via bootstrap resampling
- `return_dist=False` to get a plain array even for MC estimates

#### Unnormalized Log Prob
- `Distribution.unnormalized_log_prob()` defaults to `log_prob()`; subclasses can override
- `SequentialJointDistribution`: `log_prob()` works when conditioned set forms a root sub-graph (Markov structure makes normalized conditional tractable); raises `NotImplementedError` otherwise
- `unnormalized_log_prob()` sums all components (including conditioned ones) and is always available

#### Converter Registry
- Global `converter_registry` handles bidirectional conversion: ProbPipe ↔ TFP ↔ scipy.stats
- `converter_registry.check(source, target)` → `ConversionInfo` (method, cost, feasibility)
- `converter_registry.convert(source, target)` performs the conversion
- `converter_registry.is_distribution_type(obj)` checks if obj is any recognized distribution type
- Cross-family conversions use moment-matching via `source.mean()` and `source.variance()` directly
- Same-class conversions return the source unchanged (no copy, no provenance)
- `WorkflowFunction` auto-converts external distribution types (TFP, scipy) before broadcasting

#### Joint Distributions
- Joint distributions return **pytrees (dicts)**, supporting flat and nested structures
- `joint["x"]` → `DistributionView`; `joint["physics", "force"]` → nested path access
- `joint["physics"]` → sub-joint `ProductDistribution` (marginal over subtree)
- `DistributionView` stores parent reference and key path; multiple views of the same parent are sampled jointly in broadcasts (preserves correlation)
- Component structure uses JAX canonical ordering (sorted dict keys, depth-first traversal)

#### Random Functions
- `RandomFunction[X, Y]` extends `Distribution[Callable[[X], Y]]` – distribution over functions f: X → Y
- `ArrayRandomFunction` – specialization for X = Array, Y = Array with `input_shape`/`output_shape`
- Calling `rf(X)` returns a `Distribution` over outputs (the finite-dimensional predictive distribution)
- `_sample()`/`sample()` raise `NotImplementedError` by default (infinite-dimensional functions can't be sampled directly)
- `GaussianRandomFunction` – abstract subclass with Gaussian predictive distributions; supports algebraic operations: `A @ grf`, `grf + b`, `alpha * grf`, `grf1 + grf2`
- `LinearBasisFunction` – concrete: `f(x) = a + Phi(x) @ w`, `w ~ N(m, C)`
- `EmulatorMixin` (in `probpipe/surrogate/`) – mixin adding `fit()`, `update()`, `training_inputs`, `training_responses` to any random function
- Shape contract: `joint_inputs`/`joint_outputs` flags control batch vs event partitioning

#### MCMCSampler
- Uses TFP NUTS/HMC with automatic fallback to gradient-free RW-MH
- RW-MH uses JAX PRNG (not numpy) for consistency
- `_get_init_state` uses `jnp.atleast_1d` to handle scalar priors
- Don't force float32 in init_state – let dtype propagate from prior/data

### Testing
- ~980+ tests, target 90% coverage
- `codecov.yml` allows 2% threshold decrease, 70% patch target
- Test fixtures for MCMCSampler should use explicit `dtype=jnp.float32` (TFP operates in float32)
- Tests can use numpy for assertions but library code must use JAX

### Documentation
- MkDocs with Material theme, deployed to GitHub Pages via `.github/workflows/docs.yml`
- API docs use mkdocstrings with Python handler
- 9 example notebooks in `docs/examples/` (01–08, plus 07_random_functions)
- Suppress TFP deprecation warnings in notebooks with `warnings.filterwarnings` in import cell
- Graphviz may not be installed – use `provenance_ancestors()` + `to_dict()` instead of `provenance_dag()`
