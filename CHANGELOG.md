# Changelog

All notable changes to ProbPipe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- **`.n`** property on ``RecordDistribution`` — ``prod(batch_shape)``
  (STYLE_GUIDE §1.9).
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
