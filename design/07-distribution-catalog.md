# Part VII — The Distribution Catalog

Parts III, IV, and VI fixed what a distribution *is* and what the operations do to one. Part VII catalogs the **concrete families** the library ships: for each, its event kind, the capabilities it implements and how, and the way instances arise, whether by constructor or as the result of an operation. Every family here is an ordinary `Distribution` or `ConditionalDistribution`, and the catalog adds no new base classes.

| §     | Family                            | Event                                  | Factored?                          | Capabilities                                        | Arises by                                             |
| ----- | --------------------------------- | -------------------------------------- | ---------------------------------- | --------------------------------------------------- | ----------------------------------------------------- |
| VII.1  | parametric (`Normal`, …)          | one array field, or a fixed record     | no                                 | closed form throughout                              | constructor                                            |
| VII.2  | empirical, bootstrap, KDE         | any                                    | no                                 | sampling, sample moments, exact marginals           | constructor, or as a sampling result                   |
| VII.3  | mixture                           | the components' shared event           | no                                 | what the components jointly support                 | `mixture`, dependent marginals, or constructor      |
| VII.4  | evaluation results                | the map's output schema              | no                                 | per rule: exact density, exact moments, or sampling | `evaluate`                                             |
| VII.5  | random functions, random measures | a `FunctionSpec` or `DistributionSpec` event declaration | no                             | mean function / marginalized law, sampling          | constructor                                            |
| VII.6  | the Gaussian algebra              | numeric                                | yes | closed form, exact conditioning and marginals       | constructor, `*`, `condition_on`, linear `evaluate`    |
| VII.7  | inference-produced                | any                                    | as realized                        | whatever the realizing family supports              | `condition_on` (inference)                             |
| VII.8  | conditional families              | (given, event) pairs          | some                               | the conditional capabilities                        | constructor or composition                             |

## VII.1 — Parametric families

### Contract

A single backend adapter, `TFPDistribution`, implements the capability set on raw arrays, and every parametric family is a thin constructor over it: continuous (`Normal`, `Beta`, `Gamma`, `InverseGamma`, `Exponential`, `LogNormal`, `StudentT`, `Uniform`, `Cauchy`, `Laplace`, `HalfNormal`, `HalfCauchy`, `Pareto`, `TruncatedNormal`), discrete (`Bernoulli`, `Binomial`, `Poisson`, `Categorical`, `NegativeBinomial`), and multivariate (`MultivariateNormal`, `Dirichlet`, `Multinomial`, `Wishart`, `VonMisesFisher`). Each family derives its event term spec from its parameters, including shape, dtype, and support, and wraps it in the component declaration of II.2. The array event's component is the law's `name`, captured once at construction, unless `component_name` gives another (III.7). Each family auto-promotes to a `NumericDistribution`. The adapter is the only class that knows the backend exists, and its `raw()` is the wrapped backend distribution (II.4).

```python
class TFPDistribution(Distribution[Array]):
    def __init__(self, name: str, backend_dist: Any, *, component_name: str | None = None) -> None: ...   # the wrapped backend object
    # closed-form _sample, _log_prob, _mean, _variance, and _quantile;
    # _cov and _marginal where the family defines them

class Normal(TFPDistribution):
    def __init__(self, name: str, loc: ArrayLike, scale: ArrayLike, *,
                 component_name: str | None = None) -> None: ...
# and likewise for each family above: parameters in, event spec and capabilities derived
```

### Rationale

One adapter with thin family constructors keeps the backend a computational detail (`C3 – Computational detail hidden by default, available on demand`) and makes a new family a constructor rather than a class (`D2 – Generality first`).

## VII.2 — Empirical and resampling

### Contract

An `EmpiricalDistribution[T]` is a finite, possibly weighted set of atoms of any event type. It samples by weighted resampling, its moments are weighted sample estimates when the event is numeric, and its marginals are exact. Atoms are stored in the event type's native batch form, with the weights a parallel array. An explicit `event_spec` preserves component names and exposure form; atoms alone determine the returned term kind but cannot recover an independent whole-term component name. Without a declaration, record atoms expose their fields and any other atoms form a whole-term event under the law's `name` (III.7); a type hole is filled from the atoms. It doesn't support log probability calculations, since an empirical measure doesn't, in general, have a density.

Two bootstrap forms share one convention: the **source** may be any distribution implementing `SupportsSampling`, which covers the nonparametric bootstrap, where an empirical source is resampled, and the parametric bootstrap, where a fitted law is redrawn, in one interface; `replicate_size` defaults to the source's atom count when the source is empirical and is required otherwise.
- A `BootstrapReplicateDistribution` is the `replicate_size`-fold iid product of the source law: a draw is one **replicate**, `replicate_size` draws from the source in `T`'s batch form.
- A `BootstrapDistribution` is the corresponding random measure: a draw is the empirical measure of one replicate, an `EmpiricalDistribution`. The bootstrap distribution of a statistic is `evaluate(stat, ...)` over whichever form the statistic reads, a replicate dataset or a replicate measure. Replicate batches preserve the source event's term kind, and empirical measures built from replicates carry the source's complete event declaration. Their outer event declaration, for the batch-valued replicate or the measure-valued draw, is derived from the source and the replicate size and is distinct from the source's event interface; its component is the law's `name` unless `component_name` gives another. A new bootstrap or replicate object label never renames either interface.

A `KDEDistribution` smooths the atoms with a **smoothing kernel**: a mean-zero density `K` recentered at each atom and scaled by the bandwidth, so its law is the weighted mixture `Σᵢ wᵢ h⁻ᵈ K((x − xᵢ)/h)`. `SmoothingKernel` carries a uniform construction contract: `build_kernels(centers, scales)` returns the bank of placed copies, one per atom, whatever the concrete kernel, so the KDE holds the kernel class and never reads kernel-specific parameters. `bandwidth` accepts a value, the name of a selection rule such as `"scott"` or `"silverman"`, or `None` for the default rule, and is resolved before the copies are built. The bank supplies indexed sampling and per-copy log-densities with the scale Jacobian included. On the KDE, `_sample` draws an atom by weight and then a draw from that copy, exact for the KDE law, and `_log_prob` is the weighted log-sum-exp of the per-copy densities, also exact. The mean is the weighted atom mean, and the variance adds `h²` times the kernel's variance to the atoms' weighted sample variance. Numeric events only. Event completion follows `EmpiricalDistribution`: record atoms expose their fields, array atoms form a whole-term event under the law's `name` unless `event_spec` names the component otherwise, and every placed kernel carries the completed declaration.

```python
class EmpiricalDistribution[T](Distribution[T]):
    def __init__(self, name: str, atoms: Batch | Array, weights: Array | None = None, *,
                 event_spec: OutputSpec | None = None) -> None: ...
    # atoms are given in T's batch form; weights default to uniform

class BootstrapReplicateDistribution(Distribution):
    def __init__(self, name: str, source: SupportsSampling, replicate_size: int | None = None, *,
                 component_name: str | None = None) -> None: ...
    # a draw is one replicate in T's batch form: replicate_size iid draws from source

class BootstrapDistribution(Distribution):   # a random measure: a draw is an EmpiricalDistribution
    def __init__(self, name: str, source: SupportsSampling, replicate_size: int | None = None, *,
                 component_name: str | None = None) -> None: ...
    # the empirical measure of one replicate

class SmoothingKernel(ABC):                # a bank of mean-zero kernel copies, one per center
    @classmethod
    @abstractmethod
    def build_kernels(cls, centers: ArrayLike | NumericRecordBatch,
                      scales: ArrayLike | NumericRecord) -> SmoothingKernel: ...
    # the uniform constructor: one placed copy per center, scales broadcast over centers and coordinates
    @abstractmethod
    def _sample(self, key: Key, index: Array) -> Array: ...   # draws from the indexed copies
    @abstractmethod
    def _log_density(self, x: Array) -> Array: ...            # per-copy log-density at x, scale Jacobian included
class GaussianKernel(SmoothingKernel): ...
class EpanechnikovKernel(SmoothingKernel): ...

class KDEDistribution(Distribution[Array]):
    def __init__(self, name: str, atoms: Array | NumericRecordBatch, bandwidth: ArrayLike | str | None = None,
                 weights: Array | None = None, kernel: type[SmoothingKernel] = GaussianKernel, *,
                 event_spec: OutputSpec | None = None) -> None: ...
```

### Rationale

All four are genuine laws whose declared capabilities are those they can provide (`D1 – Mathematical fidelity`), and the empirical family is the closure family for sampling-based operations (`D4 – Closed system of objects under operations`). Accepting any `SupportsSampling` source makes the parametric bootstrap the same object as the nonparametric one (`D2 – Generality first`).

### Open points

- *Bandwidth shape.* Whether `bandwidth` admits a matrix / linear operator, with the kernel applied in the whitened space, is open.

## VII.3 — Mixtures

### Contract

A `MixtureDistribution` is a convex combination of component distributions over one shared event declaration, including names, kind, and packaging. Component object labels may differ. A rename or event transformation is explicit when their declarations differ. It implements `_sample` when all of its components do, and the same holds for `_log_prob` (as the weighted log-sum-exp). Moments combine componentwise when every component provides them: the mean is `Σ wᵢ mᵢ` and the covariance is `Σ wᵢ (Σᵢ + mᵢ mᵢᵀ) − m mᵀ`. It is what `mixture` returns for a finite mixing distribution, and the form a dependent joint's detached marginal takes under finite mixing.

```python
class MixtureDistribution(Distribution[T]):
    def __init__(self, name: str, components: Sequence[Distribution], weights: Array) -> None: ...
    # components share one event declaration; weights are nonnegative and sum to one
```

### Rationale

A mixture supports an operation exactly when its components do, the same intersection rule the factored classes use (`D3 – Capability-based operations`).

## VII.4 — Evaluation results

### Contract

Each evaluation rule returns a family from this catalog. A closed-form rule returns a parametric result, the linear-Gaussian case being a member of the Gaussian algebra. The generic linear rule returns a `LinearPushforwardDistribution`, which represents `A @ d` lazily when no family-specific rule applies. The change-of-variables rule returns a `BijectorTransformedDistribution`. The sampling fallback returns an `EmpiricalDistribution` over the pushed draws.

```python
class LinearPushforwardDistribution(Distribution):
    def __init__(self, name: str, base: Distribution, op: LinOp) -> None: ...
    # the law of op @ X for X ~ base; the event type is op's output type, under the pushforward's own component.
    # _sample pushes base draws through op; _mean and _cov delegate exactly,
    # E[A X] = A E[X] and Cov(A X) = A Cov(X) Aᵀ, lazily through the operator algebra;
    # _log_prob only when op is invertible, by change of variables

class BijectorTransformedDistribution(Distribution[T]):
    def __init__(self, name: str, base: Distribution, bijector: Function) -> None: ...
    # bijector must satisfy is_invertible and claim SupportsLogDetJacobian, checked at construction;
    # _sample pushes base draws through the bijector;
    # _log_prob(y) is the base log-density at the preimage minus the log-Jacobian determinant
```

### Rationale

Typing evaluation results as catalog families keeps the operation closed and its outputs operable (`D4 – Closed system of objects under operations`).

### Open points

- *Lazy sampling results.* The sampling rule materializes an `EmpiricalDistribution` with a fixed atom count. A lazy alternative that remains exactly samplable, drawing an input and applying the map on demand, would suit unbounded resampling such as bootstrap statistics; whether that is the sampling rule's result or an opt-in form is open.

## VII.5 — Random functions and random measures

### Contract

A `RandomFunction` is a distribution declaring a `FunctionSpec` as its event: a draw is a callable, `mean` returns the mean function, and `variance` returns the pointwise variance function when the family provides it. Calling it at a point returns a distribution over outputs, the law of `f(x)` for `f` drawn from the random function. A `RandomMeasure` is a distribution whose event is a `DistributionSpec` leaf: a draw is a `Distribution`, `mean` returns the marginalized law, and no event-typed variance is claimed in general. A draw's log-density is itself random, so `_random_log_prob()` returns the law of `x ↦ log D(x)`, a `RandomFunction`. A `BootstrapDistribution` is a member.

```python
class RandomFunction[X, Y](Distribution[Callable[[X], Y]]):
    def __call__(self, x: X) -> Distribution: ...       # the distribution over outputs at x

class RandomMeasure[T](Distribution[Distribution[T]]):
    def _random_log_prob(self) -> RandomFunction: ...   # the law of x ↦ log D(x) for D ~ M
```

### Rationale

Both are ordinary distributions over nonstandard event types, claiming only the moments those types support (`D1 – Mathematical fidelity`, `D3 – Capability-based operations`).

## VII.6 — The Gaussian algebra

### Contract

Three families form a closed algebra built on `LinOp`. A `MultivariateNormal` from the parametric families is the atomic member: its constructor accepts `cov: LinOp | Array`, a dense array wraps as a `DenseLinOp` whose domain is the event term spec and whose codomain carries the completed event declaration, and `_cov` returns the `LinOp` with its structure preserved. A `GaussianRandomFunction` is the random-function member: a `RandomFunction` whose finite-dimensional laws are Gaussian. A `FactoredMultivariateGaussian` is the factored joint whose factors are jointly Gaussian, with closed-form `log_prob`, moments, and sampling, and exact conditioning and marginals. It is derived, never constructed: `*` and `joint` return it as the most-specific class whenever every factor is a Gaussian or a linear-Gaussian conditional distribution, and its flat-coordinate pushforward is a `MultivariateNormal` obtained through the declared isomorphism of III.7. A converter may change its family while preserving the original event declaration (IV.3).

The algebra is closed under the operations: an affine pushforward of any member is again a member by a closed-form rule, and `condition_on` with a Gaussian prior and a linear-Gaussian observation is exact. A composition of Gaussian pieces built before its dimensions are bound is an ordinary factored object holding its covariances as recipes; once binding makes the `LinOp` covariances constructible, refinement re-derives the most-specific class and the object joins the algebra as a `FactoredMultivariateGaussian`.

```python
class FactoredMultivariateGaussian(FactoredNumericDistribution): ...   # derived by `*` / `joint`, never constructed
```

**The Gaussian random function.** A `GaussianRandomFunction` is abstract, covering any model with Gaussian predictions rather than Gaussian processes alone. A concrete member implements `predict_mean` and `predict_variance`, and `predict_covariance` when it supports joint evaluation; `__call__` assembles these into the exact finite-dimensional law, a `Normal` at a single point and a `MultivariateNormal` over stacked points when the covariance is available. These laws preserve the evaluated function's output component name independently of their distribution labels. The drawn function's output component and the function-valued event's component both default to the random function's `name`; `output_spec` names the former otherwise, and `component_name` the latter. A type hole in either is filled from the model, and evaluated shapes may stay symbolic until inputs bind them (II.1). Its `mean` is the mean function and its `variance` the pointwise variance function, the event-typed moments of a random function. A `GaussianProcess`, which is specified by a mean function and a covariance kernel, is the canonical member; a `LinearBasisFunction`, which is `f(x) = φ(x)ᵀw` with Gaussian weights `w`, is another. Conditioning on noisy linear observations of finitely many evaluations is exact and yields another `GaussianRandomFunction` as the posterior law, and shifts, scalings, output-side linear maps, and sums of independent members are again members by closed-form evaluation rules.

```python
class GaussianRandomFunction(RandomFunction[Array, Array], ABC):
    @abstractmethod
    def predict_mean(self, X: Array) -> Array: ...        # X stacks n input points
    @abstractmethod
    def predict_variance(self, X: Array) -> Array: ...    # marginal variance at each point
    def predict_covariance(self, X: Array) -> LinOp: ...  # joint covariance over the points, when supported
    def __call__(self, X: Array) -> Normal | MultivariateNormal: ...   # the finite-dimensional law at X

class GaussianProcess(GaussianRandomFunction):
    def __init__(self, name: str, mean_fn: Callable[[Array], Array],
                 cov_kernel: Callable[[Array, Array], Array], *,
                 output_spec: OutputSpec | None = None, component_name: str | None = None) -> None: ...

class LinearBasisFunction(GaussianRandomFunction):
    def __init__(self, name: str, basis: Callable[[Array], Array], weights: MultivariateNormal, *,
                 output_spec: OutputSpec | None = None, component_name: str | None = None) -> None: ...
    # f(x) = basis(x)ᵀ w; the covariance kernel is basis(x)ᵀ Σ_w basis(x′)
```

### Rationale

Gaussian closure under affine maps, conditioning, and marginalization is a mathematical fact, stated as class structure so that dispatch exploits it automatically (`D1 – Mathematical fidelity`, `C3 – Computational detail hidden by default, available on demand`).

## VII.7 — Inference-produced distributions

### Contract

An inference result is an ordinary member of whichever family realizes it: a variational posterior is a parametric or bijector-transformed family, an MCMC or ABC posterior is empirical, and an amortized posterior is a learned conditional evaluated at the data. Results preserve the target event's component names and packaging independently of their new object labels. What the results share is a record: each carries `provenance` naming the method, the target, and the inputs, and each exposes the capabilities its realizing family supports. Whether a result is exact or approximate, and relative to what, is read from that record.

### Rationale

Approximation is a relation between a result and its target: a variational Gaussian's density is exact for the law it *is*, and approximate only relative to the posterior it stands in for. A relation belongs in the record of how the result arose, so it is recorded in `provenance` (`C6 – Traceable and reproducible workflows`).

### Open points

- *Fidelity presentation.* II.7 fixes the recorded local guarantees and upstream history. A compact user-facing summary may be useful, but it must identify its target and derive from provenance rather than add an independent truth in `annotations`.
- *Approximation error.* Capturing a result's approximation error, for example a bound or a diagnostic, has no generic representation yet. For now it is stored in `annotations`, keyed by the producing method.

## VII.8 — Conditional families

### Contract

The conditional members of the catalog are `ConditionalDistribution`s, each fixed by its (given, event) pair.

- A **linear-Gaussian conditional distribution** is `s ↦ N(A @ s + b, Σ)` with `A` a `LinOp`. It is the conditional member of the Gaussian algebra: composed with a Gaussian prior it yields a `FactoredMultivariateGaussian`, and conditioning through it is exact.
- A **GLM likelihood** is assembled from a `GLMFamily`, a link, and the linear predictor. A `GLMFamily` is mean-parameterized: `build(name, mean, dispersion, component_name=...)` returns the law of conditionally independent observations, one per entry of `mean`, with `has_dispersion` declaring whether the family takes a dispersion parameter, such as a Gaussian scale. The likelihood's given slots are `X`, `β`, and the dispersion when the family has one, its event is the response vector, and its law is `family.build(name, link⁻¹(X @ β), dispersion, component_name=component_name)`, with the link defaulting to the family's canonical one: `GaussianFamily` with identity is linear regression, `BernoulliFamily` with logit is logistic regression, and `PoissonFamily` with log is Poisson regression. `X` and the dispersion may instead be supplied to `glm_likelihood`, which fixes them at construction as the exogenous curry of `condition_on` applied early. The response component is the likelihood's `name` unless `component_name` gives another, and it is passed through to the family. The pieces are the interface: changing the link or the family changes the likelihood without a new class.

```python
class LinearGaussianConditional(ConditionalDistribution):
    def __init__(self, name: str, A: LinOp, b: Array, cov: LinOp) -> None: ...
    # s ↦ N(A @ s + b, cov); the given slot is A's input slot; the event type is A's output type, under the kernel's own component

class GLMFamily(ABC):                     # a mean-parameterized response family
    canonical_link: Function              # invertible: is_invertible checked at construction
    has_dispersion: bool                  # whether build takes a dispersion, e.g. a Gaussian scale
    @abstractmethod
    def build(self, name: str, mean: Array, dispersion: ArrayLike | None = None, *,
              component_name: str | None = None) -> Distribution: ...
    # the law of len(mean) conditionally independent observations with the given means

class GaussianFamily(GLMFamily): ...      # canonical link: identity; dispersion: the scale
class BernoulliFamily(GLMFamily): ...     # canonical link: logit; no dispersion
class PoissonFamily(GLMFamily): ...       # canonical link: log; no dispersion

def glm_likelihood(name: str, family: GLMFamily, link: Function | None = None,
                   *, component_name: str | None = None, X: Array | None = None,
                   dispersion: ArrayLike | None = None) -> ConditionalDistribution: ...
    # shapes: X ("obs", "features"), β ("features",), y ("obs",); the dimensions are symbolic until X binds them
```

### Rationale

Assembling conditional families from uniform pieces is `D2 – Generality first`: a mean-parameterized family, a link bijector, and a linear predictor compose into an entire model class with nothing new defined.

## VII.9 — Program-defined families

### Contract

A **program-defined model** exposes what its backend provides. A program with a joint law over modeled variables, including modeled observations, may expose a `Distribution`. A program supplying a parameter target for given data exposes a `ConditionalDistribution` over those data inputs, or the data-bound `Distribution`. Data sizes, covariates, and arbitrary data-block entries are not automatically random event components.

`StanModel` uses BridgeStan and `PyMCModel` uses a PyMC model-building function. Each adapter declares its data inputs separately from the event variables, whose program names determine the output components. A model named `regression_model` may have the one-field event `OutputSpec(RecordSpec(beta=beta_spec))`; its draws remain records, and composition matches `beta`, not the model label. The existing Stan adapter's parameter-only event and separately supplied data follow the data-bound form; exposing an unbound or generative model requires the corresponding explicit declaration, not merely moving data into its event.

The adapter claims the density and sampling capabilities the program supplies. It declares unnormalized density unless normalization is established. Inference methods register against the backend interface they require; they do not require a public factor graph unless they use one (VI.6). A method records which data were bound, its target, controls, and local fidelity, and its result preserves the target event declaration (VII.7). An unconstrained parameterization is an explicit invertible map of that event (III.7, V.12).

```python
# Adapter contracts; constructors bind backend data separately from event variables.
class StanModel(Distribution): ...  # data-bound parameter target through BridgeStan
class PyMCModel(Distribution): ...  # data-bound target from a PyMC model-building function
# An adapter exposing unbound data implements ConditionalDistribution instead;
# a joint-law form requires an explicit generative contract over its modeled events.
```

### Rationale

A backend program participates through the interface it supplies (`D1 – Mathematical fidelity`). Registering its inference method by capability rather than requiring an exposed factor graph extends the same conditioning operation to opaque backend representations (`C1 – Uniform interface to functions, distributions, and values`, `D3 – Capability-based operations`).