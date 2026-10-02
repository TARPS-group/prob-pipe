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

A single backend adapter, `TFPDistribution`, implements the capability set on raw arrays, and every parametric family is a thin constructor over it: continuous (`Normal`, `Beta`, `Gamma`, `InverseGamma`, `Exponential`, `LogNormal`, `StudentT`, `Uniform`, `Cauchy`, `Laplace`, `HalfNormal`, `HalfCauchy`, `Pareto`, `TruncatedNormal`), discrete (`Bernoulli`, `Binomial`, `Poisson`, `Categorical`, `NegativeBinomial`), and multivariate (`MultivariateNormal`, `Dirichlet`, `Multinomial`, `Wishart`, `VonMisesFisher`). Each family derives its event term spec from its parameters, including shape, dtype, and support, and wraps it in the component declaration of II.2. Parameters with more axes than one law needs give one law whose extra leading axes are event axes of independent coordinates: `Normal("y", jnp.zeros(3), 1.0)` draws a vector of three independent coordinates, and a `MultivariateNormal` whose `loc` has shape `(n, d)` draws an `(n, d)` array of n independent rows. Separate laws form a `DistributionBatch` (III.10). A family's claims are the capabilities it computes in closed form, recorded in a class-level table the adapter reads (III.8): the backend's methods, `SupportsQuantile` where the backend has a quantile, and the closed forms the backend lacks: the per-coordinate quantiles of `MultivariateNormal`, and the variance of `VonMisesFisher` as the diagonal of its covariance. A moment that diverges is the extended real ±∞, returned as `inf`, as the mean of a `HalfCauchy` is, and only an undefined moment, such as the mean of a `Cauchy`, raises `MathematicalDomainError` (II.7). The array event's component defaults to the law's `name`, captured once at construction. An `event_spec` declaration names another, usually with its type pending, and the family completes it with `with_spec` (II.2), so `Normal("prior", 0.0, 1.0, event_spec=OutputSpec(beta=None))` is labeled `prior` and exports `beta`. Each family auto-promotes to a `NumericDistribution`. The adapter is the only class that knows the backend exists, and its `raw()` is the wrapped backend distribution (II.4).

```python
class TFPDistribution(Distribution):
    def __init__(self, name: str, backend_dist: Any, *, event_spec: OutputSpec | None = None) -> None: ...   # the wrapped backend object
    # closed-form _sample, _log_prob, _mean, _variance, and _quantile;
    # _cov and _marginal where the family defines them

class Normal(TFPDistribution):
    def __init__(self, name: str, loc: ArrayLike, scale: ArrayLike, *,
                 event_spec: OutputSpec | None = None) -> None: ...
# and likewise for each family above: parameters in, event spec and capabilities derived
```

### Rationale

One adapter with thin family constructors keeps the backend a computational detail (`C3 – Computational detail hidden by default, available on demand`) and makes a new family a constructor rather than a class (`D2 – Generality first`).

## VII.2 — Empirical and resampling

### Contract

An `EmpiricalDistribution` is a finite, possibly weighted set of atoms of any event type. It samples by weighted resampling, its moments are weighted sample estimates when the event is numeric, and its marginals are exact. Atoms are stored in the event type's native batch form, with the weights a parallel array. A batch of atoms keeps its own levels, every batch axis indexing atoms in row-major order. A plain array indexes atoms along its leading axis, on one level named by `level`, which defaults to the law's component; `level` is refused with a batch, whose levels `with_level_names` renames. An explicit `event_spec` preserves component names and packaging, and atoms alone determine the returned term kind. Without a declaration, record atoms expose their fields and any other atoms form a whole-term event whose component defaults to the law's `name` (III.7); a type hole is filled from the atoms. It claims no log-density, since an empirical measure in general has no density.

Two bootstrap forms share one convention: the **source** may be any distribution implementing `SupportsSampling`, which covers the nonparametric bootstrap, where an empirical source is resampled, and the parametric bootstrap, where a fitted law is redrawn, in one interface; `replicate_size` defaults to the source's atom count when the source is empirical and is required otherwise. A replicate's draws lie on one level named by `level`. It defaults to the source's atom level when the source is an empirical law with exactly one, and otherwise to the source's component when the source's event is a whole term; a source that exposes a record of several components requires it. A replicate of a dataset therefore keeps the dataset's level, so a statistic written for the data applies unchanged to every replicate.
- A `BootstrapReplicateDistribution` is the `replicate_size`-fold iid product of the source law: a draw is one **replicate**, `replicate_size` draws from the source in the event's batch form.
- A `BootstrapDistribution` is the corresponding random measure: a draw is the empirical measure of one replicate, an `EmpiricalDistribution`. The bootstrap distribution of a statistic is `evaluate(stat, ...)` over whichever form the statistic reads, a replicate dataset or a replicate measure. Replicate batches preserve the source event's term kind, and empirical measures built from replicates carry the source's complete event declaration. Their outer event declaration, for the batch-valued replicate or the measure-valued draw, is derived from the source and the replicate size and is distinct from the source's event interface; its component defaults to the law's `name`, and an `event_spec` declaration names another.

A `KDEDistribution` smooths the atoms of a numeric event with a **smoothing kernel**: a mean-zero density `K` recentered at each atom and scaled by the bandwidth, so its law is the weighted mixture `Σᵢ wᵢ h⁻ᵈ K((x − xᵢ)/h)`. `SmoothingKernel` carries a uniform construction contract: `build_kernels(centers, scales)` returns the bank of placed copies, one per atom, whatever the concrete kernel, so the KDE holds the kernel class and never reads kernel-specific parameters. The scales broadcast against centers of shape `(n, *event)` by NumPy's rules, so a per-center scalar on an event of shape `(d,)` has shape `(n, 1)`. Records enter through their flat vectors (II.3): a `NumericRecordBatch` of centers as `(n, d)` and a `NumericRecord` of scales as `(d,)`. `bandwidth` accepts a value, the name of a selection rule such as `"scott"` or `"silverman"`, or `None` for the default rule, which is Scott's, and is resolved before the copies are built. Scott's rule is `hⱼ = n_eff^(-1/(d+4)) σⱼ` and Silverman's is `hⱼ = (4/(d+2))^(1/(d+4)) n_eff^(-1/(d+4)) σⱼ`, with `σⱼ` the weighted standard deviation of coordinate `j` and `n_eff = (Σwᵢ)²/Σwᵢ²` Kish's effective sample size, so the rules stay sensible under importance weights. A coordinate whose atoms all agree has `σⱼ = 0`, and either rule then raises `ValueError` asking for an explicit bandwidth. The bank supplies indexed sampling and per-copy log-densities with the scale Jacobian included. On the KDE, `_sample` draws an atom by weight and then a draw from that copy, exact for the KDE law, and `_log_prob` is the weighted log-sum-exp of the per-copy densities, also exact. The mean is the weighted atom mean, and the variance adds `h²` times the kernel's variance to the atoms' weighted sample variance. Event completion follows `EmpiricalDistribution`: record atoms expose their fields, array atoms form a whole-term event whose component `event_spec` names or else defaults to the law's `name`, and every placed kernel carries the completed declaration.

```python
class EmpiricalDistribution(Distribution):
    def __init__(self, name: str, atoms: Batch | Array, weights: Array | None = None, *,
                 level: str | None = None, event_spec: OutputSpec | None = None) -> None: ...
    # atoms are given in the event's batch form; weights default to uniform;
    # a plain array's atoms lie on level, which defaults to the law's component
    @property
    def atoms(self) -> Batch | Array: ...    # the stored atoms, in the event's batch form
    @property
    def weights(self) -> Array: ...          # normalized, one per atom, row-major over the batch axes
    @property
    def num_atoms(self) -> int: ...

class BootstrapReplicateDistribution(Distribution):
    def __init__(self, name: str, source: SupportsSampling, replicate_size: int | None = None, *,
                 level: str | None = None, event_spec: OutputSpec | None = None) -> None: ...
    # a draw is one replicate in the event's batch form: replicate_size iid draws from source

class BootstrapDistribution(Distribution):   # a random measure: a draw is an EmpiricalDistribution
    def __init__(self, name: str, source: SupportsSampling, replicate_size: int | None = None, *,
                 level: str | None = None, event_spec: OutputSpec | None = None) -> None: ...
    # the empirical measure of one replicate

class SmoothingKernel(ABC):                # a bank of mean-zero kernel copies, one per center
    variance: ClassVar[float]                  # the unit kernel's variance per coordinate
    @classmethod
    @abstractmethod
    def build_kernels(cls, centers: ArrayLike | NumericRecordBatch,
                      scales: ArrayLike | NumericRecord) -> SmoothingKernel: ...
    # the uniform constructor: one placed copy per center
    @abstractmethod
    def _sample(self, key: Key, index: Array) -> Array: ...   # (*index.shape, *event) draws from the indexed copies
    @abstractmethod
    def _log_density(self, x: Array) -> Array: ...            # (*batch, n) for x of shape (*batch, *event): one per copy
class GaussianKernel(SmoothingKernel): ...
class EpanechnikovKernel(SmoothingKernel): ...   # the product kernel ∏ⱼ ¾(1 − uⱼ²); variance 1/5

class KDEDistribution(Distribution):
    def __init__(self, name: str, atoms: Array | NumericRecordBatch, bandwidth: ArrayLike | str | None = None,
                 weights: Array | None = None, kernel: type[SmoothingKernel] = GaussianKernel, *,
                 event_spec: OutputSpec | None = None) -> None: ...
```

### Rationale

All four are laws whose declared capabilities are those they can provide (`D1 – Mathematical fidelity`), and the empirical family is the closure family for sampling-based operations (`D4 – Closed system of objects under operations`). Accepting any `SupportsSampling` source makes the parametric bootstrap the same object as the nonparametric one (`D2 – Generality first`).

### Open points

- *Bandwidth shape.* Whether `bandwidth` admits a matrix / linear operator, with the kernel applied in the whitened space, is open.

## VII.3 — Mixtures

### Contract

A `MixtureDistribution` is a convex combination of component distributions over one shared event declaration, including names, kind, and packaging, so components whose declarations differ are renamed or transformed explicitly first. It implements `_sample` when all of its components do, and the same holds for `_log_prob` (as the weighted log-sum-exp). Moments combine componentwise when every component provides them: the mean is `Σ wᵢ mᵢ` and the covariance is `Σ wᵢ (Σᵢ + mᵢ mᵢᵀ) − m mᵀ`. It is what `mixture` returns for a finite mixing distribution, and the form a dependent joint's detached marginal takes under finite mixing.

```python
class MixtureDistribution(Distribution):
    def __init__(self, name: str, components: Sequence[Distribution], weights: Array) -> None: ...
    # components share one event declaration; weights are nonnegative and sum to one
```

### Rationale

A mixture supports an operation exactly when its components do, the same intersection rule the factored classes use (`D3 – Capability-based operations`).

## VII.4 — Evaluation results

### Contract

Each evaluation rule returns a family from this catalog. A closed-form rule returns a parametric result, the linear-Gaussian case being a member of the Gaussian algebra. The generic linear rule returns a `LinearPushforwardDistribution`, which represents `A @ d` lazily when no family-specific rule applies. The change-of-variables rule returns a `BijectorTransformedDistribution`. The sampling fallback returns an `EmpiricalDistribution` over the pushed draws.

`BijectorTransformedDistribution` is the catalog's one transformed family. A backend bijector enters as a `Function` that claims `SupportsInverse` and `SupportsLogDetJacobian` (III.3). The law claims a moment only where the bijector gives it in closed form, as an affine map does, and the moment operations estimate the others by their Monte Carlo fallback (VI.5).

```python
class LinearPushforwardDistribution(Distribution):
    def __init__(self, name: str, base: Distribution, op: LinOp) -> None: ...
    # the law of op @ X for X ~ base; the event type is op's output type, under the pushforward's own component.
    # _sample pushes base draws through op; _mean and _cov delegate exactly,
    # E[A X] = A E[X] and Cov(A X) = A Cov(X) Aᵀ, lazily through the operator algebra;
    # _log_prob only when op is invertible, by change of variables

class BijectorTransformedDistribution(Distribution):
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

A `RandomFunction` is a distribution declaring a `FunctionSpec` as its event: a draw is a callable, `mean` returns the mean function, and `variance` returns the pointwise variance function when the family provides it. Calling it at a point returns a distribution over outputs, the law of `f(x)` for `f` drawn from the random function. A `RandomMeasure` is a distribution whose event is a `DistributionSpec` leaf: a draw is a `Distribution`, `mean` returns the marginalized law, and no event-typed variance is claimed in general. A draw's log-density is itself random: a random measure that can compute it claims `SupportsRandomLogProb` (III.8), whose `_random_log_prob()` returns the law of `x ↦ log D(x)`, a `RandomFunction`. A `BootstrapDistribution` is a member.

```python
class RandomFunction(Distribution):
    def __call__(self, x: Any) -> Distribution: ...     # the distribution over outputs at x

class RandomMeasure(Distribution): ...   # a law whose draws are laws; SupportsRandomLogProb where computable
```

### Rationale

Both are ordinary distributions over nonstandard event types, claiming only the moments those types support (`D1 – Mathematical fidelity`, `D3 – Capability-based operations`).

## VII.6 — The Gaussian algebra

### Contract

Three families form a closed algebra built on `LinOp`. A `MultivariateNormal` from the parametric families is the atomic member: its constructor accepts `cov: LinOp | Array`, a dense array wraps as a `DenseLinOp` whose domain is the event term spec and whose codomain carries the completed event declaration, and `_cov` returns the `LinOp` with its structure preserved. A `GaussianRandomFunction` is the random-function member: a `RandomFunction` whose finite-dimensional laws are Gaussian. A `FactoredMultivariateGaussian` is the factored joint whose factors are jointly Gaussian, with closed-form `log_prob`, moments, and sampling, and exact conditioning and marginals. It is derived: `*` and `joint` return it as the most-specific class whenever every factor is a Gaussian or a linear-Gaussian conditional distribution, and its flat-coordinate pushforward is a `MultivariateNormal` obtained through the declared isomorphism of III.7. A converter may change its family while preserving the original event declaration (IV.3).

The algebra is closed under the operations: an affine pushforward of any member is again a member by a closed-form rule, and `condition_on` with a Gaussian prior and a linear-Gaussian observation is exact. A composition of Gaussian pieces built before its dimensions are bound is an ordinary factored object holding its covariances as recipes; once binding makes the `LinOp` covariances constructible, refinement re-derives the most-specific class and the object joins the algebra as a `FactoredMultivariateGaussian`.

```python
class FactoredMultivariateGaussian(FactoredNumericDistribution): ...   # derived by `*` / `joint`, never constructed
```

**The Gaussian random function.** A `GaussianRandomFunction` is abstract, covering any model with Gaussian predictions rather than Gaussian processes alone. A concrete member implements `predict_mean` and `predict_variance`, and `predict_covariance` when it supports joint evaluation; `__call__` assembles these into the exact finite-dimensional law, a `Normal` at a single point and a `MultivariateNormal` over stacked points when the covariance is available. These laws preserve the evaluated function's output component name independently of their distribution labels. The drawn function's output component and the function-valued event's component both default to the random function's `name`; `output_spec` names the former otherwise, and `event_spec` the latter. A type hole in either is filled from the model, and evaluated shapes may stay symbolic until inputs bind them (II.1). Its `mean` is the mean function and its `variance` the pointwise variance function, the event-typed moments of a random function. A `GaussianProcess`, which is specified by a mean function and a covariance kernel, is the canonical member; a `LinearBasisFunction`, which is `f(x) = φ(x)ᵀw` with Gaussian weights `w`, is another. Conditioning on noisy linear observations of finitely many evaluations is exact and yields another `GaussianRandomFunction` as the posterior law, and shifts, scalings, output-side linear maps, and sums of independent members are again members by closed-form evaluation rules.

```python
class GaussianRandomFunction(RandomFunction, ABC):
    @abstractmethod
    def predict_mean(self, X: Array) -> Array: ...        # X stacks n input points
    @abstractmethod
    def predict_variance(self, X: Array) -> Array: ...    # marginal variance at each point
    def predict_covariance(self, X: Array) -> LinOp: ...  # joint covariance over the points, when supported
    def __call__(self, X: Array) -> Normal | MultivariateNormal: ...   # the finite-dimensional law at X

class GaussianProcess(GaussianRandomFunction):
    def __init__(self, name: str, mean_fn: Callable[[Array], Array],
                 cov_kernel: Callable[[Array, Array], Array], *,
                 output_spec: OutputSpec | None = None, event_spec: OutputSpec | None = None) -> None: ...

class LinearBasisFunction(GaussianRandomFunction):
    def __init__(self, name: str, basis: Callable[[Array], Array], weights: MultivariateNormal, *,
                 output_spec: OutputSpec | None = None, event_spec: OutputSpec | None = None) -> None: ...
    # f(x) = basis(x)ᵀ w; the covariance kernel is basis(x)ᵀ Σ_w basis(x′)
```

### Rationale

Gaussian closure under affine maps, conditioning, and marginalization is a mathematical fact, stated as class structure so that dispatch exploits it automatically (`D1 – Mathematical fidelity`, `C3 – Computational detail hidden by default, available on demand`).

## VII.7 — Inference-produced distributions

### Contract

An inference result is an ordinary member of whichever family realizes it: a variational posterior is a parametric or bijector-transformed family, an MCMC or ABC posterior is empirical, and an amortized posterior is a learned conditional evaluated at the data. Results preserve the target event's component names and packaging independently of their new object labels. What the results share is a record: each carries `provenance` naming the method, the target, and the inputs, and each exposes the capabilities its realizing family supports. A result built from a run's draws also carries `method`, the name by which the `method` control selects the inference method that produced them, such as `blackjax_nuts`. Whether a result is exact or approximate, and relative to what, is read from that record.

An **amortized posterior** is a learned `ConditionalDistribution` q(θ | y) whose given slot is the observation and whose event is the parameters. An explicit training call, `learn_amortized_posterior`, returns it, and conditioning it on an observation evaluates it without retraining. It claims `SupportsApproximateConditioning`, since its evaluation stands in for the posterior of the joint it was trained on, so `exact_only=True` excludes it, and provenance names that joint as its target. Its laws sample, so `condition_on` returns them without further inference (VI.6). A learned likelihood p̂(y | θ), which `learn_amortized_likelihood` returns, is a `ConditionalDistribution` with a density, and a learned likelihood ratio, which `learn_amortized_ratio` returns, is one with an unnormalized density. Conditioning either one's composition with a prior on an observation is Bayes' rule, which the normalization stage completes by inference (VI.6). The three learned kernels' classes are private: each is obtained from its learner, and code that inspects one reads its capabilities.

A learned kernel claims each capability that its trained network computes exactly for the learned law. An amortized posterior therefore claims the conditional density when its network is a coupling flow, whose density is the flow's log-density with the bijectors' log-determinants. It claims no density when its network is a flow-matching model, whose density needs an ODE integration, or a consistency model, which gives none. A learned likelihood from a conditional flow claims sampling from that flow, whose draws agree with its density, and a learned likelihood ratio claims no sampling.

### Rationale

Approximation is a relation between a result and its target: a variational Gaussian's density is exact for the law it *is*, and approximate only relative to the posterior it stands in for. A relation belongs in the record of how the result arose, so it is recorded in `provenance` (`C6 – Traceable and reproducible workflows`).

### Open points

- *Fidelity presentation.* II.7 fixes the recorded local guarantees and upstream history. A compact user-facing summary may be useful, but it must identify its target and derive from provenance rather than add an independent truth in `annotations`.
- *Approximation error.* Capturing a result's approximation error, for example a bound or a diagnostic, has no generic representation yet. For now it is stored in `annotations`, keyed by the producing method.

## VII.8 — Conditional families

### Contract

The conditional members of the catalog are `ConditionalDistribution`s, each fixed by its (given, event) pair.

- A **linear-Gaussian conditional distribution** is `s ↦ N(A @ s + b, Σ)` with `A` a `LinOp`. It is the conditional member of the Gaussian algebra: composed with a Gaussian prior it yields a `FactoredMultivariateGaussian`, and conditioning through it is exact.
- A **GLM likelihood** is assembled from a `GLMFamily`, a link, and the linear predictor. A `GLMFamily` is mean-parameterized: `build(name, mean, dispersion, event_spec=...)` returns the law of conditionally independent observations, one per entry of `mean`, with `has_dispersion` declaring whether the family takes a dispersion parameter, such as a Gaussian scale. The likelihood's given slots are `X`, `beta`, and `dispersion` when the family has one, its event is the response vector, and its law is `family.build(name, link⁻¹(X @ beta), dispersion, event_spec=event_spec)`, with the link defaulting to the family's canonical one: `GaussianFamily` with identity is linear regression, `BernoulliFamily` with logit is logistic regression, and `PoissonFamily` with log is Poisson regression. The dispersion is a positive scalar, as a Gaussian scale is, and a heteroscedastic family declares a per-observation slot of its own. `X` and the dispersion may instead be supplied to `glm_likelihood`, which fixes them at construction as the exogenous curry of `condition_on` applied early. The response component defaults to the likelihood's `name`, and an `event_spec` declaration naming another is passed through to the family. The pieces are the interface: changing the link or the family changes the likelihood without a new class.

```python
class LinearGaussianConditional(ConditionalDistribution):
    def __init__(self, name: str, A: LinOp, b: Array, cov: LinOp) -> None: ...
    # s ↦ N(A @ s + b, cov); the given slot is A's input slot; the event type is A's output type, under the kernel's own component

class GLMFamily(ABC):                     # a mean-parameterized response family
    canonical_link: Function              # invertible: is_invertible checked at construction
    has_dispersion: bool                  # whether build takes a dispersion, e.g. a Gaussian scale
    @abstractmethod
    def build(self, name: str, mean: Array, dispersion: ArrayLike | None = None, *,
              event_spec: OutputSpec | None = None) -> Distribution: ...
    # the law of len(mean) conditionally independent observations with the given means

class GaussianFamily(GLMFamily): ...      # canonical link: identity; dispersion: the scale
class BernoulliFamily(GLMFamily): ...     # canonical link: logit; no dispersion
class PoissonFamily(GLMFamily): ...       # canonical link: log; no dispersion

def glm_likelihood(name: str, family: GLMFamily, link: Function | None = None,
                   *, event_spec: OutputSpec | None = None, X: Array | None = None,
                   dispersion: ArrayLike | None = None) -> ConditionalDistribution: ...
    # shapes: X ("obs", "features"), beta ("features",), y ("obs",); the dimensions are symbolic until X binds them
```

### Rationale

Assembling conditional families from uniform pieces is `D2 – Generality first`: a mean-parameterized family, a link bijector, and a linear predictor compose into an entire model class with nothing new defined.

## VII.9 — Program-defined families

### Contract

A **program-defined model** exposes the law its program defines, in the kind that law has. A program that models its observations as well as its parameters defines a joint law and exposes a `Distribution`, or a `ConditionalDistribution` over the inputs it does not model. A program that supplies a parameter target for given data exposes a `ConditionalDistribution` whose given slots are its data and whose laws are the targets. In both kinds, conditioning on data is `condition_on`, which returns a normalized law (VI.6).

- `StanModel` is a `ConditionalDistribution` through BridgeStan. Its given slots are the program's data-block variables, which the program does not divide into sizes, covariates, and observations. Each given slot carries the numeric spec of its variable's declared type, and a size that names a data variable is a symbolic dimension that the given slots share with the parameter record (III.9). Its event is the parameter record, whose fields carry their dtypes and the supports their declared constraints state, and a constraint that no ProbPipe support states, such as an ordered vector's, leaves its field's support undeclared. The adapter reads each data variable's and each parameter's element type and rank from `stanc --info` at construction, and their sizes and constraints from their declarations, so it declares both sides before any data are bound. It claims `SupportsConditionalUnnormalizedLogProb` alone, from BridgeStan's log density in the constrained parameterization without the Jacobian, so binding the data curries it to the unnormalized posterior, which `condition_on` normalizes with a method such as Stan's NUTS. Data given at construction curry the program early, and a construction that binds every data variable returns the unnormalized posterior as a `Distribution`. That posterior's class is private: the Stan methods dispatch on it, and users obtain the posterior through `StanModel`.
- `PyMCModel` is the joint law that a PyMC model-building function defines over its free variables, the parameters and the observed variables alike. An argument that the function passes as an observed variable's `observed` value is an event field, and any other argument is a given slot, so a model with covariates is a `ConditionalDistribution` over them. The adapter builds the model with its defaults to tell them apart: an argument that defaults to `None` and names a free variable of that build is observed, an argument without a default is a given slot, and an argument that defaults to `None` and names no free variable raises `ValueError`. An observed variable's shape is the one the model function declares through `shape` or `dims`, and the shape of the build without data stands in where the model declares none. Each free variable carries its dtype and the support its transform states, such as the positive reals for a log transform, and a transform that no ProbPipe support states leaves the support undeclared. A `PyMCModel` with given slots is a kernel of a private class, which the constructor returns, as `StanModel`'s constructor returns a `Distribution` once every data variable is bound. Its laws are `PyMCModel` instances with the covariates bound, and its event dimensions are symbolic, since the covariates may set them. A `PyMCModel` claims sampling, which draws from the prior predictive, and a normalized density; an instance containing a potential or an improper prior claims the unnormalized density instead. Conditioning on observed values is Bayes' rule (VI.6).
- `StanModel` and `PyMCModel` are exported from `probpipe` with the other families, and each imports its backend on first use, so neither backend is needed to import the package.
- `UnnormalizedDistribution` is the law of a user-supplied unnormalized log-density over a declared event. It claims `SupportsUnnormalizedLogProb` alone, and `sample`, `convert`, and `condition_on` normalize it through the inference-method registry (VI.3, VI.6, VI.10).

A program's variable names determine its output components: a model named `regression_model` may have the one-field event `OutputSpec(RecordSpec(beta=beta_spec))`, whose draws remain records, and composition matches `beta`, not the model label. Inference methods register against the backend interface they require (VI.6). A method records which data were bound, its target, its controls, and its local fidelity, and its result preserves the target event declaration (VII.7). An unconstrained parameterization is an explicit invertible map of that event (III.7, V.12), and the unconstrained form of a Stan target claims BridgeStan's log density with the Jacobian.

```python
class StanModel(ConditionalDistribution):
    def __init__(self, name: str, stan_file: str, *, data: Mapping[str, Any] | None = None) -> None: ...
    # given: the data-block variables that data leaves unbound; event: the parameter record

class PyMCModel(Distribution):
    def __init__(self, name: str, model_fn: Callable[..., Any]) -> None: ...
    # event: the free variables; the conditional form when model_fn takes an argument that no observed variable receives

class UnnormalizedDistribution(Distribution):
    def __init__(self, name: str, log_density: Callable[[Any], Array], event_spec: OutputSpec) -> None: ...
    # claims SupportsUnnormalizedLogProb alone
```

### Rationale

A backend program participates through the interface it supplies (`D1 – Mathematical fidelity`). A Stan program's data block makes no distinction between covariates and observations, so exposing the program as the kernel from its data to its posterior target needs no declaration beyond the program. Normalization by `condition_on` then returns the posterior from one call (`C3 – Computational detail hidden by default, available on demand`). Registering its inference method by capability rather than requiring an exposed factor graph extends the same conditioning operation to opaque backend representations (`C1 – Uniform interface to functions, distributions, and values`, `D3 – Capability-based operations`).
