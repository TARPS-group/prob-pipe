# Part VI — The Distribution Catalog

Parts III and V fixed what a distribution *is* and what the operations do to one. Part VI catalogs the **concrete families** the library ships: for each, its event kind, its place on the axes of the `Distribution` hierarchy, the capabilities it implements and how, and the way instances arise, whether by constructor or as the result of an operation. Every family here is an ordinary `Distribution` or `ConditionalDistribution`, and the catalog adds no new base classes.

## VI.0 — Overview: the catalog map

| §     | Family                            | Event                                  | Factored?                          | Capabilities                                        | Arises by                                             |
| ----- | --------------------------------- | -------------------------------------- | ---------------------------------- | --------------------------------------------------- | ----------------------------------------------------- |
| VI.1  | parametric (`Normal`, …)          | one array field, or a fixed record     | no                                 | closed form throughout                              | constructor                                            |
| VI.2  | empirical, bootstrap, KDE         | any                                    | no                                 | sampling, sample moments, exact marginals           | constructor, or as a sampling result                   |
| VI.3  | mixture                           | the components' shared event           | no                                 | what the components jointly support                 | `predictive`, dependent marginals, or constructor      |
| VI.4  | pushforward results               | the map's output template              | no                                 | per rule: exact density, exact moments, or sampling | `pushforward`                                          |
| VI.5  | random functions, random measures | a `FunctionSpec` or `DistributionSpec` leaf | no                             | mean function / marginalized law, sampling          | constructor                                            |
| VI.6  | the Gaussian algebra              | numeric                                | `FactoredMultivariateGaussian`: yes | closed form, exact conditioning and marginals       | constructor, `*`, `condition_on`, linear `pushforward` |
| VI.7  | inference-produced                | any                                    | as realized                        | whatever the realizing family supports              | `condition_on` (inference)                             |
| VI.8  | conditional families              | (given, event) template pairs          | some                               | the conditional capabilities                        | constructor or composition                             |

## VI.1 — Parametric families

### Contract

A single backend adapter, `TFPDistribution`, implements the capability set on raw arrays, and every parametric family is a thin constructor over it: continuous (`Normal`, `Beta`, `Gamma`, `InverseGamma`, `Exponential`, `LogNormal`, `StudentT`, `Uniform`, `Cauchy`, `Laplace`, `HalfNormal`, `HalfCauchy`, `Pareto`, `TruncatedNormal`), discrete (`Bernoulli`, `Binomial`, `Poisson`, `Categorical`, `NegativeBinomial`), and multivariate (`MultivariateNormal`, `Dirichlet`, `Multinomial`, `Wishart`, `VonMisesFisher`). Each family derives its `event_template` from its parameters, including shape, dtype, and the support `Constraint`, and auto-promotes to a `NumericDistribution`. The adapter is the only class that knows the backend exists.

```python
class TFPDistribution(Distribution[Array]):
    def __init__(self, name: str, backend_dist: Any) -> None: ...   # the wrapped backend object
    # closed-form _sample, _log_prob, _mean, _variance, and _quantile;
    # _cov and _marginal where the family defines them

class Normal(TFPDistribution):
    def __init__(self, name: str, loc: ArrayLike, scale: ArrayLike) -> None: ...
# and likewise for each family above: parameters in, template and capabilities derived
```

### Rationale

One adapter with thin family constructors keeps the backend a computational detail (`C3 – Computational detail hidden by default, available on demand`) and makes a new family a constructor rather than a class (`D2 – Generality first`).

## VI.2 — Empirical and resampling

### Contract

An `EmpiricalDistribution[T]` is a finite, possibly weighted set of atoms of any event type. It samples by weighted resampling, its moments are weighted sample estimates when the event is numeric, and its marginals are exact, the empirical distribution of the projected atoms. It implements neither log-prob capability, since an empirical measure has no density; `KDEDistribution` is the family to reach for when one is needed.

Two bootstrap forms share one convention: the **source** may be any distribution implementing `SupportsSampling`, which covers the nonparametric bootstrap (an empirical source, resampled) and the parametric bootstrap (a fitted law, redrawn) in one interface, with `replicate_size` defaulting to the source's atom count when the source is empirical and required otherwise.
- A `BootstrapReplicateDistribution` is the `replicate_size`-fold iid product of the source law: a draw is one **replicate**, `replicate_size` draws from the source in `T`'s batch form.
- A `BootstrapDistribution` is the corresponding random measure: a draw is the empirical measure of one replicate, an `EmpiricalDistribution`. The bootstrap distribution of a statistic is a pushforward, `pushforward(stat, ...)`, of whichever form the statistic reads, a replicate dataset or a replicate measure.

A `KDEDistribution` smooths the atoms with a kernel: its law is the weighted mixture of one kernel copy per atom, scaled by the bandwidth. `_sample` draws an atom by weight and then a draw from the kernel centered there, exact for the KDE law. `_log_prob` is the log of the weighted average of kernel densities, also exact, and its mean and variance are closed form when the kernel's are. Numeric events only.

```python
class EmpiricalDistribution[T](Distribution[T]):
    def __init__(self, name: str, atoms: Batch | Array, weights: Array | None = None) -> None: ...
    # atoms are given in T's batch form; weights default to uniform

class BootstrapReplicateDistribution(Distribution):
    def __init__(self, name: str, source: SupportsSampling, replicate_size: int | None = None) -> None: ...
    # a draw is one replicate in T's batch form: replicate_size iid draws from source

class BootstrapDistribution(Distribution):   # a random measure: a draw is an EmpiricalDistribution
    def __init__(self, name: str, source: SupportsSampling, replicate_size: int | None = None) -> None: ...
    # the empirical measure of one replicate

class KDEDistribution(Distribution[Array]):
    def __init__(self, name: str, atoms: Array | NumericRecordBatch, bandwidth: ArrayLike,
                 weights: Array | None = None, kernel: Distribution | None = None) -> None: ...
    # kernel defaults to a standard normal; the law is the bandwidth-scaled kernel mixture
```

### Rationale

All four are bona fide laws with honestly partial capabilities (`D1 – Mathematical fidelity`), and the empirical family is the closure family for sampling-based operations (`D4 – Closed system of objects under operations`). Accepting any `SupportsSampling` source makes the parametric bootstrap the same object as the nonparametric one (`D2 – Generality first`).

## VI.3 — Mixtures and predictives

### Contract

A `MixtureDistribution` is a convex combination of component distributions over one shared event template. `_sample` draws a component index by weight and then a component draw. `_log_prob` is the weighted log-sum-exp, present exactly when every component scores. Moments combine componentwise when every component provides them: the mean is `Σ wᵢ mᵢ` and the covariance is `Σ wᵢ (Σᵢ + mᵢ mᵢᵀ) − m mᵀ`. It is what `predictive` returns for a finite mixing distribution, and the general form of a dependent joint's detached marginal.

```python
class MixtureDistribution(Distribution[T]):
    def __init__(self, name: str, components: Sequence[Distribution], weights: Array) -> None: ...
    # components share one event_template; weights are nonnegative and sum to one
```

### Rationale

A mixture supports an operation exactly when its components do, the same intersection rule the factored classes use (`D3 – Capability-based operations`).

## VI.4 — Pushforward results

### Contract

Each pushforward rule returns a family from this catalog. A closed-form rule returns a parametric result, with the linear-Gaussian case landing in the Gaussian algebra. The change-of-variables rule returns a `BijectorTransformedDistribution`. The sampling fallback returns an `EmpiricalDistribution` over the pushed draws. Whatever the rule, a linear map's result delegates `mean` and `cov` exactly.

```python
class BijectorTransformedDistribution(Distribution[T]):
    def __init__(self, name: str, base: Distribution, bijector: Bijector) -> None: ...
    # _sample pushes base draws through the bijector;
    # _log_prob(y) is the base log-density at the preimage minus the log-Jacobian determinant
```

### Rationale

Typing pushforward results as catalog families keeps the operation closed and its outputs operable (`D4 – Closed system of objects under operations`).

### Open points

- *Lazy sampling pushforwards.* The sampling rule materializes an `EmpiricalDistribution` with a fixed atom count. A lazy alternative that remains exactly samplable, drawing an input and applying the map on demand, would suit unbounded resampling such as bootstrap statistics; whether that is the sampling rule's result or an opt-in form is open.

## VI.5 — Random functions and random measures

### Contract

A `RandomFunction` is a distribution whose event is a `FunctionSpec` leaf: a draw is a callable, `mean` returns the mean function, and `variance` returns the pointwise variance function when the family provides it. Calling it at a point is the evaluation pushforward, a distribution over outputs. A `RandomMeasure` is a distribution whose event is a `DistributionSpec` leaf: a draw is a `Distribution`, `mean` returns the marginalized law, and no event-typed variance is claimed in general. A draw's log-density is itself random, so `_random_log_prob()` returns the law of `x ↦ log D(x)`, a `RandomFunction`. A `BootstrapDistribution` is a member.

```python
class RandomFunction[X, Y](Distribution[Callable[[X], Y]]):
    def __call__(self, x: X) -> Distribution: ...       # the evaluation pushforward at x

class RandomMeasure[T](Distribution[Distribution[T]]):
    def _random_log_prob(self) -> RandomFunction: ...   # the law of x ↦ log D(x) for D ~ M
```

### Rationale

Both are ordinary distributions over nonstandard event types, claiming exactly the moments those types support (`D1 – Mathematical fidelity`, `D3 – Capability-based operations`).

## VI.6 — The Gaussian algebra

### Contract

Three families form a closed algebra built on `LinOp`. A `MultivariateNormal`, from the parametric families, is the atomic member: its constructor accepts `cov: LinOp | Array`, a dense array wraps as a `DenseLinOp`, and `_cov` returns the `LinOp` with its structure preserved. A `GaussianRandomFunction` is the abstract random-function member, covering any model with Gaussian predictions rather than Gaussian processes alone: a concrete subclass implements the predictive mean and marginal variance, and the joint predictive covariance when it supports joint evaluation. A `FactoredMultivariateGaussian` is the factored joint whose factors are jointly Gaussian, with closed-form `log_prob`, moments, and sampling, and exact conditioning and marginals. It is derived, never constructed: `*` and `joint` return it as the most-specific class whenever every factor is a Gaussian or a linear-Gaussian conditional distribution, and an exact conversion to `MultivariateNormal` over the flat event is registered with the converter registry.

The algebra is closed under the operations: composing Gaussian factors with linear-Gaussian conditional distributions yields a `FactoredMultivariateGaussian`, an affine pushforward of any member is again a member by a closed-form rule, and `condition_on` with a Gaussian prior and a linear-Gaussian observation is exact.

```python
class GaussianRandomFunction(RandomFunction[Array, Array], ABC):
    @abstractmethod
    def predict_mean(self, X: Array) -> Array: ...        # X stacks n input points
    @abstractmethod
    def predict_variance(self, X: Array) -> Array: ...    # marginal variance at each point
    def predict_covariance(self, X: Array) -> LinOp: ...  # joint covariance over the points, when supported
    def __call__(self, X: Array) -> Normal | MultivariateNormal: ...
    # the exact finite-dimensional law at the stacked points, joint when predict_covariance is available

class FactoredMultivariateGaussian(FactoredNumericDistribution): ...
# derived by `*` / `joint` when every factor is Gaussian or linear-Gaussian;
# log_prob, moments, sampling, conditioning, and marginals are all exact
```

### Rationale

Gaussian closure under affine maps, conditioning, and marginalization is a mathematical fact, stated as class structure so that dispatch exploits it automatically (`D1 – Mathematical fidelity`, `C3 – Computational detail hidden by default, available on demand`).

## VI.7 — Inference-produced distributions

### Contract

An inference result is an ordinary member of whichever family realizes it: a variational posterior is a parametric or bijector-transformed family, an MCMC or ABC posterior is empirical, and an amortized posterior is a learned conditional evaluated at the data. What the results share is a record: each carries `provenance` naming the method, the target, and the inputs, and each exposes exactly the capabilities its realizing family supports. Whether a result is exact or approximate, and relative to what, is read from that record.

### Rationale

Approximation is a relation between a result and its target: a variational Gaussian's density is exact for the law it *is*, and approximate only relative to the posterior it stands in for. A relation belongs in the record of how the result arose, so it lives in `provenance` (`C6 – Traceable and reproducible workflows`).

### Open points

- *Tagging approximate results.* `provenance` records how a result arose, and a lighter tag on top may be worth adding, either an `is_approximate` flag or a convention within `annotations`. Any such tag would span tracked terms generally, since conditional distributions and linear operators can be approximate too.
- *Approximation error.* Capturing a result's approximation error (a bound, a diagnostic, a fitted estimate) has no generic representation yet. For now it rides in `annotations`, keyed by the producing method.

## VI.8 — Conditional families

### Contract

The conditional members of the catalog are `ConditionalDistribution`s, each fixed by its (given, event) template pair.

- A **linear-Gaussian conditional distribution** is `s ↦ N(A @ s + b, Σ)` with `A` a `LinOp`. It is the conditional member of the Gaussian algebra: composed with a Gaussian prior it yields a `FactoredMultivariateGaussian`, and conditioning through it is exact.
- A **GLM likelihood** is assembled from pieces the catalog already has: an affine map of the given through a `LinOp`, an inverse link that is a `Bijector`, and a parametric family taking the result as its parameter. `(X, β) ↦ Normal(X @ β, σ²)` is linear regression, `Bernoulli(logistic(X @ β))` is logistic regression, and `Poisson(exp(X @ β))` is Poisson regression. The pieces are the interface: changing the link or the family changes the likelihood without a new class.
- An **amortized likelihood** is a learned conditional density, exposing conditional sampling and scoring as trained, with the fit recorded in `provenance`.

```python
class LinearGaussianConditional(ConditionalDistribution):
    def __init__(self, name: str, A: LinOp, b: Array, cov: LinOp) -> None: ...
    # s ↦ N(A @ s + b, cov); given and event templates derived from A's input and output templates

def glm_likelihood(name: str, family: Callable[[Array], Distribution], link: Bijector) -> ConditionalDistribution: ...
    # given fields X and β; the law of the response is family(link⁻¹(X @ β))
```

### Rationale

Assembling conditional families from the catalog's own pieces is `D2 – Generality first`: a `LinOp`, a `Bijector`, and a parametric family compose into an entire model class with nothing new defined.
