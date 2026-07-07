# Part VI — The Distribution Catalog

Parts III and V fixed what a distribution *is* and what the operations do to one. Part VI catalogs the **concrete families** the library ships: for each, its event kind, its place on the axes of the `Distribution` hierarchy, the capabilities it implements and how, and the way instances arise, whether by constructor or as the result of an operation. Every family here is an ordinary `Distribution` or `ConditionalDistribution`, and the catalog adds no new base classes.

## VI.0 — Overview: the catalog map

| §     | Family                            | Event                                  | Factored?                          | Capabilities                                        | Arises by                                             |
| ----- | --------------------------------- | -------------------------------------- | ---------------------------------- | --------------------------------------------------- | ----------------------------------------------------- |
| VI.1  | parametric (`Normal`, …)          | one array field, or a fixed record     | no                                 | closed form throughout                              | constructor                                            |
| VI.2  | empirical, bootstrap, KDE         | any                                    | no                                 | sampling, empirical density, sample moments         | constructor, or as a sampling result                   |
| VI.3  | mixture                           | the components' shared event           | no                                 | what the components jointly support                 | `predictive`, dependent marginals, or constructor      |
| VI.4  | pushforward results               | the map's output template              | no                                 | per rule: exact density, exact moments, or sampling | `pushforward`                                          |
| VI.5  | the Gaussian algebra              | numeric                                | `FactoredMultivariateGaussian`: yes | closed form, exact conditioning and marginals       | constructor, `*`, `condition_on`, linear `pushforward` |
| VI.6  | random functions, random measures | a `FunctionSpec` or `DistributionSpec` leaf | no                             | mean function / marginalized law, sampling          | constructor                                            |
| VI.7  | inference-produced                | any                                    | as realized                        | whatever the realizing family supports              | `condition_on` (inference)                             |
| VI.8  | conditional families              | (given, event) template pairs          | some                               | the conditional capabilities                        | constructor or composition                             |

## VI.1 — Parametric families

### Contract

A single backend adapter, `TFPDistribution`, implements the capability set on raw arrays, and every parametric family is a thin constructor over it: continuous (`Normal`, `Beta`, `Gamma`, `InverseGamma`, `Exponential`, `LogNormal`, `StudentT`, `Uniform`, `Cauchy`, `Laplace`, `HalfNormal`, `HalfCauchy`, `Pareto`, `TruncatedNormal`), discrete (`Bernoulli`, `Binomial`, `Poisson`, `Categorical`, `NegativeBinomial`), and multivariate (`MultivariateNormal`, `Dirichlet`, `Multinomial`, `Wishart`, `VonMisesFisher`). Each family derives its `event_template` from its parameters, including shape, dtype, and the support `Constraint`, and auto-promotes to a `NumericDistribution`. All of `_sample`, `_log_prob`, `_mean`, `_variance`, and `_quantile` are closed form, `_cov` is provided where covariance is defined, and the multivariate families implement `SupportsMarginals` exactly. The adapter is the only class that knows the backend exists.

### Rationale

One adapter with thin family constructors is `D2 – Generality first`: a new family is a constructor, not a class tower, and the backend is a computational detail behind the capability set (`C3 – Computational detail hidden by default, available on demand`). Deriving the template from the parameters is `D5 – Explicit, carried structure`, and array-backed families keep every operation differentiable (`D6 – Differentiability where possible`).

## VI.2 — Empirical and resampling

### Contract

An `EmpiricalDistribution[T]` is a finite, possibly weighted sample set over any event type: `_sample` resamples by weight, `_log_prob` is the density with respect to the counting measure on the atoms, moments are weighted sample estimates when the event is numeric, and `SupportsMarginals` is exact, since the marginal of an empirical distribution is the empirical distribution of the projected atoms. Two refinements share the representation: `BootstrapDistribution` resamples with replacement to produce replicate draws, and `KDEDistribution` smooths the atoms with a kernel, trading the atomic density for a continuous `_log_prob` while keeping exact sampling.

### Rationale

The empirical measure is a bona fide probability measure, so the family is an ordinary `Distribution` with honestly limited capabilities (`D1 – Mathematical fidelity`). It is also the closure family for sampling-based operations: a sampling pushforward, a Monte Carlo marginal, and a simulation result are all empirical, so returning one keeps the system closed (`D4 – Closed system of objects under operations`).

## VI.3 — Mixtures and predictives

### Contract

A `MixtureDistribution` is a convex combination of component distributions over one shared event template, with weights or a mixing distribution over the component index. `_sample` draws a component and then a draw, `_log_prob` is the weighted log-sum-exp when every component scores, and a moment combines the components' moments when every component provides them. It is what `predictive` returns for a finite mixing distribution, and the general form of a dependent joint's detached marginal.

### Rationale

A mixture supports an operation exactly when its components do, the same intersection rule the factored classes use (`D3 – Capability-based operations`). Naming the family keeps the predictive and marginal operations closed with an inspectable result rather than an anonymous wrapper (`D4 – Closed system of objects under operations`).

## VI.4 — Pushforward results

### Contract

Each pushforward rule returns a family from this catalog. A closed-form rule returns a parametric result, with the linear-Gaussian case landing in the Gaussian algebra. The change-of-variables rule returns a `BijectorTransformedDistribution`, which carries the base distribution and the `Bijector`, has an exact `_log_prob` through the log-determinant of the Jacobian, and samples by pushing base draws forward. The sampling fallback returns an `EmpiricalDistribution` over the pushed draws. Whatever the rule, a linear map's result delegates `mean` and `cov` exactly.

### Rationale

Typing the results as catalog families rather than opaque wrappers keeps `pushforward` closed and its outputs operable (`D4 – Closed system of objects under operations`), and each result records the rule that produced it, so the exactness of what came back is never a guess (`D1 – Mathematical fidelity`).

## VI.5 — The Gaussian algebra

### Contract

Three families form a closed algebra built on `LinOp`. A `MultivariateNormal` is the atomic member, its covariance a `LinOp` carrying the distribution's templates on both sides. A `FactoredMultivariateGaussian` is a factored joint whose factors are jointly Gaussian: it carries `SupportsFactors`, and every operation is exact, with conditioning on any produced field by Gaussian conditioning, Gaussian marginals, and closed-form `log_prob`, moments, and sampling. A `GaussianRandomFunction` is the random-function member: its `mean` is the mean function, its covariance is a kernel over pairs of inputs, and evaluation at any finite set of points is a `MultivariateNormal`.

The algebra is closed under the operations. Composing Gaussian factors with linear-Gaussian conditional distributions yields a `FactoredMultivariateGaussian`, an affine pushforward of any member is again a member by a closed-form rule, and `condition_on` with a Gaussian prior and a linear-Gaussian observation is exact.

### Rationale

Gaussian closure under affine maps, marginalization, and conditioning is a mathematical fact, and the catalog states it as class-level structure so that dispatch exploits it automatically (`D1 – Mathematical fidelity`, `C3 – Computational detail hidden by default, available on demand`). Building every covariance on `LinOp` keeps the structure lazy and typed by the event templates (`D5 – Explicit, carried structure`), so exactness composes instead of being reimplemented case by case.

## VI.6 — Random functions and random measures

### Contract

A `RandomFunction` is a distribution whose event is a `FunctionSpec` leaf: a draw is a callable, `mean` returns the mean function, and `variance` returns the pointwise variance function when the family provides it. Evaluating a random function at a point pushes it forward through evaluation, giving a distribution over outputs, and a `GaussianRandomFunction` returns exact Gaussians. A `RandomMeasure` is a distribution whose event is a `DistributionSpec` leaf: a draw is a `Distribution`, `mean` returns the marginalized law, and no event-typed `variance` is claimed in general. Because a draw's log-density is itself random, the random-measure capabilities `SupportsRandomLogProb` and `SupportsRandomUnnormalizedLogProb` return the law of the log-density function, itself a `RandomFunction`.

### Rationale

Both families are ordinary distributions over nonstandard event types, which is exactly what capability gating exists for: each claims the moments its event type mathematically supports and no more (`D1 – Mathematical fidelity`, `D3 – Capability-based operations`). Returning the log-density's law as a random function keeps even the density interface closed over the catalog (`D4 – Closed system of objects under operations`).

## VI.7 — Inference-produced distributions

### Contract

There is no approximate-distribution class, since any family can arise as an approximation. An inference result is an ordinary member of whichever family realizes it: a variational posterior is a parametric or bijector-transformed family, an MCMC or ABC posterior is empirical, and an amortized posterior is a learned conditional evaluated at the data. What the results share is not a type but a record: each carries `provenance` naming the method, the target, and the inputs, and each exposes exactly the capabilities its realizing family supports. Whether a result is exact or approximate, and relative to what, is read from that record.

### Rationale

Approximation is a relation between a result and its target, not a property of the law itself: a variational Gaussian's density is exact for the law it *is*, and approximate only relative to the posterior it stands in for. Reifying that relation as a class would encode a non-mathematical distinction (`D1 – Mathematical fidelity`), so the relation lives in `provenance` instead (`C6 – Traceable and reproducible workflows`), and honest capabilities come from the realizing family (`D3 – Capability-based operations`).

## VI.8 — Conditional families

### Contract

The conditional members of the catalog are `ConditionalDistribution`s, each fixed by its (given, event) template pair.

- A **linear-Gaussian conditional distribution** is `s ↦ N(A @ s + b, Σ)` with `A` a `LinOp`. It is the conditional member of the Gaussian algebra: composed with a Gaussian prior it yields a `FactoredMultivariateGaussian`, and conditioning through it is exact.
- A **GLM likelihood** is assembled from pieces the catalog already has: an affine map of the given through a `LinOp`, an inverse link that is a `Bijector`, and a parametric family taking the result as its parameter. `(X, β) ↦ Normal(X @ β, σ²)` is linear regression, `Bernoulli(logistic(X @ β))` is logistic regression, and `Poisson(exp(X @ β))` is Poisson regression. The templates are derived from the pieces, and the pieces are the interface: changing the link or the family changes the likelihood without a new class.
- An **amortized likelihood** is a learned conditional density, exposing conditional sampling and scoring as trained, with the fit recorded in `provenance`.

### Rationale

Assembling conditional families from the catalog's own pieces rather than hand-writing each is `D2 – Generality first` at the family level: the GLM likelihoods demonstrate that a `LinOp`, a `Bijector`, and a parametric family compose into an entire model class with nothing new defined. Exact members and learned members sit in one vocabulary, dispatched by capability rather than by how they were obtained (`D3 – Capability-based operations`).
