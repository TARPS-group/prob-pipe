# Bayesian inference extensions for ProbPipe — plan

Status: **draft for review.** No implementation yet.

This document follows the plan-first convention described in
[CONTRIBUTING.md § PR Workflow](./CONTRIBUTING.md). Goal: agree on
the integration shape, library choices, and phased rollout *before*
any code lands.

## 1. Motivation

ProbPipe's inference registry currently ships gradient-based MCMC
(TFP NUTS / HMC) plus gradient-free RWMH; optionally nutpie / cmdstan
/ pymc when those backends are installed. For NN-scale parameter
spaces (deep emulators, amortized models, GP hyperparameter
posteriors with thousands of inducing points, …) we need four more
families:

| Family | Why ProbPipe needs it | Status today |
|---|---|---|
| Laplace approximation | Cheap parametric posterior; popular for BNN UQ since [Daxberger et al. 2021](https://arxiv.org/abs/2106.14806), used alongside deep ensembles, MC dropout, SWAG, and HMC | none |
| SGMCMC (SGLD / SGHMC / …) | Full-batch HMC doesn't scale; SGMCMC handles minibatched targets | none |
| VI (mean-field / full-rank / Pathfinder / SVGD) | Fast parametric posteriors and warm-starts for full-rank MCMC | none |
| SMC (tempered / adaptive-tempered / persistent-sampling) | Multimodal posteriors, model-evidence estimation, smooth annealing from prior to posterior | none |

We do **not** need to re-implement these from scratch — but we *do*
need to choose carefully which pieces we depend on vs. own.

## 2. Library survey

| Library | Active? | Coverage | Verdict |
|---|---|---|---|
| [**BlackJAX**](https://github.com/blackjax-devs/blackjax) v1.5 (Apr 2026) | very active | NUTS/HMC/MALA/MCLMC, **SGLD/SGHMC/SGNHT/csgld**, **tempered_smc/adaptive_tempered_smc/persistent_sampling_smc**, **meanfield_vi/fullrank_vi/pathfinder/SVGD**, Laplace-preconditioned MCMC | **Primary dependency.** Single library covers MCMC + SGMCMC + SMC + VI + Pathfinder. |
| [laplax](https://github.com/laplax-org/laplax) v0.0.2 (Jul 2025) | active | Full / diag / low-rank / GGN Laplace; Lanczos eigensolvers | **Not used.** Speaks `(model_fn, params, data, loss)` — NN-flavoured. Doesn't fit `SupportsLogProb` cleanly; adapter would fight the abstraction. See § 5.1. |
| [posteriors](https://github.com/normal-computing/posteriors) | active | Laplace, SGMCMC, VI, EKF | **Not used.** PyTorch-native; would require a JAX↔torch bridge. Out of scope; revisit if a concrete user need surfaces with a torch model. |
| [SGMCMCJax](https://github.com/jeremiecoullon/SGMCMCJax) | **inactive** (last release Aug 2023, [Snyk: inactive](https://snyk.io/advisor/python/sgmcmcjax)) | SGLD, SGHMC, SVRG-Langevin | Skip — superseded by BlackJAX SGMCMC. |
| [JaxSGMC](https://github.com/tummfm/jax-sgmc) | newer (2024 paper) | SGLD, SGHMC, reSGLD, AMAGOLD, SGGMC | Niche; covered by BlackJAX. |
| [NumPyro](https://github.com/pyro-ppl/numpyro) (SVI + AutoGuide) | very active | AutoNormal, AutoLaplaceApproximation, AutoMultivariateNormal | **Not used.** Mature but invasive: requires expressing the model in NumPyro's PPL idiom. ProbPipe users shouldn't have to learn a second modeling DSL just to fit a posterior. |
| [Fortuna (AWS)](https://github.com/awslabs/fortuna) | maintained | Laplace, SGMCMC, ensembles, conformal | Built on JAX/Flax; opinionated UQ-workflow framing that doesn't fit `Distribution`. |

## 3. Recommendation

| Concern | Choice |
|---|---|
| **MCMC** (NUTS / HMC) | **Migrate** from TFP-on-JAX to BlackJAX. Phased: register both, benchmark, flip default, eventually drop TFP. |
| **SGMCMC** (SGLD / SGHMC / SGNHT) | **BlackJAX**, fed by a generic ProbPipe random-measure abstraction (§ 4). |
| **VI** (mean-field / full-rank / Pathfinder / SVGD) | **BlackJAX**. |
| **SMC** (tempered / adaptive-tempered / persistent-sampling) | **BlackJAX**, consuming the same `(log_prior_fn, log_likelihood_fn)` decomposition produced by `split_log_density` (§ 4.7). |
| **Laplace** (full / diag / Lanczos low-rank) | **Native** ProbPipe implementation in `probpipe/inference/_laplace.py`. |
| **PyTorch interop** | Out of scope. Revisit if/when concrete user demand. |

The result: **one external dependency** for the heavy algorithm
work (BlackJAX) plus a small native module owned by ProbPipe (Laplace).

## 4. Stochastic targets as random measures

The integration surface for stochastic-gradient inference is one that
ProbPipe **already has**. A minibatched log-density is exactly an
instance of the existing `RandomMeasure[T]` abstraction
(`probpipe/core/_random_measures.py`) plus the existing
`SupportsRandomUnnormalizedLogProb` protocol
(`probpipe/core/protocols.py`). One concrete subclass covers the
minibatched case; the same machinery covers any other "stochastic
target" variant — tempered targets, doubly stochastic estimators, ABC
/ synthetic likelihoods, debiased truncations.

### 4.1 The abstraction

`probpipe/core/_random_measures.py` defines

```python
class RandomMeasure[T](Distribution[Distribution[T]]):
    """A distribution over distributions on T."""
```

A draw $D \sim M$ from a `RandomMeasure[T]` is itself a
`Distribution[T]`. The accompanying protocol

```python
@runtime_checkable
class SupportsRandomUnnormalizedLogProb(Protocol):
    def _random_unnormalized_log_prob(
        self,
    ) -> RandomFunction[T, LogProb]: ...
```

returns a `RandomFunction[T, LogProb]` — a function-valued random
variable whose realisations are unnormalized log-density callables
$\log\tilde{D}(\cdot)$.

That **is the contract every stochastic-gradient method needs**:

1. Take one realisation from the random function (≡ sample one
   `Distribution[T]` from the random measure).
2. Differentiate the realisation's log-density wrt the parameters.

Steps 1+2 produce one unbiased gradient estimate; SGMCMC steps with
it; repeat. No new abstraction is required.

### 4.2 The parameter type is `Record`

Every `SimpleModel`-based posterior in ProbPipe carries
`Record`-shaped parameters: the prior is a
`Distribution[Record]` (TFP-backed scalars and `ProductDistribution`
both produce `Record` samples — even a single-parameter prior is a
single-field Record), and the inference target is therefore a
distribution on `Record`.

So the random measure is **`RandomMeasure[Record]`**, the inner
realisations are **`Distribution[Record]`**, and the random
log-density is a **`RandomFunction[Record, LogProb]`**. This
keeps named parameter access (`posterior["theta"]`,
`posterior["sigma"]`) all the way through, with no flatten /
unflatten round-trip in the inference loop.

> **Footnote.** A model whose prior is a bare `Distribution[Array]`
> (no Record wrapper) would parameterise the abstraction at
> `RandomMeasure[Array]` — i.e., a `NumericRandomMeasure` per the
> existing hierarchy. This is the rare case in practice; `SimpleModel`
> always lands at Record. The Array variant falls out trivially if
> needed.

### 4.3 `MinibatchedLogDensity` as a `RandomMeasure[Record]`

For a model with prior $p(\theta)$ and likelihood
$p(\mathcal{D} \mid \theta) = \prod_i p(d_i \mid \theta)$, where
$\theta$ is a `Record`, the **minibatch random measure** is the
random distribution $D_B$ whose unnormalized density is

$$\log\tilde{D}_B(\theta) = \log p(\theta) + \frac{N}{b}\sum_{d \in B} \log p(d \mid \theta),$$

where $B \subset \mathcal{D}$ is a uniform random size-$b$ subset of
the data.

The $N/b$ rescaling makes
$\mathbb{E}_B[\log\tilde{D}_B(\theta)] = \log p_{\text{full}}(\theta)$
*and*
$\mathbb{E}_B[\nabla\log\tilde{D}_B(\theta)] = \nabla\log p_{\text{full}}(\theta)$.
Note: $\mathbb{E}_B[D_B(A)] \ne D_{\text{full}}(A)$ in general
(expectation does not commute with $\exp$); `mean(measure)` is a
*mixture* of per-minibatch measures, not the full-data target. This
matters when reasoning about which ops are useful — see § 4.6.

Concrete class:

```python
# probpipe/inference/_minibatch.py
class MinibatchedLogDensity(
    RandomMeasure[Record],
    SupportsSampling,
    SupportsRandomUnnormalizedLogProb,
):
    """Random measure realised by uniform minibatching.

    Wraps a ``SupportsLogProb`` model + dataset. A draw from this
    measure is the unnormalized rescaled posterior on the model's
    parameters (a ``Record``) implied by one random minibatch
    B ⊂ D of size b.
    """

    def __init__(
        self,
        model: SupportsLogProb,
        data: ArrayLike | RecordArray,
        batch_size: int,
        *,
        with_replacement: bool = False,
        rescale: bool = True,
        name: str | None = None,
    ):
        # ``data`` must be indexable along its leading axis. Both
        # ArrayLike and RecordArray satisfy this. A scalar Record
        # (no leading axis) is rejected.
        ...

    # -- RandomMeasure interface -----------------------------------------

    def _sample(self, key, sample_shape: tuple[int, ...] = ()):
        """Draw one (or many) realised minibatched-posterior
        distribution(s) from the random measure."""
        if sample_shape == ():
            indices = self._draw_indices(key)
            return _MinibatchPosterior(
                model=self._model,
                batch=_index_along_leading(self._data, indices),
                rescale_factor=(
                    (self._N / self._batch_size) if self._rescale else 1.0
                ),
            )
        # else: stack sample_shape draws into a DistributionArray
        ...

    # -- SupportsRandomUnnormalizedLogProb interface ---------------------

    def _random_unnormalized_log_prob(self) -> RandomFunction[Record, LogProb]:
        """The function-valued random variable θ ↦ log~D_B(θ)."""
        return _MinibatchRandomLogDensity(self)

    # -- Properties ------------------------------------------------------

    @property
    def data_size(self) -> int: ...
    @property
    def batch_size(self) -> int: ...
```

The inner one-realisation distribution `_MinibatchPosterior` (a
private helper) implements `SupportsUnnormalizedLogProb` and is
JAX-pytreeable so the SGMCMC kernel JIT-traces it cleanly.

### 4.4 Backends consume the abstraction, not the concrete class

Every backend that needs stochastic gradients takes a
`RandomMeasure[Record]` (carrying
`SupportsRandomUnnormalizedLogProb`) and ignores its concrete
subclass. The minibatched case is one input among many.
BlackJAX's SGMCMC kernel signature
`sghmc.step(rng_key, state, minibatch)` is intentionally
flexible: the third argument is whatever the user passes to
`grad_estimator`, and BlackJAX forwards it opaquely. We use that
flexibility to pass a JAX key — the measure samples its
per-step realisation internally:

```python
# probpipe/inference/_blackjax_sgmcmc.py

def _execute_sghmc(
    measure: RandomMeasure[Record],
    init_position: Record,
    *,
    step_size: float,
    num_steps: int,
    key: PRNGKey,
    name: str | None = None,
) -> MCMCApproximateDistribution:
    rand_logp = random_unnormalized_log_prob(measure)
    # ↑ a RandomFunction[Record, LogProb]; calling its ._sample(k)
    # returns a deterministic log-density callable for that draw.

    def grad_estimator(theta: Record, measure_key: PRNGKey) -> Record:
        # JIT-traced. The closure captures a freshly-sampled batch
        # (or whatever per-realisation state the measure carries);
        # jax.grad differentiates through it wrt theta.
        realised_logp = rand_logp._sample(measure_key)
        return jax.grad(realised_logp)(theta)

    sghmc = blackjax.sghmc(grad_estimator, step_size, ...)
    state = sghmc.init(init_position)

    @jax.jit
    def one_step(state, key):
        k_kernel, k_measure = jax.random.split(key)
        # Distinct keys for kernel-internal noise vs measure draw.
        state, info = sghmc.step(k_kernel, state, k_measure)
        return state, info

    positions = []
    for _ in range(num_steps):
        key, sub = jax.random.split(key)
        state, _ = one_step(state, sub)
        positions.append(state.position)

    return MCMCApproximateDistribution(
        chains=[_stack_records(positions)],
        algorithm="blackjax_sghmc",
        is_approximate=True,
        name=name,
    )
```

A few correctness points the sketch is making explicit:

1. **`rand_logp._sample(measure_key)` inside JIT.** The call
   returns a Python closure during tracing; JAX traces through
   the closure body (which captures the sampled batch as a
   traced array), so the JIT'd code is a pure sequence of JAX
   primitives — no closure survives to runtime. This is the
   standard "construct-traced-closure" pattern.
2. **Two keys per step.** The BlackJAX kernel needs its own RNG
   (Langevin / HMC noise); the measure needs an independent key
   for its per-step randomness. Splitting at the top of
   `one_step` keeps them uncorrelated.
3. **BlackJAX's third `step` slot.** We pass a key rather than a
   concrete batch — the kernel forwards whatever we pass to
   `grad_estimator`, by design. Documented in the module
   docstring so future readers don't expect a literal minibatch.

User-facing entry point:
`condition_on(model, data, method="blackjax_sghmc", batch_size=256)`
constructs a `MinibatchedLogDensity` internally and hands it to
`_execute_sghmc`. Power users can build any
`RandomMeasure[Record]` implementing
`SupportsRandomUnnormalizedLogProb` and pass it explicitly to a
fitter.

#### Stacking chains

`_stack_records` turns a Python list of per-step `Record`s into a
`RecordArray` along a new leading "draw" axis, so
`MCMCApproximateDistribution` sees one chain of draws keyed by
parameter name. (See § 12.3 for the structural decision around
carrying `RecordArray` chains directly.)

### 4.5 Future random-measure subclasses

Because the SGMCMC backend keys off the protocol, adding a new
stochastic target later doesn't require touching backend code. The
following all fit the same `RandomMeasure[Record]` +
`SupportsRandomUnnormalizedLogProb` shape:

| Subclass | Source of randomness | In scope now? |
|---|---|---|
| `MinibatchedLogDensity` | Uniform minibatch sampling | ✅ Phase 1b |
| `TemperedLogDensity` | Tempering schedule (e.g. Bayesian model evidence). SMC integration itself ships in this plan (§ 7) consuming the decomposed `(prior, likelihood)` form directly; `TemperedLogDensity` is the cleaner abstraction for non-SMC tempered targets and is left to a future drop-in. | Future — same protocol |
| `DoublyStochasticLogDensity` | Combine minibatching with auxiliary-variable Monte Carlo (doubly stochastic VI) | Future |
| `SyntheticLikelihood` | Likelihood is itself a random simulator output (ABC, neural-likelihood) | Future |
| `RussianRouletteLogDensity` | Random truncation of an infinite series | Future |
| `AntitheticPair` | Coupled pair of inner measures for variance reduction | Future |

Only `MinibatchedLogDensity` ships in this proposal; the rest are
listed to make the design's reach explicit. A future contributor
adding `TemperedLogDensity` doesn't have to touch the SGMCMC backend
— the existing kernel just works with the new measure.

### 4.6 What ops give you on a `RandomMeasure[Record]`

The existing op layer applies, with the math caveats spelled out:

| Op | Result on a `MinibatchedLogDensity` |
|---|---|
| `sample(measure, key)` | One realised inner `Distribution[Record]` (the per-minibatch posterior). Useful for "give me a snapshot target to inspect / debug". |
| `mean(measure)` | The marginalised inner distribution — a *mixture* of per-minibatch posteriors. **Not** the full-data target; flag in docstring. |
| `random_unnormalized_log_prob(measure)` (no value arg) | A `RandomFunction[Record, LogProb]`. Calling `._sample(key)` on it gives a deterministic unnormalized log-density callable. The *primary* op for SGMCMC. |
| `random_unnormalized_log_prob(measure, theta)` (with value arg) | Per the protocol's documented two-arg form: a `Distribution[LogProb]` — distribution over log-density estimates at $\theta$. `mean(...)` of *that* is $\log p_{\text{full}}(\theta)$ — the unbiased log-density estimator. |
| `condition_on(measure, observed)` | Per the existing `_random_measures.py` comment, "should compose via the existing condition_on op machinery". Out of scope here but the door is open. |

### 4.7 Two supporting additions

The minibatch construction needs:

1. **`ConditionallyIndependentLikelihood`** — a new subclass of
   `Likelihood` carrying `per_datum_log_likelihood(params, datum)`,
   the log-density of a single observation given parameters,
   $\log p(d_i \mid \theta)$.

   **Semantics.** The minibatched log-density (§ 4.3) sums
   per-datum terms over $B$, so it implicitly assumes the
   observations are conditionally independent given the
   parameters:
   $$\log p(\mathcal{D} \mid \theta) = \sum_i \log p(d_i \mid \theta).$$
   "Conditionally independent" rather than "i.i.d." matters:
   each `datum` may include covariates ($d_i = (x_i, y_i)$ with
   $y_i \mid x_i, \theta$ independent across $i$ but not
   identically distributed marginally), so the per-datum density
   varies with the input. The factorization across $i$ is what
   minibatching needs; identical marginal distribution is not.
   This covers the vast majority of `Likelihood` subclasses
   users actually build in ProbPipe — regression /
   classification / count-regression / GLM likelihoods,
   mixture-of-experts conditional likelihoods, exchangeable
   observation models. Every existing concrete `Likelihood` in
   the codebase today fits this shape.

   **Where the factorization doesn't apply** (and so where
   subclassing `ConditionallyIndependentLikelihood` would be
   incorrect):

   - **Time series / state-space models** ($\log p(y_{1:T} \mid \theta) = \sum_t \log p(y_t \mid y_{1:t-1}, \theta)$): the per-step term conditions on history, so the "datum" required for evaluation isn't just $y_t$. Minibatching needs a different decomposition (e.g., random subsequences with detached starts).
   - **Pairwise-potential / MRF likelihoods** ($\log p(y) = \sum_{(i,j) \in E} \phi(y_i, y_j)$): the "datum" is an edge, not a vertex; minibatching is possible (sample edges, rescale) but along a different axis.
   - **Latent-variable likelihoods that marginalise** (HMMs, mixture models with explicit assignment): the per-datum density typically requires a forward pass / message-passing, so it isn't a pure function of $(\theta, d_i)$.
   - **Likelihoods with shared sufficient statistics** (e.g., a likelihood whose data dependence enters through $\bar{x}$): minibatching breaks unbiasedness of the gradient estimator.

   These cases stay on the base `Likelihood`; they remain usable
   for full-batch HMC / NUTS / VI, just not for SGMCMC.

   **API.**

   ```python
   class Likelihood:
       def log_likelihood(self, params, data) -> jnp.ndarray:
           """Joint log-likelihood over the full dataset."""

   class ConditionallyIndependentLikelihood(Likelihood):
       """Likelihood whose observations factorise as

           log p(D | theta) = sum_i log p(d_i | theta)

       — i.e., conditionally independent (not necessarily
       identically distributed; ``datum`` may carry covariates).
       Required by ``MinibatchedLogDensity`` and the SGMCMC
       backends; useful elsewhere too (held-out predictive
       log-likelihoods, leave-one-out cross-validation, PSIS-LOO).
       """

       def per_datum_log_likelihood(self, params, datum) -> jnp.ndarray:
           """Log-density of a single datum given parameters.

           Default uses log_likelihood on a length-1 batch;
           override for efficiency.
           """
           return self.log_likelihood(params, datum[None, ...])
   ```

   **Migration.** All existing concrete `Likelihood` subclasses
   in ProbPipe are conditionally independent (they're regression
   / classification / GLM / count-regression likelihoods). The
   migration is mechanical: reparent them from `Likelihood` to
   `ConditionallyIndependentLikelihood`. `MinibatchedLogDensity`
   raises a clear error at construction time if the source
   model's likelihood is a bare `Likelihood` rather than a
   `ConditionallyIndependentLikelihood`, naming the required
   subclass so users adding new likelihoods make a deliberate
   choice.

   New contributors adding genuinely non-factorizable likelihoods
   (time series, MRF, latent-variable) stay on the base class
   and lose minibatching access — which is the correct outcome.

2. **`split_log_density(model, observed)`** — promote
   `_build_target_log_prob` from `_tfp_mcmc.py` to a public
   utility returning `(log_prior_fn, per_datum_log_likelihood_fn)`.
   Used by `MinibatchedLogDensity.__init__`; requires
   `model.likelihood` to be a `ConditionallyIndependentLikelihood`.

Both land in **Phase 1b** alongside `MinibatchedLogDensity`
itself.

## 5. Native Laplace approximation

### 5.1 Why not laplax

laplax's API is:

```python
posterior_fn, _ = laplax.laplace(
    model_fn,           # forward pass: (input, params) -> output
    params,             # MAP point
    data,               # (X, y) tuples
    loss_fn="mse",      # or "cross_entropy", or a callable
    curv_type="full",
)
```

This is a **forward-pass-plus-loss** decomposition — a deep-learning
flavoured contract. ProbPipe speaks `_log_prob((params, data))` —
already-marginalized log-density. Two adapter paths exist:

1. Decompose `_log_prob` into a "model_fn forward pass" and a "loss"
   that don't naturally exist at the `Distribution` level.
2. Pass a no-op forward and use the full log-density as `loss_fn` —
   works mechanically, but the abstraction lies (no `model_fn(input)`
   is called) and we throw away laplax's layer-aware optimisations
   (which are why it exists).

Either path produces a fragile adapter that fights the upstream
abstraction. A native implementation is cleaner.

### 5.2 Native implementation

Laplace approximation is conceptually simple in JAX:

1. Find MAP: $\hat\theta = \operatorname*{arg\,max}_\theta \log p(\theta, \mathcal{D})$.
2. Compute curvature: $H = -\nabla^2_\theta \log p(\hat\theta, \mathcal{D})$.
3. Posterior is $\mathcal{N}(\hat\theta, H^{-1})$.

ProbPipe has all the pieces:

- `model.as_flat_distribution()` flattens parameters to a single
  vector for curvature work; results lift back via
  `as_record_distribution(template=...)` (§ 8).
- `optax` (already a transitive dep of TFP) handles MAP optimisation.
- `jax.hessian` gives the full curvature; `jax.grad ∘ jax.grad` of a
  diagonal projection gives the diagonal at near-zero extra cost.
- `MultivariateNormalPrecision` (§ 8) is the natural output type when
  the curvature is dense; `Normal` covers the diagonal case;
  `MultivariateNormalLowRank` covers the low-rank-plus-diagonal case.

The natural shape for the implementation is a thin orchestrator
(`LaplaceMethod`) parameterised by a **`CurvatureApproximation`**
strategy. Curvature schemes vary in cost / accuracy / model
assumptions; making the curvature pluggable keeps the
MAP-finding, flattening, and Record-template lifting in one place
while letting each curvature scheme own its specific Hessian
estimator and output distribution. New schemes (GGN, KFAC, …)
are new strategy subclasses with no `LaplaceMethod` changes.

### 5.3 Curvature approximation as a pluggable strategy

The protocol:

```python
# probpipe/inference/_laplace.py

@runtime_checkable
class CurvatureApproximation(Protocol):
    """Strategy for approximating −∇²log p at the MAP point.

    The strategy owns both the curvature computation and the output
    distribution shape, since the two are coupled (a low-rank
    curvature lands in MultivariateNormalLowRank; a diagonal
    curvature lands in Normal; a dense curvature lands in
    MultivariateNormalPrecision).
    """

    @property
    def name(self) -> str:
        """Short name used in the registered method name."""

    def estimate(
        self,
        log_density: Callable[[jax.Array], jax.Array],
        theta_map: jax.Array,
        **kwargs,
    ) -> Distribution:
        """Build the posterior approximation centred at theta_map.

        Returns a flat-parameter Distribution; the caller lifts it
        back to the Record template via as_record_distribution(...).
        """
```

Three shipped strategies, all in `probpipe/inference/_laplace.py`:

```python
class FullHessian(CurvatureApproximation):
    name = "full"

    def estimate(self, log_density, theta_map, **kw) -> Distribution:
        H = -jax.hessian(log_density)(theta_map)
        return MultivariateNormalPrecision(
            loc=theta_map, precision=H, name=kw.get("name"),
        )


class DiagonalHessian(CurvatureApproximation):
    name = "diag"

    def estimate(self, log_density, theta_map, **kw) -> Distribution:
        H_diag = -_diagonal_hessian(log_density, theta_map)
        return Normal(
            loc=theta_map, scale=H_diag ** -0.5, name=kw.get("name"),
        )


class LanczosLowRank(CurvatureApproximation):
    name = "lowrank"

    def __init__(
        self, *, rank: int | None = None,
        tikhonov: float | None = None,
        reorthogonalize: bool = True,
    ):
        self.rank = rank
        self.tikhonov = tikhonov
        self.reorthogonalize = reorthogonalize

    def estimate(self, log_density, theta_map, **kw) -> Distribution:
        H_lowrank = _lanczos_hessian(
            log_density, theta_map,
            rank=self.rank or min(50, theta_map.size),
            tikhonov=self.tikhonov,
            reorthogonalize=self.reorthogonalize,
        )
        return MultivariateNormalLowRank.from_inverse(
            loc=theta_map, inverse=H_lowrank, name=kw.get("name"),
        )
```

The orchestrator wires MAP + strategy + lift-back together:

```python
class LaplaceMethod(InferenceMethod):
    """Native Laplace approximation: MAP + curvature → Gaussian.

    Curvature scheme is pluggable via the ``curvature`` argument.
    """

    def __init__(
        self, *,
        curvature: CurvatureApproximation = FullHessian(),
        priority: int = 40,
    ):
        self._curvature = curvature
        self._priority = priority

    @property
    def name(self) -> str:
        return f"laplace_{self._curvature.name}"

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    @property
    def priority(self) -> int:
        return self._priority

    def check(self, dist, observed, **kw) -> MethodInfo:
        if not isinstance(dist, SupportsLogProb):
            return MethodInfo(False, self.name, "Requires SupportsLogProb")
        # Probe JAX-traceability + flat parameterisation; consult
        # the strategy for any additional requirements (e.g., GGN
        # would require a per-datum prediction map on the model).
        ...

    def execute(self, dist, observed, **kw) -> Distribution:
        target_record = _build_target_log_prob(dist, observed)
        prior = dist._prior if isinstance(dist, SimpleModel) else dist
        flat = prior.as_flat_distribution()
        record_template = prior.record_template

        def target_flat(theta_flat):
            return target_record(flat.unflatten_sample(theta_flat))

        theta_flat_map = _run_map_optim(target_flat, flat, **kw)
        approx_flat = self._curvature.estimate(
            target_flat, theta_flat_map, name=kw.get("name"),
        )
        return approx_flat.as_record_distribution(template=record_template)
```

Registered factories (each is a thin `LaplaceMethod(curvature=...)`
binding so users get `laplace_full`, `laplace_diag`,
`laplace_lowrank` as named methods that auto-select):

| Phase | Method name | Strategy | Priority |
|---|---|---|---|
| 6a | `laplace_full` | `FullHessian()` | 40 |
| 6a | `laplace_diag` | `DiagonalHessian()` | 40 |
| 6b | `laplace_lowrank` | `LanczosLowRank()` | 35 |

Implementation notes:

- `_diagonal_hessian` is `jnp.diag(jax.hessian(...))` for modest
  dim, or per-coordinate `jax.grad ∘ jax.grad` for larger dim.
  Either is correct; the latter avoids materialising the full
  $n \times n$ Hessian.
- Treating the prior as the source of the record template assumes
  a `SimpleModel`-shaped target. For raw `SupportsLogProb`
  distributions where parameters are the distribution itself
  (no separate prior), `prior = dist`; the same code path works.
- A user can supply a custom curvature without registering a new
  method: `condition_on(model, data, method="laplace_full",
  method_kwargs={"curvature": MyCurvature()})` overrides the
  bundled one; passing an explicit `CurvatureApproximation`
  instance to `LaplaceMethod` and registering it is the path for
  long-lived custom strategies.

### 5.4 Lanczos low-rank (LanczosLowRank strategy, Phase 6b)

The full-Hessian path scales poorly: dense storage is $O(n^2)$ and
`jax.hessian` materialises the full matrix. For models above a few
thousand parameters this becomes prohibitive — but
`DiagonalHessian` discards all off-diagonal information, which
loses the parameter correlations Laplace exists to capture in the
first place. `LanczosLowRank` closes that gap.

The technique:

1. **Matrix-free Hessian-vector products.** A single HVP
   $H v$ costs $O(n)$ — one forward-mode pass through a gradient
   — and uses no extra memory beyond the parameter vector. JAX
   gives this directly via the JVP-of-grad form:
   ```python
   def hvp(v):
       grad_target = jax.grad(target_flat)
       _, hv = jax.jvp(grad_target, (theta_flat_map,), (v,))
       return hv
   ```
   (The double-grad form
   `jax.grad(lambda p: jnp.vdot(jax.grad(target_flat)(p), v))(theta_flat_map)`
   gives the same result but is reverse-over-reverse, so ~2× the
   FLOPs of forward-over-reverse here.)
2. **Lanczos iteration.** Builds an orthonormal basis
   $V \in \mathbb{R}^{n \times k}$ and tridiagonal
   $T \in \mathbb{R}^{k \times k}$ such that $V^T H V = T$, using
   only HVPs. After $k$ iterations:
   $$H \approx V T V^T + \sigma^2 I,$$
   where $\sigma^2$ is a small ridge fitted from the
   tail-eigenvalue mass (or supplied by the user). $k$ is chosen
   to capture the "interesting" eigenvalues; $k \ll n$.
3. **Posterior parameterisation.** The covariance
   $\Sigma = (V T V^T + \sigma^2 I)^{-1}$ admits a low-rank +
   diagonal form via the Sherman–Morrison–Woodbury identity:
   $$\Sigma = \sigma^{-2}I + V D V^T$$
   for a diagonal $D = T^{-1} - \sigma^{-2}I$ (in the
   eigenbasis). This is exactly the parameterisation TFP's
   [`MultivariateNormalDiagPlusLowRankCovariance`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalDiagPlusLowRankCovariance)
   exposes; `MultivariateNormalLowRank` wraps it (§ 8).
   Sampling and log-prob work in $O(nk)$ per draw.

Hyperparameters surfaced via the `LanczosLowRank` constructor (or
`method_kwargs` for the registered `laplace_lowrank` factory):

- `rank: int` — Lanczos iterations $k$ (default: $\min(50, n)$).
- `tikhonov: float | None` — ridge $\sigma^2$. If `None`, fit from
  the average of the discarded eigenvalues.
- `reorthogonalize: bool` — full reorthogonalisation each step
  (default `True`; numerically robust at small extra cost).

### 5.5 Future curvature strategies (extension points, not adapters)

The `CurvatureApproximation` protocol is the extension surface.
Two well-known schemes that don't ship in Phase 6 but fit the
same protocol once the prerequisite abstractions exist:

- **`GGN`** (Gauss-Newton approximation): useful for models with
  a clean per-datum prediction structure. Requires the model to
  expose a "prediction map" separate from the full log-density —
  ProbPipe doesn't have that abstraction today. The strategy
  class would live in `_laplace.py` alongside the others; the
  block is the model abstraction, not the curvature code.
- **`KFAC`** (Kronecker-factored): layer-aware curvature for
  neural networks. Presupposes a NN layer structure ProbPipe
  doesn't inspect. Same blocker: the strategy fits the protocol,
  but ProbPipe needs to grow a layer-aware model abstraction
  first.

Either becomes a small, additive PR once its prerequisite lands.
The Phase 6 scope ships the three strategies above; later phases
add to `CurvatureApproximation` without modifying it.

### 5.6 Why Laplace is the lowest-priority phase

Among the three families on the wishlist (Laplace, SGMCMC, VI),
Laplace is the **least urgent** for ProbPipe:

- It is a parametric posterior over a flat parameter vector; users
  who want one already have analytic alternatives for small models
  and NUTS / SGMCMC / VI for large ones.
- It is the only family with no external library dependency, so it
  is *not blocked* by integration work.
- The BlackJAX migration (MCMC + SGMCMC + VI) is the higher-leverage
  work — it touches existing users (MCMC default flip), unlocks
  SGMCMC for the first time, and adds VI / Pathfinder.

Native Laplace lands **after** the BlackJAX migration completes
(see § 9, Phases 6a/6b).

## 6. Migrate MCMC: TFP → BlackJAX

ProbPipe currently registers `tfp_nuts` (priority 100) and `tfp_hmc`
(priority 90) via `probpipe/inference/_tfp_mcmc.py`. BlackJAX is the
JAX-native incumbent: same algorithms, smaller dependency surface,
better integration with the rest of the JAX ecosystem (NumPyro, Flax,
Equinox), and aligned with the SGMCMC / VI / Pathfinder choice in §3.
Migration warrants a phased rollout to retain behavioral parity.

### 6.1 Migration phases (within § 9 rollout)

1. **Add BlackJAX MCMC at lower priority.** Register `blackjax_nuts`
   (priority 95) and `blackjax_hmc` (priority 85). Auto-selection
   still picks TFP.
2. **Benchmark suite.** Compare TFP vs BlackJAX on the existing
   end-to-end inference tests. The existing `test_tfp_mcmc.py`
   becomes the parity reference. Acceptance criteria for the
   priority flip — all must pass for the same problem and pinned
   seed:
   - **Posterior accuracy on a closed-form 2-D Gaussian target:**
     posterior mean within $0.5 \sigma_{\text{MC}}$ of TFP's, and
     posterior covariance Frobenius distance from TFP's within
     $0.05 \cdot \|\Sigma_{\text{TFP}}\|_F$.
   - **Sampling throughput on a moderate model** (8-D logistic
     regression on 1000 observations, 2000 draws): BlackJAX
     throughput within 20% of TFP's; per-iteration cost should
     match given identical kernel parameters.
   - **Cold-start time** (process startup → first draw): BlackJAX
     within $1.5 \times$ TFP's. Cold start matters for short CI
     runs and notebook workflows.
   - **Convergence diagnostics:** `arviz.ess` and `arviz.rhat` from
     both backends agree to two decimal places.
   - **Dependency footprint:** documented for the CHANGELOG (no
     pass/fail criterion, but expected to *decrease* — BlackJAX's
     import surface is smaller than TFP's).
3. **Flip priorities.** When the criteria above are met:
   `registry.set_priorities(blackjax_nuts=100, blackjax_hmc=90, tfp_nuts=10, tfp_hmc=10)`.
4. **Deprecation warning.** TFP MCMC method classes emit a
   `DeprecationWarning` for one minor release.
5. **Remove TFP MCMC**. After the deprecation window, delete
   `_tfp_mcmc.py` and drop the `tensorflow_probability` MCMC import
   surface. (TFP-as-distribution-backend stays — it's used by
   `TFPDistribution` and is unaffected.)

The deprecation window is **short** in absolute terms. ProbPipe is
in alpha (transitioning to internal beta), so there is no
long-tail of pinned-version external users to support. Phases
2a → 2b → 2c can land in three consecutive minor releases — or
collapsed further once an internal user opts in to the BlackJAX
path. Pinning `method="tfp_nuts"` for one release before deletion
covers the bit-pattern-regression case (see § 6.3).

### 6.2 What gets simpler

- `MCMCApproximateDistribution` is unchanged — it's backend-agnostic,
  takes raw chains.
- `_build_target_log_prob` (in the new `_inference_utils.py`)
  already returns a JAX-traceable log-density that BlackJAX consumes
  directly.
- BlackJAX's NUTS / HMC kernels return `(state, info)` per step.
  `info` carries `acceptance_probability`, `step_size`,
  `num_integration_steps` etc., which we forward to ArviZ
  `sample_stats`.

### 6.3 Risk

The migration changes the *exact bit-pattern* of posterior samples
for any user with a pinned random seed. Document this in the
CHANGELOG as a behavioural breaking change in the release that
flips the priorities. Users who need bit-identical regression
should pin `method="tfp_nuts"` for one release before deletion.

## 7. Public API

Same `condition_on` entry point:

```python
import probpipe as pp

model = pp.SimpleModel(prior, likelihood)

# MCMC — auto-select picks blackjax_nuts after the priority flip
posterior = pp.condition_on(model, data)

# Laplace — native, fast parametric posterior
posterior = pp.condition_on(model, data, method="laplace_diag")

# SGMCMC — minibatched
posterior = pp.condition_on(
    model, data,
    method="blackjax_sghmc",
    batch_size=256,
    num_steps=10_000,
    method_kwargs={"step_size": 1e-4, "num_integration_steps": 10},
)

# VI — Pathfinder warm-start
posterior = pp.condition_on(
    model, data, method="blackjax_pathfinder", num_samples=1000,
)

# SMC — adaptive-tempered, with inner NUTS kernel
posterior = pp.condition_on(
    model, data,
    method="blackjax_adaptive_tempered_smc",
    num_particles=2000,
    method_kwargs={
        "target_ess": 0.5,          # adaptive resampling threshold
        "mcmc_kernel": "nuts",      # MCMC moves at each tempering level
        "mcmc_kwargs": {"step_size": 0.01, "num_integration_steps": 20},
    },
)
```

Power-user path with an explicit random-measure target:

```python
from probpipe.inference import MinibatchedLogDensity

# Same target the registered method builds internally:
target = MinibatchedLogDensity(model, data, batch_size=256)

# Or any other RandomMeasure[Record] subclass implementing
# SupportsRandomUnnormalizedLogProb (e.g., a future
# TemperedLogDensity, a user-defined synthetic-likelihood, …):
target = MyCustomRandomMeasure(...)

# Hand off to a custom training loop or a non-registered BlackJAX
# kernel; backends only require the protocol, not the subclass.
```

## 8. Core additions

All belong in `probpipe/core/` (or `probpipe/distributions/` for the
new MVN classes) and are independently useful, but they have different
timing because they unblock different phases:

- **`Distribution.as_record_distribution(template=...)`** — inverse
  of the existing `as_flat_distribution()`. Lifts a flat-vector
  distribution back to a Record-keyed view. Useful for any flat
  parametric posterior: VI mean-field / full-rank, Pathfinder
  draws viewed under a Record template, and Laplace.

  Signature sketch:
  ```python
  class Distribution:
      def as_record_distribution(
          self, *, template: RecordTemplate,
      ) -> RecordDistribution[Record]:
          """Lift a flat-vector distribution to a Record-keyed view
          using ``template`` as the structural skeleton. The flat
          distribution's ``event_size`` must match
          ``template.flat_size``; otherwise a ``ValueError`` is
          raised with both sizes in the message."""
  ```
  Lands in **Phase 1a** so VI users (Phase 4) get the structured
  view out of the box.

- **`MultivariateNormalPrecision`** — new `Distribution` class
  parameterised as `(loc, precision)`. Sibling of
  `MultivariateNormal` (which takes `cov` / `scale_tril`),
  mirroring TFP's pattern of one class per parameterisation
  ([`MultivariateNormalFullCovariance`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalFullCovariance),
  [`MultivariateNormalTriL`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalTriL),
  [`MultivariateNormalDiag`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalDiag)).
  Laplace natively yields a precision matrix; the
  precision-parameterised `log_prob` form `(x−μ)ᵀH(x−μ)` is also
  cheaper than going through covariance, so the parameterisation
  matters internally. Type tells you which one you're holding;
  `isinstance` checks and protocol-driven dispatch (converters
  that exploit precision structure, natural-gradient VI later)
  fall out naturally. Bundled with **Phase 6a**.

- **`MultivariateNormalLowRank`** — new `Distribution` class
  wrapping
  [`tfp.distributions.MultivariateNormalDiagPlusLowRankCovariance`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalDiagPlusLowRankCovariance).
  Parameterised as $\Sigma = \mathrm{diag}(d) + U U^T$. Carries
  whatever moment / sampling protocols the underlying TFP class
  exposes; otherwise behaves like every other TFP-backed class.
  Bundled with **Phase 6b** (Lanczos low-rank Laplace) — its sole
  consumer.

- **`PathfinderDistribution`** — new parametric `Distribution`
  class capturing Pathfinder's output verbatim. Pathfinder's
  approximating distribution is a mixture of importance-weighted
  multivariate Gaussians (one per L-BFGS iterate), and that
  mixture has a closed-form log-density and analytic moments —
  the parametric character is the *whole point* of Pathfinder
  over plain importance sampling. A `MCMCApproximateDistribution`
  return type would discard that structure, leaving only
  empirically-resampled draws and forcing `log_prob` /
  `mean` / `variance` through KDE-style approximations.

  Sketch:
  ```python
  class PathfinderDistribution(NumericRecordDistribution):
      """Pathfinder approximation: weighted mixture of MVN iterates.

      Each L-BFGS iterate produces a Gaussian approximation
      ``N(μ_i, Σ_i)``; importance weights ``w_i ∝ p(theta_i) /
      q_i(theta_i)`` reweight them against the true posterior.
      """
      def __init__(
          self, *, locs: jax.Array, scale_trils: jax.Array,
          log_weights: jax.Array,
          record_template: NumericRecordTemplate,
          name: str | None = None,
      ): ...

      # Analytic — composes the mixture log-density from the
      # per-component MVN log-probs and the importance weights.
      def _log_prob(self, x: Record) -> LogProb: ...

      # Analytic — weighted average of component means.
      @property
      def mean_(self) -> Record: ...

      # Analytic — law of total variance over the mixture.
      @property
      def variance_(self) -> Record: ...

      # Importance-resample from the mixture, optionally with
      # the systematic / stratified resampling variants Pathfinder
      # papers use.
      def _sample(self, key, sample_shape=()): ...
  ```

  Bundled with **Phase 4** alongside `blackjax_pathfinder` — its
  consumer.

## 9. Phased rollout

The BlackJAX migration (Phases 2–4) lands first; native Laplace
comes last as the lowest-priority piece (see § 5.6).

| Phase | Scope | Depends on |
|---|---|---|
| **0** | This plan PR. | — |
| **1a** | `Distribution.as_record_distribution(template=...)`. Small core add. | — |
| **1b** | `MinibatchedLogDensity` + `Likelihood.per_datum_log_likelihood`. No backend deps. | — |
| **2a** | Register `blackjax_nuts` + `blackjax_hmc` at low priority. Run benchmark suite vs TFP. | 1b (shared `_build_target_log_prob` lifted out of `_tfp_mcmc.py`) |
| **2b** | Flip priorities (BlackJAX wins auto-selection); emit TFP-MCMC deprecation warning. | 2a |
| **2c** | Remove `_tfp_mcmc.py` after the deprecation window. | 2b |
| **3** | BlackJAX SGMCMC (`blackjax_sgld`, `blackjax_sghmc`). | 1b |
| **4** | BlackJAX VI (`blackjax_meanfield_vi`, `blackjax_fullrank_vi`, `blackjax_pathfinder`) + `PathfinderDistribution`. | 1a |
| **5** | BlackJAX SMC (`blackjax_tempered_smc`, `blackjax_adaptive_tempered_smc`). | 1b, 2a |
| **6a** | Native Laplace full + diag (`laplace_full`, `laplace_diag`) + precision-parameterised MVN (see § 8 for the constructor-vs-class design choice). | 1a |
| **6b** | Lanczos low-rank Laplace (`laplace_lowrank`) + `MultivariateNormalLowRank`. | 6a |

Notes:

- **Phases 1a, 1b, 2a, 2b, 2c, 3, 4, 5** are the minimum viable
  BlackJAX-migration + integration scope. Each is a separate PR.
- **Phase 6a + 6b** are the lowest-priority additions. 6a is
  unblocked after 1a lands (no dependency on the BlackJAX phases)
  but is scheduled last to keep reviewer + maintainer attention on
  the higher-leverage migration work first. 6b builds on 6a:
  same MAP-finding / target-construction infrastructure, just a
  different curvature path.
- Phases 1a, 1b, 4, 5, and 6a are independent of each other and could
  proceed in parallel if reviewer bandwidth allows; the table
  reflects the *priority ordering*, not a strict serial gate.

## 10. Subpackage layout

```
probpipe/
├── inference/
│   ├── __init__.py               # re-exports MinibatchedLogDensity
│   ├── _minibatch.py             # NEW — MinibatchedLogDensity (RandomMeasure subclass)
│   ├── _inference_utils.py       # NEW — split_log_density + _build_target_log_prob (lifted from _tfp_mcmc.py)
│   ├── _laplace.py               # NEW — native Laplace methods
│   ├── _blackjax_sgmcmc.py       # NEW — SGLD, SGHMC
│   ├── _blackjax_vi.py           # NEW — meanfield_vi, fullrank_vi, pathfinder
│   ├── _blackjax_smc.py          # NEW — tempered_smc, adaptive_tempered_smc
│   ├── _blackjax_mcmc.py         # NEW — NUTS, HMC (Phase 2a)
│   └── _tfp_mcmc.py              # DEPRECATED in Phase 2b → DELETED in Phase 2c
└── distributions/
    ├── _multivariate.py          # MODIFIED — add MultivariateNormalPrecision (Phase 6a)
    ├── _multivariate_lowrank.py  # NEW — MultivariateNormalLowRank (Phase 6b)
    └── _pathfinder.py            # NEW — PathfinderDistribution (Phase 4)
```

Notes:

- `_minibatch.py` carries an underscore (private module) per
  STYLE_GUIDE.md § 1.6; the public symbol re-exports through
  `probpipe.inference.__init__` (and ultimately `probpipe`). Future
  `RandomMeasure` subclasses (`TemperedLogDensity`, etc.) likewise
  live in private modules and re-export.
- New distribution classes (`MultivariateNormalPrecision`,
  `MultivariateNormalLowRank`, `PathfinderDistribution`) live in
  `probpipe/distributions/` and re-export through the top-level
  package — every distribution class users construct is reachable
  via `probpipe.<ClassName>`.

Dependency graph (STYLE_GUIDE.md § 6) is unchanged — `inference/`
imports from `core/` only. Optional-extras gating:

```toml
[project.optional-dependencies]
blackjax = ["blackjax>=1.5"]
```

`pip install probpipe[blackjax]` enables every BlackJAX-backed method.
The Laplace methods have **no extra dependency**.

## 11. Testing strategy

- Skip pattern: `pytest.importorskip("blackjax")` for BlackJAX
  methods; Laplace methods always run.
- **Closed-form correctness** on 2-D Gaussian targets (all
  seed-pinned to `jax.random.PRNGKey(0)` unless stated):
  - **Laplace full / diag:** recovered precision matrix matches the
    analytic precision to `rtol=1e-3` on full, `rtol=5e-2` on diag
    (off-diagonal correlation discarded by design).
  - **VI mean-field / full-rank:** recovered mean within
    `atol=0.05` of analytic; full-rank scale within `rtol=0.1`;
    mean-field scale within `rtol=0.2` (the well-known
    underestimation bias on correlated targets).
  - **SGMCMC:** empirical mean / variance within
    $3\sigma_{\text{MC}}$ MC error.
  - **Pathfinder:** importance-weighted empirical mean / variance
    within `atol=0.1`; the *analytic* mixture mean / variance
    (from `PathfinderDistribution.mean_` / `.variance_`) within
    `atol=0.02` of the analytic target.
  - **SMC:** weighted-particle empirical mean / variance within
    $3\sigma_{\text{MC}}$ MC error. Plus an explicit
    **bimodal-target** test (mixture of two Gaussians at
    $\mu_1 = -3, \mu_2 = +3$, both unit-scale, with seed
    `PRNGKey(42)`) — assert TVD between the empirical posterior
    and the target is below `0.1` for `blackjax_tempered_smc` and
    above `0.3` for `blackjax_sghmc` (the failure mode is
    mode-trapping). The TVD threshold is the user-facing claim
    the test backs up: tempered SMC handles multimodality where
    single-chain gradient methods get stuck.
- **`MinibatchedLogDensity` as a `RandomMeasure`**:
  - `isinstance(target, RandomMeasure)` and
    `isinstance(target, SupportsRandomUnnormalizedLogProb)`.
  - `_sample(key)` returns a `Distribution[Record]` whose
    `_unnormalized_log_prob` evaluates to the rescaled minibatch
    density.
  - Index sampling produces uniformly-distributed indices
    (chi-square test, $p > 0.01$).
  - **Unbiased gradient**: averaging the gradient of
    `random_unnormalized_log_prob(target)._sample(k)(theta)` over
    1024 keys matches the full-data gradient to `atol=1e-3` per
    coordinate.
  - **Unbiased log-density**: averaging
    `random_unnormalized_log_prob(target)._sample(k)(theta)` over
    1024 keys matches the full-data log-density to `atol=1e-2`.
  - **JIT-traceability**: a SGMCMC step using the random measure
    JIT-compiles cleanly (regression test against accidental
    Python-side state in the inner distribution); compile-cache
    hit rate at 100% across 100 steps with identical pytree
    shape.
- **`PathfinderDistribution` correctness** (Phase 4):
  - `log_prob` against the analytic mixture density (built
    from the constructor's `locs`, `scale_trils`, `log_weights`)
    matches `scipy.stats.multivariate_normal`-based reference to
    `rtol=1e-5`.
  - `_sample` produces draws whose empirical mean / variance
    match `.mean_` / `.variance_` to MC tolerance (1000 draws,
    `atol=0.05`).
- **MCMC migration parity** (Phase 2a):
  - For each test in `test_tfp_mcmc.py`, run with both TFP and
    BlackJAX and assert the numeric thresholds in § 6.1.2 hold
    (mean within $0.5\sigma_{\text{MC}}$, covariance Frobenius
    distance within 5%, ESS / R-hat agreeing to two decimals).
- **Upstream-API regression** (CI cron, weekly):
  - SGMCMC integration tests run against `blackjax==<latest>`
    (rather than the project's lower-bound pin). Failure opens a
    tracking issue rather than blocking CI on `main`.
- **Convention tests**: every wrapper distribution passes the
  no-`batch_shape` invariant in `tests/test_distribution_base.py`
  and the no-iteration invariant.

## 12. Design specifications

These are the binding choices for design points that don't fall
out trivially from §§ 1–11.

### 12.1 Per-datum likelihood for non-`SimpleModel` targets

A bare `SupportsLogProb` distribution doesn't decompose into
prior + per-datum likelihood — there is *no canonical way* to
minibatch a black-box `log_prob(theta, data)`.
`MinibatchedLogDensity` requires `per_datum_log_likelihood=...`
as a constructor argument when the source is a raw
`SupportsLogProb` (no `Likelihood`-typed component to
interrogate). For `SimpleModel`-based targets, the per-datum
decomposition is derived from
`model.likelihood.per_datum_log_likelihood` automatically — but
only if `model.likelihood` is a
`ConditionallyIndependentLikelihood` (§ 4.7); a base
`Likelihood` is rejected with a clear error naming the required
subclass.

```
MinibatchedLogDensity requires either:
  - model.likelihood to be a ConditionallyIndependentLikelihood
    (the SimpleModel path), or
  - an explicit per_datum_log_likelihood=... callable (the
    bare SupportsLogProb path).
```

### 12.2 Pathfinder return type

`PathfinderDistribution` (§ 8) — parametric mixture, analytic
log-density and moments. Pathfinder's parametric character is
the whole point of the method; an `MCMCApproximateDistribution`
return type would discard that structure.

### 12.3 Record-shaped MCMC chains

`MCMCApproximateDistribution(chains=[Array])` keeps its
Array-typed chain storage. BlackJAX kernels produce Record-shaped
positions; each step's Record is flattened to an Array at
accumulation, and the structured view is exposed via
`posterior.as_record_distribution(template=...)`. The alternative
— extending `MCMCApproximateDistribution` (or adding a
Record-shaped variant) to carry `RecordArray` chains directly —
is a deferred follow-up. Trigger to switch: a user-facing
workflow where `posterior["theta"]` is invoked frequently enough
that the extra `as_record_distribution(template=...)` call
becomes friction (concretely: more than a passing reference in a
notebook or example).

### 12.4 MAP optimiser

Default `optax.adam` with `learning_rate=1e-2`, 500 max steps,
gradient-norm tolerance `1e-4`. Expose `optimizer=` in
`method_kwargs` for user override. Newton is a known good
alternative for modest-dim convex targets; not the registered
default because Newton's failure mode on non-convex targets is
silent divergence, while adam's is slow convergence (which is
detectable).

### 12.5 GPU placement for SGMCMC

For `jax.Array` data: already on-device, no plumbing. For
`RecordArray` / `NumericRecordArray` data: all leaves must be on
the same device as the model parameters.
`MinibatchedLogDensity.__init__` validates this via
`jax.tree.map(lambda x: x.device, data)` and raises a clear error
listing any heterogeneous-device leaves. Documented in the
docstring.

### 12.6 Provenance for moved utilities

Lifting `_build_target_log_prob` and `_get_init_state` out of
`_tfp_mcmc.py` into `_inference_utils.py` is a Phase 1b subtask
*and* a Phase 2a prerequisite (the BlackJAX MCMC path consumes
the same utility). Pure refactor; lands before any backend code
that depends on it.

### 12.7 Inner-distribution pytree shape

`_MinibatchPosterior` (one realisation of the random measure)
carries `(batch, rescale_factor)` as dynamic JAX leaves and
`model` as a static reference. JIT traces a fresh pytree per
SGMCMC step — should be cheap, but worth a regression test to
catch accidental Python-side bookkeeping that would force
retracing.

### 12.8 `RandomFunction[T, LogProb]` audit

The protocol's return type `RandomFunction[Record, LogProb]` is a
function-valued random variable — its realisations are callables
`theta → log_p(theta)`. The existing `RandomFunction` machinery
in `probpipe/core/_random_functions.py` has only been exercised
with numeric-array outputs (`GaussianRandomFunction`,
`LinearBasisFunction`). One-line audit task during Phase 1b:
confirm `RandomFunction` is generic enough over the output type
that a `LogProb`-valued realisation works through the existing
class, or scope a small extension if not.

### 12.9 Amortized inference scope

Out of scope for this plan. VI / Pathfinder handle parametric
amortisation (fixed-form mixture approximations); amortised
inference in the simulation-based-inference sense (training a
network to map `data → posterior parameters`, then evaluating at
inference time) is structurally different — needs a `Sampler`
abstraction, not a `RandomMeasure[Record]`. Defer until a
concrete user need surfaces.

## 13. Decision asked of reviewer

Sign-off requested on:

- The **JAX-native approach** (BlackJAX for MCMC + SGMCMC + VI +
  SMC; native Laplace; no external PyTorch dependency).
- Framing the SGMCMC target as a **`RandomMeasure[Record]`
  subclass** (§ 4); ship `MinibatchedLogDensity` as the concrete
  subclass with the door open for `TemperedLogDensity`,
  doubly-stochastic, ABC, etc. as future drop-ins.
- Introducing **`ConditionallyIndependentLikelihood`** as a
  `Likelihood` subclass carrying `per_datum_log_likelihood`
  (§ 4.7). `MinibatchedLogDensity` requires the source model's
  likelihood to be one; existing i.i.d.-style concrete
  likelihoods reparent mechanically; future non-factorizable
  likelihoods (time series, MRF, latent-variable) stay on the
  base class and lose minibatching access by design.
- The **TFP→BlackJAX migration** path (§ 6) with the
  numeric-threshold benchmark gate at Phase 2a.
- **Adding BlackJAX SMC** (tempered + adaptive-tempered) as
  Phase 5, consuming `split_log_density`'s prior/likelihood
  decomposition.
- The `LaplaceMethod` shape: thin orchestrator parameterised by a
  pluggable **`CurvatureApproximation`** strategy (§ 5.3), with
  `FullHessian`, `DiagonalHessian`, and `LanczosLowRank`
  shipping in Phase 6, and `GGN` / `KFAC` as future strategy
  classes once the prerequisite model abstractions exist (§ 5.5).
- **Deferring native Laplace to Phases 6a/6b** — after the
  BlackJAX migration completes (§ 5.6).
- The new Distribution classes in § 8:
  `Distribution.as_record_distribution(template=...)`,
  `MultivariateNormalPrecision`, `MultivariateNormalLowRank`,
  `PathfinderDistribution`.
- The phased rollout in § 9.

Once approved, Phase 1a is a separate small PR; Phase 1b is the
prerequisite refactor for Phases 2a (MCMC migration), 3 (SGMCMC),
and 5 (SMC); Phase 2 is the MCMC migration; Phases 3, 4, 5, and 6
follow.
