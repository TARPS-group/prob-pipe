"""BlackJAX-backed stochastic-gradient MCMC methods.

Two :class:`~probpipe.core._registry.UnaryDispatchMethod` subclasses registered with
:data:`~probpipe.inference.inference_method_registry`:

* ``blackjax_sgld`` — Stochastic Gradient Langevin Dynamics
  ([Welling & Teh, 2011](https://www.icml-2011.org/papers/398_icmlpaper.pdf)).
* ``blackjax_sghmc`` — Stochastic Gradient Hamiltonian Monte Carlo
  ([Chen, Fox & Guestrin, 2014](https://arxiv.org/abs/1402.4102)).

Both methods consume a :class:`~probpipe.SimpleModel` whose
``likelihood`` satisfies
:class:`~probpipe.ConditionallyIndependentLikelihood`. Internally they
construct a :class:`~probpipe.MinibatchedDistribution` to produce
unbiased stochastic gradient estimates and feed it to the BlackJAX
kernel via the ``grad_estimator(position, measure_key)`` closure
convention — the per-step ``measure_key`` is passed through BlackJAX's
opaque ``minibatch`` slot.

Priorities (45 / 42) sit in the refinement-based MC tier (1–50, above
parametric and below all exact methods); the per-method ``check()``
further requires ``batch_size=`` to be passed, so SGMCMC only applies
when the user has opted into minibatching via
``condition_on(model, observed, method="blackjax_sgld", batch_size=…)``
or by selecting it explicitly. See the inference-method registry
docs for the full priority convention.
"""

from __future__ import annotations

from typing import Any

import blackjax
import jax
import jax.numpy as jnp

from ..core._registry import MethodInfo
from ..core._random_measures import RandomMeasure
from ..custom_types import PRNGKey
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._minibatch import MinibatchedDistribution
from ._registry import InferenceMethod
from ._inference_utils import as_prng_key, get_init_state, is_simple_model

__all__ = ["BlackJAXSGLDMethod", "BlackJAXSGHMCMethod"]


# ---------------------------------------------------------------------------
# grad_estimator factory
# ---------------------------------------------------------------------------


def _build_grad_estimator(measure: RandomMeasure):
    """Build the BlackJAX-compatible gradient estimator from a random measure.

    BlackJAX's SGMCMC kernels take a callable
    ``grad_estimator(position, *opaque)`` and pass whatever the user
    provides in the ``minibatch`` slot opaquely. We exploit that by
    passing a fresh PRNG key per step; the random measure samples its
    own minibatch internally via
    :meth:`~probpipe.MinibatchedDistribution._random_unnormalized_log_prob`.
    The kernel stays oblivious to the minibatching convention, so the
    same builder works for any future ``RandomMeasure[Record]`` subclass
    that supplies :class:`SupportsRandomUnnormalizedLogProb`.
    """
    rand_logp = measure._random_unnormalized_log_prob()

    def grad_estimator(position: Any, measure_key: PRNGKey) -> Any:
        realised_log_density = rand_logp._sample(measure_key)
        return jax.grad(realised_log_density)(position)

    return grad_estimator


# ---------------------------------------------------------------------------
# Shared method base
# ---------------------------------------------------------------------------


class _BlackJAXSGMCMCMethod(InferenceMethod):
    """Base for the two BlackJAX SGMCMC methods.

    Subclasses define ``_method_name``, ``_method_priority``, and
    override :meth:`_build_algorithm` to plug in the specific BlackJAX
    kernel constructor (``blackjax.sgld`` or ``blackjax.sghmc``) with
    method-specific kwargs.
    """

    _method_name: str = ""
    _method_priority: int = 0

    @property
    def name(self) -> str:
        return self._method_name

    def supported_types(self) -> tuple[type, ...]:
        # Filter at the registry-level by Distribution; the SimpleModel +
        # ConditionallyIndependentLikelihood constraint is enforced in check().
        from ..core._distribution_base import Distribution
        return (Distribution,)

    @property
    def priority(self) -> int:
        return self._method_priority

    # -- feasibility checks --------------------------------------------------

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        """Require SimpleModel + ConditionallyIndependentLikelihood + batch_size."""
        from ..core.protocols import ConditionallyIndependentLikelihood

        if not is_simple_model(dist):
            return MethodInfo(
                feasible=False, method_name=self.name,
                description=(
                    f"{self.name} requires a SimpleModel; got "
                    f"{type(dist).__name__}."
                ),
            )
        if not isinstance(dist.likelihood, ConditionallyIndependentLikelihood):
            return MethodInfo(
                feasible=False, method_name=self.name,
                description=(
                    f"{self.name} requires model.likelihood to satisfy "
                    f"ConditionallyIndependentLikelihood; got "
                    f"{type(dist.likelihood).__name__}."
                ),
            )
        if "batch_size" not in kwargs:
            return MethodInfo(
                feasible=False, method_name=self.name,
                description=(
                    f"{self.name} requires an explicit batch_size= kwarg. "
                    f"There is no canonical default for a stochastic sampler."
                ),
            )
        return MethodInfo(feasible=True, method_name=self.name)

    # -- execution -----------------------------------------------------------

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        """Run the SGMCMC kernel; return an :class:`ApproximateDistribution`."""
        batch_size: int = kwargs["batch_size"]
        num_results: int = kwargs.get("num_results", 1000)
        num_warmup: int = kwargs.get("num_warmup", 0)
        step_size: float = kwargs.get("step_size", 1e-3)
        random_seed: int | PRNGKey = kwargs.get("random_seed", 0)
        with_replacement: bool = kwargs.get("with_replacement", False)

        # Build the minibatched random measure that supplies stochastic
        # grads. ``dist`` is a SimpleModel (validated by ``check()``);
        # unpack its prior + CIL likelihood for the random measure.
        measure = MinibatchedDistribution(
            dist.prior, dist.likelihood, observed,
            batch_size=batch_size,
            with_replacement=with_replacement,
        )

        # Build the BlackJAX-compatible gradient estimator + algorithm.
        grad_estimator = _build_grad_estimator(measure)
        algorithm = self._build_algorithm(grad_estimator, **kwargs)

        # Initial position from prior or user-supplied init.
        prior = dist.prior
        init = get_init_state(
            dist, kwargs.get("init"), random_seed=random_seed,
        )
        state = algorithm.init(init)

        # Iterate. The kernel itself jits within run_loop's step closure.
        positions = _run_sgmcmc_loop(
            algorithm, state, as_prng_key(random_seed),
            step_size, num_warmup, num_results,
        )

        # ``check()`` rejects any non-SimpleModel target, and a
        # SimpleModel prior is always a RecordDistribution, so
        # ``record_template`` is guaranteed here.
        chain = jnp.stack(positions, axis=0)
        record_template = prior.record_template
        return make_posterior(
            [chain], parents=(prior,), algorithm=self._method_name,
            auxiliary=None, record_template=record_template,
            num_results=num_results, num_warmup=num_warmup, num_chains=1,
        )

    # -- subclass hook -------------------------------------------------------

    def _build_algorithm(self, grad_estimator, **kwargs: Any):
        """Return a BlackJAX ``SamplingAlgorithm`` for this method.

        Subclasses override to supply the kernel-specific kwargs
        (e.g., ``num_integration_steps`` / ``alpha`` / ``beta`` for
        SGHMC).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Per-step loop (JIT inside)
# ---------------------------------------------------------------------------


def _run_sgmcmc_loop(
    algorithm,
    state,
    key: PRNGKey,
    step_size: float,
    num_warmup: int,
    num_results: int,
) -> list[Any]:
    """Run the SGMCMC kernel for ``num_warmup + num_results`` steps; discard warmup."""
    step = jax.jit(algorithm.step)

    def one_step(state, k):
        k_kernel, k_measure = jax.random.split(k)
        return step(k_kernel, state, k_measure, step_size)

    total = num_warmup + num_results
    keys = jax.random.split(key, total)
    positions: list[Any] = []
    for i in range(total):
        state = one_step(state, keys[i])
        if i >= num_warmup:
            positions.append(state)
    return positions


# ---------------------------------------------------------------------------
# Concrete methods
# ---------------------------------------------------------------------------


class BlackJAXSGLDMethod(_BlackJAXSGMCMCMethod):
    """BlackJAX Stochastic Gradient Langevin Dynamics.

    Kernel: :func:`blackjax.sgld`. Tier 41-50 (refinement-based:
    asymptotically exact as the step-size schedule decays). Priority
    45 — sits below every exact method, so does not auto-win over
    full-batch gradient methods; ``check()`` further requires the
    user to pass ``batch_size=`` for SGLD to be applicable at all.
    """

    _method_name = "blackjax_sgld"
    _method_priority = 45

    def _build_algorithm(self, grad_estimator, **kwargs: Any):
        return blackjax.sgld(grad_estimator)


class BlackJAXSGHMCMethod(_BlackJAXSGMCMCMethod):
    """BlackJAX Stochastic Gradient Hamiltonian Monte Carlo.

    Kernel: :func:`blackjax.sghmc`. Accepts the additional kwargs
    ``num_integration_steps`` (default 10), ``alpha`` (default 0.01),
    ``beta`` (default 0.0). Tier 41-50 by algorithm category
    (refinement-based: asymptotically exact as the step-size schedule
    decays), but registered at the opt-in-only sentinel ``priority=0``.
    Reasoning: SGHMC's ``check()`` is identical to ``blackjax_sgld``
    (same ``SimpleModel`` + ``ConditionallyIndependentLikelihood`` +
    ``batch_size=`` gate); with SGLD at 45, SGHMC is structurally
    unreachable in auto-dispatch. SGLD is also the better default —
    fewer dials (single ``step_size``) vs SGHMC's
    ``num_integration_steps`` / ``alpha`` / ``beta``. Callers who
    specifically want SGHMC pin ``method="blackjax_sghmc"``.
    """

    _method_name = "blackjax_sghmc"
    _method_priority = 0

    def _build_algorithm(self, grad_estimator, **kwargs: Any):
        num_integration_steps: int = kwargs.get("num_integration_steps", 10)
        alpha: float = kwargs.get("alpha", 0.01)
        beta: float = kwargs.get("beta", 0.0)
        return blackjax.sghmc(
            grad_estimator,
            num_integration_steps=num_integration_steps,
            alpha=alpha, beta=beta,
        )
