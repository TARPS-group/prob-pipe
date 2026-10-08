"""BlackJAX-backed stochastic-gradient MCMC methods.

Two :class:`~probpipe.core._dispatch.UnaryDispatchMethod` subclasses registered with
:data:`~probpipe.inference.inference_method_registry`:

* ``blackjax_sgld`` — Stochastic Gradient Langevin Dynamics
  ([Welling & Teh, 2011](https://www.icml-2011.org/papers/398_icmlpaper.pdf)).
* ``blackjax_sghmc`` — Stochastic Gradient Hamiltonian Monte Carlo
  ([Chen, Fox & Guestrin, 2014](https://arxiv.org/abs/1402.4102)).

Both methods consume the unnormalized conditional of a factored joint at
observed fields, reading its prior and likelihood factors (VI.6). The
likelihood is a kernel whose observations are conditionally independent and
which scores a subset of them, as a GLM likelihood does. Internally they
construct a :class:`~probpipe.MinibatchedDistribution` to produce
unbiased stochastic gradient estimates and feed it to the BlackJAX
kernel via the ``grad_estimator(position, measure_key)`` closure
convention — the per-step ``measure_key`` is passed through BlackJAX's
opaque ``minibatch`` slot.

Both methods require ``batch_size=`` to be passed, so SGMCMC applies only
when the user has opted into minibatching, as in
``condition_on(likelihood * prior, y=y, method="blackjax_sgld", batch_size=…)``.
``blackjax_sgld`` is registered at priority 45 and ``blackjax_sghmc`` is
opt-in-only; the method classes state why.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import blackjax
import jax
import jax.numpy as jnp

from ..core._dispatch import Feasibility
from ..custom_types import Array, PRNGKey
from ..distributions._capabilities import SupportsLogProb
from ..distributions._empirical import EmpiricalDistribution
from ..families._random_functions import RandomMeasure
from ..operations._condition import InferenceMethod, _UnnormalizedConditional
from ._approximate_distribution import make_posterior
from ._inference_utils import (
    as_prng_key,
    described,
    flat_unflatten,
    get_init_state,
    model_factors,
    run_seed,
    unfactored_model_reason,
)
from ._minibatch import MinibatchedDistribution, _reads_observations, _subset_scoring_reason

__all__ = ["BlackJAXSGHMCMethod", "BlackJAXSGLDMethod"]


# ---------------------------------------------------------------------------
# grad_estimator factory
# ---------------------------------------------------------------------------


def _build_grad_estimator(measure: RandomMeasure, unflatten: Callable[[Array], Any] | None = None):
    """Build the BlackJAX-compatible gradient estimator from a random measure.

    BlackJAX's SGMCMC kernels take a callable
    ``grad_estimator(position, *opaque)`` and pass whatever the user
    provides in the ``minibatch`` slot opaquely. We exploit that by
    passing a fresh PRNG key per step; the random measure samples its
    own minibatch internally via
    :meth:`~probpipe.MinibatchedDistribution._random_unnormalized_log_prob`.
    The kernel stays oblivious to the minibatching convention, so the
    same builder works for any future ``RandomMeasure`` subclass
    that supplies :class:`SupportsRandomUnnormalizedLogProb`. With
    *unflatten*, the position is a flat vector that it maps to a draw of the
    measure's parameters, and the gradient is taken in the flat coordinates.
    """
    rand_logp = measure._random_unnormalized_log_prob()

    def grad_estimator(position: Any, measure_key: PRNGKey) -> Any:
        realised_log_density = rand_logp._sample(measure_key)
        if unflatten is None:
            return jax.grad(realised_log_density)(position)
        return jax.grad(lambda flat: realised_log_density(unflatten(flat)))(position)

    return grad_estimator


# ---------------------------------------------------------------------------
# Shared method base
# ---------------------------------------------------------------------------


#: The ``method_options`` entries both SG-MCMC methods read.
_SGMCMC_OPTIONS = (
    "batch_size",
    "init",
    "num_results",
    "num_warmup",
    "random_seed",
    "step_size",
    "with_replacement",
)


class _BlackJAXSGMCMCMethod(InferenceMethod):
    """Base for the two BlackJAX SGMCMC methods.

    Subclasses define ``_method_name``, ``_method_priority``, and
    override :meth:`_build_algorithm` to plug in the specific BlackJAX
    kernel constructor (``blackjax.sgld`` or ``blackjax.sghmc``) with
    method-specific kwargs.
    """

    _method_name: str = ""
    _method_priority: int | None = None
    _method_options = _SGMCMC_OPTIONS

    @property
    def name(self) -> str:
        return self._method_name

    def supported_types(self) -> tuple[type, ...]:
        return (_UnnormalizedConditional,)

    @property
    def priority(self) -> int | None:
        return self._method_priority

    # -- feasibility checks --------------------------------------------------

    def check(self, target: Any, /, **kwargs: Any) -> Feasibility:
        """Require a joint whose likelihood scores a subset of its observations, and a batch size."""
        factors = model_factors(target)
        if factors is None:
            return Feasibility(feasible=False, description=unfactored_model_reason(target))
        if not isinstance(factors.prior, SupportsLogProb):
            return Feasibility(
                feasible=False,
                description=(
                    f"the prior must have a log-density (SupportsLogProb); got "
                    f"{described(factors.prior)}"
                ),
            )
        if not _reads_observations(factors.likelihood):
            return Feasibility(
                feasible=False,
                description=_subset_scoring_reason(factors.likelihood),
            )
        if "batch_size" not in kwargs:
            return Feasibility(
                feasible=False,
                description='batch_size is required; pass method_options={"batch_size": ...}',
                actionable=True,
            )
        return Feasibility(feasible=True)

    # -- execution -----------------------------------------------------------

    def execute(self, target: Any, /, **kwargs: Any) -> EmpiricalDistribution:
        """Run the SGMCMC kernel; return an :class:`~probpipe.EmpiricalDistribution`."""
        self._check_options(kwargs)
        factors = model_factors(target)
        batch_size: int = kwargs["batch_size"]
        num_results: int = kwargs.get("num_results", 1000)
        num_warmup: int = kwargs.get("num_warmup", 0)
        step_size: float = kwargs.get("step_size", 1e-3)
        random_seed: int | PRNGKey = run_seed(kwargs, self.name)
        with_replacement: bool = kwargs.get("with_replacement", False)

        # The minibatched random measure supplies the stochastic gradients from
        # the prior and likelihood factors ``check()`` validated.
        measure = MinibatchedDistribution(
            "measure",
            factors.prior,
            factors.likelihood,
            factors.observed,
            batch_size=batch_size,
            with_replacement=with_replacement,
        )

        # The chain moves in the prior's flat coordinates.
        grad_estimator = _build_grad_estimator(measure, flat_unflatten(factors.prior))
        algorithm = self._build_algorithm(grad_estimator, **kwargs)
        init = get_init_state(factors.prior, kwargs.get("init"), random_seed=random_seed)
        state = algorithm.init(init)

        # Iterate. The kernel itself jits within run_loop's step closure.
        positions = _run_sgmcmc_loop(
            algorithm,
            state,
            as_prng_key(random_seed),
            step_size,
            num_warmup,
            num_results,
        )

        chain = jnp.stack(positions, axis=0)
        return make_posterior(
            [chain],
            parents=(target,),
            method=self._method_name,
            annotations=None,
            event_spec=target.event_spec,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=1,
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
    """BlackJAX Stochastic Gradient Langevin Dynamics, registered as ``blackjax_sgld``.

    Kernel: :func:`blackjax.sgld`. Priority 45.

    Notes
    -----
    Refinement-based, so asymptotically exact as the step-size schedule
    decays. Ranked below every full-batch gradient method, so it never wins
    automatic selection over one.
    """

    _method_name = "blackjax_sgld"
    _method_priority = 45

    def _build_algorithm(self, grad_estimator, **kwargs: Any):
        return blackjax.sgld(grad_estimator)


class BlackJAXSGHMCMethod(_BlackJAXSGMCMCMethod):
    """BlackJAX Stochastic Gradient Hamiltonian Monte Carlo, registered as ``blackjax_sghmc``.

    Kernel: :func:`blackjax.sghmc`. Opt-in-only: runs only when the caller
    pins ``method="blackjax_sghmc"``. Accepts the additional kwargs
    ``num_integration_steps`` (default 10), ``alpha`` (default 0.01), and
    ``beta`` (default 0.0).

    Notes
    -----
    Refinement-based like SGLD, so asymptotically exact as the step-size
    schedule decays. Its ``check()`` is identical to that of
    ``blackjax_sgld``, so with SGLD ranked, SGHMC would never be selected
    automatically; ``priority=None`` makes that explicit. SGLD is also the
    better default, with a single ``step_size`` to tune against SGHMC's
    three kwargs.
    """

    _method_name = "blackjax_sghmc"
    _method_priority = None
    _method_options = (*_SGMCMC_OPTIONS, "alpha", "beta", "num_integration_steps")

    def _build_algorithm(self, grad_estimator, **kwargs: Any):
        num_integration_steps: int = kwargs.get("num_integration_steps", 10)
        alpha: float = kwargs.get("alpha", 0.01)
        beta: float = kwargs.get("beta", 0.0)
        return blackjax.sghmc(
            grad_estimator,
            num_integration_steps=num_integration_steps,
            alpha=alpha,
            beta=beta,
        )
