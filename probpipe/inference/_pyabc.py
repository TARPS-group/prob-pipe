"""pyabc SMC-ABC inference method for the registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pyabc
from pyabc.sampler import SingleCoreSampler

from ..core._dispatch import Feasibility
from ..custom_types import PRNGKey
from ..distributions._capabilities import SupportsConditionalSampling
from ..distributions._conditional import ConditionalDistribution
from ..distributions._empirical import EmpiricalDistribution
from ..operations._condition import InferenceMethod, _UnnormalizedConditional
from ._approximate_distribution import make_posterior
from ._inference_utils import (
    flat_unflatten,
    flat_vector,
    integer_seed,
    model_factors,
    parameter_given,
    run_seed,
)

if TYPE_CHECKING:
    from xarray import DataTree

    from ..distributions._distribution import Distribution


# Key under which the simulated/observed vector lives in pyabc's sumstat dict.
_DATA_KEY = "y"

# pyabc passes summary statistics as ``{_DATA_KEY: vector}`` dicts.
_SumStat = Mapping[str, Any]
_SummaryFn = Callable[[np.ndarray], np.ndarray]
_DistanceFn = Callable[[_SumStat, _SumStat], float]


def _flat_key(i: int) -> str:
    """pyabc parameter name for flat position ``i`` of the parameter vector."""
    return f"p{i}"


class PyABCDistribution(pyabc.Distribution):
    """A pyabc prior that samples and scores a ProbPipe prior jointly over its
    *flattened* parameter vector.

    Sampling (:meth:`rvs`) and density (:meth:`pdf`) read the prior's draws
    and density through its flat vector, the layout III.7 fixes for a numeric
    law, so correlated and multivariate priors are supported. The object is a dict-like, picklable ``pyabc.Distribution`` keyed
    by flat ``pN`` names, which give pyabc the parameter names and the flat
    columns its perturbation kernel operates on.
    """

    def __init__(self, prior: Distribution, key: PRNGKey):
        """Wrap *prior* as a joint pyabc prior over its flat parameter vector.

        ``key`` is the JAX key threaded through :meth:`rvs` (split per draw). The
        per-position ``pN`` keys give pyabc the parameter names
        (``get_parameter_names``) and the flat columns its perturbation kernel
        operates on; sampling and scoring go through the prior's flat vector.
        """
        self._prior = prior
        self._unflatten = flat_unflatten(prior)
        self._key = key
        self._d = prior.event_spec.spec.vector_size
        super().__init__(**{_flat_key(i): pyabc.RV("uniform", 0, 1) for i in range(self._d)})

    def rvs(self, *args: Any, **kwargs: Any) -> pyabc.Parameter:
        """One joint draw from the prior, as a flat-keyed pyabc ``Parameter``."""
        self._key, sub = jax.random.split(self._key)
        vec = np.asarray(flat_vector(self._prior._sample(sub)))
        return pyabc.Parameter(**{_flat_key(i): float(vec[i]) for i in range(self._d)})

    def pdf(self, x: Mapping[str, float]) -> float:
        """Joint prior density at *x*, the flat parameter vector reassembled
        from its ``pN`` keys and scored as the prior's draw it lays out."""
        vec = jnp.asarray([x[_flat_key(i)] for i in range(self._d)])
        return float(np.exp(np.asarray(self._prior._log_prob(self._unflatten(vec)))))


def _euclidean_distance(x: _SumStat, x0: _SumStat) -> float:
    """Euclidean distance between the simulated and observed summary vectors."""
    return float(np.linalg.norm(np.asarray(x[_DATA_KEY]) - np.asarray(x0[_DATA_KEY])))


def _summarize(data: Any, summary_fn: _SummaryFn | None) -> np.ndarray:
    """Apply ``summary_fn`` (if any) and flatten to a 1-D vector."""
    if summary_fn is not None:
        data = summary_fn(data)
    return np.asarray(data, dtype=float).ravel()


def _smc_diagnostics(history: Any) -> DataTree:
    """Per-generation SMC-ABC convergence trajectory as an ArviZ ``DataTree``.

    Builds a ``smc_diagnostics`` group indexed by generation, holding the
    epsilon (acceptance-threshold) schedule, the sample attempts, the accepted
    particles, and the acceptance rate; ``total_nr_simulations`` is a scalar
    attribute. It is stored under ``arviz/`` in the result's annotations.
    """
    import xarray as xr

    populations = history.get_all_populations()
    populations = populations[populations["t"] >= 0]  # drop the prior pre-sample row
    samples = populations["samples"].to_numpy()
    particles = populations["particles"].to_numpy()
    diagnostics = xr.Dataset(
        {
            "epsilon": ("generation", populations["epsilon"].to_numpy()),
            "samples": ("generation", samples),
            "particles": ("generation", particles),
            "acceptance_rate": ("generation", particles / samples),
        },
        coords={"generation": populations["t"].to_numpy()},
        attrs={"total_nr_simulations": int(history.total_nr_simulations)},
    )
    annotations = xr.DataTree()
    annotations["smc_diagnostics"] = xr.DataTree(dataset=diagnostics)
    return annotations


class PyABCSMCMethod(InferenceMethod):
    """pyabc SMC-ABC, registered as ``pyabc_smcabc`` at priority 6.

    Applies to the unnormalized conditional of a factored joint at observed
    fields, whose likelihood is a kernel that samples, the simulator, and
    whose prior can flatten, sample, and score jointly.

    Notes
    -----
    ABC quality is bounded by the summary statistics and the acceptance
    tolerance, so it ranks below every likelihood-based method.
    """

    _method_options = (
        "distance_fn",
        "eps",
        "eps_alpha",
        "max_populations",
        "max_total_nr_simulations",
        "max_walltime",
        "min_acceptance_rate",
        "minimum_epsilon",
        "n_particles",
        "random_seed",
        "sampler",
        "summary_fn",
        "transitions",
    )

    @property
    def name(self) -> str:
        return "pyabc_smcabc"

    def supported_types(self) -> tuple[type, ...]:
        return (_UnnormalizedConditional,)

    @property
    def priority(self) -> int:
        return 6

    def check(self, target: Any, /, **kwargs: Any) -> Feasibility:
        """Whether the target conditions a joint whose likelihood simulates and whose prior scores."""
        factors = model_factors(target)
        if factors is None:
            return Feasibility(
                feasible=False,
                description=(
                    "Requires a factored joint at observed values of its fields, whose other "
                    "factors form the prior"
                ),
            )
        if not (
            isinstance(factors.likelihood, ConditionalDistribution)
            and isinstance(factors.likelihood, SupportsConditionalSampling)
        ):
            return Feasibility(
                feasible=False,
                description=(
                    "Requires a likelihood kernel that samples, the simulator; got "
                    f"{type(factors.likelihood).__name__}"
                ),
            )
        # Feasible means the prior can flatten, sample, *and* score jointly.
        # Build the backing distribution, then score one in-support draw — this
        # exercises log_prob, so a sampleable-but-density-less prior is caught
        # here rather than crashing later in pyabc's weight computation.
        try:
            pyabc_prior = PyABCDistribution(factors.prior, jax.random.PRNGKey(0))
            density = pyabc_prior.pdf(pyabc_prior.rvs())
        except Exception as e:
            return Feasibility(feasible=False, description=str(e))
        if not np.isfinite(density):
            return Feasibility(
                feasible=False,
                description="prior has no usable joint density",
            )
        return Feasibility(feasible=True)

    def execute(self, target: Any, /, **kwargs: Any) -> EmpiricalDistribution:
        """Run SMC-ABC and return a weighted posterior.

        Parameters
        ----------
        target : Distribution
            The unnormalized conditional of a factored joint at its observed
            fields: a prior that flattens to a parameter vector and carries a
            joint density, a likelihood kernel that samples, the simulator, and
            the observed value, flattened (after ``summary_fn``) to the target
            vector.
        n_particles : int, default 100
            SMC population size.
        max_populations : int, default 4
            Number of SMC generations.
        eps_alpha : float, default 0.5
            ``QuantileEpsilon`` alpha for the default epsilon schedule; ignored
            if ``eps`` is given.
        eps : pyabc epsilon, optional
            Epsilon (acceptance-threshold) strategy. Defaults to
            ``QuantileEpsilon(alpha=eps_alpha)``; pass any pyabc epsilon (e.g.
            ``MedianEpsilon``, ``ListEpsilon``, ``AcceptanceRateScheduler``) to
            override.
        transitions : pyabc transition, optional
            Perturbation kernel. Defaults to pyabc's own (a multivariate-normal
            transition); pass a custom one to override.
        minimum_epsilon, min_acceptance_rate, max_total_nr_simulations, max_walltime : optional
            Additional stopping criteria forwarded to ``ABCSMC.run`` alongside
            ``max_populations`` (whichever is hit first stops the run); pyabc's
            defaults apply when omitted.
        random_seed : int, default 0
            Seeds the JAX keys threaded into the prior and simulator and pyabc's
            own (numpy-global) proposal RNG, so repeated calls are reproducible.
        summary_fn : callable, optional
            ``(batch, dim) -> (batch, summary_dim)`` applied to simulated and
            observed data before the distance.
        distance_fn : callable, optional
            ``(x, x0) -> float`` over the ``{"y": vector}`` sumstat dicts;
            defaults to Euclidean. Note: summary statistics are passed as a
            single flat vector under the ``"y"`` key, so pyabc's multi-statistic
            adaptive/weighted distances are not used; supply a custom
            ``summary_fn``/``distance_fn`` pair for bespoke weighting.
        sampler : pyabc sampler, optional
            Defaults to ``SingleCoreSampler``. pyabc's multicore samplers
            ``fork()``, which can deadlock alongside JAX's threads (the same
            reason the PyMC backend avoids forking); pass an explicit sampler to
            opt into local-multicore parallelism.

        Returns
        -------
        EmpiricalDistribution
            The final population's particles, keyed by parameter name, carrying
            their (non-resampled) SMC importance weights. The per-generation
            convergence trajectory (epsilon schedule, sample / particle counts,
            acceptance rate) is attached as a ``smc_diagnostics`` group on
            ``arviz_data``.

        Raises
        ------
        TypeError
            If a keyword is none of the options above.
        """
        self._check_options(kwargs)
        factors = model_factors(target)
        prior = factors.prior
        simulator = factors.likelihood

        n_particles = int(kwargs.get("n_particles", 100))
        max_populations = int(kwargs.get("max_populations", 4))
        eps_alpha = float(kwargs.get("eps_alpha", 0.5))
        random_seed = integer_seed(run_seed(kwargs, self.name))
        summary_fn: _SummaryFn | None = kwargs.get("summary_fn")
        distance_fn: _DistanceFn = kwargs.get("distance_fn") or _euclidean_distance
        sampler = kwargs.get("sampler") or SingleCoreSampler()
        eps = kwargs.get("eps")
        if eps is None:
            eps = pyabc.QuantileEpsilon(alpha=eps_alpha)
        transitions = kwargs.get("transitions")

        prior_key, sim_key0 = jax.random.split(jax.random.PRNGKey(random_seed))
        pyabc_prior = PyABCDistribution(prior, prior_key)
        d = pyabc_prior._d

        x0 = {_DATA_KEY: _summarize(flat_vector(factors.observed)[None, :], summary_fn)}
        unflatten = flat_unflatten(prior)

        sim_key = [sim_key0]  # threaded per simulator call (no numpy reseed)

        def model_fn(parameters: Mapping[str, float]) -> _SumStat:
            vec = jnp.asarray([float(parameters[_flat_key(i)]) for i in range(d)])
            sim_key[0], sub = jax.random.split(sim_key[0])
            given = parameter_given(factors, unflatten(vec))
            raw = flat_vector(simulator._conditional_sample(given, sub))[None, :]
            return {_DATA_KEY: _summarize(raw, summary_fn)}

        # Known limitation: pyabc perturbs in the prior's *constrained* space, so
        # for bounded parameters it can propose out-of-support points (density 0
        # — correct but wasteful). Perturbing in unconstrained space through the
        # prior's constraint bijectors would avoid these proposals.
        abc = pyabc.ABCSMC(
            model_fn,
            pyabc_prior,
            distance_fn,
            population_size=n_particles,
            sampler=sampler,
            eps=eps,
            transitions=transitions,
        )
        # Forward any caller-supplied stopping criteria to run(); pyabc's own
        # defaults apply for those omitted.
        run_kwargs = {
            k: kwargs[k]
            for k in (
                "minimum_epsilon",
                "min_acceptance_rate",
                "max_total_nr_simulations",
                "max_walltime",
            )
            if k in kwargs
        }
        # pyabc draws its perturbations from numpy's global RNG; seed it for a
        # reproducible run and restore the caller's state afterwards.
        np_state = np.random.get_state()
        np.random.seed(random_seed)
        try:
            abc.new("sqlite://", x0)  # in-memory history; nothing written to disk
            history = abc.run(max_nr_populations=max_populations, **run_kwargs)
        finally:
            np.random.set_state(np_state)

        # Final population: flat columns p0..p{d-1}, with SMC importance weights.
        df, weights = history.get_distribution(m=0, t=history.max_t)
        flat = df.reindex(columns=[_flat_key(i) for i in range(d)]).to_numpy(dtype=float)
        weights = np.asarray(weights, dtype=float)

        # Lift the flat columns back to name-keyed Records via the target's declaration.
        return make_posterior(
            [jnp.asarray(flat)],
            parents=(target,),
            method="pyabc_smcabc",
            weights=jnp.asarray(weights / weights.sum()),
            event_spec=target.event_spec,
            field_order=list(target.event_spec.components),
            annotations=_smc_diagnostics(history),
            n_particles=n_particles,
            max_populations=max_populations,
            eps_alpha=eps_alpha,
        )
