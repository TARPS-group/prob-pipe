"""pyabc SMC-ABC inference method for the registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pyabc
from pyabc.sampler import SingleCoreSampler

from ..core.ops import log_prob, sample
from ..core.record import RecordTemplate
from ..custom_types import PRNGKey
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._registry import InferenceMethod, MethodInfo

if TYPE_CHECKING:
    from xarray import DataTree

    from .. import NumericRecordDistribution

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

    Sampling (:meth:`rvs`) and density (:meth:`pdf`) go through
    ``prior.as_flat_distribution()``, so correlated and multivariate priors are
    supported. The object is a dict-like, picklable ``pyabc.Distribution`` keyed
    by flat ``pN`` names, which give pyabc the parameter names and the flat
    columns its perturbation kernel operates on.
    """

    def __init__(self, prior: NumericRecordDistribution, key: PRNGKey):
        """Wrap *prior* as a joint pyabc prior over its flat parameter vector.

        ``key`` is the JAX key threaded through :meth:`rvs` (split per draw). The
        per-position ``pN`` keys give pyabc the parameter names
        (``get_parameter_names``) and the flat columns its perturbation kernel
        operates on; sampling and scoring go through the joint flat view.
        """
        self._prior = prior
        self._flat = prior.as_flat_distribution()
        self._key = key
        self._d = self._flat.flat_size
        super().__init__(**{_flat_key(i): pyabc.RV("uniform", 0, 1) for i in range(self._d)})

    def rvs(self, *args: Any, **kwargs: Any) -> pyabc.Parameter:
        """One joint draw from the prior, as a flat-keyed pyabc ``Parameter``."""
        self._key, sub = jax.random.split(self._key)
        vec = np.asarray(sample(self._flat, key=sub)).ravel()
        return pyabc.Parameter(**{_flat_key(i): float(vec[i]) for i in range(self._d)})

    def pdf(self, x: Mapping[str, float]) -> float:
        """Joint prior density at *x*, the flat parameter vector reassembled
        from its ``pN`` keys and scored through the flat view's ``log_prob``."""
        vec = jnp.asarray([x[_flat_key(i)] for i in range(self._d)])
        return float(np.exp(np.asarray(log_prob(self._flat, vec))))


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
    attribute. Recovered from ``dist.auxiliary`` after a run.
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
    auxiliary = xr.DataTree()
    auxiliary["smc_diagnostics"] = xr.DataTree(dataset=diagnostics)
    return auxiliary


class PyABCSMCMethod(InferenceMethod):
    """pyabc SMC-ABC for :class:`~probpipe.modeling.SimpleGenerativeModel`."""

    @property
    def name(self) -> str:
        return "pyabc_smcabc"

    def supported_types(self) -> tuple[type, ...]:
        # lazy: avoid an inference->modeling import cycle
        from ..modeling._simple_generative import SimpleGenerativeModel
        return (SimpleGenerativeModel,)

    @property
    def priority(self) -> int:
        # Inexact (ABC quality is bounded by the summary statistics and the
        # acceptance tolerance), so it sits in the low-priority auto-dispatch
        # band for a pure SimpleGenerativeModel.
        return 6

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        # lazy: avoid an inference->modeling import cycle
        from ..modeling._simple_generative import SimpleGenerativeModel
        if not isinstance(dist, SimpleGenerativeModel):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires SimpleGenerativeModel")
        prior = dist["parameters"]
        # Feasible means the prior can flatten, sample, *and* score jointly.
        # Build the backing distribution, then score one in-support draw — this
        # exercises log_prob, so a sampleable-but-density-less prior is caught
        # here rather than crashing later in pyabc's weight computation.
        try:
            pyabc_prior = PyABCDistribution(prior, jax.random.PRNGKey(0))
            density = pyabc_prior.pdf(pyabc_prior.rvs())
        except Exception as e:
            return MethodInfo(feasible=False, method_name=self.name, description=str(e))
        if not np.isfinite(density):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="prior has no usable joint density")
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        """Run SMC-ABC and return a weighted posterior.

        Parameters
        ----------
        dist : SimpleGenerativeModel
            Prior (any distribution that flattens to a parameter vector and
            carries a joint density) plus a ``GenerativeLikelihood`` simulator.
        observed : array-like
            Observed data; flattened (after ``summary_fn``) to the target vector.
        n_particles : int, default 100
            SMC population size.
        max_populations : int, default 4
            Number of SMC generations.
        eps_alpha : float, default 0.5
            ``QuantileEpsilon`` alpha (acceptance-quantile schedule).
        random_seed : int, default 0
            Seeds the JAX keys threaded into the prior and simulator and pyabc's
            own (numpy-global) proposal RNG, so repeated calls are reproducible.
        summary_fn : callable, optional
            ``(batch, dim) -> (batch, summary_dim)`` applied to simulated and
            observed data before the distance.
        distance_fn : callable, optional
            ``(x, x0) -> float`` over the ``{"y": vector}`` sumstat dicts;
            defaults to Euclidean.
        sampler : pyabc sampler, optional
            Defaults to ``SingleCoreSampler``. pyabc's multicore samplers
            ``fork()``, which can deadlock alongside JAX's threads (the same
            reason the PyMC backend avoids forking); pass an explicit sampler to
            opt into local-multicore parallelism.

        Returns
        -------
        ApproximateDistribution
            The final population's particles, keyed by parameter name, carrying
            their (non-resampled) SMC importance weights. The per-generation
            convergence trajectory (epsilon schedule, sample / particle counts,
            acceptance rate) is attached as a ``smc_diagnostics`` group on
            ``auxiliary``.
        """
        prior = dist["parameters"]
        simulator = dist["data"]

        n_particles = int(kwargs.get("n_particles", 100))
        max_populations = int(kwargs.get("max_populations", 4))
        eps_alpha = float(kwargs.get("eps_alpha", 0.5))
        random_seed = int(kwargs.get("random_seed", 0))
        summary_fn: _SummaryFn | None = kwargs.get("summary_fn")
        distance_fn: _DistanceFn = kwargs.get("distance_fn") or _euclidean_distance
        sampler = kwargs.get("sampler") or SingleCoreSampler()

        prior_key, sim_key0 = jax.random.split(jax.random.PRNGKey(random_seed))
        pyabc_prior = PyABCDistribution(prior, prior_key)
        d = pyabc_prior._d

        x0 = {_DATA_KEY: _summarize(jnp.atleast_1d(jnp.asarray(observed))[None, :], summary_fn)}

        sim_key = [sim_key0]  # threaded per simulator call (no numpy reseed)

        def model_fn(parameters: Mapping[str, float]) -> _SumStat:
            vec = jnp.asarray([float(parameters[_flat_key(i)]) for i in range(d)])
            sim_key[0], sub = jax.random.split(sim_key[0])
            raw = jnp.atleast_2d(simulator.generate_data(vec, 1, key=sub)[0])
            return {_DATA_KEY: _summarize(raw, summary_fn)}

        # Known limitation: pyabc perturbs in the prior's *constrained* space, so
        # for bounded parameters it can propose out-of-support points (density 0
        # — correct but wasteful). Follow-up (#238): perturb in unconstrained
        # space via the prior's constraint bijectors.
        abc = pyabc.ABCSMC(
            model_fn, pyabc_prior, distance_fn,
            population_size=n_particles, sampler=sampler,
            eps=pyabc.QuantileEpsilon(alpha=eps_alpha),
        )
        # pyabc draws its perturbations from numpy's global RNG; seed it for a
        # reproducible run and restore the caller's state afterwards.
        np_state = np.random.get_state()
        np.random.seed(random_seed)
        try:
            abc.new("sqlite://", x0)  # in-memory history; nothing written to disk
            history = abc.run(max_nr_populations=max_populations)
        finally:
            np.random.set_state(np_state)

        # Final population: flat columns p0..p{d-1}, with SMC importance weights.
        df, weights = history.get_distribution(m=0, t=history.max_t)
        flat = df.reindex(columns=[_flat_key(i) for i in range(d)]).to_numpy(dtype=float)
        weights = np.asarray(weights, dtype=float)

        # Lift the flat columns back to name-keyed Records via the prior's layout.
        template = RecordTemplate(**dict(prior.event_shapes))
        return make_posterior(
            [jnp.asarray(flat)], parents=(dist,), algorithm="pyabc_smcabc",
            weights=jnp.asarray(weights / weights.sum()), record_template=template,
            field_order=list(prior.event_shapes), auxiliary=_smc_diagnostics(history),
            n_particles=n_particles, max_populations=max_populations, eps_alpha=eps_alpha,
        )
