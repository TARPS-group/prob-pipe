"""pyabc SMC-ABC inference method for the registry."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..core._registry import MethodInfo
from ..core.record import RecordTemplate
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._registry import InferenceMethod

# Key for the simulated/observed vector in pyabc's summary-statistic dict.
_DATA_KEY = "y"


def _ensure_pyabc() -> Any:
    """Import pyabc or raise a helpful error pointing at the extra."""
    try:
        import pyabc
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "pyabc_smcabc requires the optional pyabc backend: "
            "pip install 'probpipe[pyabc]'"
        ) from e
    return pyabc


def _marginal_to_pyabc_rv(leaf: Any, pyabc: Any) -> Any:
    """Wrap a ProbPipe marginal as a ``pyabc.RV`` backed by the distribution
    itself — ``sample`` for ``rvs``, ``prob`` for ``pdf`` — with no scipy
    round-trip. ``cdf`` is unused by SMC-ABC.
    """
    from probpipe import prob, sample

    class _ProbPipeRV(pyabc.RV):
        def __init__(self, dist: Any):
            self._dist = dist
            self.distribution = None  # what RV.__getattr__ proxies to
            self.name = getattr(dist, "name", "rv")
            self.args, self.kwargs = (), {}

        def rvs(self, *args: Any, **kwargs: Any) -> float:
            key = jax.random.PRNGKey(int(np.random.randint(0, 2**31 - 1)))
            return float(np.asarray(sample(self._dist, key=key)))

        def pdf(self, x: Any, *args: Any, **kwargs: Any) -> float:
            return float(np.asarray(prob(self._dist, x)))

        def pmf(self, x: Any, *args: Any, **kwargs: Any) -> float:
            return self.pdf(x)

        def cdf(self, x: Any, *args: Any, **kwargs: Any) -> float:
            raise NotImplementedError("ProbPipeRV has no cdf (unused by SMC-ABC).")

        def copy(self) -> Any:
            return _ProbPipeRV(self._dist)

    return _ProbPipeRV(leaf)


def _build_pyabc_prior(prior: Any, pyabc: Any) -> tuple[Any, list[str]]:
    """Build a ``pyabc.Distribution`` and the component name order from a prior.

    Raises ``TypeError`` if *prior* is not a product of independent scalar
    marginals — so :meth:`PyABCSMCMethod.check` can probe feasibility by calling
    this and catching the failure.
    """
    components = getattr(prior, "components", None)
    if components is None:
        raise TypeError(
            "pyabc backend requires a ProductDistribution-style prior with "
            "independent named components."
        )
    names = list(prior.keys())
    rvs: dict[str, Any] = {}
    for name in names:
        comp = components[name]
        if isinstance(comp, dict):
            raise TypeError(
                f"pyabc backend requires flat scalar marginals; component "
                f"{name!r} is a nested product."
            )
        rvs[name] = _marginal_to_pyabc_rv(comp, pyabc)
    return pyabc.Distribution(**rvs), names


def _summarize(data: Any, summary_fn: Any) -> np.ndarray:
    """Apply ``summary_fn`` (if any) and flatten to a 1-D vector."""
    if summary_fn is not None:
        data = summary_fn(data)
    return np.asarray(data, dtype=float).ravel()


def _euclidean_distance(x: dict, x0: dict) -> float:
    return float(np.sqrt(np.sum((x[_DATA_KEY] - x0[_DATA_KEY]) ** 2)))


class PyABCSMCMethod(InferenceMethod):
    """pyabc SMC-ABC for SimpleGenerativeModel."""

    @property
    def name(self) -> str:
        return "pyabc_smcabc"

    def supported_types(self) -> tuple[type, ...]:
        from ..modeling._simple_generative import SimpleGenerativeModel
        return (SimpleGenerativeModel,)

    @property
    def priority(self) -> int:
        # Inexact (ABC quality is bounded by the summary statistics and the
        # acceptance tolerance); registered at 6 as the auto-dispatch pick for
        # a pure SimpleGenerativeModel.
        return 6

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        from ..modeling._simple_generative import SimpleGenerativeModel

        if not isinstance(dist, SimpleGenerativeModel):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires SimpleGenerativeModel")
        # Build the prior to probe what execute() actually needs, so check() and
        # execute() can't disagree (a feasible check then a crash). pyabc is
        # importable here: registration is gated on it.
        try:
            _build_pyabc_prior(dist["parameters"], _ensure_pyabc())
        except (TypeError, NotImplementedError) as e:
            return MethodInfo(feasible=False, method_name=self.name, description=str(e))
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        """Run SMC-ABC and return a weighted posterior.

        Parameters
        ----------
        dist : SimpleGenerativeModel
            Prior (a product of independent scalar marginals) plus a
            ``GenerativeLikelihood`` simulator.
        observed : array-like
            Observed data; flattened (after ``summary_fn``) to the target vector.
        n_particles : int, default 100
            SMC population size.
        max_populations : int, default 4
            Number of SMC generations.
        eps_alpha : float, default 0.5
            ``QuantileEpsilon`` alpha (acceptance-quantile schedule).
        random_seed : int, default 0
            Seeds both the simulator RNG and pyabc's proposal RNG (numpy global
            state is snapshotted and restored around the run), so repeated calls
            are reproducible.
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
            their (non-resampled) SMC importance weights.
        """
        pyabc = _ensure_pyabc()
        from pyabc.sampler import SingleCoreSampler

        prior = dist["parameters"]
        simulator = dist["data"]
        pyabc_prior, names = _build_pyabc_prior(prior, pyabc)

        n_particles = int(kwargs.get("n_particles", 100))
        max_populations = int(kwargs.get("max_populations", 4))
        eps_alpha = float(kwargs.get("eps_alpha", 0.5))
        random_seed = int(kwargs.get("random_seed", 0))
        summary_fn = kwargs.get("summary_fn")
        distance_fn = kwargs.get("distance_fn") or _euclidean_distance
        sampler = kwargs.get("sampler") or SingleCoreSampler()

        x0 = {_DATA_KEY: _summarize(jnp.atleast_1d(jnp.asarray(observed))[None, :], summary_fn)}

        # Per-call RNG seeding the stochastic simulator; pyabc's own proposal RNG
        # is seeded via the numpy global state below.
        rng = np.random.default_rng(random_seed)

        def model_fn(parameters: dict) -> dict:
            vec = jnp.asarray([float(parameters[n]) for n in names])
            key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
            raw = jnp.atleast_2d(simulator.generate_data(vec, 1, key=key)[0])
            return {_DATA_KEY: _summarize(raw, summary_fn)}

        abc = pyabc.ABCSMC(
            model_fn, pyabc_prior, distance_fn,
            population_size=n_particles, sampler=sampler,
            eps=pyabc.QuantileEpsilon(alpha=eps_alpha),
        )
        # pyabc draws proposals from numpy's global RNG; seed it for a
        # reproducible run and restore the caller's state afterwards.
        np_state = np.random.get_state()
        np.random.seed(random_seed)
        try:
            abc.new("sqlite://", x0)  # in-memory history; nothing written to disk
            history = abc.run(max_nr_populations=max_populations)
        finally:
            np.random.set_state(np_state)

        # SMC-ABC returns importance-weighted particles; keep the weights rather
        # than resampling. Reorder columns to the prior's flat event order.
        df, weights = history.get_distribution(m=0, t=history.max_t)
        particles = jnp.asarray(df.reindex(columns=names).to_numpy(dtype=float))
        weights = np.asarray(weights, dtype=float)

        # Name the columns by component so draws()/mean() return Records keyed by
        # parameter name (the prior's marginals are scalar).
        template = RecordTemplate(**{n: tuple(prior.components[n].event_shape) for n in names})

        return make_posterior(
            [particles], parents=(prior,), algorithm="pyabc_smcabc",
            weights=jnp.asarray(weights / weights.sum()), record_template=template,
            n_particles=n_particles, max_populations=max_populations, eps_alpha=eps_alpha,
        )
