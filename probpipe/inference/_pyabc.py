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


def _marginal_to_pyabc_rv(leaf: Any) -> Any:
    """Map one TFP-backed prior marginal to a ``pyabc.RV``."""
    import pyabc

    tfp_dist = getattr(leaf, "_tfp_dist", None)
    if tfp_dist is None:
        raise TypeError(
            f"pyabc backend requires TFP-backed marginals; component "
            f"{getattr(leaf, 'name', leaf)!r} has no _tfp_dist."
        )

    family = type(tfp_dist).__name__
    params = tfp_dist.parameters
    if family == "Uniform":  # scipy/pyabc uniform is (loc, scale=width)
        low, high = float(params["low"]), float(params["high"])
        return pyabc.RV("uniform", low, high - low)
    if family == "Normal":
        return pyabc.RV("norm", float(params["loc"]), float(params["scale"]))
    if family == "Beta":
        return pyabc.RV(
            "beta", float(params["concentration1"]), float(params["concentration0"])
        )
    if family == "Gamma":  # tfd rate -> scipy scale=1/rate
        return pyabc.RV(
            "gamma", float(params["concentration"]),
            loc=0.0, scale=1.0 / float(params["rate"]),
        )
    raise NotImplementedError(
        f"No pyabc RV mapping for TFP marginal family {family!r}. "
        "Add it to _marginal_to_pyabc_rv."
    )


def _build_pyabc_prior(prior: Any) -> tuple[Any, list[str]]:
    """Build a ``pyabc.Distribution`` and the component name order from a prior."""
    import pyabc

    components = getattr(prior, "components", None)
    if components is None:
        raise TypeError(
            "pyabc backend requires a ProductDistribution-style prior with "
            "independent named components."
        )
    names = list(prior.keys())
    rvs = {name: _marginal_to_pyabc_rv(components[name]) for name in names}
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
        # Tier 1-10 (inexact; ABC quality is bounded by the summary statistics
        # and tolerance), and the only auto-dispatchable method for a pure
        # SimpleGenerativeModel.
        return 6

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        from ..modeling._simple_generative import SimpleGenerativeModel
        if not isinstance(dist, SimpleGenerativeModel):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires SimpleGenerativeModel")
        if not hasattr(dist["parameters"], "_tfp_dist"):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Prior must be TFP-backed for pyabc")
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        import pyabc
        from pyabc.sampler import SingleCoreSampler

        prior = dist["parameters"]
        simulator = dist["data"]
        pyabc_prior, names = _build_pyabc_prior(prior)

        n_particles = kwargs.get("n_particles", 100)
        max_populations = kwargs.get("max_populations", kwargs.get("n_rounds", 4))
        eps_alpha = kwargs.get("eps_alpha", 0.5)
        random_seed = kwargs.get("random_seed", 0)
        summary_fn = kwargs.get("summary_fn")
        distance_fn = kwargs.get("distance_fn") or _euclidean_distance
        # SingleCoreSampler default: pyabc's multicore samplers fork(), which
        # deadlocks alongside this process's JAX threads (cf. the PyMC backend's
        # spawn handling). Callers wanting parallelism pass an explicit sampler.
        sampler = kwargs.get("sampler") or SingleCoreSampler()

        y_obs = jnp.atleast_1d(jnp.asarray(observed))
        x0 = {_DATA_KEY: _summarize(y_obs[None, :], summary_fn)}

        # pyabc owns the proposal RNG; this only seeds the stochastic simulator.
        rng = np.random.default_rng(random_seed)

        def model_fn(parameters: dict) -> dict:
            vec = jnp.asarray([float(parameters[n]) for n in names], dtype=jnp.float32)
            key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
            raw = jnp.atleast_2d(simulator.generate_data(vec, 1, key=key)[0])
            return {_DATA_KEY: _summarize(raw, summary_fn)}

        abc = pyabc.ABCSMC(
            model_fn, pyabc_prior, distance_fn,
            population_size=int(n_particles), sampler=sampler,
            eps=pyabc.QuantileEpsilon(alpha=float(eps_alpha)),
        )
        abc.new("sqlite://", x0)  # in-memory history; nothing written to disk
        history = abc.run(max_nr_populations=int(max_populations))

        # SMC-ABC returns importance-weighted particles; keep the weights rather
        # than resampling. Reorder columns to the prior's flat event order.
        df, weights = history.get_distribution(m=0, t=history.max_t)
        particles = jnp.asarray(df.reindex(columns=names).to_numpy(dtype=float))
        weights = np.asarray(weights, dtype=float)

        # Name the columns by component so ``draws()`` / ``mean()`` return Records
        # keyed by parameter name (the prior's marginals are scalar).
        template = RecordTemplate(**{n: tuple(prior.components[n].event_shape) for n in names})

        return make_posterior(
            [particles], parents=(prior,), algorithm="pyabc_smcabc",
            weights=jnp.asarray(weights / weights.sum()), record_template=template,
            n_particles=n_particles, max_populations=max_populations, eps_alpha=eps_alpha,
        )
