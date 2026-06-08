"""BayesFlow amortized-SBI backend for ProbPipe.

Trains amortized conditional posterior estimators -- NPE (neural posterior
estimation), FMPE (flow-matching) and CMPE (consistency-model) -- with BayesFlow
(keras-on-JAX) and wraps the trained estimator as a
:class:`~probpipe.core.protocols.SupportsConditioning` distribution.
``condition_on(model, observed)`` then draws from ``p(theta | observed)`` in a
single forward pass through the trained network: no MCMC, no gradient bridge,
and no prior translation (the prior is used only to draw ``theta`` at train
time via the :func:`~probpipe.sample` op).

BayesFlow / keras is imported lazily on first use, so ``import probpipe`` does
not pull keras.
"""

from __future__ import annotations

import os
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..core._record_array import NumericRecordArray
from ..core.distribution import Distribution
from ..core.node import workflow_function
from ..core.ops import sample as _sample_op
from ..core.protocols import GenerativeLikelihood, SupportsConditioning
from ..custom_types import Array, PRNGKey
from ._approximate_distribution import ApproximateDistribution, make_posterior

if TYPE_CHECKING:
    # Type-only imports of the optional ``bayesflow`` dependency: stringized by
    # ``from __future__ import annotations`` and never evaluated at runtime
    # (bayesflow is loaded only inside ``_ensure_bayesflow``).
    from bayesflow import Adapter, ContinuousApproximator
    from bayesflow.networks import InferenceNetwork

AmortizedMethod = Literal["npe", "fmpe", "cmpe"]
# Offline-simulation execution backend; values mirror ``WorkflowFunction``'s
# dispatch names ("jax" = vmap the simulator, "sequential" = eager per-draw loop).
SimBackend = Literal["jax", "sequential"]

# Reserved key for the simulated observation in BayesFlow's named-array dicts.
_OBSERVATION_KEY = "observation"

_bayesflow_module: ModuleType | None = None


def _ensure_bayesflow() -> ModuleType:
    """Import BayesFlow on the keras JAX backend, or raise a helpful error.

    Imported lazily (and cached) so that ``import probpipe`` does not load keras.
    """
    global _bayesflow_module
    if _bayesflow_module is not None:
        return _bayesflow_module
    # keras resolves its backend at import time from ``KERAS_BACKEND``; pin jax
    # before keras / bayesflow load. ``setdefault`` leaves an explicit choice.
    os.environ.setdefault("KERAS_BACKEND", "jax")
    try:
        import bayesflow as bf
        import keras
    except ImportError as e:
        raise ImportError(
            "BayesFlow is required for learn_amortized_posterior: "
            "pip install probpipe[bayesflow]"
        ) from e
    if keras.backend.backend() != "jax":
        # ``setdefault`` cannot re-bind an already-imported keras, so a process
        # that imported keras with another backend first lands here.
        raise ImportError(
            "ProbPipe's BayesFlow backend requires the keras JAX backend, but "
            f"keras reports {keras.backend.backend()!r}. Set KERAS_BACKEND=jax "
            "before keras or bayesflow is first imported."
        )
    _bayesflow_module = bf
    return bf


# ---------------------------------------------------------------------------
# Bridge helpers: ProbPipe prior / simulator <-> BayesFlow named-array dicts
# ---------------------------------------------------------------------------


def _make_inference_network(
    bf: ModuleType, method: AmortizedMethod, total_steps: int, *, event_size: int
) -> InferenceNetwork:
    """The BayesFlow inference network for each amortized method.

    NPE's coupling flow splits the parameter vector in two and so needs at least
    two parameters; for a one-parameter model (``event_size < 2``) the NPE default
    falls back to a flow-matching network, which has no such constraint (the
    posterior is still exposed under ``method="npe"``).
    """
    if method == "npe":
        if event_size < 2:
            return bf.networks.FlowMatching()
        return bf.networks.CouplingFlow()
    if method == "fmpe":
        return bf.networks.FlowMatching()
    if method == "cmpe":
        return bf.networks.ConsistencyModel(total_steps=total_steps)
    raise ValueError(
        f"Unknown amortized SBI method: {method!r}. Supported: 'npe', 'fmpe', 'cmpe'."
    )


def _build_adapter(bf: ModuleType, fields: tuple[str, ...]) -> Adapter:
    """Route named theta fields to ``inference_variables`` and the observation
    to ``inference_conditions``. The adapter is invertible, so ``sample``
    returns the parameters split back into named fields."""
    return (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(list(fields), into="inference_variables")
        .concatenate([_OBSERVATION_KEY], into="inference_conditions")
    )


def _simulate_offline(
    prior: Distribution,
    simulator: GenerativeLikelihood,
    num_simulations: int,
    key: PRNGKey,
    *,
    sim_backend: SimBackend,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Draw ``(theta, y)`` pairs offline: ``theta ~ prior``, ``y ~ simulator(theta)``.

    Returns ``(named_theta, y)`` where ``named_theta`` maps each prior
    record-template field to a ``(num_simulations, d_field)`` float32 array and
    ``y`` is the ``(num_simulations, d_y)`` float32 array of flattened
    simulated observations.
    """
    template = prior.record_template
    fields = template.fields
    k_theta, k_sim = jax.random.split(key)
    theta = _sample_op(prior, key=k_theta, sample_shape=(num_simulations,))
    # ``theta.flatten()`` is the prior's canonical flat layout (record-template
    # field order); the simulator's ``generate_data(params, ...)`` consumes it
    # positionally, and ``unflatten`` splits it back into per-field arrays for
    # the adapter via the same canonical layout -- uniform across record
    # containers and single-field priors (whose raw draws are not field-indexable
    # by name, but whose flat layout is well defined).
    theta_flat = jnp.asarray(theta.flatten()).reshape(num_simulations, -1)
    record = NumericRecordArray.unflatten(
        theta_flat, template=template, batch_shape=(num_simulations,)
    )
    named = {
        f: np.asarray(jnp.asarray(record[f]).reshape(num_simulations, -1), dtype="float32")
        for f in fields
    }
    sim_keys = jax.random.split(k_sim, num_simulations)

    def _one(p: Array, k: PRNGKey) -> Array:
        # One simulated dataset per theta, flattened to a fixed (d_y,) vector.
        return jnp.ravel(simulator.generate_data(p, 1, key=k)[0])

    if sim_backend == "jax":
        # JAX-traceable simulators: vmap the whole batch (fast path).
        y = jax.vmap(_one)(theta_flat, sim_keys)
    else:  # "sequential"
        # Non-traceable simulators (numpy / external code): one eager call per
        # draw, so generate_data runs concretely rather than under a jax trace.
        y = jnp.stack([_one(theta_flat[i], sim_keys[i]) for i in range(num_simulations)])
    return named, np.asarray(y, dtype="float32")


# ---------------------------------------------------------------------------
# Trained-model wrapper: a SupportsConditioning direct sampler
# ---------------------------------------------------------------------------


class BayesFlowPosterior(Distribution, SupportsConditioning):
    """A trained BayesFlow amortized posterior estimator.

    Implements :class:`~probpipe.core.protocols.SupportsConditioning`, so
    ``condition_on(model, observed)`` draws from ``p(theta | observed)`` in one
    forward pass through the trained network.  Because the estimator is
    amortized, the same instance conditions on any observation with no
    retraining.  The amortized path honours ``num_results`` and (best-effort)
    ``random_seed``; ``num_warmup`` / ``num_chains`` do not apply (a forward
    pass yields a single draw block).
    """

    def __init__(
        self,
        approximator: ContinuousApproximator,
        prior: Distribution,
        *,
        method: AmortizedMethod,
        data_dim: int,
        num_results: int = 2000,
        random_seed: int = 0,
    ):
        self._approximator = approximator
        self._prior = prior
        self._fields = tuple(prior.record_template.fields)
        self._method = method
        self._data_dim = data_dim
        self._num_results = num_results
        self._random_seed = random_seed
        # The ``Distribution`` metaclass requires a non-empty name.
        self._name = f"BayesFlowPosterior({method})"

    def _condition_on(self, observed: Any, /, **kwargs: Any) -> ApproximateDistribution:
        num_results = int(kwargs.get("num_results", self._num_results))
        random_seed = int(kwargs.get("random_seed", self._random_seed))

        # One observation -> a length-1 condition batch, matching the flattened
        # ``(d_y,)`` layout the network was trained on.
        obs_flat = np.ravel(np.asarray(observed, dtype="float32"))
        if obs_flat.size != self._data_dim:
            raise ValueError(
                f"observation has {obs_flat.size} values but the estimator was "
                f"trained on observations of size {self._data_dim} (one simulated "
                "dataset); pass a single observation of the simulator's output "
                "shape, not a stacked multi-observation dataset."
            )
        out = self._approximator.sample(
            num_samples=num_results,
            conditions={_OBSERVATION_KEY: obs_flat[None, :]},
            seed=random_seed,
        )
        # ``out`` maps each field to ``(1, num_results, d_field)``; squeeze the
        # observation axis and concatenate in record-template field order (the
        # layout ``make_posterior`` lifts back to named draws).
        cols = [np.asarray(out[f])[0].reshape(num_results, -1) for f in self._fields]
        flat = jnp.asarray(np.concatenate(cols, axis=-1))
        return make_posterior(
            [flat],
            parents=(self._prior,),
            algorithm=f"bayesflow_{self._method}",
            record_template=self._prior.record_template,
            num_results=num_results,
        )

    def __repr__(self) -> str:
        return (
            f"BayesFlowPosterior(method={self._method!r}, "
            f"num_results={self._num_results})"
        )


# ---------------------------------------------------------------------------
# Workflow function
# ---------------------------------------------------------------------------


@workflow_function
def learn_amortized_posterior(
    prior: Distribution,
    simulator: GenerativeLikelihood,
    *,
    method: AmortizedMethod = "npe",
    num_simulations: int = 10_000,
    epochs: int = 50,
    batch_size: int = 128,
    sim_backend: SimBackend = "jax",
    inference_network: Any | None = None,
    num_results: int = 2000,
    random_seed: int = 0,
    optimizer: Any = "adam",
    **fit_kwargs: Any,
) -> BayesFlowPosterior:
    """Learn an amortized conditional posterior ``p(theta | y)`` with BayesFlow.

    Trains an amortized neural posterior estimator (NPE / FMPE / CMPE) from a
    ``prior`` and a ``simulator`` and returns a :class:`BayesFlowPosterior`
    implementing :class:`~probpipe.core.protocols.SupportsConditioning`.
    ``condition_on(result, observed)`` then produces fast amortized draws in a
    single forward pass -- no MCMC.

    Parameters
    ----------
    prior : Distribution
        Prior over the model parameters.  Must be a ``RecordDistribution`` --
        typically a ``ProductDistribution`` of named distributions, or a single
        named distribution for a one-parameter model.  It is sampled via the
        :func:`~probpipe.sample` op to draw training thetas; it is *not*
        otherwise translated.
    simulator : GenerativeLikelihood
        Must implement ``generate_data(params, num_observations, *, key)``; it
        must be JAX-vmappable unless ``sim_backend="sequential"`` (see below). Training
        uses one simulated dataset per ``theta``, so ``condition_on`` must be
        given a single observation of that same flattened shape (not a stacked
        multi-observation dataset).
    method : {"npe", "fmpe", "cmpe"}
        Amortized estimator: NPE (coupling flow), FMPE (flow matching), or CMPE
        (consistency model). NPE's coupling flow needs at least two parameters,
        so for a one-parameter prior the NPE default falls back to a flow-matching
        network (still reported as ``method="npe"``).
    num_simulations : int
        Number of ``(theta, y)`` pairs simulated offline for training.
    epochs, batch_size : int
        keras training schedule.
    sim_backend : {"jax", "sequential"}
        How the offline simulation is executed. ``"jax"`` (default) vmaps the
        simulator and requires it to be JAX-traceable; ``"sequential"`` runs an
        eager per-draw loop, supporting non-JAX simulators (numpy / external
        code) at the cost of speed. Mirrors ``WorkflowFunction``'s dispatch names.
    inference_network : bayesflow.networks.InferenceNetwork or None
        Overrides the method default (``CouplingFlow`` / ``FlowMatching`` /
        ``ConsistencyModel``).
    num_results : int
        Default number of posterior draws per ``condition_on`` call.
    random_seed : int
        Seed for offline simulation, training, and sampling (sampling
        reproducibility is best-effort via keras seeding).
    optimizer : str or keras.Optimizer
        Passed to ``approximator.compile``.
    **fit_kwargs
        Forwarded to ``approximator.fit`` (e.g. ``callbacks``, ``verbose``).

    Returns
    -------
    BayesFlowPosterior
        A ``SupportsConditioning`` distribution wrapping the trained estimator.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``"npe"`` / ``"fmpe"`` / ``"cmpe"``,
        ``sim_backend`` is not ``"jax"`` / ``"sequential"``, or a prior field
        collides with the reserved ``"observation"`` key.
    TypeError
        If ``simulator`` lacks ``generate_data``, or ``prior`` is not a
        ``RecordDistribution`` (has no ``record_template``).
    ImportError
        If the ``[bayesflow]`` extra is not installed.
    """
    if method not in ("npe", "fmpe", "cmpe"):
        raise ValueError(
            f"Unknown amortized SBI method: {method!r}. Supported: 'npe', 'fmpe', 'cmpe'."
        )
    if sim_backend not in ("jax", "sequential"):
        raise ValueError(
            f"Unknown sim_backend: {sim_backend!r}. Supported: 'jax', 'sequential'."
        )
    if not hasattr(simulator, "generate_data"):
        raise TypeError(
            "simulator must be a GenerativeLikelihood with a generate_data method, "
            f"got {type(simulator).__name__}"
        )
    record_template = getattr(prior, "record_template", None)
    if record_template is None:
        raise TypeError(
            "learn_amortized_posterior requires a RecordDistribution prior with "
            "named parameter fields -- typically a ProductDistribution of named "
            f"distributions -- but got {type(prior).__name__}, which has no "
            "record_template."
        )
    fields = record_template.fields
    if _OBSERVATION_KEY in fields:
        raise ValueError(
            f"prior field name {_OBSERVATION_KEY!r} collides with the reserved "
            "observation key used internally by the BayesFlow adapter; rename the field."
        )
    bf = _ensure_bayesflow()
    key = jax.random.PRNGKey(random_seed)
    named, y = _simulate_offline(prior, simulator, num_simulations, key, sim_backend=sim_backend)
    sims = {**named, _OBSERVATION_KEY: y}

    adapter = _build_adapter(bf, fields)
    num_batches = max(1, -(-num_simulations // batch_size))  # ceil: count the partial batch
    net = inference_network or _make_inference_network(
        bf, method, total_steps=epochs * num_batches, event_size=prior.event_size
    )

    approximator = bf.ContinuousApproximator(inference_network=net, adapter=adapter)
    approximator.compile(optimizer=optimizer)
    dataset = bf.OfflineDataset(data=sims, batch_size=batch_size, adapter=adapter)
    approximator.fit(dataset=dataset, epochs=epochs, **fit_kwargs)

    return BayesFlowPosterior(
        approximator, prior, method=method, data_dim=int(y.shape[-1]),
        num_results=num_results, random_seed=random_seed,
    )
