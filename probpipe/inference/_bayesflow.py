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
import random
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
    # Type-only: the optional bayesflow dependency loads at runtime in _ensure_bayesflow.
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
    bf: ModuleType, method: AmortizedMethod, total_steps: int, *, unconstrained_size: int
) -> InferenceNetwork:
    """The BayesFlow inference network for each amortized method.

    NPE's coupling flow splits the parameter vector in two and so needs at least
    two dimensions. The network operates in *unconstrained* space, so the relevant
    count is the unconstrained width, not the prior's event size -- e.g. a 2-simplex
    field contributes one dimension, not two. Below two unconstrained dimensions the
    NPE default falls back to a flow-matching network, which has no such constraint
    (the posterior is still exposed under ``method="npe"``).
    """
    if method == "npe":
        if unconstrained_size < 2:
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


def _field_bijectors(prior: Distribution, fields: tuple[str, ...]) -> dict[str, Any]:
    """Per-field bijector mapping each parameter's unconstrained R^d to its support.

    BayesFlow's ``ContinuousApproximator`` operates in unconstrained, real space, so a
    constrained prior (positive, an interval, ...) is trained on the *unconstrained*
    parameters (``bijector.inverse``) and the network's draws are mapped back to the
    support (``bijector.forward``). For a real-valued prior every bijector is the
    identity, so the round-trip is a no-op. Discrete priors have no smooth bijector and
    are rejected here with a clear error.
    """
    from ..distributions import bijector_for  # lazy: inference/ -> distributions/

    try:
        supports = prior.supports
    except NotImplementedError:
        if len(fields) > 1:
            raise TypeError(
                f"{type(prior).__name__} has {len(fields)} fields but does not "
                "implement per-field `supports`; cannot infer per-field constraints "
                "from the single `support`. Implement `supports` on the prior."
            ) from None
        # Single-field priors: the whole-distribution support is the field's support.
        supports = {f: prior.support for f in fields}
    bijectors: dict[str, Any] = {}
    for f in fields:
        constraint = supports[f]
        try:
            bijectors[f] = bijector_for(constraint)
        except NotImplementedError as e:
            raise ValueError(
                f"learn_amortized_posterior cannot handle prior field {f!r} with support "
                f"{constraint!r}: {e}. Amortized SBI requires a continuous prior whose "
                "support admits a smooth bijector to R^d (e.g. real, positive, an "
                "interval); discrete priors are not supported."
            ) from e
    return bijectors


def _simulate_offline(
    prior: Distribution,
    simulator: GenerativeLikelihood,
    num_simulations: int,
    key: PRNGKey,
    *,
    sim_backend: SimBackend,
    bijectors: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Draw ``(theta, y)`` pairs offline: ``theta ~ prior``, ``y ~ simulator(theta)``.

    Returns ``(named_theta, y)`` where ``named_theta`` maps each prior
    record-template field to a ``(num_simulations, d_field)`` float32 array of
    *unconstrained* parameters (each field's draws pushed through its inverse
    bijector at the field's native event shape, then flattened) and ``y`` is the
    ``(num_simulations, d_y)`` float32 array of flattened simulated observations.
    The simulator itself always sees the constrained, structured draws.
    """
    template = prior.record_template
    fields = template.fields
    k_theta, k_sim = jax.random.split(key)
    theta = _sample_op(prior, key=k_theta, sample_shape=(num_simulations,))
    # Round-trip through the canonical flat layout: single-field priors' raw draws
    # are not field-indexable by name, so unflatten gives uniform named access.
    theta_flat = jnp.asarray(theta.flatten()).reshape(num_simulations, -1)
    record = NumericRecordArray.unflatten(
        theta_flat, template=template, batch_shape=(num_simulations,)
    )
    # Invert before flattening: matrix-valued bijectors (positive-definite) require
    # the field's native (..., n, n) event shape, not the flat adapter layout.
    named = {}
    for f in fields:
        u = bijectors[f].inverse(jnp.asarray(record[f]))
        named[f] = np.asarray(jnp.reshape(u, (num_simulations, -1)), dtype="float32")
    sim_keys = jax.random.split(k_sim, num_simulations)

    def _one(flat_row: Array, k: PRNGKey) -> Array:
        # Per-draw structured params (named-field access), per the
        # GenerativeLikelihood contract.
        params = NumericRecordArray.unflatten(flat_row, template=template, batch_shape=())
        return jnp.ravel(simulator.generate_data(params, 1, key=k)[0])

    if sim_backend == "jax":
        # JAX-traceable simulators: vmap the whole batch (fast path).
        y = jax.vmap(_one)(theta_flat, sim_keys)
    else:  # "sequential"
        # Non-traceable simulators (numpy / external code): one eager call per
        # draw. Theta is hosted once -- per-draw device indexing would sync
        # every iteration.
        theta_rows = np.asarray(theta_flat)
        y = jnp.stack([_one(theta_rows[i], sim_keys[i]) for i in range(num_simulations)])
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
    retraining.  The network samples in unconstrained space; draws are mapped
    back to each field's support via the per-field forward bijectors recorded
    at training time (identity for real-valued fields).  The amortized path
    honours ``num_results`` (a positive integer) and ``random_seed``;
    ``num_warmup`` / ``num_chains`` do not apply (a forward pass yields a
    single draw block).
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
        bijectors: dict[str, Any] | None = None,
    ):
        self._approximator = approximator
        self._prior = prior
        self._fields = tuple(prior.record_template.fields)
        self._method = method
        self._data_dim = data_dim
        self._num_results = num_results
        self._random_seed = random_seed
        # Forward bijectors (unconstrained -> support); missing entries mean identity.
        self._bijectors = bijectors or {}
        # The ``Distribution`` metaclass requires a non-empty name.
        self._name = f"BayesFlowPosterior({method})"

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Any:
        # Present so sampleability probes get NotImplementedError -- which
        # WorkflowFunction's dispatch fallback catches -- not AttributeError.
        raise NotImplementedError(
            "BayesFlowPosterior is an amortized conditional estimator with no "
            "unconditional sampler; draw from it by conditioning on data: "
            "condition_on(model, observed)."
        )

    def _condition_on(self, observed: Any, /, **kwargs: Any) -> ApproximateDistribution:
        num_results = int(kwargs.get("num_results", self._num_results))
        if num_results < 1:
            raise ValueError(f"num_results must be a positive integer, got {num_results}.")
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
        # ``out`` maps each field to ``(1, num_results, d_field)``. Stays in jnp
        # end-to-end: this is the latency-critical amortized path, so no per-field
        # host round-trips.
        cols = []
        for f in self._fields:
            draws = jnp.asarray(out[f])[0]
            bij = self._bijectors.get(f)
            if bij is not None:
                draws = bij.forward(draws)
            cols.append(jnp.reshape(draws, (num_results, -1)))
        flat = jnp.concatenate(cols, axis=-1)
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
        otherwise translated.  Constrained fields (positive, an interval, a simplex,
        positive-definite matrices, ...) are trained in unconstrained space via the
        per-field bijector from :func:`~probpipe.bijector_for` -- applied at the
        field's native event shape -- and mapped back to the support at sample time;
        real-valued fields use the identity.  Discrete priors are not supported.
    simulator : GenerativeLikelihood
        Must implement ``generate_data(params, num_observations, *, key)``.  As with
        ``SimpleGenerativeModel`` / ``PriorPredictiveCheck``, ``params`` is the prior's
        native per-draw sample -- a record whose fields are accessible by name
        (``params["a"]``), not a flattened vector.  It must be JAX-vmappable unless
        ``sim_backend="sequential"`` (see below). Training uses one simulated dataset
        per ``theta``, so ``condition_on`` must be given a single observation of that
        same flattened shape (not a stacked multi-observation dataset).
    method : {"npe", "fmpe", "cmpe"}
        Amortized estimator: NPE (coupling flow), FMPE (flow matching), or CMPE
        (consistency model). NPE's coupling flow needs at least two *unconstrained*
        parameter dimensions (a one-parameter prior has one; so does a single
        2-simplex), below which the NPE default falls back to a flow-matching
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
        Seed for offline simulation (``jax.random``), keras network init + training
        (via ``keras.utils.set_random_seed``), and sampling. The caller's global
        NumPy / Python RNG state is snapshotted and restored after training, so the
        call does not perturb unrelated random streams.
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
        ``sim_backend`` is not ``"jax"`` / ``"sequential"``, any of
        ``num_simulations`` / ``batch_size`` / ``epochs`` / ``num_results`` is not a
        positive integer, a prior field collides with the reserved ``"observation"``
        key, or a prior field's support admits no smooth bijector to ``R^d`` (e.g. a
        discrete prior).
    TypeError
        If ``simulator`` lacks ``generate_data``, ``prior`` is not a
        ``RecordDistribution`` (has no ``record_template``), or a multi-field
        prior implements no per-field ``supports`` accessor.
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
    for _name, _val in (
        ("num_simulations", num_simulations), ("batch_size", batch_size),
        ("epochs", epochs), ("num_results", num_results),
    ):
        if _val < 1:
            raise ValueError(f"{_name} must be a positive integer, got {_val}.")
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
    # Built up front: also rejects discrete / unsupported-support priors before
    # any simulation runs.
    bijectors = _field_bijectors(prior, fields)
    # The network trains on *unconstrained* widths, which differ from the prior's
    # event sizes for dimension-shifting bijectors (a d-simplex contributes d-1).
    leaf_shapes = record_template.numeric_leaf_shapes
    unconstrained_size = sum(
        int(np.prod(tuple(bijectors[f].inverse_event_shape(leaf_shapes[f])), dtype=int))
        for f in fields
    )

    bf = _ensure_bayesflow()
    import keras

    # keras reads the global RNG for network init + fit, so seed it -- but snapshot
    # and restore the caller's Python/NumPy state so the process-wide seeding does
    # not leak. keras keeps its own seeded generator state across the restore.
    py_state = random.getstate()
    np_state = np.random.get_state()
    keras.utils.set_random_seed(random_seed)
    try:
        key = jax.random.PRNGKey(random_seed)
        named, y = _simulate_offline(
            prior, simulator, num_simulations, key,
            sim_backend=sim_backend, bijectors=bijectors,
        )
        sims = {**named, _OBSERVATION_KEY: y}

        adapter = _build_adapter(bf, fields)
        num_batches = max(1, -(-num_simulations // batch_size))  # ceil: count partial batch
        net = inference_network or _make_inference_network(
            bf, method, total_steps=epochs * num_batches, unconstrained_size=unconstrained_size
        )

        approximator = bf.ContinuousApproximator(inference_network=net, adapter=adapter)
        approximator.compile(optimizer=optimizer)
        dataset = bf.OfflineDataset(data=sims, batch_size=batch_size, adapter=adapter)
        approximator.fit(dataset=dataset, epochs=epochs, **fit_kwargs)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

    return BayesFlowPosterior(
        approximator, prior, method=method, data_dim=int(y.shape[-1]),
        num_results=num_results, random_seed=random_seed, bijectors=bijectors,
    )
