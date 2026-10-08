"""BayesFlow amortized-SBI backend for ProbPipe.

Trains amortized conditional posterior estimators -- NPE (neural posterior
estimation), FMPE (flow-matching) and CMPE (consistency-model) -- with BayesFlow
(keras-on-JAX) and returns the learned kernel ``q(theta | y)`` from the
observation to the parameters: ``condition_on(q, {"observation": y})`` returns
the amortized posterior's law at ``y``, each of whose draws is one forward pass
through the trained network -- no MCMC, no gradient bridge, and no prior
translation (the prior is used only to draw ``theta`` at train time via the
:func:`~probpipe.sample` op). When the network is a coupling flow, NPE's
default, the kernel and its laws also have the flow's density.

The shared bridge (lazy import, validation, offline simulation, adapter keying,
seeded training) lives in :mod:`._bayesflow_common`;
:mod:`._bayesflow_likelihoods` builds the NLE/NRE likelihood surrogates on the
same pipeline.

BayesFlow / keras is imported lazily on first use, so ``import probpipe`` does
not pull keras.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..core._dispatch import Feasibility, MathematicalDomainError, ResolutionError
from ..core._numeric_record import _reconstruct_from_vector
from ..core._spec_base import NumericArraySpec
from ..core._specs import _components_record
from ..core.record import Record
from ..custom_types import Array
from ..distributions._capabilities import (
    SupportsApproximateConditioning,
    SupportsConditionalLogProb,
    SupportsConditionalSampling,
    SupportsLogProb,
    SupportsSampling,
    _capability_subclass,
)
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution
from ..distributions._factored import _raw_record
from ..functions import function
from ..values import Function
from ._approximate_distribution import _record_run
from ._bayesflow_common import (
    _OBSERVATION_KEY,
    SimBackend,
    _adapter_field_keys,
    _import_bayesflow,
    _isolated_keras_seeding,
    _leaf_draws,
    _observation_slot,
    _simulate_offline,
    _validate_learn_inputs,
    _without_progress_bar,
)
from ._inference_utils import integer_seed, run_seed

if TYPE_CHECKING:
    # Type-only: bayesflow/keras load at runtime in _import_bayesflow.
    from bayesflow import Adapter, ContinuousApproximator
    from bayesflow.networks import InferenceNetwork
    from keras.optimizers import Optimizer as KerasOptimizer
else:
    # ``@function`` resolves the decorated signature's hints at runtime
    # (``get_type_hints``), so names appearing there must exist outside
    # TYPE_CHECKING; the optional-dependency types degrade to ``Any``.
    InferenceNetwork = KerasOptimizer = Any

AmortizedMethod = Literal["npe", "fmpe", "cmpe"]
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
    raise ValueError(f"Unknown amortized SBI method: {method!r}. Supported: 'npe', 'fmpe', 'cmpe'.")


def _build_adapter(bf: ModuleType, internal_keys: tuple[str, ...]) -> Adapter:
    """Route the internal theta keys to ``inference_variables`` and the
    observation to ``inference_conditions``. The adapter is invertible, so
    ``sample`` returns the parameters split back under the same keys."""
    return (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(list(internal_keys), into="inference_variables")
        .concatenate([_OBSERVATION_KEY], into="inference_conditions")
    )


def _field_bijectors(prior: Distribution, keys: tuple[str, ...]) -> dict[str, Function]:
    """Per-leaf bijector mapping each parameter's unconstrained R^d to its support.

    ``keys`` are the prior's numeric leaves (slash paths like ``"outer/a"`` for a
    nested prior; top-level field names for a flat one), matching the keying of
    ``prior.supports``.

    BayesFlow's ``ContinuousApproximator`` operates in unconstrained, real space, so a
    constrained prior (positive, an interval, ...) is trained on the *unconstrained*
    parameters (the bijector's inverse) and the network's draws are mapped back to the
    support (its forward map). For a real-valued prior every bijector is the
    identity, so the round-trip is a no-op. Discrete priors have no smooth bijector and
    are rejected here with a clear error.
    """
    from ..functions import bijector_for

    supports = prior.supports
    bijectors: dict[str, Function] = {}
    for k in keys:
        constraint = supports[k]
        if constraint is None:
            raise ValueError(
                f"learn_amortized_posterior cannot handle prior parameter {k!r}: its "
                "support is not declared, so no bijector to R^d can be chosen. Give the "
                "prior a declared support, for example by building it from a family."
            )
        try:
            bijectors[k] = bijector_for(constraint)
        except (MathematicalDomainError, ResolutionError) as e:
            raise ValueError(
                f"learn_amortized_posterior cannot handle prior parameter {k!r} with "
                f"support {constraint!r}: {e}. Amortized SBI requires a continuous prior "
                "whose support admits a smooth bijector to R^d (e.g. real, positive, an "
                "interval); discrete priors are not supported."
            ) from e
    return bijectors


def _unconstrained_shape(bijector: Function, shape: tuple[int, ...]) -> tuple[int, ...]:
    """The shape of the unconstrained point the inverse of *bijector* gives at a point of *shape*.

    A dimension-shifting bijector changes it: a point of a ``d``-simplex has
    ``d - 1`` unconstrained coordinates.
    """
    point = jax.ShapeDtypeStruct(tuple(shape), jnp.float32)
    return tuple(jax.eval_shape(bijector._inverse, point).shape)


# ---------------------------------------------------------------------------
# The amortized posterior: a learned kernel from the observation to the parameters
# ---------------------------------------------------------------------------


def _flow_log_density(kernel: _AmortizedPosterior, observation: np.ndarray, value: Any) -> Array:
    """The coupling flow's log-density of the parameters *value* at *observation*.

    Each leaf is mapped by the inverse of its bijector to the unconstrained
    coordinates the network was trained in. The flow scores the point as
    ``approximator.log_prob`` does, adding the standardization's
    log-Jacobian, and each bijector's log-Jacobian at the point is subtracted.

    Parameters
    ----------
    kernel : _AmortizedPosterior
        The amortized posterior, whose trained network and bijectors score the
        point.
    observation : numpy.ndarray
        The flattened observation the network conditions on.
    value : Any
        The parameters, with any leading axes: a record or nested mapping of
        the prior's leaves, or one array.

    Returns
    -------
    Array
        One log-density per value, shaped like the value's leading axes.
    """
    raw = _raw_record(value)
    leaves = {leaf: jnp.asarray(_leaf_draws(raw, leaf)) for leaf in kernel._leaf_keys}
    first = kernel._leaf_keys[0]
    batch = leaves[first].shape[: leaves[first].ndim - len(kernel._leaf_shapes[first])]
    count = prod(batch)
    points, log_jacobian = [], jnp.zeros(count)
    for leaf, x in leaves.items():
        x = jnp.reshape(x, (count, *kernel._leaf_shapes[leaf]))
        bijector = kernel._bijectors.get(leaf)
        if bijector is not None:
            x = jax.vmap(bijector._inverse)(x)
            log_jacobian = log_jacobian + jax.vmap(bijector._log_det_jacobian)(x)
        points.append(jnp.reshape(x, (count, -1)))
    standardizer = kernel._approximator.standardizer
    conditions = standardizer.maybe_standardize(
        jnp.tile(jnp.asarray(observation)[None, :], (count, 1)),
        key="inference_conditions",
        stage="inference",
    )
    z, standardization = standardizer.maybe_standardize(
        jnp.concatenate(points, axis=-1).astype(jnp.float32),
        key="inference_variables",
        stage="inference",
        log_det_jac=True,
    )
    network = kernel._approximator.inference_network
    log_density = network.log_prob(z, conditions=conditions) + standardization
    return jnp.reshape(log_density - log_jacobian, batch)


def _amortized_conditional_log_prob(
    self: _AmortizedPosterior, given: Record | Mapping[str, Any], value: Any
) -> Array:
    """The flow's log-density of the parameters *value* at the observation *given* binds."""
    return _flow_log_density(self, self._observation(given), value)


def _amortized_log_prob(self: _AmortizedPosteriorLaw, value: Any) -> Array:
    """The flow's log-density of the parameters *value* at the law's observation."""
    return _flow_log_density(self._kernel, self._observation_value, value)


class _AmortizedPosterior(
    ConditionalDistribution, SupportsApproximateConditioning, SupportsConditionalSampling
):
    """A learned amortized posterior ``q(theta | y)``: a kernel from the observation to the parameters.

    Its given slot is the observation, named ``observation`` unless the prior
    declares that name, and its event is the parameters, declared as the prior
    declares them. Evaluating it at an observation yields the law whose draws
    each run the trained network once, with no retraining and no inference.
    The evaluation stands in for the posterior of the joint of the prior and
    the simulator the network was trained on, so the kernel claims
    ``SupportsApproximateConditioning`` and ``exact_only=True`` excludes it. The
    network samples in unconstrained space, and the draws are mapped back to each
    leaf's support by the forward bijectors recorded at training.

    When the network is a coupling flow, the kernel claims
    ``SupportsConditionalLogProb`` and its law at an observation claims
    ``SupportsLogProb``. The log-density at a value of the parameters is the
    flow's log-density at the value's unconstrained coordinates less each
    bijector's log-Jacobian there, which is exact for the learned law. A
    flow-matching or consistency network gives no density.

    Parameters
    ----------
    approximator : ContinuousApproximator
        The trained BayesFlow approximator.
    prior : Distribution
        The prior the network was trained against.
    simulator : ConditionalDistribution
        The simulator the network was trained against.
    method : {"npe", "fmpe", "cmpe"}
        The amortized estimator.
    data_dim : int
        The flattened size of the observation the network conditions on.
    bijectors : dict of str to Function, optional
        The forward bijector of each constrained leaf.
    has_density : bool, default False
        Whether the network computes the learned law's density, as a coupling
        flow does.
    """

    _capability_table: ClassVar = {
        SupportsConditionalLogProb: {"_conditional_log_prob": _amortized_conditional_log_prob}
    }

    def __new__(
        cls,
        approximator: ContinuousApproximator,
        prior: Distribution,
        simulator: ConditionalDistribution,
        *,
        method: AmortizedMethod,
        data_dim: int,
        bijectors: dict[str, Function] | None = None,
        has_density: bool = False,
    ) -> _AmortizedPosterior:
        claimed = [SupportsConditionalLogProb] if has_density else []
        return object.__new__(_capability_subclass(_AmortizedPosterior, claimed))

    def __init__(
        self,
        approximator: ContinuousApproximator,
        prior: Distribution,
        simulator: ConditionalDistribution,
        *,
        method: AmortizedMethod,
        data_dim: int,
        bijectors: dict[str, Function] | None = None,
        has_density: bool = False,
    ):
        slot = _observation_slot(prior)
        super().__init__(
            f"amortized_posterior_{method}", {slot: NumericArraySpec((data_dim,))}, prior.event_spec
        )
        leaf_shapes = dict(_components_record(prior.event_spec).leaf_shapes)
        attributes = {
            "_slot": slot,
            "_approximator": approximator,
            "_prior": prior,
            "_simulator": simulator,
            # Numeric leaves (slash paths for a nested prior; == fields for a flat
            # one) -- the column order the network emits, matching training.
            "_leaf_keys": tuple(leaf_shapes),
            "_leaf_shapes": leaf_shapes,
            "_method": method,
            "_data_dim": data_dim,
            # Forward bijectors (unconstrained -> support); missing entries mean identity.
            "_bijectors": bijectors or {},
        }
        for name, value in attributes.items():
            object.__setattr__(self, name, value)

    @property
    def prior(self) -> Distribution:
        """The prior the network was trained against."""
        return self._prior

    @property
    def simulator(self) -> ConditionalDistribution:
        """The simulator the network was trained against."""
        return self._simulator

    def _observation(self, given: Any) -> np.ndarray:
        """The observation *given* binds, flattened to the size the network was trained on.

        Parameters
        ----------
        given : Any
            The observation, or a mapping or ``Record`` that holds it at the
            observation slot.

        Returns
        -------
        numpy.ndarray
            A one-dimensional ``float32`` array.

        Raises
        ------
        KeyError
            If a mapping or record *given* lacks the observation slot or names
            another key.
        ValueError
            If the observation's size is not the trained one.
        """
        if isinstance(given, Record):
            given = given.children
        if isinstance(given, Mapping):
            others = sorted(set(given) - {self._slot})
            if others:
                raise KeyError(f"{others} are not the given slot {self._slot!r} of {self.label!r}")
            given = given[self._slot]
        obs_flat = np.ravel(np.asarray(given, dtype="float32"))
        if obs_flat.size != self._data_dim:
            raise ValueError(
                f"observed data has {obs_flat.size} values but the estimator was "
                f"trained to condition on size {self._data_dim} (the simulator's "
                "per-draw output) -- the conditioning shape is fixed at training "
                "time. Pass observed data of that shape; for datasets of other "
                "sizes use learn_amortized_likelihood or learn_amortized_ratio."
            )
        return obs_flat

    def _network_draws(self, observation: np.ndarray, count: int, seed: int) -> Array:
        """``count`` flat draws of the network at *observation*, in the prior's supports."""
        with _without_progress_bar():
            out = self._approximator.sample(
                num_samples=count,
                conditions={_OBSERVATION_KEY: observation[None, :]},
                seed=seed,
            )
        # ``out`` maps each internal theta key to ``(1, count, d_leaf)``.
        # Stays in jnp end-to-end: this is the latency-critical amortized path,
        # so no per-leaf host round-trips. Columns are concatenated in leaf order,
        # which is the canonical flatten order the posterior's record unflattens by.
        cols = []
        for k, leaf in zip(_adapter_field_keys(self._leaf_keys), self._leaf_keys):
            draws = jnp.asarray(out[k])[0]
            bij = self._bijectors.get(leaf)
            if bij is not None:
                draws = bij.apply(draws)
            cols.append(jnp.reshape(draws, (count, -1)))
        return jnp.concatenate(cols, axis=-1)

    def _condition_on_guard(self, paths: tuple[str, ...]) -> Feasibility:
        """Every path names the observation slot, since a parameter is conditioned by Bayes' rule."""
        others = [path for path in paths if path != self._slot]
        if others:
            return Feasibility(
                False,
                f"{others} are not the given slot {self._slot!r} of the amortized posterior",
            )
        return Feasibility(True)

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Distribution:
        """The learned law at the observation *given* binds, which samples the network.

        *given* is a mapping or record keyed by the observation slot, or the
        observation itself. Evaluating the kernel runs no network and takes no
        method options. Each draw of the law runs the network at the
        observation, seeded by the draw's key, so ``workflow_run(seed=...)``
        fixes the draws.

        Parameters
        ----------
        given : Any
            The value the call conditions on.
        **kwargs : Any
            The call's ``method_options``.

        Returns
        -------
        Distribution
            The law, whose provenance names the kernel and the method
            ``bayesflow_<method>``.

        Raises
        ------
        TypeError
            If a method option is given.
        KeyError
            If *given* names a key other than the observation slot.
        ValueError
            If the observation's size is not the trained one.
        """
        if kwargs:
            raise TypeError(
                f"method_options {sorted(kwargs)} are not options of the amortized posterior "
                f"{self.label!r}, which takes none"
            )
        law = _AmortizedPosteriorLaw(self, self._observation(given))
        return _record_run(law, (self,), f"bayesflow_{self._method}")

    def _conditional_sample(
        self, given: Any, key: Array, sample_shape: tuple[int, ...] = ()
    ) -> Any:
        """Draws of the network at the observation *given* binds, seeded by *key*.

        One draw for ``sample_shape=()``, and the sample axes before the event
        otherwise, at the kind the event declares.
        """
        count = int(np.prod(sample_shape)) if sample_shape else 1
        seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
        flat = self._network_draws(self._observation(given), count, seed)
        spec = self.event_spec.spec
        if not self.event_spec.exposes_record:
            return flat.reshape(*sample_shape, *spec.shape)
        vector = flat.reshape(*sample_shape, flat.shape[-1]) if sample_shape else flat[0]
        return _reconstruct_from_vector(self.label, spec, vector)

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The network the posterior was learned with."""
        return [("method", repr(self._method))]


class _AmortizedPosteriorLaw(Distribution, SupportsSampling):
    """An amortized posterior's law at an observation, whose draws are the network's.

    Its event is the parameters, declared as the prior declares them, and each
    draw runs the trained network at the observation. The law claims
    ``SupportsLogProb`` when its kernel claims the conditional density.
    """

    _capability_table: ClassVar = {SupportsLogProb: {"_log_prob": _amortized_log_prob}}

    def __new__(
        cls, kernel: _AmortizedPosterior, observation: np.ndarray
    ) -> _AmortizedPosteriorLaw:
        claimed = [SupportsLogProb] if isinstance(kernel, SupportsConditionalLogProb) else []
        return object.__new__(_capability_subclass(_AmortizedPosteriorLaw, claimed))

    def __init__(self, kernel: _AmortizedPosterior, observation: np.ndarray) -> None:
        super().__init__("posterior", kernel.event_spec)
        self._kernel = kernel
        self._observation_value = observation

    def _sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Any:
        """Draws of the network at the observation, seeded by *key*."""
        return self._kernel._conditional_sample(self._observation_value, key, sample_shape)


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------


@function
def learn_amortized_posterior(
    prior: Distribution,
    simulator: ConditionalDistribution,
    *,
    method: AmortizedMethod = "npe",
    num_simulations: int = 10_000,
    epochs: int = 50,
    batch_size: int = 128,
    sim_backend: SimBackend = "jax",
    inference_network: InferenceNetwork | None = None,
    optimizer: str | KerasOptimizer = "adam",
    **fit_kwargs: Any,
) -> ConditionalDistribution:
    """Learn an amortized conditional posterior ``q(theta | y)`` with BayesFlow.

    Trains an amortized neural posterior estimator (NPE / FMPE / CMPE) from a
    ``prior`` and a ``simulator`` and returns the learned kernel from the
    observation to the parameters, whose given slot is ``observation``.
    ``condition_on(result, {"observation": y})`` evaluates it without retraining
    or MCMC and returns the law at ``y``, each of whose draws is one forward pass
    of the network; the evaluation is approximate, so ``exact_only=True`` refuses
    it. Provenance names the prior
    and the simulator it was trained on.

    The training's seed is drawn from a workflow-owned random event, so
    ``workflow_run(seed=...)`` reproduces the trained network, and an unscoped
    call trains afresh. The seed fixes the offline simulation (``jax.random``)
    and keras's network initialization and training, which
    ``keras.utils.set_random_seed`` seeds. The caller's global NumPy and Python
    random states are restored after training, and keras's global seed
    generator keeps the state training leaves. Each draw of the learned
    posterior's law at an observation is a workflow-owned random event of its
    own.

    Parameters
    ----------
    prior : Distribution
        Prior over the model parameters.  Must be a numeric distribution --
        typically a factored joint of named distributions, or a single named
        distribution for a one-parameter model.  It is
        sampled via the :func:`~probpipe.sample` op to draw training thetas; it is
        *not* otherwise translated.  Constrained leaves (positive, an interval, a
        simplex, positive-definite matrices, ...) are trained in unconstrained space
        via the per-leaf bijector from :func:`~probpipe.bijector_for` -- applied at
        the leaf's native event shape -- and mapped back to the support at sample
        time; real-valued leaves use the identity.  Discrete priors are not supported.
    simulator : ConditionalDistribution
        The kernel of one observation given the prior's fields, which claims
        ``SupportsConditionalSampling``. Its given values are the prior's native
        per-draw sample -- a record whose fields are accessible by name
        (``given["a"]``), not a flattened vector.  It must be JAX-vmappable unless
        ``sim_backend="sequential"`` (see below). Training uses one simulated
        observation per ``theta``, fixing the conditioning shape: ``condition_on``
        must be given observed data of that same flattened size (datasets of any
        size are the NLE/NRE learners' case).
    method : {"npe", "fmpe", "cmpe"}
        Amortized estimator: NPE (coupling flow), FMPE (flow matching), or CMPE
        (consistency model). NPE's coupling flow needs at least two *unconstrained*
        parameter dimensions (a one-parameter prior has one; so does a single
        2-simplex), below which the NPE default falls back to a flow-matching
        network (still reported as ``method="npe"``), which gives no density.
    num_simulations : int
        Number of ``(theta, y)`` pairs simulated offline for training.
    epochs : int
        Number of keras training passes over the simulations.
    batch_size : int
        Number of simulations in each keras training batch.
    sim_backend : {"jax", "sequential"}
        How the offline simulation is executed. ``"jax"`` (default) vmaps the
        simulator and requires it to be JAX-traceable; ``"sequential"`` runs an
        eager per-draw loop, supporting non-JAX simulators (numpy / external
        code) at the cost of speed. Mirrors ``Function``'s dispatch names.
    inference_network : bayesflow.networks.InferenceNetwork or None
        Overrides the method default (``CouplingFlow`` / ``FlowMatching`` /
        ``ConsistencyModel``). A ``CouplingFlow`` gives the posterior a density.
    optimizer : str or keras.Optimizer
        Passed to ``approximator.compile``.
    **fit_kwargs
        Forwarded to ``approximator.fit`` (e.g. ``callbacks``, ``verbose``).

    Returns
    -------
    ConditionalDistribution
        The amortized posterior ``q(theta | y)``, which claims
        ``SupportsApproximateConditioning`` and ``SupportsConditionalSampling``,
        and ``SupportsConditionalLogProb`` when its network is a coupling flow.
        Its ``prior`` and ``simulator`` are the joint it was trained on.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``"npe"`` / ``"fmpe"`` / ``"cmpe"``,
        ``sim_backend`` is not ``"jax"`` / ``"sequential"``, any of
        ``num_simulations`` / ``batch_size`` / ``epochs`` is less than one, or a prior field's support is not declared or admits no
        smooth bijector to ``R^d`` (e.g. a discrete prior).
    TypeError
        If a count parameter is not an integer, ``simulator`` is not a kernel
        that samples, ``prior`` is not a numeric distribution, or
        ``fit_kwargs`` holds ``random_seed`` or ``seed``.
    ImportError
        If the ``[bayesflow]`` extra is not installed.
    """
    if method not in ("npe", "fmpe", "cmpe"):
        raise ValueError(
            f"Unknown amortized SBI method: {method!r}. Supported: 'npe', 'fmpe', 'cmpe'."
        )
    record = _validate_learn_inputs(
        prior,
        simulator,
        caller="learn_amortized_posterior",
        sim_backend=sim_backend,
        counts=(
            ("num_simulations", num_simulations),
            ("batch_size", batch_size),
            ("epochs", epochs),
        ),
        fit_kwargs=fit_kwargs,
    )
    # Per numeric leaf (slash paths for a nested prior; == fields for a flat
    # one). supports / bijectors are leaf-keyed, so this serves both uniformly.
    leaf_shapes = record.leaf_shapes
    leaf_keys = tuple(leaf_shapes)
    # Built up front: also rejects discrete / unsupported-support priors before
    # any simulation runs.
    bijectors = _field_bijectors(prior, leaf_keys)
    # The network trains on *unconstrained* widths, which differ from the prior's
    # event sizes for dimension-shifting bijectors (a d-simplex contributes d-1).
    unconstrained_size = sum(
        int(np.prod(_unconstrained_shape(bijectors[k], leaf_shapes[k]), dtype=int))
        for k in leaf_keys
    )

    bf = _import_bayesflow()
    random_seed = integer_seed(run_seed("learn_amortized_posterior"))
    with _isolated_keras_seeding(random_seed):
        key = jax.random.PRNGKey(random_seed)
        named, y = _simulate_offline(
            prior,
            simulator,
            num_simulations,
            key,
            sim_backend=sim_backend,
            bijectors=bijectors,
        )
        # Re-key theta leaves to positional internal names so user field names
        # never enter BayesFlow's key namespace (no collision with the
        # observation key or the adapter's inference_variables/conditions).
        internal_keys = _adapter_field_keys(leaf_keys)
        sims = {k: named[lk] for k, lk in zip(internal_keys, leaf_keys)}
        sims[_OBSERVATION_KEY] = y

        adapter = _build_adapter(bf, internal_keys)
        num_batches = max(1, -(-num_simulations // batch_size))  # ceil: count partial batch
        net = inference_network or _make_inference_network(
            bf, method, total_steps=epochs * num_batches, unconstrained_size=unconstrained_size
        )

        approximator = bf.ContinuousApproximator(inference_network=net, adapter=adapter)
        approximator.compile(optimizer=optimizer)
        dataset = bf.OfflineDataset(data=sims, batch_size=batch_size, adapter=adapter)
        approximator.fit(dataset=dataset, epochs=epochs, **fit_kwargs)

    return _AmortizedPosterior(
        approximator,
        prior,
        simulator,
        method=method,
        data_dim=int(y.shape[-1]),
        bijectors=bijectors,
        has_density=isinstance(net, bf.networks.CouplingFlow),
    )
