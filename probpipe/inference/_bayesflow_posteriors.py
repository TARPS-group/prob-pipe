"""BayesFlow amortized-SBI backend for ProbPipe.

Trains amortized conditional posterior estimators -- NPE (neural posterior
estimation), FMPE (flow-matching) and CMPE (consistency-model) -- with BayesFlow
(keras-on-JAX) and returns the learned kernel ``q(theta | y)`` from the
observation to the parameters: ``condition_on(q, {"observation": y})`` draws from
the amortized posterior in a single forward pass through the trained network --
no MCMC, no gradient bridge, and no prior translation (the prior is used only
to draw ``theta`` at train time via the :func:`~probpipe.sample` op).

The shared bridge (lazy import, validation, offline simulation, adapter keying,
seeded training) lives in :mod:`._bayesflow_common`;
:mod:`._bayesflow_likelihoods` builds the NLE/NRE likelihood surrogates on the
same pipeline.

BayesFlow / keras is imported lazily on first use, so ``import probpipe`` does
not pull keras.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

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
    SupportsConditionalSampling,
)
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution
from ..functions import function
from ..values import Function
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._bayesflow_common import (
    _OBSERVATION_KEY,
    SimBackend,
    _adapter_field_keys,
    _import_bayesflow,
    _isolated_keras_seeding,
    _observation_slot,
    _simulate_offline,
    _validate_learn_inputs,
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


class _AmortizedPosterior(
    ConditionalDistribution, SupportsApproximateConditioning, SupportsConditionalSampling
):
    """A learned amortized posterior ``q(theta | y)``: a kernel from the observation to the parameters.

    Its given slot is the observation, named ``observation`` unless the prior
    declares that name, and its event is the parameters, declared as the prior
    declares them. Evaluating it at an observation runs the trained
    network once, with no retraining and no inference, and the law it yields
    samples. The evaluation stands in for the posterior of the joint of the prior
    and the simulator the network was trained on, so the kernel claims
    ``SupportsApproximateConditioning`` and ``exact_only=True`` excludes it. The
    network samples in unconstrained space, and the draws are mapped back to each
    leaf's support by the forward bijectors recorded at training.

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
    num_results : int
        The default number of draws of the law at an observation.
    bijectors : dict of str to Function, optional
        The forward bijector of each constrained leaf.
    """

    def __init__(
        self,
        approximator: ContinuousApproximator,
        prior: Distribution,
        simulator: ConditionalDistribution,
        *,
        method: AmortizedMethod,
        data_dim: int,
        num_results: int = 2000,
        bijectors: dict[str, Function] | None = None,
    ):
        slot = _observation_slot(prior)
        super().__init__(
            f"amortized_posterior_{method}", {slot: NumericArraySpec((data_dim,))}, prior.event_spec
        )
        attributes = {
            "_slot": slot,
            "_approximator": approximator,
            "_prior": prior,
            "_simulator": simulator,
            # Numeric leaves (slash paths for a nested prior; == fields for a flat
            # one) -- the column order the network emits, matching training.
            "_leaf_keys": tuple(_components_record(prior.event_spec).leaf_shapes),
            "_method": method,
            "_data_dim": data_dim,
            "_num_results": num_results,
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
                raise KeyError(f"{others} are not the given slot {self._slot!r} of {self.name!r}")
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

    def _network_draws(self, observation: np.ndarray, num_results: int, seed: int) -> Array:
        """``num_results`` flat draws of the network at *observation*, in the prior's supports."""
        out = self._approximator.sample(
            num_samples=num_results,
            conditions={_OBSERVATION_KEY: observation[None, :]},
            seed=seed,
        )
        # ``out`` maps each internal theta key to ``(1, num_results, d_leaf)``.
        # Stays in jnp end-to-end: this is the latency-critical amortized path,
        # so no per-leaf host round-trips. Columns are concatenated in leaf order,
        # which is the canonical flatten order the posterior's record unflattens by.
        cols = []
        for k, leaf in zip(_adapter_field_keys(self._leaf_keys), self._leaf_keys):
            draws = jnp.asarray(out[k])[0]
            bij = self._bijectors.get(leaf)
            if bij is not None:
                draws = bij.apply(draws)
            cols.append(jnp.reshape(draws, (num_results, -1)))
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

    def _condition_on(self, given: Any, /, **kwargs: Any) -> ApproximateDistribution:
        """The network's draws at the observation *given* binds, as an empirical posterior.

        *given* is a mapping or record keyed by the observation slot, or the
        observation itself. ``num_results`` sets the number of draws, and
        ``random_seed`` their seed; without it the draws are a workflow-owned
        random event, so ``workflow_run(seed=...)`` fixes them.

        Raises
        ------
        TypeError
            If a keyword is neither ``num_results`` nor ``random_seed``.
        KeyError
            If *given* names a key other than the observation slot.
        ValueError
            If ``num_results`` is not positive, or the observation's size is not
            the trained one.
        """
        unread = sorted(set(kwargs) - {"num_results", "random_seed"})
        if unread:
            raise TypeError(
                f"method_options {unread} are not options of the amortized posterior "
                f"{self.name!r}, which reads ['num_results', 'random_seed']"
            )
        num_results = int(kwargs.get("num_results", self._num_results))
        if num_results < 1:
            raise ValueError(f"num_results must be a positive integer, got {num_results}.")
        seed = integer_seed(run_seed(kwargs, f"bayesflow_{self._method}"))
        flat = self._network_draws(self._observation(given), num_results, seed)
        return make_posterior(
            [flat],
            parents=(self,),
            method=f"bayesflow_{self._method}",
            event_spec=self._prior.event_spec,
            num_results=num_results,
        )

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
        return _reconstruct_from_vector(self.name, spec, vector)

    def __repr__(self) -> str:
        return f"AmortizedPosterior(method={self._method!r}, num_results={self._num_results})"


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
    num_results: int = 2000,
    random_seed: int = 0,
    optimizer: str | KerasOptimizer = "adam",
    **fit_kwargs: Any,
) -> ConditionalDistribution:
    """Learn an amortized conditional posterior ``q(theta | y)`` with BayesFlow.

    Trains an amortized neural posterior estimator (NPE / FMPE / CMPE) from a
    ``prior`` and a ``simulator`` and returns the learned kernel from the
    observation to the parameters, whose given slot is ``observation``.
    ``condition_on(result, {"observation": y})`` evaluates it in a single forward
    pass, with no MCMC, and returns a law that samples; the evaluation is
    approximate, so ``exact_only=True`` refuses it. Provenance names the prior
    and the simulator it was trained on.

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
        network (still reported as ``method="npe"``).
    num_simulations : int
        Number of ``(theta, y)`` pairs simulated offline for training.
    epochs, batch_size : int
        keras training schedule.
    sim_backend : {"jax", "sequential"}
        How the offline simulation is executed. ``"jax"`` (default) vmaps the
        simulator and requires it to be JAX-traceable; ``"sequential"`` runs an
        eager per-draw loop, supporting non-JAX simulators (numpy / external
        code) at the cost of speed. Mirrors ``Function``'s dispatch names.
    inference_network : bayesflow.networks.InferenceNetwork or None
        Overrides the method default (``CouplingFlow`` / ``FlowMatching`` /
        ``ConsistencyModel``).
    num_results : int
        Default number of posterior draws per ``condition_on`` call.
    random_seed : int
        Seed for offline simulation (``jax.random``) and keras network init +
        training (via ``keras.utils.set_random_seed``). The caller's global
        NumPy / Python RNG state is snapshotted and restored after training, so the
        call does not perturb unrelated random streams. The learned posterior's
        draws are seeded by the workflow scope, or by a ``random_seed`` method
        option of the conditioning call.
    optimizer : str or keras.Optimizer
        Passed to ``approximator.compile``.
    **fit_kwargs
        Forwarded to ``approximator.fit`` (e.g. ``callbacks``, ``verbose``).

    Returns
    -------
    ConditionalDistribution
        The amortized posterior ``q(theta | y)``, which claims
        ``SupportsApproximateConditioning`` and ``SupportsConditionalSampling``;
        its ``prior`` and ``simulator`` are the joint it was trained on.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``"npe"`` / ``"fmpe"`` / ``"cmpe"``,
        ``sim_backend`` is not ``"jax"`` / ``"sequential"``, any of
        ``num_simulations`` / ``batch_size`` / ``epochs`` / ``num_results`` is
        less than one, or a prior field's support is not declared or admits no
        smooth bijector to ``R^d`` (e.g. a discrete prior).
    TypeError
        If a count parameter is not an integer, ``simulator`` is not a kernel
        that samples, or ``prior`` is not a numeric distribution.
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
            ("num_results", num_results),
        ),
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
        num_results=num_results,
        bijectors=bijectors,
    )
