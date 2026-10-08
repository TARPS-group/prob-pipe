"""Shared bridge for the BayesFlow backends (optional ``[bayesflow]`` extra).

Hosts the pieces every BayesFlow-based learner uses: the lazy keras-pinned
import, train-time input validation, the seeded-training RNG bracket, the
offline ``(theta, y)`` simulation pipeline, and the internal adapter keying
that keeps user field names out of BayesFlow's key namespace. Consumed by
:mod:`._bayesflow_posteriors` (NPE/FMPE/CMPE) and
:mod:`._bayesflow_likelihoods` (NLE/NRE).

BayesFlow / keras is imported lazily on first use, so ``import probpipe`` does
not pull keras.
"""

from __future__ import annotations

import functools
import os
import random
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .._messages import unknown_names
from ..core._specs import _components_record
from ..core.record import Record
from ..custom_types import Array, PRNGKey
from ..distributions._capabilities import SupportsConditionalSampling
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution, NumericDistribution
from ._inference_utils import refuse_seed_keywords

if TYPE_CHECKING:
    from ..values import Function

# Offline-simulation execution backend; values mirror ``Function``'s
# dispatch names ("jax" = vmap the simulator, "sequential" = eager per-draw loop).
SimBackend = Literal["jax", "sequential"]

# Internal adapter key for the simulated observation. User field names never
# enter BayesFlow's key namespace -- theta fields are re-keyed via
# ``_adapter_field_keys`` -- so this (and BayesFlow's own ``inference_variables``
# / ``inference_conditions`` targets) can never collide with a prior field.
_OBSERVATION_KEY = "observation"

_bayesflow_module: ModuleType | None = None


def _import_bayesflow() -> ModuleType:
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
            "BayesFlow is required for the amortized SBI learners: "
            "pip install 'probpipe-core[bayesflow]'"
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


def _observation_slot(prior: Any) -> str:
    """The name of a learned kernel's observation field: ``observation``, unless the prior declares it.

    A kernel's given slots and event components are disjoint, and the prior's
    components name the parameters, so the observation takes the first of
    ``observation``, ``observation_``, and so on that the prior leaves free.
    """
    slot = _OBSERVATION_KEY
    while slot in prior.event_spec.components:
        slot += "_"
    return slot


def _adapter_field_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    """Positional internal keys (``theta_0``, ``theta_1``, ...) for the adapter.

    Both the training dict and the sample-side extraction derive these from the
    leaf order of the record the prior's components form (``leaf_shapes`` keys; ==
    ``fields`` for a flat prior), so the mapping is deterministic across train
    and inference without storing it, and slash-delimited nested leaf paths
    never reach BayesFlow's key namespace.
    """
    return tuple(f"theta_{i}" for i in range(len(keys)))


def _validate_learn_inputs(
    prior: Distribution,
    simulator: ConditionalDistribution,
    *,
    caller: str,
    sim_backend: SimBackend,
    counts: tuple[tuple[str, Any], ...],
    fit_kwargs: Mapping[str, Any],
) -> Any:
    """Shared train-time validation for the amortized learners; returns the
    record the prior's components form. Raises before any simulation runs.

    Parameters
    ----------
    prior : Distribution
        The learner's prior, which must declare a numeric event.
    simulator : ConditionalDistribution
        The learner's simulator, a kernel from the prior's fields to one
        observation, which must sample.
    caller : str
        The name of the public learner, which the error messages name.
    sim_backend : {"jax", "sequential"}
        The learner's simulation backend.
    counts : tuple of (str, Any)
        Each count argument's name and value, which must be a positive integer.
    fit_kwargs : Mapping[str, Any]
        The keywords the learner passes to ``approximator.fit``, which name no
        seed, since the training's seed is drawn from a workflow-owned random
        event.

    Returns
    -------
    Any
        The record that the components of the prior's event form.

    Raises
    ------
    ValueError
        If *sim_backend* is unknown or a count is less than one.
    TypeError
        If a count is not an integer, *fit_kwargs* holds ``random_seed`` or
        ``seed``, *simulator* is not a kernel that samples, or *prior* is not
        a numeric distribution.
    """
    refuse_seed_keywords(caller, fit_kwargs)
    if sim_backend not in ("jax", "sequential"):
        raise ValueError(unknown_names("sim_backend", [sim_backend], ["jax", "sequential"]))
    for _name, _val in counts:
        if not isinstance(_val, (int, np.integer)):
            raise TypeError(f"{_name} must be an integer; got {type(_val).__name__}")
        if _val < 1:
            raise ValueError(f"{_name} must be a positive integer; got {_val}")
    if not (
        isinstance(simulator, ConditionalDistribution)
        and isinstance(simulator, SupportsConditionalSampling)
    ):
        raise TypeError(
            "simulator must be a ConditionalDistribution that can be sampled, giving one "
            f"observation at the prior's parameters; got {type(simulator).__name__}"
        )
    if not isinstance(prior, NumericDistribution):
        raise TypeError(
            f"{caller} requires a numeric prior over named parameters, such as a product of "
            f"named distributions; got {type(prior).__name__}, whose draws are not numeric"
        )
    return _components_record(prior.event_spec)


@contextmanager
def _isolated_keras_seeding(random_seed: int):
    """Seed keras (which reads the global RNG for init + fit) for reproducible
    training, snapshotting and restoring the caller's Python/NumPy RNG state so
    the process-wide seeding does not leak. keras keeps its own seeded generator
    state across the restore."""
    import keras

    py_state = random.getstate()
    np_state = np.random.get_state()
    keras.utils.set_random_seed(random_seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


@contextmanager
def _without_progress_bar() -> Iterator[None]:
    """Turn off the progress bar of BayesFlow's sampler for the block.

    BayesFlow's sampler draws a ``tqdm`` bar on every call and has no setting
    that turns it off, so each draw of an amortized posterior would print one.
    A BayesFlow whose sampler module has another layout keeps its bar.
    """
    samplers = sys.modules.get("bayesflow.approximators.helpers.samplers")
    bar = getattr(samplers, "tqdm", None)
    if bar is None:
        yield
        return
    samplers.tqdm = functools.partial(bar, disable=True)
    try:
        yield
    finally:
        samplers.tqdm = bar


def _simulator_given(simulator: ConditionalDistribution, params: Any) -> Any:
    """The simulator's given values at *params*, the record of one prior draw.

    A simulator whose given slots are the prior's components receives the record
    itself, so it may read a nested field by its leaf path; any other receives
    the record of its slots.
    """
    slots = tuple(simulator.given_spec)
    if set(slots) == set(params.fields):
        return params
    return Record("given", {slot: params[slot] for slot in slots})


def _leaf_draws(draws: Any, leaf: str) -> Any:
    """The draws of the numeric leaf at the path *leaf*, from a law's raw batched draws.

    A record-drawing law's raw draws are a nested mapping of stacked leaves, and
    an array-drawing law's are the stacked array of its one leaf.
    """
    if isinstance(draws, Record):
        return draws.raw(leaf)
    if isinstance(draws, Mapping):
        node = draws
        for part in leaf.split("/"):
            node = node[part]
        return node
    return draws


def _simulate_offline(
    prior: Distribution,
    simulator: ConditionalDistribution,
    num_simulations: int,
    key: PRNGKey,
    *,
    sim_backend: SimBackend,
    bijectors: dict[str, Function] | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Draw ``(theta, y)`` pairs offline: ``theta ~ prior``, ``y ~ simulator(theta)``.

    Returns ``(named_theta, y)`` where ``named_theta`` maps each prior
    record-template numeric leaf (slash paths like ``"outer/a"`` for a nested
    prior; top-level fields for a flat one) to a ``(num_simulations, d_leaf)``
    float32 array and ``y`` is the ``(num_simulations, d_y)`` float32 array of
    flattened simulated observations.  With ``bijectors`` given (the NPE path),
    each theta leaf is *unconstrained* (pushed through its inverse bijector at the
    leaf's native event shape, then flattened); with ``bijectors=None`` (the
    NLE/NRE paths, where theta is a network *input* rather than a modeled
    density), the raw constrained draws are returned.  The simulator itself always
    sees the constrained, structured draws.
    """
    template = _components_record(prior.event_spec)
    # Iterate numeric leaves (slash paths like "outer/a" for nested priors; ==
    # top-level fields for flat priors). The adapter re-keys positionally, so
    # leaf paths never reach BayesFlow's namespace.
    leaf_keys = tuple(template.leaf_shapes)
    k_theta, k_sim = jax.random.split(key)
    draws = prior._sample(k_theta, (num_simulations,))
    columns = {leaf: jnp.asarray(_leaf_draws(draws, leaf)) for leaf in leaf_keys}
    theta_flat = jnp.concatenate(
        [jnp.reshape(column, (num_simulations, -1)) for column in columns.values()], axis=1
    )
    # Invert before flattening: matrix-valued bijectors (positive-definite) require
    # the leaf's native (..., n, n) event shape, not the flat adapter layout.
    named = {}
    for leaf, arr in columns.items():
        if bijectors is not None:
            arr = bijectors[leaf]._inverse(arr)
        named[leaf] = np.asarray(jnp.reshape(arr, (num_simulations, -1)), dtype="float32")
    sim_keys = jax.random.split(k_sim, num_simulations)

    def _one(flat_row: Array, k: PRNGKey) -> Array:
        # One observation of the simulator kernel at the per-draw structured
        # params. ``flat_row`` is 1-D, so from_vector rebuilds a single
        # NumericRecord.
        from ..core._numeric_record import _reconstruct_from_vector
        from ._inference_utils import flat_vector

        params = _reconstruct_from_vector("params", template, flat_row)
        return flat_vector(simulator._conditional_sample(_simulator_given(simulator, params), k))

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
