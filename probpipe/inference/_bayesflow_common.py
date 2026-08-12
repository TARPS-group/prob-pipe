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

import os
import random
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..core._record_array import NumericRecordArray
from ..core.distribution import Distribution
from ..core.ops import sample as _sample_op
from ..core.protocols import GenerativeLikelihood
from ..custom_types import Array, PRNGKey

if TYPE_CHECKING:
    # Type-only: tfp is a hard dependency but is only needed here for
    # bijector annotations.
    import tensorflow_probability.substrates.jax.bijectors as tfb

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


def _adapter_field_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    """Positional internal keys (``theta_0``, ``theta_1``, ...) for the adapter.

    Both the training dict and the sample-side extraction derive these from the
    prior's ``event_template`` leaf order (``leaf_shapes`` keys; ==
    ``fields`` for a flat prior), so the mapping is deterministic across train
    and inference without storing it, and slash-delimited nested leaf paths
    never reach BayesFlow's key namespace.
    """
    return tuple(f"theta_{i}" for i in range(len(keys)))


def _validate_learn_inputs(
    prior: Distribution,
    simulator: GenerativeLikelihood,
    *,
    caller: str,
    sim_backend: SimBackend,
    counts: tuple[tuple[str, Any], ...],
) -> Any:
    """Shared train-time validation for the amortized learners; returns the
    prior's record template. Raises before any simulation runs."""
    if sim_backend not in ("jax", "sequential"):
        raise ValueError(f"Unknown sim_backend: {sim_backend!r}. Supported: 'jax', 'sequential'.")
    for _name, _val in counts:
        if not isinstance(_val, (int, np.integer)):
            raise TypeError(f"{_name} must be an integer, got {type(_val).__name__}.")
        if _val < 1:
            raise ValueError(f"{_name} must be a positive integer, got {_val}.")
    if not hasattr(simulator, "generate_data"):
        raise TypeError(
            "simulator must be a GenerativeLikelihood with a generate_data method, "
            f"got {type(simulator).__name__}"
        )
    event_template = getattr(prior, "event_template", None)
    if event_template is None:
        raise TypeError(
            f"{caller} requires a RecordDistribution prior with named parameter "
            "fields -- typically a ProductDistribution of named distributions -- "
            f"but got {type(prior).__name__}, which has no event_template."
        )
    return event_template


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


def _simulate_offline(
    prior: Distribution,
    simulator: GenerativeLikelihood,
    num_simulations: int,
    key: PRNGKey,
    *,
    sim_backend: SimBackend,
    bijectors: dict[str, tfb.Bijector] | None,
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
    template = prior.event_template
    # Iterate numeric leaves (slash paths like "outer/a" for nested priors; ==
    # top-level fields for flat priors). The adapter re-keys positionally, so
    # leaf paths never reach BayesFlow's namespace.
    leaf_keys = tuple(template.leaf_shapes)
    k_theta, k_sim = jax.random.split(key)
    theta = _sample_op(prior, key=k_theta, sample_shape=(num_simulations,))
    # Round-trip through the canonical 1-D vector layout: single-field priors'
    # raw draws are not field-indexable by name, so from_vector gives uniform
    # named access. Structured draws serialize via to_vector; raw arrays ravel.
    if isinstance(theta, NumericRecordArray):
        theta_flat = jnp.asarray(theta.to_vector()).reshape(num_simulations, -1)
    else:
        theta_flat = jnp.asarray(theta).reshape(num_simulations, -1)
    from ..core._numeric_record import _reconstruct_from_vector

    record = _reconstruct_from_vector(prior.name, template, theta_flat, name_is_auto=True)
    # Invert before flattening: matrix-valued bijectors (positive-definite) require
    # the leaf's native (..., n, n) event shape, not the flat adapter layout.
    named = {}
    for leaf in leaf_keys:
        arr = jnp.asarray(record[leaf])
        if bijectors is not None:
            arr = bijectors[leaf].inverse(arr)
        named[leaf] = np.asarray(jnp.reshape(arr, (num_simulations, -1)), dtype="float32")
    sim_keys = jax.random.split(k_sim, num_simulations)

    def _one(flat_row: Array, k: PRNGKey) -> Array:
        # Per-draw structured params (named-field access), per the
        # GenerativeLikelihood contract. ``flat_row`` is 1-D, so from_vector
        # rebuilds a single NumericRecord.
        from ..core._numeric_record import _reconstruct_from_vector

        params = _reconstruct_from_vector("params", template, flat_row, name_is_auto=True)
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
