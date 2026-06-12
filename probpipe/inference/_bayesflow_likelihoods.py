"""Amortized neural likelihood (NLE) and ratio (NRE) surrogates via BayesFlow.

Trains a BayesFlow estimator of the conditional density ``p(y | theta)`` (NLE: a
conditional coupling flow) or of the likelihood-to-evidence ratio (NRE: an NRE-C
classifier) and wraps it as a :class:`~probpipe.core.protocols.Likelihood`
component whose ``log_likelihood`` is **jax.grad-transparent** -- so
``SimpleModel(prior, learned)`` + ``condition_on(model, data)`` runs ProbPipe's
existing BlackJAX/TFP NUTS machinery with no new samplers and no PyTorch.

Both wrappers are :class:`~probpipe.ConditionallyIndependentLikelihood`: the
estimator is trained on single ``(theta, y_i)`` pairs and a dataset's
log-likelihood is the sum of per-row scores, so multi-observation datasets work
natively. The networks condition on (NLE) or classify (NRE) the *raw constrained*
``theta``, matching what the MCMC log-density assembly passes at sampling time.

BayesFlow / keras load lazily on first use, so ``import probpipe`` does not pull
keras.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ..core.distribution import Distribution
from ..core.protocols import ConditionallyIndependentLikelihood, GenerativeLikelihood
from ..custom_types import Array, ArrayLike, PRNGKey
from ._bayesflow_common import (
    _OBSERVATION_KEY,
    SimBackend,
    _adapter_field_keys,
    _ensure_bayesflow,
    _seeded_keras_training,
    _simulate_offline,
    _validate_learn_inputs,
)

if TYPE_CHECKING:
    from bayesflow.networks import InferenceNetwork
    from keras import Layer as KerasLayer
    from keras.optimizers import Optimizer as KerasOptimizer
else:
    # Plain functions (deliberately NOT @workflow_function: the workflow result
    # boundary coerces returns into Record/Distribution, which would wrap these
    # Likelihood components); the runtime aliases keep the annotations
    # resolvable for any get_type_hints consumer.
    InferenceNetwork = KerasLayer = KerasOptimizer = Any


# ---------------------------------------------------------------------------
# Likelihood wrappers
# ---------------------------------------------------------------------------


class _BayesFlowLikelihoodBase(ConditionallyIndependentLikelihood, GenerativeLikelihood):
    """Shared surface of the learned-likelihood wrappers.

    Subclasses implement ``_row_scores(theta_rows, data_rows)`` -- the
    jax-traceable per-row score (a log-density for NLE, a log-ratio for NRE).
    Everything else is common: params/data coercion, the CIL sum over rows, and
    the ``generate_data`` passthrough to the training simulator.
    """

    def __init__(
        self,
        approximator: Any,
        prior: Distribution,
        simulator: GenerativeLikelihood,
        *,
        data_dim: int,
    ):
        self._approximator = approximator
        self._prior = prior
        self._simulator = simulator
        self._theta_dim = int(prior.event_size)
        self._data_dim = data_dim

    @property
    def prior(self) -> Distribution:
        """The prior the estimator was trained against."""
        return self._prior

    @property
    def simulator(self) -> GenerativeLikelihood:
        """The training simulator (provides ``generate_data``)."""
        return self._simulator

    @property
    def approximator(self) -> Any:
        """The trained BayesFlow approximator (for direct/advanced use)."""
        return self._approximator

    def _theta_row(self, params: Any) -> Array:
        """Coerce params (structured record or flat vector) to a ``(d_theta,)`` row."""
        if hasattr(params, "flatten"):
            t = params.flatten()
        elif hasattr(params, "to_numeric"):
            t = params.to_numeric().flatten()
        else:
            t = params
        t = jnp.ravel(jnp.asarray(t))
        if t.shape[0] != self._theta_dim:
            raise ValueError(
                f"params has {t.shape[0]} values but the estimator was trained on "
                f"{self._theta_dim}-dimensional parameters."
            )
        return t

    def _data_rows(self, data: ArrayLike | np.ndarray) -> Array:
        """Coerce data to ``(n, d_y)`` rows of the trained observation width."""
        rows = jnp.atleast_2d(jnp.asarray(data))
        if rows.shape[-1] != self._data_dim:
            raise ValueError(
                f"data rows have {rows.shape[-1]} values but the estimator was "
                f"trained on observations of size {self._data_dim}; pass a dataset "
                "of shape (n, d_y) (or a single (d_y,) observation) matching the "
                "simulator's flattened output."
            )
        return rows.reshape(-1, self._data_dim)

    def log_likelihood(self, params: Any, data: ArrayLike | np.ndarray) -> Array:
        """Sum of per-row scores over the dataset (conditionally independent rows)."""
        rows = self._data_rows(data)
        theta = self._theta_row(params)
        theta_rows = jnp.tile(theta[None, :], (rows.shape[0], 1))
        return jnp.sum(self._row_scores(theta_rows, rows))

    def per_datum_log_likelihood(self, params: Any, datum: Any) -> Array:
        """Score of a single observation row."""
        theta = self._theta_row(params)
        row = jnp.reshape(jnp.asarray(datum), (1, -1))
        if row.shape[-1] != self._data_dim:
            raise ValueError(
                f"datum has {row.shape[-1]} values but the estimator was trained "
                f"on observations of size {self._data_dim}."
            )
        return self._row_scores(theta[None, :], row)[0]

    def generate_data(
        self, params: Any, num_observations: int, *, key: PRNGKey | None = None,
    ) -> Any:
        """Delegate to the training simulator (``GenerativeLikelihood`` passthrough)."""
        return self._simulator.generate_data(params, num_observations, key=key)

    def _row_scores(self, theta_rows: Array, data_rows: Array) -> Array:
        raise NotImplementedError


class BayesFlowLikelihood(_BayesFlowLikelihoodBase):
    """A learned amortized likelihood: ``log_likelihood(theta, data)`` evaluates the
    trained conditional density ``sum_i log p_net(y_i | theta)``.

    The score path is pure keras-jax ops (standardize, conditional-flow
    ``log_prob``, plus the standardization log-det-jacobian), so it is
    ``jax.grad``-transparent and jit-stable -- a drop-in
    :class:`~probpipe.core.protocols.Likelihood` /
    :class:`~probpipe.ConditionallyIndependentLikelihood` for
    ``SimpleModel`` + ``condition_on`` gradient-based MCMC. Values are faithful
    to the public ``approximator.log_prob`` (same standardization and
    log-det-jacobian); the only difference is staying on-device.
    """

    def _row_scores(self, theta_rows: Array, data_rows: Array) -> Array:
        a = self._approximator
        conds = a.standardizer.maybe_standardize(
            theta_rows, key="inference_conditions", stage="inference")
        z, ldj = a.standardizer.maybe_standardize(
            data_rows, key="inference_variables", stage="inference", log_det_jac=True)
        return a.inference_network.log_prob(z, conditions=conds) + ldj

    def __repr__(self) -> str:
        return (
            f"BayesFlowLikelihood(theta_dim={self._theta_dim}, "
            f"data_dim={self._data_dim})"
        )


class BayesFlowRatio(_BayesFlowLikelihoodBase):
    """A learned likelihood-to-evidence ratio: per-row scores are the NRE-C
    classifier logits, which converge to ``log[p(y_i | theta) / p(y_i)]``.

    ``log_likelihood`` sums the per-row log-ratios, which equals the true joint
    log-likelihood **up to a theta-independent constant** (``sum_i log p(y_i)``).
    That makes it a valid drop-in for conditioning / MCMC -- the constant cancels
    -- but the values are *not* normalized log-likelihoods: do not use them for
    model comparison, information criteria (LOO / WAIC), or any reading of
    absolute likelihood magnitudes.

    Because the estimator is a classifier (an MLP over ``concat(theta, y)``), it
    has no continuous-density machinery: it handles **discrete-valued
    observations** natively and has no minimum observation dimension -- the two
    cases where :class:`BayesFlowLikelihood`'s coupling flow does not apply.
    """

    def _row_scores(self, theta_rows: Array, data_rows: Array) -> Array:
        a = self._approximator
        thv = a.standardizer.maybe_standardize(
            theta_rows, key="inference_variables", stage="inference")
        conds = a.standardizer.maybe_standardize(
            data_rows, key="inference_conditions", stage="inference")
        return a.logits(thv, conds, stage="inference")

    def __repr__(self) -> str:
        return (
            f"BayesFlowRatio(theta_dim={self._theta_dim}, "
            f"data_dim={self._data_dim})"
        )


# ---------------------------------------------------------------------------
# Learner entry points (plain functions; see the module-top note on why these
# are not @workflow_function)
# ---------------------------------------------------------------------------


def _train_offline(
    prior: Distribution,
    simulator: GenerativeLikelihood,
    *,
    caller: str,
    num_simulations: int,
    epochs: int,
    batch_size: int,
    sim_backend: SimBackend,
    random_seed: int,
    theta_role: str,
    build_approximator: Any,
    optimizer: str | KerasOptimizer,
    fit_kwargs: dict[str, Any],
) -> tuple[Any, int]:
    """Shared NLE/NRE training loop: validate, simulate, adapt, fit.

    ``theta_role`` is the adapter slot the (raw, constrained) theta fields feed
    -- ``"inference_conditions"`` for NLE, ``"inference_variables"`` for NRE --
    with the observation taking the other slot. Returns ``(approximator, d_y)``.
    """
    record_template = _validate_learn_inputs(
        prior, simulator, caller=caller, sim_backend=sim_backend,
        counts=(("num_simulations", num_simulations), ("batch_size", batch_size),
                ("epochs", epochs)),
    )
    fields = record_template.fields

    bf = _ensure_bayesflow()
    with _seeded_keras_training(random_seed):
        key = jax.random.PRNGKey(random_seed)
        named, y = _simulate_offline(
            prior, simulator, num_simulations, key,
            sim_backend=sim_backend, bijectors=None,   # raw theta: it is a net input
        )
        internal_keys = _adapter_field_keys(fields)
        sims = {k: named[f] for k, f in zip(internal_keys, fields)}
        sims[_OBSERVATION_KEY] = y

        obs_role = ("inference_variables" if theta_role == "inference_conditions"
                    else "inference_conditions")
        adapter = (
            bf.Adapter()
            .convert_dtype("float64", "float32")
            .concatenate(list(internal_keys), into=theta_role)
            .concatenate([_OBSERVATION_KEY], into=obs_role)
        )
        approximator = build_approximator(bf, adapter, int(y.shape[-1]))
        approximator.compile(optimizer=optimizer)
        dataset = bf.OfflineDataset(data=sims, batch_size=batch_size, adapter=adapter)
        approximator.fit(dataset=dataset, epochs=epochs, **fit_kwargs)
    return approximator, int(y.shape[-1])


def learn_amortized_likelihood(
    prior: Distribution,
    simulator: GenerativeLikelihood,
    *,
    num_simulations: int = 10_000,
    epochs: int = 50,
    batch_size: int = 128,
    sim_backend: SimBackend = "jax",
    inference_network: InferenceNetwork | None = None,
    random_seed: int = 0,
    optimizer: str | KerasOptimizer = "adam",
    **fit_kwargs: Any,
) -> BayesFlowLikelihood:
    """Learn an amortized likelihood ``p(y | theta)`` (NLE) with BayesFlow.

    Trains a conditional coupling flow on offline ``(theta, y)`` simulations and
    returns a :class:`BayesFlowLikelihood` -- a
    :class:`~probpipe.ConditionallyIndependentLikelihood` whose jax-traceable
    ``log_likelihood`` plugs into ``SimpleModel(prior, learned)`` +
    :func:`~probpipe.condition_on`, so the existing gradient-based MCMC machinery
    (BlackJAX/TFP NUTS) samples the posterior -- for single observations or
    multi-observation datasets (per-row scores sum under conditional
    independence). The network conditions on the raw constrained ``theta``.

    Parameters
    ----------
    prior : Distribution
        Prior over the model parameters; a ``RecordDistribution`` with named,
        non-nested fields. Sampled (only) to draw training thetas; constrained
        and discrete-valued parameter fields are both fine here, since theta is
        a network *input* (whether the downstream sampler can handle the prior
        is the sampler's concern).
    simulator : GenerativeLikelihood
        ``generate_data(params, num_observations, *, key)``; receives the
        prior's structured per-draw record (named-field access). Must be
        JAX-vmappable unless ``sim_backend="sequential"``.
    num_simulations, epochs, batch_size : int
        Offline simulation count and keras training schedule.
    sim_backend : {"jax", "sequential"}
        ``"jax"`` (default) vmaps the simulator; ``"sequential"`` runs an eager
        per-draw loop for non-JAX simulators.
    inference_network : bayesflow.networks.InferenceNetwork or None
        Overrides the default ``CouplingFlow``. The density must be
        reverse-mode differentiable for gradient-based MCMC -- adaptive-ODE
        networks (``FlowMatching``, ``DiffusionModel``) are **not** (their
        ``log_prob`` integrates with a dynamic-bound ``while_loop``).
    random_seed : int
        Seeds simulation, network init, and training; the caller's global
        NumPy / Python RNG state is restored afterwards.
    optimizer : str or keras.Optimizer
        Passed to ``approximator.compile``.
    **fit_kwargs
        Forwarded to ``approximator.fit`` (e.g. ``callbacks``, ``verbose``).

    Returns
    -------
    BayesFlowLikelihood

    Raises
    ------
    ValueError
        If ``sim_backend`` is unknown, a count parameter is less than one, or
        the simulated observations are one-dimensional with the default network
        (the coupling flow needs ``d_y >= 2``; use
        :func:`learn_amortized_ratio`, whose classifier has no minimum, or pass
        a custom ``inference_network``).
    TypeError
        If a count parameter is not an integer, ``simulator`` lacks
        ``generate_data``, or ``prior`` is not a flat ``RecordDistribution``.
    ImportError
        If the ``[bayesflow]`` extra is not installed.
    """

    def _build(bf: Any, adapter: Any, data_dim: int) -> Any:
        if inference_network is None and data_dim < 2:
            raise ValueError(
                "NLE's default coupling flow requires observations with at least 2 "
                f"dimensions, but the simulator emits {data_dim}-dimensional data. "
                "Use learn_amortized_ratio (the NRE classifier has no minimum "
                "dimension) or pass a custom inference_network."
            )
        net = inference_network or bf.networks.CouplingFlow()
        return bf.ContinuousApproximator(inference_network=net, adapter=adapter)

    approximator, data_dim = _train_offline(
        prior, simulator, caller="learn_amortized_likelihood",
        num_simulations=num_simulations, epochs=epochs, batch_size=batch_size,
        sim_backend=sim_backend, random_seed=random_seed,
        theta_role="inference_conditions", build_approximator=_build,
        optimizer=optimizer, fit_kwargs=fit_kwargs,
    )
    return BayesFlowLikelihood(approximator, prior, simulator, data_dim=data_dim)


def learn_amortized_ratio(
    prior: Distribution,
    simulator: GenerativeLikelihood,
    *,
    num_simulations: int = 10_000,
    epochs: int = 50,
    batch_size: int = 128,
    sim_backend: SimBackend = "jax",
    inference_network: KerasLayer | None = None,
    random_seed: int = 0,
    optimizer: str | KerasOptimizer = "adam",
    **fit_kwargs: Any,
) -> BayesFlowRatio:
    """Learn an amortized likelihood-to-evidence ratio (NRE) with BayesFlow.

    Trains an NRE-C classifier (``RatioApproximator``; contrastive pairs are
    built internally by shuffling theta within each batch, so the training data
    are the same offline ``(theta, y)`` simulations as NLE) and returns a
    :class:`BayesFlowRatio` whose summed per-row log-ratios stand in for the
    log-likelihood **up to a theta-independent constant** -- valid for
    ``SimpleModel`` + ``condition_on`` MCMC, invalid for absolute-likelihood
    uses (model comparison, LOO/WAIC); see the class docstring. The classifier
    handles discrete-valued observations and one-dimensional data natively.

    Parameters
    ----------
    prior : Distribution
        Prior over the model parameters; a flat ``RecordDistribution`` (as in
        :func:`learn_amortized_likelihood` -- constrained and discrete-valued
        parameter fields are fine, theta is a network input).
    simulator : GenerativeLikelihood
        ``generate_data(params, num_observations, *, key)``; receives the
        prior's structured per-draw record.
    num_simulations : int
        Number of ``(theta, y)`` pairs simulated offline for training.
    epochs, batch_size : int
        keras training schedule.
    sim_backend : {"jax", "sequential"}
        ``"jax"`` (default) vmaps the simulator; ``"sequential"`` runs an eager
        per-draw loop for non-JAX simulators.
    inference_network : keras.Layer or None
        The classifier body (defaults to ``bayesflow.networks.MLP()``); the
        ``RatioApproximator`` adds its own scalar projection head.
    random_seed : int
        Seeds simulation, network init, and training; the caller's global
        NumPy / Python RNG state is restored afterwards.
    optimizer : str or keras.Optimizer
        Passed to ``approximator.compile``.
    **fit_kwargs
        Forwarded to ``approximator.fit`` (e.g. ``callbacks``, ``verbose``).

    Returns
    -------
    BayesFlowRatio

    Raises
    ------
    ValueError
        If ``sim_backend`` is unknown or a count parameter is less than one
        (no minimum observation dimension, unlike NLE).
    TypeError
        If a count parameter is not an integer, ``simulator`` lacks
        ``generate_data``, or ``prior`` is not a flat ``RecordDistribution``.
    ImportError
        If the ``[bayesflow]`` extra is not installed.
    """

    def _build(bf: Any, adapter: Any, data_dim: int) -> Any:
        net = inference_network if inference_network is not None else bf.networks.MLP()
        return bf.approximators.RatioApproximator(
            inference_network=net, adapter=adapter)

    approximator, data_dim = _train_offline(
        prior, simulator, caller="learn_amortized_ratio",
        num_simulations=num_simulations, epochs=epochs, batch_size=batch_size,
        sim_backend=sim_backend, random_seed=random_seed,
        theta_role="inference_variables", build_approximator=_build,
        optimizer=optimizer, fit_kwargs=fit_kwargs,
    )
    return BayesFlowRatio(approximator, prior, simulator, data_dim=data_dim)
