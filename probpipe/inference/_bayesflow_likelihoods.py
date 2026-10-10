"""Amortized neural likelihood (NLE) and ratio (NRE) kernels via BayesFlow.

Trains a BayesFlow estimator of the conditional density ``p(y | theta)`` (NLE: a
conditional coupling flow) or of the likelihood-to-evidence ratio (NRE: an NRE-C
classifier) and returns it as a ``ConditionalDistribution`` from the parameters
to datasets of observation rows: NLE's kernel has the network's density and
NRE's an unnormalized one. The scores are **jax.grad-transparent**, so
``condition_on(learned * prior, {"observation": y})`` runs ProbPipe's registered
BlackJAX/TFP NUTS methods with no new samplers and no PyTorch.

Both kernels treat a dataset's rows as conditionally independent: the estimator
is trained on single ``(theta, y_i)`` pairs and a dataset's score is the sum of
per-row scores, so datasets of any size work natively (NPE, by contrast,
conditions on a shape fixed at training time). The networks condition on (NLE)
or classify (NRE) the *raw constrained* ``theta``, matching what the MCMC
log-density assembly passes at sampling time.

BayesFlow / keras load lazily on first use, so ``import probpipe`` does not pull
keras.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from .._messages import unknown_names
from ..core._spec_base import NumericArraySpec
from ..core._specs import OutputSpec
from ..core.record import Record
from ..custom_types import Array, ArrayLike
from ..distributions._capabilities import (
    SupportsConditionalLogProb,
    SupportsConditionalUnnormalizedLogProb,
    SupportsLogProb,
    SupportsUnnormalizedLogProb,
)
from ..distributions._conditional import ConditionalDistribution, ConditionalDistributionSpec
from ..distributions._distribution import Distribution
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
    from bayesflow.approximators import ContinuousApproximator, RatioApproximator
    from bayesflow.networks import InferenceNetwork
    from keras import Layer as KerasLayer
    from keras.optimizers import Optimizer as KerasOptimizer
else:
    # Runtime aliases so the optional-dependency names in signatures stay
    # resolvable for get_type_hints consumers.
    ContinuousApproximator = RatioApproximator = Any
    InferenceNetwork = KerasLayer = KerasOptimizer = Any


# ---------------------------------------------------------------------------
# Learned kernels from the parameters to the observations
# ---------------------------------------------------------------------------


class _LearnedLaw(Distribution):
    """The law a learned kernel yields at a value of every parameter, over datasets of rows."""

    def __init__(self, kernel: _BayesFlowLikelihoodBase, values: Mapping[str, Any]) -> None:
        super().__init__(
            kernel.event_spec,
            label=kernel.label,
        )
        self._kernel = kernel
        self._values = dict(values)

    def _score(self, value: Any) -> Array:
        """The kernel's score of the dataset *value* at this law's parameters."""
        return self._kernel._score(self._values, value)


class _LearnedDensity(_LearnedLaw, SupportsLogProb):
    """A learned likelihood's law at a parameter value, whose density is the network's."""

    def _log_prob(self, value: Any) -> Array:
        """The sum of the network's log-densities of the dataset's rows."""
        return self._score(value)


class _LearnedRatioLaw(_LearnedLaw, SupportsUnnormalizedLogProb):
    """A learned ratio's law at a parameter value, whose density is known up to a constant."""

    def _unnormalized_log_prob(self, value: Any) -> Array:
        """The sum of the classifier's log-ratios of the dataset's rows."""
        return self._score(value)


class _BayesFlowLikelihoodBase(ConditionalDistribution):
    """A learned kernel from the parameters to conditionally independent observation rows.

    Its given slots are the prior's components, so composing it with the prior
    it was trained against, as ``learned * prior``, gives the joint law, and
    conditioning that joint on an observation is Bayes' rule, which the
    normalization stage completes by inference. Its event is a dataset of
    observation rows, of any number, whose score is the sum of per-row scores
    under conditional independence. Subclasses implement :meth:`_row_scores`,
    the jax-traceable per-row score, and :meth:`_law`.

    Parameters
    ----------
    approximator : ContinuousApproximator or RatioApproximator
        The trained BayesFlow approximator whose components the subclass's
        ``_row_scores`` calls.
    prior : Distribution
        The prior the estimator was trained against; its ``event_size`` fixes
        the expected ``theta`` width, and its components are the given slots.
    simulator : ConditionalDistribution
        The training simulator.
    data_dim : int
        Flattened per-row observation width the network was trained on (fixed
        by the simulator's per-draw output at training time).
    label : str
        The kernel's label.
    """

    def __init__(
        self,
        approximator: ContinuousApproximator | RatioApproximator,
        prior: Distribution,
        simulator: ConditionalDistribution,
        *,
        data_dim: int,
        label: str,
    ):
        super().__init__(
            dict(prior.event_spec.components),
            OutputSpec(**{_observation_slot(prior): NumericArraySpec(("observations", data_dim))}),
            label=label,
        )
        attributes = {
            "_approximator": approximator,
            "_prior": prior,
            "_simulator": simulator,
            "_theta_dim": int(prior.event_spec.spec.vector_size),
            "_data_dim": data_dim,
            "_bound": {},
        }
        for attribute, value in attributes.items():
            object.__setattr__(self, attribute, value)

    @property
    def prior(self) -> Distribution:
        """The prior the estimator was trained against."""
        return self._prior

    @property
    def simulator(self) -> ConditionalDistribution:
        """The training simulator."""
        return self._simulator

    @property
    def approximator(self) -> ContinuousApproximator | RatioApproximator:
        """The trained BayesFlow approximator (for direct/advanced use)."""
        return self._approximator

    def _theta_row(self, values: Mapping[str, Any]) -> Array:
        """The parameters, keyed by the prior's components, as a ``(d_theta,)`` row.

        The row is the record's canonical 1-D vector layout, which matches the
        training layout. Its width is validated against the prior's
        ``event_size`` (static under jit: shapes are concrete at trace time).
        """
        components = tuple(self._prior.event_spec.components)
        t = (
            Record(
                {name: values[name] for name in components},
                label="params",
            )
            .to_numeric()
            .to_vector()
        )
        t = jnp.ravel(jnp.asarray(t))
        if t.shape[0] != self._theta_dim:
            raise ValueError(
                f"params has {t.shape[0]} values, but {self.label!r} was trained on "
                f"parameters of {self._theta_dim} values"
            )
        return t

    def _data_rows(self, data: ArrayLike | np.ndarray) -> Array:
        """Coerce ``data`` to ``(n, d_y)`` rows of the trained observation width.

        A single ``(d_y,)`` observation becomes one row and ``(n, d_y)`` passes
        through; for scalar observations (``d_y == 1``) a 1-D array is read as
        ``n`` observations, not one ``n``-wide row. Higher-rank inputs collapse
        to rows provided the trailing axis is ``d_y`` (validated; static under
        jit).
        """
        rows = jnp.asarray(data)
        if self._data_dim == 1 and rows.ndim == 1:
            rows = rows[:, None]
        else:
            rows = jnp.atleast_2d(rows)
        if rows.shape[-1] != self._data_dim:
            raise ValueError(
                f"each observation in data has {rows.shape[-1]} values, but {self.label!r} was "
                f"trained on observations of size {self._data_dim}. Pass data of shape "
                f"(n, {self._data_dim}) or a single observation of shape ({self._data_dim},)."
            )
        return rows.reshape(-1, self._data_dim)

    def _values(self, given: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """The parameter values bound so far and those *given* binds, by slot.

        Parameters
        ----------
        given : Record or Mapping of str to Any
            The values ``condition_on`` passes positionally, by slot.
        kwargs : Mapping of str to Any
            The values ``condition_on`` passes by keyword, which take precedence
            over *given*.

        Returns
        -------
        dict of str to Any
            The earlier bindings, updated with *given* and then with *kwargs*.

        Raises
        ------
        KeyError
            If a name is not a given slot.
        """
        values = {**dict(given.children if isinstance(given, Record) else given), **kwargs}
        unknown = sorted(set(values) - set(self.given_spec))
        if unknown:
            raise KeyError(unknown_names("parameter", unknown, list(self.given_spec)))
        return {**self._bound, **values}

    def _score(self, values: Mapping[str, Any], data: Any) -> Array:
        """The dataset's score at the parameters: the sum of per-row scores.

        The sum is the conditionally-independent joint (rows are iid given
        ``theta``), evaluated in **one batched network call** -- ``theta`` is
        tiled across rows rather than looped -- so a NUTS step costs a single
        forward pass regardless of dataset size.
        """
        if isinstance(data, Record):
            data = data[_observation_slot(self._prior)]
        rows = self._data_rows(data)
        theta = self._theta_row(values)
        theta_rows = jnp.tile(theta[None, :], (rows.shape[0], 1))
        return jnp.sum(self._row_scores(theta_rows, rows))

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **kwargs: Any
    ) -> Distribution | ConditionalDistribution:
        """The law of the observations at a value of every parameter, or the kernel over the rest."""
        values = self._values(given, kwargs)
        left = {slot: spec for slot, spec in self.given_spec.items() if slot not in values}
        if not left:
            return self._law(values)
        curried = self._shallow_copy()
        object.__setattr__(curried, "_provenance", None)
        object.__setattr__(curried, "_bound", values)
        object.__setattr__(curried, "_spec", ConditionalDistributionSpec(left, self.event_spec))
        return curried

    @abstractmethod
    def _law(self, values: Mapping[str, Any]) -> _LearnedLaw:
        """The law of the observations at the parameter *values*."""

    @abstractmethod
    def _row_scores(self, theta_rows: Array, data_rows: Array) -> Array:
        """Per-row scores for ``theta_rows`` ``(n, d_theta)`` against ``data_rows``
        ``(n, d_y)``, returned as ``(n,)``. Must be jax-traceable and
        reverse-mode differentiable in ``theta_rows`` -- this is the gradient-MCMC
        hot path."""


class BayesFlowLikelihood(_BayesFlowLikelihoodBase, SupportsConditionalLogProb):
    """A learned amortized likelihood ``p̂(y | theta)``: a kernel with the network's density.

    The density of a dataset at ``theta`` is ``sum_i log p_net(y_i | theta)``.
    The score path is pure keras-jax ops (standardize, conditional-flow
    ``log_prob``, plus the standardization log-det-jacobian), so it is
    ``jax.grad``-transparent and jit-stable, and gradient-based MCMC normalizes
    ``condition_on(learned * prior, {"observation": y})``. Values are faithful
    to the public ``approximator.log_prob`` (same standardization and
    log-det-jacobian); the only difference is staying on-device.

    With ``dequantized=True`` (set by ``learn_amortized_likelihood``'s
    ``dequantize`` flag), the flow was trained on uniformly jittered
    observations ``y + U[0,1)^d``, and scoring shifts integer-valued data to
    the unit-cell midpoint -- values then equal the public ``log_prob``
    evaluated at ``y + 1/2``, a one-point approximation of the implied pmf
    ``P(y | theta) = integral over [y, y+1)^d of p(u | theta) du``
    (Theis et al., 2016, arXiv:1511.01844). Pass raw integer-valued
    observations; the kernel owns the cell convention.
    """

    def __init__(
        self,
        approximator: ContinuousApproximator,
        prior: Distribution,
        simulator: ConditionalDistribution,
        *,
        data_dim: int,
        label: str | None = None,
        dequantized: bool = False,
    ):
        super().__init__(
            approximator,
            prior,
            simulator,
            data_dim=data_dim,
            label="BayesFlowLikelihood" if label is None else label,
        )
        object.__setattr__(self, "_dequantized", dequantized)

    def _law(self, values: Mapping[str, Any]) -> _LearnedDensity:
        return _LearnedDensity(self, values)

    def _conditional_log_prob(self, given: Record | Mapping[str, Any], value: Any) -> Array:
        """The network's log-density of the dataset *value* at the parameters *given* binds."""
        return self._score(self._values(given, {}), value)

    def _row_scores(self, theta_rows: Array, data_rows: Array) -> Array:
        # The public approximator.log_prob is host-bound at both edges (adapter
        # numpy in, convert_to_numpy out), so this re-traces its math on-device:
        # standardize both slots exactly as training did -- theta feeds the
        # "inference_conditions" slot, y is the modeled "inference_variables" --
        # then ask the flow for the density of standardized y. Adding the
        # standardization log-det-jacobian converts that density back to the
        # original y units, making the values equal to the public log_prob.
        if self._dequantized:
            data_rows = data_rows + 0.5  # midpoint of the trained cell [y, y+1)
        a = self._approximator
        conds = a.standardizer.maybe_standardize(
            theta_rows, key="inference_conditions", stage="inference"
        )
        z, ldj = a.standardizer.maybe_standardize(
            data_rows, key="inference_variables", stage="inference", log_det_jac=True
        )
        return a.inference_network.log_prob(z, conditions=conds) + ldj

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The parameter and data dimensions, and whether the data are dequantized."""
        fields = [("theta_dim", repr(self._theta_dim)), ("data_dim", repr(self._data_dim))]
        if self._dequantized:
            fields.append(("dequantized", "True"))
        return fields


class BayesFlowRatio(_BayesFlowLikelihoodBase, SupportsConditionalUnnormalizedLogProb):
    """A learned likelihood-to-evidence ratio: a kernel with an unnormalized density.

    The per-row scores are the NRE-C classifier logits, which converge to
    ``log[p(y_i | theta) / p(y_i)]``, so a dataset's score equals its joint
    log-likelihood **up to a theta-independent constant** (``sum_i log p(y_i)``).
    The kernel therefore claims ``SupportsConditionalUnnormalizedLogProb``
    alone: conditioning ``learned * prior`` on an observation cancels the
    constant, but the values are not normalized log-likelihoods, so do not use
    them for model comparison, information criteria (LOO / WAIC), or any
    reading of absolute likelihood magnitudes.

    Because the estimator is a classifier (an MLP over ``concat(theta, y)``), it
    has no continuous-density machinery: it handles **discrete-valued
    observations** natively (no ``dequantize`` flag needed, and mixed
    discrete/continuous rows are fine) and has no minimum observation
    dimension -- the two cases where :class:`BayesFlowLikelihood`'s coupling
    flow needs, respectively, dequantization or a custom network.
    """

    def __init__(
        self,
        approximator: RatioApproximator,
        prior: Distribution,
        simulator: ConditionalDistribution,
        *,
        data_dim: int,
        label: str | None = None,
    ):
        super().__init__(
            approximator,
            prior,
            simulator,
            data_dim=data_dim,
            label="BayesFlowRatio" if label is None else label,
        )

    def _law(self, values: Mapping[str, Any]) -> _LearnedRatioLaw:
        return _LearnedRatioLaw(self, values)

    def _conditional_unnormalized_log_prob(
        self, given: Record | Mapping[str, Any], value: Any
    ) -> Array:
        """The classifier's log-ratios of the dataset *value* at the parameters, summed."""
        return self._score(self._values(given, {}), value)

    def _row_scores(self, theta_rows: Array, data_rows: Array) -> Array:
        # NRE swaps the adapter roles relative to NLE: theta is the *classified*
        # quantity ("inference_variables"), y the conditioning input
        # ("inference_conditions"). The classifier head's logits converge to the
        # log density ratio. No log-det-jacobian term here: standardization is
        # the same fixed transform in the ratio's numerator and denominator, so
        # its jacobian cancels -- the logits already live in original units.
        a = self._approximator
        thv = a.standardizer.maybe_standardize(
            theta_rows, key="inference_variables", stage="inference"
        )
        conds = a.standardizer.maybe_standardize(
            data_rows, key="inference_conditions", stage="inference"
        )
        return a.logits(thv, conds, stage="inference")

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The parameter and data dimensions."""
        return [("theta_dim", repr(self._theta_dim)), ("data_dim", repr(self._data_dim))]


# ---------------------------------------------------------------------------
# Learner entry points. Plain functions, as they were for the likelihood
# components these kernels replace; see STYLE_GUIDE 1.4.
# ---------------------------------------------------------------------------


def _train_offline(
    prior: Distribution,
    simulator: ConditionalDistribution,
    *,
    caller: str,
    num_simulations: int,
    epochs: int,
    batch_size: int,
    sim_backend: SimBackend,
    theta_role: str,
    build_approximator: Any,
    optimizer: str | KerasOptimizer,
    fit_kwargs: dict[str, Any],
    dequantize: bool = False,
) -> tuple[ContinuousApproximator | RatioApproximator, int]:
    """Shared NLE/NRE training loop: validate, simulate, adapt, fit.

    ``theta_role`` is the adapter slot the (raw, constrained) theta fields feed
    -- ``"inference_conditions"`` for NLE, ``"inference_variables"`` for NRE --
    with the observation taking the other slot. With ``dequantize``, U[0,1)
    jitter is added to the simulated observations after simulation (the
    simulator stays untouched). Once the inputs are valid, the training's seed
    is drawn from a workflow-owned random event, and it seeds the simulation,
    the jitter, and keras.

    Parameters
    ----------
    prior : Distribution
        The learner's prior.
    simulator : ConditionalDistribution
        The learner's simulator, a kernel from the prior's fields to one
        observation.
    caller : str
        The name of the public learner, which the error messages name and the
        seed's event records as its provider.
    num_simulations : int
        The number of simulations the network trains on.
    epochs : int
        The number of training epochs.
    batch_size : int
        The size of each training batch.
    sim_backend : {"jax", "sequential"}
        The learner's simulation backend.
    theta_role : str
        The adapter slot of the theta fields.
    build_approximator : callable
        ``build_approximator(bf, adapter, data_dim)``, the BayesFlow
        approximator to train.
    optimizer : str or keras.Optimizer
        The optimizer that ``approximator.compile`` receives.
    fit_kwargs : dict
        The keyword arguments of ``approximator.fit``.
    dequantize : bool
        Whether to add ``U[0,1)`` jitter to the simulated observations.

    Returns
    -------
    tuple
        ``(approximator, d_y)``: the trained approximator and the flattened
        size of one observation.
    """
    record = _validate_learn_inputs(
        prior,
        simulator,
        caller=caller,
        sim_backend=sim_backend,
        counts=(
            ("num_simulations", num_simulations),
            ("batch_size", batch_size),
            ("epochs", epochs),
        ),
        fit_kwargs=fit_kwargs,
    )
    # Numeric leaves (slash paths for a nested prior; == fields for a flat one).
    # NLE/NRE feed raw theta to the network, so no bijectors -- just the keying.
    leaf_keys = tuple(record.leaf_shapes)

    bf = _import_bayesflow()
    random_seed = integer_seed(run_seed(caller))
    with _isolated_keras_seeding(random_seed):
        key = jax.random.PRNGKey(random_seed)
        k_jitter = None
        if dequantize:
            # A dedicated sibling key, split off before the simulator sees the
            # key. Never derive twice from one key: jax's threefry makes
            # fold_in(key, i) IDENTICAL to split(key, n)[i], so a folded
            # "extra" stream would alias the simulation keys. Splitting only
            # under the flag keeps the dequantize=False key path -- and every
            # tolerance measured on it -- bit-identical.
            key, k_jitter = jax.random.split(key)
        named, y = _simulate_offline(
            prior,
            simulator,
            num_simulations,
            key,
            sim_backend=sim_backend,
            bijectors=None,  # raw theta: it is a net input
        )
        if dequantize:
            if float(np.abs(y).max()) >= 2.0**23:
                raise ValueError(
                    f"dequantize=True requires simulated counts below 2**23, where float32 "
                    f"can still hold the added jitter; the largest simulated count is "
                    f"{float(np.abs(y).max()):g}. Rescale the observations or use "
                    f"learn_amortized_ratio."
                )
            y = np.asarray(jnp.asarray(y) + jax.random.uniform(k_jitter, y.shape), dtype="float32")
        internal_keys = _adapter_field_keys(leaf_keys)
        sims = {k: named[lk] for k, lk in zip(internal_keys, leaf_keys)}
        sims[_OBSERVATION_KEY] = y

        obs_role = (
            "inference_variables"
            if theta_role == "inference_conditions"
            else "inference_conditions"
        )
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
    simulator: ConditionalDistribution,
    *,
    num_simulations: int = 10_000,
    epochs: int = 50,
    batch_size: int = 128,
    sim_backend: SimBackend = "jax",
    inference_network: InferenceNetwork | None = None,
    dequantize: bool = False,
    optimizer: str | KerasOptimizer = "adam",
    **fit_kwargs: Any,
) -> BayesFlowLikelihood:
    """Learn an amortized likelihood ``p(y | theta)`` (NLE) with BayesFlow.

    Trains a conditional coupling flow on offline ``(theta, y)`` simulations and
    returns a :class:`BayesFlowLikelihood`, the kernel ``p̂(y | theta)`` from the
    prior's components to datasets of observation rows, with the network's
    jax-traceable density. ``condition_on(learned * prior, {"observation": y})``
    then runs a registered gradient-based MCMC method (BlackJAX/TFP NUTS) for
    datasets of any size, since per-row scores sum under conditional
    independence. The network conditions on the raw constrained ``theta``.

    The training's seed is drawn from a workflow-owned random event, so
    ``workflow_run(seed=...)`` reproduces the trained network, and an unscoped
    call trains afresh. The seed fixes the offline simulation, the
    dequantization jitter, and keras's network initialization and training.
    The caller's global NumPy and Python random states are restored after
    training.

    Parameters
    ----------
    prior : Distribution
        Prior over the model parameters; a numeric distribution whose
        components name them, which may be nested (a factored joint of named
        distributions). Sampled (only) to draw training thetas;
        constrained and discrete-valued parameter fields are both fine here,
        since theta is a network *input* (whether the downstream sampler can
        handle the prior is the sampler's concern).
    simulator : ConditionalDistribution
        The kernel of one observation given the prior's fields, which samples;
        its given values are the prior's structured per-draw record (named-field
        access). Must be JAX-vmappable unless ``sim_backend="sequential"``.
    num_simulations : int
        Number of ``(theta, y)`` pairs simulated offline for training.
    epochs : int
        Number of keras training passes over the simulations.
    batch_size : int
        Number of simulations in each keras training batch.
    sim_backend : {"jax", "sequential"}
        ``"jax"`` (default) vmaps the simulator; ``"sequential"`` runs an eager
        per-draw loop for non-JAX simulators.
    inference_network : bayesflow.networks.InferenceNetwork or None
        Overrides the default ``CouplingFlow``. The density must be
        reverse-mode differentiable for gradient-based MCMC -- adaptive-ODE
        networks (``FlowMatching``, ``DiffusionModel``) are **not** (their
        ``log_prob`` integrates with a dynamic-bound ``while_loop``).
    dequantize : bool
        Set for **integer-valued observations** (counts and other *ordered*
        integer encodings; unordered categoricals would inherit a meaningless
        cell adjacency). Fitting a continuous flow to atoms is ill-posed -- the
        MLE collapses density onto the data points, which in practice shows up
        as overdispersed, seed-unstable posteriors as observations concentrate
        on few values. Uniform dequantization fixes this: training adds
        ``U[0,1)^d`` jitter to the simulated ``y`` (the simulator itself stays
        untouched and keeps emitting raw integers), making the target
        absolutely continuous, and the returned wrapper scores integer data at
        the unit-cell midpoint ``y + 1/2``, approximating the implied pmf
        ``P(y | theta) = integral over [y, y+1)^d of p(u | theta) du``. This is
        the fixed-``q`` special case of variational dequantization: the
        cell-integral identity is exact, and the training objective is its
        Jensen lower bound (Theis et al., 2016, arXiv:1511.01844; Ho et al.,
        2019 "Flow++", arXiv:1902.00275, section 3.1). Pass raw integers as
        data; do **not** pre-jitter or pre-shift. Counts must stay below
        ``2**23`` (~8.4e6): the pipeline is float32, whose spacing reaches 1.0
        there, silently rounding away the jitter and the midpoint shift
        (enforced on the simulated training observations). For
        mixed discrete/continuous rows or when a learned density is not
        needed, prefer :func:`learn_amortized_ratio`, whose classifier
        consumes discrete observations natively (cf. MNLE, Boelts et al.,
        2022, for the mixed-data approach in the torch ``sbi`` ecosystem).
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
        a custom ``inference_network``); with ``dequantize=True``, also if the
        simulated observations reach ``2**23``.
    TypeError
        If a count parameter is not an integer, ``simulator`` is not a kernel
        that samples, ``prior`` is not a numeric distribution, or
        ``fit_kwargs`` holds ``random_seed`` or ``seed``.
    ImportError
        If the ``[bayesflow]`` extra is not installed.
    """

    def _build(bf: Any, adapter: Any, data_dim: int) -> Any:
        if inference_network is None and data_dim < 2:
            raise ValueError(
                "the default network of learn_amortized_likelihood requires observations of "
                f"at least 2 values, but the simulator gives {data_dim}. Pass an "
                "inference_network, or use learn_amortized_ratio, which has no minimum."
            )
        net = inference_network or bf.networks.CouplingFlow()
        return bf.ContinuousApproximator(inference_network=net, adapter=adapter)

    approximator, data_dim = _train_offline(
        prior,
        simulator,
        caller="learn_amortized_likelihood",
        num_simulations=num_simulations,
        epochs=epochs,
        batch_size=batch_size,
        sim_backend=sim_backend,
        theta_role="inference_conditions",
        build_approximator=_build,
        optimizer=optimizer,
        fit_kwargs=fit_kwargs,
        dequantize=dequantize,
    )
    return BayesFlowLikelihood(
        approximator, prior, simulator, data_dim=data_dim, dequantized=dequantize
    )


def learn_amortized_ratio(
    prior: Distribution,
    simulator: ConditionalDistribution,
    *,
    num_simulations: int = 10_000,
    epochs: int = 50,
    batch_size: int = 128,
    sim_backend: SimBackend = "jax",
    inference_network: KerasLayer | None = None,
    optimizer: str | KerasOptimizer = "adam",
    **fit_kwargs: Any,
) -> BayesFlowRatio:
    """Learn an amortized likelihood-to-evidence ratio (NRE) with BayesFlow.

    Trains an NRE-C classifier (``RatioApproximator``; contrastive pairs are
    built internally by shuffling theta within each batch, so the training data
    are the same offline ``(theta, y)`` simulations as NLE) and returns a
    :class:`BayesFlowRatio`, the kernel from the prior's components to datasets
    of observation rows whose summed per-row log-ratios are its unnormalized
    density: the log-likelihood **up to a theta-independent constant** -- valid
    for ``condition_on(learned * prior, {"observation": y})``, invalid for
    absolute-likelihood uses (model comparison, LOO/WAIC); see the class
    docstring. The classifier handles discrete-valued observations and
    one-dimensional data natively.

    The training's seed is drawn from a workflow-owned random event, so
    ``workflow_run(seed=...)`` reproduces the trained classifier, and an
    unscoped call trains afresh. The seed fixes the offline simulation and
    keras's network initialization and training. The caller's global NumPy and
    Python random states are restored after training.

    Parameters
    ----------
    prior : Distribution
        Prior over the model parameters; a numeric distribution, possibly
        nested (as in :func:`learn_amortized_likelihood` -- constrained and
        discrete-valued parameter fields are fine, theta is a network input).
    simulator : ConditionalDistribution
        The kernel of one observation given the prior's fields, which samples;
        its given values are the prior's structured per-draw record.
    num_simulations : int
        Number of ``(theta, y)`` pairs simulated offline for training.
    epochs : int
        Number of keras training passes over the simulations.
    batch_size : int
        Number of simulations in each keras training batch.
    sim_backend : {"jax", "sequential"}
        ``"jax"`` (default) vmaps the simulator; ``"sequential"`` runs an eager
        per-draw loop for non-JAX simulators.
    inference_network : keras.Layer or None
        The classifier body (defaults to ``bayesflow.networks.MLP()``); the
        ``RatioApproximator`` adds its own scalar projection head.
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
        If a count parameter is not an integer, ``simulator`` is not a kernel
        that samples, ``prior`` is not a numeric distribution, or
        ``fit_kwargs`` holds ``random_seed`` or ``seed``.
    ImportError
        If the ``[bayesflow]`` extra is not installed.
    """

    def _build(bf: Any, adapter: Any, data_dim: int) -> Any:
        net = inference_network if inference_network is not None else bf.networks.MLP()
        return bf.approximators.RatioApproximator(inference_network=net, adapter=adapter)

    approximator, data_dim = _train_offline(
        prior,
        simulator,
        caller="learn_amortized_ratio",
        num_simulations=num_simulations,
        epochs=epochs,
        batch_size=batch_size,
        sim_backend=sim_backend,
        theta_role="inference_variables",
        build_approximator=_build,
        optimizer=optimizer,
        fit_kwargs=fit_kwargs,
    )
    return BayesFlowRatio(approximator, prior, simulator, data_dim=data_dim)
