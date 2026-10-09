"""Simulation-based calibration (SBC) and interval coverage for method validation.

Where :mod:`probpipe.validation._comparison` scores an approximation against a
*reference*, these check a posterior for self-consistency with the model that
generated the data, whether the posterior is a method's fit or a kernel such as
an amortized posterior:

- :func:`simulation_based_calibration` (Talts et al. 2018) draws ``θ★`` and
  ``y`` from the joint, forms the posterior at ``y``, and ranks each ``θ★``
  component among the posterior's draws. The ranks are uniform on
  ``{0, …, L}`` when the posterior is calibrated, so non-uniform ranks expose a
  biased or mis-tuned method.
- :func:`interval_coverage` is the companion frequentist check of one posterior:
  whether a central credible interval contains the truth.

Calibration runs a Python loop over its replications, which works with every
backend, such as blackjax, Stan, PyMC, or a trained network. The loop itself is
not jit-compatible. The model is a joint that samples, such as a GLM likelihood
times its prior.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .._messages import unknown_names
from ..core._record_spec import RecordSpec
from ..core._specs import OutputSpec
from ..custom_types import Array, ArrayLike
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution, _array_leaves
from ..distributions._empirical import EmpiricalDistribution, _coordinates
from ..distributions._factored import _raw_record
from ..functions import _context
from ..operations._condition import condition_on
from ..operations._sample import sample
from ._workflow_rng import _validate_positive_int

__all__ = ["SBCResult", "interval_coverage", "simulation_based_calibration"]


# -- helpers ----------------------------------------------------------------


def _flatten(value: Any, event_spec: OutputSpec, batch_ndim: int = 0) -> Array:
    """Flatten a parameter value, or a batch of them, to the flat layout of the posterior's draws.

    The posterior's event declaration *event_spec* gives the layout. An array
    value is raveled, and a record value, a ``Record`` or the nested mapping of
    its leaves, has its leaves raveled and concatenated in the declaration's
    leaf order. The *batch_ndim* leading axes of a batch are kept, so one value
    gives a ``(p,)`` array and a batch of ``n`` draws an ``(n, p)`` array.
    """
    spec = event_spec.spec
    if isinstance(spec, RecordSpec):
        raw = _raw_record(value)
        leaves = []
        for path in spec:
            leaf = raw
            for segment in path.split("/"):
                leaf = leaf[segment]
            leaves.append(jnp.asarray(leaf))
    else:
        leaves = [jnp.asarray(value)]
    lead = leaves[0].shape[:batch_ndim]
    return jnp.concatenate([jnp.reshape(leaf, (*lead, -1)) for leaf in leaves], axis=-1)


def _ranks(draws: Array, point: Array) -> Array:
    """SBC rank of each component of *point*: ``#{draws < point}``, in ``{0, …, n}``.

    Uses strict ``<`` (ties, measure-zero for a continuous posterior, count as
    not-below). ``draws`` is ``(n, d)``; ``point`` is ``(d,)``; returns ``(d,)``.
    """
    return jnp.sum(draws < point[None, :], axis=0)


def _component_names(posterior: Any) -> tuple[str, ...]:
    """Per-flattened-component parameter names matching the posterior's flat coordinates.

    A scalar leaf keeps its path, and a leaf with ``k > 1`` flattened
    components becomes ``path[0] … path[k-1]`` (row-major), in the posterior's
    leaf order. A whole-term posterior's one leaf is its component.
    """
    names: list[str] = []
    for path, spec in _array_leaves(posterior.event_spec).items():
        size = int(np.prod(spec.shape, dtype=int))
        names.extend([path] if size == 1 else [f"{path}[{i}]" for i in range(size)])
    return tuple(names)


def _kolmogorov_sf(d: float, n: int) -> float:
    """Asymptotic survival function of the one-sample KS statistic.

    ``P(D_n ≥ d)`` under the null, via the Kolmogorov limit
    ``Q_KS(λ) = 2 Σ_k (−1)^{k−1} e^{−2k²λ²}`` with Stephens' (1970) small-sample
    correction ``λ = (√n + 0.12 + 0.11/√n) d``. Dependency-free (no SciPy);
    accurate for the sample sizes SBC uses (``n ≳ 30``).
    """
    if d <= 0.0:
        return 1.0
    en = np.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * d
    k = np.arange(1, 101)  # series decays as e^{-2k²λ²}; 100 terms is far more than enough
    p = 2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * (k * lam) ** 2))
    return float(min(max(p, 0.0), 1.0))


def _ks_uniform(ranks: np.ndarray, num_draws: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-parameter KS distance + p-value of the ranks vs ``Uniform{0..L}``.

    ``ranks`` is ``(num_simulations, num_params)`` in ``{0, …, num_draws}``; the
    ranks are mapped to ``(0, 1)`` via the midpoint ``(r + 0.5)/(L + 1)`` and
    compared to ``Uniform[0, 1]``.
    """
    u = (ranks + 0.5) / (num_draws + 1)
    v = np.sort(u, axis=0)
    s = v.shape[0]
    i_over_s = (np.arange(1, s + 1) / s)[:, None]
    im1_over_s = (np.arange(0, s) / s)[:, None]
    d = np.maximum(np.max(i_over_s - v, axis=0), np.max(v - im1_over_s, axis=0))
    pvals = np.array([_kolmogorov_sf(float(d[j]), s) for j in range(d.shape[0])])
    return d, pvals


def _credible_levels(levels: Iterable[float]) -> tuple[float, ...]:
    """*levels* as floats, each a credible level in ``(0, 1)``.

    Parameters
    ----------
    levels : iterable of float
        The credible levels the caller gave, such as ``(0.5, 0.9)``.

    Returns
    -------
    tuple of float
        The levels, in the order given.

    Raises
    ------
    TypeError
        If *levels* is not an iterable of numbers.
    ValueError
        If a level is not in ``(0, 1)``.
    """
    if isinstance(levels, str) or not isinstance(levels, Iterable):
        raise TypeError(f"levels must be a sequence of credible levels; got {levels!r}")
    checked = []
    for level in levels:
        if isinstance(level, bool):
            raise TypeError(f"levels must be numbers in (0, 1); got {level!r}")
        try:
            value = float(level)
        except (TypeError, ValueError):
            raise TypeError(f"levels must be numbers in (0, 1); got {level!r}") from None
        if not 0.0 < value < 1.0:
            raise ValueError(f"levels must be numbers in (0, 1); got {level!r}")
        checked.append(value)
    return tuple(checked)


def _slot_binding(kernel: ConditionalDistribution, observed: tuple[str, ...]) -> dict[str, str]:
    """The observed field that each given slot of *kernel* takes, keyed by the slot.

    The slots take the observed fields of their names when every observed field
    names a slot and every required slot is observed. Otherwise a kernel with one
    given slot takes the one observed field, whatever its name.

    Parameters
    ----------
    kernel : ConditionalDistribution
        The posterior kernel, whose given slots take the observed fields.
    observed : tuple of str
        The names of the observed fields of the model's draw.

    Returns
    -------
    dict of str to str
        The observed field's name by slot name, which is the identity map when
        the slots take the fields of their names.

    Raises
    ------
    ValueError
        If neither rule binds the slots.
    """
    slots = tuple(kernel.given_spec)
    if set(observed) <= set(slots) and set(kernel.given_spec.required) <= set(observed):
        return {name: name for name in observed}
    if len(slots) == 1 and len(observed) == 1:
        return {slots[0]: observed[0]}
    raise ValueError(
        f"the given slots {list(slots)} of posterior {kernel.label!r} do not match the "
        f"observed fields {list(observed)}. Name the slots after the observed fields, or "
        f"use a posterior with one given slot for one observed field."
    )


def _check_parameters(event_spec: OutputSpec, parameters: tuple[str, ...], label: str) -> None:
    """Raise unless a draw of the posterior holds the parameters.

    A draw that exposes a record holds the parameters as its components, and a
    whole-term draw is the value of the one parameter.

    Parameters
    ----------
    event_spec : OutputSpec
        The posterior's event declaration.
    parameters : tuple of str
        The names of the model's unobserved fields.
    label : str
        The posterior's label, which the error names.

    Raises
    ------
    ValueError
        If the components of an exposed record are not the parameters, or a
        whole-term draw meets several parameters.
    """
    if event_spec.exposes_record:
        if set(event_spec.components) != set(parameters):
            raise ValueError(
                f"posterior {label!r} draws the fields {list(event_spec.components)}, but "
                f"they must be the model's parameters {list(parameters)}"
            )
    elif len(parameters) != 1:
        raise ValueError(
            f"posterior {label!r} draws a single unnamed value, but the model has the "
            f"parameters {list(parameters)}. It must draw a record with one field per parameter."
        )


# -- simulation-based calibration -------------------------------------------


@dataclass(frozen=True)
class SBCResult:
    """Result of :func:`simulation_based_calibration`.

    Attributes
    ----------
    ranks : np.ndarray
        Integer ranks of ``θ★`` among the posterior draws, shape
        ``(num_simulations, num_params)``, each in ``{0, …, num_posterior_draws}``.
        The rank counts draws strictly below ``θ★``; this assumes continuous
        draws (ties are measure-zero for a continuous posterior, but would bias
        ranks low for a discrete-valued parameter).
    num_posterior_draws : int
        ``L``: the number of posterior draws of each replication, which is the
        largest rank.
    param_names : tuple[str, ...] or None
        The name of each column of ``ranks`` and ``ks_*``: ``field`` for a
        scalar leaf, and ``field[i]`` for each entry of a larger leaf, in the
        posterior's leaf order.
    ks_statistic : np.ndarray
        Per-parameter KS distance of the normalized ranks from ``Uniform[0, 1]``.
    ks_pvalue : np.ndarray
        Per-parameter KS p-value against ``Uniform[0, 1]``; small ⇒ ranks are
        non-uniform ⇒ miscalibrated. This is a diagnostic — inspect it (and the
        rank histogram) and apply your own threshold, with a multiple-comparison
        correction across parameters, rather than reading off a pass/fail verdict.
    """

    ranks: np.ndarray
    num_posterior_draws: int
    param_names: tuple[str, ...] | None
    ks_statistic: np.ndarray
    ks_pvalue: np.ndarray

    def rank_histogram(self, num_bins: int = 20) -> np.ndarray:
        """Rank histogram per parameter, shape ``(num_params, num_bins)``.

        Bins the normalized ranks into ``num_bins`` equal-width bins over
        ``[0, 1]``; a calibrated method gives approximately flat histograms.
        """
        u = (self.ranks + 0.5) / (self.num_posterior_draws + 1)
        edges = np.linspace(0.0, 1.0, num_bins + 1)
        return np.stack([np.histogram(u[:, j], bins=edges)[0] for j in range(u.shape[1])])

    def coverage(self, levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95)) -> dict[float, np.ndarray]:
        """The share of replications whose ``θ★`` lies in the central interval of each level.

        ``θ★`` lies in the central ``level`` interval of its posterior when its
        normalized rank ``u = (r + 0.5) / (L + 1)`` lies in
        ``[(1 − level) / 2, (1 + level) / 2]``, so the shares are computed from
        the ranks alone. Under a calibrated posterior the expected share is the
        level, within ``1 / (L + 1)``. The shares match the mean of
        :func:`interval_coverage` over the replications' posterior draws, except
        at a rank next to an end of the interval, where
        :func:`interval_coverage` interpolates between two draws.

        Parameters
        ----------
        levels : sequence of float
            The credible levels, each in ``(0, 1)``.

        Returns
        -------
        dict of float to np.ndarray
            For each level, the share of replications that cover ``θ★``, one per
            parameter, of shape ``(num_params,)``.

        Raises
        ------
        TypeError
            If *levels* is not a sequence of numbers.
        ValueError
            If a level is not in ``(0, 1)``.
        """
        u = (self.ranks + 0.5) / (self.num_posterior_draws + 1)
        shares: dict[float, np.ndarray] = {}
        for level in _credible_levels(levels):
            lo, hi = (1.0 - level) / 2.0, (1.0 + level) / 2.0
            shares[level] = np.mean((u >= lo) & (u <= hi), axis=0)
        return shares


def simulation_based_calibration(
    model: Distribution,
    *,
    observed: str | Sequence[str],
    num_simulations: int,
    num_posterior_draws: int,
    posterior: ConditionalDistribution | None = None,
    method: str | None = None,
    method_options: Mapping[str, Any] | None = None,
) -> SBCResult:
    """Simulation-based calibration of a posterior (Talts et al. 2018).

    Each of *num_simulations* replications draws the parameters ``θ★`` and the
    *observed* fields ``y`` from the joint *model*, forms the posterior at
    ``y``, draws from it, and ranks each flattened ``θ★`` component among the
    draws. Under a calibrated posterior each rank is uniform on
    ``{0, …, num_posterior_draws}``. The result summarizes each parameter's
    ranks by their KS distance from uniform and its p-value.

    The posterior at ``y`` is formed in one of two ways:

    1. with *posterior*, it is the kernel's law at ``y``, which
       ``condition_on(posterior, {slot: y})`` returns without a fit, as for an
       amortized posterior or a closed-form posterior;
    2. otherwise, it is the fit of *model* at ``y`` that
       ``condition_on.with_options(method=method, method_options=method_options)(model, y)``
       returns.

    *posterior*'s given slots take the observed fields of their names, and a
    kernel with one given slot, such as an amortized posterior's
    ``observation``, takes the one observed field. The draws are
    ``sample(posterior_at_y, sample_shape=(num_posterior_draws,))`` for every
    kind of posterior: the atoms of a weighted empirical law, such as SMC-ABC's
    particles, are resampled by weight, a network posterior is evaluated, and an
    MCMC posterior's atoms are drawn at random.

    The calibration takes its randomness from the enclosing workflow scope,
    whose workflow-owned random events it claims in program order: one
    ``sample(model, sample_shape=(num_simulations,))`` draws the ``θ★`` and
    ``y`` of every replication, and then each replication forms its posterior
    and draws from it. A call inside ``workflow_run(seed=...)`` therefore
    reproduces its ranks, and an unscoped call draws afresh.

    Parameters
    ----------
    model : Distribution
        A joint that samples, over the parameters and the *observed* fields,
        such as ``likelihood * prior``.
    observed : str or sequence of str
        The fields of a draw that the posterior conditions on; the others are
        the parameters.
    num_simulations : int
        The number of replications.
    num_posterior_draws : int
        The number of draws of each replication's posterior, which is the
        largest rank. The ranks assume nearly independent draws, so for an
        empirical posterior, such as MCMC chains or weighted particles, it
        should not exceed the effective sample size of the atoms. For an MCMC
        method, set the chain length in *method_options* to make it so.
    posterior : ConditionalDistribution, optional
        A posterior kernel from the observed fields to the parameters, such as
        the amortized posterior that ``learn_amortized_posterior`` returns.
        Without it, each replication fits *model* at its observed values.
    method : str, optional
        The inference method of each fit, by name; ``None`` selects the method
        as :func:`condition_on` does.
    method_options : Mapping, optional
        The options of each fit's method, such as ``{"num_warmup": 500,
        "num_results": 2000}``. Each fit's seed is a workflow-owned random event
        of the enclosing scope.

    Returns
    -------
    SBCResult
        The ranks, of shape ``(num_simulations, num_params)``, with the KS
        statistic and p-value of each parameter.

    Raises
    ------
    TypeError
        If *model* does not sample; if *posterior* is not a
        ``ConditionalDistribution``; if a count is not an integer; or if
        *method* is not a non-empty string or *method_options* is not a mapping
        of option names. A method's ``TypeError`` for an option it does not
        read propagates.
    ValueError
        If a count is not positive; if an *observed* name is not a field of a
        draw, or no parameter is left; if *method* or *method_options* is given
        with *posterior*; if *posterior*'s given slots do not take the observed
        fields; or if the posterior's draw does not hold the parameters: a draw
        that exposes a record names other fields, or a whole-term draw meets
        several parameters.
    ResolutionError
        If no route of :func:`condition_on` forms the posterior at ``y``, or
        that posterior does not sample.
    ReplayCompatibilityError
        If a ``replay_run`` scope is active, since a replay accepts one
        top-level ``Function`` call and the calibration makes several.

    Notes
    -----
    The draws of ``θ★`` and ``y`` are one event, which the call claims first,
    so they do not depend on the posterior or the method. Two calls with the
    same *model* and *num_simulations*, at the same position of scopes with the
    same seed, therefore rank their posteriors' draws against the same ``θ★``
    and ``y``.

    Examples
    --------
    The exact posterior kernel of a conjugate normal model covers ``θ★`` at
    about the nominal rates:

    >>> import jax.numpy as jnp
    >>> from probpipe import Normal, NumericArraySpec, conditional_distribution, workflow_run
    >>> prior = Normal("mu", 0.0, 2.0)
    >>> likelihood = conditional_distribution(
    ...     lambda mu: Normal("y", mu * jnp.ones(5), 1.0),
    ...     label="y_given_mu",
    ...     given_spec=prior.event_spec.components,
    ... )
    >>> precision = 1 / 4 + 5
    >>> exact = conditional_distribution(
    ...     lambda y: Normal("mu", jnp.sum(y) / precision, precision**-0.5),
    ...     label="posterior",
    ...     given_spec={"y": NumericArraySpec((5,))},
    ... )
    >>> with workflow_run(seed=0):
    ...     result = simulation_based_calibration(
    ...         likelihood * prior,
    ...         observed="y",
    ...         posterior=exact,
    ...         num_simulations=100,
    ...         num_posterior_draws=99,
    ...     )
    >>> result.ranks.shape
    (100, 1)
    >>> result.coverage((0.5, 0.9))
    {0.5: array([0.53]), 0.9: array([0.87])}
    """
    _context._assert_workflow_admission()
    num_simulations = _validate_positive_int("num_simulations", num_simulations)
    num_posterior_draws = _validate_positive_int(
        "num_posterior_draws",
        num_posterior_draws,
    )
    if posterior is not None:
        if method is not None or method_options is not None:
            raise ValueError(
                "simulation_based_calibration takes either posterior or method and "
                "method_options, not both"
            )
        if not isinstance(posterior, ConditionalDistribution):
            raise TypeError(
                "posterior must be a ConditionalDistribution from the observed fields to "
                f"the parameters; got {type(posterior).__name__}"
            )
    # Configured before any draw, so a malformed method or option fails first.
    fit = condition_on.with_options(method=method, method_options=method_options)
    if not callable(getattr(model, "_sample", None)):
        raise TypeError(
            f"simulation_based_calibration: model must be a distribution that can be sampled; "
            f"got {type(model).__name__}"
        )
    observed = (observed,) if isinstance(observed, str) else tuple(observed)
    components = tuple(model.event_spec.components)
    unknown = [name for name in observed if name not in components]
    if unknown:
        raise ValueError(unknown_names("observed field", unknown, components, "fields"))
    parameters = tuple(name for name in components if name not in observed)
    if not parameters:
        raise ValueError(
            f"observed={list(observed)} includes every field of {model.label!r}, so no "
            f"parameters are left to calibrate"
        )
    binding: dict[str, str] = {}
    if posterior is not None:
        binding = _slot_binding(posterior, observed)
        _check_parameters(posterior.event_spec, parameters, posterior.label)

    simulations = _raw_record(sample.with_options(raw=True)(model, sample_shape=(num_simulations,)))
    rank_rows: list[np.ndarray] = []
    component_names: tuple[str, ...] | None = None
    for index in range(num_simulations):
        draw = jax.tree.map(lambda column, i=index: column[i], simulations)
        theta_star = {name: draw[name] for name in parameters}
        if posterior is None:
            law = fit(model, {name: draw[name] for name in observed})
        else:
            law = condition_on(posterior, {slot: draw[field] for slot, field in binding.items()})
        draws = sample.with_options(raw=True)(law, sample_shape=(num_posterior_draws,))
        if component_names is None:
            _check_parameters(law.event_spec, parameters, law.label)
            component_names = _component_names(law)
        # A whole-term posterior's draw is its one parameter's value.
        point = theta_star if law.event_spec.exposes_record else next(iter(theta_star.values()))
        flat_draws = _flatten(draws, law.event_spec, batch_ndim=1)  # (L, p)
        rank_rows.append(np.asarray(_ranks(flat_draws, _flatten(point, law.event_spec))))

    ranks = np.stack(rank_rows).astype(int)  # (num_simulations, p)
    ks_stat, ks_pvalue = _ks_uniform(ranks, num_posterior_draws)
    return SBCResult(
        ranks=ranks,
        num_posterior_draws=num_posterior_draws,
        param_names=component_names,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pvalue,
    )


# -- interval coverage ------------------------------------------------------


def interval_coverage(
    draws_or_dist: Any,
    truth: ArrayLike,
    *,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> dict[float, Array]:
    """Central-credible-interval coverage of *truth*, per parameter.

    For each ``level``, checks whether each component of *truth* lies in the
    central ``level`` interval ``[q_{(1−level)/2}, q_{(1+level)/2}]`` of the
    per-parameter posterior. Returns ``{level: covered}`` with ``covered`` a
    boolean array over parameters. Averaging the indicators over many
    ``(posterior, truth)`` pairs gives the frequentist coverage, which matches the
    nominal level for a calibrated method.

    *draws_or_dist* is an ``(n, d)`` (or 1-D ``(n,)``) array of draws or an
    empirical law, read as the flat coordinates of its atoms (treated as
    equally weighted, as MCMC draws are); *truth* is the matching ``(d,)``.
    """
    draws = jnp.asarray(
        _coordinates(draws_or_dist)
        if isinstance(draws_or_dist, EmpiricalDistribution)
        else draws_or_dist
    )
    if draws.ndim == 1:
        draws = draws[:, None]
    truth = jnp.atleast_1d(jnp.asarray(truth))
    if truth.shape[0] != draws.shape[1]:
        raise ValueError(
            f"truth has {truth.shape[0]} values, but the draws have dimension {draws.shape[1]}; "
            f"they must match"
        )
    out: dict[float, Array] = {}
    for level in levels:
        lo, hi = (1.0 - level) / 2.0, (1.0 + level) / 2.0
        q = jnp.quantile(draws, jnp.array([lo, hi]), axis=0)  # (2, d)
        out[float(level)] = (truth >= q[0]) & (truth <= q[1])
    return out
