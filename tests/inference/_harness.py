"""The cross-method validation harness: one binding validates an inference method on every case.

``validate_method(name)`` returns a test function parametrized over the
canonical cases (:mod:`tests.inference.canonical`), and a backend's own test
file binds it in one line::

    test_blackjax_nuts_canonical = validate_method("blackjax_nuts")

For each case, the test selects the model in the representation the method
consumes, asks ``condition_on.check`` whether the method applies, and skips
with the method's own reason when it does not. Otherwise it conditions the
model on the case's data with ``method=name`` and compares the posterior with
the case's reference, leaf by leaf: the mean, the variance, and the endpoints
of the central 90% interval of every coordinate, each read through the
operation model's ``mean``, ``variance``, and ``quantile``.

**Tolerances.** Every comparison is ``|estimate - reference| <= Z * MCSE`` with
``Z = 4``, where the Monte Carlo standard error is estimated from the
posterior's chains with their effective sample size: ArviZ's ``mcse`` of the
mean and of the quantile, and for the variance the MCSE of the mean of the
squared deviations, with their own effective sample size. A method whose draws
target the posterior fails one comparison with probability about 6e-5, and a
case's few dozen comparisons together with probability below 0.003, so a
failure is a finding rather than noise. The keys are fixed, so every outcome is
reproducible. An MCSE is valid only for chains that have mixed, so a method
that runs several chains must also reach a rank-normalized R-hat below 1.05.

**Kinds.** A *consistent* method's draws converge to the posterior as its
budget grows, as an MCMC chain's do, and it meets every comparison above. A
*biased* method stands in for the posterior even in the limit, as a mean-field
variational fit does, so the harness holds it to the location alone: each
coordinate's mean lies within a quarter of a posterior standard deviation of
the reference, beyond the ``Z * MCSE`` band.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import NumericArrayBatch, workflow_run
from tests._ops import (
    EmpiricalDistribution,
    condition_on,
    inference_method_registry,
    mean,
    quantile,
    sample,
    variance,
)
from tests.inference import canonical
from tests.inference.canonical import INTERVAL_LEVELS, ModelTestCase, PosteriorReference

__all__ = [
    "PROFILES",
    "MethodProfile",
    "Z",
    "assert_matches",
    "calibration_ranks",
    "chains_at",
    "uniformity_pvalues",
    "validate_method",
]

#: The half-width of every comparison band, in Monte Carlo standard errors.
Z = 4.0

#: The largest rank-normalized R-hat at which a several-chain result counts as mixed.
RHAT_LIMIT = 1.05

#: The location allowance of a biased method, in posterior standard deviations.
BIAS_ALLOWANCE = 0.25


@dataclass(frozen=True)
class MethodProfile:
    """How the harness runs one inference method.

    Attributes
    ----------
    representation : str
        The model representation the method consumes: ``"probpipe"`` for the
        joint composed with ``*``, ``"pymc"``, or ``"stan"``.
    consistent : bool
        Whether the method's draws converge to the posterior, which selects the
        contract it is held to.
    method_options : Mapping[str, Any]
        The ``method_options`` control ``condition_on.with_options`` receives,
        budgets and seed included.
    """

    representation: str
    consistent: bool
    method_options: Mapping[str, Any]


#: The budget of a gradient-based sampler: two chains of 2000 draws after 1000
#: warmup steps, which gives every case well over a thousand effective draws.
_GRADIENT = {"num_results": 2000, "num_warmup": 1000, "num_chains": 2, "random_seed": 0}

#: The budget of the random walk, whose draws are more autocorrelated.
_RANDOM_WALK = {"num_results": 4000, "num_warmup": 2000, "num_chains": 2, "random_seed": 0}

#: The budget of a stochastic-gradient sampler, which needs a minibatch size.
_STOCHASTIC = {"num_results": 2000, "num_warmup": 1000, "batch_size": 10, "random_seed": 0}

#: Each registered method's profile. The tolerances do not depend on the
#: budgets, since the MCSE scales with them.
PROFILES: dict[str, MethodProfile] = {
    "blackjax_nuts": MethodProfile("probpipe", True, _GRADIENT),
    "blackjax_hmc": MethodProfile("probpipe", True, _GRADIENT),
    "blackjax_rwmh": MethodProfile("probpipe", True, _RANDOM_WALK),
    "blackjax_elliptical_slice": MethodProfile("probpipe", True, _GRADIENT),
    "blackjax_sgld": MethodProfile("probpipe", False, _STOCHASTIC),
    "blackjax_sghmc": MethodProfile("probpipe", False, _STOCHASTIC),
    "tfp_nuts": MethodProfile("probpipe", True, _GRADIENT),
    "tfp_hmc": MethodProfile("probpipe", True, _GRADIENT),
    "nutpie_nuts": MethodProfile("pymc", True, _GRADIENT),
    "pymc_nuts": MethodProfile("pymc", True, {**_GRADIENT, "cores": 1}),
    "pymc_advi": MethodProfile(
        "pymc", False, {"num_results": 2000, "num_iterations": 20000, "random_seed": 0}
    ),
    "cmdstan_nuts": MethodProfile("stan", True, _GRADIENT),
    "pyabc_smcabc": MethodProfile("probpipe", False, {"n_particles": 200, "random_seed": 0}),
}


#: The (method, representation, case) runs that fail because of a library bug,
#: with the bug and the exception the failure raises. Each is collected as a
#: pending test, so the suite stays green and the ledger lists it.
KNOWN_FAILURES: dict[tuple[str, str, str], tuple[str, type[BaseException]]] = {}

_ABC_BUDGET = (
    "the harness's SMC-ABC budget, 200 particles over four populations, leaves a "
    "coordinate's mean further from the reference than a quarter of its posterior sd"
)
_ABC_OUTSIDE_THE_SUPPORT = (
    "bug: pyABC perturbs a bounded parameter in its constrained coordinate, and "
    "particles outside the unit interval reach the posterior"
)
KNOWN_FAILURES[("pyabc_smcabc", "probpipe", "gaussian_linear")] = (_ABC_BUDGET, AssertionError)
KNOWN_FAILURES[("pyabc_smcabc", "probpipe", "eight_schools")] = (_ABC_BUDGET, AssertionError)
KNOWN_FAILURES[("pyabc_smcabc", "probpipe", "beta_bernoulli")] = (
    _ABC_OUTSIDE_THE_SUPPORT,
    ValueError,
)


# ---------------------------------------------------------------------------
# The posterior's draws and summaries
# ---------------------------------------------------------------------------


def _at(raw: Any, path: str) -> Any:
    """The node of the nested mapping *raw* at the path *path*, which may be empty.

    Raises
    ------
    AssertionError
        If *raw* has no node at *path*, since a result keeps its target's paths.
    """
    node = raw
    for segment in filter(None, path.split("/")):
        if not isinstance(node, Mapping) or segment not in node:
            raise AssertionError(f"the result has no value at the event path {path!r}: {raw!r}")
        node = node[segment]
    return node


def _node(raw: Any, law: Any, path: str) -> Any:
    """The node at the event path *path* of *raw*, a raw value shaped like a draw of *law*.

    An exposed record's paths index its nested mapping, and a whole term's
    paths start with its component, which stands for the whole raw value.

    Raises
    ------
    AssertionError
        If *raw* has no node at *path*.
    """
    from probpipe.distributions._factored import _raw_record

    declaration = law.event_spec
    raw = _raw_record(raw)
    if declaration.exposes_record:
        return _at(raw, path)
    (component,) = declaration.components
    head, _, rest = path.partition("/")
    if head != component:
        raise AssertionError(f"{path!r} is not an event path of the whole term {component!r}")
    return _at(raw, rest)


def _as_leaf(value: Any, shape: tuple[int, ...], path: str, what: str) -> np.ndarray:
    """*value* as a float64 array of the reference leaf's *shape*, which it must fill.

    The values are compared at the reference's shape; whether the result keeps
    the target's declared shape is a contract of its own, checked elsewhere.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.size != math.prod(shape):
        raise AssertionError(
            f"the {what} at {path!r} has {array.size} values, and the reference has "
            f"{math.prod(shape)}"
        )
    return array.reshape(shape)


#: The number of independent draws the harness takes from a result that keeps no chains.
INDEPENDENT_DRAWS = 4000


def independent_draws(posterior: Any) -> Any:
    """*posterior* when it keeps its draws, and otherwise the empirical law of independent draws of it.

    An MCMC result keeps its chains, and an empirical law its atoms. Any other
    law gives :data:`INDEPENDENT_DRAWS` draws under a seeded workflow, so that
    every summary the harness compares, and its MCSE, is read from one set of
    draws, whatever routes the law's own moments and quantiles take.
    """
    if getattr(posterior, "num_chains", None) is not None or isinstance(
        posterior, EmpiricalDistribution
    ):
        return posterior
    with workflow_run(seed=0):
        draws = sample(posterior, sample_shape=(INDEPENDENT_DRAWS,))
    return EmpiricalDistribution(posterior.label, draws, event_spec=posterior.event_spec)


def chains_at(posterior: Any, path: str, shape: tuple[int, ...]) -> np.ndarray:
    """The draws of the leaf at *path*, as ``(chains, draws, *shape)``.

    An MCMC result gives one row per chain, read from its draws, whose layout
    is the target's. An empirical law gives its atoms, one row per entry of its
    outer level when its atoms have several levels and one row otherwise. Any
    other law is drawn from first, as :func:`independent_draws` does.
    """
    from probpipe.distributions._factored import _raw_record

    num_chains = getattr(posterior, "num_chains", None)
    if num_chains is not None:
        draws = posterior.draws()
        column = _at(_raw_record(draws), path) if hasattr(draws, "element_spec") else draws
        values = np.asarray(column, dtype=np.float64)
        return values.reshape(num_chains, values.shape[0] // num_chains, *shape)
    posterior = independent_draws(posterior)
    atoms = posterior.atoms
    raw = atoms.values if isinstance(atoms, NumericArrayBatch) else _raw_record(atoms)
    values = np.asarray(_node(raw, posterior, path), dtype=np.float64)
    chains = atoms.batch_shape[0] if len(atoms.axis_groups) > 1 else 1
    return values.reshape(chains, -1, *shape)


def _coordinates(chains: np.ndarray) -> np.ndarray:
    """*chains* ``(C, D, *shape)`` as ``(d, C, D)``, one row per coordinate."""
    C, D = chains.shape[:2]
    return np.moveaxis(chains.reshape(C, D, -1), -1, 0)


def _mcse(chains: np.ndarray, method: str, prob: float | None = None) -> np.ndarray:
    """ArviZ's MCSE of each coordinate's mean, or of its quantile at *prob*."""
    from arviz_stats.base import array_stats

    keywords = {"prob": prob} if prob is not None else {}
    return np.asarray(
        array_stats.mcse(
            _coordinates(chains), method=method, chain_axis=-2, draw_axis=-1, **keywords
        )
    )


def _mcse_variance(chains: np.ndarray) -> np.ndarray:
    """The MCSE of each coordinate's variance estimate, the mean of its squared deviations.

    The squared deviations form a chain of their own, so their MCSE is that of
    a mean, with the effective sample size of the squares. ArviZ's MCSE of the
    standard deviation takes the effective sample size of the draws instead,
    which overstates the precision of a second moment under a sampler whose
    draws are antithetic, as NUTS's often are.
    """
    from arviz_stats.base import array_stats

    coordinates = _coordinates(chains)
    pooled_mean = coordinates.mean(axis=(-2, -1), keepdims=True)
    squares = (coordinates - pooled_mean) ** 2
    return np.asarray(array_stats.mcse(squares, method="mean", chain_axis=-2, draw_axis=-1))


def _rhat(chains: np.ndarray) -> np.ndarray:
    from arviz_stats.base import array_stats

    return np.asarray(array_stats.rhat(_coordinates(chains), chain_axis=-2, draw_axis=-1))


@dataclass
class _Comparisons:
    """The failed comparisons of one result, collected so a failure lists them all."""

    failures: list[str] = field(default_factory=list)

    def within(
        self, what: str, path: str, estimate: np.ndarray, reference: np.ndarray, band: np.ndarray
    ) -> None:
        estimate, reference = np.ravel(estimate), np.ravel(reference)
        band = np.broadcast_to(np.ravel(band), estimate.shape)
        for index, (e, r, b) in enumerate(zip(estimate, reference, band)):
            if not abs(e - r) <= b:
                self.failures.append(
                    f"{what} of {path}[{index}]: {e:.5g} vs reference {r:.5g}, band {b:.3g}"
                )

    def check(self, label: str) -> None:
        if self.failures:
            raise AssertionError(f"{label}:\n  " + "\n  ".join(self.failures))


def assert_matches(
    posterior: Any, reference: PosteriorReference, *, consistent: bool = True, label: str = ""
) -> None:
    """Assert that *posterior* matches *reference* under the contract of its kind.

    A consistent result meets the four-MCSE band on every coordinate's mean,
    variance, and interval endpoints, and mixes to an R-hat below 1.05 when it
    has several chains. A biased result meets the band widened by a quarter of
    a posterior standard deviation, on the means alone.

    Raises
    ------
    AssertionError
        Listing every comparison that fails.
    """
    posterior = independent_draws(posterior)
    means = mean.with_options(raw=True)(posterior)
    variances = variance.with_options(raw=True)(posterior)
    levels = jnp.asarray(INTERVAL_LEVELS)
    quantiles = quantile.with_options(raw=True)(posterior, levels)
    comparisons = _Comparisons()
    for path, leaf in reference.leaves.items():
        shape = leaf.mean.shape
        chains = chains_at(posterior, path, shape)
        estimate = _as_leaf(_node(means, posterior, path), shape, path, "mean")
        mcse_mean = _mcse(chains, "mean")
        sd = np.ravel(np.sqrt(leaf.variance))
        if not consistent:
            band = Z * mcse_mean + BIAS_ALLOWANCE * sd
            comparisons.within("mean", path, estimate, leaf.mean, band)
            continue
        comparisons.within("mean", path, estimate, leaf.mean, Z * mcse_mean)
        estimate_var = _as_leaf(_node(variances, posterior, path), shape, path, "variance")
        mcse_var = _mcse_variance(chains)
        comparisons.within("variance", path, estimate_var, leaf.variance, Z * mcse_var)
        node = np.asarray(_node(quantiles, posterior, path), dtype=np.float64)
        node = _as_leaf(node, (len(INTERVAL_LEVELS), *shape), path, "quantiles")
        for index, level in enumerate(INTERVAL_LEVELS):
            comparisons.within(
                f"quantile {level}",
                path,
                node[index],
                leaf.quantiles[level],
                Z * _mcse(chains, "quantile", prob=level),
            )
        if chains.shape[0] > 1:
            rhat = _rhat(chains)
            for index, value in enumerate(rhat):
                if not value < RHAT_LIMIT:
                    comparisons.failures.append(f"R-hat of {path}[{index}]: {value:.4f}")
    comparisons.check(label or "the posterior does not match the reference")


# ---------------------------------------------------------------------------
# The representations
# ---------------------------------------------------------------------------


def _model_and_data(
    case: ModelTestCase, representation: str, request: pytest.FixtureRequest
) -> tuple[Any, Mapping[str, Any]]:
    """The case's model in *representation*, with the given ``condition_on`` binds.

    Skips when the representation's backend is absent here, or the case does
    not provide the representation.
    """
    if representation == "probpipe":
        return case.model, case.data
    if representation == "pymc":
        pytest.importorskip("pymc")
        if case.pymc_model is None:
            pytest.skip(f"the case {case.name!r} has no PyMC representation")
        return case.pymc_model(), case.data
    if representation == "stan":
        request.getfixturevalue("_stan_toolchain")
        if case.stan_program is None:
            pytest.skip(f"the case {case.name!r} has no Stan program")
        directory = request.getfixturevalue("tmp_path")
        return case.stan_model(directory), dict(case.stan_data)
    raise ValueError(f"unknown representation {representation!r}")


def _skip_reason(report: Any) -> str:
    return getattr(report, "description", "") or "the check reports no route"


# ---------------------------------------------------------------------------
# The binding
# ---------------------------------------------------------------------------


def _params(name: str, profile: MethodProfile) -> list[Any]:
    params = []
    for case_name in canonical.CASES:
        marks = []
        failure = KNOWN_FAILURES.get((name, profile.representation, case_name))
        if failure is not None:
            reason, raises = failure
            marks.append(pytest.mark.pending(reason=reason, raises=raises))
        params.append(pytest.param(case_name, marks=marks, id=case_name))
    return params


def validate_method(
    name: str, *, representation: str | None = None, **method_options: Any
) -> Callable[..., None]:
    """The test function that validates the inference method *name* on every canonical case.

    Parameters
    ----------
    name : str
        A method of the inference-method registry, with a profile in :data:`PROFILES`.
    representation : str, optional
        The model representation, overriding the profile's: ``"probpipe"``,
        ``"pymc"``, or ``"stan"``.
    **method_options
        Method options that override the profile's.

    Returns
    -------
    callable
        A test parametrized over the case names, to bind at module scope.
    """
    profile = PROFILES[name]
    profile = replace(
        profile,
        representation=representation or profile.representation,
        method_options={**profile.method_options, **method_options},
    )

    @pytest.mark.parametrize("case_name", _params(name, profile))
    def test(case_name: str, request: pytest.FixtureRequest) -> None:
        if name not in inference_method_registry.list_methods():
            pytest.skip(f"{name} is not registered here, since its backend is not installed")
        case = canonical.case(case_name)
        model, data = _model_and_data(case, profile.representation, request)
        view = condition_on.with_options(method=name, method_options=profile.method_options)
        report = view.check(model, data)
        if report.feasible is not True:
            pytest.skip(f"{name} does not apply to {case_name}: {_skip_reason(report)}")
        posterior = view(model, data)
        assert_matches(
            posterior,
            case.reference,
            consistent=profile.consistent,
            label=f"{name} on {case_name} ({profile.representation})",
        )

    test.__name__ = test.__qualname__ = f"test_{name}_canonical"
    test.__doc__ = (
        f"``{name}`` recovers each canonical case's reference posterior within its contract.\n\n"
        f"Each case runs one fit of the method, a second or two of sampling, "
        f"so the parametrization takes several seconds in all."
    )
    return test


# ---------------------------------------------------------------------------
# Simulation-based calibration
# ---------------------------------------------------------------------------


def calibration_ranks(
    method: str | None,
    case: ModelTestCase,
    *,
    replications: int,
    draws: int,
    method_options: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """The SBC ranks of *method* on *case*: one row per replication, one column per coordinate.

    Each replication draws the parameters and the observations from the
    model's joint under its own seeded workflow, conditions the model on the
    drawn observations with *method*, or by the route ``condition_on`` selects
    when *method* is ``None``, thins the posterior's draws to *draws* evenly
    spaced ones, and ranks each drawn parameter coordinate among them. Under a
    calibrated method each rank is uniform on ``{0, ..., draws}``.
    """
    observed = tuple(case.data)
    rows = []
    for replication in range(replications):
        with workflow_run(seed=10_000 + replication):
            joint_draw = sample.with_options(raw=True)(case.model)
        given = {name: joint_draw[name] for name in observed}
        view = condition_on
        if method is not None:
            view = condition_on.with_options(
                method=method,
                method_options={**(method_options or {}), "random_seed": replication},
            )
        posterior = view(case.model, given)
        ranks = []
        for path, leaf in case.reference.leaves.items():
            truth = np.ravel(np.asarray(_at(joint_draw, path), dtype=np.float64))
            chains = chains_at(posterior, path, leaf.mean.shape)
            pooled = chains.reshape(-1, truth.size)
            keep = np.linspace(0, pooled.shape[0] - 1, draws).round().astype(int)
            ranks.append((pooled[keep] < truth).sum(axis=0))
        rows.append(np.concatenate(ranks))
    return np.stack(rows)


def uniformity_pvalues(ranks: np.ndarray, draws: int) -> np.ndarray:
    """The Kolmogorov-Smirnov p-value of each column of *ranks* against ``Uniform{0, ..., draws}``."""
    from probpipe.validation._calibration import _ks_uniform

    _, pvalues = _ks_uniform(ranks, draws)
    return pvalues
