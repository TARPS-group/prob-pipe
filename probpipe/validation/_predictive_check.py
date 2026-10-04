"""Predictive checks: replicated data from a model against the observed data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import numpy as np

from ..core._array_backend import array_backend_for
from ..core._numeric_record import NumericRecord
from ..core._record_spec import _reshaped_template
from ..core.record import Record
from ..custom_types import PRNGKey
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution
from ..distributions._empirical import EmpiricalDistribution, _batch_form
from ..distributions._factored import FactoredConditionalDistribution, _event_of, _raw_record
from ..functions import function
from ..functions._broker import _PROBPIPE_DISTRIBUTION_PROVIDER_ABI
from ._workflow_rng import _resolve_validation_key, _validate_positive_int

__all__ = ["predictive_check"]


@function
def predictive_check[D](
    kernel: ConditionalDistribution,
    law: Distribution,
    test_fns: Callable[[D], float] | Sequence[Callable[[D], float]],
    observed_data: D | None = None,
    *,
    num_replications: int = 500,
    key: PRNGKey | None = None,
) -> dict:
    """Compare replicated data with the observed data through test statistics.

    A replication is a draw of the kernel's event from the composition
    ``kernel * law``, which draws the given slots from *law* and then the
    kernel's event at those values. With the posterior as *law* the check is a
    posterior predictive check, and with the prior it is a prior predictive
    check. Each statistic is computed on the same replications and, when
    *observed_data* is given, on the observed data; its p-value is the fraction
    of replications whose statistic is at least the observed one.

    A statistic receives a replication in the form the kernel declares for its
    event: a whole term is its value, such as an array, and an exposed record is
    the mapping of its fields. A statistic that JAX can trace is computed on
    every replication in one ``jax.vmap`` call, and any other in a loop.

    Parameters
    ----------
    kernel : ConditionalDistribution
        The law of the observations given the parameters, such as the
        likelihood of a model ``likelihood * prior``.
    law : Distribution
        A law over the kernel's given slots: the posterior for a posterior
        predictive check, or the prior for a prior predictive check. A
        ``NumericRecord`` of stacked draws is read as the empirical law of its
        rows.
    test_fns : callable or sequence of callables
        One or more test statistics, each mapping a dataset to a scalar. A
        statistic is named by its ``__name__``, and the names in a sequence
        are distinct.
    observed_data : optional
        The observed data, in the form of a replication, or as a mapping from
        the kernel's components to their values, the form ``condition_on``
        takes.
    num_replications : int
        The number of replications.
    key : PRNGKey, optional
        JAX PRNG key. When it is omitted, the workflow supplies the key, so a
        call inside ``workflow_run(seed=...)`` is reproducible.

    Returns
    -------
    Record
        For a single statistic, the record has the fields:

        - ``replicated_statistics``: an ``EmpiricalDistribution`` over the
          statistic's values at the replications.
        - ``test_fn_name``: the statistic's name.
        - ``observed_statistic``: the statistic of *observed_data*, when
          *observed_data* is given.
        - ``p_value``: the fraction of replications whose statistic is at
          least ``observed_statistic``, when *observed_data* is given.

        For a sequence of statistics, each statistic's fields are nested
        under its name, so ``check["mean/p_value"]`` is the p-value of a
        statistic named ``mean``.

    Raises
    ------
    TypeError
        If *kernel* is not a ``ConditionalDistribution``, *law* does not
        sample, or a statistic is not callable.
    ValueError
        If *law* does not produce every given slot of *kernel*, naming the
        missing slots; if *law* produces a component that *kernel* produces;
        if *test_fns* is empty or holds two statistics of the same name; or if
        *num_replications* is not a positive integer.

    Notes
    -----
    Each statistic's result is also recorded in ``law.annotations``, as the
    next child ``check_N`` of the group ``predictive_check``: a dataset with
    the variable ``replicated_statistics`` over the dimension
    ``replication``, and the attributes ``test_fn_name`` and, when
    *observed_data* is given, ``observed_statistic`` and ``p_value``.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from probpipe import Normal, conditional_distribution
    >>> prior = Normal("mu", 0.0, 1.0)
    >>> likelihood = conditional_distribution(
    ...     "y_given_mu",
    ...     lambda mu: Normal("y", mu * jnp.ones(10), 1.0),
    ...     given_spec=prior.event_spec.components,
    ... )
    >>> check = predictive_check(
    ...     likelihood, prior, jnp.mean, jnp.zeros(10), key=jax.random.key(0)
    ... )
    >>> check["replicated_statistics"].num_atoms
    500
    >>> 0.0 <= float(check["p_value"]) <= 1.0
    True
    """
    statistics = _planned_statistics(test_fns)
    num_replications = _validate_positive_int("num_replications", num_replications)
    # A NumericRecord of stacked draws is the empirical law of its rows.
    if isinstance(law, NumericRecord):
        name = getattr(law, "label", "posterior")
        row = _reshaped_template(law.event_template, lambda shape: shape[1:])
        law = EmpiricalDistribution(name, _batch_form(name, law, "draw", row))
    joint = _predictive_joint(kernel, law, "predictive_check")
    observed = None if observed_data is None else _observed_event(kernel, observed_data)

    if key is None:
        # The composition is a ProbPipe law, so its draws follow the distribution ABI.
        key = _resolve_validation_key(
            None,
            operation_kind="predictive-check",
            execution_mode="sampled",
            sample_shape=(num_replications,),
            provider_abi=_PROBPIPE_DISTRIBUTION_PROVIDER_ABI,
        )
    replicated = _replicated_statistics(joint, kernel, statistics, num_replications, key)

    results = {}
    for name, fn in statistics:
        stats_array = replicated[name]
        result: dict[str, Any] = {
            "replicated_statistics": EmpiricalDistribution("replicated_statistics", stats_array),
            "test_fn_name": name,
        }
        if observed is not None:
            obs_stat = float(fn(observed))
            result["observed_statistic"] = obs_stat
            result["p_value"] = float(np.mean(stats_array >= obs_stat))
        results[name] = result
    # The law's annotations collect its validation history, one child per statistic.
    for name, result in results.items():
        _record_check_in_annotations(law, replicated[name], result)

    if callable(test_fns):
        return results[statistics[0][0]]
    return results


def _planned_statistics(test_fns: Any) -> tuple[tuple[str, Callable], ...]:
    """Each statistic of *test_fns* with its name, checked before any draw.

    A statistic is named by its ``__name__``, or by its ``repr`` when it has
    none.

    Raises
    ------
    TypeError
        If *test_fns* is neither a callable nor an iterable, or holds an
        element that is not callable.
    ValueError
        If *test_fns* is empty or holds two statistics of the same name.
    """
    if callable(test_fns):
        candidates: tuple[Any, ...] = (test_fns,)
    else:
        try:
            candidates = tuple(test_fns)
        except TypeError as exc:
            raise TypeError("test_fns must be a callable or an iterable of callables") from exc
    if not candidates:
        raise ValueError("test_fns must contain at least one callable")
    planned: list[tuple[str, Callable]] = []
    names: set[str] = set()
    for index, fn in enumerate(candidates):
        if not callable(fn):
            raise TypeError(f"test_fns[{index}] must be callable; got {type(fn).__name__}")
        name = getattr(fn, "__name__", None)
        name = repr(fn) if name is None else name
        if name in names:
            raise ValueError(
                f"test_fns must have unique names; duplicate name {name!r}. Use named "
                f"functions with distinct names."
            )
        names.add(name)
        planned.append((name, fn))
    return tuple(planned)


def _predictive_joint(
    kernel: ConditionalDistribution, law: Distribution, operation: str
) -> Distribution:
    """The composition ``kernel * law``, from which a replication is drawn.

    Raises
    ------
    TypeError
        If *kernel* is not a ``ConditionalDistribution`` or *law* is not a
        ``Distribution`` that samples.
    ValueError
        If *law* does not produce every given slot of *kernel*, or produces a
        component that *kernel* produces.
    """
    if not isinstance(kernel, ConditionalDistribution):
        raise TypeError(
            f"{operation} takes the kernel of the observations, a ConditionalDistribution; "
            f"got {type(kernel).__name__}"
        )
    if not isinstance(law, Distribution) or not callable(getattr(law, "_sample", None)):
        raise TypeError(
            f"{operation} draws the kernel's given slots from a Distribution that samples; "
            f"got {type(law).__name__}"
        )
    joint = kernel * law
    if isinstance(joint, FactoredConditionalDistribution):
        raise ValueError(
            f"{operation}: the law {law.label!r} does not produce the given slots "
            f"{sorted(joint.given_spec.required)} of the kernel {kernel.label!r}"
        )
    return joint


def _observed_event(kernel: ConditionalDistribution, observed_data: Any) -> Any:
    """*observed_data* in the form of a replication of *kernel*'s event.

    A mapping or a ``Record`` keyed by the kernel's components is
    reconstructed by the kernel's event declaration, so a whole term is its
    component's value; any other value is the observed event as given. A
    value held in a registered array host, such as a pandas or xarray object,
    converts to the JAX array a replication holds, so a statistic receives the
    observed data in the form of a replication.
    """
    if isinstance(observed_data, Record):
        observed_data = _raw_record(observed_data)
    declaration = kernel.event_spec
    if isinstance(observed_data, Mapping) and set(observed_data) == set(declaration.components):
        return _event_of(
            declaration, {key: _as_replication(value) for key, value in observed_data.items()}
        )
    return _as_replication(observed_data)


def _as_replication(value: Any) -> Any:
    """*value* as a replication holds it: a registered array host's JAX array, else *value*."""
    backend = array_backend_for(value)
    return value if backend is None else backend.to_jax(value)


def _replicated_statistics(
    joint: Distribution,
    kernel: ConditionalDistribution,
    statistics: Sequence[tuple[str, Callable]],
    num_replications: int,
    key: PRNGKey,
) -> dict[str, np.ndarray]:
    """Each statistic's values at *num_replications* replications drawn from *joint*.

    One call that samples *joint* gives every replication, and each statistic
    reads the same replications, reconstructed from the joint's draws by
    *kernel*'s event declaration.
    """
    draws = _raw_record(joint._sample(key, (num_replications,)))
    replications = _event_of(kernel.event_spec, draws)
    return {name: _statistic_values(fn, replications, num_replications) for name, fn in statistics}


def _statistic_values(fn: Callable, replications: Any, num_replications: int) -> np.ndarray:
    """*fn* at each replication, a float array of shape ``(num_replications,)``.

    The statistic is vectorized with ``jax.vmap`` over the replications'
    leading axis, and a statistic that JAX cannot trace, or that returns no
    scalar per replication, is called on each replication in a loop.
    """
    try:
        values = np.asarray(jax.vmap(fn)(replications), dtype=np.float64)
    except Exception:
        # The statistic uses Python control flow or a host conversion, which vmap cannot trace.
        values = None
    if values is not None and values.shape == (num_replications,):
        return values
    return np.array(
        [
            float(fn(jax.tree.map(lambda leaf, i=i: leaf[i], replications)))
            for i in range(num_replications)
        ],
        dtype=np.float64,
    )


def _record_check_in_annotations(
    distribution: Any,
    stats_array: Any,
    result: dict[str, Any],
) -> None:
    """Append one statistic's result Dataset under
    ``distribution.annotations["predictive_check/check_N"]``.

    Mutates ``distribution._annotations`` in place. This is the
    documented exception to ``Distribution`` immutability (see
    :attr:`Distribution.annotations` and design II.4) — diagnostic ops
    attach results under named groups rather than returning renamed
    clones, which would break source/identity tracking.

    Encoding:

    - ``replicated_statistics`` becomes a ``DataArray`` of dims
      ``("replication",)``.
    - ``test_fn_name`` + optional ``observed_statistic`` /
      ``p_value`` become Dataset attrs.

    Frozen/slotted distributions (where ``_annotations`` can't be set
    via ``object.__setattr__``) skip the attachment silently — the
    caller still gets the ``result`` dict via the public return.
    """
    try:
        import xarray as xr
        from xarray import DataTree
    except ImportError:
        # xarray isn't available — skip the attachment silently. The
        # caller still gets the ``result`` dict via the return value.
        return

    attrs = {"test_fn_name": result["test_fn_name"]}
    if "observed_statistic" in result:
        attrs["observed_statistic"] = result["observed_statistic"]
        attrs["p_value"] = result["p_value"]
    ds = xr.Dataset(
        {"replicated_statistics": (("replication",), np.asarray(stats_array))},
        attrs=attrs,
    )

    aux = getattr(distribution, "_annotations", None)
    if aux is None:
        aux = DataTree()
        try:
            object.__setattr__(distribution, "_annotations", aux)
        except (AttributeError, TypeError):
            # Frozen/immutable distribution — give up silently.
            return
    group = aux.get("predictive_check")
    if group is None:
        aux["predictive_check"] = DataTree()
        group = aux["predictive_check"]
    n_existing = len(list(group.children))
    aux[f"predictive_check/check_{n_existing}"] = DataTree(dataset=ds)
