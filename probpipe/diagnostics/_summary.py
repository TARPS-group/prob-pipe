"""``summarize``: a law's standard summaries in one record that displays as a table.

Usage::

    from probpipe import Gamma, Normal, summarize
    s = summarize(Normal("mu", 0.0, 2.0) * Gamma("tau", 2.0, 1.0))
    s["sd/mu"]   # the standard deviation of mu
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import jax.numpy as jnp
import numpy as np

from .._messages import count
from ..core._numeric_array import NumericArray
from ..core._numeric_record import NumericRecord
from ..core._numeric_record_batch import NumericRecordBatch
from ..core._repr import type_name
from ..core._table import COMPUTATIONS, column_provenance, computation, path_shapes
from ..core.named_tree import _PATH_SEP
from ..core.provenance import Provenance
from ..core.record import Record
from ..distributions._batches import DistributionBatch
from ..distributions._distribution import Distribution
from ..functions._recipe import monte_carlo_draws
from ..operations import mean, quantile, standard_deviation

__all__ = ["summarize"]


@dataclass(frozen=True)
class _Statistic:
    """One column of a summary: its name, the call that computes it, and its MCMC standard error.

    *mcse_method* and *prob* name the ArviZ Monte Carlo standard error of the
    column for an MCMC posterior, and ``None`` where ArviZ has none.
    """

    name: str
    compute: Callable[[Any], Any]
    mcse_method: str | None = None
    prob: float | None = None


#: The operations whose ArviZ Monte Carlo standard error has a method of its own.
_MCSE_METHODS: tuple[tuple[Callable[[Any], Any], str], ...] = (
    (mean, "mean"),
    (standard_deviation, "sd"),
)


def _mcse_method(compute: Callable[[Any], Any]) -> str | None:
    """The ArviZ standard error method of the operation *compute*, or ``None``."""
    return next((method for operation, method in _MCSE_METHODS if compute is operation), None)


def _quantile_at(prob: float) -> Callable[[Any], Any]:
    """The statistic ``quantile(d, prob)``."""

    def at(d: Any) -> Any:
        return quantile(d, prob)

    return at


def _percent_name(prob: float) -> str:
    """The column name of the quantile at *prob*, as ``q10`` for 0.1 and ``q2.5`` for 0.025."""
    return f"q{round(100 * prob, 6):g}"


def _default_statistics(interval: float) -> list[_Statistic]:
    """The default columns: the mean, the standard deviation, the median, and the interval's ends."""
    low, high = (1.0 - interval) / 2.0, (1.0 + interval) / 2.0
    return [
        _Statistic("mean", mean, "mean"),
        _Statistic("sd", standard_deviation, "sd"),
        _Statistic("median", _quantile_at(0.5), "median"),
        _Statistic(_percent_name(low), _quantile_at(low), "quantile", low),
        _Statistic(_percent_name(high), _quantile_at(high), "quantile", high),
    ]


def _given_statistics(statistics: tuple[Any, ...]) -> list[_Statistic]:
    """The columns that *statistics* name, in order.

    A mapping names each callable it holds by its key. A callable is named by
    its label, which an operation's name is, or by its ``__name__``, and a name
    that is not an identifier, as a lambda's ``<lambda>``, names no statistic.
    """
    given: list[_Statistic] = []
    for item in statistics:
        if isinstance(item, Mapping):
            pairs = list(item.items())
        elif callable(item):
            name = getattr(item, "label", None) or getattr(item, "__name__", None)
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(
                    f"summarize cannot name the statistic {name!r}. Pass it in a mapping, "
                    "as summarize(d, {'name': statistic})"
                )
            pairs = [(name, item)]
        else:
            raise TypeError(
                f"summarize: each statistic must be a callable or a mapping of names to "
                f"callables, got {type_name(item)}"
            )
        for name, compute in pairs:
            if not isinstance(name, str) or not name or _PATH_SEP in name:
                raise ValueError(
                    f"summarize: a statistic's name must be a non-empty str without "
                    f"{_PATH_SEP!r}, got {name!r}"
                )
            if not callable(compute):
                raise TypeError(
                    f"summarize: the statistic {name!r} must be callable, got {type_name(compute)}"
                )
            given.append(_Statistic(name, compute, _mcse_method(compute)))
    names = [statistic.name for statistic in given]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(
            f"summarize: each statistic must have its own name, got {repeated} more than once"
        )
    return given


def _column(statistic: _Statistic, d: Distribution) -> Record:
    """The statistic of *d* as a record over the law's event paths.

    A record is the term the statistic returns, and an array, which a law of one
    whole term gives, is held under the law's component.
    """
    value = statistic.compute(d)
    if isinstance(value, NumericRecord):
        return value
    components = tuple(d.event_spec.components)
    if isinstance(value, NumericArray) and len(components) == 1:
        return Record({components[0]: value})
    raise TypeError(
        f"summarize: the statistic {statistic.name!r} must return a numeric value over the "
        f"components {list(components)} of the law, got {type_name(value)}"
    )


def _filled(shapes: Mapping[str, tuple[int, ...]], value: float) -> dict[str, Any]:
    """A column that holds *value* at each path, in the shape of that path."""
    return {path: jnp.full(shape, value) for path, shape in shapes.items()}


def _chain_columns(
    d: Distribution, statistics: list[_Statistic], mcse: bool
) -> dict[str, dict[str, Any]]:
    """An MCMC posterior's MCSE columns, when *mcse*, then ``r_hat``, ``ess_bulk``, and ``ess_tail``.

    The values come from the ArviZ computations that ``add_mcmc_diagnostics``
    runs, keyed by the posterior's event paths, and the posterior's annotations
    are left as they are. R-hat of fewer than two chains is ``nan``.
    """
    from ..inference._approximate_distribution import _num_chains
    from ._datatree_store import to_named_posterior_dataset
    from ._mcmc import arviz_ess, arviz_mcse, arviz_rhat

    dataset = to_named_posterior_dataset(d)
    shapes = {str(path): tuple(dataset[path].shape[2:]) for path in dataset.data_vars}

    def by_path(result: Any) -> dict[str, Any]:
        return {path: jnp.asarray(result[path].values) for path in shapes}

    columns: dict[str, dict[str, Any]] = {}
    if mcse:
        for statistic in statistics:
            columns[f"mcse_{statistic.name}"] = (
                _filled(shapes, np.nan)
                if statistic.mcse_method is None
                else by_path(arviz_mcse(dataset, method=statistic.mcse_method, prob=statistic.prob))
            )
    columns["r_hat"] = (
        by_path(arviz_rhat(dataset, method="rank"))
        if _num_chains(d) >= 2
        else _filled(shapes, np.nan)
    )
    columns["ess_bulk"] = by_path(arviz_ess(dataset, method="bulk"))
    columns["ess_tail"] = by_path(arviz_ess(dataset, method="tail"))
    return columns


def _chain_computation(d: Distribution) -> str:
    """How each statistic of an MCMC posterior was computed, as ``MCMC, 4 chains of 1,000 draws``."""
    chains, draws = (int(n) for n in d.atoms.batch_shape)
    return f"MCMC, {count(chains, 'chain')} of {draws:,} draws"


def _monte_carlo_errors(
    d: Distribution, statistics: list[_Statistic], columns: Mapping[str, Record]
) -> dict[str, dict[str, Any]]:
    """The Monte Carlo standard error column of each statistic of a law that is not an MCMC posterior.

    A path that the statistic's route computed exactly has ``0``: every path of
    an exact route, and each path an approximate route took no Monte Carlo
    draws of. The error of a mean estimated from ``n`` independent draws is
    ``sd / sqrt(n)``. No estimator applies to another statistic without its
    draws, nor to a column whose provenance records neither its exactness nor
    its draws, so their paths hold ``nan``.
    """
    sd: Record | None = None

    def standard_deviations() -> Record:
        nonlocal sd
        if sd is None:
            named = [other.name for other in statistics if other.compute is standard_deviation]
            sd = columns[named[0]] if named else _column(_Statistic("sd", standard_deviation), d)
        return sd

    errors: dict[str, dict[str, Any]] = {}
    for statistic in statistics:
        column = columns[statistic.name]
        shapes = path_shapes(column)
        provenance = column_provenance(column)
        if computation(provenance) == "exact":
            errors[f"mcse_{statistic.name}"] = _filled(shapes, 0.0)
            continue
        draws = monte_carlo_draws(provenance)
        error: dict[str, Any] = {}
        for path, shape in shapes.items():
            parts = tuple(path.split(_PATH_SEP))
            counts = [n for part, n in draws.items() if parts[: len(part)] == part]
            if draws and not counts:
                error[path] = jnp.zeros(shape)
            elif counts and statistic.mcse_method == "mean":
                error[path] = jnp.asarray(standard_deviations().raw(path)) / np.sqrt(counts[0])
            else:
                error[path] = jnp.full(shape, np.nan)
        errors[f"mcse_{statistic.name}"] = error
    return errors


def _summary(
    d: Distribution, statistics: list[_Statistic], mcse: bool
) -> tuple[NumericRecord, dict[str, str | None]]:
    """The summary record of one law, and how each of its statistic columns was computed."""
    from ..inference._approximate_distribution import _has_chains

    columns = {statistic.name: _column(statistic, d) for statistic in statistics}
    first, *others = statistics
    shapes = path_shapes(columns[first.name])
    for statistic in others:
        given = path_shapes(columns[statistic.name])
        if given != shapes:
            raise ValueError(
                f"summarize: the statistics {first.name!r} and {statistic.name!r} must give "
                f"values over the same paths and shapes, got {shapes} and {given}"
            )
    fields: dict[str, Any] = dict(columns)
    computations: dict[str, str | None]
    if _has_chains(d):
        fields.update(_chain_columns(d, statistics, mcse))
        computations = dict.fromkeys(columns, _chain_computation(d))
    else:
        if mcse:
            fields.update(_monte_carlo_errors(d, statistics, columns))
        computations = {
            name: computation(column_provenance(column)) for name, column in columns.items()
        }
    return Record(fields, label=f"summary of {d}"), computations


def _summary_provenance(d: Any, computations: Mapping[str, Any]) -> Provenance | None:
    """The provenance of a summary of *d*, which describes how each statistic column was computed."""
    return Provenance.create(
        "summarize", parents=[d], metadata={"func": "summarize", COMPUTATIONS: dict(computations)}
    )


def _summary_batch(
    d: DistributionBatch, statistics: list[_Statistic], mcse: bool
) -> NumericRecordBatch:
    """The batch of the summary records of the laws of *d*, on its levels.

    A column that only some laws give, as ``r_hat`` of a batch that mixes MCMC
    posteriors with other laws, holds ``nan`` for the others.
    """
    shape = tuple(d.batch_shape)
    summaries = [_summary(d[index], statistics, mcse) for index in np.ndindex(*shape)]
    if not summaries:
        raise ValueError(f"summarize: the batch must hold at least one law, got shape {shape}")
    shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    for record, _ in summaries:
        for name in record.children:
            given = path_shapes(record.at_path(name))
            expected = shapes.setdefault(name, given)
            if given != expected:
                raise ValueError(
                    f"summarize: the laws of the batch must give the column {name!r} over the "
                    f"same paths and shapes, got {expected} and {given}"
                )

    def stacked(name: str, path: str) -> Any:
        values = [
            jnp.asarray(record.raw(f"{name}{_PATH_SEP}{path}"))
            if name in record.children
            else jnp.full(shapes[name][path], np.nan)
            for record, _ in summaries
        ]
        return jnp.stack(values).reshape(*shape, *shapes[name][path])

    columns = {
        f"{name}{_PATH_SEP}{path}": stacked(name, path)
        for name, paths in shapes.items()
        for path in paths
    }
    computations = {
        statistic.name: [described[statistic.name] for _, described in summaries]
        for statistic in statistics
    }
    batch = NumericRecordBatch(
        columns,
        tuple(d.level_names),
        axes_per_level=tuple(len(group) for group in d.axis_groups),
        label=f"summary of {d}",
    )
    return batch.with_provenance(_summary_provenance(d, computations))


# summarize is a plain function rather than a Function (STYLE_GUIDE.md §1.4): its
# result is a record of the terms each statistic returns, each with its own
# provenance, and a Function call would wrap that record under one label that
# renders every statistic it is given.
def summarize(
    d: Distribution | DistributionBatch,
    *statistics: Callable[[Any], Any] | Mapping[str, Callable[[Any], Any]],
    interval: float = 0.8,
    mcse: bool = False,
) -> NumericRecord | NumericRecordBatch:
    """Summarize a law in a record of standard summaries, which displays as a table.

    The result is a ``NumericRecord`` nested by statistic: each top-level field
    is the term that the statistic returns for *d*, over the law's event paths,
    so ``summarize(d)["sd/mu"]`` is the standard deviation of ``mu``. A law of
    one whole term holds each statistic under its component, as ``mean/theta``.
    The record displays as a table with one row for each path and one column
    for each statistic, under the label ``summary of`` and the law's notation.
    The line below the table states how each statistic was computed:
    ``exact``, ``Monte Carlo`` with its draws, or ``approximate``, as the
    statistic's provenance records it, and, for an MCMC posterior, ``MCMC``
    with its chains and draws. The record's provenance records those
    descriptions. ``pd.DataFrame(summarize(d).raw())`` gives the table of a law
    whose components are scalars as a DataFrame.

    The default columns are:

    - ``mean``: ``mean(d)``;
    - ``sd``: ``standard_deviation(d)``;
    - ``median``: ``quantile(d, 0.5)``;
    - the ends of the central interval of probability *interval*, as ``q10`` and
      ``q90`` for ``0.8``: ``quantile(d, (1 - interval) / 2)`` and
      ``quantile(d, (1 + interval) / 2)``.

    A posterior whose draws lie on the levels ``chain`` and ``draw``, as an MCMC
    method returns it, also has the columns ``r_hat``, ``ess_bulk``, and
    ``ess_tail``, computed as ``add_mcmc_diagnostics`` computes them and keyed by
    the same paths. The posterior's annotations are left as they are.

    With ``mcse=True``, each statistic gains a column ``mcse_<name>`` of its
    Monte Carlo standard error, after the statistic columns:

    1. **An MCMC posterior:** each statistic of its draws estimates the posterior
       the sampler targets, so each column holds the ArviZ standard error of
       its kind: ``mean``, ``sd``, ``median``, or the quantile at its
       probability. A statistic of another kind has ``nan``.
    2. **Another law:** a path that the statistic's route computed exactly has
       ``0``. For ``mean``, a path its route drew ``n`` draws of has
       ``sd / sqrt(n)``. Any other path has ``nan``, since its standard error
       needs the draws the route took.

    Parameters
    ----------
    d : Distribution or DistributionBatch
        The law to summarize. A ``DistributionBatch`` gives a batch of summary
        records, one for each law, on the batch's levels. A column that only
        some laws give, as ``r_hat`` of a batch that mixes MCMC posteriors with
        other laws, holds ``nan`` for the others.
    *statistics : callable or Mapping of str to callable
        The columns to compute in place of the default ones, each a callable
        that takes *d*, such as an operation. A callable is named by its label,
        as ``mean`` and ``variance`` are, or by its ``__name__``, and the name
        must be an identifier. A mapping names its callables by its keys, as
        ``{"sd": standard_deviation}`` does.
    interval : float
        The probability of the central interval whose ends the default columns
        hold, strictly between 0 and 1. It does not apply to given statistics.
    mcse : bool
        Whether to add the Monte Carlo standard error columns.

    Returns
    -------
    NumericRecord or NumericRecordBatch
        The summary record labeled ``summary of`` and the law's notation, with one
        top-level field for each column, or, for a ``DistributionBatch``, the
        batch of the summary records of its laws.

    Raises
    ------
    TypeError
        If *d* is not a ``Distribution`` or a ``DistributionBatch``, *interval* is
        not a real number, a statistic is neither a callable nor a mapping of
        names to callables, or a statistic returns a value that is not numeric
        over the law's components.
    ValueError
        If *interval* is not strictly between 0 and 1, a callable's name is not
        an identifier, a name is empty, has ``/``, or names another statistic,
        two statistics return values over different paths or shapes, a batch
        holds no law, or the laws of a batch give a column over different paths
        or shapes.

    Examples
    --------
    >>> from probpipe import Gamma, Normal, summarize
    >>> s = summarize((Normal("mu", 0.0, 2.0) * Gamma("tau", 2.0, 1.0)).with_label("model"))
    >>> s.label
    'summary of model(mu, tau)'
    >>> float(s["sd/mu"])
    2.0
    """
    if not isinstance(d, (Distribution, DistributionBatch)):
        raise TypeError(
            f"summarize: d must be a Distribution or a DistributionBatch, got {type_name(d)}"
        )
    if isinstance(interval, bool) or not isinstance(interval, Real):
        raise TypeError(f"summarize: interval must be a number, got {type_name(interval)}")
    if not 0.0 < float(interval) < 1.0:
        raise ValueError(f"summarize: interval must be strictly between 0 and 1, got {interval}")
    chosen = _given_statistics(statistics) if statistics else _default_statistics(float(interval))
    if isinstance(d, DistributionBatch):
        return _summary_batch(d, chosen, mcse)
    record, computations = _summary(d, chosen, mcse)
    return record.with_provenance(_summary_provenance(d, computations))
