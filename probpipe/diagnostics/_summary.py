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

import numpy as np

from ..core._array_backend import _event_shape_of
from ..core._numeric_array import NumericArray
from ..core._numeric_record import NumericRecord
from ..core._numeric_record_batch import NumericRecordBatch
from ..core._repr import type_name
from ..core._table import column_provenance, monte_carlo_draws
from ..core.named_tree import _PATH_SEP
from ..core.record import Record
from ..distributions._batches import DistributionBatch
from ..distributions._distribution import Distribution
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


#: The ArviZ Monte Carlo standard error of each operation that has one.
_MCSE_METHODS: dict[int, str] = {id(mean): "mean", id(standard_deviation): "sd"}


def _given_statistics(statistics: tuple[Any, ...]) -> list[_Statistic]:
    """The columns that *statistics* name, in order.

    A callable is named by its label, which an operation's name is, or by its
    ``__name__``, and a mapping names each callable it holds by its key.
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
            given.append(_Statistic(name, compute, _MCSE_METHODS.get(id(compute))))
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


def _shapes(column: Record) -> dict[str, tuple[int, ...]]:
    """Each path of *column* and the shape of its value."""
    return {path: tuple(_event_shape_of(column.raw(path))) for path in column}


def _chain_summaries(d: Distribution, statistics: list[_Statistic], mcse: bool) -> dict[str, Any]:
    """The columns ``r_hat``, ``ess_bulk``, and ``ess_tail`` of an MCMC posterior, and its MCSEs.

    The values come from the ArviZ computations that ``add_mcmc_diagnostics``
    runs, keyed by the posterior's event paths, and the posterior's annotations
    are left as they are. R-hat of fewer than two chains is ``nan``.
    """
    from ..inference._approximate_distribution import _num_chains
    from ._datatree_store import to_named_posterior_dataset
    from ._mcmc import _check_arviz_stats, arviz_ess, arviz_rhat

    _check_arviz_stats()
    import arviz_stats

    dataset = to_named_posterior_dataset(d)
    paths = list(dataset.data_vars)

    def by_path(result: Any) -> dict[str, np.ndarray]:
        return {path: np.asarray(result[path], dtype=float) for path in paths}

    shapes = {path: dataset[path].shape[2:] for path in paths}
    columns: dict[str, Any] = {}
    if mcse:
        for statistic in statistics:
            if statistic.mcse_method is None:
                columns[f"mcse_{statistic.name}"] = {
                    path: np.full(shape, np.nan) for path, shape in shapes.items()
                }
            else:
                result = arviz_stats.mcse(
                    dataset, method=statistic.mcse_method, prob=statistic.prob
                )
                columns[f"mcse_{statistic.name}"] = by_path(result)
    if _num_chains(d) >= 2:
        columns["r_hat"] = by_path(arviz_rhat(dataset, method="rank"))
    else:
        columns["r_hat"] = {path: np.full(shape, np.nan) for path, shape in shapes.items()}
    columns["ess_bulk"] = by_path(arviz_ess(dataset, method="bulk"))
    columns["ess_tail"] = by_path(arviz_ess(dataset, method="tail"))
    return columns


def _monte_carlo_errors(
    d: Distribution, statistics: list[_Statistic], columns: Mapping[str, Record]
) -> dict[str, Any]:
    """A Monte Carlo standard error column for each column whose route is approximate.

    A path that the route computed exactly, which its provenance records by
    drawing no Monte Carlo draws of it, has ``0``. The error of a mean estimated
    from ``n`` independent draws is ``sd / sqrt(n)``. No estimator applies to
    another statistic without the draws, nor to a column whose provenance
    records no draws, so their other paths hold ``nan``.
    """
    errors: dict[str, Any] = {}
    sd: Record | None = None
    for statistic in statistics:
        column = columns[statistic.name]
        provenance = column_provenance(column)
        if provenance is None or (provenance.metadata or {}).get("exact") is not False:
            continue
        draws = monte_carlo_draws(provenance)
        if statistic.mcse_method == "mean" and sd is None:
            named = [other.name for other in statistics if other.compute is standard_deviation]
            sd = columns[named[0]] if named else _column(_Statistic("sd", standard_deviation), d)
        error: dict[str, Any] = {}
        for path, shape in _shapes(column).items():
            parts = tuple(path.split(_PATH_SEP))
            counts = [n for part, n in draws.items() if parts[: len(part)] == part]
            if draws and not counts:
                error[path] = np.zeros(shape)
            elif counts and statistic.mcse_method == "mean" and sd is not None:
                error[path] = np.asarray(sd.raw(path), dtype=float) / np.sqrt(counts[0])
            else:
                error[path] = np.full(shape, np.nan)
        errors[f"mcse_{statistic.name}"] = error
    return errors


def _summarize_law(d: Distribution, statistics: list[_Statistic], mcse: bool) -> NumericRecord:
    """The summary record of one law."""
    from ..inference._approximate_distribution import _has_chains

    columns = {statistic.name: _column(statistic, d) for statistic in statistics}
    shapes = None
    for name, column in columns.items():
        if shapes is None:
            shapes = _shapes(column)
        elif _shapes(column) != shapes:
            raise ValueError(
                f"summarize: the statistic {name!r} has the paths and shapes {_shapes(column)}, "
                f"but the first has {shapes}"
            )
    fields: dict[str, Any] = dict(columns)
    if _has_chains(d):
        fields.update(_chain_summaries(d, statistics, mcse))
    elif mcse:
        fields.update(_monte_carlo_errors(d, statistics, columns))
    return Record(fields, label=f"summary of {d}")


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
    for each statistic, under the label ``summary of`` and the law's notation,
    and the line below the table states which columns are exact and which are
    Monte Carlo, with their draws, as each column's provenance records it.
    ``pd.DataFrame(summarize(d).raw())`` gives the table of a law whose
    components are scalars as a DataFrame.

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

    With ``mcse=True``, each column that is a Monte Carlo estimate gains a
    column ``mcse_<name>`` of its Monte Carlo standard error:

    1. **An MCMC posterior:** each statistic of its draws estimates the posterior
       the sampler targets, so every statistic column gains the ArviZ standard
       error of its kind: ``mean``, ``sd``, ``median``, or the quantile at its
       probability. A statistic of another kind has ``nan``.
    2. **Another law:** a column whose provenance records an approximate route
       gains one. For ``mean`` it is ``sd / sqrt(n)`` at each path its route drew
       ``n`` draws of, and ``0`` at each path its route computed exactly. Any
       other statistic has ``nan``, since its standard error needs the draws the
       route took.

    Parameters
    ----------
    d : Distribution or DistributionBatch
        The law to summarize. A ``DistributionBatch`` gives a batch of summary
        records, one for each law, on the batch's levels.
    *statistics : callable or Mapping of str to callable
        The columns to compute in place of the default ones, each a callable
        that takes *d*, such as an operation. A callable is named by its label,
        as ``mean`` and ``variance`` are, and a mapping names its callables by
        its keys, as ``{"sd": standard_deviation}`` does.
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
        If *interval* is not strictly between 0 and 1, a statistic has no name,
        a name with ``/``, or the name of another, two statistics return values
        over different paths or shapes, or the laws of a batch give summaries
        over different fields.

    Notes
    -----
    ``summarize`` is a plain function rather than a ``Function``: its result is a
    record of the terms each statistic returns, each with its own provenance,
    and a ``Function`` call would wrap that record under a label that renders
    every statistic it is given.

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
    if isinstance(d, Distribution):
        return _summarize_law(d, chosen, mcse)
    shape = tuple(d.batch_shape)
    records = [_summarize_law(d[index], chosen, mcse) for index in np.ndindex(*shape)]
    paths = records[0].keys()
    for record in records[1:]:
        if record.keys() != paths:
            raise ValueError(
                f"summarize: the laws of the batch must give summaries over the same fields, "
                f"got {list(paths)} and {list(record.keys())}"
            )
    columns = {
        path: np.stack([np.asarray(record.raw(path)) for record in records]).reshape(
            *shape, *np.shape(records[0].raw(path))
        )
        for path in paths
    }
    return NumericRecordBatch(
        columns,
        tuple(d.level_names),
        axes_per_level=tuple(len(group) for group in d.axis_groups),
        label=f"summary of {d}",
    )
