"""The resampling families: the bootstrap and the kernel density estimate.

The **bootstrap** takes any law that samples as its source, which covers the
nonparametric bootstrap, where an empirical law is resampled, and the
parametric bootstrap, where a fitted law is redrawn. A replicate is
``replicate_size`` iid draws from the source in the event's batch form, on one
level. A **kernel density estimate** smooths its atoms with a **smoothing
kernel**: a mean-zero density ``K`` recentered at each atom and scaled by the
bandwidth, so its law is the weighted mixture ``Σᵢ wᵢ h⁻ᵈ K((x − xᵢ)/h)``. A
kernel class builds the bank of placed copies through one uniform constructor,
so the estimate holds the kernel class and never reads kernel-specific
parameters.

Provides:
  - ``BootstrapReplicateDistribution`` – the law of one replicate.
  - ``BootstrapDistribution`` – the bootstrap random measure, whose draw is the
    empirical measure of one replicate.
  - ``SmoothingKernel`` – a bank of mean-zero kernel copies, one per center.
  - ``GaussianKernel`` – the standard normal kernel.
  - ``EpanechnikovKernel`` – the product Epanechnikov kernel, compactly
    supported on ``[-1, 1]`` in each coordinate.
  - ``KDEDistribution`` – the kernel density estimate of weighted atoms.
"""

from __future__ import annotations

import math
import operator
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .._messages import unknown_names
from .._weights import Weights, weighted_choice, weighted_covariance, weighted_mean
from ..core._batch import BatchSpec
from ..core._numeric_record import NumericRecord
from ..core._numeric_record_batch import NumericRecordBatch
from ..core._record_spec import NumericRecordSpec, RecordSpec
from ..core._repr import format_value
from ..core._spec_base import NumericArraySpec, TermSpec
from ..core._specs import OutputSpec, _check_component_name
from ..core.constraints import real
from ..core.named_tree import _unflatten_paths
from ..distributions._capabilities import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from ..distributions._distribution import (
    Distribution,
    DistributionSpec,
    _class_label,
    _constructor_label,
    _label_given_first,
    _whole_term_event,
)
from ..distributions._empirical import EmpiricalDistribution, _atoms_declaration, _batch_form
from ..distributions._factored import _raw_record
from ..linalg import DenseLinOp
from ._random_functions import RandomMeasure

if TYPE_CHECKING:
    from ..custom_types import Array, ArrayLike, PRNGKey
    from ..linalg import LinOp

__all__ = [
    "BootstrapDistribution",
    "BootstrapReplicateDistribution",
    "EpanechnikovKernel",
    "GaussianKernel",
    "KDEDistribution",
    "SmoothingKernel",
]

_PATH_SEP = "/"


class _BankNames(NamedTuple):
    """How a kernel bank's messages name its arguments: the scales, the centers, and one center."""

    scales: str
    centers: str
    center: str


#: The names a ``SmoothingKernel`` caller passes the bank by.
_KERNEL_NAMES = _BankNames("scales", "centers", "center")

#: The names a ``KDEDistribution`` caller passes the bank by.
_KDE_NAMES = _BankNames("bandwidth", "atoms", "atom")


def _type_name(value: Any) -> str:
    """The type of *value* as a message names it, with any array reported by its shape."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return f"an array of shape {tuple(value.shape)}"
    return type(value).__name__


# ---------------------------------------------------------------------------
# The bootstrap
# ---------------------------------------------------------------------------


def _sampling_source(source: Any) -> Distribution:
    """*source*, checked to be a law that samples.

    Parameters
    ----------
    source : Any
        The bootstrap's source, as the constructor received it.

    Returns
    -------
    Distribution
        *source* itself, which implements ``SupportsSampling``.

    Raises
    ------
    TypeError
        If *source* is not a ``Distribution`` that implements ``SupportsSampling``.
    """
    if not isinstance(source, Distribution) or not isinstance(source, SupportsSampling):
        raise TypeError(
            f"source must be a Distribution that supports sampling, got {_type_name(source)}"
        )
    return source


def _replicate_size(source: Distribution, replicate_size: Any) -> int:
    """The number of draws in one replicate of *source*.

    It defaults to the atom count of an empirical source.

    Parameters
    ----------
    source : Distribution
        The bootstrap's source, whose atom count is the default for an empirical law.
    replicate_size : Any
        The requested number of draws as the constructor received it, or None for the
        default.

    Returns
    -------
    int
        A positive integer: *replicate_size* as an ``int``, or the source's atom count.

    Raises
    ------
    TypeError
        If *replicate_size* is not an integer.
    ValueError
        If *replicate_size* is not positive, or is omitted for a source that is not
        empirical.
    """
    if replicate_size is None:
        if isinstance(source, EmpiricalDistribution):
            return source.num_atoms
        raise ValueError(
            f"replicate_size is required unless the source is an EmpiricalDistribution, got "
            f"{type(source).__name__} {source.label!r}"
        )
    if isinstance(replicate_size, bool):
        raise TypeError("replicate_size must be an integer, got bool")
    try:
        size = operator.index(replicate_size)
    except TypeError:
        raise TypeError(
            f"replicate_size must be an integer, got {_type_name(replicate_size)}"
        ) from None
    if size < 1:
        raise ValueError(f"replicate_size must be positive, got {size}")
    return size


def _replicate_level(source: Distribution, level: str | None) -> str:
    """The level a replicate's draws lie on.

    It defaults to the atom level of an empirical source with exactly one, and
    otherwise to the source's component when the source has one component, as a
    whole-term event does.

    Parameters
    ----------
    source : Distribution
        The bootstrap's source, whose atom level or component gives the default.
    level : str or None
        The requested level name, or None for the default.

    Returns
    -------
    str
        The name of that level, which is *level* itself when it is given.

    Raises
    ------
    TypeError
        If *level* is not a string.
    ValueError
        If *level* is empty or contains ``/``, or is omitted for a source that
        exposes a record of several components.
    """
    if level is not None:
        if not isinstance(level, str):
            raise TypeError(f"level must be a string, got {_type_name(level)}")
        _check_component_name(level, context="level names")
        return level
    if isinstance(source, EmpiricalDistribution) and len(source.atoms.level_names) == 1:
        return source.atoms.level_names[0]
    components = list(source.event_spec.components)
    if len(components) == 1:
        return components[0]
    raise ValueError(
        f"level is required because the source {source.label!r} has several components "
        f"{components}; pass level='<name>' to name the level of each replicate's draws"
    )


def _replicates(
    source: Distribution, key: PRNGKey, sample_shape: tuple[int, ...], size: int
) -> Any:
    """Raw draws of replicates of *source*, ``(*sample_shape, size)`` iid draws.

    Every replicate's draws are independent draws of the source, so they are
    drawn in one call with the replicate axis after the sample axes. Record
    draws are returned as the nested mapping of their raw leaves.
    """
    return _raw_record(source._sample(key, (*tuple(sample_shape), size)))


def _replicate_at(raw: Any, position: tuple[int, ...]) -> Any:
    """The replicate at *position* of the sample axes of *raw*, raw draws of several replicates."""
    return jax.tree.map(lambda leaf: leaf[position], raw)


class BootstrapReplicateDistribution(Distribution, SupportsSampling):
    """The law of one bootstrap replicate: ``replicate_size`` iid draws of a source.

    A draw is one **replicate**, the source's draws in the event's batch form on
    one level. The source is any law that samples: an empirical source is
    resampled by weight, which is the nonparametric bootstrap, and a fitted law
    is redrawn, which is the parametric bootstrap.

    **The event declaration.** One draw is a batch of the source's event term on
    the replicate's level, so a replicate keeps the source's term kind. The
    declaration is the law's own, derived from the source and the replicate
    size, under *component*.

    Parameters
    ----------
    component : str
        The component of the law's event.
    source : Distribution
        The law the replicate draws from, which implements ``SupportsSampling``.
    replicate_size : int, optional
        The number of draws in one replicate. It defaults to the atom count of
        an empirical source and is required otherwise.
    label : str, optional
        The law's label, ``BootstrapReplicateDistribution`` by default.
    level : str, optional
        The level a replicate's draws lie on. It defaults to the source's atom
        level when the source is an empirical law with exactly one, and
        otherwise to the source's component when the source has one, as a
        whole-term event does.
    event_spec : OutputSpec, optional
        A declaration of *component* that declares the type of one draw.

    Raises
    ------
    TypeError
        If *source* is not a law that samples, *component* or *level* is not a
        string, *label* is not a non-empty string, *replicate_size* is not an
        integer, or *event_spec* is not an ``OutputSpec`` or exposes a record.
    ValueError
        If *replicate_size* is not positive or is omitted for a source without
        atoms, *level* is omitted for a source exposing several components or is
        not a valid level name, *event_spec* names another component, or it
        declares a type that does not unify with the replicate's.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> data = EmpiricalDistribution(jnp.array([1.0, 2.0, 4.0]), component="y")
    >>> replicate = BootstrapReplicateDistribution("boot", data)
    >>> replicate.replicate_size
    3
    >>> replicate.event_spec.spec.level_names
    ('y',)
    """

    def __init__(
        self,
        component: str,
        source: SupportsSampling,
        replicate_size: int | None = None,
        *,
        label: str | None = None,
        level: str | None = None,
        event_spec: OutputSpec | None = None,
    ) -> None:
        law = _sampling_source(source)
        size = _replicate_size(law, replicate_size)
        on_level = _replicate_level(law, level)
        term = _replicate_spec(law, size, on_level)
        owner = _class_label(self)
        super().__init__(
            _constructor_label(self, label, owner),
            _whole_term_event(component, term, event_spec, owner),
        )
        self._source = law
        self._replicate_size = size
        self._level = on_level

    @property
    def source(self) -> Distribution:
        """The law a replicate draws from."""
        return self._source

    @property
    def replicate_size(self) -> int:
        """The number of draws in one replicate."""
        return self._replicate_size

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Any:
        """Draw replicates, each ``replicate_size`` iid draws of the source.

        Parameters
        ----------
        key : PRNGKey
            The key of the draws.
        sample_shape : tuple of int, optional
            The batch axes of the replicates.

        Returns
        -------
        Any
            One replicate in its raw form for ``sample_shape=()``, the source's
            raw draws along a leading axis of length ``replicate_size``: an array
            for array draws, the nested mapping of raw leaves for record draws,
            and an object array otherwise. A non-empty shape prepends its axes.
        """
        return _replicates(self._source, key, sample_shape, self._replicate_size)

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The source law and the replicate size."""
        return [("source", repr(self._source)), ("replicate_size", repr(self._replicate_size))]


def _replicate_spec(source: Distribution, size: int, level: str) -> TermSpec:
    """The term spec of one replicate: *size* of the source's draws on *level*."""
    return BatchSpec(source.event_spec.spec, {level: size})


class BootstrapDistribution(RandomMeasure, SupportsSampling, SupportsMean):
    """The bootstrap random measure: a draw is the empirical measure of one replicate.

    A draw is an ``EmpiricalDistribution`` whose atoms are one replicate,
    ``replicate_size`` iid draws of the source on one level, with uniform
    weights. The bootstrap distribution of a statistic is ``evaluate`` of the
    statistic over the measure, or over the replicate law
    :class:`BootstrapReplicateDistribution` for a statistic that reads a
    dataset.

    **The event declaration.** One draw is declared as a law carrying the
    source's complete event declaration, so its component names and packaging
    are the source's. The measure's own declaration is distinct: a whole term
    under *component*.

    **Capabilities.** The measure samples, and its mean, the marginalized law
    ``E[D](A)`` of a draw ``D``, is the source itself, since each atom of a
    replicate is a draw of the source. A draw has no density, so the measure
    claims no random log-density.

    Parameters
    ----------
    component : str
        The component of the measure's event.
    source : Distribution
        The law a replicate draws from, which implements ``SupportsSampling``.
    replicate_size : int, optional
        The number of atoms of a drawn measure. It defaults to the atom count of
        an empirical source and is required otherwise.
    label : str, optional
        The law's label, ``BootstrapDistribution`` by default, which also labels
        each drawn empirical measure.
    level : str, optional
        The level the atoms of a drawn measure lie on, defaulting as for
        :class:`BootstrapReplicateDistribution`.
    event_spec : OutputSpec, optional
        A declaration of *component* that declares the type of one draw.

    Raises
    ------
    TypeError, ValueError
        As :class:`BootstrapReplicateDistribution` raises.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> data = EmpiricalDistribution(jnp.array([1.0, 2.0, 4.0]), component="y")
    >>> measure = BootstrapDistribution("boot", data)
    >>> draw = measure._sample(jax.random.PRNGKey(0))
    >>> (type(draw).__name__, draw.num_atoms, list(draw.event_spec.components))
    ('EmpiricalDistribution', 3, ['y'])
    """

    def __init__(
        self,
        component: str,
        source: SupportsSampling,
        replicate_size: int | None = None,
        *,
        label: str | None = None,
        level: str | None = None,
        event_spec: OutputSpec | None = None,
    ) -> None:
        law = _sampling_source(source)
        size = _replicate_size(law, replicate_size)
        on_level = _replicate_level(law, level)
        term = DistributionSpec(law.event_spec)
        if event_spec is not None:
            term = _whole_term_event(component, term, event_spec, _class_label(self))
        super().__init__(component, term, label=label)
        self._source = law
        self._replicate_size = size
        self._level = on_level

    @property
    def source(self) -> Distribution:
        """The law a replicate draws from."""
        return self._source

    @property
    def replicate_size(self) -> int:
        """The number of atoms of a drawn measure."""
        return self._replicate_size

    def _measure(self, raw: Any) -> EmpiricalDistribution:
        """The empirical measure of the replicate *raw*, the source's raw draws along one axis."""
        atoms = _batch_form(self.label, raw, self._level, self._source.event_spec.spec)
        return EmpiricalDistribution(atoms, label=self.label, event_spec=self._source.event_spec)

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Any:
        """Draw empirical measures, each of one replicate of the source.

        Parameters
        ----------
        key : PRNGKey
            The key of the draws.
        sample_shape : tuple of int, optional
            The batch axes of the measures.

        Returns
        -------
        EmpiricalDistribution or numpy.ndarray
            One measure for ``sample_shape=()``, and otherwise an object array
            of that shape holding one measure per position.
        """
        shape = tuple(sample_shape)
        raw = _replicates(self._source, key, shape, self._replicate_size)
        if not shape:
            return self._measure(raw)
        measures = np.empty(shape, dtype=object)
        for position in np.ndindex(*shape):
            measures[position] = self._measure(_replicate_at(raw, position))
        return measures

    def _mean(self) -> Distribution:
        """The marginalized law of a draw, which is the source."""
        return self._source

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The source law and the replicate size."""
        return [("source", repr(self._source)), ("replicate_size", repr(self._replicate_size))]


# ---------------------------------------------------------------------------
# The smoothing kernels
# ---------------------------------------------------------------------------


def _flat_centers(
    centers: ArrayLike | NumericRecordBatch, names: _BankNames = _KERNEL_NAMES
) -> Array:
    """The centers as an array ``(n, *event)``, with a record batch flattened to ``(n, d)``.

    Parameters
    ----------
    centers : ArrayLike or NumericRecordBatch
        One center per atom along the leading axis, or a batch of numeric records on one
        batch axis.
    names : _BankNames
        The names the caller passed the bank by, which the error messages use.

    Returns
    -------
    Array
        The centers in their own dtype when that is floating, and in the default floating
        dtype otherwise.

    Raises
    ------
    ValueError
        If the centers have no leading atom axis or no atoms.
    """
    if isinstance(centers, NumericRecordBatch):
        if len(centers.batch_shape) != 1:
            raise ValueError(
                f"{names.centers} must have a single batch axis, got batch shape "
                f"{centers.batch_shape}"
            )
        array = jnp.asarray(centers.to_vector())
    else:
        array = jnp.asarray(centers)
    if array.ndim == 0:
        raise ValueError(
            f"{names.centers} must have a leading axis indexing the {names.centers}, got a 0-d "
            f"array"
        )
    if array.shape[0] == 0:
        raise ValueError(
            f"{names.centers} must hold at least one {names.center}, got shape {array.shape}"
        )
    if not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype(jnp.result_type(float))
    return array


def _record_scales(
    scales: NumericRecord, fields: NumericRecordSpec | None, names: _BankNames = _KERNEL_NAMES
) -> Array:
    """A record of scales as one scale per coordinate, in the order of the centers' fields.

    Each field of *scales* is matched to the centers' field at the same leaf
    path and broadcast over that field's coordinates.

    Parameters
    ----------
    scales : NumericRecord
        One scale per field of the centers, each broadcastable over that field's shape.
    fields : NumericRecordSpec or None
        The element spec of record centers, or None for array centers.
    names : _BankNames
        The names the caller passed the bank by, which the error messages use.

    Returns
    -------
    Array
        A vector of length ``d``, which concatenates the fields' raveled scales.

    Raises
    ------
    ValueError
        If the centers are not records, the leaf paths of *scales* are not
        those of the centers, or a field's scale does not broadcast over that
        field's coordinates.
    """
    if fields is None:
        raise ValueError(
            f"{names.scales} is a record but {names.centers} is an array; pass {names.scales} "
            f"as an array"
        )
    expected, given = list(fields), list(scales)
    if set(given) != set(expected):
        missing = sorted(set(expected) - set(given))
        unexpected = sorted(set(given) - set(expected))
        differences = [f"missing {missing}"] * bool(missing) + [f"unexpected {unexpected}"] * bool(
            unexpected
        )
        raise ValueError(
            f"{names.scales} has fields {given} but {names.centers} has fields {expected} "
            f"({', '.join(differences)})"
        )
    shapes = fields.leaf_shapes
    blocks = []
    for path in expected:
        scale = jnp.asarray(scales.raw(path))
        try:
            block = jnp.broadcast_to(scale, shapes[path])
        except ValueError:
            raise ValueError(
                f"{names.scales} for field {path!r} has shape {scale.shape}, which does not "
                f"broadcast to the field's shape {shapes[path]}"
            ) from None
        blocks.append(jnp.reshape(block, -1))
    return jnp.concatenate(blocks)


def _flat_scales(
    scales: ArrayLike | NumericRecord,
    centers: Array,
    fields: NumericRecordSpec | None,
    names: _BankNames = _KERNEL_NAMES,
) -> Array:
    """The scales broadcast to the centers' shape ``(n, *event)``.

    Parameters
    ----------
    scales : ArrayLike or NumericRecord
        One scale, one per coordinate, or one per center and coordinate; a record of
        scales is matched to *fields* by :func:`_record_scales`.
    centers : Array
        The centers as :func:`_flat_centers` returns them.
    fields : NumericRecordSpec or None
        The element spec of record centers, or None for array centers.
    names : _BankNames
        The names the caller passed the bank by, which the error messages use.

    Returns
    -------
    Array
        The scales in the dtype of *centers*.

    Raises
    ------
    ValueError
        If a record of scales does not match the centers' fields, the scales do
        not broadcast against the centers, or a concrete scale is not positive.
    """
    if isinstance(scales, NumericRecord):
        array = _record_scales(scales, fields, names)
    else:
        array = jnp.asarray(scales)
    try:
        array = jnp.broadcast_to(array, centers.shape).astype(centers.dtype)
    except ValueError:
        raise ValueError(
            f"{names.scales} has shape {array.shape}, which does not broadcast against "
            f"{names.centers} of shape {centers.shape}; pass a scalar, one value per coordinate, or one per "
            f"{names.center} and coordinate"
        ) from None
    if not isinstance(array, jax.core.Tracer) and not bool(np.all(np.asarray(array) > 0)):
        raise ValueError(
            f"{names.scales} must be positive, got a minimum of {float(np.min(np.asarray(array)))}"
        )
    return array


class SmoothingKernel(ABC):
    """A bank of mean-zero kernel copies, one placed at each center.

    A smoothing kernel is a mean-zero density ``K`` over the coordinates of one
    atom. The copy placed at center ``c`` with scales ``h`` has the density
    ``x ↦ K((x − c) / h) / ∏ⱼ hⱼ``, where the product runs over the coordinates.
    :meth:`build_kernels` is the uniform constructor: it takes the centers and
    the scales and returns the bank, whatever the concrete kernel, so a kernel
    density estimate holds the kernel class and never reads kernel-specific
    parameters. The bank draws from indexed copies with :meth:`_sample` and
    scores a point under every copy with :meth:`_log_density`.

    Parameters
    ----------
    centers : ArrayLike or NumericRecordBatch
        One center per atom along the leading axis, ``(n, *event)``. A batch of
        numeric records is flattened to its coordinates, ``(n, d)``.
    scales : ArrayLike or NumericRecord
        The scales, broadcast against the centers: one scale, one per
        coordinate, or one per center and coordinate. A numeric record of
        scales requires record centers with the same fields, matched by leaf
        path. Each field's scale broadcasts over that field's coordinates, and
        the scales are flattened in the centers' field order.

    Attributes
    ----------
    variance : float
        The variance of ``K`` in each coordinate, so the copy with scale ``h``
        has the variance ``h² * variance`` in that coordinate.

    Raises
    ------
    ValueError
        If the centers have no atom axis, a record of scales does not have the
        centers' fields, or the scales do not broadcast against the centers or
        are not positive.
    """

    variance: ClassVar[float]

    def __init__(
        self, centers: ArrayLike | NumericRecordBatch, scales: ArrayLike | NumericRecord
    ) -> None:
        fields = centers.event_template if isinstance(centers, NumericRecordBatch) else None
        self._centers = _flat_centers(centers)
        self._scales = _flat_scales(scales, self._centers, fields)

    @classmethod
    @abstractmethod
    def build_kernels(
        cls, centers: ArrayLike | NumericRecordBatch, scales: ArrayLike | NumericRecord
    ) -> SmoothingKernel:
        """The bank of copies placed at *centers* with *scales*, one copy per center.

        Parameters
        ----------
        centers : ArrayLike or NumericRecordBatch
            One center per atom along the leading axis, as for :class:`SmoothingKernel`.
        scales : ArrayLike or NumericRecord
            The scales, broadcast against the centers as for :class:`SmoothingKernel`.

        Returns
        -------
        SmoothingKernel
            An instance of this kernel class holding the placed copies.

        Raises
        ------
        ValueError
            If the centers have no atom axis, a record of scales does not have
            the centers' fields, or the scales do not broadcast against the
            centers or are not positive.
        """

    @abstractmethod
    def _sample(self, key: PRNGKey, index: Array) -> Array:
        """One draw from each indexed copy.

        Parameters
        ----------
        key : PRNGKey
            The key of the draws.
        index : Array
            Integer indices of copies, of any shape.

        Returns
        -------
        Array
            An array ``(*index.shape, *event)`` that holds the draw from each indexed copy.
        """

    @abstractmethod
    def _log_density(self, x: Array) -> Array:
        """The log-density of each copy at *x*, the scale Jacobian included.

        Parameters
        ----------
        x : Array
            A point ``(*event)``, or points ``(*batch, *event)``.

        Returns
        -------
        Array
            An array ``(*batch, n)``: the log-density of every copy at each point.
        """

    # -- shared by the concrete kernels -----------------------------------------

    def _standardized(self, x: Array) -> Array:
        """``(x − c) / h`` for every copy, an array ``(*batch, n, *event)``."""
        x = jnp.asarray(x, dtype=self._centers.dtype)
        event_ndim = self._centers.ndim - 1
        return (jnp.expand_dims(x, x.ndim - event_ndim) - self._centers) / self._scales

    def _summed_over_event(self, per_coordinate: Array) -> Array:
        """*per_coordinate* ``(*batch, n, *event)`` summed over the event axes."""
        event_ndim = self._centers.ndim - 1
        if event_ndim == 0:
            return per_coordinate
        return jnp.sum(per_coordinate, axis=tuple(range(-event_ndim, 0)))

    def _placed(self, index: Array, standard: Array) -> Array:
        """Standard draws *standard* ``(*index.shape, *event)`` moved to the indexed copies."""
        index = jnp.asarray(index)
        return self._centers[index] + self._scales[index] * standard

    def _log_scale(self) -> Array:
        """``Σⱼ log hⱼ`` of every copy, an array ``(n,)``."""
        log_scales = jnp.log(self._scales)
        return log_scales.reshape(log_scales.shape[0], -1).sum(axis=-1)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_copies={self._centers.shape[0]})"


class GaussianKernel(SmoothingKernel):
    """The Gaussian smoothing kernel: a standard normal density in each coordinate.

    Its variance is one, so the copy with scale ``h`` is the normal law with the
    center as its mean and ``h`` as its standard deviation in each coordinate.
    Parameters and errors are those of :class:`SmoothingKernel`.
    """

    variance: ClassVar[float] = 1.0

    @classmethod
    def build_kernels(
        cls, centers: ArrayLike | NumericRecordBatch, scales: ArrayLike | NumericRecord
    ) -> GaussianKernel:
        """The bank of normal copies placed at *centers* with the standard deviations *scales*."""
        return cls(centers, scales)

    def _sample(self, key: PRNGKey, index: Array) -> Array:
        """One normal draw from each indexed copy, ``(*index.shape, *event)``."""
        index = jnp.asarray(index)
        shape = (*index.shape, *self._centers.shape[1:])
        return self._placed(index, jax.random.normal(key, shape, dtype=self._centers.dtype))

    def _log_density(self, x: Array) -> Array:
        """The normal log-density of each copy at *x*, ``(*batch, n)``."""
        u = self._standardized(x)
        per_coordinate = -0.5 * u**2 - 0.5 * math.log(2.0 * math.pi)
        return self._summed_over_event(per_coordinate) - self._log_scale()


class EpanechnikovKernel(SmoothingKernel):
    """The product Epanechnikov kernel: ``K(u) = ¾ (1 − u²)`` on ``[-1, 1]`` per coordinate.

    Its variance is one fifth in each coordinate. A copy has compact support, so
    its log-density is ``-inf`` beyond one scale from its center in any
    coordinate. Parameters and errors are those of :class:`SmoothingKernel`.
    """

    variance: ClassVar[float] = 0.2

    @classmethod
    def build_kernels(
        cls, centers: ArrayLike | NumericRecordBatch, scales: ArrayLike | NumericRecord
    ) -> EpanechnikovKernel:
        """The bank of Epanechnikov copies placed at *centers* with the half-widths *scales*."""
        return cls(centers, scales)

    def _sample(self, key: PRNGKey, index: Array) -> Array:
        """One draw from each indexed copy, ``(*index.shape, *event)``.

        A standard draw is ``2 sin(arcsin(2p − 1) / 3)`` for a uniform ``p``, which
        inverts the kernel's distribution function ``(2 + 3u − u³) / 4``.
        """
        index = jnp.asarray(index)
        shape = (*index.shape, *self._centers.shape[1:])
        p = jax.random.uniform(key, shape, dtype=self._centers.dtype)
        return self._placed(index, 2.0 * jnp.sin(jnp.arcsin(2.0 * p - 1.0) / 3.0))

    def _log_density(self, x: Array) -> Array:
        """The Epanechnikov log-density of each copy at *x*, ``(*batch, n)``."""
        u = self._standardized(x)
        inside = jnp.abs(u) < 1.0
        per_coordinate = jnp.where(
            inside, math.log(0.75) + jnp.log1p(-(jnp.where(inside, u, 0.0) ** 2)), -jnp.inf
        )
        return self._summed_over_event(per_coordinate) - self._log_scale()


# ---------------------------------------------------------------------------
# The kernel density estimate
# ---------------------------------------------------------------------------

#: The bandwidth selection rules, by name.
_BANDWIDTH_RULES = ("scott", "silverman")


def _kde_atoms(atoms: Any) -> tuple[Array | NumericRecordBatch, TermSpec]:
    """The atoms of a KDE with the term spec of one draw.

    An array's leading axis indexes its atoms. A draw takes the atoms' shape or
    record, with a floating dtype and the real line as each leaf's support,
    since a smoothed draw is any real value.

    Parameters
    ----------
    atoms : Any
        The atoms the constructor received: a numeric array whose leading axis indexes
        them, or a ``NumericRecordBatch``.

    Returns
    -------
    stored : Array or NumericRecordBatch
        The atoms to store, which are an array cast to a floating dtype or the record batch
        as it is.
    atom_spec : TermSpec
        The term spec of one draw.

    Raises
    ------
    TypeError
        If *atoms* is neither a numeric array nor a ``NumericRecordBatch``.
    ValueError
        If an array of atoms has no leading axis or holds no atom.
    """
    if isinstance(atoms, NumericRecordBatch):
        spec = atoms.element_spec.map(
            lambda leaf: NumericArraySpec(leaf.shape, _floating(leaf.dtype), real)
        )
        return atoms, spec
    if isinstance(atoms, NumericRecord) or not hasattr(atoms, "shape"):
        raise TypeError(
            f"atoms must be an array (one atom per row) or a NumericRecordBatch, got "
            f"{type(atoms).__name__}"
        )
    values = jnp.asarray(atoms)
    if not jnp.issubdtype(values.dtype, jnp.number):
        raise TypeError(f"atoms must be numeric, got dtype {values.dtype}")
    if values.ndim == 0:
        raise ValueError("atoms must have a leading axis indexing the atoms, got a 0-d array")
    if values.shape[0] == 0:
        raise ValueError(f"atoms must hold at least one atom, got shape {values.shape}")
    values = values.astype(_floating(values.dtype))
    return values, NumericArraySpec(tuple(values.shape[1:]), values.dtype, real)


def _floating(dtype: Any) -> Any:
    """*dtype* when it is floating, else the default floating dtype."""
    if dtype is not None and jnp.issubdtype(dtype, jnp.floating):
        return dtype
    return jnp.result_type(float)


def _kde_weights(weights: Any, count: int) -> Weights:
    """The weights of *count* atoms, uniform when *weights* is ``None``."""
    if weights is None:
        return Weights.uniform(count)
    if isinstance(weights, Weights):
        return Weights(n=count, weights=weights)
    return Weights(n=count, weights=jnp.asarray(weights))


def _selected_bandwidth(rule: str, centers: Array, weights: Weights) -> Array:
    """The bandwidth *rule* selects for *centers*, one scale per coordinate.

    Scott's rule is ``hⱼ = n_eff^(-1/(d+4)) σⱼ`` and Silverman's is
    ``hⱼ = (4/(d+2))^(1/(d+4)) n_eff^(-1/(d+4)) σⱼ``, with ``σⱼ`` the weighted
    standard deviation of coordinate ``j``, ``d`` the number of coordinates, and
    ``n_eff = (Σwᵢ)²/Σwᵢ²`` Kish's effective sample size.

    Parameters
    ----------
    rule : str
        The name of the rule, ``"scott"`` or ``"silverman"``.
    centers : Array
        The centers ``(n, *event)``, as :func:`_flat_centers` returns them.
    weights : Weights
        The atoms' weights, which give the weighted spread and the effective sample size.

    Returns
    -------
    Array
        An array of the centers' event shape.

    Raises
    ------
    ValueError
        If *rule* names no rule, or a coordinate's atoms have no spread, so the
        rule selects a zero bandwidth.
    """
    if rule not in _BANDWIDTH_RULES:
        raise ValueError(unknown_names("bandwidth rule", [rule], _BANDWIDTH_RULES))
    probabilities = None if weights.is_uniform else weights.normalized
    mean = weighted_mean(probabilities, centers)
    spread = jnp.sqrt(weighted_mean(probabilities, (centers - mean) ** 2))
    d = math.prod(centers.shape[1:])
    factor = weights.effective_sample_size ** (-1.0 / (d + 4))
    if rule == "silverman":
        factor = factor * (4.0 / (d + 2)) ** (1.0 / (d + 4))
    bandwidth = factor * spread
    if not isinstance(bandwidth, jax.core.Tracer) and not bool(np.all(np.asarray(spread) > 0)):
        raise ValueError(
            f"bandwidth rule {rule!r} gives a zero bandwidth because the atoms do not vary in "
            f"some coordinate; pass bandwidth values instead"
        )
    return bandwidth


class KDEDistribution(
    Distribution,
    SupportsSampling,
    SupportsLogProb,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
):
    """The kernel density estimate of weighted atoms under a smoothing kernel.

    The law is the weighted mixture ``Σᵢ wᵢ h⁻ᵈ K((x − xᵢ)/h)`` of the kernel's
    copies placed at the atoms, with the bandwidth ``h`` as their scales. The
    kernel class builds the bank of copies through
    :meth:`SmoothingKernel.build_kernels`, so the estimate reads no
    kernel-specific parameter. Records enter through their flat vectors: a
    ``NumericRecordBatch`` of atoms is flattened to its coordinates, and a
    ``NumericRecord`` of scales matches its fields by path.

    **The bandwidth.** A value is one scale, one per coordinate, or one per atom
    and coordinate. A string names a selection rule, ``"scott"`` or
    ``"silverman"``, and ``None`` selects Scott's rule. Both rules count Kish's
    effective sample size, so they stay sensible under importance weights.

    **The event declaration.** Record atoms expose their fields, and array atoms
    form a whole-term event under *component*. An *event_spec* names the
    components, as for ``EmpiricalDistribution``. Every leaf is declared
    floating, on the real line.

    **Capabilities.**

    ==========================  ===================================================
    capability                  realized by
    ==========================  ===================================================
    ``_sample``                 an atom drawn by weight, then a draw from its copy
    ``_log_prob``               the weighted log-sum-exp of the copies' densities
    ``_mean``                   the weighted mean of the atoms
    ``_variance``               the atoms' weighted variance plus the mean squared
                                scale times the kernel's variance
    ``_cov``                    the atoms' weighted covariance plus that diagonal,
                                a ``DenseLinOp`` over the flat coordinates
    ==========================  ===================================================

    Every capability is exact for the law. A record event's draw and moments
    are nested mappings of raw leaves, and its density reads a record value in
    raw form or a batch of records.

    Parameters
    ----------
    atoms : Array or NumericRecordBatch
        The centers of the copies, along a leading axis of atoms.
    bandwidth : ArrayLike, NumericRecord, or str, optional
        The scales of the copies, or the name of a selection rule; ``None``
        selects Scott's rule.
    weights : Array or Weights, optional
        Nonnegative weights with a positive sum, one per atom; uniform when
        omitted.
    kernel : type of SmoothingKernel
        The smoothing kernel, ``GaussianKernel`` by default.
    component : str, optional
        The component of a whole-term event, required for array atoms unless
        *event_spec* names it, and refused for record atoms.
    label : str, optional
        The law's label, ``KDEDistribution`` by default.
    event_spec : OutputSpec, optional
        The declaration of one draw, completed with the atoms' spec.

    Raises
    ------
    TypeError
        If *atoms* is neither a numeric array nor a ``NumericRecordBatch``,
        *kernel* is not a ``SmoothingKernel`` class, *component* is missing for
        array atoms or given for record atoms, *label* is not a non-empty
        string, *event_spec* is not an ``OutputSpec``, or *event_spec* exposes a
        record for array atoms.
    ValueError
        If the atoms hold none or have no leading axis, the weights are invalid,
        *bandwidth* names no rule or a rule selects a zero scale, the scales do not
        broadcast against the atoms or are not positive, or *event_spec* names
        another component or declares a type that does not unify with the atoms'.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> kde = KDEDistribution(jnp.array([0.0, 1.0, 3.0]), 0.5, component="x")
    >>> round(float(kde._mean()), 4)
    1.3333
    >>> round(float(kde._variance()), 4)  # the atoms' variance 14/9, plus 0.5 ** 2
    1.8056
    """

    #: The constructor takes the atoms first and the component as a keyword.
    _repr_component: ClassVar[str | None] = "keyword"

    def __init__(
        self,
        atoms: Array | NumericRecordBatch,
        bandwidth: ArrayLike | NumericRecord | str | None = None,
        weights: Array | Weights | None = None,
        kernel: type[SmoothingKernel] = GaussianKernel,
        *,
        component: str | None = None,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ) -> None:
        if isinstance(atoms, str):
            raise TypeError(_label_given_first("KDEDistribution", "atoms", atoms))
        if not (isinstance(kernel, type) and issubclass(kernel, SmoothingKernel)):
            raise TypeError(
                f"kernel must be a SmoothingKernel subclass such as GaussianKernel, got {kernel!r}"
            )
        stored, atom_spec = _kde_atoms(atoms)
        owner = _class_label(self)
        declared = _atoms_declaration(atom_spec, component, event_spec, owner)
        super().__init__(_constructor_label(self, label, owner), declared)
        centers = _flat_centers(stored, _KDE_NAMES)
        atom_weights = _kde_weights(weights, centers.shape[0])
        if bandwidth is None or isinstance(bandwidth, str):
            scales = _selected_bandwidth(bandwidth or "scott", centers, atom_weights)
        else:
            scales = bandwidth
            # Checked here first, so that a bad bandwidth is reported in the KDE's terms.
            fields = stored.event_template if isinstance(stored, NumericRecordBatch) else None
            _flat_scales(scales, centers, fields, _KDE_NAMES)
        bank = kernel.build_kernels(stored, scales)
        object.__setattr__(self, "_atoms", stored)
        object.__setattr__(self, "_kernel", kernel)
        object.__setattr__(self, "_bank", bank)
        object.__setattr__(self, "_w", atom_weights)
        probabilities = None if atom_weights.is_uniform else atom_weights.normalized
        object.__setattr__(self, "_p", probabilities)

    @property
    def num_atoms(self) -> int:
        """The number of atoms, one copy of the kernel placed at each."""
        return int(self._bank._centers.shape[0])

    # -- the raw form of a draw ---------------------------------------------------

    def _record_spec(self) -> RecordSpec | None:
        """The record one draw is, or ``None`` for an array event."""
        spec = self.event_spec.spec
        return spec if isinstance(spec, RecordSpec) else None

    def _unflattened(self, flat: Array) -> Any:
        """*flat*, the coordinates ``(*batch, d)`` of record values, as their raw form.

        An array event's values are returned as they are.
        """
        record = self._record_spec()
        if record is None:
            return flat
        lead = flat.shape[:-1]
        leaves, offset = {}, 0
        for path, shape in record.leaf_shapes.items():
            width = math.prod(shape)
            leaves[path] = jnp.reshape(flat[..., offset : offset + width], (*lead, *shape))
            offset += width
        return _unflatten_paths(leaves)

    def _flattened(self, value: Any) -> Array:
        """*value* in the centers' layout: ``(*batch, *event)``, or ``(*batch, d)`` for records.

        A record value's leaves are read by the KDE's leaf paths, so a batch of
        records whose fields are declared in another order is read correctly.

        Parameters
        ----------
        value : Any
            A value or a batch of values: an array for an array event, and a record in raw
            form or a batch of records for a record event.

        Returns
        -------
        Array
            The coordinates in the dtype of the centers.

        Raises
        ------
        TypeError
            If a record event's value is neither a record in raw form nor a
            batch of records.
        """
        record = self._record_spec()
        if record is None:
            return jnp.asarray(value, dtype=self._bank._centers.dtype)
        raw = _raw_record(value)
        if not isinstance(raw, dict):
            raise TypeError(
                f"{self.label!r} draws records, so a value must be a Record, a RecordBatch, or a "
                f"mapping of its fields, got {_type_name(value)}"
            )
        blocks = []
        for path, shape in record.leaf_shapes.items():
            leaf = raw
            for segment in path.split(_PATH_SEP):
                leaf = leaf[segment]
            leaf = jnp.asarray(leaf, dtype=self._bank._centers.dtype)
            lead = leaf.shape[: leaf.ndim - len(shape)]
            blocks.append(jnp.reshape(leaf, (*lead, math.prod(shape))))
        return jnp.concatenate(blocks, axis=-1)

    def _flat_coordinates(self, values: Array) -> Array:
        """*values* ``(n, *event)`` in the centers' layout, raveled to ``(n, d)``."""
        return jnp.reshape(values, (values.shape[0], -1))

    # -- the capabilities -----------------------------------------------------------

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Any:
        """Draw an atom by weight, then a draw from the copy placed at it.

        Parameters
        ----------
        key : PRNGKey
            The key of the draws, split into a key that picks the atoms and a key that draws
            from their copies.
        sample_shape : tuple of int, optional
            The batch axes of the draws.

        Returns
        -------
        Any
            One draw in its raw form for ``sample_shape=()``: an array for an
            array event and the nested mapping of raw leaves for a record event.
            A non-empty shape prepends its axes.
        """
        pick, draw = jax.random.split(key)
        index = weighted_choice(pick, self.num_atoms, weights=self._p, shape=tuple(sample_shape))
        return self._unflattened(self._bank._sample(draw, index))

    def _log_prob(self, value: Any) -> Array:
        """``log Σᵢ wᵢ Kₕ(x − xᵢ)``, the weighted log-sum-exp of the copies' log-densities.

        Parameters
        ----------
        value : Any
            A value or a batch of values along leading axes: an array for an array event,
            and a record in raw form or a batch of records for a record event.

        Returns
        -------
        Array
            One log-density per value, shaped like the value's leading axes.
        """
        log_weights = jnp.log(self._w.normalized)
        densities = self._bank._log_density(self._flattened(value))
        return jax.scipy.special.logsumexp(densities + log_weights, axis=-1)

    def _unnormalized_log_prob(self, value: Any) -> Array:
        """The normalized log-density, which is also a log-density up to a constant."""
        return self._log_prob(value)

    def _mean(self) -> Any:
        """The weighted mean of the atoms, in the raw form of one draw."""
        return self._unflattened(weighted_mean(self._p, self._bank._centers))

    def _variance(self) -> Any:
        """``Var_w(xᵢ) + κ E_w[hᵢ²]`` per coordinate, with ``κ`` the kernel's variance."""
        centers, scales = self._bank._centers, self._bank._scales
        mean = weighted_mean(self._p, centers)
        spread = weighted_mean(self._p, (centers - mean) ** 2)
        smoothing = self._kernel.variance * weighted_mean(self._p, scales**2)
        return self._unflattened(spread + smoothing)

    def _cov(self) -> LinOp:
        """The atoms' weighted covariance plus ``κ E_w[hᵢ²]`` on the diagonal, over the flat coordinates."""
        centers = self._flat_coordinates(self._bank._centers)
        scales = self._flat_coordinates(self._bank._scales)
        smoothing = self._kernel.variance * weighted_mean(self._p, scales**2)
        return DenseLinOp(weighted_covariance(self._p, centers) + jnp.diag(smoothing))

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The atoms, the weights when nonuniform, and the kernel."""
        fields = [("atoms", format_value(self._atoms))]
        if self._p is not None:
            fields.append(("weights", format_value(self._p)))
        return [*fields, ("kernel", self._kernel.__name__)]
