"""The program-defined families: laws that a backend program or a user's density defines (VII.9).

A program-defined model exposes the law its program defines, in the kind that
law has, and its variable names determine its output components.

Provides:
  - ``StanModel`` – the kernel from a Stan program's data-block entries to its
    unnormalized posterior over the parameter record, through BridgeStan; a
    construction that binds every entry returns the posterior itself;
  - ``PyMCModel`` – the joint law a PyMC model-building function defines over
    its free variables, the parameters and the observed variables alike, and a
    kernel over the arguments no observed variable receives;
  - ``UnnormalizedDistribution`` – the law of a user-supplied unnormalized
    log-density over a declared event.

Each claims its density as its program supplies it, and ``condition_on``
normalizes a law that claims only an unnormalized one through the
inference-method registry.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..core._immutable import transient_memo
from ..core._record_spec import NumericRecordSpec, RecordSpec
from ..core._repr import format_names, format_value
from ..core._spec_base import NumericArraySpec
from ..core._specs import OpaqueSpec, OutputSpec
from ..core.record import Record
from ..custom_types import Array, ArrayLike
from ..distributions._capabilities import (
    SupportsConditionalLogProb,
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    _capability_subclass,
)
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution

__all__ = ["PyMCModel", "StanModel", "UnnormalizedDistribution"]


#: Make arguments for a Stan model library: TBB without its malloc proxy.
_BRIDGESTAN_MAKE_ARGS: tuple[str, ...] = ("TBB_LIBRARIES=tbb",)


def _given_values(owner: str, given: Any, kwargs: Mapping[str, Any], slots: Any) -> dict[str, Any]:
    """The values *given* and *kwargs* bind, by slot name.

    Raises
    ------
    KeyError
        If a name is not one of *slots*.
    """
    values = {**dict(given.children if isinstance(given, Record) else given), **kwargs}
    unknown = sorted(set(values) - set(slots))
    if unknown:
        raise KeyError(f"{unknown} are not given slots of {owner!r}")
    return values


# ---------------------------------------------------------------------------
# UnnormalizedDistribution
# ---------------------------------------------------------------------------


class UnnormalizedDistribution(Distribution, SupportsUnnormalizedLogProb):
    """The law of a user-supplied unnormalized log-density over a declared event.

    It claims ``SupportsUnnormalizedLogProb`` alone, so it is unnormalized, and
    ``sample``, ``convert``, and ``condition_on`` normalize it through the
    inference-method registry.

    Parameters
    ----------
    name : str
        The law's label.
    log_density : callable
        ``log_density(value)``, the log-density of a draw of the event up to an
        additive constant.
    event_spec : OutputSpec
        The declaration of one draw.

    Raises
    ------
    TypeError
        If *log_density* is not callable, or as ``Distribution`` raises.
    """

    def __init__(
        self, name: str, log_density: Callable[[Any], Array], event_spec: OutputSpec
    ) -> None:
        if not callable(log_density):
            raise TypeError(
                f"UnnormalizedDistribution needs a callable log_density; got "
                f"{type(log_density).__name__}"
            )
        super().__init__(name, event_spec)
        self._log_density = log_density

    def _unnormalized_log_prob(self, value: Any) -> Array:
        """The user's log-density at *value*, known up to an additive constant."""
        return jnp.asarray(self._log_density(value))

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The log-density, by its name."""
        return [("log_density", format_value(self._log_density))]


# ---------------------------------------------------------------------------
# Stan programs
# ---------------------------------------------------------------------------


def _to_f64(x: ArrayLike) -> np.ndarray:
    """*x* as the contiguous ``float64`` ndarray BridgeStan's ctypes interface requires."""
    return np.asarray(x, dtype=np.float64)


class _StanBlock(NamedTuple):
    """One Stan parameter block recovered from BridgeStan's flattened names."""

    name: str
    shape: tuple[int, ...]  # () for a scalar parameter
    # advanced-index arrays mapping a ``shape``-shaped array onto the block's
    # flat slice in BridgeStan's order; () for a scalar.
    gather: tuple[Array, ...]


def _param_blocks(flat_names: Sequence[str]) -> tuple[_StanBlock, ...]:
    """Group BridgeStan's dot-flattened parameter names into typed blocks.

    BridgeStan reports one flattened name per scalar (``mu``, ``theta.1``,
    ``L.1.1``, dot-separated and 1-indexed), each block's scalars consecutive.
    Placing each scalar by its parsed index makes the round trip exact for
    matrices, which Stan flattens column-major, and for arrays alike.
    """
    blocks: list[_StanBlock] = []
    i, n = 0, len(flat_names)
    while i < n:
        block = flat_names[i].split(".", 1)[0]
        idx_tuples: list[tuple[int, ...]] = []
        while i < n and flat_names[i].split(".", 1)[0] == block:
            idx_tuples.append(tuple(int(x) - 1 for x in flat_names[i].split(".")[1:]))
            i += 1
        if idx_tuples == [()]:
            blocks.append(_StanBlock(block, (), ()))
            continue
        ndim = len(idx_tuples[0])
        shape = tuple(max(t[d] for t in idx_tuples) + 1 for d in range(ndim))
        gather = tuple(jnp.asarray([t[d] for t in idx_tuples]) for d in range(ndim))
        blocks.append(_StanBlock(block, shape, gather))
    return tuple(blocks)


def _pack_block_params(
    owner: str, blocks: tuple[_StanBlock, ...], values: Mapping[str, Any]
) -> Array:
    """BridgeStan's flat parameter vector from one array per block.

    Raises
    ------
    TypeError
        If a block is missing, a name is not a block, or a value does not have
        its block's shape.
    """
    expected = [b.name for b in blocks]
    missing = [name for name in expected if name not in values]
    extra = [k for k in values if k not in set(expected)]
    if missing or extra:
        detail = [f"missing {missing}"] * bool(missing) + [f"unexpected {extra}"] * bool(extra)
        raise TypeError(
            f"{owner}: the keyword form expects the Stan parameter blocks "
            f"{tuple(expected)}: {'; '.join(detail)}."
        )
    out: list[Array] = []
    for b in blocks:
        arr = jnp.asarray(values[b.name])
        if tuple(arr.shape) != b.shape:
            raise TypeError(
                f"{owner}: parameter {b.name!r} expects shape {b.shape}, got {tuple(arr.shape)}."
            )
        out.append(jnp.reshape(arr, (1,)) if b.shape == () else arr[b.gather])
    return jnp.concatenate(out) if out else jnp.zeros((0,))


_STAN_BLOCK = re.compile(
    r"\b(functions|transformed\s+data|data|transformed\s+parameters|parameters|model|"
    r"generated\s+quantities)\s*\{"
)
_DECLARATION = re.compile(
    r"^(?:array\s*\[(?P<array>[^\]]*)\]\s*)?(?P<type>[A-Za-z_]\w*)\s*(?:\[(?P<sizes>[^\]]*)\])?\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*(?:\[(?P<old>[^\]]*)\])?$"
)
_SCALAR_TYPES = frozenset({"int", "real", "complex"})
_VECTOR_TYPES = frozenset(
    {
        "vector",
        "row_vector",
        "simplex",
        "unit_vector",
        "ordered",
        "positive_ordered",
        "sum_to_zero_vector",
        "complex_vector",
        "complex_row_vector",
    }
)
_MATRIX_TYPES = frozenset(
    {
        "matrix",
        "complex_matrix",
        "cov_matrix",
        "corr_matrix",
        "cholesky_factor_corr",
        "cholesky_factor_cov",
        "column_stochastic_matrix",
        "row_stochastic_matrix",
        "sum_to_zero_matrix",
    }
)


def _stan_blocks(source: str) -> dict[str, str]:
    """The bodies of a Stan program's top-level blocks, keyed by block name.

    Raises
    ------
    ValueError
        If a block's braces do not balance.
    """
    text = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*|#[^\n]*", " ", text)
    blocks: dict[str, str] = {}
    position = 0
    while (match := _STAN_BLOCK.search(text, position)) is not None:
        depth, index = 1, match.end()
        while depth and index < len(text):
            depth += {"{": 1, "}": -1}.get(text[index], 0)
            index += 1
        if depth:
            raise ValueError(f"the braces of the Stan block {match.group(1)!r} do not balance")
        blocks[" ".join(match.group(1).split())] = text[match.end() : index - 1]
        position = index
    return blocks


def _without_bounds(statement: str) -> str:
    """*statement* without its ``<...>`` bounds, offsets, and multipliers."""
    kept: list[str] = []
    depth = 0
    for char in statement:
        if char == "<":
            depth += 1
        elif char == ">" and depth:
            depth -= 1
        elif not depth:
            kept.append(char)
    return "".join(kept)


def _sizes(expression: str | None) -> list[str]:
    return [part.strip() for part in expression.split(",")] if expression else []


def _declared_variables(block: str) -> list[tuple[str, tuple[str, ...]]]:
    """Each variable a data or parameters block declares, with its size expressions, in order.

    Raises
    ------
    ValueError
        If a statement is not a declaration of a known Stan type.
    """
    variables: list[tuple[str, tuple[str, ...]]] = []
    for statement in block.split(";"):
        statement = " ".join(_without_bounds(statement).split())
        if not statement:
            continue
        match = _DECLARATION.match(statement)
        if match is None:
            raise ValueError(f"cannot read the Stan declaration {statement!r}")
        kind, sizes = match["type"], _sizes(match["sizes"])
        if kind in _SCALAR_TYPES:
            shape: list[str] = []
        elif kind in _VECTOR_TYPES:
            shape = sizes[:1]
        elif kind in _MATRIX_TYPES:
            shape = sizes * 2 if len(sizes) == 1 else sizes
        else:
            raise ValueError(f"the Stan type {kind!r} of {match['name']!r} is not read")
        variables.append((match["name"], (*_sizes(match["array"]), *_sizes(match["old"]), *shape)))
    return variables


def _dimension(expression: str, name: str, axis: int, data: Mapping[str, Any]) -> int | str:
    """A size of *name*: a literal, a scalar data entry's value, or else a symbolic dimension.

    A size that names a data entry is the dimension of that name until the
    entry is bound.
    """
    if expression.isdigit():
        return int(expression)
    if re.fullmatch(r"[A-Za-z_]\w*", expression):
        value = data.get(expression)
        return int(value) if value is not None and np.ndim(value) == 0 else expression
    return f"{name}_{axis}"


@dataclass(frozen=True)
class _StanProgram:
    """A Stan program's file, its data-block entries, and its parameters' size expressions."""

    stan_file: str
    data_entries: tuple[str, ...]
    parameters: tuple[tuple[str, tuple[str, ...]], ...]

    @classmethod
    def read(cls, stan_file: str) -> _StanProgram:
        """The program in *stan_file*.

        Raises
        ------
        ValueError
            If the program declares no parameters, or a declaration cannot be
            read.
        """
        blocks = _stan_blocks(Path(stan_file).read_text())
        parameters = tuple(_declared_variables(blocks.get("parameters", "")))
        if not parameters:
            raise ValueError(f"the Stan program {stan_file} declares no parameters")
        entries = tuple(name for name, _ in _declared_variables(blocks.get("data", "")))
        return cls(str(stan_file), entries, parameters)

    def parameter_record(self, data: Mapping[str, Any]) -> RecordSpec:
        """The parameter record, each size a scalar entry of *data* names bound to its value."""
        return RecordSpec(
            {
                name: NumericArraySpec(
                    tuple(_dimension(size, name, axis, data) for axis, size in enumerate(sizes))
                )
                for name, sizes in self.parameters
            }
        )


class _StanPosterior(Distribution, SupportsUnnormalizedLogProb):
    """The unnormalized posterior of a Stan program at a value of every data-block entry.

    Its event is the parameter record, and its unnormalized log-density is
    BridgeStan's log density in the constrained parameterization without the
    Jacobian, so it claims no normalized capability. The Stan methods of the
    inference-method registry read its program and data.

    Parameters
    ----------
    name : str
        The law's label.
    program : _StanProgram
        The program.
    data : Mapping[str, Any]
        A value of every data-block entry.
    """

    #: The BridgeStan model, built on first use, is not state.
    _transient_state = ("_memo",)

    def __init__(self, name: str, program: _StanProgram, data: Mapping[str, Any]) -> None:
        super().__init__(name, OutputSpec(program.parameter_record(data)))
        self._program = program
        self._data = dict(data)

    @property
    def stan_file(self) -> str:
        """The Stan program's file."""
        return self._program.stan_file

    @property
    def data(self) -> Mapping[str, Any]:
        """The value of every data-block entry."""
        return MappingProxyType(self._data)

    def _bridgestan_model(self) -> Any:
        """The BridgeStan model of the program at its data, built once.

        Raises
        ------
        ImportError
            If ``bridgestan`` is not installed.
        """
        memo = transient_memo(self)
        if "bridgestan" not in memo:
            try:
                import bridgestan
            except ImportError as e:
                raise ImportError(
                    "bridgestan is required for a Stan model's density. Install it with: "
                    "pip install bridgestan"
                ) from e
            # Stan links TBB's malloc proxy by default, which replaces the process
            # allocator and makes JAX's zero-size allocations fail afterwards, so the
            # model library links TBB without it.
            memo["bridgestan"] = bridgestan.StanModel(
                self.stan_file,
                data={k: _to_numpy(v) for k, v in self._data.items()} or None,
                make_args=list(_BRIDGESTAN_MAKE_ARGS),
            )
        return memo["bridgestan"]

    def _blocks(self) -> tuple[_StanBlock, ...]:
        """The constrained parameter blocks, from BridgeStan's ``param_names()``."""
        return _param_blocks(self._bridgestan_model().param_names())

    def _pack_value(self, **field_kwargs: Any) -> Array:
        """The keyword form: one array per parameter block, as BridgeStan's flat vector."""
        return _pack_block_params(type(self).__name__, self._blocks(), field_kwargs)

    def _flat(self, value: Any) -> np.ndarray:
        """*value*, a record of the parameter blocks or BridgeStan's flat vector, as the vector."""
        if isinstance(value, (Record, Mapping)):
            fields = value.children if isinstance(value, Record) else value
            value = _pack_block_params(type(self).__name__, self._blocks(), dict(fields))
        return _to_f64(value)

    def _unnormalized_log_prob(self, value: Any) -> Array:
        """BridgeStan's log density at *value*, in the constrained space, without the Jacobian."""
        model = self._bridgestan_model()
        unconstrained = model.param_unconstrain(self._flat(value))
        return jnp.asarray(model.log_density(unconstrained, jacobian=False))

    def as_unconstrained_distribution(self) -> _UnconstrainedStanView:
        """The posterior in the unconstrained parameterization, whose density has the Jacobian."""
        return _UnconstrainedStanView(self)

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The Stan file and the data-block entries bound."""
        return [("stan_file", repr(self.stan_file)), ("data", format_names(sorted(self._data)))]


class _UnconstrainedStanView(Distribution, SupportsUnnormalizedLogProb):
    """A Stan posterior in the unconstrained parameterization.

    Its unnormalized log-density is BridgeStan's log density with the Jacobian
    of the constraining transform, and its event is the record of the
    unconstrained parameter blocks.
    """

    def __init__(self, posterior: _StanPosterior) -> None:
        self._posterior = posterior
        self._blocks = _param_blocks(posterior._bridgestan_model().param_unc_names())
        super().__init__(
            f"{posterior.name}_unconstrained",
            OutputSpec(NumericRecordSpec({b.name: b.shape for b in self._blocks})),
        )

    def _pack_value(self, **field_kwargs: Any) -> Array:
        """The keyword form: one array per unconstrained block, as BridgeStan's flat vector."""
        return _pack_block_params(type(self).__name__, self._blocks, field_kwargs)

    def _unnormalized_log_prob(self, value: Any) -> Array:
        """BridgeStan's log density at the unconstrained *value*, with the Jacobian."""
        if isinstance(value, (Record, Mapping)):
            fields = value.children if isinstance(value, Record) else value
            value = _pack_block_params(type(self).__name__, self._blocks, dict(fields))
        return jnp.asarray(self._posterior._bridgestan_model().log_density(_to_f64(value)))

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The posterior this law reparameterizes."""
        return [("posterior", repr(self._posterior))]


class _StanModelMeta(type(ConditionalDistribution)):
    """The metaclass of ``StanModel``: binding every data-block entry returns the posterior."""

    def __call__(cls, name: str, stan_file: str, *, data: Mapping[str, Any] | None = None) -> Any:
        program = _StanProgram.read(stan_file)
        bound = dict(data or {})
        unknown = sorted(set(bound) - set(program.data_entries))
        if unknown:
            raise KeyError(f"{unknown} are not data-block entries of {stan_file}")
        if set(program.data_entries) <= set(bound):
            return _StanPosterior(name, program, bound)
        return super().__call__(name, stan_file, data=data)


class StanModel(
    ConditionalDistribution, SupportsConditionalUnnormalizedLogProb, metaclass=_StanModelMeta
):
    """The kernel from a Stan program's data-block entries to its unnormalized posterior.

    The given slots are the data-block entries *data* leaves unbound, which the
    program does not divide into sizes, covariates, and observations, and the
    event is the parameter record, whose sizes that name an unbound entry stay
    symbolic. It claims ``SupportsConditionalUnnormalizedLogProb`` alone:
    binding every entry yields the unnormalized posterior, whose density is
    BridgeStan's in the constrained parameterization without the Jacobian, and
    ``condition_on`` normalizes it with a method such as Stan's NUTS. The
    program is read at construction and compiled by BridgeStan when a density
    is first evaluated.

    Parameters
    ----------
    name : str
        The kernel's label.
    stan_file : str
        Path to a ``.stan`` file.
    data : Mapping[str, Any], optional
        Values of some data-block entries, bound at construction. A
        construction that binds every entry returns the posterior, a
        ``Distribution``.

    Raises
    ------
    KeyError
        If *data* names an entry the data block does not declare.
    ValueError
        If the program declares no parameters, or a declaration cannot be read.
    """

    def __init__(self, name: str, stan_file: str, *, data: Mapping[str, Any] | None = None) -> None:
        program = _StanProgram.read(stan_file)
        bound = dict(data or {})
        super().__init__(
            name,
            {entry: OpaqueSpec() for entry in program.data_entries if entry not in bound},
            OutputSpec(program.parameter_record(bound)),
        )
        object.__setattr__(self, "_program", program)
        object.__setattr__(self, "_data", bound)

    @property
    def stan_file(self) -> str:
        """The Stan program's file."""
        return self._program.stan_file

    @property
    def data(self) -> Mapping[str, Any]:
        """The data-block entries bound at construction or by currying."""
        return MappingProxyType(self._data)

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **kwargs: Any
    ) -> Distribution | ConditionalDistribution:
        """The posterior at a value of every data-block entry, or the kernel over the rest.

        Raises
        ------
        KeyError
            If a name is not an unbound data-block entry.
        """
        values = _given_values(self.name, given, kwargs, self.given_spec)
        return StanModel(self.name, self.stan_file, data={**self._data, **values})

    def _conditional_unnormalized_log_prob(
        self, given: Record | Mapping[str, Any], value: Any
    ) -> Array:
        """The unnormalized posterior density at *value*, the data bound to *given*.

        Raises
        ------
        KeyError
            If *given* leaves a data-block entry unbound.
        """
        law = self._condition_on(given)
        if isinstance(law, ConditionalDistribution):
            raise KeyError(f"{self.name!r} needs a value for every data-block entry")
        return law._unnormalized_log_prob(value)

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The Stan file; the given slots are the data-block entries left unbound."""
        return [("stan_file", repr(self.stan_file))]


# ---------------------------------------------------------------------------
# PyMC programs
# ---------------------------------------------------------------------------


def _to_numpy(value: Any) -> Any:
    """*value* as a numpy array when it is array-like, for PyMC's tensor backend."""
    return np.asarray(value) if hasattr(value, "__array__") else value


#: The PyMC distributions whose density does not integrate to one.
_IMPROPER_OPS = frozenset({"FlatRV", "HalfFlatRV"})


class _PyMCProgram:
    """A PyMC model-building function with some arguments bound, and the variables its build has.

    The function is built with its arguments' defaults. An argument whose
    default is ``None`` and for which that build has a free variable of the
    same name is an observed variable, since passing ``None`` as its
    ``observed`` value leaves it free; every other argument is a given slot.

    Raises
    ------
    TypeError
        If the function cannot be built with its defaults and the bound values.
    """

    def __init__(self, model_fn: Callable[..., Any], bound: Mapping[str, Any] | None = None):
        self.model_fn = model_fn
        self.bound = dict(bound or {})
        arguments = [
            parameter
            for parameter in inspect.signature(model_fn).parameters.values()
            if parameter.name not in self.bound
        ]
        try:
            model = self.build()
        except TypeError as error:
            raise TypeError(
                f"PyMCModel builds {getattr(model_fn, '__name__', model_fn)!r} with its "
                f"arguments' defaults to read its variables; give each argument a default: "
                f"{error}"
            ) from error
        free = {rv.name: rv for rv in model.free_RVs}
        self.observed = tuple(p.name for p in arguments if p.default is None and p.name in free)
        self.given = tuple(p.name for p in arguments if p.name not in self.observed)
        self.parameters = tuple(name for name in free if name not in self.observed)
        self.shapes = {name: tuple(rv.type.shape) for name, rv in free.items()}
        self.dtypes = {name: np.dtype(rv.dtype) for name, rv in free.items()}
        self.normalized = not model.potentials and not any(
            type(rv.owner.op).__name__ in _IMPROPER_OPS for rv in model.free_RVs
        )

    def build(self, data: Mapping[str, Any] | None = None) -> Any:
        """The PyMC model at the bound values and *data*."""
        values = {**self.bound, **(data or {})}
        return self.model_fn(**{name: _to_numpy(value) for name, value in values.items()})

    def bind(self, values: Mapping[str, Any]) -> _PyMCProgram:
        """This program with *values* bound as well."""
        return _PyMCProgram(self.model_fn, {**self.bound, **values})

    def event_record(self, *, symbolic: bool) -> RecordSpec:
        """The record of the free variables, a dimension the build leaves unknown symbolic.

        Under *symbolic* every dimension is symbolic, for a kernel whose shapes
        its given values may set.
        """
        return RecordSpec(
            {
                name: NumericArraySpec(
                    tuple(
                        f"{name}_{axis}" if symbolic or size is None else int(size)
                        for axis, size in enumerate(shape)
                    )
                )
                for name, shape in self.shapes.items()
            }
        )

    def log_density(self) -> Callable[..., Any]:
        """The compiled joint log-density of the free variables, in their constrained space."""
        import pytensor
        import pytensor.tensor as pt
        from pymc.logprob.basic import conditional_logp
        from pytensor.graph.replace import clone_replace

        model = self.build()
        values = {
            rv: pt.tensor(name=f"{rv.name}_value", shape=rv.type.shape, dtype=rv.dtype)
            for rv in model.free_RVs
        }
        terms = [pt.sum(term) for term in conditional_logp(values).values()]
        if model.potentials:
            terms += [pt.sum(p) for p in clone_replace(list(model.potentials), values)]
        return pytensor.function(list(values.values()), pt.sum(terms))


def _pymc_density(self: PyMCModel, value: Any) -> Array:
    """The joint log-density of the free variables at *value*, a record of them.

    Each value is cast to its variable's dtype, since PyMC scores a count as an
    integer, and a count at a value that is not an integer has density zero.
    """
    memo = transient_memo(self)
    if "log_density" not in memo:
        memo["log_density"] = self._program.log_density()
    arguments = []
    for name, dtype in self._program.dtypes.items():
        given = np.asarray(value[name])
        cast = given.astype(dtype)
        if np.issubdtype(dtype, np.integer) and not np.array_equal(cast, given):
            return jnp.asarray(-jnp.inf)
        arguments.append(cast)
    return jnp.asarray(memo["log_density"](*arguments))


def _pymc_sample(self: PyMCModel, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
    """Draws of every free variable from the prior predictive, as a record.

    One draw for ``sample_shape=()``; otherwise each field carries the sample
    axes before its shape. The key seeds PyMC's sampler.
    """
    import pymc as pm

    n = int(np.prod(sample_shape)) if sample_shape else 1
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    prior = pm.sample_prior_predictive(draws=n, model=self._program.build(), random_seed=seed).prior
    fields = {}
    for name in self.event_spec.components:
        values = jnp.asarray(prior[name].values[0])
        fields[name] = (
            values[0] if not sample_shape else values.reshape(*sample_shape, *values.shape[1:])
        )
    return Record(self.name, fields)


class _PyMCModelMeta(type(Distribution)):
    """The metaclass of ``PyMCModel``: a function with a given slot defines a kernel."""

    def __call__(cls, name: str, model_fn: Callable[..., Any]) -> Any:
        program = model_fn if isinstance(model_fn, _PyMCProgram) else _PyMCProgram(model_fn)
        if program.given:
            return _PyMCKernel(name, program)
        return super().__call__(name, program)


class PyMCModel(Distribution, metaclass=_PyMCModelMeta):
    """The joint law a PyMC model-building function defines over its free variables.

    The free variables are the parameters and the observed variables alike: an
    argument that the function passes as an observed variable's ``observed``
    value is an event field, and an observed variable's shape is the one the
    build without its data gives. Any other argument is a given slot, so a
    model with covariates constructs the kernel over them, whose curried law is
    a ``PyMCModel``. It claims sampling, which draws from the prior predictive,
    and a normalized density. A model with a potential or an improper prior
    claims the unnormalized density alone, since its prior predictive does not
    draw from its law. Conditioning on observed values is Bayes' rule, which the
    PyMC methods of the inference-method registry normalize.

    Parameters
    ----------
    name : str
        The law's label.
    model_fn : callable
        A function returning a ``pymc.Model``, whose observed variables take
        their ``observed`` values from arguments that default to ``None``.

    Raises
    ------
    ImportError
        If ``pymc`` is not installed.
    TypeError
        If *model_fn* cannot be built with its arguments' defaults.
    """

    _capability_table: ClassVar = {
        SupportsLogProb: {"_log_prob": _pymc_density},
        SupportsUnnormalizedLogProb: {"_unnormalized_log_prob": _pymc_density},
        SupportsSampling: {"_sample": _pymc_sample},
    }
    #: The compiled density, built on first use, is not state.
    _transient_state = ("_memo",)

    def __new__(cls, name: str, model_fn: Callable[..., Any]) -> Any:
        program = model_fn if isinstance(model_fn, _PyMCProgram) else _PyMCProgram(model_fn)
        claimed = (
            (SupportsLogProb, SupportsSampling)
            if program.normalized
            else (SupportsUnnormalizedLogProb,)
        )
        return object.__new__(_capability_subclass(PyMCModel, claimed))

    def __init__(self, name: str, model_fn: Callable[..., Any]) -> None:
        try:
            import pymc  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pymc is required for PyMCModel. Install it with: pip install pymc"
            ) from e
        program = model_fn if isinstance(model_fn, _PyMCProgram) else _PyMCProgram(model_fn)
        super().__init__(name, OutputSpec(program.event_record(symbolic=False)))
        self._program = program

    # -- the program ------------------------------------------------------------

    @property
    def _param_names(self) -> tuple[str, ...]:
        """The free variables that are not observed variables."""
        return self._program.parameters

    @property
    def _observed_names(self) -> tuple[str, ...]:
        """The arguments the function passes as observed variables' values."""
        return self._program.observed

    def _pymc_model(self, data: Any = None) -> Any:
        """The PyMC model at the observed values *data*.

        *data* is ``None`` for the build without data, a mapping or record
        keyed by observed variables, or a bare array for the first of them. A
        value that a mapping or record holds for a parameter fixes it: the
        model observes the parameter at that value, as conditioning on it
        requires.
        """
        from ..core._record_batch import RecordBatch

        if data is None:
            return self._program.build()
        if isinstance(data, RecordBatch):
            return self._program.build(
                {
                    name: data._raw_column(name)
                    for name in self._observed_names
                    if name in data.event_template
                }
            )
        if isinstance(data, Record):
            values = {name: data[name] for name in data.fields}
            arguments = {name: values[name] for name in self._observed_names if name in values}
        elif isinstance(data, Mapping):
            values = dict(data)
            arguments = {
                name: value for name, value in values.items() if name not in self._param_names
            }
        else:
            return self._program.build({self._observed_names[0]: data})
        model = self._program.build(arguments)
        fixed = {name: _to_numpy(values[name]) for name in self._param_names if name in values}
        if not fixed:
            return model
        import pymc as pm

        return pm.observe(model, fixed)

    def _conditioned_param_names(self, model: Any) -> tuple[str, ...]:
        """The free variables to infer in a data-conditioned *model*, in order.

        They are the parameters the data leave free and any observed variable
        the data left free.

        Raises
        ------
        ValueError
            If *model*'s free variables differ from the build without data by a
            variable that the data neither observe nor fix, since the set of
            random variables must not change with the data.
        """
        free = {rv.name for rv in model.free_RVs}
        fixed = {rv.name for rv in model.observed_RVs}
        missing = [n for n in self._param_names if n not in free and n not in fixed]
        extra = free - set(self._param_names) - set(self._observed_names)
        if missing or extra:
            raise ValueError(
                f"PyMC random variable(s) {sorted(missing) or sorted(extra)} differ between "
                f"the build without data and this build. ProbPipe does not support models "
                f"whose set of free random variables changes with the data (dynamic random "
                f"variables); only per-variable shapes may depend on data size."
            )
        return tuple(n for n in self._param_names if n in free) + tuple(
            n for n in self._observed_names if n in free
        )

    def _parameter_record_for(self, model: Any, names: Sequence[str]) -> NumericRecordSpec:
        """The record of *names*, shaped as a build *model* shapes them.

        Raises
        ------
        ValueError
            If a named free variable has a dimension the build leaves unknown.
        """
        free_rvs = {rv.name: rv for rv in model.free_RVs}
        fields: dict[str, tuple[int, ...]] = {}
        for name in names:
            shape = tuple(free_rvs[name].type.shape)
            if any(s is None for s in shape):
                raise ValueError(
                    f"PyMC RV {name!r} has a non-concrete shape {shape}; declare its shape "
                    f"explicitly, as in pm.Normal({name!r}, 0, 1, shape=k)."
                )
            fields[name] = tuple(int(s) for s in shape)
        return NumericRecordSpec(**fields)

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The model function, by its name, and its free variables."""
        return _pymc_repr_arguments(self._program)


def _pymc_repr_arguments(program: _PyMCProgram) -> list[tuple[str, str]]:
    """The model function, by its name, and its free variables, for a repr."""
    return [
        ("model_fn", format_value(program.model_fn)),
        ("variables", format_names(program.shapes)),
    ]


def _pymc_kernel_density(self: _PyMCKernel, given: Any, value: Any) -> Array:
    """The joint log-density of the free variables at *value*, the given slots bound."""
    return self._condition_on(given)._unnormalized_log_prob(value)


def _pymc_kernel_log_prob(self: _PyMCKernel, given: Any, value: Any) -> Array:
    """The normalized joint log-density at *value*, the given slots bound."""
    return self._condition_on(given)._log_prob(value)


def _pymc_kernel_sample(
    self: _PyMCKernel, given: Any, key: Any, sample_shape: tuple[int, ...] = ()
) -> Any:
    """Prior predictive draws of the free variables, the given slots bound."""
    return self._condition_on(given)._sample(key, sample_shape)


class _PyMCKernel(ConditionalDistribution):
    """The kernel from a PyMC model's given arguments to the joint law of its free variables.

    Binding every given slot yields the ``PyMCModel`` of the function with those
    arguments bound. Its event dimensions are symbolic, since the given values
    may set them, and it claims the conditional twins of the capabilities that
    model claims: conditional sampling and the normalized density, or the
    unnormalized density alone.
    """

    _capability_table: ClassVar = {
        SupportsConditionalLogProb: {"_conditional_log_prob": _pymc_kernel_log_prob},
        SupportsConditionalUnnormalizedLogProb: {
            "_conditional_unnormalized_log_prob": _pymc_kernel_density
        },
        SupportsConditionalSampling: {"_conditional_sample": _pymc_kernel_sample},
    }

    def __new__(cls, name: str, program: _PyMCProgram) -> Any:
        claimed = (
            (SupportsConditionalLogProb, SupportsConditionalSampling)
            if program.normalized
            else (SupportsConditionalUnnormalizedLogProb,)
        )
        return object.__new__(_capability_subclass(_PyMCKernel, claimed))

    def __init__(self, name: str, program: _PyMCProgram) -> None:
        super().__init__(
            name,
            {slot: OpaqueSpec() for slot in program.given},
            OutputSpec(program.event_record(symbolic=True)),
        )
        object.__setattr__(self, "_program", program)

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **kwargs: Any
    ) -> Distribution | ConditionalDistribution:
        """The model at a value of every given slot, or the kernel over the slots left.

        Raises
        ------
        KeyError
            If a name is not a given slot.
        """
        values = _given_values(self.name, given, kwargs, self.given_spec)
        return PyMCModel(self.name, self._program.bind(values))

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The model function, by its name, and its free variables."""
        return _pymc_repr_arguments(self._program)
