"""The program-defined families: laws that a backend program defines (VII.9).

A program-defined model exposes the law its program defines, in the kind that
law has, and its variable names determine its output components.

Provides:
  - ``StanModel`` – the kernel from a Stan program's data-block variables to its
    unnormalized posterior over the parameter record, through BridgeStan; a
    construction that binds every data variable returns the posterior itself;
  - ``PyMCModel`` – the joint law a PyMC model-building function defines over
    its free variables, the parameters and the observed variables alike, and a
    kernel over the arguments no observed variable receives.

Each claims its density as its program supplies it, and ``condition_on``
normalizes a law that claims only an unnormalized one through the
inference-method registry.
"""

from __future__ import annotations

import functools
import inspect
import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .._messages import count, unknown_names
from ..core._immutable import transient_memo
from ..core._record_spec import NumericRecordSpec, RecordSpec
from ..core._repr import format_names, format_value
from ..core._spec_base import NumericArraySpec
from ..core._specs import OpaqueSpec, OutputSpec
from ..core.constraints import (
    Constraint,
    greater_than,
    interval,
    positive,
    positive_definite,
    real,
    simplex,
    sphere,
    unit_interval,
)
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

__all__ = ["PyMCModel", "StanModel"]


#: Make arguments for a Stan model library: TBB without its malloc proxy.
_BRIDGESTAN_MAKE_ARGS: tuple[str, ...] = ("TBB_LIBRARIES=tbb",)


def _given_values(owner: str, given: Any, kwargs: Mapping[str, Any], slots: Any) -> dict[str, Any]:
    """The values *given* and *kwargs* bind, by slot name.

    Parameters
    ----------
    owner : str
        The kernel's label, which the error message names.
    given : Record or Mapping[str, Any]
        Values of some given slots, by slot name; a record contributes its top-level fields.
    kwargs : Mapping[str, Any]
        Further values by slot name, which take precedence over those of *given*.
    slots : Iterable of str
        The names of the kernel's given slots, such as its ``given_spec``.

    Returns
    -------
    dict[str, Any]
        Each value as received, by slot name.

    Raises
    ------
    KeyError
        If a name is not one of *slots*.
    """
    values = {**dict(given.children if isinstance(given, Record) else given), **kwargs}
    unknown = sorted(set(values) - set(slots))
    if unknown:
        raise KeyError(f"cannot condition {owner!r}: {unknown_names('given slot', unknown, slots)}")
    return values


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

    Parameters
    ----------
    owner : str
        The label of the calling law, which the error messages name.
    blocks : tuple of _StanBlock
        The parameter blocks, in BridgeStan's order.
    values : Mapping[str, Any]
        One array per block, by block name, each of its block's shape.

    Returns
    -------
    Array
        A vector that concatenates the blocks in order, with each block's entries in
        BridgeStan's order.

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
            f"{owner!r} expects values for the Stan parameters {tuple(expected)} "
            f"({', '.join(detail)})"
        )
    out: list[Array] = []
    for b in blocks:
        arr = jnp.asarray(values[b.name])
        if tuple(arr.shape) != b.shape:
            raise TypeError(
                f"parameter {b.name!r} of {owner!r} must have shape {b.shape}, got "
                f"{tuple(arr.shape)}"
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

    Parameters
    ----------
    source : str
        The text of a Stan program.

    Returns
    -------
    dict[str, str]
        The text inside each block's braces with its comments replaced by spaces, keyed by
        the block's name with single spaces, such as ``"transformed data"``.

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


def _bounds_of(statement: str) -> str:
    """The text of *statement*'s first ``<...>`` group, which declares its bounds, or ``""``."""
    opening = statement.find("<")
    if opening < 0:
        return ""
    depth = 0
    for index in range(opening, len(statement)):
        depth += {"<": 1, ">": -1}.get(statement[index], 0)
        if not depth:
            return statement[opening + 1 : index]
    return statement[opening + 1 :]


def _top_level_parts(text: str) -> list[str]:
    """*text* split at the commas outside parentheses and brackets."""
    parts, depth, current = [], 0, []
    for char in text:
        depth += {"(": 1, "[": 1, ")": -1, "]": -1}.get(char, 0)
        if char == "," and not depth:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


#: A bound that is an expression rather than a number, which no support states.
_EXPRESSION = object()


def _bounds(text: str) -> dict[str, Any]:
    """Each bound *text* declares, by keyword: a number, or :data:`_EXPRESSION`."""
    declared: dict[str, Any] = {}
    for part in _top_level_parts(text):
        keyword, _, value = part.partition("=")
        try:
            declared[keyword.strip()] = float(value)
        except ValueError:
            declared[keyword.strip()] = _EXPRESSION
    return declared


def _sizes(expression: str | None) -> list[str]:
    return [part.strip() for part in expression.split(",")] if expression else []


#: The constrained Stan types with a support of their own.
_TYPE_SUPPORTS: dict[str, Constraint] = {
    "simplex": simplex,
    "unit_vector": sphere,
    "positive_ordered": positive,
    "cov_matrix": positive_definite,
}

#: The real Stan types whose support only their bounds constrain.
_BOUNDED_TYPES = frozenset({"real", "vector", "row_vector", "matrix"})


def _bounded_support(lower: Any, upper: Any) -> Constraint | None:
    """The support of a real value with the bounds *lower* and *upper*.

    Each bound is a number, ``None`` when absent, or :data:`_EXPRESSION`. Both
    bounds give an interval, a lower bound alone the values above it, and no
    bound the reals. An upper bound alone, or a bound that is an expression,
    leaves the support undeclared.
    """
    if lower is _EXPRESSION or upper is _EXPRESSION:
        return None
    if lower is not None and upper is not None:
        return interval(lower, upper)
    if lower is not None:
        return positive if isinstance(lower, float) and lower == 0 else greater_than(lower)
    return real if upper is None else None


def _stan_support(kind: str, bounds: Mapping[str, Any]) -> Constraint | None:
    """The support of a parameter of the Stan type *kind* with the declared *bounds*.

    A constrained type has its own support, and a real type the support of its
    bounds. Any other type leaves the support undeclared, as an ordered vector
    or a correlation matrix does.
    """
    if kind in _TYPE_SUPPORTS:
        return _TYPE_SUPPORTS[kind]
    if kind not in _BOUNDED_TYPES:
        return None
    return _bounded_support(bounds.get("lower"), bounds.get("upper"))


@dataclass(frozen=True)
class _StanVariable:
    """A variable a Stan program's data or parameters block declares.

    Attributes
    ----------
    name : str
        The variable's name.
    sizes : tuple of str
        The size expression of each axis, outermost first.
    kind : str
        The declared type, such as ``"vector"`` or ``"simplex"``.
    bounds : Mapping[str, Any]
        The declared bounds by keyword, each a number or :data:`_EXPRESSION`.
    """

    name: str
    sizes: tuple[str, ...]
    kind: str
    bounds: Mapping[str, Any]


def _declared_variables(block: str) -> list[_StanVariable]:
    """Each variable a data or parameters block declares, in order.

    Parameters
    ----------
    block : str
        The body of a data or parameters block, as :func:`_stan_blocks` returns it.

    Returns
    -------
    list of _StanVariable
        One entry per declaration statement.

    Raises
    ------
    ValueError
        If a statement is not a declaration of a known Stan type.
    """
    variables: list[_StanVariable] = []
    for statement in block.split(";"):
        bounds = _bounds(_bounds_of(statement))
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
            raise ValueError(
                f"StanModel does not support the Stan type {kind!r} (variable {match['name']!r})"
            )
        axes = (*_sizes(match["array"]), *_sizes(match["old"]), *shape)
        variables.append(_StanVariable(match["name"], axes, kind, bounds))
    return variables


def _dimension(expression: str, name: str, axis: int, data: Mapping[str, Any]) -> int | str:
    """A size of *name*: a literal, a scalar data variable's value, or else a symbolic dimension.

    A size that names a data variable is the dimension of that name until the
    variable is bound.
    """
    if expression.isdigit():
        return int(expression)
    if re.fullmatch(r"[A-Za-z_]\w*", expression):
        value = data.get(expression)
        return int(value) if value is not None and np.ndim(value) == 0 else expression
    return f"{name}_{axis}"


def _stanc(*, fetch: bool = True) -> Path:
    """The location of BridgeStan's stanc compiler, fetched on first use when *fetch* is true.

    BridgeStan keeps its source tree in the directory ``$BRIDGESTAN`` names, or
    else under ``~/.bridgestan``, and its Makefile fetches the stanc3 binary into
    the tree's ``bin/``.

    Parameters
    ----------
    fetch : bool
        Whether to download a missing source tree and fetch a missing compiler
        with the Makefile's target, as BridgeStan does before it first compiles
        a model. With ``False``, the compiler is only located.

    Returns
    -------
    Path
        The stanc executable in the source tree's ``bin/``.

    Raises
    ------
    ImportError
        If ``bridgestan`` is not installed, or its stanc compiler is absent and
        *fetch* is false or the fetch fails. The message gives the command that
        fetches the compiler.
    """
    try:
        from bridgestan.compile import IS_WINDOWS, MAKE, get_bridgestan_path
    except ImportError as e:
        raise ImportError(
            "StanModel reads its program's declarations with stanc, which bridgestan provides. "
            "Install it with: pip install bridgestan"
        ) from e
    root = get_bridgestan_path(download=fetch)
    if not root:
        raise ImportError(
            "BridgeStan's source tree is not installed; StanModel downloads it to "
            "~/.bridgestan on first use, or set $BRIDGESTAN to an existing tree"
        )
    target = "bin/stanc.exe" if IS_WINDOWS else "bin/stanc"
    stanc = Path(root) / target
    command = f"{MAKE} -C {root} {target}"
    if not stanc.exists() and fetch:
        completed = subprocess.run(
            [MAKE, target], cwd=root, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout).strip()
            raise ImportError(
                f"BridgeStan could not fetch its stanc compiler with `{command}`: {message}"
            )
    if not stanc.exists():
        raise ImportError(
            f"BridgeStan's stanc compiler is not installed; fetch it with `{command}`"
        )
    return stanc


@functools.lru_cache(maxsize=128)
def _stanc_info_of(stan_file: str, modified: int, size: int) -> Mapping[str, Any]:
    """``stanc --info`` of *stan_file*, cached while the file is unchanged.

    Parameters
    ----------
    stan_file : str
        The ``.stan`` file that stanc reads.
    modified : int
        The file's modification time in nanoseconds, which is part of the cache key.
    size : int
        The file's size in bytes, which is part of the cache key.

    Returns
    -------
    Mapping[str, Any]
        The parsed JSON report, whose ``"inputs"`` and ``"parameters"`` entries give each
        variable's element type and number of dimensions.

    Raises
    ------
    ValueError
        If stanc rejects the program, quoting its message.
    """
    completed = subprocess.run(
        [str(_stanc()), "--info", stan_file], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise ValueError(f"stanc rejected the Stan program {stan_file}: {message}")
    return json.loads(completed.stdout)


def _stanc_info(stan_file: str) -> Mapping[str, Any]:
    """``stanc --info`` of *stan_file*: each data variable and parameter, with its type and rank."""
    status = Path(stan_file).stat()
    return _stanc_info_of(str(stan_file), status.st_mtime_ns, status.st_size)


#: The dtype the array backend gives each element type stanc reports.
_STAN_DTYPES: dict[str, Callable[[], Any]] = {
    "int": lambda: np.dtype(jnp.result_type(int)),
    "real": lambda: np.dtype(jnp.result_type(float)),
    "complex": lambda: np.dtype(jnp.result_type(complex)),
}


def _checked_against_stanc(
    variables: Sequence[_StanVariable], reported: Mapping[str, Any], block: str, stan_file: str
) -> tuple[tuple[_StanVariable, Any], ...]:
    """Each variable of *block* with the dtype of the type stanc reports for it.

    Parameters
    ----------
    variables : Sequence[_StanVariable]
        The variables read from the block's text, in declaration order.
    reported : Mapping[str, Any]
        The block's entry in stanc's report, which gives each variable's element type and
        number of dimensions by variable name.
    block : str
        The block's name, ``"data"`` or ``"parameters"``, for the error messages.
    stan_file : str
        The program's file, for the error messages.

    Returns
    -------
    tuple of (_StanVariable, np.dtype)
        Each variable paired with the dtype of its element type, in declaration order.

    Raises
    ------
    ValueError
        If the declarations read from the program name other variables than
        stanc reports, or a variable's rank differs from stanc's.
    """
    names = [variable.name for variable in variables]
    if names != list(reported):
        raise ValueError(
            f"StanModel read the {block} block of {stan_file} as declaring {names}, but stanc "
            f"reports {list(reported)}; StanModel does not support this declaration syntax"
        )
    checked = []
    for variable in variables:
        entry = reported[variable.name]
        if len(variable.sizes) != entry["dimensions"]:
            raise ValueError(
                f"StanModel read {variable.name!r} in {stan_file} as having "
                f"{count(len(variable.sizes), 'axis', 'axes')}, but stanc reports "
                f"{entry['dimensions']}; StanModel does not support this declaration syntax"
            )
        checked.append((variable, _STAN_DTYPES[entry["type"]]()))
    return tuple(checked)


@dataclass(frozen=True)
class _StanProgram:
    """A Stan program's file, its data-block variables, and its parameters.

    Each data variable and parameter carries the dtype of the element type ``stanc
    --info`` reports, and its size expressions and bounds as the program
    declares them.
    """

    stan_file: str
    data: tuple[tuple[_StanVariable, Any], ...]
    parameters: tuple[tuple[_StanVariable, Any], ...]

    @classmethod
    def read(cls, stan_file: str) -> _StanProgram:
        """The program in *stan_file*.

        Parameters
        ----------
        stan_file : str
            The ``.stan`` file to read.

        Returns
        -------
        _StanProgram
            The program with its data-block variables and parameters, each checked against
            stanc's report and paired with its dtype.

        Raises
        ------
        ImportError
            If ``bridgestan`` is not installed, or its stanc compiler is absent
            and cannot be fetched.
        ValueError
            If stanc rejects the program, the program declares no parameters,
            or a declaration cannot be read.
        """
        info = _stanc_info(stan_file)
        blocks = _stan_blocks(Path(stan_file).read_text())
        parameters = _checked_against_stanc(
            _declared_variables(blocks.get("parameters", "")),
            info["parameters"],
            "parameters",
            stan_file,
        )
        if not parameters:
            raise ValueError(f"the Stan program {stan_file} declares no parameters")
        data = _checked_against_stanc(
            _declared_variables(blocks.get("data", "")), info["inputs"], "data", stan_file
        )
        return cls(str(stan_file), data, parameters)

    @property
    def data_entries(self) -> tuple[str, ...]:
        """The names of the data-block variables, in declaration order."""
        return tuple(variable.name for variable, _ in self.data)

    def given_spec(self, bound: Mapping[str, Any]) -> dict[str, NumericArraySpec]:
        """The given slot of each data variable *bound* leaves unbound, typed as it is declared.

        A size that names a scalar variable of *bound* is that variable's value, and
        one that names another variable is the dimension of its name.
        """
        return {
            variable.name: NumericArraySpec(_shape(variable, bound), dtype)
            for variable, dtype in self.data
            if variable.name not in bound
        }

    def parameter_record(self, data: Mapping[str, Any]) -> RecordSpec:
        """The parameter record, each size that names a scalar variable of *data* bound to its value.

        Each parameter carries its dtype and the support its declaration states.
        """
        return RecordSpec(
            {
                variable.name: NumericArraySpec(
                    _shape(variable, data), dtype, _stan_support(variable.kind, variable.bounds)
                )
                for variable, dtype in self.parameters
            }
        )


def _shape(variable: _StanVariable, data: Mapping[str, Any]) -> tuple[int | str, ...]:
    """The shape of *variable*, each size that names a scalar variable of *data* bound to its value."""
    return tuple(
        _dimension(size, variable.name, axis, data) for axis, size in enumerate(variable.sizes)
    )


def _parameter_record_at(
    record: RecordSpec, shapes: Mapping[str, tuple[int, ...]]
) -> NumericRecordSpec:
    """*record*, a program's parameter record, with each field at the shape *shapes* gives it.

    A run's draws fix each shape, and the record keeps each field's dtype and
    support.
    """
    return NumericRecordSpec(
        {
            name: NumericArraySpec(tuple(shapes[name]), spec.dtype, spec.support)
            for name, spec in record.children.items()
        }
    )


class _StanPosterior(Distribution, SupportsUnnormalizedLogProb):
    """The unnormalized posterior of a Stan program at a value of every data-block variable.

    Its event is the parameter record, and its unnormalized log-density is
    BridgeStan's log density in the constrained parameterization without the
    Jacobian, so it claims no normalized capability. The Stan methods of the
    inference-method registry read its program and data.

    Parameters
    ----------
    label : str
        The law's label.
    program : _StanProgram
        The program.
    data : Mapping[str, Any]
        A value of every data-block variable.
    """

    #: The program reads its observed variables' shapes from the data it
    #: receives, so ``condition_on`` leaves their givens to the program.
    _shapes_from_data: ClassVar[bool] = True

    #: The BridgeStan model, built on first use, is not state.
    _transient_state = ("_memo",)

    def __init__(self, label: str, program: _StanProgram, data: Mapping[str, Any]) -> None:
        super().__init__(label, OutputSpec(program.parameter_record(data)))
        self._program = program
        self._data = dict(data)

    @property
    def stan_file(self) -> str:
        """The Stan program's file."""
        return self._program.stan_file

    @property
    def data(self) -> Mapping[str, Any]:
        """The value of every data-block variable."""
        return MappingProxyType(self._data)

    def _bridgestan_model(self) -> Any:
        """The BridgeStan model of the program at its data, built once.

        Returns
        -------
        bridgestan.StanModel
            The model, which the law's transient memo keeps for later calls.

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
        return _pack_block_params(self.label, self._blocks(), field_kwargs)

    def _flat(self, value: Any) -> np.ndarray:
        """*value*, a record of the parameter blocks or BridgeStan's flat vector, as the vector."""
        if isinstance(value, (Record, Mapping)):
            fields = value.children if isinstance(value, Record) else value
            value = _pack_block_params(self.label, self._blocks(), dict(fields))
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

    #: The program reads its observed variables' shapes from the data it
    #: receives, so ``condition_on`` leaves their givens to the program.
    _shapes_from_data: ClassVar[bool] = True

    def __init__(self, posterior: _StanPosterior) -> None:
        self._posterior = posterior
        self._blocks = _param_blocks(posterior._bridgestan_model().param_unc_names())
        super().__init__(
            f"{posterior.label}_unconstrained",
            OutputSpec(NumericRecordSpec({b.name: b.shape for b in self._blocks})),
        )

    def _pack_value(self, **field_kwargs: Any) -> Array:
        """The keyword form: one array per unconstrained block, as BridgeStan's flat vector."""
        return _pack_block_params(self.label, self._blocks, field_kwargs)

    def _unnormalized_log_prob(self, value: Any) -> Array:
        """BridgeStan's log density at the unconstrained *value*, with the Jacobian."""
        if isinstance(value, (Record, Mapping)):
            fields = value.children if isinstance(value, Record) else value
            value = _pack_block_params(self.label, self._blocks, dict(fields))
        return jnp.asarray(self._posterior._bridgestan_model().log_density(_to_f64(value)))

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The posterior this law reparameterizes."""
        return [("posterior", repr(self._posterior))]


class _StanModelMeta(type(ConditionalDistribution)):
    """The metaclass of ``StanModel``: binding every data-block variable returns the posterior."""

    def __call__(cls, label: str, stan_file: str, *, data: Mapping[str, Any] | None = None) -> Any:
        program = _StanProgram.read(stan_file)
        bound = dict(data or {})
        unknown = sorted(set(bound) - set(program.data_entries))
        if unknown:
            raise KeyError(
                f"cannot bind data for {stan_file}: "
                f"{unknown_names('data variable', unknown, program.data_entries)}"
            )
        if set(program.data_entries) <= set(bound):
            return _StanPosterior(label, program, bound)
        return super().__call__(label, stan_file, data=data)


class StanModel(
    ConditionalDistribution, SupportsConditionalUnnormalizedLogProb, metaclass=_StanModelMeta
):
    """The kernel from a Stan program's data-block variables to its unnormalized posterior.

    The given slots are the data-block variables *data* leaves unbound, which the
    program does not divide into sizes, covariates, and observations, and the
    event is the parameter record. Each slot is typed as its variable is declared,
    and each parameter carries its dtype and the support its declared
    constraint states; a size that names an unbound variable is the dimension of
    that name on both sides. It claims ``SupportsConditionalUnnormalizedLogProb``
    alone: binding every data variable yields the unnormalized posterior, whose density
    is BridgeStan's in the constrained parameterization without the Jacobian,
    and ``condition_on`` normalizes it with a method such as Stan's NUTS. The
    declarations are read at construction from ``stanc --info`` and the
    program's text, and BridgeStan compiles the program when a density is
    first evaluated. The first construction on a machine downloads BridgeStan's
    source tree and its stanc compiler, as BridgeStan's first compile does.

    Parameters
    ----------
    label : str
        The kernel's label.
    stan_file : str
        The location of the ``.stan`` file that holds the program.
    data : Mapping[str, Any], optional
        Values of some data-block variables, bound at construction. A
        construction that binds every data variable returns the posterior, a
        ``Distribution``.

    Raises
    ------
    ImportError
        If ``bridgestan`` is not installed, or its stanc compiler is absent
        and cannot be fetched.
    KeyError
        If *data* names a variable the data block does not declare.
    ValueError
        If stanc rejects the program, the program declares no parameters, or a
        declaration cannot be read.
    """

    #: The program reads its observed variables' shapes from the data it
    #: receives, so ``condition_on`` leaves their givens to the program.
    _shapes_from_data: ClassVar[bool] = True

    def __init__(
        self, label: str, stan_file: str, *, data: Mapping[str, Any] | None = None
    ) -> None:
        program = _StanProgram.read(stan_file)
        bound = dict(data or {})
        super().__init__(
            label, program.given_spec(bound), OutputSpec(program.parameter_record(bound))
        )
        object.__setattr__(self, "_program", program)
        object.__setattr__(self, "_data", bound)

    @property
    def stan_file(self) -> str:
        """The Stan program's file."""
        return self._program.stan_file

    @property
    def data(self) -> Mapping[str, Any]:
        """The data-block variables bound at construction or by currying."""
        return MappingProxyType(self._data)

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **kwargs: Any
    ) -> Distribution | ConditionalDistribution:
        """The posterior at a value of every data-block variable, or the kernel over the rest.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values of some unbound data-block variables, by variable name.
        **kwargs : Any
            Further values by variable name, which take precedence over those of *given*.

        Returns
        -------
        Distribution or ConditionalDistribution
            The posterior when the values bind every data-block variable, and otherwise a
            ``StanModel`` with the values added to its bound data.

        Raises
        ------
        KeyError
            If a name is not an unbound data-block variable.
        """
        values = _given_values(self.label, given, kwargs, self.given_spec)
        return StanModel(self.label, self.stan_file, data={**self._data, **values})

    def _conditional_unnormalized_log_prob(
        self, given: Record | Mapping[str, Any], value: Any
    ) -> Array:
        """The unnormalized posterior density at *value*, the data bound to *given*.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            A value of every unbound data-block variable, by variable name.
        value : Any
            A record of the parameter blocks, or BridgeStan's flat parameter vector.

        Returns
        -------
        Array
            The scalar log-density in the constrained parameterization, without the
            Jacobian.

        Raises
        ------
        KeyError
            If *given* leaves a data-block variable unbound.
        """
        law = self._condition_on(given)
        if isinstance(law, ConditionalDistribution):
            raise KeyError(
                f"{self.label!r} is missing values for the data variables {list(law.given_spec)}"
            )
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


def _backend_dtype(dtype: Any) -> np.dtype:
    """*dtype*, a PyMC variable's, as the array backend holds its values."""
    return np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))


def _constant(bound: Any) -> Any:
    """The value of the bound *bound* when it is a constant of the graph, else :data:`_EXPRESSION`."""
    from pytensor.graph.basic import Constant

    if bound is None:
        return None
    if isinstance(bound, Constant):
        value = np.asarray(bound.data)
        return float(value) if value.ndim == 0 else jnp.asarray(value)
    return _EXPRESSION


def _pymc_support(model: Any, rv: Any) -> Constraint | None:
    """The support of the free variable *rv* of *model*, read from its transform and dtype.

    A continuous variable without a transform is real, and a transform states
    the support it maps onto: a log transform the positive reals, a log-odds
    transform the unit interval, a simplex transform the simplex, and an
    interval transform the interval of its bounds. A transform that states no
    support ProbPipe declares, a bound that is not a constant, and a discrete
    variable leave the support undeclared.
    """
    from pymc.distributions import transforms
    from pymc.logprob import transforms as logprob

    transform = model.rvs_to_transforms.get(rv)
    if transform is None:
        return real if np.issubdtype(np.dtype(rv.dtype), np.floating) else None
    if isinstance(transform, (logprob.LogTransform, transforms.LogExpM1)):
        return positive
    if isinstance(transform, logprob.LogOddsTransform):
        return unit_interval
    if isinstance(transform, (logprob.SimplexTransform, transforms.SumTo1)):
        return simplex
    if isinstance(transform, logprob.CircularTransform):
        return interval(-np.pi, np.pi)
    if isinstance(transform, transforms.Ordered):
        return positive if transform.positive else None
    if isinstance(transform, logprob.IntervalTransform):
        lower, upper = (_constant(bound) for bound in transform.args_fn(*rv.owner.inputs))
        return _bounded_support(lower, upper)
    return None


class _PyMCProgram:
    """A PyMC model-building function with some arguments bound, and the variables its build has.

    The function is built with its arguments' defaults. An argument whose
    default is ``None`` and for which that build has a free variable of the
    same name is an observed variable, since passing ``None`` as its
    ``observed`` value leaves it free; every other argument is a given slot.

    Parameters
    ----------
    model_fn : callable
        The model-building function, which returns a ``pymc.Model``.
    bound : Mapping[str, Any], optional
        Values of some arguments of *model_fn*, by argument name, which every build passes.

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
                f"PyMCModel could not build {getattr(model_fn, '__name__', model_fn)!r} from its "
                f"default arguments ({error}); give every argument a default, such as None for "
                f"observed data"
            ) from error
        free = {rv.name: rv for rv in model.free_RVs}
        self.observed = tuple(p.name for p in arguments if p.default is None and p.name in free)
        self.given = tuple(p.name for p in arguments if p.name not in self.observed)
        self.parameters = tuple(name for name in free if name not in self.observed)
        self.shapes = {name: tuple(rv.type.shape) for name, rv in free.items()}
        self.dtypes = {name: np.dtype(rv.dtype) for name, rv in free.items()}
        self.supports = {name: _pymc_support(model, rv) for name, rv in free.items()}
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

        Each variable carries its dtype and the support its transform states.
        Under *symbolic* every dimension is symbolic, for a kernel whose shapes
        its given values may set.
        """
        return RecordSpec(
            {
                name: NumericArraySpec(
                    tuple(
                        f"{name}_{axis}" if symbolic or size is None else int(size)
                        for axis, size in enumerate(shape)
                    ),
                    _backend_dtype(self.dtypes[name]),
                    self.supports[name],
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
    return Record(self.label, fields)


class _PyMCModelMeta(type(Distribution)):
    """The metaclass of ``PyMCModel``: a function with a given slot defines a kernel."""

    def __call__(cls, label: str, model_fn: Callable[..., Any]) -> Any:
        program = model_fn if isinstance(model_fn, _PyMCProgram) else _PyMCProgram(model_fn)
        if program.given:
            return _PyMCKernel(label, program)
        return super().__call__(label, program)


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
    label : str
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

    #: The program reads its observed variables' shapes from the data it
    #: receives, so ``condition_on`` leaves their givens to the program.
    _shapes_from_data: ClassVar[bool] = True

    _capability_table: ClassVar = {
        SupportsLogProb: {"_log_prob": _pymc_density},
        SupportsUnnormalizedLogProb: {"_unnormalized_log_prob": _pymc_density},
        SupportsSampling: {"_sample": _pymc_sample},
    }
    #: The compiled density, built on first use, is not state.
    _transient_state = ("_memo",)

    def __new__(cls, label: str, model_fn: Callable[..., Any]) -> Any:
        program = model_fn if isinstance(model_fn, _PyMCProgram) else _PyMCProgram(model_fn)
        claimed = (
            (SupportsLogProb, SupportsSampling)
            if program.normalized
            else (SupportsUnnormalizedLogProb,)
        )
        return object.__new__(_capability_subclass(PyMCModel, claimed))

    def __init__(self, label: str, model_fn: Callable[..., Any]) -> None:
        try:
            import pymc  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pymc is required for PyMCModel. Install it with: pip install pymc"
            ) from e
        program = model_fn if isinstance(model_fn, _PyMCProgram) else _PyMCProgram(model_fn)
        super().__init__(label, OutputSpec(program.event_record(symbolic=False)))
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

        Parameters
        ----------
        model : pymc.Model
            A build of the model function at observed values, as :meth:`_pymc_model` returns
            it.

        Returns
        -------
        tuple of str
            The free parameters in the order of ``_param_names``, then the free observed
            variables in the order of ``_observed_names``.

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
            names, change = (sorted(missing), "disappear") if missing else (sorted(extra), "appear")
            raise ValueError(
                f"PyMC random variables {names} {change} when the model is built with this "
                f"data, but the free random variables must not change with the data (only "
                f"their shapes may)"
            )
        return tuple(n for n in self._param_names if n in free) + tuple(
            n for n in self._observed_names if n in free
        )

    def _parameter_record_for(self, model: Any, names: Sequence[str]) -> NumericRecordSpec:
        """The record of *names*, shaped as a build *model* shapes them.

        Each variable carries its dtype and the support its transform states.

        Parameters
        ----------
        model : pymc.Model
            A build of the model function, whose free variables give the shapes.
        names : Sequence[str]
            Names of free variables of *model*, in the record's field order.

        Returns
        -------
        NumericRecordSpec
            One field per name, at the variable's concrete shape in *model*.

        Raises
        ------
        ValueError
            If a named free variable has a dimension the build leaves unknown.
        """
        free_rvs = {rv.name: rv for rv in model.free_RVs}
        fields: dict[str, NumericArraySpec] = {}
        for name in names:
            rv = free_rvs[name]
            shape = tuple(rv.type.shape)
            if any(s is None for s in shape):
                raise ValueError(
                    f"PyMC RV {name!r} has a non-concrete shape {shape}; declare its shape "
                    f"explicitly, as in pm.Normal({name!r}, 0, 1, shape=k)."
                )
            fields[name] = NumericArraySpec(
                tuple(int(s) for s in shape), _backend_dtype(rv.dtype), _pymc_support(model, rv)
            )
        return NumericRecordSpec(fields)

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

    #: The program reads its observed variables' shapes from the data it
    #: receives, so ``condition_on`` leaves their givens to the program.
    _shapes_from_data: ClassVar[bool] = True

    _capability_table: ClassVar = {
        SupportsConditionalLogProb: {"_conditional_log_prob": _pymc_kernel_log_prob},
        SupportsConditionalUnnormalizedLogProb: {
            "_conditional_unnormalized_log_prob": _pymc_kernel_density
        },
        SupportsConditionalSampling: {"_conditional_sample": _pymc_kernel_sample},
    }

    def __new__(cls, label: str, program: _PyMCProgram) -> Any:
        claimed = (
            (SupportsConditionalLogProb, SupportsConditionalSampling)
            if program.normalized
            else (SupportsConditionalUnnormalizedLogProb,)
        )
        return object.__new__(_capability_subclass(_PyMCKernel, claimed))

    def __init__(self, label: str, program: _PyMCProgram) -> None:
        super().__init__(
            label,
            {slot: OpaqueSpec() for slot in program.given},
            OutputSpec(program.event_record(symbolic=True)),
        )
        object.__setattr__(self, "_program", program)

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **kwargs: Any
    ) -> Distribution | ConditionalDistribution:
        """The model at a value of every given slot, or the kernel over the slots left.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values of some given slots, by slot name.
        **kwargs : Any
            Further values by slot name, which take precedence over those of *given*.

        Returns
        -------
        Distribution or ConditionalDistribution
            The ``PyMCModel`` of the program with the values bound, or a ``_PyMCKernel``
            over the given slots left.

        Raises
        ------
        KeyError
            If a name is not a given slot.
        """
        values = _given_values(self.label, given, kwargs, self.given_spec)
        return PyMCModel(self.label, self._program.bind(values))

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The model function, by its name, and its free variables."""
        return _pymc_repr_arguments(self._program)
