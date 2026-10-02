"""Function values, their declarations, and the installable call-engine boundary.

Provides:
  - :class:`FunctionSpec`, the function kind's term spec;
  - :class:`Function`, the base of the function kind, with its declared sides,
    identity, controls, plain evaluation, and the call path;
  - :func:`install_call_engine`, which installs the engine of design Part V on
    the call path;
  - the function capabilities :class:`SupportsInverse`,
    :class:`SupportsLogDetJacobian`, and :class:`SupportsDifferentiation`, with
    :func:`is_invertible` and :func:`is_differentiable`.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, Self, cast, runtime_checkable

import jax.numpy as jnp

from ..core._dispatch import Feasibility
from ..core._record_spec import RecordSpec
from ..core._repr import format_names, public_class_name, term_repr
from ..core._spec_base import NumericArraySpec, NumericSpec, TermSpec, _unify_specs
from ..core._specs import InputSpec, OutputSpec
from ..core.config import WorkflowKind
from ..core.node import Node
from ..core.tracked import Annotated, TrackedTerm
from ._binding import (
    WorkflowSignatureInfo,
    make_signature_info,
    make_signature_info_from_signature,
    resolve_workflow_values,
    values_to_bound_arguments,
)

if TYPE_CHECKING:
    from ..core._numeric import Numeric
    from ..core.named_tree import NamedTree
    from ..custom_types import Array


@dataclass(frozen=True, init=False)
class FunctionSpec(TermSpec):
    """A callable's optional input slots and output component declaration.

    Parameters
    ----------
    input_spec : InputSpec or None
        Named input slots. None leaves the input side unspecified.
    output_spec : OutputSpec or None
        The returned term and its exposed components. None leaves that side
        unspecified; a named type hole leaves only its term kind unspecified.

    Raises
    ------
    TypeError
        If either side has the wrong declaration type.

    Notes
    -----
    ``is_valid`` admits any callable without executing it. Dimension binding
    reads declarations when available, sharing one dimension scope across both
    sides. Missing declarations add no bindings. Incompatible slot names,
    output exposure, kinds, or dimensions raise ValueError. Labels are outside
    this spec; component names participate in declaration matching.
    """

    input_spec: InputSpec | None
    output_spec: OutputSpec | None

    def __init__(
        self, input_spec: InputSpec | None = None, output_spec: OutputSpec | None = None
    ) -> None:
        if input_spec is not None and not isinstance(input_spec, InputSpec):
            raise TypeError("FunctionSpec.input_spec must be an InputSpec or None")
        if output_spec is not None and not isinstance(output_spec, OutputSpec):
            raise TypeError("FunctionSpec.output_spec must be an OutputSpec or None")
        object.__setattr__(self, "input_spec", input_spec)
        object.__setattr__(self, "output_spec", output_spec)

    @property
    def free_dims(self) -> frozenset[str]:
        """Unbound dimensions across both sides, in one scope."""
        inputs = self.input_spec.free_dims if self.input_spec is not None else frozenset()
        output = self.output_spec.spec if self.output_spec is not None else None
        return inputs | (output.free_dims if output is not None else frozenset())

    def _substitute_dims(self, bindings: Mapping[str, int | str]) -> FunctionSpec:
        inputs = (
            None
            if self.input_spec is None
            else InputSpec(
                {name: spec._substitute_dims(bindings) for name, spec in self.input_spec.items()}
            )
        )
        output = self.output_spec
        if output is not None and output.spec is not None:
            output = output._with_spec(output.spec._substitute_dims(bindings))
        return FunctionSpec(inputs, output)

    def _bind_dims_from_value(self, value: Callable, bindings: dict[str, int], path: str) -> None:
        if not callable(value):
            raise ValueError(f"{path} does not conform to FunctionSpec: expected a callable")
        actual = getattr(value, "spec", None)
        if isinstance(actual, FunctionSpec):
            self._bind_dims_from_spec(actual, bindings, path)

    def _bind_dims_from_spec(self, actual: TermSpec, bindings: dict[str, int], path: str) -> bool:
        if not isinstance(actual, FunctionSpec):
            return False
        if self.input_spec is not None and actual.input_spec is not None:
            if self.input_spec.keys() != actual.input_spec.keys():
                raise ValueError(f"{path} has incompatible input slots")
            for name, spec in self.input_spec.items():
                _unify_specs(spec, actual.input_spec[name], bindings, f"{path}/input/{name}")
        if self.output_spec is not None and actual.output_spec is not None:
            expected, observed = self.output_spec, actual.output_spec
            if (
                expected._component_name != observed._component_name
                or expected.components.keys() != observed.components.keys()
            ):
                raise ValueError(f"{path} has incompatible output components")
            if expected.spec is not None and observed.spec is not None:
                _unify_specs(expected.spec, observed.spec, bindings, f"{path}/output")
        return True

    def is_valid(self, value: Any) -> bool:
        """Whether value is callable; this does not execute or certify its body."""
        return callable(value)

    def __repr__(self) -> str:
        """The declared sides, as the constructor takes them; an unspecified side is omitted."""
        fields = [
            (side, repr(declaration))
            for side, declaration in (
                ("input_spec", self.input_spec),
                ("output_spec", self.output_spec),
            )
            if declaration is not None
        ]
        return term_repr("FunctionSpec", None, fields)


@dataclass(frozen=True)
class _FunctionInvocationContext:
    """Immutable dimension bindings shared with one raw evaluation."""

    dimension_bindings: Mapping[str, int]

    def __init__(self, dimension_bindings: Mapping[str, int] | None = None):
        object.__setattr__(
            self, "dimension_bindings", MappingProxyType(dict(dimension_bindings or {}))
        )


class _FunctionImplementation(Protocol):
    """Private execution payload for dynamically constructed Function values."""

    def invoke(
        self, bound_inputs: inspect.BoundArguments, *, context: _FunctionInvocationContext
    ) -> Any: ...


@dataclass(frozen=True)
class _CallableFunctionImplementation:
    """A plain callable as a Function execution payload."""

    callable: Callable[..., Any]

    def invoke(
        self, bound_inputs: inspect.BoundArguments, *, context: _FunctionInvocationContext
    ) -> Any:
        return self.callable(*bound_inputs.args, **bound_inputs.kwargs)


def _complete_output_spec(
    output_spec: OutputSpec | TermSpec | None, output_label: str
) -> OutputSpec | None:
    if output_spec is None or isinstance(output_spec, OutputSpec):
        return output_spec
    if isinstance(output_spec, RecordSpec):
        return OutputSpec(output_spec)
    if isinstance(output_spec, TermSpec):
        return OutputSpec(**{output_label: output_spec})
    raise TypeError("output_spec must be an OutputSpec, TermSpec, or None")


def _bind_function_inputs(
    *,
    function_name: str,
    input_spec: InputSpec | None,
    values: Mapping[str, Any],
    bindings: Mapping[str, int] | None = None,
) -> tuple[InputSpec | None, dict[str, int]]:
    resolved = dict(bindings or {})
    if input_spec is None:
        return None, resolved
    if input_spec.keys() != values.keys():
        raise ValueError(f"Function {function_name!r} input slots do not match its declaration")
    for name, spec in input_spec.items():
        spec._bind_dims_from_value(
            values[name], resolved, f"Function {function_name!r} input/{name}"
        )
    return input_spec.with_dim_sizes(**resolved), resolved


def _validate_function_output(
    *, function_name: str, output_spec: OutputSpec | None, result: Any, bindings: Mapping[str, int]
) -> OutputSpec | None:
    """Validate one returned term without wrapping it or changing its declaration.

    Each declared array's dtype admits a returned dtype of the same kind (bool,
    integer, floating, or complex) at any width, and its support must hold.

    Raises
    ------
    ValueError
        If the result's structure, dimensions, dtype, or support does not
        conform to the declaration.
    """
    if output_spec is None:
        return None
    spec = output_spec.spec
    if spec is None:
        spec = RecordSpec.infer_from({"result": result}).children["result"]
    resolved = dict(bindings)
    path = f"Function {function_name!r} output"
    if output_spec._component_name is not None:
        path += f"/{output_spec._component_name}"
    actual_spec = getattr(result, "spec", None)
    if isinstance(actual_spec, TermSpec):
        _unify_specs(spec, actual_spec, resolved, path)
        _validate_declared_support(spec, actual_spec, path)
    spec._bind_dims_from_value(result, resolved, path)
    concrete = spec._substitute_dims(resolved)
    _validate_output_values(concrete, result, path)
    return output_spec._with_spec(concrete)


def _validate_declared_support(expected: TermSpec, actual: TermSpec, path: str) -> None:
    """Refuse an actual spec whose support a declared support does not contain, leaf by leaf.

    Records and batches are read field by field and element by element, and a
    law's declaration component by component of its event, so a returned law
    whose draws leave the declared support is refused, as completion checks the
    produced value (II.2). A law's declaration has no type hole, so both sides
    are concrete there. Supports are compared only where both are declared.

    Raises
    ------
    ValueError
        If a declared support does not contain the actual one, naming the path.
    """
    from ..core._batch import BatchSpec
    from ..core.constraints import _supports_compatible

    expected_event = getattr(expected, "event_spec", None)
    actual_event = getattr(actual, "event_spec", None)
    if isinstance(expected, RecordSpec) and isinstance(actual, RecordSpec):
        for key, child in expected.items():
            _validate_declared_support(child, actual[key], f"{path}/{key}")
    elif isinstance(expected, BatchSpec) and isinstance(actual, BatchSpec):
        _validate_declared_support(expected.element_spec, actual.element_spec, path)
    elif isinstance(expected_event, OutputSpec) and isinstance(actual_event, OutputSpec):
        components = actual_event.components
        for component, child in expected_event.components.items():
            _validate_declared_support(child, components[component], f"{path}/{component}")
    elif isinstance(expected, NumericArraySpec) and isinstance(actual, NumericArraySpec):
        if (
            expected.support is not None
            and actual.support is not None
            and not _supports_compatible(actual.support, expected.support)
        ):
            raise ValueError(
                f"{path} support {actual.support!r} does not conform to {expected.support!r}"
            )


def _validate_output_values(spec: TermSpec, value: Any, path: str) -> None:
    """Check *value* against the dtype and support each declared array of *spec* states."""
    from ..core._batch import BatchSpec

    if isinstance(spec, BatchSpec):
        from ..core._record_batch import RecordBatch

        element = spec.element_spec
        if isinstance(element, RecordSpec) and isinstance(value, RecordBatch):
            for field, child in element.items():
                column = value._raw_column(field)
                _validate_output_column(child, column, value.batch_shape, f"{path}/{field}")
        elif isinstance(element, NumericArraySpec):
            _validate_output_column(element, value.values, value.batch_shape, path)
    elif isinstance(spec, RecordSpec):
        children = getattr(value, "children", value)
        for name, child in spec.children.items():
            _validate_output_values(child, children[name], f"{path}/{name}")
    elif isinstance(spec, NumericArraySpec):
        _validate_output_dtype(spec, value, path)
        if spec.support is not None and not bool(jnp.all(spec.support.check(value))):
            raise ValueError(f"{path} does not conform to declared support {spec.support!r}")


#: The kinds of numeric dtype; a returned dtype conforms to a declared one of its kind.
_DTYPE_KINDS = (jnp.bool_, jnp.integer, jnp.floating, jnp.complexfloating)


def _dtype_kind(dtype: Any) -> Any:
    return next((kind for kind in _DTYPE_KINDS if jnp.issubdtype(dtype, kind)), None)


def _validate_output_dtype(spec: NumericArraySpec, value: Any, path: str) -> None:
    """Refuse a returned value whose dtype is of another kind than *spec* declares.

    A within-kind width difference conforms, as a ``float32`` value for a
    ``float64`` declaration does, and a cast across kinds, such as an integer
    value for a floating declaration, does not.
    """
    if spec.dtype is None:
        return
    import numpy as np

    from ..core._array_backend import _numpy_dtype_of

    dtype = _numpy_dtype_of(value)
    if (
        dtype is None
        or not np.can_cast(dtype, spec.dtype, casting="same_kind")
        or _dtype_kind(dtype) is not _dtype_kind(spec.dtype)
    ):
        raise ValueError(f"{path} dtype {dtype} does not conform to {spec.dtype}")


def _validate_output_column(
    spec: TermSpec, value: Any, batch_shape: tuple[int, ...], path: str
) -> None:
    if not isinstance(spec, NumericArraySpec):
        return
    from ..core._spec_base import _full_array_shape_or_none

    shape = _full_array_shape_or_none(value)
    if shape != (*batch_shape, *spec.shape):
        raise ValueError(f"{path} has shape {shape}, expected {(*batch_shape, *spec.shape)}")
    _validate_output_values(spec, value, path)


def _validate_function_declarations(
    *,
    function_name: str,
    signature: inspect.Signature,
    input_spec: InputSpec | None,
    construction_bindings: Mapping[str, Any],
) -> None:
    parameters = signature.parameters
    variadic = [p.name for p in parameters.values() if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    if input_spec is not None:
        if variadic:
            raise ValueError(
                f"Function {function_name!r} cannot use an authoritative input_spec "
                f"with variadic parameters {variadic}"
            )
        if set(input_spec) != set(parameters):
            raise ValueError(
                f"Function {function_name!r} input_spec slots {sorted(input_spec)} "
                f"must exactly match signature parameters {sorted(parameters)}"
            )
    unexpected = set(construction_bindings).difference(parameters)
    if unexpected and not any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        raise ValueError(
            f"Function {function_name!r} has invalid construction bindings: "
            f"unexpected names {sorted(unexpected)}"
        )
    if input_spec is None:
        return
    effective: dict[str, int] = {}
    for name, parameter in parameters.items():
        spec = input_spec[name]
        if parameter.default is not inspect.Parameter.empty:
            spec._bind_dims_from_value(
                parameter.default, {}, f"Function {function_name!r} default/{name}"
            )
        if name in construction_bindings:
            value = construction_bindings[name]
            source = "construction binding"
        elif parameter.default is not inspect.Parameter.empty:
            value = parameter.default
            source = "default"
        else:
            continue
        spec._bind_dims_from_value(value, effective, f"Function {function_name!r} {source}/{name}")


#: The check a result declaration with a type hole defers to the return.
_COMPLETED_AT_RETURN = "the result's type is completed from the returned value"

#: The engine's controls, each with its default. A default sample count of None
#: stands for the constructing class's ``DEFAULT_N_BROADCAST_SAMPLES``.
_CONTROL_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "workflow_kind": WorkflowKind.DEFAULT,
        "n_broadcast_samples": None,
        "dispatch": "auto",
        "max_workers": None,
        "include_inputs": False,
        "method": None,
        "exact_only": False,
        "conversions": MappingProxyType({}),
        "method_options": MappingProxyType({}),
        "raw": False,
    }
)

#: The controls a construction keyword of None leaves at their default; None is
#: inadmissible for every other control.
_DEFAULTED_BY_NONE = frozenset(
    {"n_broadcast_samples", "max_workers", "method", "conversions", "method_options"}
)

#: Removed construction keywords, which warn: ``func`` aliases ``fn``, and the rest are ignored.
_REMOVED_KEYWORDS = frozenset({"seed", "input_template", "output_template", "func"})


def _refuse_unknown_controls(controls: Mapping[str, Any]) -> None:
    """Refuse every keyword that is neither an engine control nor a removed keyword.

    Raises
    ------
    TypeError
        Naming each unknown keyword.
    """
    unknown = controls.keys() - _CONTROL_DEFAULTS.keys() - _REMOVED_KEYWORDS
    if unknown:
        raise TypeError(
            f"Unknown Function controls: {sorted(unknown)}; an argument of the wrapped "
            f"function binds at construction through bind="
        )


class Function(Node, TrackedTerm, Annotated):
    """An immutable callable with a frozen signature and optional declarations.

    ``apply`` binds arguments and validates one raw evaluation, returning the
    body's result unchanged. Calling the Function runs the installed workflow
    engine, which adds lifting, dispatch, result identity, and provenance.

    Parameters
    ----------
    name : str
        Required non-empty function label, independent of its output interface.
    fn : Callable
        The wrapped Python callable. Its signature is captured at construction.
    input_spec : InputSpec or Mapping[str, TermSpec] or None
        Authoritative input slots matching fixed signature parameters by name.
        Declared inputs cannot accompany variadic parameters. Defaults and
        construction bindings must satisfy the declaration.
    output_spec : OutputSpec or TermSpec or None
        Authoritative result declaration. A bare RecordSpec exposes its fields;
        any other bare term spec declares a whole term under output_label. A
        named type hole is inferred independently for each call.
    output_label : str or None
        Result label. Defaults to the initial name and survives with_label.
        Whole-term components default to this name, which must then be a Python
        identifier; an explicit OutputSpec can supply a different component.
    differentiable : NumericSpec or None
        The differentiability claim: exactly the numeric input values gradients
        propagate through. None makes no claim.
    bind : Mapping or None
        Construction-time values of the wrapped callable's arguments, overridden
        by call arguments, and the only way to bind an argument at construction.
        An entry keyed by a name the signature lacks goes to its variadic keyword
        parameter.
    module : object or None
        Experimental shared-input container consulted for missing arguments.
    **controls : Any
        The engine's controls, which ``with_options`` revises:

        - ``workflow_kind`` (WorkflowKind): orchestration selection; DEFAULT,
          the default, inherits the workflow configuration.
        - ``n_broadcast_samples`` (int or None): positive sampling-lift count,
          defaulting to the class's ``DEFAULT_N_BROADCAST_SAMPLES``, 128.
        - ``dispatch`` ({"auto", "jax", "sequential", "thread"}): evaluation
          dispatch interpreted by the engine, "auto" by default.
        - ``max_workers`` (int or None): positive thread-worker count, or the
          executor default.
        - ``include_inputs`` (bool): whether the sampling lift retains inputs
          alongside outputs, False by default.
        - ``method`` (str or None): the name of the route that realizes a
          call; None, the default, selects automatically.
        - ``exact_only`` (bool): whether route selection excludes approximate
          routes, False by default.
        - ``conversions`` (Mapping or None): per-parameter conversion settings,
          keyed by parameter name, each a mapping of the converter's settings.
        - ``method_options`` (Mapping or None): the numerical budgets the
          selected method reads, keyed by option name, such as an MCMC
          method's warmup and draw counts; the selected method validates the
          entries when it runs. Empty by default.
        - ``raw`` (bool): whether a call returns its result detached from the
          workflow, False by default.

    Raises
    ------
    TypeError
        For an invalid name, callable, declaration type, workflow kind, or
        worker-count type, a control of the wrong type, or a keyword that is
        no control, which the message names.
    ValueError
        For mismatched input slots, invalid defaults or bindings, unknown
        dispatch, nonpositive worker or sample counts, conversions for a
        parameter the signature lacks, or invalid component names.
    NotImplementedError
        If a differentiability claim is given, which Function does not carry
        yet.

    Notes
    -----
    Legacy constructor keywords emit ``FutureWarning``: ``func`` overrides
    ``fn``; ``seed``, ``input_template``, and ``output_template`` are ignored.
    Use ``workflow_run(seed=...)`` for workflow randomness or ``bind`` for a
    wrapped callable's seed parameter. ``name`` and ``fn`` remain required.
    Only the engine's controls are admitted, since a registered method declares
    no controls of its own: its budgets are entries of ``method_options``.

    ``spec`` contains only input/output declarations. ``with_label`` changes the
    function label and callable metadata; output_label and component names are
    preserved. ``with_options`` returns a shallow copy with revised controls.
    A Function stores only the controls set on it, so ``options`` reads every
    other control's default when it is read.

    The engine reads three declarations from the Function it runs (V.1): what
    each parameter accepts, the result declaration, and the realization. A
    Function declares them by its input declaration and annotations, its
    ``output_spec``, and its body; a subclass realized by routes, as an
    operation is, declares them through ``_roles``, :meth:`_plan_point`, and
    :meth:`_route_candidates`.
    """

    #: The kinds each parameter with a role accepts, each named by its spec class
    #: (VI.0). A parameter with no role is admitted by its input declaration and
    #: lifted as its annotation states, as every parameter of a Function is.
    _roles: Mapping[str, tuple[type[TermSpec], ...]] = MappingProxyType({})

    _signature_info: WorkflowSignatureInfo
    _bind: Mapping[str, Any]
    _module: Any | None
    _implementation: _FunctionImplementation
    _spec: FunctionSpec
    _output_label: str
    _options: Mapping[str, Any]

    DEFAULT_N_BROADCAST_SAMPLES = 128

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        input_spec: InputSpec | Mapping[str, TermSpec] | None = None,
        output_spec: OutputSpec | TermSpec | None = None,
        output_label: str | None = None,
        differentiable: NumericSpec | None = None,
        bind: Mapping[str, Any] | None = None,
        module: Any | None = None,
        **controls: Any,
    ) -> None:
        _refuse_unknown_controls(controls)
        removed = _REMOVED_KEYWORDS.intersection(controls)
        if removed:
            warnings.warn(
                f"Removed Function options {sorted(removed)} detected: func aliases fn; "
                "input_template, output_template and seed are ignored. "
                "Use fn, input_spec and output_spec instead; use workflow_run(seed=...) "
                "or bind={'seed': ...} for a wrapped-function seed.",
                FutureWarning,
                stacklevel=2,
            )
            fn = controls.pop("func", fn) if "func" in removed else fn
            for key in removed - {"func"}:
                controls.pop(key, None)
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn).__name__}")
        self._initialize(
            _CallableFunctionImplementation(fn),
            make_signature_info(fn),
            name,
            input_spec=input_spec,
            output_spec=output_spec,
            output_label=output_label,
            differentiable=differentiable,
            metadata_source=fn,
            bind=bind,
            module=module,
            **controls,
        )

    def _initialize(
        self,
        implementation: _FunctionImplementation,
        signature_info: Any,
        name: str,
        *,
        input_spec: InputSpec | Mapping[str, TermSpec] | None = None,
        output_spec: OutputSpec | TermSpec | None = None,
        output_label: str | None = None,
        differentiable: NumericSpec | None = None,
        metadata_source: Any = None,
        bind: Mapping[str, Any] | None = None,
        module: Any = None,
        **controls: Any,
    ) -> None:
        unknown = controls.keys() - _CONTROL_DEFAULTS.keys()
        if unknown:
            raise TypeError(f"Unknown Function controls: {sorted(unknown)}")
        if differentiable is not None:
            raise NotImplementedError("Function.__init__: the differentiability claim")
        if not isinstance(name, str) or not name:
            raise TypeError("Function requires a non-empty label")
        if output_label is None:
            output_label = name
        if not isinstance(output_label, str) or not output_label:
            raise TypeError("Function output_label must be a non-empty string")
        if input_spec is not None and not isinstance(input_spec, InputSpec):
            if not isinstance(input_spec, Mapping):
                raise TypeError("input_spec must be an InputSpec, a mapping, or None")
            input_spec = InputSpec(input_spec)
        output_spec = _complete_output_spec(output_spec, output_label)
        construction_bindings = dict(bind or {})
        _validate_function_declarations(
            function_name=name,
            signature=signature_info.signature,
            input_spec=input_spec,
            construction_bindings=construction_bindings,
        )
        given = {
            name: value
            for name, value in controls.items()
            if value is not None or name not in _DEFAULTED_BY_NONE
        }
        options = _set_controls(self._control_defaults(), {}, given, signature_info.signature)
        set_attribute = partial(object.__setattr__, self)
        self._init_tracked(name)
        set_attribute("_annotations", {})
        set_attribute("_implementation", implementation)
        set_attribute("_signature_info", signature_info)
        set_attribute("_spec", FunctionSpec(input_spec, output_spec))
        set_attribute("_output_label", output_label)
        set_attribute("_options", MappingProxyType(options))
        set_attribute("_bind", MappingProxyType(construction_bindings))
        set_attribute("_module", module)
        set_attribute("__doc__", getattr(metadata_source, "__doc__", None))
        set_attribute("__name__", name)
        set_attribute("__qualname__", getattr(metadata_source, "__qualname__", name))
        set_attribute(
            "__module__", getattr(metadata_source, "__module__", None) or type(self).__module__
        )
        set_attribute("__signature__", signature_info.signature)
        Node.__init__(self)

    @staticmethod
    def _from_implementation(
        implementation: _FunctionImplementation,
        *,
        signature: inspect.Signature,
        name: str,
        **kwargs: Any,
    ) -> Function:
        """Construct a Function from a private payload and explicit signature."""
        if not callable(getattr(implementation, "invoke", None)):
            raise TypeError(
                "implementation must provide an invoke(bound_inputs, *, context) method"
            )
        instance = object.__new__(Function)
        instance._initialize(
            implementation, make_signature_info_from_signature(signature), name, **kwargs
        )
        return instance

    @property
    def signature(self) -> inspect.Signature:
        """The independently captured Python calling contract."""
        return self._signature_info.signature

    @property
    def spec(self) -> FunctionSpec:
        """The stored function-kind declaration."""
        return self._spec

    @property
    def input_spec(self) -> InputSpec | None:
        """The authoritative input slots, or None when undeclared."""
        return self.spec.input_spec

    @property
    def output_spec(self) -> OutputSpec | None:
        """The authoritative output interface, or None when undeclared."""
        return self.spec.output_spec

    @property
    def output_label(self) -> str:
        """The result label captured at construction."""
        return self._output_label

    @property
    def options(self) -> Mapping[str, Any]:
        """The effective engine controls, separate from domain arguments.

        A control set at construction or by :meth:`with_options` keeps its value,
        and every other control reports the framework's default as it reads now.
        """
        return MappingProxyType(self._control_defaults() | dict(self._options))

    def _control_defaults(self) -> dict[str, Any]:
        """Every control's framework default, the class's sample count included."""
        return dict(_CONTROL_DEFAULTS) | {"n_broadcast_samples": self.DEFAULT_N_BROADCAST_SAMPLES}

    def with_options(self, **controls: Any) -> Self:
        """Return a copy with revised controls, preserving identity and declarations.

        Raises TypeError for unknown controls, including construction metadata
        and seed. None leaves an existing control unchanged. Invalid control
        values raise the same errors as construction.
        """
        unknown = controls.keys() - _CONTROL_DEFAULTS.keys()
        if unknown:
            raise TypeError(f"Unknown Function controls: {sorted(unknown)}")
        revisions = {name: value for name, value in controls.items() if value is not None}
        options = _set_controls(self._control_defaults(), self._options, revisions, self.signature)
        clone = self._shallow_copy()
        object.__setattr__(clone, "_options", MappingProxyType(options))
        return clone

    def check(self, *args: Any, **kwargs: Any) -> Any:
        """Probe a call through the first six steps of the stack, without executing it.

        The probe runs the steps a call with the same arguments would run, from
        resolving the controls to selecting the route, and a view from
        :meth:`with_options` probes under the controls of its call. It executes
        nothing and causes no random event: the body and every conversion stay
        unexecuted.

        Parameters
        ----------
        *args, **kwargs : Any
            The call's arguments, bound to the signature as the call binds them.

        Returns
        -------
        CallReport
            The call's report: for each candidate route whether it is feasible,
            infeasible with a reason, or unresolved with the declarations it
            still needs; the selected route when selection can be decided; the
            planned result declaration; the checks deferred to return; the
            lifted parameters; and the planned conversions.

        Raises
        ------
        TypeError
            If an argument does not bind to the signature.
        ApplicabilityError
            If an argument's kind is not one its parameter accepts, an
            applicability condition fails, or the declarations conflict, as the
            call raises.
        ResolutionError
            If a conversion has no converter, or the ``method`` control names no
            route of a Function realized by routes.
        NotImplementedError
            Until the installed engine provides the probe.
        """
        return _check_engine(self, *args, **kwargs)

    def with_label(self, label: str) -> Self:
        """Relabel the function, preserving output_label and its declaration."""
        renamed = cast(Self, TrackedTerm.with_label(self, label))
        object.__setattr__(renamed, "__name__", label)
        object.__setattr__(renamed, "__qualname__", label)
        return renamed

    def raw(self) -> Callable[..., Any]:
        """Return the wrapped callable, or the raw evaluator of a private payload."""
        if isinstance(self._implementation, _CallableFunctionImplementation):
            return self._implementation.callable
        return self.apply

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate one point, validating declarations and returning the raw result.

        Python binding errors raise TypeError. Input or output declaration
        violations raise ValueError. Dimension bindings are local to this call.
        Existing returned objects retain their identity and metadata. A Function
        realized by routes admits each argument by its role, with no lifting,
        raising ``ApplicabilityError`` for a kind its role refuses; runs the
        route the engine selects, raising ``ResolutionError`` when none
        applies; and returns the raw form of the result.
        """
        with _apply_scope():
            bound = self.signature.bind_partial(*args, **kwargs)
            values = resolve_workflow_values(
                self._signature_info,
                dict(bound.arguments),
                bind=self._bind,
                module=self._module,
                dependency_type=Node,
                workflow_name=self.label,
            )
            _, bindings = _bind_function_inputs(
                function_name=self.label, input_spec=self.input_spec, values=values
            )
            result = self._invoke_resolved(values, context=_FunctionInvocationContext(bindings))
            _validate_function_output(
                function_name=self.label,
                output_spec=self.output_spec,
                result=result,
                bindings=bindings,
            )
            return result

    def _invoke_resolved(
        self, values: Mapping[str, Any], *, context: _FunctionInvocationContext
    ) -> Any:
        """Realize one point with no lifting, as the installed engine does.

        A Function's body runs on *values*; one realized by routes runs the
        route the engine selects and returns the result's raw form.
        """
        return _invoke_engine(self, values, context)

    def _route_candidates(self, controls: Mapping[str, Any]) -> tuple[Any, ...] | None:
        """The routes that may realize one point of a call under *controls*, in selection order.

        A Function's body is its one route, so it lists none and the engine
        runs the body. A subclass realized by routes returns them, each with
        the members :mod:`probpipe.functions._resolution` names.
        """
        return None

    def _plan_point(
        self, values: Mapping[str, Any], controls: Mapping[str, Any]
    ) -> tuple[Any, OutputSpec | None, tuple[str, ...]]:
        """What the routes read at one point: its call object, result declaration, and deferrals.

        A Function's call object is *values*, and its result declaration its
        ``output_spec``, whose type hole defers the type to the return.

        Returns
        -------
        tuple
            The object the routes' checks read, the result declaration, and the
            checks deferred to the return.
        """
        declared = self.output_spec
        deferred = (
            () if declared is not None and declared.spec is not None else (_COMPLETED_AT_RETURN,)
        )
        return values, declared, deferred

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _call_engine(self, *args, **kwargs)

    def __repr__(self) -> str:
        """The public class, the label, the parameters, and the declarations set on the function.

        The result label is shown where it differs from the function's own.
        """
        return term_repr(public_class_name(type(self)), self.label, self._repr_arguments())

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The arguments the repr shows after the label, each by name and formatted value."""
        fields = [("parameters", format_names(self.signature.parameters))]
        if self.output_label != self.label:
            fields.append(("output_label", repr(self.output_label)))
        for side, declaration in (
            ("input_spec", self.input_spec),
            ("output_spec", self.output_spec),
        ):
            if declaration is not None:
                fields.append((side, repr(declaration)))
        return fields


def _set_controls(
    defaults: Mapping[str, Any],
    current: Mapping[str, Any],
    revisions: Mapping[str, Any],
    signature: inspect.Signature,
) -> dict[str, Any]:
    """The controls set once *revisions* revise *current*, validated against the defaults.

    Only set controls are stored, so a default is read when a control is read.

    Raises
    ------
    TypeError, ValueError
        As :func:`_validate_options` raises for the effective controls.
    """
    set_controls = dict(current) | dict(revisions)
    validated = _validate_options(dict(defaults) | set_controls, signature)
    return {name: validated[name] for name in set_controls}


def _validate_options(options: Mapping[str, Any], signature: inspect.Signature) -> dict[str, Any]:
    """The controls with their values validated, and conversions frozen.

    Raises
    ------
    TypeError
        If a control has the wrong type.
    ValueError
        If a control's value is inadmissible, such as a nonpositive count.
    """
    dispatch = options["dispatch"]
    if dispatch not in ("auto", "jax", "sequential", "thread"):
        raise ValueError(f"dispatch must be one of auto, jax, sequential, thread; got {dispatch!r}")
    workers = options["max_workers"]
    if workers is not None:
        if not isinstance(workers, int):
            raise TypeError("max_workers must be a positive integer or None")
        if workers <= 0:
            raise ValueError("max_workers must be a positive integer or None")
        if dispatch != "thread":
            warnings.warn(
                f"max_workers configures only dispatch='thread'; ignoring it for dispatch={dispatch!r}.",
                stacklevel=3,
            )
    if not isinstance(options["workflow_kind"], WorkflowKind):
        raise TypeError("workflow_kind must be a WorkflowKind enum member")
    count = options["n_broadcast_samples"]
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError(f"n_broadcast_samples must be an integer; got {count!r}")
    if count <= 0:
        raise ValueError(f"n_broadcast_samples must be a positive integer; got {count!r}")
    for flag in ("include_inputs", "exact_only", "raw"):
        if not isinstance(options[flag], bool):
            raise TypeError(f"{flag} must be a bool; got {options[flag]!r}")
    method = options["method"]
    if method is not None and (not isinstance(method, str) or not method):
        raise TypeError(f"method must be a non-empty string or None; got {method!r}")
    conversions = options["conversions"]
    if not isinstance(conversions, Mapping) or not all(
        isinstance(settings, Mapping) for settings in conversions.values()
    ):
        raise TypeError("conversions must map parameter names to mappings of converter settings")
    unknown = set(conversions).difference(signature.parameters)
    if unknown:
        raise ValueError(f"conversions name parameters the signature lacks: {sorted(unknown)}")
    method_options = options["method_options"]
    if not isinstance(method_options, Mapping) or not all(
        isinstance(name, str) and name for name in method_options
    ):
        raise TypeError("method_options must map option names to their values")
    return dict(options) | {
        "conversions": MappingProxyType(
            {name: MappingProxyType(dict(settings)) for name, settings in conversions.items()}
        ),
        "method_options": MappingProxyType(dict(method_options)),
    }


def _plain_call(function: Function, *args: Any, **kwargs: Any) -> Any:
    return function.apply(*args, **kwargs)


def _plain_check(function: Function, *args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("Function.check")


def _plain_invoke(
    function: Function, values: Mapping[str, Any], context: _FunctionInvocationContext
) -> Any:
    return function._implementation.invoke(
        values_to_bound_arguments(function.signature, values), context=context
    )


_call_engine: Callable[..., Any] = _plain_call
_check_engine: Callable[..., Any] = _plain_check
_apply_scope: Callable[[], AbstractContextManager[Any]] = nullcontext
_invoke_engine: Callable[..., Any] = _plain_invoke


def install_call_engine(engine: Callable[..., Any]) -> None:
    """Install the engine of the Function call path, once, at package initialization.

    Until installation a call evaluates plainly, as :meth:`Function.apply`
    does. The engine reads the controls a Function carries and agrees with
    plain evaluation on concrete values.

    Parameters
    ----------
    engine : callable
        Called as ``engine(function, *args, **kwargs)`` for every call of a
        Function. It may also provide ``check(function, *args, **kwargs)``,
        which serves :meth:`Function.check`; ``apply_scope()``, which returns
        the context manager :meth:`Function.apply` enters around plain
        evaluation; and ``invoke(function, values, context)``, which realizes
        the one point :meth:`Function.apply` evaluates.

    Raises
    ------
    TypeError
        If *engine* is not callable.
    RuntimeError
        If a different engine is already installed. Installing the same engine
        again changes nothing.
    """
    global _call_engine, _check_engine, _apply_scope, _invoke_engine
    if not callable(engine):
        raise TypeError("The Function call engine must be callable")
    if _call_engine is not _plain_call and _call_engine is not engine:
        raise RuntimeError("The Function call engine is already installed")
    _call_engine = engine
    _check_engine = getattr(engine, "check", _plain_check)
    _apply_scope = getattr(engine, "apply_scope", nullcontext)
    _invoke_engine = getattr(engine, "invoke", _plain_invoke)


# ---------------------------------------------------------------------------
# Function capabilities
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsInverse(Protocol):
    """An invertible map, whose forward is the claiming Function's ``apply``.

    A class whose instances are invertible only in some configurations also
    defines the guard ``_inverse_guard()``, and :func:`is_invertible` reads the
    claim together with it. The guard returns one of:

    - a bool: whether the inverse is available;
    - None: the answer is unresolved;
    - a :class:`~probpipe.core._dispatch.Feasibility`: the full report.
    """

    def _inverse(self, y: Numeric) -> Numeric:
        """The point the forward map sends to *y*."""
        ...


@runtime_checkable
class SupportsLogDetJacobian(Protocol):
    """A map with a tractable log-determinant of its Jacobian."""

    def _log_det_jacobian(self, x: Numeric) -> Array:
        """The log of the absolute Jacobian determinant of the forward map at *x*."""
        ...


def _guard_admits(report: bool | Feasibility | None) -> bool:
    """Whether a guard's report establishes feasibility."""
    if isinstance(report, Feasibility):
        return report.feasible is True
    return report is True


def is_invertible(f: Any) -> bool:
    """Whether *f* claims :class:`SupportsInverse` and its guard admits the claim.

    Parameters
    ----------
    f : Any
        The object to test, usually a :class:`Function`.

    Returns
    -------
    bool
        True when *f* claims the capability and either defines no
        ``_inverse_guard`` or its guard reports the inverse feasible. False when
        *f* does not claim the capability, or its guard reports the inverse
        infeasible or unresolved.
    """
    if not isinstance(f, SupportsInverse):
        return False
    guard = getattr(f, "_inverse_guard", None)
    return True if guard is None else _guard_admits(guard())


@runtime_checkable
class SupportsDifferentiation(Protocol):
    """An object that declares which of its values gradients propagate through.

    The claim is fixed at construction. For a map the template is a sub-schema
    of its numeric input slots, and for a distribution a sub-schema of its
    numeric event schema.
    """

    @property
    def differentiable_template(self) -> NumericSpec:
        """Exactly the values gradients propagate through."""
        ...


def is_differentiable(x: Any, values: NamedTree | None = None) -> bool:
    """Whether gradients propagate through the named values of *x*.

    Parameters
    ----------
    x : Any
        A map or a distribution.
    values : NamedTree or None
        The values to test, by their paths. None tests every numeric value of
        *x*: its numeric input slots for a map, its numeric event values for a
        distribution.

    Returns
    -------
    bool
        True when every value named in *values* lies in the differentiable
        template of *x*, or with no *values* when the template covers every
        numeric value of *x*. False when *x* does not declare
        :class:`SupportsDifferentiation`.

    Raises
    ------
    NotImplementedError
        For an object that declares the capability, until the template
        comparison is implemented.
    """
    if not isinstance(x, SupportsDifferentiation):
        return False
    raise NotImplementedError("values.is_differentiable")
