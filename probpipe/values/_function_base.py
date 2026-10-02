"""Function values, their declarations, and the installable call-engine boundary."""

from __future__ import annotations

import inspect
import os
import warnings
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from functools import partial
from types import MappingProxyType
from typing import Any, Literal, Protocol, Self, cast

import jax.numpy as jnp

from ..core._record_spec import RecordSpec
from ..core._spec_base import NumericArraySpec, TermSpec, _unify_specs
from ..core._specs import InputSpec, OutputSpec
from ..core.config import WorkflowKind
from ..core.node import Node
from ..core.tracked import Annotated, TrackedTerm
from ._binding import (
    FunctionSignatureInfo,
    make_signature_info,
    make_signature_info_from_signature,
    resolve_function_values,
    values_to_bound_arguments,
)

_FunctionDispatch = Literal["auto", "jax", "sequential", "thread"]
_WARNING_SKIP_PREFIXES = (os.path.dirname(os.path.dirname(__file__)) + os.sep,)


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

    Examples
    --------
    Bind a dimension shared by the input and output declarations:

    >>> from probpipe import FunctionSpec, InputSpec, NumericArraySpec, OutputSpec
    >>> declared = FunctionSpec(
    ...     InputSpec(x=NumericArraySpec(("n",))),
    ...     OutputSpec(result=NumericArraySpec(("n",))),
    ... )
    >>> actual = FunctionSpec(
    ...     InputSpec(x=NumericArraySpec((3,))),
    ...     OutputSpec(result=NumericArraySpec((3,))),
    ... )
    >>> declared.bind_dims_from_spec(actual) == actual
    True
    >>> declared.free_dims == {"n"}
    True
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
    output_spec: OutputSpec | TermSpec | None, output_name: str
) -> OutputSpec | None:
    if output_spec is None or isinstance(output_spec, OutputSpec):
        return output_spec
    if isinstance(output_spec, RecordSpec):
        return OutputSpec(output_spec)
    if isinstance(output_spec, TermSpec):
        return OutputSpec(**{output_name: output_spec})
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
    """Validate one returned term without wrapping it or changing its declaration."""
    if output_spec is None:
        return None
    spec = output_spec.spec
    if spec is None:
        # A hole imposes no return constraint. The engine infers its kind using
        # the same wrapping rules as an undeclared return (including sequences).
        return output_spec
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
    if isinstance(concrete, FunctionSpec) and concrete.input_spec is not None:
        _validate_function_declarations(
            function_name=function_name,
            signature=(
                result.signature
                if isinstance(result, Function)
                else make_signature_info(result).signature
            ),
            input_spec=concrete.input_spec,
            construction_bindings=result._bind if isinstance(result, Function) else {},
        )
    _validate_output_support(concrete, result, path)
    if actual_spec is None:
        actual_spec = RecordSpec.infer_from({"result": result}).children["result"]
    concrete = _complete_output_metadata(concrete, actual_spec)
    return output_spec._with_spec(concrete)


def _complete_output_metadata(expected: TermSpec, actual: TermSpec) -> TermSpec:
    """Fill unspecified array metadata from a validated result, recursively."""
    from ..core._batch import BatchSpec

    if isinstance(expected, NumericArraySpec) and isinstance(actual, NumericArraySpec):
        return replace(
            expected,
            dtype=actual.dtype if expected.dtype is None else expected.dtype,
            support=actual.support if expected.support is None else expected.support,
        )
    if isinstance(expected, RecordSpec) and isinstance(actual, RecordSpec):
        return RecordSpec(
            {
                name: _complete_output_metadata(child, actual.children[name])
                for name, child in expected.children.items()
            }
        )
    if isinstance(expected, BatchSpec) and isinstance(actual, BatchSpec):
        return replace(
            expected,
            element_spec=_complete_output_metadata(expected.element_spec, actual.element_spec),
        )
    return expected


def _validate_declared_support(expected: TermSpec, actual: TermSpec, path: str) -> None:
    from ..core._batch import BatchSpec
    from ..core.constraints import _supports_compatible
    from ..distributions import DistributionSpec

    if isinstance(expected, RecordSpec) and isinstance(actual, RecordSpec):
        for key, child in expected.items():
            _validate_declared_support(child, actual[key], f"{path}/{key}")
    elif isinstance(expected, BatchSpec) and isinstance(actual, BatchSpec):
        _validate_declared_support(expected.element_spec, actual.element_spec, path)
    elif isinstance(expected, DistributionSpec) and isinstance(actual, DistributionSpec):
        actual_components = actual.event_spec.components
        for name, child in expected.event_spec.components.items():
            actual_child = actual_components[name]
            if child is None or actual_child is None:
                raise ValueError(f"{path}/{name} requires concrete expected and actual term specs")
            _validate_declared_support(child, actual_child, f"{path}/{name}")
    elif isinstance(expected, NumericArraySpec) and isinstance(actual, NumericArraySpec):
        if (
            expected.support is not None
            and actual.support is not None
            and not _supports_compatible(actual.support, expected.support)
        ):
            raise ValueError(
                f"{path} support {actual.support!r} does not conform to {expected.support!r}"
            )


def _validate_output_support(spec: TermSpec, value: Any, path: str) -> None:
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
            _validate_output_support(child, children[name], f"{path}/{name}")
    elif isinstance(spec, NumericArraySpec) and spec.support is not None:
        if not bool(jnp.all(spec.support.check(value))):
            raise ValueError(f"{path} does not conform to declared support {spec.support!r}")


def _validate_output_column(
    spec: TermSpec, value: Any, batch_shape: tuple[int, ...], path: str
) -> None:
    if not isinstance(spec, NumericArraySpec):
        return
    import numpy as np

    from ..core._array_backend import _numpy_dtype_of
    from ..core._spec_base import _full_array_shape_or_none

    shape = _full_array_shape_or_none(value)
    if shape != (*batch_shape, *spec.shape):
        raise ValueError(f"{path} has shape {shape}, expected {(*batch_shape, *spec.shape)}")
    dtype = _numpy_dtype_of(value)
    if spec.dtype is not None and (
        dtype is None or not np.can_cast(dtype, spec.dtype, casting="same_kind")
    ):
        raise ValueError(f"{path} dtype {dtype} does not conform to {spec.dtype}")
    _validate_output_support(spec, value, path)


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


class Function(Node, TrackedTerm, Annotated):
    """An immutable callable with a frozen signature and optional declarations.

    ``apply`` binds arguments and validates one raw evaluation, returning the
    body's result unchanged. Calling the Function runs the installed Function
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
        any other bare term spec declares a whole term under output_name. A
        named type hole is inferred independently for each call.
    output_name : str or None
        Result label. Defaults to the initial name and survives with_name.
        Whole-term components default to this name, which must then contain no
        ``/``. Non-identifier names such as ``Model.fit`` and ``<lambda>`` are
        allowed; an explicit OutputSpec can supply a different component.
    bind : Mapping or None
        Construction-time argument defaults, overridden by call arguments.
    module : object or None
        Experimental shared-input container consulted for missing arguments.
    workflow_kind : WorkflowKind
        Orchestration selection; DEFAULT inherits the orchestration configuration.
    n_broadcast_samples : int or None
        Sampling-lift count, defaulting to 128.
    dispatch : {"auto", "jax", "sequential", "thread"}
        Evaluation dispatch selection interpreted by the engine.
    max_workers : int or None
        Positive thread-worker count, or the executor default.
    include_inputs : bool
        Whether the sampling lift retains inputs alongside outputs.
    **kwargs : Any
        Additional construction bindings. Use bind for a domain argument named
        seed; workflow randomness is configured by workflow_run.

    Raises
    ------
    TypeError
        For an invalid name or output_name, callable, declaration type, orchestration mode, or
        worker-count type.
    ValueError
        For mismatched input slots, invalid defaults or bindings, unknown
        dispatch, nonpositive worker counts, or invalid component names.

    Warns
    -----
    FutureWarning
        Once per supplied legacy keyword: func, seed, input_template, or
        output_template. See Notes for their remaining behavior.
    UserWarning
        If max_workers is supplied with a dispatch other than thread.

    Notes
    -----
    Legacy constructor keywords emit ``FutureWarning``: ``func`` overrides
    ``fn``; ``seed``, ``input_template``, and ``output_template`` are ignored.
    Use ``workflow_run(seed=...)`` for workflow randomness or ``bind`` for a
    wrapped callable's seed parameter. ``name`` and ``fn`` remain required.
    Old templates do not install validation; ``func`` cannot be supplied without
    ``fn``. Each warning identifies the supplied option at the user's call site.

    ``spec`` contains only input/output declarations. ``with_name`` changes the
    function label and callable metadata; output_name and component names are
    preserved. ``with_options`` returns a shallow copy with revised controls.
    """

    _signature_info: FunctionSignatureInfo
    _bind: Mapping[str, Any]
    _module: Any | None
    _implementation: _FunctionImplementation
    _spec: FunctionSpec
    _output_name: str
    _options: Mapping[str, Any]

    DEFAULT_N_BROADCAST_SAMPLES = 128

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        input_spec: InputSpec | Mapping[str, TermSpec] | None = None,
        output_spec: OutputSpec | TermSpec | None = None,
        output_name: str | None = None,
        workflow_kind: WorkflowKind = WorkflowKind.DEFAULT,
        bind: Mapping[str, Any] | None = None,
        module: Any | None = None,
        n_broadcast_samples: int | None = None,
        dispatch: _FunctionDispatch = "auto",
        max_workers: int | None = None,
        include_inputs: bool = False,
        **kwargs: Any,
    ) -> None:
        removed = {"seed", "input_template", "output_template", "func"}.intersection(kwargs)
        if removed:
            messages = {
                "func": "aliases fn and overrides it; fn remains required. Use fn instead.",
                "input_template": "is ignored; its validation is not retained. Use input_spec instead.",
                "output_template": "is ignored; its validation is not retained. Use output_spec instead.",
                "seed": "is ignored. Use workflow_run(seed=...) or bind={'seed': ...} for a wrapped-function seed.",
            }
            for key in sorted(removed):
                warnings.warn(
                    f"Legacy Function option {key!r} {messages[key]}",
                    FutureWarning,
                    skip_file_prefixes=_WARNING_SKIP_PREFIXES,
                )
            fn = kwargs.pop("func", fn) if "func" in removed else fn
            for key in removed - {"func"}:
                kwargs.pop(key, None)
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn).__name__}")
        self._initialize(
            _CallableFunctionImplementation(fn),
            make_signature_info(fn),
            name,
            input_spec=input_spec,
            output_spec=output_spec,
            output_name=output_name,
            metadata_source=fn,
            bind=dict(bind or {}) | kwargs,
            module=module,
            workflow_kind=workflow_kind,
            n_broadcast_samples=n_broadcast_samples,
            dispatch=dispatch,
            max_workers=max_workers,
            include_inputs=include_inputs,
        )

    def _initialize(
        self,
        implementation: _FunctionImplementation,
        signature_info: Any,
        name: str,
        *,
        input_spec: InputSpec | Mapping[str, TermSpec] | None = None,
        output_spec: OutputSpec | TermSpec | None = None,
        output_name: str | None = None,
        metadata_source: Any = None,
        bind: Mapping[str, Any] | None = None,
        module: Any = None,
        workflow_kind: WorkflowKind = WorkflowKind.DEFAULT,
        n_broadcast_samples: int | None = None,
        dispatch: str = "auto",
        max_workers: int | None = None,
        include_inputs: bool = False,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("Function requires a non-empty name")
        if output_name is None:
            output_name = name
        if not isinstance(output_name, str) or not output_name:
            raise TypeError("Function output_name must be a non-empty string")
        if input_spec is not None and not isinstance(input_spec, InputSpec):
            if not isinstance(input_spec, Mapping):
                raise TypeError("input_spec must be an InputSpec, a mapping, or None")
            input_spec = InputSpec(input_spec)
        output_spec = _complete_output_spec(output_spec, output_name)
        construction_bindings = dict(bind or {})
        _validate_function_declarations(
            function_name=name,
            signature=signature_info.signature,
            input_spec=input_spec,
            construction_bindings=construction_bindings,
        )
        options = dict(
            workflow_kind=workflow_kind,
            n_broadcast_samples=self.DEFAULT_N_BROADCAST_SAMPLES
            if n_broadcast_samples is None
            else n_broadcast_samples,
            dispatch=dispatch,
            max_workers=max_workers,
            include_inputs=include_inputs,
        )
        _validate_options(options)
        set_attribute = partial(object.__setattr__, self)
        self._init_tracked(name)
        set_attribute("_annotations", {})
        set_attribute("_implementation", implementation)
        set_attribute("_signature_info", signature_info)
        set_attribute("_spec", FunctionSpec(input_spec, output_spec))
        set_attribute("_output_name", output_name)
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
    def output_name(self) -> str:
        """The result label captured at construction."""
        return self._output_name

    @property
    def options(self) -> Mapping[str, Any]:
        """Read-only engine controls, separate from domain arguments."""
        return self._options

    @property
    def effective_workflow_kind(self) -> WorkflowKind:
        """Resolve orchestration from instance controls and current global config.

        Recomputed on each access. Requested Prefect modes warn and fall back
        to OFF when Prefect is unavailable. Before engine installation, plain
        evaluation always uses OFF.
        """
        return _workflow_kind_resolver(self)

    def with_options(self, **controls: Any) -> Self:
        """Return a reusable shallow copy with revised execution controls.

        Parameters
        ----------
        **controls : Any
            Any of workflow_kind, n_broadcast_samples, dispatch, max_workers,
            and include_inputs. None leaves that setting unchanged. The
            revised controls apply to every call of the returned copy.

        Returns
        -------
        Self
            A distinct Function with the same label, output_name, signature,
            declarations, implementation, and provenance. Its annotations
            container is independent; the original's controls are unchanged.

        Raises
        ------
        TypeError
            For unknown controls (including seed and construction metadata),
            a non-WorkflowKind mode, or a non-integer worker count.
        ValueError
            For an unknown dispatch or nonpositive worker count. Broadcast
            sample counts are validated when a distribution lift executes.

        Warns
        -----
        UserWarning
            If the resulting controls specify max_workers outside thread dispatch.
        """
        unknown = controls.keys() - self.options.keys()
        if unknown:
            raise TypeError(f"Unknown Function controls: {sorted(unknown)}")
        options = dict(self.options) | {k: v for k, v in controls.items() if v is not None}
        _validate_options(options)
        clone = self._shallow_copy()
        object.__setattr__(clone, "_options", MappingProxyType(options))
        return clone

    def with_name(self, name: str) -> Self:
        """Return a shallow copy with a new function label.

        Parameters
        ----------
        name : str
            Non-empty label for the copy and its Python callable metadata.

        Returns
        -------
        Self
            A distinct Function retaining output_name, declarations, controls,
            signature, and implementation. Its annotations container is
            independent. Rename provenance points to the original when
            provenance recording is enabled; the original is unchanged.

        Raises
        ------
        TypeError
            If name is not a non-empty string.
        """
        return cast(Self, TrackedTerm.with_name(self, name))

    def _with_name(self, name: str) -> Self:
        """Relabel a copy's Python names without rename provenance."""
        renamed = cast(Self, TrackedTerm._with_name(self, name))
        object.__setattr__(renamed, "__name__", name)
        object.__setattr__(renamed, "__qualname__", name)
        return renamed

    def raw(self) -> Callable[..., Any]:
        """Return the wrapped callable, or the raw evaluator of a private payload."""
        if isinstance(self._implementation, _CallableFunctionImplementation):
            return self._implementation.callable
        return self.apply

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate one point, validating declarations and returning the raw result.

        Parameters
        ----------
        *args : Any
            Positional arguments bound against the frozen Python signature.
        **kwargs : Any
            Domain keyword arguments. Missing inputs are resolved from
            construction bindings, the optional Module, and Python defaults.

        Returns
        -------
        Any
            The implementation's exact result after declaration validation,
            preserving identity, shape, annotations, and provenance. There is
            no lifting, batching, result wrapping, or result renaming.

        Raises
        ------
        TypeError
            If Python argument binding or missing-input resolution fails.
        ValueError
            If input or output kinds, structures, dimensions, dtypes, or
            declared supports disagree, or a returned callable's signature
            cannot satisfy its declared input slots.
        RuntimeError
            If evaluation violates workflow scope or replay admission rules.

        Notes
        -----
        Dimension bindings are local to this evaluation; stored declarations
        are unchanged. A returned callable is not executed for validation.
        Exceptions from the wrapped implementation propagate. Support checks
        require concrete values and can raise tracer errors inside JAX transforms.
        """
        with _apply_scope():
            bound = self.signature.bind_partial(*args, **kwargs)
            values = resolve_function_values(
                self._signature_info,
                dict(bound.arguments),
                bind=self._bind,
                module=self._module,
                dependency_type=Node,
                function_name=self.name,
            )
            _, bindings = _bind_function_inputs(
                function_name=self.name, input_spec=self.input_spec, values=values
            )
            result = self._invoke_resolved(values, context=_FunctionInvocationContext(bindings))
            _validate_function_output(
                function_name=self.name,
                output_spec=self.output_spec,
                result=result,
                bindings=bindings,
            )
            return result

    def _invoke_resolved(
        self, values: Mapping[str, Any], *, context: _FunctionInvocationContext
    ) -> Any:
        return self._implementation.invoke(
            values_to_bound_arguments(self.signature, values), context=context
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate through the installed engine, lifting inputs when needed.

        Parameters
        ----------
        *args : Any
            Positional domain inputs interpreted using the frozen signature.
        **kwargs : Any
            Keyword domain inputs. Set engine controls with with_options.

        Returns
        -------
        TrackedTerm or Any
            A point result wrapped in its value kind under output_name. An
            existing tracked result is shallow-copied, with shared value data,
            independent annotations, and the current call's provenance.
            A batch sweep returns the result kind's batch form, with swept
            axes preceding each point's shape or existing batch levels.
            Distribution lifting returns an output law, or an input/output
            joint when include_inputs is true. Explicit declarations determine
            component exposure independently of the result label.
            Before engine installation, this method returns the raw apply result.

        Raises
        ------
        TypeError
            For argument binding errors, unsupported input/output types, or
            invalid broadcast sample-count types.
        ValueError
            For declaration violations, incompatible sweep levels or shapes,
            nonpositive broadcast sample counts, or unsupported explicit JAX
            dispatch (including concrete output-support checks).
        RuntimeError
            For workflow scope, managed-execution, or replay admission failures.

        Notes
        -----
        User implementation exceptions propagate. Prefect modes warn and fall
        back to OFF when Prefect is unavailable; automatic dispatch may use a
        row-wise route when JAX cannot execute the call.
        """
        return _call_engine(self, *args, **kwargs)


def _validate_options(options: Mapping[str, Any]) -> None:
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


def _plain_call(function: Function, /, *args: Any, **kwargs: Any) -> Any:
    return function.apply(*args, **kwargs)


def _plain_workflow_kind(function: Function, /) -> WorkflowKind:
    """Plain evaluation has no orchestration, regardless of stored controls."""
    return WorkflowKind.OFF


_call_engine: Callable[..., Any] = _plain_call
_apply_scope: Callable[[], AbstractContextManager[Any]] = nullcontext
_workflow_kind_resolver: Callable[[Function], WorkflowKind] = _plain_workflow_kind


def install_call_engine(
    engine: Callable[..., Any],
    *,
    apply_scope: Callable[[], AbstractContextManager[Any]] = nullcontext,
    workflow_kind_resolver: Callable[[Function], WorkflowKind] = _plain_workflow_kind,
) -> None:
    """Install the process's Function call engine once at package initialization.

    Parameters
    ----------
    engine : Callable
        Receives the Function as a positional-only first parameter followed
        by its domain call arguments, leaving keywords such as ``function``
        available to the wrapped callable.
    apply_scope : Callable, optional
        Zero-argument context-manager factory for raw-evaluation admission.
        Defaults to nullcontext.
    workflow_kind_resolver : Callable, optional
        Maps a Function to its effective WorkflowKind on each property access.
        Defaults to OFF for plain evaluation.

    Returns
    -------
    None
        Installs the callbacks. Reinstalling the same engine is a no-op,
        including when different callbacks are supplied.

    Raises
    ------
    TypeError
        If engine is not callable.
    RuntimeError
        If a different engine has already been installed. Failed installation
        leaves all existing callbacks unchanged.

    Notes
    -----
    Before installation, Function calls perform plain apply. Callback
    installation keeps the value-layer base independent of the engine.
    """
    global _call_engine, _apply_scope, _workflow_kind_resolver
    if not callable(engine):
        raise TypeError("The Function call engine must be callable")
    if _call_engine is engine:
        return
    if _call_engine is not _plain_call:
        raise RuntimeError("The Function call engine is already installed")
    _call_engine = engine
    _apply_scope = apply_scope
    _workflow_kind_resolver = workflow_kind_resolver
