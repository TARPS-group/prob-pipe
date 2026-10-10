"""Experimental containers of Functions sharing inputs and dependencies."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ..values import _binding

try:
    from graphviz import Digraph
except ImportError:
    Digraph = None

from ..core.config import WorkflowKind
from ..core.node import Node
from ..values import Function


def workflow_method(func: Callable):
    """Mark a method as an experimental workflow method for :class:`Module` subclasses.

    Methods decorated with ``@workflow_method`` are automatically
    converted to :class:`Function` instances when the
    ``Module`` is instantiated.
    """
    func._is_function_method = True
    return func


def abstract_workflow_method(func: Callable):
    """Mark a method as an experimental abstract workflow interface.

    Combines ``@abstractmethod`` with ``@workflow_method`` so that
    :class:`AbstractModule` subclasses can declare workflow-shaped
    interfaces without providing implementations.
    """
    return abstractmethod(workflow_method(func))


class Module(Node):
    """
    Experimental container for workflow nodes with shared inputs and child nodes.

    New user-facing API:
        MyModule(data=data_node, horizon=30, alpha=0.1)

    Internally:
        - kwargs whose values are Node instances become child_nodes
        - everything else becomes inputs

    Parameters
    ----------
    workflow_kind : WorkflowKind
        Prefect orchestration mode propagated to workflow methods built
        from this module.
    **kwargs : Any
        Shared child nodes and inputs available to workflow methods.

    Raises
    ------
    TypeError
        If ``workflow_kind`` is not a ``WorkflowKind`` enum member.
    """

    def __init__(
        self,
        *,
        workflow_kind: WorkflowKind = WorkflowKind.DEFAULT,
        **kwargs: Any,
    ):
        if not isinstance(workflow_kind, WorkflowKind):
            raise TypeError(
                f"workflow_kind must be a WorkflowKind enum member, "
                f"got {type(workflow_kind).__name__}"
            )
        self._workflow_kind = workflow_kind
        super().__init__(**kwargs)
        # validate abstract workflow implementations before wrapping
        self._validate_abstract_function_implementations()

        self._build_functions()

    def _build_functions(self):
        """
        Replace @workflow_method methods with Function instances.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not callable(attr) or not getattr(attr, "_is_function_method", False):
                continue

            func = attr

            # skip abstract workflows
            if getattr(func, "__isabstractmethod__", False):
                continue

            function_instance = Function(
                func,
                output_label=func.__name__,
                workflow_kind=self._workflow_kind,
                label=f"{self.__class__.__name__}.{func.__name__}",
                module=self,
            )

            setattr(self, attr_name, function_instance)

    def dag(self):
        """Return a Graphviz DAG visualization of this module."""
        if Digraph is None:
            raise ImportError(
                "graphviz is required for dag visualization. "
                "Install it with: pip install probpipe[viz]"
            )
        dot = Digraph(
            name=self.__class__.__name__,
            graph_attr={
                "rankdir": "LR",
                "fontsize": "12",
                "fontname": "Helvetica",
            },
            node_attr={
                "fontname": "Helvetica",
                "fontsize": "11",
            },
        )

        # -------------------------
        # Child nodes (outside)
        # -------------------------
        for name in self._child_nodes:
            dot.node(
                name,
                label=name,
                shape="ellipse",
                style="filled",
                fillcolor="#E8E8E8",
            )

        # -------------------------
        # Module cluster
        # -------------------------
        with dot.subgraph(name=f"cluster_{self.__class__.__name__}") as cluster:
            cluster.attr(
                label=self.__class__.__name__,
                style="rounded",
                color="#4F81BD",
                fontname="Helvetica-Bold",
                fontsize="12",
            )

            # Function nodes inside the module
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if not isinstance(attr, Function):
                    continue

                function_name = attr._label  # e.g. PM25ForecastingModule.fit
                function_label = function_name.split(".")[-1]

                cluster.node(
                    function_name,
                    label=function_label,
                    shape="box",
                    style="filled",
                    fillcolor="#C6DBEF",
                )

        # -------------------------
        # Dependency edges
        # -------------------------
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not isinstance(attr, Function):
                continue

            function_name = attr._label

            # Infer dependencies from workflow signature
            # (Functions don't store child_nodes; they resolve dependencies at runtime)
            for param_name in attr._signature_info.param_names:
                is_dependency = _binding.is_dependency_param(
                    attr._signature_info,
                    param_name,
                    dependency_type=Node,
                )
                if is_dependency and param_name in self._child_nodes:
                    dot.edge(param_name, function_name)

        return dot

    def _validate_abstract_function_implementations(self) -> None:
        """
        Ensure that any abstract workflow interfaces in the MRO are implemented
        by a concrete workflow with a compatible signature.

        This prevents a common failure mode:
          - base class declares @abstract_workflow_method interface
          - subclass defines a method with same name but forgets @workflow_method
          - or implements it with a mismatched signature
        """
        cls = self.__class__

        # Walk MRO to find abstract workflow interfaces
        for base in cls.mro():
            for name, obj in base.__dict__.items():
                if not callable(obj):
                    continue
                if not getattr(obj, "_is_function_method", False):
                    continue
                if not getattr(obj, "__isabstractmethod__", False):
                    continue

                abstract_func = obj  # unbound function

                # Get the attribute as seen on the instance (could be method override)
                impl_attr = getattr(self, name, None)
                if impl_attr is None:
                    continue  # ABCMeta will usually catch this on AbstractModule anyway

                # If still abstract, ABCMeta will also catch it; but this provides better errors
                if getattr(impl_attr, "__isabstractmethod__", False):
                    raise TypeError(f"{cls.__name__} does not implement abstract workflow '{name}'")

                # Must be marked as workflow (@workflow_method)
                if not getattr(impl_attr, "_is_function_method", False):
                    raise TypeError(
                        f"{cls.__name__}.{name} implements an abstract workflow interface "
                        f"but is not marked with @workflow_method"
                    )

                # Compare signatures (use unbound function signatures to include 'self')
                impl_func = impl_attr.__func__ if hasattr(impl_attr, "__func__") else impl_attr

                self._assert_function_signature_compatible(
                    abstract_func=abstract_func,
                    impl_func=impl_func,
                    name=name,
                )

    @staticmethod
    def _assert_function_signature_compatible(
        *,
        abstract_func: Callable,
        impl_func: Callable,
        name: str,
    ) -> None:
        abs_sig = inspect.signature(abstract_func)
        impl_sig = inspect.signature(impl_func)

        def drop_self(sig: inspect.Signature):
            params = list(sig.parameters.values())
            if params and params[0].name == "self":
                params = params[1:]
            return params

        abs_params = drop_self(abs_sig)
        impl_params = drop_self(impl_sig)

        # Build dict for implementation params by name (supports keyword usage)
        impl_by_name = {p.name: p for p in impl_params}

        # Require: every abstract param exists in impl with same kind
        for ap in abs_params:
            ip = impl_by_name.get(ap.name)
            if ip is None:
                raise TypeError(
                    f"Function '{name}' implementation is missing parameter '{ap.name}'.\n"
                    f"Expected (abstract): {abs_sig}\n"
                    f"Got (impl):          {impl_sig}"
                )
            if ip.kind != ap.kind:
                raise TypeError(
                    f"Function '{name}' parameter '{ap.name}' kind mismatch.\n"
                    f"Expected (abstract): {abs_sig}\n"
                    f"Got (impl):          {impl_sig}"
                )


class AbstractModule(Module, ABC):
    """
    Experimental base class for modules that declare workflow interfaces via @abstract_workflow_method.

    ABCMeta will prevent instantiation until all abstract workflows are implemented
    by a concrete subclass.
    """

    pass
