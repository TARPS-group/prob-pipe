"""Portable callable-definition anchors for workflow RNG replay."""

from __future__ import annotations

import base64
import dataclasses
import functools
import hashlib
import importlib
import inspect
import json
import os
import sys
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ._function_contract import _CallableFunctionImplementation
from .constraints import Constraint
from .event_template import EventTemplate

_CALLABLE_DEFINITION_ABI = "probpipe.callable_definition/v1"
_CALLABLE_DEFINITION_DOMAIN = b"ProbPipe-callable-definition-v1\0"
_PROBPIPE_REPLAY_ABI = "probpipe.replay/v1"
_PYTHON_REPLAY_ABI = f"{sys.implementation.name}-{sys.version_info.major}.{sys.version_info.minor}"


class _UnsupportedDefinition(TypeError):
    """Internal signal that a definition cannot be encoded strongly."""


@dataclass(frozen=True)
class CallableAnchor:
    """Immutable supported-or-closed callable replay anchor."""

    supported: bool
    form: str
    module: str | None
    qualname: str | None
    sha256: str | None = None
    signature_and_templates_json: str | None = None
    source_location: str | None = None
    source_artifact_digest: str | None = None

    def controls(self) -> dict[str, Any]:
        """Return the exact JSON-native replay control record."""
        result: dict[str, Any] = {
            "supported": self.supported,
            "module": self.module,
            "qualname": self.qualname,
            "definition_abi": _CALLABLE_DEFINITION_ABI,
        }
        if not self.supported:
            result["form"] = self.form
            return result
        if self.sha256 is None or self.signature_and_templates_json is None:
            raise RuntimeError("supported callable anchor is incomplete")
        result.update(
            {
                "sha256": self.sha256,
                "signature_and_templates": json.loads(self.signature_and_templates_json),
                "python_replay_abi": _PYTHON_REPLAY_ABI,
                "probpipe_replay_abi": _PROBPIPE_REPLAY_ABI,
            }
        )
        return result

    def diagnostics(self) -> dict[str, str]:
        """Return source-artifact observations, which never admit replay."""
        result: dict[str, str] = {}
        if self.source_location is not None:
            result["source_location"] = self.source_location
        if self.source_artifact_digest is not None:
            result["source_artifact_digest"] = self.source_artifact_digest
        return result


def capture_function_anchor(function: Any) -> CallableAnchor:
    """Capture a strong anchor or a closed unsupported-form record."""
    implementation = getattr(function, "_implementation", None)
    if not isinstance(implementation, _CallableFunctionImplementation):
        return _unsupported_anchor(function, None, "private_function_implementation")
    candidate = implementation.callable
    source_location, source_digest = _source_artifact(candidate)
    form = _callable_form(candidate)
    if form != "python_function":
        return _unsupported_anchor(
            function,
            candidate,
            form,
            source_location=source_location,
            source_artifact_digest=source_digest,
        )

    module_name = getattr(candidate, "__module__", None)
    qualname = getattr(candidate, "__qualname__", None)
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        return _unsupported_anchor(function, candidate, "missing_import_identity")
    if "<locals>" in qualname:
        return _unsupported_anchor(function, candidate, "local_function")
    if candidate.__closure__ or candidate.__code__.co_freevars:
        return _unsupported_anchor(function, candidate, "closure")
    if not _resolves_to_callable(function, candidate, module_name, qualname):
        return _unsupported_anchor(function, candidate, "module_resolution_mismatch")

    try:
        signature_and_templates = _signature_and_templates(function, candidate)
        definition = {
            "code": _canonical_value(candidate.__code__),
            "defaults": _canonical_value(candidate.__defaults__),
            "kwdefaults": _canonical_value(candidate.__kwdefaults__),
            "annotations": _canonical_value(candidate.__annotations__),
            "signature_and_templates": signature_and_templates,
        }
        encoded = _canonical_json(definition)
    except _UnsupportedDefinition:
        return _unsupported_anchor(
            function,
            candidate,
            "unsupported_definition_state",
            source_location=source_location,
            source_artifact_digest=source_digest,
        )

    return CallableAnchor(
        supported=True,
        form="python_function",
        module=module_name,
        qualname=qualname,
        sha256=hashlib.sha256(_CALLABLE_DEFINITION_DOMAIN + encoded).hexdigest(),
        signature_and_templates_json=_canonical_json(signature_and_templates).decode("utf-8"),
        source_location=source_location,
        source_artifact_digest=source_digest,
    )


def _callable_form(candidate: Any) -> str:
    if isinstance(candidate, functools.partial):
        return "partial"
    if inspect.ismethod(candidate):
        return "bound_method"
    if inspect.isclass(candidate):
        return "class"
    if inspect.isbuiltin(candidate):
        return "builtin"
    if inspect.isfunction(candidate):
        if candidate.__name__ == "<lambda>":
            return "lambda"
        if candidate.__closure__ or candidate.__code__.co_freevars:
            return "closure"
        if "<locals>" in candidate.__qualname__:
            return "local_function"
        return "python_function"
    return "callable_object" if callable(candidate) else "non_callable"


def _unsupported_anchor(
    function: Any,
    candidate: Any,
    form: str,
    *,
    source_location: str | None = None,
    source_artifact_digest: str | None = None,
) -> CallableAnchor:
    module = getattr(candidate, "__module__", None) or getattr(function, "__module__", None)
    qualname = getattr(candidate, "__qualname__", None) or getattr(function, "__qualname__", None)
    return CallableAnchor(
        supported=False,
        form=form,
        module=module if isinstance(module, str) else None,
        qualname=qualname if isinstance(qualname, str) else None,
        source_location=source_location,
        source_artifact_digest=source_artifact_digest,
    )


def _resolves_to_callable(
    function: Any,
    candidate: Any,
    module_name: str,
    qualname: str,
) -> bool:
    try:
        resolved: Any = importlib.import_module(module_name)
        for segment in qualname.split("."):
            resolved = getattr(resolved, segment)
    except (AttributeError, ImportError, ValueError):
        return False
    if resolved is candidate:
        return True
    implementation = getattr(resolved, "_implementation", None)
    if isinstance(implementation, _CallableFunctionImplementation):
        return implementation.callable is candidate
    return False


def _signature_and_templates(function: Any, candidate: Any) -> dict[str, Any]:
    signature = inspect.signature(candidate, follow_wrapped=False)
    parameters = [
        {
            "name": parameter.name,
            "kind": parameter.kind.name,
            "default": _canonical_value(parameter.default),
            "annotation": _canonical_value(parameter.annotation),
        }
        for parameter in signature.parameters.values()
    ]
    return {
        "parameters": parameters,
        "return_annotation": _canonical_value(signature.return_annotation),
        "input_template": _canonical_value(getattr(function, "_input_template", None)),
        "output_template": _canonical_value(getattr(function, "_output_template", None)),
    }


def _canonical_value(value: Any) -> dict[str, Any]:
    if value is inspect.Parameter.empty or value is inspect.Signature.empty:
        return {"tag": "empty"}
    if value is None:
        return {"tag": "none"}
    if value is Ellipsis:
        return {"tag": "ellipsis"}
    if isinstance(value, bool):
        return {"tag": "bool", "value": value}
    if isinstance(value, int):
        return {"tag": "int", "value": str(value)}
    if isinstance(value, float):
        return {"tag": "float", "value": value.hex()}
    if isinstance(value, complex):
        return {
            "tag": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if isinstance(value, str):
        return {"tag": "str", "value": value}
    if isinstance(value, bytes):
        return {"tag": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return {"tag": "tuple", "items": [_canonical_value(item) for item in value]}
    if isinstance(value, list):
        return {"tag": "list", "items": [_canonical_value(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        items.sort(key=_canonical_json)
        return {
            "tag": "frozenset" if isinstance(value, frozenset) else "set",
            "items": items,
        }
    if isinstance(value, Mapping):
        entries = [[_canonical_value(key), _canonical_value(item)] for key, item in value.items()]
        entries.sort(key=lambda entry: _canonical_json(entry[0]))
        return {"tag": "mapping", "entries": entries}
    if isinstance(value, types.CodeType):
        return _canonical_code(value)
    if isinstance(value, np.dtype):
        return {
            "tag": "numpy_dtype",
            "value": value.str,
            "descr": _canonical_value(value.descr) if value.fields else {"tag": "none"},
        }
    if isinstance(value, np.generic):
        return _canonical_array(np.asarray(value), tag="numpy_scalar")
    if isinstance(value, np.ndarray):
        return _canonical_array(value, tag="numpy_array")
    if isinstance(value, EventTemplate):
        return {
            "tag": "event_template",
            "type": _type_identity(type(value)),
            "children": [[name, _canonical_value(child)] for name, child in value.children.items()],
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "tag": "dataclass",
            "type": _type_identity(type(value)),
            "fields": [
                [field.name, _canonical_value(getattr(value, field.name))]
                for field in dataclasses.fields(value)
            ],
        }
    if isinstance(value, Constraint):
        return {
            "tag": "constraint",
            "type": _type_identity(type(value)),
            "state": _canonical_value(vars(value)),
        }
    if isinstance(value, Enum):
        return {
            "tag": "enum",
            "type": _type_identity(type(value)),
            "name": value.name,
        }
    if isinstance(value, (types.GenericAlias, types.UnionType)):
        return {
            "tag": "generic",
            "origin": _canonical_value(typing.get_origin(value)),
            "args": _canonical_value(typing.get_args(value)),
        }
    origin = typing.get_origin(value)
    if origin is not None:
        return {
            "tag": "typing",
            "origin": _canonical_value(origin),
            "args": _canonical_value(typing.get_args(value)),
        }
    if isinstance(value, type):
        return {"tag": "type", "value": _type_identity(value)}
    raise _UnsupportedDefinition(f"unsupported callable-definition value {type(value).__name__}")


def _canonical_code(code: types.CodeType) -> dict[str, Any]:
    return {
        "tag": "code",
        "name": code.co_name,
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code": _canonical_value(code.co_code),
        "exceptiontable": _canonical_value(code.co_exceptiontable),
        "consts": _canonical_value(code.co_consts),
        "names": _canonical_value(code.co_names),
        "varnames": _canonical_value(code.co_varnames),
        "freevars": _canonical_value(code.co_freevars),
        "cellvars": _canonical_value(code.co_cellvars),
    }


def _canonical_array(value: np.ndarray, *, tag: str) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    dtype = contiguous.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and sys.byteorder == "big"):
        little_dtype = dtype.newbyteorder("<")
        contiguous = contiguous.astype(little_dtype, copy=False)
        dtype = little_dtype
    return {
        "tag": tag,
        "dtype": dtype.str,
        "shape": list(contiguous.shape),
        "base64": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
    }


def _type_identity(value: type) -> dict[str, str]:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise _UnsupportedDefinition("type has no import identity")
    return {"module": module, "qualname": qualname}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _source_artifact(candidate: Any) -> tuple[str | None, str | None]:
    try:
        source = inspect.getsourcefile(candidate)
        if source is None:
            return None, None
        path = Path(os.path.abspath(source))
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError):
        return None, None
