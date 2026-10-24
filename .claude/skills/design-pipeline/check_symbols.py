"""Check a pipeline's ProbPipe names and call arguments against the installed package.

The ``design-pipeline`` assistant runs this script on each version of a user's
``pipeline.py``, or of a notebook, before it shows the code. The script parses
each file with ``ast`` and checks five things against the ``probpipe`` that this
Python imports:

1. each ``import probpipe...`` and ``from probpipe... import ...`` resolves;
2. each attribute chain that starts at an imported ProbPipe name, such as
   ``pp.condition_on.using``, resolves;
3. the arguments of each call of a ProbPipe callable bind to its signature,
   which ``inspect.signature`` reports;
4. each ``method=`` given to ``condition_on.using`` names an inference method
   registered in this environment, when it is a string or a name that the file
   assigns a string once at its top level, such as ``METHOD = "tfp_nuts"``;
5. the file calls no top-level export that raises ``NotImplementedError`` on
   every call, which ``build_corpus.unimplemented_exports`` finds in the
   source of the imported ``probpipe``.

The script also checks each module beside a file that the file imports, such
as ``report.py`` for a notebook that runs ``import report``. A name that the
file assigns again, for example by ``Normal = ...``, is not checked, because
the script cannot tell which value a later use reads. The script needs only
the standard library, ``probpipe``, and ``build_corpus.py`` beside it.

Usage::

    python check_symbols.py pipeline.py [more .py or .ipynb files]

The script prints one line for each problem, and it exits with status 0 when
there are none, 1 when there is one, and 2 when a file cannot be read or parsed
or ``probpipe`` does not import.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import sys
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_corpus

_UNRESOLVED = object()


@dataclass
class Source:
    """The code of one file, with the label of each line for messages."""

    path: Path
    code: str
    labels: list[str] = field(default_factory=list)

    def where(self, line: int) -> str:
        """The label of the line *line*, as ``file:line`` or ``file[cell n]:line``."""
        if 0 < line <= len(self.labels):
            return self.labels[line - 1]
        return f"{self.path}:{line}"


def read_source(path: Path) -> Source:
    """The code of the script or notebook *path*, with each IPython magic replaced by ``pass``.

    A magic line becomes ``pass`` at its indentation, so that a block whose only
    statement is a magic, such as ``%pip install`` under ``if IN_COLAB:``, still
    parses. The lines that continue a magic after a trailing backslash become
    blank.
    """
    if path.suffix != ".ipynb":
        code = path.read_text()
        return Source(path, code, [f"{path}:{n}" for n in range(1, code.count("\n") + 2)])
    cells = json.loads(path.read_text()).get("cells", [])
    lines: list[str] = []
    labels: list[str] = []
    for index, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        continued = False
        for number, line in enumerate("".join(cell.get("source", "")).splitlines(), start=1):
            stripped = line.lstrip()
            if continued:
                lines.append("")
            elif stripped.startswith(("%", "!")):
                lines.append(line[: len(line) - len(stripped)] + "pass")
            else:
                lines.append(line)
            continued = (continued or stripped.startswith(("%", "!"))) and line.endswith("\\")
            labels.append(f"{path}[cell {index}]:{number}")
    return Source(path, "\n".join(lines) + "\n", labels)


def resolve(dotted: str) -> Any:
    """The object that *dotted* names, importing submodules as needed, or ``_UNRESOLVED``."""
    parts = dotted.split(".")
    try:
        target: Any = importlib.import_module(parts[0])
    except ImportError:
        return _UNRESOLVED
    for index, part in enumerate(parts[1:], start=1):
        if hasattr(target, part):
            target = getattr(target, part)
            continue
        try:
            target = importlib.import_module(".".join(parts[: index + 1]))
        except ImportError:
            return _UNRESOLVED
    return target


def signature_of(obj: Any) -> inspect.Signature | None:
    """The call signature of *obj*, from ``__init__`` for a class whose metaclass hides it."""
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    kinds = {p.kind for p in signature.parameters.values()}
    generic = kinds <= {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    if inspect.isclass(obj) and generic:
        try:
            init = inspect.signature(obj.__init__)
        except (TypeError, ValueError):
            return signature
        parameters = list(init.parameters.values())[1:]
        return init.replace(parameters=parameters)
    return signature


def _registered_methods() -> list[str] | None:
    """The names of the inference methods registered in this environment."""
    registry = resolve("probpipe.inference_method_registry")
    if registry is _UNRESOLVED:
        return None
    return list(registry.list_methods())


@cache
def _unimplemented() -> dict[int, str]:
    """The exports of the imported ``probpipe`` that raise ``NotImplementedError``, by object id."""
    probpipe = resolve("probpipe")
    if probpipe is _UNRESOLVED:
        return {}
    source = Path(probpipe.__file__).resolve().parents[1]
    found = {}
    for name in build_corpus.unimplemented_exports(source):
        target = resolve(f"probpipe.{name}")
        if target is not _UNRESOLVED:
            found[id(target)] = name
    return found


class Checker(ast.NodeVisitor):
    """Collects the problems of one source file."""

    def __init__(self, source: Source, tree: ast.Module) -> None:
        self.source = source
        self.problems: list[str] = []
        self.names_checked = 0
        self.calls_checked = 0
        self.bound: dict[str, str] = {}
        self.reassigned = self._reassigned(tree)
        self.strings = self._top_level_strings(tree)
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node  # type: ignore[attr-defined]

    @staticmethod
    def _reassigned(tree: ast.Module) -> set[str]:
        """The names that *tree* binds other than by an import."""
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                names.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
        return names

    @staticmethod
    def _top_level_strings(tree: ast.Module) -> dict[str, str]:
        """The names that *tree* binds once, at its top level, to a string."""
        stores: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                stores[node.id] = stores.get(node.id, 0) + 1
        return {
            node.targets[0].id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and stores.get(node.targets[0].id) == 1
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        }

    def report(self, node: ast.AST, message: str) -> None:
        self.problems.append(f"{self.source.where(getattr(node, 'lineno', 0))}: {message}")

    # -- imports ----------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] != "probpipe":
                continue
            self.names_checked += 1
            if resolve(alias.name) is _UNRESOLVED:
                self.report(node, f"`import {alias.name}` fails: no such module")
                continue
            if alias.asname:
                self._bind(alias.asname, alias.name)
            else:
                self._bind("probpipe", "probpipe")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level or module.split(".")[0] != "probpipe":
            return
        for alias in node.names:
            if alias.name == "*":
                self.report(node, f"`from {module} import *` hides which names it uses")
                continue
            self.names_checked += 1
            dotted = f"{module}.{alias.name}"
            if resolve(dotted) is _UNRESOLVED:
                self.report(node, f"`from {module} import {alias.name}` fails: no such name")
                continue
            self._bind(alias.asname or alias.name, dotted)

    def _bind(self, name: str, dotted: str) -> None:
        if name not in self.reassigned:
            self.bound[name] = dotted

    # -- names and calls --------------------------------------------------------

    def dotted(self, node: ast.AST) -> str | None:
        """The ProbPipe path that the name or attribute chain *node* spells, if it spells one."""
        attributes: list[str] = []
        while isinstance(node, ast.Attribute):
            attributes.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name) and node.id in self.bound:
            return ".".join([self.bound[node.id], *reversed(attributes)])
        return None

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(getattr(node, "parent", None), ast.Attribute):
            self.generic_visit(node)
            return
        dotted = self.dotted(node)
        if dotted is not None:
            self.names_checked += 1
            if resolve(dotted) is _UNRESOLVED:
                self.report(node, f"`{ast.unparse(node)}` does not resolve: `{dotted}` is missing")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = self.dotted(node.func)
        target = resolve(dotted) if dotted is not None else _UNRESOLVED
        if target is not _UNRESOLVED and callable(target):
            self.calls_checked += 1
            self._check_arguments(node, dotted, target)
            if id(target) in _unimplemented():
                self.report(
                    node,
                    f"`{ast.unparse(node.func)}` is not implemented: every call raises "
                    "NotImplementedError",
                )
            if dotted.endswith("condition_on.using"):
                self._check_method(node)
        self.generic_visit(node)

    def _check_arguments(self, node: ast.Call, dotted: str, target: Any) -> None:
        signature = signature_of(target)
        if signature is None:
            return
        name = ast.unparse(node.func)
        starred = any(isinstance(a, ast.Starred) for a in node.args)
        double = any(k.arg is None for k in node.keywords)
        keywords = {k.arg: None for k in node.keywords if k.arg is not None}
        if starred or double:
            accepts_any = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
            )
            for keyword in keywords:
                parameter = signature.parameters.get(keyword)
                positional_only = (
                    parameter is not None and parameter.kind is parameter.POSITIONAL_ONLY
                )
                if (parameter is None and not accepts_any) or positional_only:
                    self.report(node, f"`{name}` takes no keyword argument `{keyword}`")
            return
        try:
            signature.bind(*[None] * len(node.args), **keywords)
        except TypeError as error:
            self.report(node, f"`{name}{signature}` rejects this call: {error}")

    def _check_method(self, node: ast.Call) -> None:
        for keyword in node.keywords:
            if keyword.arg != "method":
                continue
            if isinstance(keyword.value, ast.Constant):
                method = keyword.value.value
            elif isinstance(keyword.value, ast.Name) and keyword.value.id in self.strings:
                method = self.strings[keyword.value.id]
            else:
                continue
            methods = _registered_methods()
            if methods is not None and method not in methods:
                self.report(
                    node,
                    f"no inference method `{method}` is registered in this "
                    f"environment; the registered methods are {', '.join(methods)}",
                )


def local_modules(path: Path) -> list[Path]:
    """The modules beside *path* that it imports, such as ``report.py`` for ``import report``."""
    tree = ast.parse(read_source(path).code, filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names.add(node.module.split(".")[0])
    return sorted(
        path.parent / f"{name}.py" for name in names if (path.parent / f"{name}.py").is_file()
    )


def check(path: Path) -> tuple[list[str], int, int]:
    """The problems of the file *path*, and the counts of names and calls checked."""
    source = read_source(path)
    tree = ast.parse(source.code, filename=str(path))
    checker = Checker(source, tree)
    checker.visit(tree)
    return checker.problems, checker.names_checked, checker.calls_checked


def main(argv: list[str] | None = None) -> int:
    """Check each file named on the command line, print the problems, and return the status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        probpipe = importlib.import_module("probpipe")
    except ImportError as error:
        print(f"check_symbols.py: probpipe does not import: {error}", file=sys.stderr)
        return 2
    status, names, calls = 0, 0, 0
    paths = list(args.files)
    for path in paths:
        try:
            paths.extend(module for module in local_modules(path) if module not in paths)
            problems, file_names, file_calls = check(path)
        except (OSError, SyntaxError, ValueError) as error:
            print(f"check_symbols.py: cannot check {path}: {error}", file=sys.stderr)
            return 2
        names, calls = names + file_names, calls + file_calls
        for problem in problems:
            print(problem)
        status = max(status, 1 if problems else 0)
    print(
        f"check_symbols.py: probpipe {probpipe.__version__}, {names} names and {calls} calls "
        f"checked, {'problems found' if status else 'no problems'}"
    )
    return status


if __name__ == "__main__":
    sys.exit(main())
