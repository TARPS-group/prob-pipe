"""Reverse-dependency import graph for ProbPipe's CI test/notebook selection.

This module is the single source of truth for the AST import-graph logic that
``.github/workflows/ci.yml`` uses to decide which artifacts a source change
affects. It previously lived as two near-identical inline heredocs (one in the
``test`` job, one in the ``notebooks`` job); see issue #266.

It builds a reverse import graph over ``probpipe/`` by AST-parsing every module
and answers two questions:

* which probpipe modules are *transitively affected* by a set of changed source
  files (``affected_modules``), and
* which notebooks under a directory import symbols defined in those affected
  modules (``notebook_affected`` + ``symbol_definition_map``).

CLI
---
``python3 scripts/ci/import_graph.py expand <changed_files...>``
    Print the source file of every probpipe module transitively affected by the
    changed files, one per line. Consumed by the ``test`` job.

``python3 scripts/ci/import_graph.py notebooks <nb_dir> <changed_files...>``
    Print the notebooks under ``<nb_dir>`` affected by the changed files, one
    per line. Consumed by the ``notebooks`` job.

Stdlib-only: the change-detection step runs before dependencies are installed,
on the runner's system ``python3``.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

#: The package whose import graph is analysed. Repo-specific by design.
PACKAGE = "probpipe"


def file_to_module(path: str) -> str:
    """Convert a ``probpipe/.../mod.py`` path to its dotted module name."""
    return str(Path(path).with_suffix("")).replace("/", ".")


def module_to_file(module: str) -> str | None:
    """Return the file backing a dotted module name, or ``None`` if absent."""
    base = module.replace(".", "/")
    for candidate in (base + ".py", base + "/__init__.py"):
        if Path(candidate).exists():
            return candidate
    return None


def _resolve_import_from(node: ast.ImportFrom, importer: str) -> str | None:
    """Resolve an ``ast.ImportFrom`` to its absolute intra-package target.

    Handles relative imports via ``node.level``. Returns ``None`` for imports
    that do not target :data:`PACKAGE`.
    """
    level = node.level or 0
    if level == 0:
        if not node.module or not node.module.startswith(PACKAGE):
            return None
        return node.module
    parts = importer.split(".")
    base = parts[:-level] if level < len(parts) else []
    if node.module:
        return ".".join(base + node.module.split("."))
    return ".".join(base)


def reverse_dependency_map(package_root: str = PACKAGE) -> dict[str, set[str]]:
    """Build ``module -> {modules that directly import it}`` over the package.

    Walks every ``*.py`` file under *package_root*, parsing intra-package
    imports (absolute ``from probpipe... import`` / ``import probpipe...`` and
    relative ``from . import`` with ``level`` handling) into reverse edges.
    """
    rdeps: dict[str, set[str]] = defaultdict(set)
    for fpath in Path(package_root).rglob("*.py"):
        importer = file_to_module(str(fpath))
        try:
            tree = ast.parse(fpath.read_text())
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = _resolve_import_from(node, importer)
                if imported is not None:
                    rdeps[imported].add(importer)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(PACKAGE):
                        rdeps[alias.name].add(importer)
    return rdeps


def affected_modules(
    changed_files: Iterable[str],
    rdeps: dict[str, set[str]],
) -> set[str]:
    """Transitively expand changed source files over the reverse graph.

    Seeds the walk with the modules of *changed_files* and follows reverse
    edges. ``__init__`` modules are skipped *during traversal*: they are sinks
    in the reverse graph (nothing imports a package by its dunder name), so
    they add no real dependents, and traversing into the top-level
    ``probpipe/__init__.py`` would otherwise make the test job treat any change
    to a re-exported module as a root-level change. A directly changed
    ``__init__.py`` is still honoured (it enters as a seed).
    """
    seen = {file_to_module(f) for f in changed_files}
    queue = list(seen)
    while queue:
        module = queue.pop(0)
        for dep in rdeps.get(module, ()):
            if dep not in seen and dep.rsplit(".", 1)[-1] != "__init__":
                seen.add(dep)
                queue.append(dep)
    return seen


def symbol_definition_map(package_root: str = PACKAGE) -> dict[str, str]:
    """Map each top-level symbol name to the module that defines it.

    ``__init__`` files are skipped so the map records true definition sites;
    used to resolve ``from probpipe import Name`` re-exports back to where
    ``Name`` is actually defined. First definition wins on collisions.
    """
    sym_defs: dict[str, str] = {}
    for fpath in Path(package_root).rglob("*.py"):
        if fpath.name == "__init__.py":
            continue
        module = file_to_module(str(fpath))
        try:
            tree = ast.parse(fpath.read_text())
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                sym_defs.setdefault(node.name, module)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        sym_defs.setdefault(target.id, module)
    return sym_defs


def _cell_imported_modules(tree: ast.AST, sym_defs: dict[str, str]) -> set[str]:
    """Resolve the probpipe modules a parsed notebook cell depends on.

    ``from probpipe import Name`` is resolved through *sym_defs* to the module
    defining ``Name`` (falling back to the package for unknowns / ``*``);
    submodule imports are already definition sites; bare ``import probpipe`` is
    treated as the package prefix.
    """
    resolved: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module or not node.module.startswith(PACKAGE):
                continue
            src_file = module_to_file(node.module)
            if src_file and src_file.endswith("__init__.py"):
                for alias in node.names:
                    if alias.name != "*" and alias.name in sym_defs:
                        resolved.add(sym_defs[alias.name])
                    else:
                        resolved.add(node.module)
            else:
                resolved.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PACKAGE):
                    resolved.add(alias.name)
    return resolved


def notebook_affected(
    nb_path: str,
    affected: set[str],
    sym_defs: dict[str, str],
) -> bool:
    """Return whether *nb_path* imports any symbol defined in *affected*."""
    nb = json.loads(Path(nb_path).read_text())
    resolved: set[str] = set()
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        resolved |= _cell_imported_modules(tree, sym_defs)
    return any(
        af_mod == nb_mod or af_mod.startswith(nb_mod + ".")
        for nb_mod in resolved
        for af_mod in affected
    )


def _cmd_expand(args: argparse.Namespace) -> int:
    rdeps = reverse_dependency_map()
    changed_sources = [f for f in args.files if f.startswith(PACKAGE)]
    for module in sorted(affected_modules(changed_sources, rdeps)):
        path = module_to_file(module)
        if path:
            print(path)
    return 0


def _cmd_notebooks(args: argparse.Namespace) -> int:
    nb_dir = args.nb_dir
    changed = args.files
    rdeps = reverse_dependency_map()
    sym_defs = symbol_definition_map()

    changed_sources = [f for f in changed if f.startswith(PACKAGE + "/")]
    affected = affected_modules(changed_sources, rdeps)

    to_run = {
        f
        for f in changed
        if f.startswith(nb_dir + "/") and f.endswith(".ipynb") and Path(f).exists()
    }
    if affected:
        for nb_path in sorted(Path(nb_dir).glob("*.ipynb")):
            nb_str = str(nb_path)
            if nb_str not in to_run and notebook_affected(nb_str, affected, sym_defs):
                to_run.add(nb_str)
    for nb in sorted(to_run):
        print(nb)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reverse-dependency import-graph selection for CI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_expand = sub.add_parser(
        "expand",
        help="print source files transitively affected by changed files",
    )
    p_expand.add_argument("files", nargs="*", help="changed file paths")
    p_expand.set_defaults(func=_cmd_expand)

    p_nb = sub.add_parser(
        "notebooks",
        help="print notebooks under <nb_dir> affected by changed files",
    )
    p_nb.add_argument("nb_dir", help="notebook directory for this CI leg")
    p_nb.add_argument("files", nargs="*", help="changed file paths")
    p_nb.set_defaults(func=_cmd_notebooks)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
