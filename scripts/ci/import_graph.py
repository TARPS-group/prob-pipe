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


def _resolve_import_from(node: ast.ImportFrom, importer: str, package: str) -> str | None:
    """Resolve an ``ast.ImportFrom`` to its absolute intra-package target.

    Handles relative imports via ``node.level``. Returns ``None`` for imports
    that do not target *package*.
    """
    level = node.level or 0
    if level == 0:
        if not node.module or not node.module.startswith(package):
            return None
        return node.module
    parts = importer.split(".")
    base = parts[:-level] if level < len(parts) else []
    if node.module:
        return ".".join(base + node.module.split("."))
    return ".".join(base)


def _scan_package(package_root: str = PACKAGE) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Single AST pass over *package_root*, returning ``(rdeps, sym_defs)``.

    Each file is read and parsed exactly once. ``rdeps`` (module ->
    {modules that directly import it}) is built from every file, including
    ``__init__``, via a full ``ast.walk`` so imports nested under
    ``TYPE_CHECKING`` or inside functions are still captured. ``sym_defs``
    (top-level symbol name -> defining module, first-wins) is built only from
    the *module-level* defs/assignments of non-``__init__`` files, so it
    records true definition sites for resolving ``probpipe/__init__.py``
    re-exports.
    """
    rdeps: dict[str, set[str]] = defaultdict(set)
    sym_defs: dict[str, str] = {}
    for fpath in Path(package_root).rglob("*.py"):
        module = file_to_module(str(fpath))
        try:
            tree = ast.parse(fpath.read_text())
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = _resolve_import_from(node, module, package_root)
                if imported is not None:
                    rdeps[imported].add(module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(package_root):
                        rdeps[alias.name].add(module)
        if fpath.name != "__init__.py":
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    sym_defs.setdefault(node.name, module)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            sym_defs.setdefault(target.id, module)
    return rdeps, sym_defs


def reverse_dependency_map(package_root: str = PACKAGE) -> dict[str, set[str]]:
    """Build ``module -> {modules that directly import it}`` over the package.

    Parses every ``*.py`` file under *package_root*, recording intra-package
    imports (absolute ``from probpipe... import`` / ``import probpipe...`` and
    relative ``from . import`` with ``level`` handling) as reverse edges.
    """
    return _scan_package(package_root)[0]


def symbol_definition_map(package_root: str = PACKAGE) -> dict[str, str]:
    """Map each top-level symbol name to the module that defines it.

    ``__init__`` files are skipped and only module-level definitions are
    recorded, so the map holds true definition sites; used to resolve
    ``from probpipe import Name`` re-exports back to where ``Name`` is defined.
    First definition wins on collisions.
    """
    return _scan_package(package_root)[1]


def affected_modules(
    changed_files: Iterable[str],
    rdeps: dict[str, set[str]],
    *,
    skip_init: bool = False,
) -> set[str]:
    """Transitively expand changed source files over the reverse graph.

    Seeds the walk with the modules of *changed_files* and follows reverse
    edges, returning every transitively affected module.

    *skip_init* selects each job's historical behavior:

    * ``False`` (the ``test`` job): keep ``__init__`` modules. A change to a
      module re-exported by ``probpipe/__init__.py`` then reaches the
      top-level ``__init__``, which the ci.yml mapping treats as a root-level
      change and so runs the root ``tests/test_*.py`` files (e.g.
      ``test_coverage_gaps.py``, which regression-tests re-exported classes).
    * ``True`` (the ``notebooks`` job): skip ``__init__`` modules during
      traversal. That job resolves package-level imports through
      :func:`symbol_definition_map` rather than the reverse graph, so it never
      relied on ``__init__`` edges.

    A directly changed ``__init__.py`` always enters as a seed regardless of
    *skip_init*.
    """
    seen = {file_to_module(f) for f in changed_files}
    queue = list(seen)
    while queue:
        module = queue.pop(0)
        for dep in rdeps.get(module, ()):
            if dep in seen:
                continue
            if skip_init and dep.rsplit(".", 1)[-1] == "__init__":
                continue
            seen.add(dep)
            queue.append(dep)
    return seen


def _cell_imported_modules(tree: ast.AST, sym_defs: dict[str, str], package: str) -> set[str]:
    """Resolve the probpipe modules a parsed notebook cell depends on.

    ``from probpipe import Name`` is resolved through *sym_defs* to the module
    defining ``Name`` (falling back to the package for unknowns / ``*``);
    submodule imports are already definition sites; bare ``import probpipe`` is
    treated as the package prefix.
    """
    resolved: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module or not node.module.startswith(package):
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
                if alias.name.startswith(package):
                    resolved.add(alias.name)
    return resolved


def notebook_affected(
    nb_path: str,
    affected: set[str],
    sym_defs: dict[str, str],
    package: str = PACKAGE,
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
        resolved |= _cell_imported_modules(tree, sym_defs, package)
    return any(
        af_mod == nb_mod or af_mod.startswith(nb_mod + ".")
        for nb_mod in resolved
        for af_mod in affected
    )


def _cmd_expand(args: argparse.Namespace) -> int:
    rdeps = reverse_dependency_map()
    # The two jobs historically filtered changed sources slightly differently
    # (test: startswith("probpipe"); notebooks: startswith("probpipe/")); each
    # is preserved verbatim. skip_init=False preserves the test job's behavior:
    # a change to a module re-exported by probpipe/__init__.py still triggers
    # the root test files.
    changed_sources = [f for f in args.files if f.startswith(PACKAGE)]
    for module in sorted(affected_modules(changed_sources, rdeps, skip_init=False)):
        path = module_to_file(module)
        if path:
            print(path)
    return 0


#: Inference test files that exercise registration/dispatch across all methods.
#: A method module is pulled into the affected set alongside ``inference/__init__``
#: (the registration facade imports every method), so mapping that __init__ edge
#: to these integration tests -- rather than the whole folder -- is what lets a
#: single method change run only its own test file plus these.
_INFERENCE_INTEGRATION_TESTS = (
    "tests/inference/test_inference.py",
    "tests/inference/test_inference_registry.py",
)


def _inference_test_targets(module_file: str) -> set[str]:
    """Map one affected ``probpipe/inference/*.py`` file to its pytest target(s).

    Filename convention ``_<name>.py -> tests/inference/test_<name>.py`` for the
    per-method modules; the ``__init__`` registration facade maps to the
    registry/dispatch integration tests (:data:`_INFERENCE_INTEGRATION_TESTS`);
    anything without a matching test file (shared core such as ``_registry.py``,
    cross-cutting helpers) falls back to the whole ``tests/inference/`` folder.
    """
    name = Path(module_file).name
    if name == "__init__.py":
        hits = {t for t in _INFERENCE_INTEGRATION_TESTS if Path(t).exists()}
        return hits or {"tests/inference"}
    stem = name[:-3].lstrip("_")  # _blackjax_sgmcmc.py -> blackjax_sgmcmc
    candidate = f"tests/inference/test_{stem}.py"
    return {candidate} if Path(candidate).exists() else {"tests/inference"}


def resolve_test_targets(changed_files: Iterable[str]) -> list[str]:
    """Map changed source files to the pytest targets that cover them.

    Source files are transitively expanded over the reverse import graph
    (``skip_init=False``, as the test job requires), then each affected module
    is mapped to a target: a root-level ``probpipe/X.py`` selects the root
    ``tests/test_*.py`` files; ``probpipe/inference/...`` uses the
    filename-convention granularity of :func:`_inference_test_targets`; any
    other ``probpipe/<subpkg>/...`` selects the ``tests/<subpkg>/`` folder. A
    folder target supersedes individual file targets beneath it. Returns a
    sorted list.
    """
    rdeps = reverse_dependency_map()
    changed_sources = [f for f in changed_files if f.startswith(PACKAGE)]
    targets: set[str] = set()
    root_change = False
    for module in affected_modules(changed_sources, rdeps, skip_init=False):
        path = module_to_file(module)
        if not path:
            continue
        rel = path[len(PACKAGE) + 1 :]  # strip leading "probpipe/"
        if "/" not in rel:
            root_change = True  # root-level module -> root tests/test_*.py
            continue
        subpkg, mod = rel.split("/", 1)
        if subpkg == "inference":
            targets |= _inference_test_targets(mod)
        elif Path(f"tests/{subpkg}").is_dir():
            targets.add(f"tests/{subpkg}")
    if root_change:
        targets |= {str(p) for p in Path("tests").glob("test_*.py")}
    # A folder target supersedes the individual files beneath it (else pytest
    # would collect the same file twice).
    folders = {t for t in targets if not t.endswith(".py")}
    targets = {t for t in targets if not any(t.startswith(f"{d}/") for d in folders)}
    return sorted(targets)


def _cmd_test_targets(args: argparse.Namespace) -> int:
    for target in resolve_test_targets(args.files):
        print(target)
    return 0


def _cmd_notebooks(args: argparse.Namespace) -> int:
    nb_dir = args.nb_dir
    changed = args.files
    rdeps, sym_defs = _scan_package()  # single pass builds both maps
    changed_sources = [f for f in changed if f.startswith(PACKAGE + "/")]
    affected = affected_modules(changed_sources, rdeps, skip_init=True)

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

    p_targets = sub.add_parser(
        "test-targets",
        help="print the pytest targets covering the changed source files",
    )
    p_targets.add_argument("files", nargs="*", help="changed file paths")
    p_targets.set_defaults(func=_cmd_test_targets)

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
