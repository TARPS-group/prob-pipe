"""Repository hygiene checks for removed private imports."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (ROOT / "probpipe", ROOT / "tests")
THIS_FILE = Path(__file__).resolve()


def _python_files():
    for source_root in SOURCE_ROOTS:
        for path in source_root.rglob("*.py"):
            if path.resolve() == THIS_FILE:
                continue
            if "__pycache__" in path.parts:
                continue
            yield path


def test_no_legacy_utils_prod_imports():
    offenders: list[str] = []

    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module not in {"_utils", "probpipe._utils"}:
                continue
            if any(alias.name == "prod" for alias in node.names):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert offenders == []


def test_utils_module_does_not_define_prod_wrapper():
    tree = ast.parse((ROOT / "probpipe" / "_utils.py").read_text())

    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    assert "prod" not in function_names
