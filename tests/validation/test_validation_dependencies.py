"""Validation subpackage dependency-boundary tests."""

from __future__ import annotations

import ast
from pathlib import Path

import probpipe.validation


def test_validation_does_not_import_modeling():
    validation_root = Path(probpipe.validation.__file__).parent
    violations = []

    for source_path in validation_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported_modules = (f"probpipe.{module}" if node.level == 2 else module,)
            else:
                continue
            if any(
                module == "probpipe.modeling" or module.startswith("probpipe.modeling.")
                for module in imported_modules
            ):
                violations.append(f"{source_path.relative_to(validation_root)}:{node.lineno}")

    assert violations == []
