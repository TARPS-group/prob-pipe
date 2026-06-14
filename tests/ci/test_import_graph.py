"""Unit tests for the shared CI import-graph builder (scripts/ci/import_graph.py).

The builder previously lived as two inline heredocs in ``ci.yml`` and could not
be tested; see issue #266. Each test materialises a tiny fake ``probpipe``
package in a temp cwd and exercises the graph functions against it.
"""

from __future__ import annotations

import json

import pytest
from import_graph import (
    affected_modules,
    notebook_affected,
    reverse_dependency_map,
    symbol_definition_map,
)


@pytest.fixture
def fake_pkg(tmp_path, monkeypatch):
    """Write a ``{relpath: source}`` tree under tmp_path and chdir into it."""

    def _build(files: dict[str, str]) -> None:
        for relpath, content in files.items():
            path = tmp_path / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        monkeypatch.chdir(tmp_path)

    return _build


def test_relative_imports_resolve(fake_pkg):
    """``from .a`` / ``from ..a`` resolve to the correct absolute target."""
    fake_pkg(
        {
            "probpipe/__init__.py": "",
            "probpipe/a.py": "x = 1\n",
            "probpipe/b.py": "from .a import x\n",
            "probpipe/core/__init__.py": "",
            "probpipe/core/d.py": "from ..a import x\n",
        }
    )
    rdeps = reverse_dependency_map()
    assert "probpipe.b" in rdeps["probpipe.a"]
    assert "probpipe.core.d" in rdeps["probpipe.a"]


def test_alias_import_recorded(fake_pkg):
    """``import probpipe.sub as s`` records an edge under ``probpipe.sub``."""
    fake_pkg(
        {
            "probpipe/__init__.py": "",
            "probpipe/sub.py": "y = 2\n",
            "probpipe/user.py": "import probpipe.sub as s\n",
        }
    )
    rdeps = reverse_dependency_map()
    assert "probpipe.user" in rdeps["probpipe.sub"]


def test_transitive_bfs_reaches_indirect_dependents(fake_pkg):
    """A change to a leaf surfaces modules that import it transitively."""
    fake_pkg(
        {
            "probpipe/__init__.py": "",
            "probpipe/leaf.py": "class Thing:\n    pass\n",
            "probpipe/mid.py": "from .leaf import Thing\n",
            "probpipe/top.py": "from .mid import Thing\n",
        }
    )
    rdeps = reverse_dependency_map()
    affected = affected_modules(["probpipe/leaf.py"], rdeps)
    assert {"probpipe.leaf", "probpipe.mid", "probpipe.top"} <= affected


def test_affected_modules_skips_init_reexport(fake_pkg):
    """__init__ re-exports are not pulled in transitively (the unify rule)...

    ...while genuine non-__init__ dependents still are, and a directly changed
    __init__ is honoured as a seed.
    """
    fake_pkg(
        {
            "probpipe/__init__.py": "from .leaf import Thing\n",
            "probpipe/leaf.py": "class Thing:\n    pass\n",
            "probpipe/mid.py": "from .leaf import Thing\n",
        }
    )
    rdeps = reverse_dependency_map()

    affected = affected_modules(["probpipe/leaf.py"], rdeps)
    assert "probpipe.mid" in affected  # real dependent kept
    assert "probpipe.__init__" not in affected  # re-export hub skipped

    # A directly changed __init__ still enters as a seed.
    assert "probpipe.__init__" in affected_modules(["probpipe/__init__.py"], rdeps)


def test_symbol_map_and_notebook_affected_via_init(fake_pkg, tmp_path):
    """from-package imports resolve through sym_defs to the defining module."""
    fake_pkg(
        {
            "probpipe/__init__.py": "from .leaf import Normal\n",
            "probpipe/leaf.py": "class Normal:\n    pass\n",
            "probpipe/other.py": "class Other:\n    pass\n",
        }
    )
    sym_defs = symbol_definition_map()
    assert sym_defs["Normal"] == "probpipe.leaf"

    nb = {"cells": [{"cell_type": "code", "source": ["from probpipe import Normal\n"]}]}
    nb_path = tmp_path / "demo.ipynb"
    nb_path.write_text(json.dumps(nb))

    assert notebook_affected(str(nb_path), {"probpipe.leaf"}, sym_defs) is True
    assert notebook_affected(str(nb_path), {"probpipe.other"}, sym_defs) is False
