"""Unit tests for the shared CI import-graph builder (scripts/ci/import_graph.py).

The builder previously lived as two inline heredocs in ``ci.yml`` and could not
be tested; see issue #266. Each test materialises a tiny fake ``probpipe``
package in a temp cwd and exercises the graph functions against it.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
from import_graph import (
    affected_modules,
    main,
    notebook_affected,
    resolve_test_targets,
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


class TestImportGraph:
    def test_relative_imports_resolve(self, fake_pkg):
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

    def test_alias_import_recorded(self, fake_pkg):
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

    def test_transitive_bfs_reaches_indirect_dependents(self, fake_pkg):
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

    def test_affected_modules_skip_init_flag(self, fake_pkg):
        """skip_init=True drops __init__ hubs; the default (False) keeps them.

        skip_init=False is what preserves the test job's root-test trigger; a
        directly changed __init__ is always a seed regardless of the flag.
        """
        fake_pkg(
            {
                "probpipe/__init__.py": "from .leaf import Thing\n",
                "probpipe/leaf.py": "class Thing:\n    pass\n",
                "probpipe/mid.py": "from .leaf import Thing\n",
            }
        )
        rdeps = reverse_dependency_map()

        # Notebooks behavior: skip __init__, keep real dependents.
        skipped = affected_modules(["probpipe/leaf.py"], rdeps, skip_init=True)
        assert "probpipe.mid" in skipped
        assert "probpipe.__init__" not in skipped

        # Test-job behavior (default): keep __init__ so the root trigger fires.
        kept = affected_modules(["probpipe/leaf.py"], rdeps)
        assert "probpipe.__init__" in kept
        assert affected_modules(["probpipe/leaf.py"], rdeps, skip_init=False) == kept

        # A directly changed __init__ is honoured as a seed either way.
        assert "probpipe.__init__" in affected_modules(
            ["probpipe/__init__.py"], rdeps, skip_init=True
        )

    def test_symbol_map_and_notebook_affected_via_init(self, fake_pkg, tmp_path):
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

    def test_symbol_map_ignores_nested_definitions(self, fake_pkg):
        """Only module-level defs are recorded (tree.body, not a full walk)."""
        fake_pkg(
            {
                "probpipe/__init__.py": "",
                "probpipe/leaf.py": (
                    "def factory():\n    class Local:\n        pass\n    return Local\n"
                ),
            }
        )
        sym_defs = symbol_definition_map()
        assert sym_defs.get("factory") == "probpipe.leaf"  # module-level def kept
        assert "Local" not in sym_defs  # nested inside a function body, ignored


class TestCLI:
    def test_expand_keeps_init_for_root_trigger(self, fake_pkg, capsys):
        """`expand` (skip_init=False) emits probpipe/__init__.py for re-exports."""
        fake_pkg(
            {
                "probpipe/__init__.py": "from .leaf import Thing\n",
                "probpipe/leaf.py": "class Thing:\n    pass\n",
                "probpipe/mid.py": "from .leaf import Thing\n",
            }
        )
        assert main(["expand", "probpipe/leaf.py"]) == 0
        printed = capsys.readouterr().out.split()
        assert "probpipe/leaf.py" in printed
        assert "probpipe/mid.py" in printed
        assert "probpipe/__init__.py" in printed

    def test_notebooks_selects_affected(self, fake_pkg, tmp_path, capsys):
        """`notebooks` prints notebooks importing a changed module's symbols."""
        fake_pkg(
            {
                "probpipe/__init__.py": "from .leaf import Normal\n",
                "probpipe/leaf.py": "class Normal:\n    pass\n",
            }
        )
        nb_dir = tmp_path / "nbs"
        nb_dir.mkdir()
        nb = {"cells": [{"cell_type": "code", "source": ["from probpipe import Normal\n"]}]}
        (nb_dir / "demo.ipynb").write_text(json.dumps(nb))

        assert main(["notebooks", str(nb_dir), "probpipe/leaf.py"]) == 0
        printed = capsys.readouterr().out.split()
        assert str(nb_dir / "demo.ipynb") in printed


class TestResolveTestTargets:
    """Source-change -> pytest-target mapping, incl. inference filename granularity."""

    # A fake inference subpackage: the __init__ facade imports every method (so a
    # method change pulls __init__ into the affected set); each method imports the
    # shared _registry; probpipe/__init__ stays empty so a method change does NOT
    # reach the root (mirrors the real tree, verified via `expand`).
    _INFERENCE_TREE: ClassVar[dict[str, str]] = {
        "probpipe/__init__.py": "",
        # Name-imports (`from ._mod import X`), matching the real registration
        # facade — these record a reverse edge to the *submodule*, so a method
        # change pulls __init__ into the affected set. (`from . import _mod` would
        # not, and would not mirror the real graph.)
        "probpipe/inference/__init__.py": (
            "from ._blackjax_sgmcmc import x\n"
            "from ._pyabc import x\n"
            "from ._pymc_method import x\n"
            "from ._registry import reg\n"
        ),
        "probpipe/inference/_registry.py": "reg = 1\n",
        "probpipe/inference/_blackjax_sgmcmc.py": "from ._registry import reg\n\nx = 1\n",
        "probpipe/inference/_pyabc.py": "from ._registry import reg\n\nx = 1\n",
        "probpipe/inference/_pymc_method.py": "from ._registry import reg\n\nx = 1\n",
        "tests/inference/test_blackjax_sgmcmc.py": "",
        "tests/inference/test_pyabc.py": "",
        "tests/inference/test_inference.py": "",
        "tests/inference/test_inference_registry.py": "",
    }

    def test_method_maps_to_own_test_plus_integration(self, fake_pkg):
        """A single method change runs its own test + the registry integration
        tests (from the __init__ edge), not the whole folder or sibling methods."""
        fake_pkg(self._INFERENCE_TREE)
        assert set(resolve_test_targets(["probpipe/inference/_blackjax_sgmcmc.py"])) == {
            "tests/inference/test_blackjax_sgmcmc.py",
            "tests/inference/test_inference.py",
            "tests/inference/test_inference_registry.py",
        }

    def test_direct_init_change_maps_to_whole_folder(self, fake_pkg):
        """A *direct* edit to the registration facade (probpipe/inference/__init__.py)
        can make or break any method's registration -- including optional backends --
        so it maps to the whole folder, unlike the transitive __init__ edge above
        (a method change) which narrows to the integration tests."""
        fake_pkg(self._INFERENCE_TREE)
        assert set(resolve_test_targets(["probpipe/inference/__init__.py"])) == {"tests/inference"}

    def test_shared_core_maps_to_whole_folder(self, fake_pkg):
        """A shared-core change (_registry) fans out to every method, so the
        folder target supersedes the individual files."""
        fake_pkg(self._INFERENCE_TREE)
        assert set(resolve_test_targets(["probpipe/inference/_registry.py"])) == {"tests/inference"}

    def test_module_without_matching_test_falls_back_to_folder(self, fake_pkg):
        """An inference module with no test_<name>.py (here _pymc_method) falls
        back to the whole folder rather than silently selecting nothing."""
        fake_pkg(self._INFERENCE_TREE)
        assert set(resolve_test_targets(["probpipe/inference/_pymc_method.py"])) == {
            "tests/inference"
        }

    def test_other_subpackage_maps_to_its_folder(self, fake_pkg):
        """Non-inference subpackages keep the folder-granular behavior."""
        fake_pkg(
            {
                "probpipe/__init__.py": "",
                "probpipe/core/__init__.py": "",
                "probpipe/core/thing.py": "v = 1\n",
                "tests/core/test_thing.py": "",
            }
        )
        assert set(resolve_test_targets(["probpipe/core/thing.py"])) == {"tests/core"}

    def test_cli_prints_targets(self, fake_pkg, capsys):
        fake_pkg(self._INFERENCE_TREE)
        assert main(["test-targets", "probpipe/inference/_pyabc.py"]) == 0
        printed = capsys.readouterr().out.split()
        assert "tests/inference/test_pyabc.py" in printed
        assert "tests/inference/test_inference_registry.py" in printed
        assert "tests/inference" not in printed  # not the whole folder

    def test_root_module_maps_to_root_tests(self, fake_pkg):
        """A root-level ``probpipe/X.py`` change selects the root ``tests/test_*.py``
        files, never a subpackage folder."""
        fake_pkg(
            {
                "probpipe/__init__.py": "",
                "probpipe/_utils.py": "v = 1\n",
                "tests/test_utils.py": "",
                "tests/test_coverage_gaps.py": "",
                "tests/core/test_thing.py": "",  # subpackage test — must NOT be selected
            }
        )
        assert set(resolve_test_targets(["probpipe/_utils.py"])) == {
            "tests/test_utils.py",
            "tests/test_coverage_gaps.py",
        }

    def test_reexported_leaf_reaches_root_tests(self, fake_pkg):
        """A subpackage module re-exported through ``probpipe/__init__.py`` reaches
        the top-level __init__ (skip_init=False), pulling in the root
        ``tests/test_*.py`` alongside its own subpackage folder — the
        regression-suite trigger (e.g. test_coverage_gaps.py) the test job relies on."""
        fake_pkg(
            {
                "probpipe/__init__.py": "from .core.thing import Thing\n",
                "probpipe/core/__init__.py": "",
                "probpipe/core/thing.py": "class Thing:\n    pass\n",
                "tests/core/test_thing.py": "",
                "tests/test_coverage_gaps.py": "",
            }
        )
        assert set(resolve_test_targets(["probpipe/core/thing.py"])) == {
            "tests/core",
            "tests/test_coverage_gaps.py",
        }

    def test_nested_inference_module_falls_back_to_folder(self, fake_pkg):
        """A nested inference subpackage module has no flat ``test_<name>.py``
        convention, so it maps to the whole folder rather than a basename guess."""
        fake_pkg(
            {
                "probpipe/__init__.py": "",
                "probpipe/inference/__init__.py": "",
                "probpipe/inference/sub/__init__.py": "",
                "probpipe/inference/sub/_foo.py": "v = 1\n",
                "tests/inference/test_foo.py": "",  # basename match — must NOT be selected
            }
        )
        assert set(resolve_test_targets(["probpipe/inference/sub/_foo.py"])) == {"tests/inference"}
