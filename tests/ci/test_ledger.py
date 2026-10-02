"""The design ledger's list of stale uses in the docs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "design"))

import ledger


def _notebook(path: Path, *sources: str) -> None:
    cells = [{"cell_type": "code", "source": [source], "metadata": {}} for source in sources]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": cells}))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "example_scripts").mkdir()
    return tmp_path


def _details(root: Path) -> list[tuple[str, str, str]]:
    return [(use.path, use.location, use.detail) for use in ledger.stale_docs(root)]


class TestStaleDocs:
    def test_an_import_of_a_name_probpipe_lacks_is_listed(self, root):
        _notebook(root / "docs" / "guide.ipynb", "from probpipe import Normal, NoSuchName")
        assert _details(root) == [
            ("docs/guide.ipynb", "cell 0", "imports NoSuchName from probpipe, which lacks it")
        ]

    def test_an_import_from_a_module_that_is_gone_is_listed(self, root):
        (root / "example_scripts" / "demo.py").write_text("from probpipe.no_such_module import x\n")
        assert _details(root) == [
            (
                "example_scripts/demo.py",
                "script",
                "imports from probpipe.no_such_module, which is gone",
            )
        ]

    def test_a_retired_keyword_is_listed_with_its_replacement(self, root):
        _notebook(root / "docs" / "guide.ipynb", "x = 1", "expectation(d, f, return_dist=False)")
        ((path, location, detail),) = _details(root)
        assert (path, location) == ("docs/guide.ipynb", "cell 1")
        assert detail.startswith("passes return_dist=, which is removed")

    def test_a_function_keyword_that_is_no_control_is_listed(self, root):
        _notebook(
            root / "docs" / "guide.ipynb",
            "add = Function('add', lambda x, y: x + y, dispatch='sequential', y=2.0)",
            "@pp.function(n_broadcast_samples=8, scale=2.0)\ndef scaled(x, scale):\n    return x",
        )
        (root / "example_scripts" / "demo.py").write_text(
            "wrapped = function(lambda x: x, name='identity', seed=0)\n"
        )
        detail = "which is neither a construction parameter nor a control"
        assert [
            (path, location, text.split(",")[0]) for path, location, text in _details(root)
        ] == [
            ("docs/guide.ipynb", "cell 0", "passes y= to Function"),
            ("docs/guide.ipynb", "cell 1", "passes scale= to function"),
            ("example_scripts/demo.py", "script", "passes seed= to function"),
        ]
        assert all(detail in text for _path, _location, text in _details(root))

    def test_construction_parameters_controls_and_bindings_are_not_listed(self, root):
        _notebook(
            root / "docs" / "guide.ipynb",
            "add = Function(name='add', fn=lambda x, y: x + y, bind={'y': 2.0}, raw=True)",
            "@function(name='f', output_label='value', dispatch='jax', workflow_kind=None)\n"
            "def f(x):\n    return x",
            "g = Function('g', lambda **kw: 0, **controls)",
        )
        assert _details(root) == []

    def test_current_imports_and_magics_are_not_listed(self, root):
        _notebook(root / "docs" / "guide.ipynb", "%matplotlib inline\nfrom probpipe import Normal")
        (root / "example_scripts" / "demo.py").write_text("import probpipe\n")
        assert _details(root) == []
