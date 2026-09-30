"""Source-level guards for unified automatic-key ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "probpipe"
_JAX_KEY_CONSTRUCTORS = frozenset({"PRNGKey", "key"})


def _python_sources(directory: Path) -> tuple[Path, ...]:
    return tuple(sorted(directory.rglob("*.py")))


def _jax_key_bindings(
    tree: ast.AST,
) -> tuple[set[str], set[str], set[str]]:
    """Return bound names for JAX, ``jax.random``, and key constructors."""
    jax_modules = set()
    random_modules = set()
    constructors = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "jax":
                    jax_modules.add(alias.asname or "jax")
                elif alias.name == "jax.random":
                    if alias.asname is None:
                        jax_modules.add("jax")
                    else:
                        random_modules.add(alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.module == "jax":
            for alias in node.names:
                if alias.name == "random":
                    random_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "jax.random":
            for alias in node.names:
                if alias.name in _JAX_KEY_CONSTRUCTORS:
                    constructors.add(alias.asname or alias.name)
    return jax_modules, random_modules, constructors


def _is_jax_key_constructor_call(
    node: ast.Call,
    *,
    jax_modules: set[str],
    random_modules: set[str],
    constructors: set[str],
) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in constructors
    if not isinstance(func, ast.Attribute) or func.attr not in _JAX_KEY_CONSTRUCTORS:
        return False
    if isinstance(func.value, ast.Name):
        return func.value.id in random_modules
    return (
        isinstance(func.value, ast.Attribute)
        and func.value.attr == "random"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id in jax_modules
    )


def _direct_jax_key_constructor_calls(paths: tuple[Path, ...]) -> list[str]:
    calls = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        jax_modules, random_modules, constructors = _jax_key_bindings(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_jax_key_constructor_call(
                node,
                jax_modules=jax_modules,
                random_modules=random_modules,
                constructors=constructors,
            ):
                relative = path.relative_to(_REPOSITORY_ROOT)
                calls.append(f"{relative}:{node.lineno}")
    return calls


@pytest.mark.parametrize(
    "source",
    [
        "import jax\njax.random.PRNGKey(0)\n",
        "import jax\njax.random.key(0)\n",
        "import jax as j\nj.random.key(0)\n",
        "import jax.random as jr\njr.PRNGKey(0)\n",
        "from jax import random as jr\njr.key(0)\n",
        "from jax.random import PRNGKey as make_key\nmake_key(0)\n",
        "from jax.random import key as make_key\nmake_key(0)\n",
    ],
)
def test_source_scan_detects_jax_key_constructor_import_forms(
    source,
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "candidate.py"
    path.write_text(source)
    monkeypatch.setattr("tests.core.test_automatic_key_source_scan._REPOSITORY_ROOT", tmp_path)

    assert _direct_jax_key_constructor_calls((path,)) == ["candidate.py:2"]


def test_source_scan_ignores_unrelated_key_methods_and_functions(tmp_path, monkeypatch):
    path = tmp_path / "candidate.py"
    path.write_text(
        "def key(value):\n"
        "    return value\n"
        "class Provider:\n"
        "    def PRNGKey(self, value):\n"
        "        return value\n"
        "key(0)\n"
        "Provider().PRNGKey(0)\n"
    )
    monkeypatch.setattr("tests.core.test_automatic_key_source_scan._REPOSITORY_ROOT", tmp_path)

    assert _direct_jax_key_constructor_calls((path,)) == []


def test_legacy_process_global_key_symbols_are_absent():
    offenders = []
    for path in _python_sources(_PACKAGE_ROOT):
        source = path.read_text()
        for symbol in ("_AUTO_KEY_COUNTER", "_auto_key"):
            if symbol in source:
                offenders.append(f"{path.relative_to(_REPOSITORY_ROOT)}: {symbol}")

    assert offenders == []


def test_broker_owned_modules_have_no_direct_prng_key_fallbacks():
    core_paths = tuple(
        path for path in _python_sources(_PACKAGE_ROOT / "core") if path.name != "transition.py"
    )
    # ``transition.with_resampling(seed=...)`` retains its caller-controlled
    # public seed contract; it is not an omitted-key fallback.
    broker_owned_paths = core_paths + tuple(
        path
        for package in ("converters", "validation", "diagnostics")
        for path in _python_sources(_PACKAGE_ROOT / package)
    )

    assert _direct_jax_key_constructor_calls(broker_owned_paths) == []


def test_caller_owned_and_provider_local_seed_paths_remain():
    inference_source = "\n".join(
        path.read_text() for path in _python_sources(_PACKAGE_ROOT / "inference")
    )
    glm_source = (_PACKAGE_ROOT / "modeling" / "_glm.py").read_text()

    assert "jax.random.PRNGKey(random_seed)" in inference_source
    assert "self._key = jax.random.PRNGKey(seed)" in glm_source
    assert "self._key, key = jax.random.split(self._key)" in glm_source
