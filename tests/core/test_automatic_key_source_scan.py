"""Source-level guards for unified automatic-key ownership."""

from __future__ import annotations

import ast
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "probpipe"


def _python_sources(directory: Path) -> tuple[Path, ...]:
    return tuple(sorted(directory.rglob("*.py")))


def _direct_prng_key_calls(paths: tuple[Path, ...]) -> list[str]:
    calls = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "PRNGKey"
            ):
                relative = path.relative_to(_REPOSITORY_ROOT)
                calls.append(f"{relative}:{node.lineno}")
    return calls


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

    assert _direct_prng_key_calls(broker_owned_paths) == []


def test_caller_owned_and_provider_local_seed_paths_remain():
    inference_source = "\n".join(
        path.read_text() for path in _python_sources(_PACKAGE_ROOT / "inference")
    )
    glm_source = (_PACKAGE_ROOT / "modeling" / "_glm.py").read_text()

    assert "jax.random.PRNGKey(random_seed)" in inference_source
    assert "self._key = jax.random.PRNGKey(seed)" in glm_source
    assert "self._key, key = jax.random.split(self._key)" in glm_source
