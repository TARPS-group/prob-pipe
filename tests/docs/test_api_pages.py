"""The API reference documents each public object of ProbPipe once, with a summary.

A public module is a module of ``probpipe`` whose dotted path has no component
that starts with an underscore, as ``STYLE_GUIDE.md`` §2.3 states, and its
``__all__`` lists its public names (§9.1). A name that two public modules export
is one object, such as ``probpipe.Normal`` and ``probpipe.families.Normal``, so
the test matches the pages' paths to the public names by object identity.

Each ``::: dotted.path`` line of ``docs/api/*.md`` is a mkdocstrings directive,
which renders the object at that path. The test checks three things:

1. each directive names a public object;
2. each public object has a directive on exactly one page;
3. the docstring of each documented object starts with a summary, a line that
   ends with a period and that a blank line or the end of the docstring follows.

mkdocstrings reads docstrings from the source through griffe, so the third check
reads them through griffe too. An instance at module level, such as
``probpipe.real``, therefore needs a docstring of its own after its assignment,
since the docstring of its class is not rendered for it.
"""

from __future__ import annotations

import functools
import importlib
import pkgutil
import re
from collections import defaultdict
from pathlib import Path

import griffe

import probpipe

ROOT = Path(__file__).resolve().parents[2]
API_PAGES = ROOT / "docs" / "api"

_DIRECTIVE = re.compile(r"^:::\s+([\w.]+)\s*$")


@functools.cache
def _public_modules() -> tuple[str, ...]:
    """Every public module of ``probpipe`` that defines ``__all__``, the package first."""
    names = ["probpipe"] + [
        info.name
        for info in pkgutil.walk_packages(probpipe.__path__, "probpipe.")
        if not any(part.startswith("_") for part in info.name.split("."))
    ]
    return tuple(name for name in names if hasattr(importlib.import_module(name), "__all__"))


@functools.cache
def _public_objects() -> dict[int, str]:
    """The first public path of each public object, keyed by the object's identity."""
    objects: dict[int, str] = {}
    for module_name in _public_modules():
        module = importlib.import_module(module_name)
        for name in module.__all__:
            objects.setdefault(id(getattr(module, name)), f"{module_name}.{name}")
    return objects


@functools.cache
def _directives() -> tuple[tuple[str, str], ...]:
    """The page and the dotted path of each directive of the API pages, outside code blocks."""
    directives: list[tuple[str, str]] = []
    for page in sorted(API_PAGES.glob("*.md")):
        in_fence = False
        for line in page.read_text().splitlines():
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            elif not in_fence and (match := _DIRECTIVE.match(line)):
                directives.append((page.name, match.group(1)))
    return tuple(directives)


def _resolve(path: str) -> object:
    """The object at the dotted *path*: its longest importable prefix, then attributes."""
    parts = path.split(".")
    for end in range(len(parts), 0, -1):
        try:
            target: object = importlib.import_module(".".join(parts[:end]))
        except ImportError:
            continue
        for attribute in parts[end:]:
            target = getattr(target, attribute)
        return target
    raise ImportError(f"{path} does not import")


@functools.cache
def _rendered_package() -> griffe.Module:
    """``probpipe`` as mkdocstrings reads it: statically, from the source of the repository."""
    return griffe.load("probpipe", search_paths=[ROOT])


def _rendered_docstring(path: str) -> str:
    """The docstring the directive for *path* renders, or ``""`` for none."""
    member = _rendered_package()[path.removeprefix("probpipe.")]
    target = member.final_target if member.is_alias else member
    return target.docstring.value if target.docstring else ""


def _summary_problem(docstring: str) -> str | None:
    """Why *docstring* does not start with a one-line summary, or ``None`` if it does."""
    lines = docstring.strip().splitlines()
    if not lines:
        return "has no docstring"
    if len(lines) > 1 and lines[1].strip():
        return f"has a summary of more than one line: {lines[0].strip()!r}"
    if not lines[0].rstrip().endswith("."):
        return f"has a summary that does not end with a period: {lines[0].strip()!r}"
    return None


def test_each_directive_names_a_public_object():
    public = _public_objects()
    stray = [f"{page}: {path}" for page, path in _directives() if id(_resolve(path)) not in public]
    assert stray == []


def test_each_public_object_is_documented_on_one_page():
    """A public object with no directive maps to ``[]``, and one documented twice to two pages."""
    pages: dict[int, list[str]] = defaultdict(list)
    for page, path in _directives():
        pages[id(_resolve(path))].append(page)
    misplaced = {
        path: pages[key] for key, path in _public_objects().items() if len(pages[key]) != 1
    }
    assert misplaced == {}


def test_each_documented_object_has_a_summary():
    problems = {
        path: problem
        for _, path in _directives()
        if (problem := _summary_problem(_rendered_docstring(path)))
    }
    assert problems == {}
