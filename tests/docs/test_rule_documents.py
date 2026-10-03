"""The rule documents and the agent files cite only paths and names that exist.

The rule documents are ``CONTRIBUTING.md``, ``STYLE_GUIDE.md``,
``CONTRACTS.md``, and the pull-request template. The agent files are
``AGENTS.md``, ``CLAUDE.md``, and the project skills under ``.claude/skills/``.
Each code span, link, and section pointer outside a fenced code block is a
citation, and each kind of citation is checked:

1. a repository path, such as ``probpipe/core/tracked.py``, exists under the
   repository root or under ``probpipe/``;
2. a file name, such as ``_nutpie.py``, names a file of the repository;
3. a dotted name under ``probpipe``, such as ``probpipe.core.tracked``, imports;
4. a class name, such as ``Record`` or ``Record.from_field_values``, and an
   underscore name, such as ``_memo``, are defined in ``probpipe/``, unless the
   name belongs to Python or to :data:`EXTERNAL_NAMES`;
5. a link to a file resolves, and its anchor names a heading of that file;
6. a section pointer, such as "STYLE_GUIDE.md §8.6" or "design II.4", names a
   heading of the file or a section of the design reference.

``AGENTS.md`` also stays within :data:`AGENTS_LINE_BUDGET` lines and imports no
other file, and ``CLAUDE.md`` imports it.
"""

from __future__ import annotations

import abc
import ast
import builtins
import collections.abc
import dataclasses
import enum
import functools
import importlib
import inspect
import re
import types
import typing
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "probpipe"

RULE_DOCUMENTS = (
    "CONTRIBUTING.md",
    "STYLE_GUIDE.md",
    "CONTRACTS.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
)
AGENT_FILES = ("AGENTS.md", "CLAUDE.md")

#: The most lines ``AGENTS.md`` may have, since every agent session loads all of it.
AGENTS_LINE_BUDGET = 120

#: Skills that describe documentation systems in general, so the paths they cite are examples.
GENERIC_SKILLS = frozenset({"criticize-with-docs"})

#: Names the documents cite that ProbPipe does not define: the NumPy docstring
#: sections, and the Claude Code tools that the skills call.
EXTERNAL_NAMES = frozenset(
    {
        "Attributes",
        "Examples",
        "Notes",
        "Parameters",
        "Raises",
        "Returns",
        "Warns",
        "Yields",
        "Agent",
        "Bash",
        "Edit",
        "Glob",
        "Grep",
        "Read",
        "Write",
    }
)

#: Files each contributor keeps for themselves, which the repository does not hold.
PERSONAL_FILES = frozenset({".claude/settings.local.json"})

#: Path-like prefixes that name git refs, such as a branch, rather than files.
REF_PREFIXES = ("dev/", "claude/", "origin/", "consolidated/")

_FILE_SUFFIXES = (".py", ".md", ".yml", ".yaml", ".toml", ".json", ".ipynb", ".cfg", ".lock")
_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\(\))?")
_CLASS_NAME = re.compile(r"_?[A-Z]\w*")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_DESIGN_SECTION = re.compile(r"^## ([IVX]+\.\d+) — ", re.M)
_FILE_POINTER = re.compile(
    r"([\w./-]*[\w-]\.md|\b[A-Z][A-Z_]+[A-Z])\s*§\s*(\d+(?:\.\d+)*|[IVX]+\.\d+|[A-Z][^.,;:)\]]*)"
)
_LOCAL_POINTER = re.compile(r"§\s?(\d+(?:\.\d+)*)")
_DESIGN_POINTER = re.compile(r"\bdesign (?:sections? )?([IVX]+\.\d+)\b")


def _documents() -> list[Path]:
    """Every rule document and agent file that exists, in a stable order."""
    paths = [ROOT / name for name in (*RULE_DOCUMENTS, *AGENT_FILES) if (ROOT / name).exists()]
    skills = ROOT / ".claude" / "skills"
    paths += sorted(
        path
        for path in skills.rglob("*.md")
        if path.relative_to(skills).parts[0] not in GENERIC_SKILLS
    )
    return paths


def _prose_lines(path: Path) -> Iterator[tuple[int, str]]:
    """The numbered lines of *path* outside its fenced code blocks."""
    in_fence = False
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            yield number, line


@functools.cache
def _package_trees() -> tuple[ast.Module, ...]:
    return tuple(ast.parse(path.read_text()) for path in sorted(PACKAGE.rglob("*.py")))


@functools.cache
def _definitions() -> frozenset[str]:
    """Every class, function, alias, and assigned name that ``probpipe/`` defines."""
    names: set[str] = set()
    for tree in _package_trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
                names.add(node.name.id)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(t.id for t in targets if isinstance(t, ast.Name))
    return frozenset(names)


@functools.cache
def _identifiers() -> frozenset[str]:
    """Every identifier ``probpipe/`` uses, and the fixtures of ``tests/conftest.py``.

    An identifier of the package is a module, a name, an attribute, or a string.
    """
    names: set[str] = set(_definitions())
    names.update(path.stem for path in PACKAGE.rglob("*.py"))
    conftest = ast.parse((ROOT / "tests" / "conftest.py").read_text())
    names.update(node.name for node in ast.walk(conftest) if isinstance(node, ast.FunctionDef))
    for tree in _package_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.arg) or (isinstance(node, ast.keyword) and node.arg):
                names.add(node.arg)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.isidentifier():
                    names.add(node.value)
    return frozenset(names)


@functools.cache
def _python_names() -> frozenset[str]:
    """The names of the builtins and of the standard modules the documents cite."""
    modules = (builtins, typing, collections.abc, abc, dataclasses, enum, functools, inspect, types)
    return frozenset(name for module in modules for name in dir(module))


@functools.cache
def _repository_names() -> frozenset[str]:
    """The name of every file and directory of the repository, outside caches."""
    skipped = {".git", ".venv", "site", "node_modules", "__pycache__"}
    return frozenset(
        path.name
        for path in ROOT.rglob("*")
        if not skipped.intersection(path.relative_to(ROOT).parts)
    )


@functools.cache
def _design_sections() -> frozenset[str]:
    return frozenset(
        section
        for path in (ROOT / "design").glob("*.md")
        for section in _DESIGN_SECTION.findall(path.read_text())
    )


def _slug(heading: str) -> str:
    """The anchor GitHub gives a heading."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


@functools.cache
def _headings(path: Path) -> tuple[str, ...]:
    """The text of each heading of *path*, with code and emphasis marks removed."""
    return tuple(
        re.sub(r"[`*]", "", match.group(1)).strip()
        for _, line in _prose_lines(path)
        if (match := _HEADING.match(line))
    )


@functools.cache
def _anchors(path: Path) -> frozenset[str]:
    """The anchors of the headings of *path*, numbered as GitHub numbers repeats."""
    anchors: set[str] = set()
    for heading in _headings(path):
        slug, count = _slug(heading), 1
        candidate = slug
        while candidate in anchors:
            candidate, count = f"{slug}-{count}", count + 1
        anchors.add(candidate)
    return frozenset(anchors)


def _resolve(target: str, document: Path) -> Path | None:
    """The file *target* names, read against the document, the root, and ``probpipe/``."""
    for base in (document.parent, ROOT, PACKAGE):
        if (base / target).exists():
            return base / target
    return None


def _path_problem(span: str, document: Path) -> str | None:
    target = re.sub(r"(:\d+(-\d+)?|::.*)$", "", span).rstrip("/")
    if target in PERSONAL_FILES or not target:
        return None
    if "/" not in target:
        return None if target in _repository_names() else f"`{span}` names no file"
    return None if _resolve(target, document) else f"`{span}` is not a path"


def _is_path(span: str) -> bool:
    if re.search(r"[\s<>*{}$()=,\"'|]|\.\.\.", span) or span.startswith(
        ("http", "~", "/", "-", "#", "@", *REF_PREFIXES)
    ):
        return False
    return "/" in span or span.endswith(_FILE_SUFFIXES)


def _dotted_problem(span: str) -> str | None:
    parts = span.split(".")
    for end in range(len(parts), 0, -1):
        try:
            target = importlib.import_module(".".join(parts[:end]))
        except ImportError:
            continue
        for attribute in parts[end:]:
            if not hasattr(target, attribute):
                return f"`{span}` does not resolve"
            target = getattr(target, attribute)
        return None
    return f"`{span}` does not import"


def _name_problem(span: str) -> str | None:
    head, *attributes = span.removesuffix("()").split(".")
    if head in _python_names() or head in EXTERNAL_NAMES:
        return None
    if _CLASS_NAME.fullmatch(head) and re.search("[a-z]", head):
        if head not in _definitions():
            return f"`{span}` names no class or function of probpipe"
    elif head.startswith("_"):
        if head not in _identifiers():
            return f"`{span}` names nothing in probpipe"
    else:
        return None
    missing = [attribute for attribute in attributes if attribute not in _identifiers()]
    return f"`{span}` names a missing attribute {missing[0]}" if missing else None


def _span_problems(line: str, document: Path) -> Iterator[str]:
    for match in _SPAN.finditer(line):
        span = match.group(2).strip()
        if re.fullmatch(r"probpipe(\.\w+)+", span):
            problem = _dotted_problem(span)
        elif _is_path(span):
            problem = _path_problem(span, document)
        elif _IDENTIFIER.fullmatch(span):
            problem = _name_problem(span)
        else:
            problem = None
        if problem:
            yield problem


def _link_problems(line: str, document: Path) -> Iterator[str]:
    for target in _LINK.findall(line):
        if target.startswith(("http:", "https:", "mailto:")):
            continue
        file, _, anchor = target.partition("#")
        path = document if not file else _resolve(file, document)
        if path is None:
            yield f"link to {target} names no file"
        elif anchor and path.suffix == ".md" and anchor not in _anchors(path):
            yield f"link to {target} names no heading of {path.name}"


def _heading_matches(section: str, path: Path) -> bool:
    wanted = " ".join(section.lower().split())
    for heading in _headings(path):
        heading = " ".join(heading.lower().split())
        if re.fullmatch(r"\d+(\.\d+)*", wanted):
            if re.match(rf"{re.escape(wanted)}[.\s]", heading + " "):
                return True
        elif wanted.startswith(heading) or heading.startswith(wanted):
            return True
    return False


def _pointer_problems(line: str, document: Path) -> Iterator[str]:
    text = line.replace("`", "")
    in_file_pointers: list[tuple[int, int]] = []
    for match in _FILE_POINTER.finditer(text):
        in_file_pointers.append(match.span())
        file, section = match.group(1), match.group(2).strip()
        path = _resolve(file if file.endswith(".md") else f"{file}.md", document)
        if path is None:
            yield f"pointer to {file} names no file"
        elif not _heading_matches(section, path):
            yield f"pointer to {file} § {section} names no heading"
    for match in _LOCAL_POINTER.finditer(text):
        if any(start <= match.start() < end for start, end in in_file_pointers):
            continue
        if not _heading_matches(match.group(1), document):
            yield f"pointer to §{match.group(1)} names no heading of {document.name}"
    for section in _DESIGN_POINTER.findall(text):
        if section not in _design_sections():
            yield f"pointer to design {section} names no section of design/"


def _problems(document: Path) -> list[str]:
    problems: list[str] = []
    for number, line in _prose_lines(document):
        for check in (_span_problems, _link_problems, _pointer_problems):
            problems.extend(f"line {number}: {problem}" for problem in check(line, document))
    return problems


@pytest.mark.parametrize("document", _documents(), ids=lambda path: str(path.relative_to(ROOT)))
def test_every_citation_exists(document):
    """Each path, name, link, and section pointer of the document names something that exists.

    A name ProbPipe does not define, such as a third-party class, belongs in
    ``EXTERNAL_NAMES``.
    """
    assert _problems(document) == []


def test_agents_md_is_short_and_imports_nothing():
    """``AGENTS.md`` fits its line budget and holds its own content.

    Codex and Copilot read an ``@path`` line as text rather than as an import,
    so the file states everything it says itself.
    """
    lines = (ROOT / "AGENTS.md").read_text().splitlines()
    assert len(lines) <= AGENTS_LINE_BUDGET
    assert [line for line in lines if line.startswith("@")] == []


def test_claude_md_imports_agents_md():
    """Claude Code reads ``AGENTS.md`` through the import in ``CLAUDE.md``."""
    assert "@AGENTS.md" in (ROOT / "CLAUDE.md").read_text().splitlines()
