"""Flatten the ProbPipe documentation into a corpus that an agent searches.

The ``design-pipeline`` assistant reads the corpus's overview tier and its index
in full, and it searches the other files when it needs them. The script needs
only the standard library, so it runs before ProbPipe or its dependencies are
installed. It builds the corpus in four steps:

1. it finds the ProbPipe checkout: the ``--repo`` argument, else the
   environment variable ``PROBPIPE_REPO``, else the nearest ancestor of the
   working directory whose ``pyproject.toml`` names ProbPipe;
2. it reads the pages of the documentation site from the ``nav`` of
   ``mkdocs.yml``, flattens each notebook to its markdown and code cells with
   short text outputs, and resolves each mkdocstrings ``::: probpipe.X``
   directive of an API page statically, with ``ast``, to the symbol's
   signature and docstring;
3. it writes the result as tiered part files, each at most ``PART_LINES``
   lines and ``PART_CHARS`` characters, which an agent reads in a few line
   ranges that start at the headings ``MANIFEST.md`` lists;
4. it writes ``INDEX.md``, which lists the public exports with their
   signatures, marks each export that raises ``NotImplementedError`` on every
   call, and lists the inference methods with their priorities and the contents
   of the user guide and the tutorials, and ``MANIFEST.md``, which records the
   checkout's branch, commit, and version and the headings of each part file.

The script reports each missing source, and each page of the ``nav`` that no
tier includes, on standard error, so a change to the documentation layout shows
when the corpus is built. It also warns when the ``probpipe`` that this Python
imports has another version than the checkout, or comes from another
directory. Parsing every module of ``probpipe/`` needs Python 3.12 or later,
which ProbPipe requires. On an older Python, a symbol whose module does not
parse is marked as unresolved.

Usage::

    python3 build_corpus.py [--repo PATH] [OUT_DIR]

``OUT_DIR`` defaults to ``corpus/`` under ``$PROBPIPE_DESIGN_HOME``, else under
``~/.probpipe-design``.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import functools
import json
import os
import re
import subprocess
import sys
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PART_LINES = 1_500
PART_CHARS = 60_000
#: The lines and characters each part keeps free for its title and a continuation heading.
PART_HEADROOM = (12, 1_200)
OUTPUT_CHARS = 800
SIGNATURE_CHARS = 110
SUMMARY_CHARS = 120
PROJECT_NAMES = frozenset({"probpipe", "probpipe-core"})

_NAV_PAGE = re.compile(r"([\w./-]+\.(?:md|ipynb))\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PART = re.compile(r"^\d-[\w-]+\.\d+\.md$")
_DOCS_IMPORT = re.compile(r"from (probpipe(?:\.\w+)+) import \(?([\w,\s]+?)\)?\s*(?:\n|$|\\n)")
_REGISTERED = re.compile(r"registered as ``(\w+)``", re.I)
_PRIORITY = re.compile(r"[Pp]riority (\d+)")
#: The node of a ``type`` statement, which Python 3.12 added; an empty tuple matches no node.
_TYPE_ALIAS = getattr(ast, "TypeAlias", ())


@dataclass(frozen=True)
class Tier:
    """One tier of the corpus: its file slug, title, purpose, and sources.

    A source is a repository path, such as ``README.md``, or a directory of the
    documentation site ending in ``/``, such as ``docs/tutorials/``, which
    stands for the pages of the ``nav`` under it, in ``nav`` order.
    """

    slug: str
    title: str
    purpose: str
    sources: tuple[str, ...]


TIERS = (
    Tier(
        "1-overview",
        "Overview and orientation",
        "What ProbPipe is, a short example, installation, the quickstart, citation, and help.",
        ("README.md", "docs/index.md", "docs/get_started/", "docs/cite.md", "docs/help.md"),
    ),
    Tier(
        "2-tutorials",
        "Tutorials",
        "End-to-end worked analyses of one dataset, from a first model to "
        "forecasts, sequential updating, and a model that can only be simulated.",
        ("docs/tutorials/",),
    ),
    Tier(
        "3-user-guide",
        "User guide",
        "One feature per page: terms, values and records, distributions, joint "
        "models, functions and lifting, conditioning and inference methods, "
        "checking results, conversion and updating, provenance, and Ray.",
        ("docs/user_guide/",),
    ),
    Tier(
        "4-api-reference",
        "API reference",
        "Signatures and docstrings of the public API, resolved from source.",
        ("docs/api/",),
    ),
    Tier(
        "5-design",
        "Design reference (TARGET STATE)",
        "The design intent. It describes the target state and may be ahead of "
        "the code: verify any API taken from here against tiers 1-4 or the "
        "source before proposing it to a user.",
        (
            "design/README.md",
            "design/01-design-principles.md",
            "design/02-shared-abstractions.md",
            "design/03-term-kinds.md",
            "design/04-distributions.md",
            "design/05-functions.md",
            "design/06-operations.md",
            "design/07-distribution-catalog.md",
            "design/glossary.md",
            "design/package-structure.md",
        ),
    ),
)

#: The tiers whose pages the contents of ``INDEX.md`` lists.
INDEXED_TIERS = ("2-tutorials", "3-user-guide")


def warn(message: str) -> None:
    """Report a problem with a source on standard error."""
    print(f"build_corpus.py: {message}", file=sys.stderr)


# --- the checkout ------------------------------------------------------------


def is_checkout(path: Path) -> bool:
    """Whether *path* is the root of a ProbPipe checkout with its documentation."""
    pyproject = path / "pyproject.toml"
    if not (pyproject.is_file() and (path / "probpipe" / "__init__.py").is_file()):
        return False
    try:
        name = tomllib.loads(pyproject.read_text()).get("project", {}).get("name")
    except tomllib.TOMLDecodeError:
        return False
    return name in PROJECT_NAMES


def find_repo(explicit: Path | None) -> Path:
    """The ProbPipe checkout to read, or exit with a message that says how to name one."""
    for source, candidate in (
        ("--repo", explicit),
        ("PROBPIPE_REPO", os.environ.get("PROBPIPE_REPO")),
    ):
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if not is_checkout(path):
                sys.exit(
                    f"build_corpus.py: {source} names {path}, which is not a ProbPipe "
                    "checkout: it needs a pyproject.toml whose project is probpipe or "
                    "probpipe-core, and a probpipe/ package."
                )
            return path
    for path in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if is_checkout(path):
            return path
    sys.exit(
        "build_corpus.py: no ProbPipe checkout found. Clone "
        "https://github.com/TARPS-group/prob-pipe.git and pass its directory with "
        "--repo PATH, set PROBPIPE_REPO, or run the script from inside the clone."
    )


def _git(repo: Path, *args: str) -> str:
    """The output of ``git -C repo args``, or ``unknown`` when git cannot answer."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def checkout_facts(repo: Path) -> dict[str, str]:
    """The branch, commit, build date, and ``pyproject.toml`` version of the checkout."""
    project = tomllib.loads((repo / "pyproject.toml").read_text()).get("project", {})
    return {
        "repository": str(repo),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "built": datetime.date.today().isoformat(),
        "version": str(project.get("version", "unknown")),
    }


def installed_probpipe(cwd: Path) -> tuple[str, str] | None:
    """The version and directory of the ``probpipe`` this Python imports, or ``None``.

    The import runs in a subprocess from *cwd*, so the checkout itself is not on
    the path and the result describes the installed environment.
    """
    code = "import os, probpipe; print(probpipe.__version__); print(os.path.dirname(probpipe.__file__))"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=cwd, timeout=300
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = result.stdout.strip().splitlines()
    return (lines[-2], lines[-1]) if result.returncode == 0 and len(lines) >= 2 else None


# --- sources -----------------------------------------------------------------


def nav_pages(repo: Path) -> list[Path]:
    """The pages of the ``nav`` of ``mkdocs.yml``, in the order the site shows them."""
    pages: list[Path] = []
    in_nav = False
    for line in (repo / "mkdocs.yml").read_text().splitlines():
        if line.startswith("nav:"):
            in_nav = True
        elif in_nav and line and not line[0].isspace():
            break
        elif in_nav and (match := _NAV_PAGE.search(line)):
            pages.append(repo / "docs" / match.group(1))
    return pages


def expand(repo: Path, sources: tuple[str, ...], pages: list[Path]) -> list[Path]:
    """The files *sources* name, in order, with the ``nav`` *pages* for each directory."""
    paths: list[Path] = []
    for source in sources:
        if source.endswith("/"):
            matches = [page for page in pages if page.is_relative_to(repo / source)]
        else:
            matches = [repo / source]
        if not matches:
            warn(f"no page of the mkdocs.yml nav is under {source}")
        for path in matches:
            if not path.is_file():
                warn(f"{path.relative_to(repo)} does not exist")
            elif path not in paths:
                paths.append(path)
    return paths


# --- notebooks ---------------------------------------------------------------


def _text_outputs(cell: dict) -> str:
    """The text outputs of the notebook *cell*, truncated to ``OUTPUT_CHARS``."""
    chunks = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            chunks.append("".join(out.get("text", "")))
        elif "data" in out and "text/plain" in out["data"]:
            chunks.append("".join(out["data"]["text/plain"]))
    text = "\n".join(c.rstrip() for c in chunks if c.strip())
    if len(text) > OUTPUT_CHARS:
        text = text[:OUTPUT_CHARS] + "\n... [output truncated]"
    return text


def notebook_to_markdown(path: Path) -> str:
    """The markdown cells, code cells, and text outputs of the notebook *path*."""
    nb = json.loads(path.read_text())
    parts = []
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", "")).rstrip()
        if not source:
            continue
        if cell["cell_type"] == "markdown":
            parts.append(source)
        elif cell["cell_type"] == "code":
            parts.append(f"```python\n{source}\n```")
            output = _text_outputs(cell)
            if output:
                parts.append(f"Output:\n```text\n{output}\n```")
    return "\n\n".join(parts)


def page_text(path: Path) -> str:
    """The text of the documentation page *path*, with a notebook flattened to markdown."""
    return notebook_to_markdown(path) if path.suffix == ".ipynb" else path.read_text()


# --- static symbol resolution ------------------------------------------------


@functools.cache
def _parse(path: Path) -> ast.Module | None:
    """The syntax tree of *path*, or ``None`` when this Python cannot parse it."""
    try:
        return ast.parse(path.read_text())
    except SyntaxError:
        return None


def _module_file(repo: Path, dotted: str) -> Path | None:
    """The source file of the module *dotted*, or ``None`` when there is none."""
    base = repo.joinpath(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _absolute(module: str, node: ast.ImportFrom, is_package: bool) -> str:
    """The absolute module that the import *node* of *module* imports from."""
    if not node.level:
        return node.module or ""
    parts = module.split(".")
    keep = len(parts) - node.level + (1 if is_package else 0)
    base = parts[:keep]
    return ".".join(base + ([node.module] if node.module else []))


def _find(repo: Path, module: str, name: str, depth: int = 0) -> tuple[ast.AST, str] | None:
    """Locate ``name`` in ``module``, following ``from ... import`` re-exports."""
    path = _module_file(repo, module)
    tree = None if path is None or depth > 8 else _parse(path)
    if tree is None:
        return None
    for node in tree.body:
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node, module
        if isinstance(node, _TYPE_ALIAS) and node.name.id == name:
            return node, module
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return node, module
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    source = _absolute(module, node, path.name == "__init__.py")
                    submodule = f"{source}.{alias.name}"
                    submodule_file = _module_file(repo, submodule)
                    if submodule_file:
                        submodule_tree = _parse(submodule_file)
                        return (submodule_tree, submodule) if submodule_tree else None
                    return _find(repo, source, alias.name, depth + 1)
    return _find_lazy(repo, module, tree, name, depth)


def _find_lazy(
    repo: Path, module: str, tree: ast.Module, name: str, depth: int
) -> tuple[ast.AST, str] | None:
    """Locate ``name`` through the module ``__getattr__`` of *module* that names it.

    A package that exports a name lazily, as ``probpipe`` exports ``PyMCModel``,
    imports a module inside its ``__getattr__`` and returns the attribute of that
    name, so the name is looked up in each module that the function imports.
    """
    is_package = (_module_file(repo, module) or Path()).name == "__init__.py"
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name == "__getattr__"):
            continue
        names = {c.value for c in ast.walk(node) if isinstance(c, ast.Constant)}
        if name not in names:
            continue
        for statement in ast.walk(node):
            if not isinstance(statement, ast.ImportFrom):
                continue
            source = _absolute(module, statement, is_package)
            for alias in statement.names:
                imported = f"{source}.{alias.name}"
                target = imported if _module_file(repo, imported) else source
                if found := _find(repo, target, name, depth + 1):
                    return found
    return None


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """The signature of the function *node*, as ``name(args) -> returns``."""
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{node.name}({ast.unparse(node.args)}){returns}"


def _first_line(doc: str | None) -> str:
    """The first paragraph of the docstring *doc*, on one line."""
    return " ".join((doc or "").strip().split("\n\n")[0].split())


def _plain(text: str) -> str:
    """*text* without Sphinx roles and design-section citations, for a one-line summary."""
    text = re.sub(r":\w+:`~?([^`]+)`", r"`\1`", text)
    return re.sub(r"\s*\((?:[IVX]+\.\d+(?:, )?)+\)", "", text)


def _clip(text: str, limit: int) -> str:
    """*text*, cut to *limit* characters with an ellipsis when it is longer."""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def render_symbol(repo: Path, dotted: str) -> str:
    """The signature and docstring of the symbol *dotted*, as markdown."""
    module, _, name = dotted.rpartition(".")
    found = _find(repo, module, name) if module else None
    if found is None:
        found_module = _module_file(repo, dotted)
        module_tree = _parse(found_module) if found_module else None
        if module_tree:
            doc = ast.get_docstring(module_tree)
            return f"#### `{dotted}` (module)\n\n{doc or '(no docstring)'}"
        return f"#### `{dotted}`\n\n(could not resolve statically; read the source)"
    node, where = found
    header = f"#### `{dotted}`\n\n*defined in `{where}`*"
    if isinstance(node, ast.Module):
        return f"{header}\n\n{ast.get_docstring(node) or '(no docstring)'}"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        doc = ast.get_docstring(node) or "(no docstring)"
        return f"{header}\n\n```python\n{_signature(node)}\n```\n\n{doc}"
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        lines = [
            header,
            f"\n```python\nclass {node.name}({bases})\n```\n",
            ast.get_docstring(node) or "(no docstring)",
        ]
        members = []
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name == "__init__":
                lines.append(f"\nConstructor: `{_signature(item)}`")
            elif not item.name.startswith("_"):
                members.append(f"- `{_signature(item)}` — {_first_line(ast.get_docstring(item))}")
        if members:
            lines.append("\nPublic members:\n" + "\n".join(members))
        return "\n".join(lines)
    return f"{header}\n\n```python\n{ast.unparse(node)[:600]}\n```"


def summarize_symbol(repo: Path, dotted: str) -> tuple[str, str]:
    """The one-line signature and the one-line summary of the symbol *dotted*."""
    module, _, name = dotted.rpartition(".")
    found = _find(repo, module, name)
    if found is None:
        return name, "(could not resolve statically; read the source)"
    node, _ = found
    if isinstance(node, ast.Module):
        return f"{name} (module)", _first_line(ast.get_docstring(node))
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _signature(node), _first_line(ast.get_docstring(node))
    if isinstance(node, ast.ClassDef):
        init = next(
            (
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "__init__"
            ),
            None,
        )
        if init is None:
            return f"class {name}", _first_line(ast.get_docstring(node))
        args = ast.unparse(init.args).removeprefix("self, ").removeprefix("self")
        return f"class {name}({args})", _first_line(ast.get_docstring(node))
    return name, _clip(ast.unparse(node), SUMMARY_CHARS)


def resolve_directives(repo: Path, markdown: str) -> str:
    """*markdown* with each ``::: probpipe.X`` directive replaced by its symbol."""
    out = []
    for line in markdown.splitlines():
        if line.startswith("::: "):
            out.append(render_symbol(repo, line[4:].strip()))
        elif line.startswith("    ") and out and out[-1].startswith("####"):
            continue  # mkdocstrings options block
        else:
            out.append(line)
    return "\n".join(out)


# --- part files --------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """A run of lines of one source document, which a part file holds whole."""

    source: str
    text: str
    starts_document: bool


def render(repo: Path, path: Path) -> str:
    """The file *path* as one corpus document, headed by its repository path."""
    rel = path.relative_to(repo)
    body = page_text(path)
    if rel.parts[:2] == ("docs", "api"):
        body = resolve_directives(repo, body)
    return f"# FILE: {rel}\n\n{body.strip()}\n"


def _cut(lines: list[str], at_heading: bool) -> list[list[str]]:
    """*lines* cut before each heading, or each blank line, that is outside a code fence."""
    runs: list[list[str]] = [[]]
    fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            fence = not fence
        boundary = _HEADING.match(line) if at_heading else not line.strip()
        if not fence and boundary and runs[-1]:
            runs.append([])
        runs[-1].append(line)
    return runs


def _fits(lines: list[str], max_lines: int, max_chars: int) -> bool:
    return len(lines) <= max_lines and sum(len(line) + 1 for line in lines) <= max_chars


def _pack(runs: list[list[str]], max_lines: int, max_chars: int) -> list[list[str]]:
    """*runs* joined greedily into groups that each fit the limits, unless one run is longer."""
    groups: list[list[str]] = []
    for run in runs:
        if groups and _fits(groups[-1] + run, max_lines, max_chars):
            groups[-1].extend(run)
        else:
            groups.append(list(run))
    return groups


def blocks(source: str, document: str, max_lines: int, max_chars: int) -> list[Block]:
    """*document* cut into blocks that each fit the limits.

    The cuts fall before headings first, then at blank lines, and only then
    inside a paragraph, each outside a code fence where the text allows it.
    """
    lines = document.rstrip("\n").split("\n")
    groups: list[list[str]] = []
    for section in _pack(_cut(lines, at_heading=True), max_lines, max_chars):
        if _fits(section, max_lines, max_chars):
            groups.append(section)
            continue
        for paragraph in _pack(_cut(section, at_heading=False), max_lines, max_chars):
            while not _fits(paragraph, max_lines, max_chars):
                size = 1
                while _fits(paragraph[: size + 1], max_lines, max_chars):
                    size += 1
                groups.append(paragraph[:size])
                paragraph = paragraph[size:]
            groups.append(paragraph)
    return [
        Block(source, "\n".join(group) + "\n", starts_document=index == 0)
        for index, group in enumerate(groups)
    ]


def parts(tier: Tier, documents: list[tuple[str, str]]) -> list[str]:
    """The part files of *tier*, each within ``PART_LINES`` lines and ``PART_CHARS`` characters."""
    max_lines, max_chars = PART_LINES - PART_HEADROOM[0], PART_CHARS - PART_HEADROOM[1]
    found: list[str] = []
    current: list[str] = []
    for source, document in documents:
        for block in blocks(source, document, max_lines, max_chars):
            text = block.text
            candidate = [*current, "", *text.split("\n")]
            if current and not _fits(candidate, max_lines, max_chars):
                found.append("\n".join(current))
                current = []
            if not current and not block.starts_document:
                text = f"# FILE: {block.source} (continued)\n\n{text}"
            current.extend(["", *text.rstrip("\n").split("\n")])
    if current:
        found.append("\n".join(current))
    return [
        f"# {tier.title} (part {number})\n\n> {tier.purpose}\n{body}\n"
        for number, body in enumerate(found, start=1)
    ]


def headings(text: str, deepest: int = 2) -> list[tuple[int, int, str]]:
    """The line number, level, and text of each heading of *text* outside code fences."""
    found = []
    fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            fence = not fence
        elif not fence and (match := _HEADING.match(line)) and len(match.group(1)) <= deepest:
            found.append((number, len(match.group(1)), match.group(2)))
    return found


def anchor(heading: str) -> str:
    """The anchor that the site's table of contents gives *heading*."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading).replace("`", "")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


# --- the index ---------------------------------------------------------------


def _all_names(repo: Path, module: str) -> list[str]:
    """The names of the literal ``__all__`` list of *module*, or none when it has none."""
    path = _module_file(repo, module)
    tree = _parse(path) if path else None
    names: list[str] = []
    for node in tree.body if tree else ():
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        if isinstance(target, ast.Name) and target.id == "__all__":
            try:
                names.extend(ast.literal_eval(node.value))
            except ValueError:
                continue
    return names


def _only_raises_not_implemented(node: ast.AST) -> bool:
    """Whether the body of the function *node*, after its docstring, is one ``raise NotImplementedError``."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
    if len(body) != 1 or not isinstance(body[0], ast.Raise) or body[0].exc is None:
        return False
    error = body[0].exc.func if isinstance(body[0].exc, ast.Call) else body[0].exc
    return isinstance(error, ast.Name) and error.id == "NotImplementedError"


def _unconditional_calls(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """The names that the function *node* calls on every run.

    A call counts when it is in a simple statement of the function's own body,
    outside a branch, a loop, a nested function, a lambda, a conditional
    expression, a boolean operator, and a comprehension.
    """
    comprehensions = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    skipped = (ast.Lambda, ast.IfExp, ast.BoolOp, *comprehensions)
    names: set[str] = set()
    pending: list[ast.AST] = [item for item in node.body if _is_simple(item)]
    while pending:
        current = pending.pop()
        if isinstance(current, skipped):
            continue
        if isinstance(current, ast.Call) and isinstance(current.func, ast.Name):
            names.add(current.func.id)
        pending.extend(ast.iter_child_nodes(current))
    return names


def _is_simple(statement: ast.stmt) -> bool:
    """Whether *statement* holds no other statement, as an assignment or a ``return`` does."""
    return not any(isinstance(child, ast.stmt) for child in ast.iter_child_nodes(statement))


def unimplemented_exports(repo: Path) -> list[str]:
    """The top-level exports of ``probpipe`` whose every call raises ``NotImplementedError``.

    An exported function is unimplemented when its body is one ``raise
    NotImplementedError``, when a simple statement of its body calls a
    function of its module whose body is one, as ``mixture`` does, or when
    each route that its module registers for it, as
    ``inverse.register_route(...)`` does, names such a function. A call under
    a branch, such as ``if``, can be skipped, so it does not count.
    """
    found_names = []
    for name in _all_names(repo, "probpipe"):
        found = _find(repo, "probpipe", name)
        if found is None or not isinstance(found[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        node, module = found
        path = _module_file(repo, module)
        tree = _parse(path) if path else None
        if tree is None:
            continue
        stubs = {item.name for item in tree.body if _only_raises_not_implemented(item)}
        called = _unconditional_calls(node)
        routes = [
            {n.id for n in ast.walk(item) if isinstance(n, ast.Name)}
            for item in tree.body
            if isinstance(item, ast.Expr)
            and isinstance(item.value, ast.Call)
            and isinstance(item.value.func, ast.Attribute)
            and isinstance(item.value.func.value, ast.Name)
            and item.value.func.value.id == node.name
        ]
        if (
            _only_raises_not_implemented(node)
            or called & stubs
            or (routes and all(route & stubs for route in routes))
        ):
            found_names.append(name)
    return found_names


def docs_subpackage_names(repo: Path, paths: list[Path]) -> dict[str, list[str]]:
    """The names that the documentation imports from, or documents in, each subpackage."""
    found: dict[str, list[str]] = {}
    for path in paths:
        text = path.read_text()
        for module, names in _DOCS_IMPORT.findall(text):
            for name in re.findall(r"\w+", names):
                found.setdefault(module, [])
                if name not in found[module]:
                    found[module].append(name)
        for dotted in re.findall(r"^::: (probpipe\.\w+(?:\.\w+)+)\s*$", text, re.M):
            module, _, name = dotted.rpartition(".")
            if _module_file(repo, module) and name not in found.setdefault(module, []):
                found[module].append(name)
    return dict(sorted(found.items()))


def _returned_constant(node: ast.ClassDef, method: str) -> object:
    """The constant that the method *method* of the class *node* returns, if it returns one."""
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == method:
            for statement in ast.walk(item):
                if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Constant):
                    return statement.value.value
    return None


def _class_constant(node: ast.ClassDef, attribute: str) -> object:
    """The constant that the class *node* assigns to *attribute*, if it assigns one."""
    for item in node.body:
        if isinstance(item, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == attribute for t in item.targets
        ):
            if isinstance(item.value, ast.Constant):
                return item.value.value
    return None


def _check_doc(node: ast.ClassDef | None, classes: dict[str, ast.ClassDef]) -> str:
    """The first line of the docstring of the ``check`` of *node* or of a base in its module."""
    seen: set[str] = set()
    while node is not None and node.name not in seen:
        seen.add(node.name)
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "check":
                return _first_line(ast.get_docstring(item))
        bases = [b.id for b in node.bases if isinstance(b, ast.Name) and b.id in classes]
        node = classes[bases[0]] if bases else None
    return ""


def _feasibility(node: ast.ClassDef | ast.FunctionDef, classes: dict[str, ast.ClassDef]) -> str:
    """Where the method *node* applies: its docstring's "Applies to" sentence, else its check's."""
    doc = " ".join((ast.get_docstring(node) or "").split())
    if match := re.search(r"(Applies to .+?\.)(?:\s|$)", doc):
        return match.group(1)
    if isinstance(node, ast.ClassDef):
        return _check_doc(node, classes)
    for statement in ast.walk(node):
        if (
            isinstance(statement, ast.Return)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
        ):
            return _check_doc(classes.get(statement.value.func.id), classes)
    return ""


def inference_methods(repo: Path) -> list[tuple[str, int | None, str, str]]:
    """The name, priority, module, and feasibility of each inference method in the source."""
    methods = []
    for path in sorted((repo / "probpipe" / "inference").glob("_*.py")):
        tree = _parse(path)
        body = tree.body if tree else []
        classes = {node.name: node for node in body if isinstance(node, ast.ClassDef)}
        for node in body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                continue
            doc = ast.get_docstring(node) or ""
            registered = _REGISTERED.search(doc)
            if not registered:
                continue
            priority: object = None
            if match := _PRIORITY.search(" ".join(doc.split())):
                priority = int(match.group(1))
            elif isinstance(node, ast.ClassDef):
                priority = _class_constant(node, "_method_priority")
                if priority is None:
                    priority = _returned_constant(node, "priority")
            module = f"probpipe.inference.{path.stem}"
            priority = priority if isinstance(priority, int) else None
            methods.append((registered.group(1), priority, module, _feasibility(node, classes)))
    return sorted(methods, key=lambda m: (m[1] is None, -(m[1] or 0), m[0]))


def build_index(
    repo: Path, facts: dict[str, str], pages: list[Path], located: dict[str, tuple[str, int]]
) -> str:
    """The text of ``INDEX.md``: the exports, the inference methods, and the guide's contents.

    *located* maps each source path to the part file and line where it starts.
    """
    out = [
        "# ProbPipe corpus index",
        "",
        f"ProbPipe {facts['version']}, branch `{facts['branch']}`, "
        f"commit `{facts['commit'][:12]}`, built {facts['built']}.",
        "Each signature is cut to one line, so search the API tier or the source for the full one.",
        "",
        "## Top-level exports of `probpipe`",
        "",
        "An export marked **not implemented** raises `NotImplementedError` on every call, "
        "so a pipeline uses none of them.",
        "",
    ]
    unimplemented = set(unimplemented_exports(repo))
    for name in _all_names(repo, "probpipe"):
        signature, summary = summarize_symbol(repo, f"probpipe.{name}")
        mark = "**not implemented**: " if name in unimplemented else ""
        out.append(
            f"- `{_clip(signature, SIGNATURE_CHARS)}`: "
            f"{mark}{_clip(_plain(summary), SUMMARY_CHARS)}"
        )
    out += ["", "## Subpackage names that the documentation uses", ""]
    for module, names in docs_subpackage_names(repo, pages).items():
        out += [f"### `{module}`", ""]
        for name in names:
            signature, summary = summarize_symbol(repo, f"{module}.{name}")
            out.append(
                f"- `{_clip(signature, SIGNATURE_CHARS)}`: {_clip(_plain(summary), SUMMARY_CHARS)}"
            )
        out.append("")
    out += [
        "## Inference methods",
        "",
        "`condition_on` selects the feasible method of highest priority. A method "
        "without a priority runs only when `condition_on.using(method=...)` names it. "
        "Methods of an optional backend are registered only when the backend imports.",
        "",
        "| Method | Priority | Module | Applies to |",
        "|---|---|---|---|",
    ]
    for name, priority, module, feasibility in inference_methods(repo):
        shown = "named only" if priority is None else str(priority)
        out.append(f"| `{name}` | {shown} | `{module}` | {feasibility.replace('|', '/')} |")
    out += ["", "## Contents of the tutorials and the user guide", ""]
    for path in pages:
        rel = str(path.relative_to(repo))
        if not any(
            path.is_relative_to(repo / s)
            for t in TIERS
            if t.slug in INDEXED_TIERS
            for s in t.sources
        ):
            continue
        part, line = located.get(rel, ("?", 0))
        out.append(f"- `{rel}` (corpus `{part}`, line {line})")
        for _, level, text in headings(page_text(path)):
            if level == 2:
                out.append(f"  - {text} (`#{anchor(text)}`)")
    return "\n".join(out) + "\n"


# --- assembly ----------------------------------------------------------------


def default_out_dir() -> Path:
    """The corpus directory under ``$PROBPIPE_DESIGN_HOME``, else under ``~/.probpipe-design``."""
    home = os.environ.get("PROBPIPE_DESIGN_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".probpipe-design") / "corpus"


def main() -> None:
    """Build the corpus into the output directory and print its manifest."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--repo", type=Path, default=None, help="the ProbPipe checkout to read")
    args = parser.parse_args()
    repo = find_repo(args.repo)
    out_dir = (args.out_dir or default_out_dir()).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():
        if _PART.match(old.name) or old.name in {"INDEX.md", "MANIFEST.md"}:
            old.unlink()

    facts = checkout_facts(repo)
    installed = installed_probpipe(out_dir)
    if installed is None:
        environment = f"`probpipe` does not import with `{sys.executable}`."
    else:
        version, location = installed
        environment = f"`probpipe` {version} imports from `{location}`."
        from_checkout = Path(location).resolve().is_relative_to(repo)
        if version != facts["version"] and from_checkout:
            warn(
                f"probpipe imports from the checkout, and its installed metadata names "
                f"version {version} while the checkout's pyproject.toml names version "
                f"{facts['version']}. The metadata is stale, and installing the checkout "
                "again with pip install -e refreshes it"
            )
        elif version != facts["version"]:
            warn(
                f"the installed probpipe is version {version}, and the checkout is "
                f"version {facts['version']}. Install the checkout into this "
                "environment so that the code matches the corpus"
            )
        if not from_checkout:
            warn(f"the installed probpipe comes from {location}, outside the checkout {repo}")

    pages = nav_pages(repo)
    included: set[Path] = set()
    located: dict[str, tuple[str, int]] = {}
    rows: list[str] = []
    index_of_parts: list[str] = []
    for tier in TIERS:
        paths = expand(repo, tier.sources, pages)
        included.update(paths)
        documents = [(str(p.relative_to(repo)), render(repo, p)) for p in paths]
        for number, text in enumerate(parts(tier, documents), start=1):
            name = f"{tier.slug}.{number}.md"
            (out_dir / name).write_text(text)
            rows.append(f"| `{name}` | {tier.title} | {len(text.splitlines()):,} | {len(text):,} |")
            index_of_parts += ["", f"### `{name}`", ""]
            for line, level, heading in headings(text):
                if heading.startswith("FILE: "):
                    source = heading.removeprefix("FILE: ")
                    continued = source.endswith(" (continued)")
                    source = source.removesuffix(" (continued)")
                    located.setdefault(source, (name, line))
                    suffix = " (continued)" if continued else ""
                    index_of_parts.append(f"- line {line}: `{source}`{suffix}")
                elif level == 2:
                    index_of_parts.append(f"  - line {line}: {heading}")
    for page in pages:
        if page not in included:
            warn(f"no tier includes the nav page {page.relative_to(repo)}")

    index = build_index(repo, facts, pages, located)
    (out_dir / "INDEX.md").write_text(index)
    manifest = [
        "# ProbPipe knowledge corpus",
        "",
        f"- Repository: `{facts['repository']}`",
        f"- Branch: `{facts['branch']}`",
        f"- Commit: `{facts['commit']}`",
        f"- Built: {facts['built']}",
        f"- ProbPipe version in `pyproject.toml`: {facts['version']}",
        f"- Environment: {environment}",
        "",
        "Read `INDEX.md` and the `1-overview` parts in full, and search the other parts "
        "when a question needs them.",
        "",
        "| File | Tier | Lines | Chars |",
        "|---|---|---|---|",
        f"| `INDEX.md` | Index | {len(index.splitlines()):,} | {len(index):,} |",
        *rows,
        "",
        "## Headings of each part",
        *index_of_parts,
    ]
    (out_dir / "MANIFEST.md").write_text("\n".join(manifest) + "\n")
    print(f"Corpus written to {out_dir}")
    print("\n".join(manifest[: 14 + len(rows)]))


if __name__ == "__main__":
    main()
