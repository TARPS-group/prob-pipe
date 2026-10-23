"""Flatten the ProbPipe documentation into a readable knowledge corpus.

The ``design-pipeline`` skill reads this corpus before it talks to the user.
The script needs only the standard library, so it runs before ProbPipe or its
dependencies are installed. It builds the corpus in three steps:

1. it reads the pages of the documentation site from the ``nav`` of
   ``mkdocs.yml``, in the order the site shows them;
2. it flattens each notebook to its markdown and code cells, with short text
   outputs, and resolves each mkdocstrings ``::: probpipe.X`` directive of an
   API page statically, with ``ast``, to the symbol's signature and docstring;
3. it writes the result as tiered files, each small enough to read in one pass,
   and a ``MANIFEST.md`` that lists them.

The script reports each missing source, and each page of the ``nav`` that no
tier includes, on standard error, so a change to the documentation layout shows
when the corpus is built. Parsing every module of ``probpipe/`` needs Python
3.12 or later, which ProbPipe requires. On an older Python, a symbol whose
module does not parse is marked as unresolved.

Usage::

    python3 .claude/skills/design-pipeline/build_corpus.py OUT_DIR
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs"
MKDOCS = REPO / "mkdocs.yml"
PART_CHARS = 120_000
OUTPUT_CHARS = 800

_NAV_PAGE = re.compile(r"([\w./-]+\.(?:md|ipynb))\s*$")
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


def nav_pages() -> list[Path]:
    """The pages of the ``nav`` of ``mkdocs.yml``, in the order the site shows them."""
    pages: list[Path] = []
    in_nav = False
    for line in MKDOCS.read_text().splitlines():
        if line.startswith("nav:"):
            in_nav = True
        elif in_nav and line and not line[0].isspace():
            break
        elif in_nav and (match := _NAV_PAGE.search(line)):
            pages.append(DOCS / match.group(1))
    return pages


def warn(message: str) -> None:
    """Report a problem with a source on standard error."""
    print(f"build_corpus.py: {message}", file=sys.stderr)


def expand(sources: tuple[str, ...], pages: list[Path]) -> list[Path]:
    """The files *sources* name, in order, with the ``nav`` *pages* for each directory."""
    paths: list[Path] = []
    for source in sources:
        if source.endswith("/"):
            matches = [page for page in pages if page.is_relative_to(REPO / source)]
        else:
            matches = [REPO / source]
        if not matches:
            warn(f"no page of the mkdocs.yml nav is under {source}")
        for path in matches:
            if not path.is_file():
                warn(f"{path.relative_to(REPO)} does not exist")
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


# --- mkdocstrings directive resolution ---------------------------------------


def _parse(path: Path) -> ast.Module | None:
    """The syntax tree of *path*, or ``None`` when this Python cannot parse it."""
    try:
        return ast.parse(path.read_text())
    except SyntaxError:
        return None


def _module_file(dotted: str) -> Path | None:
    """The source file of the module *dotted*, or ``None`` when there is none."""
    base = REPO.joinpath(*dotted.split("."))
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


def _find(module: str, name: str, depth: int = 0) -> tuple[ast.AST, str] | None:
    """Locate ``name`` in ``module``, following ``from ... import`` re-exports."""
    path = _module_file(module)
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
                    submodule_file = _module_file(submodule)
                    if submodule_file:
                        submodule_tree = _parse(submodule_file)
                        return (submodule_tree, submodule) if submodule_tree else None
                    return _find(source, alias.name, depth + 1)
    return None


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """The signature of the function *node*, as ``name(args) -> returns``."""
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{node.name}({ast.unparse(node.args)}){returns}"


def _first_line(doc: str | None) -> str:
    """The first paragraph of the docstring *doc*, on one line."""
    return (doc or "").strip().split("\n\n")[0].replace("\n", " ")


def render_symbol(dotted: str) -> str:
    """The signature and docstring of the symbol *dotted*, as markdown."""
    module, _, name = dotted.rpartition(".")
    found = _find(module, name) if module else None
    if found is None:
        found_module = _module_file(dotted)
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


def resolve_directives(markdown: str) -> str:
    """*markdown* with each ``::: probpipe.X`` directive replaced by its symbol."""
    out = []
    for line in markdown.splitlines():
        if line.startswith("::: "):
            out.append(render_symbol(line[4:].strip()))
        elif line.startswith("    ") and out and out[-1].startswith("####"):
            continue  # mkdocstrings options block
        else:
            out.append(line)
    return "\n".join(out)


# --- assembly ----------------------------------------------------------------


def render(path: Path) -> str:
    """The file *path* as one corpus document, headed by its repository path."""
    rel = path.relative_to(REPO)
    body = notebook_to_markdown(path) if path.suffix == ".ipynb" else path.read_text()
    if rel.parts[:2] == ("docs", "api"):
        body = resolve_directives(body)
    return f"\n\n<!-- source: {rel} -->\n\n# FILE: {rel}\n\n{body.strip()}\n"


def split(documents: list[str]) -> list[str]:
    """*documents* joined into parts of at most ``PART_CHARS``, unless one is longer."""
    parts, current = [], ""
    for doc in documents:
        if current and len(current) + len(doc) > PART_CHARS:
            parts.append(current)
            current = ""
        current += doc
    if current:
        parts.append(current)
    return parts


def main() -> None:
    """Build the corpus into the output directory and print its manifest."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = [
        "# ProbPipe knowledge corpus",
        "",
        f"Built from `{REPO}`. Read the files in order.",
        "",
        "| File | Tier | Purpose | Chars |",
        "|---|---|---|---|",
    ]
    pages = nav_pages()
    included: set[Path] = set()
    for tier in TIERS:
        paths = expand(tier.sources, pages)
        included.update(paths)
        docs = [render(p) for p in paths]
        for i, part in enumerate(split(docs), start=1):
            name = f"{tier.slug}.{i}.md"
            text = f"# {tier.title} (part {i})\n\n> {tier.purpose}\n{part}"
            (args.out_dir / name).write_text(text)
            manifest.append(f"| `{name}` | {tier.title} | {tier.purpose} | {len(text):,} |")
    for page in pages:
        if page not in included:
            warn(f"no tier includes the nav page {page.relative_to(REPO)}")
    (args.out_dir / "MANIFEST.md").write_text("\n".join(manifest) + "\n")
    print((args.out_dir / "MANIFEST.md").read_text())


if __name__ == "__main__":
    main()
