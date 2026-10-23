"""Flatten the ProbPipe documentation into a readable knowledge corpus.

The ``design-pipeline`` skill loads this corpus before talking to the user.
It needs only the standard library, so it runs before ProbPipe or its
dependencies are installed:

* notebooks are flattened to markdown and code cells, with short text outputs;
* mkdocstrings ``::: probpipe.X`` directives are resolved statically (via
  ``ast``) to the symbol's signature and docstring;
* the result is written as a handful of tiered files, each small enough to
  read in one pass, plus a ``MANIFEST.md`` describing them.

Usage::

    python3 .claude/skills/design-pipeline/build_corpus.py OUT_DIR
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PART_CHARS = 120_000
OUTPUT_CHARS = 800


@dataclass(frozen=True)
class Tier:
    slug: str
    title: str
    purpose: str
    sources: tuple[str, ...]


TIERS = (
    Tier(
        "1-overview",
        "Overview and orientation",
        "What ProbPipe is, its core types and ops, installation, and help.",
        ("README.md", "docs/index.md", "docs/user_guide.md", "docs/help.md"),
    ),
    Tier(
        "2-tutorials",
        "Tutorials",
        "End-to-end worked analyses; the best source of idiomatic current usage.",
        ("docs/tutorials/*.ipynb",),
    ),
    Tier(
        "3-user-guide",
        "User guide",
        "Feature walkthroughs: records, broadcasting, joints, backends, "
        "conversion, sequential updating, JAX, bagging, emulators, Prefect, "
        "diagnostics.",
        ("docs/user_guide/*.ipynb", "docs/orchestration/*.md"),
    ),
    Tier(
        "4-api-reference",
        "API reference",
        "Signatures and docstrings of the public API, resolved from source.",
        (
            "docs/api/index.md",
            "docs/api/operations.md",
            "docs/api/distributions/*.md",
            "docs/api/records.md",
            "docs/api/inference.md",
            "docs/api/diagnostics.md",
            "docs/api/validation.md",
            "docs/api/workflows.md",
            "docs/api/constraints.md",
            "docs/api/converters.md",
            "docs/api/provenance.md",
            "docs/api/extending.md",
            "docs/api/internals.md",
        ),
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
        ),
    ),
)


def expand(patterns: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(REPO.glob(pattern)) if "*" in pattern else [REPO / pattern]
        paths.extend(p for p in matches if p.is_file() and p not in paths)
    return paths


# --- notebooks ---------------------------------------------------------------


def _text_outputs(cell: dict) -> str:
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


def _module_file(dotted: str) -> Path | None:
    base = REPO.joinpath(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _absolute(module: str, node: ast.ImportFrom, is_package: bool) -> str:
    if not node.level:
        return node.module or ""
    parts = module.split(".")
    keep = len(parts) - node.level + (1 if is_package else 0)
    base = parts[:keep]
    return ".".join(base + ([node.module] if node.module else []))


def _find(module: str, name: str, depth: int = 0) -> tuple[ast.AST, str] | None:
    """Locate ``name`` in ``module``, following ``from ... import`` re-exports."""
    path = _module_file(module)
    if path is None or depth > 8:
        return None
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
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
                    if _module_file(submodule):
                        return ast.parse(_module_file(submodule).read_text()), submodule
                    return _find(source, alias.name, depth + 1)
    return None


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{node.name}({ast.unparse(node.args)}){returns}"


def _first_line(doc: str | None) -> str:
    return (doc or "").strip().split("\n\n")[0].replace("\n", " ")


def render_symbol(dotted: str) -> str:
    module, _, name = dotted.rpartition(".")
    found = _find(module, name) if module else None
    if found is None:
        found_module = _module_file(dotted)
        if found_module:
            doc = ast.get_docstring(ast.parse(found_module.read_text()))
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
    rel = path.relative_to(REPO)
    body = notebook_to_markdown(path) if path.suffix == ".ipynb" else path.read_text()
    if rel.parts[:2] == ("docs", "api"):
        body = resolve_directives(body)
    return f"\n\n<!-- source: {rel} -->\n\n# FILE: {rel}\n\n{body.strip()}\n"


def split(documents: list[str]) -> list[str]:
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
    for tier in TIERS:
        docs = [render(p) for p in expand(tier.sources)]
        for i, part in enumerate(split(docs), start=1):
            name = f"{tier.slug}.{i}.md"
            text = f"# {tier.title} (part {i})\n\n> {tier.purpose}\n{part}"
            (args.out_dir / name).write_text(text)
            manifest.append(f"| `{name}` | {tier.title} | {tier.purpose} | {len(text):,} |")
    (args.out_dir / "MANIFEST.md").write_text("\n".join(manifest) + "\n")
    print((args.out_dir / "MANIFEST.md").read_text())


if __name__ == "__main__":
    main()
