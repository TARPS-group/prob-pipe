"""List candidate breaches of the writing rules in Markdown files, for a reader to judge.

The checks are of two kinds. The battery checks the mechanics of a document:

1. a section reference, such as ``II.4``, names a section of ``design/``;
2. every code fence closes;
3. every row of a table has the header's number of columns;
4. no line ends in whitespace, and no line outside a table holds a double space.

The flag scan reads the prose against ``STYLE_GUIDE.md`` §10:

1. a word of the vocabulary of rule 6, or an intensifier of rule 7;
2. an appositive comma after a code span, as in "`raw()`, the one access point";
3. a parenthetical of five or more words;
4. a comma list of four or more items.

Each line the script prints is a candidate, which a reader confirms or
dismisses. The scan skips fenced code blocks and code spans, and the vocabulary
scan skips quoted words, which a document mentions rather than uses.

Usage::

    python scripts/design/prose.py                    # design/*.md
    python scripts/design/prose.py STYLE_GUIDE.md     # the named Markdown files
    python scripts/design/prose.py --strict           # exit 1 while a candidate remains

The script exits with status 0 unless ``--strict`` is given.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "design"

_SECTION_ID = re.compile(r"^## ([IVX]+\.\d+[a-z]?) —", re.M)
_PART_ID = re.compile(r"^# Part ([IVX]+) —", re.M)
_REFERENCE = re.compile(r"(?<![A-Za-z])((?:I{1,3}|IV|VI{0,3}|V)\.\d+[a-z]?)(?!\d)")
_CODE_SPAN = re.compile(r"(`+)(?!`)(.+?)(?<!`)\1(?!`)")
_VOCABULARY = re.compile(
    r"\b(?:surfaces?|surfaced|load-bearing|machinery|escape hatch(?:es)?|doors?|hook(?:s|ed)?"
    r"|plumbing|under the hood|sugar|elid(?:e|es|ed|ing)|knobs?|dials?|seams?|fine print"
    r"|story|stories|pictures?|reach(?:es|ed|ing)?|touch(?:es|ed|ing)?|hand(?:s|ed|ing)? back"
    r"|walk(?:s|ed|ing)?|liv(?:e|es|ed|ing)|rid(?:e|es|ing)|rode|land(?:s|ed|ing)?"
    r"|sit(?:s|ting)?|sat)\b",
    re.I,
)
#: "exactly" in its mathematical sense, as in "exactly one" or "exactly when", is no intensifier.
_INTENSIFIER = re.compile(
    r"\b(?:precisely|deliberately|genuinely|of course|exactly(?!\s+(?:(?:one|two|three|four|zero"
    r"|once|twice|when|if|the|those|its|as)\b|\d)))\b",
    re.I,
)
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”")
_APPOSITIVE = re.compile(r"`[^`]+`, (?:the|a|an) \w")
_PARENTHETICAL = re.compile(r"(?<!\])\(([^()]*)\)")
#: An item of a comma list: up to 80 characters, with no comma and no "and" or "or".
_LIST_ITEM = r"(?:(?!\b(?:and|or)\b)[^,;:()]){1,80}"
_COMMA_LIST = re.compile(rf"(?:{_LIST_ITEM},\s+){{3,}}(?:and|or)\s+\S")
_SENTENCE_END = re.compile(r"[.;!?][*_\"')\]]*\s+")
_UNIT_START = re.compile(r"^\s*(?:[-*+]|\d+\.)\s|^#{1,6}\s|^\s*\|")


@dataclass(frozen=True)
class Candidate:
    """One place a reader should judge: its file, line, kind, and text."""

    path: str
    line: int
    kind: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}: {self.text}"


def section_ids(design: Path = DESIGN) -> frozenset[str]:
    """The ids of the numbered sections and parts of the design reference."""
    ids: set[str] = set()
    for path in design.glob("0*.md"):
        text = path.read_text()
        ids.update(_SECTION_ID.findall(text))
        ids.update(_PART_ID.findall(text))
    return frozenset(ids)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def battery(path: Path, ids: frozenset[str]) -> Iterator[Candidate]:
    """The mechanical candidates of *path*: references, fences, tables, and whitespace."""
    name = _relative(path)
    in_fence = False
    columns: int | None = None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_fence, columns = not in_fence, None
            continue
        if line != line.rstrip():
            yield Candidate(name, number, "trailing whitespace", line.strip())
        if in_fence:
            continue
        for reference in _REFERENCE.findall(line):
            if reference not in ids:
                yield Candidate(name, number, "unknown section", reference)
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            count = stripped.count("|") - stripped.count("\\|") - 1
            if columns is None:
                columns = count
            elif count != columns:
                yield Candidate(
                    name, number, "table columns", f"{count} columns, expected {columns}"
                )
        else:
            columns = None
            # The indentation of a blockquote or a list item is no double space.
            if re.search(r"\S  +\S", re.sub(r"^[>\s]*", "", line)):
                yield Candidate(name, number, "double space", stripped)
    if in_fence:
        yield Candidate(name, 0, "unbalanced fence", "a code fence never closes")


def _units(path: Path) -> Iterator[tuple[int, str]]:
    """The prose units of *path* with their first line: paragraphs, list items, and rows."""
    start, parts = 0, []
    in_fence = False
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line.strip() or _UNIT_START.match(line):
            if parts:
                yield start, " ".join(parts)
            start, parts = number, []
            if in_fence or not line.strip():
                continue
        if not parts:
            start = number
        parts.append(line.strip())
    if parts:
        yield start, " ".join(parts)


def flags(path: Path) -> Iterator[Candidate]:
    """The candidates of *path* against the writing rules of STYLE_GUIDE.md §10."""
    name = _relative(path)
    for number, unit in _units(path):
        if unit.startswith("|") and set(unit) <= set("|-: "):
            continue
        for match in _APPOSITIVE.finditer(unit):
            yield Candidate(name, number, "appositive", match.group(0))
        prose = _CODE_SPAN.sub("CODE", unit)
        used = _QUOTED.sub("QUOTE", prose)
        for pattern, kind in ((_VOCABULARY, "vocabulary"), (_INTENSIFIER, "intensifier")):
            for match in pattern.finditer(used):
                yield Candidate(name, number, kind, match.group(0))
        for match in _PARENTHETICAL.finditer(prose):
            if sum(1 for word in match.group(1).split() if re.search(r"\w", word)) >= 5:
                yield Candidate(name, number, "parenthetical", f"({match.group(1)})")
        for sentence in _SENTENCE_END.split(prose):
            for match in _COMMA_LIST.finditer(sentence):
                yield Candidate(name, number, "comma list", match.group(0).strip())


def candidates(paths: Sequence[Path]) -> list[Candidate]:
    """Every candidate of *paths*, the battery's first, in file order."""
    ids = section_ids()
    found: list[Candidate] = []
    for path in paths:
        found += [*battery(path, ids), *flags(path)]
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", type=Path, help="Markdown files; default design/*.md")
    parser.add_argument("--strict", action="store_true", help="exit 1 while a candidate remains")
    args = parser.parse_args(argv)
    paths = args.paths or sorted(DESIGN.glob("*.md"))
    found = candidates([path.resolve() for path in paths])
    for candidate in found:
        print(candidate)
    print(f"{len(found)} candidates in {len(paths)} files")
    return 1 if args.strict and found else 0


if __name__ == "__main__":
    raise SystemExit(main())
