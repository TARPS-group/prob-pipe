"""Rewrite the rendered HTML of each page: Sphinx roles and design citations.

ProbPipe docstrings use Sphinx role syntax (``:class:`Foo```,
``:meth:`Bar.baz```, ``:attr:`x```, etc.) but the docs are rendered by
mkdocstrings, which does not interpret these roles. Without this hook,
the role prefix appears verbatim in the rendered output, e.g.::

    :class:`Foo`     -> :class:<code>Foo</code>

This hook runs at the ``on_page_content`` stage (after mkdocs has
converted markdown to HTML) and rewrites those patterns to plain
``<code>`` spans. The Sphinx ``~module.path.Name`` short-form is also
honoured: ``:class:`~probpipe.Weights``` renders as ``Weights`` rather
than ``probpipe.Weights``.

Docstrings also cite sections of the design reference by id, as "(II.4)" and
"design V.8" do. The hook links each cited id that a ``## II.4 — Title``
heading of ``design/*.md`` names to that heading on GitHub, and an id that no
heading names stays text. A citation inside a link, a code span, or a code
block stays text too.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping
from html import escape
from pathlib import Path

#: The design reference, beside this file at the repository root.
DESIGN = Path(__file__).resolve().parent / "design"

#: The URL of the design reference on GitHub, which a cited id links into.
DESIGN_URL = "https://github.com/TARPS-group/prob-pipe/blob/main/design"

# Roles we strip. ``math`` is intentionally excluded — those should stay
# until the docstrings are migrated to the arithmatex ``$x$`` form.
_SPHINX_ROLE_RE = re.compile(
    r":(?:class|meth|func|attr|mod|data|obj|ref|exc):<code>(~?)([^<]+)</code>"
)

_SECTION_HEADING = re.compile(r"^## ([IVX]+\.\d+) — .*$", re.M)
_CITATION = re.compile(r"(?<![\w.])([IVX]+\.\d+)(?![\w]|\.\d)")
_MARKUP = re.compile(r"<!--.*?-->|<(?:[^>\"']|\"[^\"]*\"|'[^']*')*>", re.S)
_TAG = re.compile(r"<(/?)([A-Za-z][\w-]*)")
_INLINE_MARKS = re.compile(r"[`*]")

#: The elements whose text keeps its citations as text.
_TEXT_ONLY_ELEMENTS = frozenset({"a", "code", "pre", "script", "style", "textarea"})


def _replace(match: re.Match[str]) -> str:
    tilde, name = match.group(1), match.group(2)
    if tilde and "." in name:
        # Sphinx convention: ``~module.path.Foo`` displays as ``Foo``.
        name = name.rsplit(".", 1)[-1]
    return f"<code>{name}</code>"


def heading_anchor(heading: str) -> str:
    """The anchor GitHub gives a Markdown heading whose source text is *heading*.

    GitHub reads the heading without its code and emphasis marks, lowercases it,
    removes each character that is not a word character, a hyphen, or a space,
    and replaces each space with a hyphen.
    """
    text = _INLINE_MARKS.sub("", heading).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def design_sections(design: Path) -> dict[str, tuple[str, str]]:
    """The URL and the heading text of each section of the design reference, by section id.

    A section is a ``## <id> — <title>`` heading of a Markdown file of *design*,
    such as ``## II.4 — Identity, type & metadata``.
    """
    sections: dict[str, tuple[str, str]] = {}
    for path in sorted(design.glob("*.md")):
        for match in _SECTION_HEADING.finditer(path.read_text()):
            heading = match.group(0).removeprefix("## ").strip()
            url = f"{DESIGN_URL}/{path.name}#{heading_anchor(heading)}"
            sections[match.group(1)] = (url, _INLINE_MARKS.sub("", heading))
    return sections


def link_design_citations(content: str, sections: Mapping[str, tuple[str, str]]) -> str:
    """The HTML *content* with each cited section id of *sections* linked to its heading.

    A citation in the text of an element of :data:`_TEXT_ONLY_ELEMENTS` stays
    text, and so does an id that *sections* does not hold. The title of each
    link is the section's heading.
    """

    def link(match: re.Match[str]) -> str:
        section = sections.get(match.group(1))
        if section is None:
            return match.group(0)
        url, heading = section
        return f'<a href="{url}" title="Design {escape(heading)}">{match.group(1)}</a>'

    pieces: list[str] = []
    depth = 0  # the number of open elements of _TEXT_ONLY_ELEMENTS
    start = 0
    for markup in _MARKUP.finditer(content):
        text = content[start : markup.start()]
        pieces += [text if depth else _CITATION.sub(link, text), markup.group(0)]
        start = markup.end()
        tag = _TAG.match(markup.group(0))
        if tag and tag.group(2).lower() in _TEXT_ONLY_ELEMENTS:
            if tag.group(1):
                depth = max(depth - 1, 0)
            elif not markup.group(0).endswith("/>"):
                depth += 1
    text = content[start:]
    pieces.append(text if depth else _CITATION.sub(link, text))
    return "".join(pieces)


@functools.cache
def _sections() -> dict[str, tuple[str, str]]:
    return design_sections(DESIGN)


def on_page_content(html: str, **_: object) -> str:
    """MkDocs hook: plain code spans for Sphinx roles, and links for design citations."""
    return link_design_citations(_SPHINX_ROLE_RE.sub(_replace, html), _sections())
