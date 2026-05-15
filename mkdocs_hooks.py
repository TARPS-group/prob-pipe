"""Strip Sphinx-style cross-reference role prefixes from rendered HTML.

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

Fixing this in source would be the right long-term move, but it's a
mechanical edit across hundreds of files that's out of scope for the
docs-reorganisation PR.
"""

from __future__ import annotations

import re

# Roles we strip. ``math`` is intentionally excluded — those should stay
# until the docstrings are migrated to the arithmatex ``$x$`` form.
_SPHINX_ROLE_RE = re.compile(
    r":(?:class|meth|func|attr|mod|data|obj|ref|exc):<code>(~?)([^<]+)</code>"
)


def _replace(match: re.Match[str]) -> str:
    tilde, name = match.group(1), match.group(2)
    if tilde and "." in name:
        # Sphinx convention: ``~module.path.Foo`` displays as ``Foo``.
        name = name.rsplit(".", 1)[-1]
    return f"<code>{name}</code>"


def on_page_content(html: str, **_: object) -> str:
    """MkDocs hook: rewrite Sphinx role spans into plain code spans."""
    return _SPHINX_ROLE_RE.sub(_replace, html)
