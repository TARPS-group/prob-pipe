"""Render a plan written in GitHub-flavoured Markdown as one self-contained HTML page.

Run ``uv run python plans/render/render_plan.py <input.md> <output.html>`` from the
repository root. The output is page content only, with no doctype, ``<html>``,
``<head>``, or ``<body>``, since the host that serves it wraps the file.

Beyond the ``markdown`` package's tables, fenced code, and sane lists, the renderer
handles:

- GitHub alerts (``> [!IMPORTANT]``, ``> [!NOTE]``, ``> [!TIP]``, ``> [!WARNING]``,
  ``> [!CAUTION]``), drawn as labelled callouts. A leading ``**Kind (name).**``
  becomes the callout's label and name.
- ``<details><summary>Technical note: ...</summary> ... </details>`` with Markdown
  inside, drawn as a secondary disclosure.
- A figure: an image alone on its line, optionally followed by an italic caption
  line ``*Figure: ...*``. A local SVG is inlined, with its ids made unique.
- Math: ``$...$``, ``$$...$$``, and ``math`` fenced blocks, typeset by MathJax 3.
- GitHub-style heading ids, so the Markdown's own ``#anchor`` links work, and a
  table of contents built from the headings.
- List indentation as GitHub reads it: a nested item indented under its parent's
  text, and a list that starts right after a paragraph line.

Relative links to other files are rewritten to the repository on GitHub, at the
current branch, unless ``--link-base`` says otherwise.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import markdown

HERE = Path(__file__).resolve().parent
CSS_PATH = HERE / "plan.css"

FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Mono:ital,wght@0,400;0,500;1,400"
    "&family=IBM+Plex+Sans:wght@400;500;600"
    "&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400;1,8..60,600"
    "&display=swap"
)
MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"
HLJS_URL = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"

# The alert types, each with its CSS modifier and its default label.
ALERTS = {
    "IMPORTANT": ("definition", "Definition"),
    "NOTE": ("property", "Note"),
    "TIP": ("intuition", "Intuition"),
    "WARNING": ("warning", "Caution"),
    "CAUTION": ("caution", "Caution"),
}

FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([\w+-]*)")
ALERT_RE = re.compile(r"^(\s*)>\s*\[!(\w+)\]\s*(.*)$")
FIGURE_RE = re.compile(r"^(\s*)!\[([^\]]*)\]\(\s*<?([^\s)>]+)>?(?:\s+\"[^\"]*\")?\s*\)\s*$")
CAPTION_RE = re.compile(r"^\s*([*_])(Figure\b.*)\1\s*$")
LIST_RE = re.compile(r"^(\d{1,9}[.)]|[-*+])( +|$)")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
PLACEHOLDER = "PPBLOCK{}Z"
PLACEHOLDER_RE = re.compile(r"<p>\s*PPBLOCK(\d+)Z\s*</p>|PPBLOCK(\d+)Z")
MATH_TOKEN = "PPMATH{}Z"
MATH_RE = re.compile(r"<p>\s*PPMATH(\d+)Z\s*</p>|PPMATH(\d+)Z")
INLINE_MATH_RE = re.compile(r"(?<![\\$\w])\$(?=[^\s$])((?:\\.|[^$\\\n])+?)(?<=\S)\$(?![\w$])")
DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
HEADING_RE = re.compile(r"<h([1-6])>(.*?)</h\1>", re.S)
SECNUM_RE = re.compile(r"^((?:\d+(?:\.\d+)*\.?)|(?:[A-Z]\.))(\s+)")
LEAD_RE = re.compile(r"^<p><strong>([A-Z][a-z]+)(?: \((.+?)\))?\.</strong>\s*", re.S)


@dataclass
class Context:
    """The state shared by the nested renders of one document."""

    source_dir: Path
    blocks: list[str] = field(default_factory=list)
    math: list[tuple[str, bool]] = field(default_factory=list)
    figure_count: int = 0
    has_code: bool = False

    def stash(self, fragment: str) -> str:
        self.blocks.append(fragment)
        return PLACEHOLDER.format(len(self.blocks) - 1)


# ---------------------------------------------------------------------------
# Line-level passes, each aware of fenced code


def iter_fenced(lines: list[str]):
    """Yield (line, inside_fence, is_fence_line) for each line."""
    fence: str | None = None
    for line in lines:
        m = FENCE_RE.match(line)
        if fence is None and m:
            fence = m.group(2)[0] * len(m.group(2))
            yield line, True, True
        elif (
            fence is not None
            and line.strip().startswith(fence)
            and not line.strip()[len(fence) :].strip()
        ):
            fence = None
            yield line, True, True
        else:
            yield line, fence is not None, False


def math_fences_to_display(lines: list[str]) -> list[str]:
    """Turn each ```math fenced block into a $$ display block."""
    out: list[str] = []
    in_math = False
    for line, _in_fence, is_fence in iter_fenced(lines):
        m = FENCE_RE.match(line) if is_fence else None
        if m and not in_math and m.group(3) == "math":
            in_math = True
            out.append(m.group(1) + "$$")
        elif in_math and is_fence:
            in_math = False
            out.append(line[: len(line) - len(line.lstrip())] + "$$")
        else:
            out.append(line)
    return out


def extract_blocks(lines: list[str], ctx: Context) -> list[str]:
    """Replace alerts, technical-note details, and figures with placeholders."""
    out: list[str] = []
    flagged = list(iter_fenced(lines))
    i = 0
    while i < len(flagged):
        line, in_fence, _ = flagged[i]
        if in_fence:
            out.append(line)
            i += 1
            continue
        alert = ALERT_RE.match(line)
        if alert and alert.group(2).upper() in ALERTS:
            indent, kind, rest = alert.groups()
            body = [rest] if rest.strip() else []
            i += 1
            while i < len(flagged) and flagged[i][0].lstrip().startswith(">"):
                body.append(re.sub(r"^\s*> ?", "", flagged[i][0]))
                i += 1
            out += ["", indent + ctx.stash(render_alert(kind.upper(), body, ctx)), ""]
            continue
        if line.strip().lower().startswith("<details"):
            indent = line[: len(line) - len(line.lstrip())]
            depth, j = 0, i
            inner: list[str] = []
            while j < len(flagged):
                text = flagged[j][0]
                depth += len(re.findall(r"<details\b", text, re.I))
                depth -= len(re.findall(r"</details>", text, re.I))
                inner.append(text)
                j += 1
                if depth <= 0:
                    break
            out += ["", indent + ctx.stash(render_details(inner, ctx)), ""]
            i = j
            continue
        figure = FIGURE_RE.match(line)
        if figure:
            indent, alt, src = figure.groups()
            caption = None
            k = i + 1
            if k < len(flagged) and not flagged[k][0].strip():
                k += 1
            if k < len(flagged) and CAPTION_RE.match(flagged[k][0]):
                caption = CAPTION_RE.match(flagged[k][0]).group(2)
                i = k + 1
            else:
                i += 1
            out += ["", indent + ctx.stash(render_figure(alt, src, caption, ctx)), ""]
            continue
        out.append(line)
        i += 1
    return out


def normalize_lists(lines: list[str]) -> list[str]:
    """Re-indent nested list content to four spaces, as Python-Markdown expects.

    GitHub nests a line under a list item when it is indented to the item's text,
    such as three spaces under ``1.``, while Python-Markdown needs four. A list
    that follows a paragraph line directly also gets the blank line that
    Python-Markdown needs, and so does a table.
    """
    out: list[str] = []
    stack: list[tuple[int, int]] = []  # (source content indent, output content indent)
    prev_blank = True
    prev_table = False
    for line, in_fence, is_fence in iter_fenced(lines):
        if not line.strip():
            out.append("")
            prev_blank = True
            continue
        n = len(line) - len(line.lstrip(" "))
        body = line.lstrip(" ")
        if in_fence and not is_fence:
            base_src, base_out = stack[-1] if stack else (0, 0)
            out.append(" " * (base_out + max(n - base_src, 0)) + body if n >= base_src else line)
            prev_blank = False
            continue
        was_in_list = bool(stack)
        while stack and n < stack[-1][0]:
            stack.pop()
        base_src, base_out = stack[-1] if stack else (0, 0)
        new_indent = base_out + (n - base_src)
        marker = None if is_fence else LIST_RE.match(body)
        if marker:
            if not was_in_list and not prev_blank:
                out.append("")
            spaces = len(marker.group(2)) or 1
            if spaces > 4:
                spaces = 1
            stack.append((n + len(marker.group(1)) + spaces, new_indent + 4))
        is_table = body.startswith("|")
        if is_table and not prev_table and not prev_blank and not marker:
            out.append("")
        out.append(" " * new_indent + body)
        prev_blank = False
        prev_table = is_table
    return out


def extract_math(lines: list[str], ctx: Context) -> list[str]:
    """Replace display and inline math outside code with placeholders."""
    text_parts: list[str] = []
    for line, in_fence, _ in iter_fenced(lines):
        text_parts.append(("\x00" if in_fence else "") + line)

    def stash(tex: str, display: bool) -> str:
        ctx.math.append((tex, display))
        return MATH_TOKEN.format(len(ctx.math) - 1)

    # Display math may span lines, so join the runs of lines outside fences.
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if not run:
            return
        joined = "\n".join(run)
        pieces = re.split(r"(`+[^`]*?`+)", joined)
        for idx, piece in enumerate(pieces):
            if idx % 2:
                continue
            piece = DISPLAY_MATH_RE.sub(lambda m: stash(m.group(1).strip(), True), piece)
            piece = INLINE_MATH_RE.sub(lambda m: stash(m.group(1), False), piece)
            pieces[idx] = piece
        out.extend("".join(pieces).split("\n"))
        run.clear()

    for part in text_parts:
        if part.startswith("\x00"):
            flush()
            out.append(part[1:])
        else:
            run.append(part)
    flush()
    return out


# ---------------------------------------------------------------------------
# Rendering


def render(text: str, ctx: Context) -> str:
    """Render one chunk of Markdown, with its nested blocks, to HTML."""
    lines = text.split("\n")
    lines = math_fences_to_display(lines)
    lines = extract_blocks(lines, ctx)
    lines = normalize_lists(lines)
    lines = extract_math(lines, ctx)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])
    out = md.convert("\n".join(lines))
    if "<pre><code" in out:
        ctx.has_code = True
    return replace_placeholders(out, ctx)


def replace_placeholders(fragment: str, ctx: Context) -> str:
    def block(m: re.Match[str]) -> str:
        return ctx.blocks[int(m.group(1) or m.group(2))]

    previous = None
    while previous != fragment:
        previous = fragment
        fragment = PLACEHOLDER_RE.sub(block, fragment)
    return fragment


def render_inline(text: str, ctx: Context) -> str:
    out = render(text.strip(), ctx).strip()
    if out.startswith("<p>") and out.endswith("</p>") and out.count("<p>") == 1:
        out = out[3:-4]
    return out


def render_alert(kind: str, body: list[str], ctx: Context) -> str:
    modifier, label = ALERTS[kind]
    inner = render("\n".join(body), ctx).strip()
    if kind == "NOTE" and re.match(r"^<(ol|ul)>", inner):
        label = "Properties"
    name = ""
    lead = LEAD_RE.match(inner)
    if lead:
        label = lead.group(1)
        if lead.group(2):
            name = f'<span class="pp-callout-name">{lead.group(2)}</span>'
        inner = "<p>" + inner[lead.end() :]
        inner = re.sub(r"^<p>\s*</p>\s*", "", inner)
        if kind == "NOTE" and label in {"Property", "Properties", "Consequence", "Corollary"}:
            modifier = "property"
    return (
        f'<div class="pp-callout pp-callout--{modifier}" role="note">'
        f'<p class="pp-callout-label">{html.escape(label)}{name}</p>'
        f'<div class="pp-callout-body">\n{inner}\n</div></div>'
    )


def render_details(lines: list[str], ctx: Context) -> str:
    text = "\n".join(lines)
    text = re.sub(r"^\s*<details[^>]*>", "", text, count=1, flags=re.I)
    text = re.sub(r"</details>\s*$", "", text, count=1, flags=re.I)
    summary = ""
    m = re.search(r"<summary>(.*?)</summary>", text, re.S | re.I)
    if m:
        summary = m.group(1).strip()
        text = text[: m.start()] + text[m.end() :]
    kind, title = "", summary
    km = re.match(r"^(Technical note|Note|Proof|Aside|Details?)\s*:\s*(.*)$", summary, re.S | re.I)
    if km:
        kind, title = km.group(1), km.group(2)
    kind_html = f'<span class="pp-technote-kind">{html.escape(kind)}</span>' if kind else ""
    title_html = f'<span class="pp-technote-title">{render_inline(title, ctx)}</span>'
    body = render(text.strip("\n"), ctx)
    return (
        f'<details class="pp-technote"><summary>{kind_html}{title_html}</summary>'
        f'<div class="pp-technote-body">\n{body}\n</div></details>'
    )


def inline_svg(path: Path, alt: str, ctx: Context) -> str:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"<\?xml[^>]*\?>|<!DOCTYPE[^>]*>|<!--.*?-->", "", svg, flags=re.S).strip()
    prefix = f"pp-fig{ctx.figure_count}-"
    ids = re.findall(r'\bid="([^"]+)"', svg)
    for ident in ids:
        svg = svg.replace(f'id="{ident}"', f'id="{prefix}{ident}"')
        svg = svg.replace(f"url(#{ident})", f"url(#{prefix}{ident})")
        svg = svg.replace(f'href="#{ident}"', f'href="#{prefix}{ident}"')
    root = re.search(r"<svg\b[^>]*>", svg)
    if root is None:
        raise ValueError(f"{path} has no <svg> element")
    tag = root.group(0)
    width = re.search(r'\swidth="([\d.]+)', tag)
    view = re.search(r'viewBox="[\d.\s-]+?\s([\d.]+)\s[\d.]+"', tag)
    natural = float(width.group(1)) if width else float(view.group(1)) if view else 600.0
    new_tag = re.sub(r'\s(width|height|role|aria-label)="[^"]*"', "", tag)
    style = (
        f"width:100%;max-width:{natural:g}px;min-width:{min(natural, round(natural * 0.72)):g}px"
    )
    new_tag = new_tag[:-1] + (
        f' role="img" aria-label="{html.escape(alt, quote=True)}" focusable="false" style="{style}">'
    )
    return svg.replace(tag, new_tag, 1)


def render_figure(alt: str, src: str, caption: str | None, ctx: Context) -> str:
    ctx.figure_count += 1
    path = (ctx.source_dir / src).resolve()
    if src.startswith(("http://", "https://", "data:")) or not path.exists():
        media = f'<img src="{html.escape(src, quote=True)}" alt="{html.escape(alt, quote=True)}">'
    elif path.suffix.lower() == ".svg":
        media = inline_svg(path, alt, ctx)
    else:
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
        }
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        media = (
            f'<img src="data:{mime.get(path.suffix.lower(), "image/png")};base64,{data}" '
            f'alt="{html.escape(alt, quote=True)}" style="max-width:100%;height:auto">'
        )
    cap = ""
    if caption:
        text = re.sub(r"^Figure\s*[:.]?\s*", "", caption)
        text = text[:1].upper() + text[1:]
        cap = (
            f'<figcaption><span class="pp-figure-label">Figure {ctx.figure_count}.</span>'
            f"{render_inline(text, ctx)}</figcaption>"
        )
    return f'<figure class="pp-figure"><div class="pp-figure-scroll">{media}</div>{cap}</figure>'


# ---------------------------------------------------------------------------
# Post-processing of the whole document


def github_slug(text: str) -> str:
    slug = text.strip().lower()
    slug = re.sub(r"[^\w\- ]", "", slug)
    return slug.replace(" ", "-")


def plain_text(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment))


@dataclass
class Heading:
    level: int
    ident: str
    inner: str
    text: str


def add_heading_ids(body: str) -> tuple[str, list[Heading]]:
    """Give every heading a GitHub-style id, and number spans for its section."""
    seen: dict[str, int] = {}
    headings: list[Heading] = []

    def sub(m: re.Match[str]) -> str:
        level, inner = int(m.group(1)), m.group(2).strip()
        text = plain_text(inner)
        base = github_slug(text)
        count = seen.get(base, 0)
        seen[base] = count + 1
        ident = base if count == 0 else f"{base}-{count}"
        styled = SECNUM_RE.sub(
            lambda s: f'<span class="pp-secnum">{s.group(1)}</span>', inner, count=1
        )
        headings.append(Heading(level, ident, styled, text))
        anchor = ""
        if level > 1:
            anchor = f'<a class="pp-anchor" href="#{ident}" aria-label="Link to this section">#</a>'
        return f'<h{level} id="{ident}">{styled}{anchor}</h{level}>'

    return HEADING_RE.sub(sub, body), headings


def mark_contents_section(body: str) -> str:
    """Wrap the document's own contents list, so that it is set as a compact list."""
    m = re.search(
        r'(<h2 id="(?:table-of-)?contents">.*?</h2>)(.*?)(?=<hr\s*/?>|<h2\b|$)', body, re.S
    )
    if not m:
        return body
    wrapped = f'{m.group(1)}<div class="pp-doc-contents">{m.group(2)}</div>\n'
    return body[: m.start()] + wrapped + body[m.end() :]


def wrap_tables(body: str) -> str:
    def sub(m: re.Match[str]) -> str:
        table = m.group(0)
        head = re.search(r"<tr>(.*?)</tr>", table, re.S)
        cols = len(re.findall(r"<th\b", head.group(1))) if head else 0
        wide = " pp-table-wide" if cols >= 3 else ""
        # GitHub reads an escaped pipe in a cell as a pipe, inside code too.
        table = table.replace("\\|", "|")
        return f'<div class="pp-table-wrap{wide}">{table}</div>'

    return re.sub(r"<table>.*?</table>", sub, body, flags=re.S)


def rewrite_links(body: str, link_base: str | None) -> str:
    """Open links to other pages in a new tab, resolving relative ones against link_base."""

    def sub(m: re.Match[str]) -> str:
        attrs, href = m.group(1), html.unescape(m.group(2))
        if href.startswith("#"):
            return m.group(0)
        if not re.match(r"^[a-z][a-z0-9+.-]*:", href, re.I) and link_base:
            href = urljoin(link_base, href)
        safe = html.escape(href, quote=True)
        return f'<a{attrs} href="{safe}" target="_blank" rel="noopener noreferrer"'

    return re.sub(r'<a\b([^>]*?)\shref="([^"]*)"', sub, body)


def restore_math(body: str, ctx: Context) -> str:
    def sub(m: re.Match[str]) -> str:
        tex, display = ctx.math[int(m.group(1) or m.group(2))]
        escaped = html.escape(tex, quote=False)
        if display:
            return f'<div class="pp-math">$${escaped}$$</div>'
        return f"${escaped}$"

    return MATH_RE.sub(sub, body)


def build_toc(headings: list[Heading]) -> str:
    items: list[str] = []
    open_sub = False
    for h in headings:
        if h.level not in (2, 3) or h.text.strip().lower() in {"contents", "table of contents"}:
            continue
        link = f'<a href="#{h.ident}">{h.inner}</a>'
        if h.level == 2:
            if open_sub:
                items.append("</ol>")
                open_sub = False
            if items:
                items.append("</li>")
            items.append(f"<li>{link}")
        else:
            if not items:
                items.append("<li>")
            if not open_sub:
                items.append("<ol>")
                open_sub = True
            items.append(f"<li>{link}</li>")
    if open_sub:
        items.append("</ol>")
    if items:
        items.append("</li>")
    return "<ol>" + "".join(items) + "</ol>"


def short_title(title: str) -> str:
    """A name of at most four words for the page's <title>."""
    words = title.split()
    if len(words) <= 4:
        return title
    items = [s.strip() for s in re.split(r",\s*(?:and\s+)?|\s+and\s+", title) if s.strip()]
    if len(items) >= 3:
        picked = [*items[:2], items[-1]]
        return ", ".join(p[:1].upper() + p[1:] for p in picked)
    return " ".join(words[:4])


def git_link_base(source: Path) -> str | None:
    """The GitHub URL of the source's directory at the current branch, if any."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(source.parent), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        remote = git("remote", "get-url", "origin")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        top = Path(git("rev-parse", "--show-toplevel"))
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.match(r"^(?:git@github\.com:|https://github\.com/)([^/]+/[^/]+?)(?:\.git)?$", remote)
    if not m:
        return None
    rel = source.resolve().parent.relative_to(top.resolve()).as_posix()
    return f"https://github.com/{m.group(1)}/blob/{branch}/{rel + '/' if rel != '.' else ''}"


SCRIPT = """
<script>
(() => {
  const toc = document.getElementById("pp-toc");
  if (!toc) return;
  const wide = window.matchMedia("(min-width: 1080px)");
  const sync = () => { if (wide.matches) toc.open = true; };
  sync();
  wide.addEventListener("change", (e) => { toc.open = e.matches; });
  toc.addEventListener("click", (e) => {
    if (!wide.matches && e.target.closest("a")) toc.open = false;
  });
  const links = new Map();
  toc.querySelectorAll('a[href^="#"]').forEach((a) => {
    links.set(decodeURIComponent(a.getAttribute("href").slice(1)), a);
  });
  const targets = [...links.keys()].map((id) => document.getElementById(id)).filter(Boolean);
  if (!("IntersectionObserver" in window) || !targets.length) return;
  let current = null;
  const mark = (id) => {
    if (id === current) return;
    if (current && links.get(current)) links.get(current).removeAttribute("aria-current");
    current = id;
    if (links.get(id)) links.get(id).setAttribute("aria-current", "true");
  };
  const observer = new IntersectionObserver(() => {
    let best = targets[0];
    for (const t of targets) {
      if (t.getBoundingClientRect().top < window.innerHeight * 0.3) best = t; else break;
    }
    mark(best.id);
  }, { rootMargin: "0px 0px -60% 0px", threshold: [0, 1] });
  targets.forEach((t) => observer.observe(t));
})();
</script>
"""

MATHJAX = f"""
<script>
window.MathJax = {{
  tex: {{ inlineMath: [["$", "$"]], displayMath: [["$$", "$$"]], processEscapes: true }},
  svg: {{ fontCache: "global" }},
  options: {{ skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"] }}
}};
</script>
<script async src="{MATHJAX_URL}"></script>
"""

HLJS = f"""
<script src="{HLJS_URL}"></script>
<script>
if (window.hljs) {{
  hljs.configure({{ cssSelector: 'pre code[class*="language-"]', ignoreUnescapedHTML: true }});
  hljs.highlightAll();
}}
</script>
"""


def render_page(source: Path, title: str | None, eyebrow: str, link_base: str | None) -> str:
    ctx = Context(source_dir=source.parent)
    body = render(source.read_text(encoding="utf-8"), ctx)
    body = restore_math(body, ctx)
    body = wrap_tables(body)
    body, headings = add_heading_ids(body)
    body = mark_contents_section(body)
    body = rewrite_links(body, link_base)

    h1 = re.search(r"<h1 id=\"[^\"]*\">(.*?)</h1>\s*", body, re.S)
    doc_title = plain_text(h1.group(1)).strip() if h1 else source.stem.replace("-", " ").title()
    h1_html = h1.group(0).strip() if h1 else f"<h1>{html.escape(doc_title)}</h1>"
    if h1:
        body = body[: h1.start()] + body[h1.end() :]
    page_title = title or short_title(doc_title)

    css = CSS_PATH.read_text(encoding="utf-8")
    today = dt.date.today().isoformat()
    try:
        shown = source.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        shown = source.name
    has_math = bool(ctx.math)

    parts = [
        f"<title>{html.escape(page_title)}</title>",
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        f'<link rel="stylesheet" href="{FONTS_URL}">',
        f"<style>\n{css}</style>",
        '<div class="pp-page"><div class="pp-layout">',
        f'<header class="pp-header"><p class="pp-eyebrow">{html.escape(eyebrow)}</p>{h1_html}</header>',
        '<aside class="pp-toc-aside"><details class="pp-toc" id="pp-toc">'
        '<summary>Contents</summary><nav aria-label="Contents">'
        f'<p class="pp-toc-title">Contents</p>{build_toc(headings)}</nav></details></aside>',
        f'<article class="pp-doc">\n{body}\n</article>',
        f'<footer class="pp-footer">Rendered from <code>{html.escape(shown)}</code> on {today}.</footer>',
        "</div></div>",
        SCRIPT,
    ]
    if has_math:
        parts.append(MATHJAX)
    if ctx.has_code:
        parts.append(HLJS)
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="the Markdown file")
    parser.add_argument("output", type=Path, help="the HTML file to write")
    parser.add_argument("--title", help="the page's <title>, a name of two to four words")
    parser.add_argument(
        "--eyebrow", default="ProbPipe design", help="the small line above the title"
    )
    parser.add_argument(
        "--link-base",
        help="the URL that relative links resolve against (default: the file's directory on GitHub)",
    )
    args = parser.parse_args()
    link_base = args.link_base or git_link_base(args.input)
    page = render_page(args.input, args.title, args.eyebrow, link_base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(f"wrote {args.output} ({len(page) // 1024} KiB)")


if __name__ == "__main__":
    main()
