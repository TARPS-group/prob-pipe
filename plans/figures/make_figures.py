"""Draw the figures of the declarations design document as SVG files.

Run ``python3 plans/figures/make_figures.py`` from the repository root. It writes
one ``.svg`` file per figure next to this module, and needs only the standard
library.

The visual language, which ``legend.svg`` states for the reader:

- data flows right to left, as a merge ``a & b`` and a density ``p(y | mu)`` are
  written, so a term's outputs are ports on its left edge and its inputs are
  ports on its right edge;
- a term is a rounded box whose badge names its kind;
- an optional input is a hollow port;
- an output at the root, a base value offered whole under one name, has its name
  outside the box;
- the parts of a value are a tree of fields and groups inside the box;
- a batch is a stack of offset boxes, with its levels at the bottom;
- a merge draws its two operands side by side as written, wires each output of
  the right operand into the left operand's input of the same name, fans one
  port of the result out to an input both operands share, and draws the result
  as a dashed boundary with the result's ports on its edges.

Every figure is drawn on a light panel with dark text, so it reads the same on
GitHub's light and dark themes. Each figure is declared at the bottom of this
module as data (terms, ports, and parts), and the layout functions place it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

FIGURE_DIR = Path(__file__).resolve().parent

# Colours: dark ink on a white panel, one hue per kind.
PANEL_FILL = "#ffffff"
PANEL_STROKE = "#d0d7de"
INK = "#1f2328"
MUTED = "#59636e"
WIRE = "#424a53"
BOUNDARY = "#6e7781"
KIND_COLOURS = {  # kind: (stroke and badge, box fill)
    "value": ("#0969da", "#eef6ff"),
    "record": ("#0969da", "#eef6ff"),
    "law": ("#8250df", "#f6f0ff"),
    "kernel": ("#bf3989", "#fff0f7"),
    "function": ("#1a7f37", "#eefbf1"),
    "batch": ("#9a6700", "#fff8d6"),
}

MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

ROW = 22  # the height of one port row
HEADER = 30  # the height of a box's header, which holds the badge and the title
PORT_RADIUS = 4.5
PANEL_PADDING = 18


def text_width(text: str, size: float = 13, mono: bool = True) -> float:
    """Estimate the rendered width of ``text``, generously, for layout."""
    return len(text) * size * (0.61 if mono else 0.56)


# ----------------------------------------------------------------------------
# The declarative model of a figure
# ----------------------------------------------------------------------------


@dataclass
class Part:
    """A node of a value: a field, or a group with children."""

    name: str
    children: list[Part] = field(default_factory=list)


@dataclass
class Port:
    """An input of a term."""

    name: str
    optional: bool = False


@dataclass
class Term:
    """A term as a figure draws it.

    ``outputs`` are the top-level parts the term offers, each drawn with a port
    and with its own parts as a tree below it. ``root_output`` is the name of an
    output at the root, a base value offered whole, drawn outside the box.
    ``body`` holds muted lines of explanation inside the box, and ``levels``
    makes the term a batch, drawn as a stack of boxes.
    """

    name: str
    kind: str
    outputs: list[Part] = field(default_factory=list)
    inputs: list[Port] = field(default_factory=list)
    root_output: str | None = None
    body: list[str] = field(default_factory=list)
    levels: str | None = None
    show_ports: bool = True  # False draws the parts as a bare tree, before outputs are introduced

    def output_names(self) -> list[str]:
        if self.root_output is not None:
            return [self.root_output]
        return [part.name for part in self.outputs]


# ----------------------------------------------------------------------------
# A canvas of SVG primitives that tracks its own extent
# ----------------------------------------------------------------------------


class Canvas:
    def __init__(self, title: str) -> None:
        self.title = title
        self.elements: list[str] = []
        self.min_x = self.min_y = float("inf")
        self.max_x = self.max_y = float("-inf")

    def _extend(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.min_x = min(self.min_x, x0, x1)
        self.min_y = min(self.min_y, y0, y1)
        self.max_x = max(self.max_x, x0, x1)
        self.max_y = max(self.max_y, y0, y1)

    def rect(self, x, y, w, h, *, rx=8, fill="none", stroke=INK, width=1.4, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.elements.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
        )
        self._extend(x, y, x + w, y + h)

    def circle(self, x, y, r, *, fill, stroke, width=1.6):
        self.elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width}"/>'
        )
        self._extend(x - r, y - r, x + r, y + r)

    def line(self, points, *, stroke=WIRE, width=1.5, arrow=False, dash=None):
        """Draw a polyline through ``points``, with an arrowhead at its end."""
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.path(d, points, stroke=stroke, width=width, arrow=arrow, dash=dash)

    def path(self, d, extent_points, *, stroke=WIRE, width=1.5, arrow=False, dash=None):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.elements.append(
            f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash_attr}{marker}/>'
        )
        for x, y in extent_points:
            self._extend(x, y, x, y)

    def text(
        self,
        x,
        y,
        content,
        *,
        size=13,
        mono=True,
        anchor="start",
        fill=INK,
        weight="normal",
        italic=False,
    ):
        family = MONO if mono else SANS
        style = ' font-style="italic"' if italic else ""
        self.elements.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{style}>'
            f"{escape(content)}</text>"
        )
        width = text_width(content, size, mono)
        x0 = {"start": x, "middle": x - width / 2, "end": x - width}[anchor]
        self._extend(x0, y - size, x0 + width, y + size * 0.3)

    def render(self) -> str:
        pad = PANEL_PADDING
        width = self.max_x - self.min_x + 2 * pad
        height = self.max_y - self.min_y + 2 * pad
        dx, dy = pad - self.min_x, pad - self.min_y
        body = "\n    ".join(self.elements)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
            f'viewBox="0 0 {width:.0f} {height:.0f}" role="img">\n'
            f"  <title>{escape(self.title)}</title>\n"
            "  <defs>\n"
            '    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" '
            'markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse">\n'
            f'      <path d="M 0 1 L 9 5 L 0 9 z" fill="{WIRE}"/>\n'
            "    </marker>\n"
            "  </defs>\n"
            f'  <rect x="0.5" y="0.5" width="{width - 1:.0f}" height="{height - 1:.0f}" rx="12" '
            f'fill="{PANEL_FILL}" stroke="{PANEL_STROKE}"/>\n'
            f'  <g transform="translate({dx:.1f},{dy:.1f})">\n    {body}\n  </g>\n'
            "</svg>\n"
        )


# ----------------------------------------------------------------------------
# Drawing one term
# ----------------------------------------------------------------------------


@dataclass
class PlacedTerm:
    """A term drawn at a position, with the heights of its ports."""

    term: Term
    x: float
    y: float
    w: float
    h: float
    output_y: dict[str, float]
    input_y: dict[str, float]

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h


def _rows_of(parts: list[Part], depth: int = 0) -> list[tuple[int, Part]]:
    rows: list[tuple[int, Part]] = []
    for part in parts:
        rows.append((depth, part))
        rows.extend(_rows_of(part.children, depth + 1))
    return rows


def badge_width(kind: str) -> float:
    return text_width(kind, 10.5, mono=False) + 14


def draw_badge(canvas: Canvas, x: float, y: float, kind: str) -> float:
    """Draw a kind badge with its top-left corner at (x, y); return its width."""
    colour = KIND_COLOURS[kind][0]
    w = badge_width(kind)
    canvas.rect(x, y, w, 17, rx=8.5, fill=colour, stroke=colour, width=1)
    canvas.text(
        x + w / 2,
        y + 12.5,
        kind,
        size=10.5,
        mono=False,
        anchor="middle",
        fill="#ffffff",
        weight="bold",
    )
    return w


def term_size(term: Term) -> tuple[float, float]:
    rows = _rows_of(term.outputs)
    left_w = 10.0
    if rows:
        left_w = max(12 + depth * 16 + text_width(part.name) for depth, part in rows) + 12
    right_w = (max(text_width(port.name) for port in term.inputs) + 24) if term.inputs else 0.0
    body_w = (
        (max(text_width(line, 12, mono=False) for line in term.body) + 20) if term.body else 0.0
    )
    header_w = 10 + badge_width(term.kind) + 8 + text_width(term.name, 13.5) + 14
    w = max(header_w, left_w + body_w + right_w, 112)
    n_rows = max(len(rows), 1 if term.root_output else 0, len(term.inputs), len(term.body), 1)
    if term.levels:
        n_rows += 1
    return w, HEADER + n_rows * ROW + 6


def draw_term(canvas: Canvas, term: Term, x: float, y: float) -> PlacedTerm:
    stroke, fill = KIND_COLOURS[term.kind]
    w, h = term_size(term)
    if term.levels:  # a batch: two offset boxes behind the front one
        for offset in (10, 5):
            canvas.rect(x + offset, y - offset, w, h, fill=fill, stroke=stroke, width=1.2)
    canvas.rect(x, y, w, h, fill=fill, stroke=stroke, width=1.6)
    badge_w = draw_badge(canvas, x + 10, y + 7, term.kind)
    canvas.text(x + 10 + badge_w + 8, y + 20, term.name, size=13.5, weight="bold")

    def row_y(index: int) -> float:
        return y + HEADER + index * ROW + ROW / 2

    output_y: dict[str, float] = {}
    rows = _rows_of(term.outputs)
    if term.root_output is not None:
        output_y[term.root_output] = row_y(0)
        canvas.text(x - 11, row_y(0) - 6, term.root_output, anchor="end", weight="bold")
    row_of_part = {}
    for index, (depth, part) in enumerate(rows):
        row_of_part[id(part)] = index
        text_x = x + 12 + depth * 16
        canvas.text(text_x, row_y(index) + 4.5, part.name, fill=INK if depth == 0 else MUTED)
        if depth == 0:
            output_y[part.name] = row_y(index)
    for depth, part in rows:  # the tree's connectors
        parent_index = row_of_part[id(part)]
        stem_x = x + 12 + depth * 16 + 4
        for child in part.children:
            child_y = row_y(row_of_part[id(child)])
            canvas.line(
                [(stem_x, row_y(parent_index) + 10), (stem_x, child_y), (stem_x + 9, child_y)],
                stroke=MUTED,
                width=1.1,
            )

    input_y: dict[str, float] = {}
    for index, port in enumerate(term.inputs):
        input_y[port.name] = row_y(index)
        canvas.text(x + w - 12, row_y(index) + 4.5, port.name, anchor="end")

    if term.body:
        left_w = 10.0
        if rows:
            left_w = max(12 + depth * 16 + text_width(part.name) for depth, part in rows) + 12
        right_w = (max(text_width(port.name) for port in term.inputs) + 24) if term.inputs else 0.0
        centre = x + left_w + (w - left_w - right_w) / 2
        first = (max(len(rows), len(term.inputs), len(term.body)) - len(term.body)) / 2
        for index, line in enumerate(term.body):
            canvas.text(
                centre,
                row_y(first + index) + 4.5,
                line,
                size=12,
                mono=False,
                anchor="middle",
                fill=MUTED,
                italic=True,
            )
    if term.levels:
        canvas.text(x + 12, y + h - 10, f"levels: {term.levels}", size=11.5, mono=False, fill=MUTED)

    for port_y in output_y.values():
        if term.show_ports:
            canvas.circle(x, port_y, PORT_RADIUS, fill=stroke, stroke=stroke)
    for port in term.inputs:
        canvas.circle(
            x + w,
            input_y[port.name],
            PORT_RADIUS,
            fill=PANEL_FILL if port.optional else stroke,
            stroke=stroke,
        )
    return PlacedTerm(term, x, y, w, h, output_y, input_y)


# ----------------------------------------------------------------------------
# Figures: one term, a row of terms, and a merge
# ----------------------------------------------------------------------------


def single_term_figure(
    title: str, term: Term, *, caption: list[str] = (), annotate_sides: bool = False
) -> Canvas:
    canvas = Canvas(title)
    placed = draw_term(canvas, term, 0, 0)
    if annotate_sides:
        canvas.text(
            -6,
            -12,
            "outputs leave on the left",
            size=11.5,
            mono=False,
            anchor="end",
            fill=MUTED,
            italic=True,
        )
        canvas.text(
            placed.right + 6,
            -12,
            "inputs enter on the right",
            size=11.5,
            mono=False,
            fill=MUTED,
            italic=True,
        )
    for index, line in enumerate(caption):
        canvas.text(
            placed.w / 2,
            placed.bottom + 24 + index * 18,
            line,
            size=12,
            mono=False,
            anchor="middle",
            fill=MUTED,
        )
    return canvas


def row_figure(title: str, panels: list[tuple[Term, list[str]]], *, gap: float = 70) -> Canvas:
    """Draw several terms side by side, each with a caption below it."""
    canvas = Canvas(title)
    x = 0.0
    tops = []
    for term, caption in panels:
        w, _ = term_size(term)
        lead = text_width(term.root_output) + 16 if term.root_output else 0
        caption_w = max((text_width(line, 12, mono=False) for line in caption), default=0)
        cell_w = max(w + lead, caption_w)
        box_x = x + lead + (cell_w - w - lead) / 2
        tops.append((term, box_x, caption, x, cell_w))
        x += cell_w + gap
    tallest = max(term_size(term)[1] for term, *_ in tops)
    for term, box_x, caption, cell_x, cell_w in tops:
        placed = draw_term(canvas, term, box_x, 0)
        for index, line in enumerate(caption):
            canvas.text(
                cell_x + cell_w / 2,
                tallest + 26 + index * 18,
                line,
                size=12,
                mono=False,
                anchor="middle",
                fill=MUTED,
            )
        del placed
    return canvas


def merge_figure(
    title: str, left: Term, right: Term, *, result_name: str, result_kind: str
) -> Canvas:
    """Draw ``left & right``: the operands as written, their wires, and the result.

    Each output of ``right`` that names an input of ``left`` feeds it; an input
    of ``left`` that is not fed and that ``right`` also takes is shared, and one
    port of the result fans out to both. The result offers ``left``'s outputs,
    then ``right``'s; it takes ``left``'s remaining inputs, then ``right``'s.
    """
    canvas = Canvas(title)
    right_outputs = right.output_names()
    right_inputs = [port.name for port in right.inputs]
    fed = [port.name for port in left.inputs if port.name in right_outputs]
    left_remaining = [port for port in left.inputs if port.name not in fed]
    shared = [port.name for port in left_remaining if port.name in right_inputs]
    right_remaining = [port for port in right.inputs if port.name not in shared]
    optional_of = {port.name: port.optional for port in left.inputs + right.inputs}
    for port in left_remaining:
        if port.name in shared:  # a shared input is optional only when optional in both
            optional_of[port.name] = port.optional and next(
                p.optional for p in right.inputs if p.name == port.name
            )

    w_left, _ = term_size(left)
    w_right, _ = term_size(right)
    result_outputs = left.output_names() + right_outputs
    label_room = max((text_width(name) for name in result_outputs), default=0)

    boundary_left = 0.0
    boundary_top = 0.0
    band = 16 + 22 * len(left_remaining)
    box_top = boundary_top + band + (6 if not left_remaining else 0)
    left_lead = text_width(left.root_output) + 16 if left.root_output else 0
    left_x = boundary_left + 30 + left_lead
    right_lead = text_width(right.root_output) + 18 if right.root_output else 0
    gap = max(150, 90 + 12 * len(right_outputs) + 11 * len(left_remaining) + right_lead)
    right_x = left_x + w_left + gap
    boundary_right = right_x + w_right + 26 + 10 * max(len(shared), 1)

    placed_left = draw_term(canvas, left, left_x, box_top)
    placed_right = draw_term(canvas, right, right_x, box_top)
    lowest = max(placed_left.bottom, placed_right.bottom)

    def boundary_port(x: float, y: float, name: str, side: str, optional: bool = False) -> None:
        canvas.circle(x, y, PORT_RADIUS, fill=PANEL_FILL if optional else BOUNDARY, stroke=BOUNDARY)
        if side == "left":
            canvas.text(x - 10, y + 4.5, name, anchor="end", weight="bold")
        else:
            canvas.text(x + 10, y + 4.5, name, weight="bold")

    # The left operand's outputs leave straight through the result's left edge.
    for name in left.output_names():
        y = placed_left.output_y[name]
        canvas.line([(left_x - PORT_RADIUS, y), (boundary_left + PORT_RADIUS + 2, y)], arrow=True)
        boundary_port(boundary_left, y, name, "left")

    # The right operand's outputs: each runs left through the gap, feeds the left
    # operand's input of the same name, and leaves through the result's left edge
    # below the left operand.
    channel_top = lowest + 22
    for k, name in enumerate(right_outputs):
        y_port = placed_right.output_y[name]
        lane_x = right_x - 22 - 12 * (len(right_outputs) - 1 - k)
        y_out = channel_top + 22 * k
        canvas.line(
            [
                (right_x - PORT_RADIUS, y_port),
                (lane_x, y_port),
                (lane_x, y_out),
                (boundary_left + PORT_RADIUS + 2, y_out),
            ],
            arrow=True,
        )
        boundary_port(boundary_left, y_out, name, "left")
        if name in fed:
            y_in = placed_left.input_y[name]
            target_x = placed_left.right + PORT_RADIUS + 2
            canvas.path(
                f"M {lane_x:.1f},{y_port:.1f} C {lane_x - 40:.1f},{y_port:.1f} "
                f"{target_x + 40:.1f},{y_in:.1f} {target_x:.1f},{y_in:.1f}",
                [(lane_x, y_port), (target_x, y_in)],
                width=2.2,
                arrow=True,
            )
            canvas.circle(lane_x, y_port, 3, fill=WIRE, stroke=WIRE, width=1)

    # The left operand's remaining inputs enter through the top band of the
    # result's right edge; a shared one fans out to the right operand too.
    for k, port in enumerate(left_remaining):
        y_band = boundary_top + 16 + 22 * k
        y_in = placed_left.input_y[port.name]
        lane_x = placed_left.right + 30 + 11 * k
        canvas.line(
            [
                (boundary_right - PORT_RADIUS - 2, y_band),
                (lane_x, y_band),
                (lane_x, y_in),
                (placed_left.right + PORT_RADIUS + 2, y_in),
            ],
            arrow=True,
        )
        boundary_port(boundary_right, y_band, port.name, "right", optional_of[port.name])
        if port.name in shared:
            fan_x = placed_right.right + 16 + 10 * shared.index(port.name)
            y_right = placed_right.input_y[port.name]
            canvas.line(
                [
                    (fan_x, y_band),
                    (fan_x, y_right),
                    (placed_right.right + PORT_RADIUS + 2, y_right),
                ],
                arrow=True,
            )
            canvas.circle(fan_x, y_band, 3, fill=WIRE, stroke=WIRE, width=1)

    # The right operand's other inputs enter straight through the right edge.
    for port in right_remaining:
        y = placed_right.input_y[port.name]
        canvas.line(
            [(boundary_right - PORT_RADIUS - 2, y), (placed_right.right + PORT_RADIUS + 2, y)],
            arrow=True,
        )
        boundary_port(boundary_right, y, port.name, "right", port.optional)

    boundary_bottom = (
        (channel_top + 22 * (len(right_outputs) - 1) + 16) if right_outputs else lowest + 16
    )
    canvas.rect(
        boundary_left,
        boundary_top,
        boundary_right - boundary_left,
        boundary_bottom - boundary_top,
        rx=14,
        stroke=BOUNDARY,
        width=1.5,
        dash="6 4",
    )
    badge_w = draw_badge(canvas, boundary_left + 4, boundary_top - 27, result_kind)
    canvas.text(
        boundary_left + 4 + badge_w + 8, boundary_top - 14, result_name, size=13.5, weight="bold"
    )
    canvas._extend(boundary_left - label_room - 12, boundary_top, boundary_left, boundary_top)
    return canvas


# ----------------------------------------------------------------------------
# The legend
# ----------------------------------------------------------------------------


def legend_figure(*, terms: bool = True, merges: bool = True) -> Canvas:
    """Draw the legend: the conventions for terms, for merges, or both."""
    title = {
        (True, True): "Legend of the figures",
        (True, False): "How a figure draws a term",
        (False, True): "How a figure draws a merge",
    }[(terms, merges)]
    canvas = Canvas(title)
    text_x = 330
    y = 0.0

    def explain(lines: list[str], top: float, height: float) -> None:
        first = top + height / 2 - (len(lines) - 1) * 9 + 4.5
        for index, line in enumerate(lines):
            canvas.text(text_x, first + index * 18, line, size=13, mono=False)

    if not terms:
        return _legend_merge_rows(canvas, y, explain)

    # Row 1: the direction of flow.
    canvas.line([(250, y + 16), (40, y + 16)], width=2.2, arrow=True)
    canvas.text(
        250, y + 36, "inputs enter", size=11.5, mono=False, anchor="end", fill=MUTED, italic=True
    )
    canvas.text(40, y + 36, "outputs leave", size=11.5, mono=False, fill=MUTED, italic=True)
    explain(
        [
            "Data flows from right to left, as a density",
            "p(y | mu) is written: produced, then received.",
        ],
        y,
        40,
    )
    y += 64

    # Row 2: a term and its kind.
    draw_term(canvas, Term("prior", "law"), 40, y)
    explain(["A term is a rounded box. Its badge names its kind."], y, 38)
    y += 72
    x = 40.0
    for kind in ("value", "law", "kernel", "function", "batch"):
        x += draw_badge(canvas, x, y, kind) + 8
    explain(["The kinds of term: values, laws, kernels,", "functions, and batches."], y - 10, 38)
    y += 50

    # Row 3: outputs.
    placed = draw_term(canvas, Term("r", "value", outputs=[Part("a"), Part("b")]), 40, y)
    explain(
        ["An output is a port on the left edge,", "labelled inside the box with its name."],
        y,
        placed.h,
    )
    y += placed.h + 22

    # Row 4: an output at the root.
    placed = draw_term(
        canvas, Term("k", "kernel", root_output="y", body=["a draw is an array"]), 70, y
    )
    explain(
        ["An output at the root offers a base value whole.", "Its name stands outside the box."],
        y,
        placed.h,
    )
    y += placed.h + 22

    # Row 5: inputs.
    placed = draw_term(
        canvas, Term("k", "kernel", inputs=[Port("mu"), Port("s", optional=True)]), 40, y
    )
    explain(
        ["An input is a port on the right edge.", "A hollow port is an optional input."],
        y,
        placed.h,
    )
    y += placed.h + 22

    # Row 6: parts.
    placed = draw_term(
        canvas, Term("r", "value", outputs=[Part("a"), Part("g", [Part("c"), Part("e")])]), 40, y
    )
    explain(
        [
            "The parts of a value are a tree inside the box:",
            "here the field a and the group g, which holds",
            "the fields c and e.",
        ],
        y,
        placed.h,
    )
    y += placed.h + 30

    # Row 7: a batch.
    placed = draw_term(
        canvas, Term("draws", "batch", outputs=[Part("mu"), Part("tau")], levels="sample"), 40, y
    )
    explain(["A batch is a stack of boxes.", "Its levels are listed at the bottom."], y, placed.h)
    y += placed.h + 26

    if not merges:
        return canvas
    return _legend_merge_rows(canvas, y, explain)


def _legend_merge_rows(canvas: Canvas, y: float, explain) -> Canvas:
    # Row 8: feeding.
    a = draw_term(canvas, Term("a", "kernel", inputs=[Port("x")]), 20, y)
    b = draw_term(canvas, Term("b", "law", outputs=[Part("x")]), 200, y)
    yy = a.input_y["x"]
    canvas.line([(b.x - PORT_RADIUS, yy), (a.right + PORT_RADIUS + 2, yy)], width=2.2, arrow=True)
    explain(
        [
            "In a & b, an output of b feeds the input of a",
            "with the same name: a wire runs from the",
            "output to the input.",
        ],
        y,
        a.h,
    )
    y += a.h + 26

    # Row 9: a merge result and a shared input.
    canvas.rect(20, y, 270, 64, rx=14, stroke=BOUNDARY, width=1.5, dash="6 4")
    canvas.rect(
        40,
        y + 10,
        70,
        44,
        rx=8,
        fill=KIND_COLOURS["kernel"][1],
        stroke=KIND_COLOURS["kernel"][0],
        width=1.6,
    )
    canvas.rect(
        170,
        y + 10,
        70,
        44,
        rx=8,
        fill=KIND_COLOURS["kernel"][1],
        stroke=KIND_COLOURS["kernel"][0],
        width=1.6,
    )
    canvas.circle(
        110, y + 32, PORT_RADIUS, fill=KIND_COLOURS["kernel"][0], stroke=KIND_COLOURS["kernel"][0]
    )
    canvas.circle(
        240, y + 32, PORT_RADIUS, fill=KIND_COLOURS["kernel"][0], stroke=KIND_COLOURS["kernel"][0]
    )
    canvas.line(
        [
            (290 - PORT_RADIUS - 2, y + 12),
            (130, y + 12),
            (130, y + 32),
            (110 + PORT_RADIUS + 2, y + 32),
        ],
        arrow=True,
    )
    canvas.line([(258, y + 12), (258, y + 32), (240 + PORT_RADIUS + 2, y + 32)], arrow=True)
    canvas.circle(258, y + 12, 3, fill=WIRE, stroke=WIRE, width=1)
    canvas.circle(290, y + 12, PORT_RADIUS, fill=BOUNDARY, stroke=BOUNDARY)
    explain(
        [
            "The result of a merge is a dashed boundary,",
            "with the result's ports on its edges. One port",
            "that fans out to both operands is a shared input.",
        ],
        y,
        64,
    )
    return canvas


# ----------------------------------------------------------------------------
# The figures of the document
# ----------------------------------------------------------------------------


def lik() -> Term:
    return Term("lik", "kernel", root_output="y", inputs=[Port("mu")], body=["a draw is an array"])


def prior() -> Term:
    return Term("prior", "law", outputs=[Part("mu"), Part("tau")])


def figures() -> dict[str, Canvas]:
    return {
        "legend": legend_figure(),
        "legend-terms": legend_figure(merges=False),
        "legend-merge": legend_figure(terms=False),
        "record-tree": single_term_figure(
            "The record {a: 1.0, g: {c: 2.0, e: 3.0}} as a tree",
            Term(
                "r",
                "value",
                outputs=[Part("a"), Part("g", [Part("c"), Part("e")])],
                show_ports=False,
            ),
            caption=["paths: a, g, g/c, g/e", "fields: a, g/c, g/e · group: g"],
        ),
        "signature": single_term_figure(
            "The signature of the kernel lik_opt: the output y, and the inputs mu and s",
            Term(
                "lik_opt",
                "kernel",
                root_output="y",
                inputs=[Port("mu"), Port("s", optional=True)],
                body=["a draw is an array"],
            ),
            caption=[
                "lik_opt(y | mu, s): its draw y is an array offered under the name y;",
                "it takes mu, and s is optional",
            ],
            annotate_sides=True,
        ),
        "record-parts": single_term_figure(
            "The record {a, g: {c, e}} and its parts",
            Term("r", "value", outputs=[Part("a"), Part("g", [Part("c"), Part("e")])]),
            caption=["its outputs are a and g, its children"],
        ),
        "output-naming": row_figure(
            "Three ways a term offers what it produces",
            [
                (
                    Term(
                        "lik",
                        "kernel",
                        root_output="y",
                        inputs=[Port("mu")],
                        body=["a draw is an array"],
                    ),
                    ["a base value: offered whole,", "under a name outside it"],
                ),
                (
                    Term("params_prior", "law", outputs=[Part("beta"), Part("sigma")]),
                    ["a record: offers its parts,", "beta and sigma"],
                ),
                (
                    Term(
                        "params_prior",
                        "law",
                        outputs=[Part("params", [Part("beta"), Part("sigma")])],
                    ),
                    ["the record wrapped under params:", "one output, params"],
                ),
            ],
        ),
        "merge-general": merge_figure(
            "The merge a & b of two kernels",
            Term("a", "kernel", outputs=[Part("u"), Part("v")], inputs=[Port("x"), Port("w")]),
            Term("b", "kernel", outputs=[Part("x"), Part("z")], inputs=[Port("w"), Port("q")]),
            result_name="a & b",
            result_kind="kernel",
        ),
        "merge-kernel-law": merge_figure(
            "The merge lik & prior of a likelihood and a prior",
            lik(),
            prior(),
            result_name="lik & prior",
            result_kind="law",
        ),
        "merge-shared-input": merge_figure(
            "The merge lik_nee & lik_lai of two kernels that share the input theta",
            Term("lik_nee", "kernel", root_output="nee", inputs=[Port("theta")]),
            Term("lik_lai", "kernel", root_output="lai", inputs=[Port("theta")]),
            result_name="lik_nee & lik_lai",
            result_kind="kernel",
        ),
        "merge-records": merge_figure(
            "The merge of two records",
            Term("r1", "value", outputs=[Part("a")]),
            Term("r2", "value", outputs=[Part("b"), Part("g", [Part("c")])]),
            result_name="r1 & r2",
            result_kind="value",
        ),
    }


def main() -> None:
    for name, canvas in figures().items():
        (FIGURE_DIR / f"{name}.svg").write_text(canvas.render(), encoding="utf-8")
        print(f"wrote plans/figures/{name}.svg")


if __name__ == "__main__":
    main()
