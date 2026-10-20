"""Draw the figures of the declarations design document as SVG files.

Run ``python3 plans/figures/make_figures.py`` from the repository root. It writes
one ``.svg`` file per figure next to this module, and needs only the standard
library.

The visual language, which the legend figures state for the reader:

- data flows from right to left, as a density ``p(y | mu)`` is written, so an
  object's outputs are ports on its left edge and its inputs are ports on its
  right edge;
- an object is a rounded box whose badge names its kind;
- an optional input is a hollow port;
- an output at the root, which names a base value whole, has its name outside
  the box;
- the parts of a value are a tree of fields and groups inside the box;
- a batch is a stack of offset boxes, with its levels at the bottom;
- a merge draws its two operands side by side as written, wires each output of
  the right operand into the left operand's input of the same name, fans one
  port of the result out to an input both operands share, and draws the result
  as a dashed boundary with the result's ports on its edges;
- an invalid merge has a red boundary and a red cross, and the ports, wires, or
  operand at fault are red.

Every figure is drawn on a light panel with dark text, so it reads the same on
GitHub's light and dark themes. Each figure is declared at the bottom of this
module as data (objects, ports, and parts), and the layout functions place it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

FIGURE_DIR = Path(__file__).resolve().parent

# Colours: dark ink on a white panel, one hue per kind, and red for a fault.
PANEL_FILL = "#ffffff"
PANEL_STROKE = "#d0d7de"
INK = "#1f2328"
MUTED = "#59636e"
WIRE = "#424a53"
BOUNDARY = "#6e7781"
FAULT = "#cf222e"
KIND_COLOURS = {  # kind: (stroke and badge, box fill)
    "value": ("#0969da", "#eef6ff"),
    "record": ("#0550ae", "#eef6ff"),
    "law": ("#8250df", "#f6f0ff"),
    "kernel": ("#bf3989", "#fff0f7"),
    "function": ("#1a7f37", "#eefbf1"),
    "batch": ("#9a6700", "#fff8d6"),
}

MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

ROW = 22  # the height of one port row
HEADER = 30  # the height of a box's header, which holds the badge and the name
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
    """A node of a value: a field, or a group with children.

    ``label`` is the text drawn, such as ``"x: (3,)"``, when it differs from the
    name that wires match on.
    """

    name: str
    children: list[Part] = field(default_factory=list)
    label: str | None = None

    @property
    def text(self) -> str:
        return self.label or self.name


@dataclass
class Port:
    """An input of an object."""

    name: str
    optional: bool = False
    label: str | None = None
    factor: bool = False  # a factor input of a law: drawn hollow and dashed (§ 5.2)

    @property
    def text(self) -> str:
        return self.label or self.name


@dataclass
class Obj:
    """An object as a figure draws it.

    ``outputs`` are the object's outputs, each drawn with a port and with its own
    parts as a tree below it. ``root_output`` is the name of an output at the
    root, which names a base value whole, drawn outside the box.
    ``body`` holds muted lines of explanation inside the box, and ``levels``
    makes the object a batch, drawn as a stack of boxes. ``faults`` names the
    ports drawn red, and ``faulty`` draws the whole box red.
    """

    name: str
    kind: str
    outputs: list[Part] = field(default_factory=list)
    inputs: list[Port] = field(default_factory=list)
    root_output: str | None = None
    body: list[str] = field(default_factory=list)
    levels: str | None = None
    show_ports: bool = True  # False draws the parts as a bare tree, before outputs are introduced
    paths_only: bool = False  # True mutes every part: paths that are not outputs
    faults: set[str] = field(default_factory=set)
    faulty: bool = False

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

    def extend(self, x0: float, y0: float, x1: float, y1: float) -> None:
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
        self.extend(x, y, x + w, y + h)

    def circle(self, x, y, r, *, fill, stroke, width=1.6, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width}"{dash_attr}/>'
        )
        self.extend(x - r, y - r, x + r, y + r)

    def line(self, points, *, stroke=WIRE, width=1.5, arrow=False, dash=None):
        """Draw a polyline through ``points``, with an arrowhead at its end."""
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.path(d, points, stroke=stroke, width=width, arrow=arrow, dash=dash)

    def path(self, d, extent_points, *, stroke=WIRE, width=1.5, arrow=False, dash=None):
        marker_id = "arrow-fault" if stroke == FAULT else "arrow"
        marker = f' marker-end="url(#{marker_id})"' if arrow else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.elements.append(
            f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash_attr}{marker}/>'
        )
        for x, y in extent_points:
            self.extend(x, y, x, y)

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
        self.extend(x0, y - size, x0 + width, y + size * 0.3)

    def cross(self, x, y):
        """Draw the red cross that marks an invalid merge, centred at (x, y)."""
        self.circle(x, y, 10, fill=FAULT, stroke=FAULT, width=1)
        for dx in (-1, 1):
            self.line(
                [(x - 4.5, y - 4.5 * dx), (x + 4.5, y + 4.5 * dx)], stroke="#ffffff", width=2.2
            )

    def render(self) -> str:
        pad = PANEL_PADDING
        width = self.max_x - self.min_x + 2 * pad
        height = self.max_y - self.min_y + 2 * pad
        dx, dy = pad - self.min_x, pad - self.min_y
        body = "\n    ".join(self.elements)
        markers = "".join(
            f'    <marker id="{marker_id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" '
            'markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse">\n'
            f'      <path d="M 0 1 L 9 5 L 0 9 z" fill="{colour}"/>\n'
            "    </marker>\n"
            for marker_id, colour in (("arrow", WIRE), ("arrow-fault", FAULT))
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
            f'viewBox="0 0 {width:.0f} {height:.0f}" role="img">\n'
            f"  <title>{escape(self.title)}</title>\n"
            f"  <defs>\n{markers}  </defs>\n"
            f'  <rect x="0.5" y="0.5" width="{width - 1:.0f}" height="{height - 1:.0f}" rx="12" '
            f'fill="{PANEL_FILL}" stroke="{PANEL_STROKE}"/>\n'
            f'  <g transform="translate({dx:.1f},{dy:.1f})">\n    {body}\n  </g>\n'
            "</svg>\n"
        )


# ----------------------------------------------------------------------------
# Drawing one object
# ----------------------------------------------------------------------------


@dataclass
class Placed:
    """An object drawn at a position, with the heights of its ports."""

    obj: Obj
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


BADGE_TEXT = {"value": "base value"}


def badge_width(kind: str) -> float:
    return text_width(BADGE_TEXT.get(kind, kind), 10.5, mono=False) + 14


def draw_badge(canvas: Canvas, x: float, y: float, kind: str) -> float:
    """Draw a kind badge with its top-left corner at (x, y); return its width."""
    colour = KIND_COLOURS[kind][0]
    w = badge_width(kind)
    canvas.rect(x, y, w, 17, rx=8.5, fill=colour, stroke=colour, width=1)
    canvas.text(
        x + w / 2,
        y + 12.5,
        BADGE_TEXT.get(kind, kind),
        size=10.5,
        mono=False,
        anchor="middle",
        fill="#ffffff",
        weight="bold",
    )
    return w


def _side_widths(obj: Obj) -> tuple[float, float]:
    rows = _rows_of(obj.outputs)
    left_w = 10.0
    if rows:
        left_w = max(12 + depth * 16 + text_width(part.text) for depth, part in rows) + 12
    right_w = (max(text_width(port.text) for port in obj.inputs) + 24) if obj.inputs else 0.0
    return left_w, right_w


def obj_size(obj: Obj) -> tuple[float, float]:
    left_w, right_w = _side_widths(obj)
    body_w = (max(text_width(line, 12, mono=False) for line in obj.body) + 20) if obj.body else 0.0
    header_w = 10 + badge_width(obj.kind) + 8 + text_width(obj.name, 13.5) + 14
    w = max(header_w, left_w + body_w + right_w, 112)
    rows = _rows_of(obj.outputs)
    n_rows = max(len(rows), 1 if obj.root_output else 0, len(obj.inputs), len(obj.body), 1)
    if obj.levels:
        n_rows += 1
    return w, HEADER + n_rows * ROW + 6


def draw_obj(canvas: Canvas, obj: Obj, x: float, y: float) -> Placed:
    stroke, fill = KIND_COLOURS[obj.kind]
    w, h = obj_size(obj)
    if obj.levels:  # a batch: two offset boxes behind the front one
        for offset in (10, 5):
            canvas.rect(x + offset, y - offset, w, h, fill=fill, stroke=stroke, width=1.2)
    canvas.rect(
        x,
        y,
        w,
        h,
        fill=fill,
        stroke=FAULT if obj.faulty else stroke,
        width=2.6 if obj.faulty else 1.6,
    )
    badge_w = draw_badge(canvas, x + 10, y + 7, obj.kind)
    canvas.text(x + 10 + badge_w + 8, y + 20, obj.name, size=13.5, weight="bold")

    def row_y(index: float) -> float:
        return y + HEADER + index * ROW + ROW / 2

    output_y: dict[str, float] = {}
    rows = _rows_of(obj.outputs)
    if obj.root_output is not None:
        output_y[obj.root_output] = row_y(0)
        colour = FAULT if obj.root_output in obj.faults else INK
        canvas.text(x - 11, row_y(0) - 6, obj.root_output, anchor="end", weight="bold", fill=colour)
    row_of_part = {}
    for index, (depth, part) in enumerate(rows):
        row_of_part[id(part)] = index
        text_x = x + 12 + depth * 16
        colour = FAULT if part.name in obj.faults and depth == 0 else (INK if depth == 0 else MUTED)
        if obj.paths_only:
            colour = MUTED
        canvas.text(text_x, row_y(index) + 4.5, part.text, fill=colour)
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
    for index, port in enumerate(obj.inputs):
        input_y[port.name] = row_y(index)
        colour = FAULT if port.name in obj.faults else INK
        canvas.text(
            x + w - 12,
            row_y(index) + 4.5,
            port.text,
            anchor="end",
            fill=MUTED if port.factor and colour == INK else colour,
            italic=port.factor,
        )

    if obj.body:
        left_w, right_w = _side_widths(obj)
        centre = x + left_w + (w - left_w - right_w) / 2
        first = (max(len(rows), len(obj.inputs), len(obj.body)) - len(obj.body)) / 2
        for index, line in enumerate(obj.body):
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
    if obj.levels:
        canvas.text(x + 12, y + h - 10, f"levels: {obj.levels}", size=11.5, mono=False, fill=MUTED)

    if obj.show_ports:
        for name, port_y in output_y.items():
            colour = FAULT if name in obj.faults else stroke
            canvas.circle(x, port_y, PORT_RADIUS, fill=colour, stroke=colour)
    for port in obj.inputs:
        colour = FAULT if port.name in obj.faults else stroke
        canvas.circle(
            x + w,
            input_y[port.name],
            PORT_RADIUS + (0.8 if port.factor else 0),
            fill=PANEL_FILL if port.optional or port.factor else colour,
            stroke=colour,
            dash="2.2 1.8" if port.factor else None,
        )
    return Placed(obj, x, y, w, h, output_y, input_y)


# ----------------------------------------------------------------------------
# Figures: one object, a row of objects, and a merge
# ----------------------------------------------------------------------------


def caption_lines(canvas: Canvas, centre: float, top: float, lines: list[str]) -> None:
    for index, line in enumerate(lines):
        canvas.text(
            centre, top + index * 18, line, size=12, mono=False, anchor="middle", fill=MUTED
        )


def single_figure(title: str, obj: Obj, *, notes: list[str] = (), annotate_sides=False) -> Canvas:
    canvas = Canvas(title)
    placed = draw_obj(canvas, obj, 0, 0)
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
    caption_lines(canvas, placed.w / 2, placed.bottom + 24, list(notes))
    return canvas


def row_figure(title: str, panels: list[tuple[Obj, list[str]]], *, gap: float = 70) -> Canvas:
    """Draw several objects side by side, each with a short note below it."""
    canvas = Canvas(title)
    x = 0.0
    cells = []
    for obj, notes in panels:
        w, _ = obj_size(obj)
        lead = text_width(obj.root_output) + 16 if obj.root_output else 0
        notes_w = max((text_width(line, 12, mono=False) for line in notes), default=0)
        cell_w = max(w + lead, notes_w)
        cells.append((obj, x + lead + (cell_w - w - lead) / 2, notes, x, cell_w))
        x += cell_w + gap
    tallest = max(obj_size(obj)[1] for obj, *_ in cells)
    for obj, box_x, notes, cell_x, cell_w in cells:
        draw_obj(canvas, obj, box_x, 0)
        caption_lines(canvas, cell_x + cell_w / 2, tallest + 26, notes)
    return canvas


def merge_figure(
    title: str,
    left: Obj,
    right: Obj,
    *,
    result_name: str,
    result_kind: str,
    invalid: bool = False,
    fault_wires: set[str] = frozenset(),
    fault_results: set[str] = frozenset(),
) -> Canvas:
    """Draw ``left & right``: the operands as written, their wires, and the result.

    Each output of ``right`` that names an input of ``left`` feeds it; an input
    of ``left`` that is not fed and that ``right`` also takes is shared, and one
    port of the result fans out to both. The result has ``left``'s outputs,
    then ``right``'s; it takes ``left``'s remaining inputs, then ``right``'s.
    ``invalid`` marks the merge as refused, and ``fault_wires`` and
    ``fault_results`` name the wires and result ports drawn red.
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

    def wire_colour(name: str) -> str:
        return FAULT if name in fault_wires else WIRE

    w_left, _ = obj_size(left)
    w_right, _ = obj_size(right)
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

    placed_left = draw_obj(canvas, left, left_x, box_top)
    placed_right = draw_obj(canvas, right, right_x, box_top)
    lowest = max(placed_left.bottom, placed_right.bottom)

    def boundary_port(
        x: float, y: float, name: str, side: str, optional: bool = False, factor: bool = False
    ) -> None:
        colour = FAULT if name in fault_results else BOUNDARY
        canvas.circle(
            x,
            y,
            PORT_RADIUS,
            fill=PANEL_FILL if optional or factor else colour,
            stroke=colour,
            dash="2.2 1.8" if factor else None,
        )
        text_colour = FAULT if name in fault_results else INK
        if side == "left":
            canvas.text(x - 10, y + 4.5, name, anchor="end", weight="bold", fill=text_colour)
        else:
            canvas.text(x + 10, y + 4.5, name, weight="bold", fill=text_colour)

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
                stroke=wire_colour(name),
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
        colour = wire_colour(port.name)
        canvas.line(
            [
                (boundary_right - PORT_RADIUS - 2, y_band),
                (lane_x, y_band),
                (lane_x, y_in),
                (placed_left.right + PORT_RADIUS + 2, y_in),
            ],
            stroke=colour,
            arrow=True,
        )
        boundary_port(
            boundary_right, y_band, port.name, "right", optional_of[port.name], port.factor
        )
        if port.name in shared:
            fan_x = placed_right.right + 16 + 10 * shared.index(port.name)
            y_right = placed_right.input_y[port.name]
            canvas.line(
                [
                    (fan_x, y_band),
                    (fan_x, y_right),
                    (placed_right.right + PORT_RADIUS + 2, y_right),
                ],
                stroke=colour,
                arrow=True,
            )
            canvas.circle(fan_x, y_band, 3, fill=colour, stroke=colour, width=1)

    # The right operand's other inputs enter straight through the right edge.
    for port in right_remaining:
        y = placed_right.input_y[port.name]
        canvas.line(
            [(boundary_right - PORT_RADIUS - 2, y), (placed_right.right + PORT_RADIUS + 2, y)],
            stroke=wire_colour(port.name),
            arrow=True,
        )
        boundary_port(boundary_right, y, port.name, "right", port.optional, port.factor)

    boundary_bottom = (
        (channel_top + 22 * (len(right_outputs) - 1) + 16) if right_outputs else lowest + 16
    )
    canvas.rect(
        boundary_left,
        boundary_top,
        boundary_right - boundary_left,
        boundary_bottom - boundary_top,
        rx=14,
        stroke=FAULT if invalid else BOUNDARY,
        width=2.2 if invalid else 1.5,
        dash="6 4",
    )
    badge_w = draw_badge(canvas, boundary_left + 4, boundary_top - 27, result_kind)
    name_x = boundary_left + 4 + badge_w + 8
    canvas.text(name_x, boundary_top - 14, result_name, size=13.5, weight="bold")
    if invalid:
        cross_x = name_x + text_width(result_name, 13.5) + 18
        canvas.cross(cross_x, boundary_top - 18)
        canvas.text(
            cross_x + 16,
            boundary_top - 14,
            "refused",
            size=12.5,
            mono=False,
            fill=FAULT,
            weight="bold",
        )
    canvas.extend(boundary_left - label_room - 12, boundary_top, boundary_left, boundary_top)
    return canvas


# ----------------------------------------------------------------------------
# The legends
# ----------------------------------------------------------------------------


class Legend:
    """Rows of a glyph on the left and its explanation on the right."""

    TEXT_X = 330

    def __init__(self, title: str) -> None:
        self.canvas = Canvas(title)
        self.y = 0.0

    def explain(self, lines: list[str], top: float, height: float) -> None:
        first = top + height / 2 - (len(lines) - 1) * 9 + 4.5
        for index, line in enumerate(lines):
            self.canvas.text(self.TEXT_X, first + index * 18, line, size=13, mono=False)

    def object_rows(self) -> None:
        canvas = self.canvas
        y = self.y
        canvas.line([(250, y + 16), (40, y + 16)], width=2.2, arrow=True)
        canvas.text(
            250,
            y + 36,
            "inputs enter",
            size=11.5,
            mono=False,
            anchor="end",
            fill=MUTED,
            italic=True,
        )
        canvas.text(40, y + 36, "outputs leave", size=11.5, mono=False, fill=MUTED, italic=True)
        self.explain(
            [
                "Data flows from right to left, as a density",
                "p(y | mu) is written: produced, then received.",
            ],
            y,
            40,
        )
        y += 64

        draw_obj(canvas, Obj("d", "law"), 40, y)
        self.explain(["An object is a rounded box.", "Its badge names its kind."], y, 38)
        y += 72
        for row, kinds in enumerate((("value", "record", "law"), ("kernel", "function", "batch"))):
            x = 40.0
            for kind in kinds:  # two rows, so the badges stay clear of the explanation
                x += draw_badge(canvas, x, y + 24 * row, kind) + 8
        self.explain(
            ["The kinds: base values, records, laws,", "kernels, functions, and batches."],
            y,
            41,
        )
        y += 68

        placed = draw_obj(canvas, Obj("r", "record", outputs=[Part("a"), Part("b")]), 40, y)
        self.explain(
            ["An output is a port on the left edge,", "labelled inside the box."], y, placed.h
        )
        y += placed.h + 22

        placed = draw_obj(
            canvas, Obj("k", "kernel", root_output="y", body=["over an array"]), 70, y
        )
        self.explain(
            [
                "An output at the root names a base value",
                "whole; its name stands outside the box.",
            ],
            y,
            placed.h,
        )
        y += placed.h + 22

        placed = draw_obj(
            canvas, Obj("k", "kernel", inputs=[Port("mu"), Port("sigma", optional=True)]), 40, y
        )
        self.explain(
            ["An input is a port on the right edge.", "A hollow port is an optional input."],
            y,
            placed.h,
        )
        y += placed.h + 22

        placed = draw_obj(
            canvas,
            Obj("r", "record", outputs=[Part("a"), Part("g", [Part("b"), Part("c")])]),
            40,
            y,
        )
        self.explain(
            [
                "The parts of a value are a tree inside the",
                "box: the field a, and the group g holding",
                "the fields b and c.",
            ],
            y,
            placed.h,
        )
        y += placed.h + 30

        placed = draw_obj(
            canvas,
            Obj("samples", "batch", outputs=[Part("mu"), Part("sigma")], levels="sample"),
            40,
            y,
        )
        self.explain(
            ["A batch is a stack of boxes,", "with its levels at the bottom."], y, placed.h
        )
        self.y = y + placed.h + 26

    def merge_rows(self) -> None:
        canvas = self.canvas
        y = self.y
        a = draw_obj(canvas, Obj("a", "kernel", inputs=[Port("x")]), 20, y)
        b = draw_obj(canvas, Obj("b", "law", outputs=[Part("x")]), 200, y)
        yy = a.input_y["x"]
        canvas.line(
            [(b.x - PORT_RADIUS, yy), (a.right + PORT_RADIUS + 2, yy)], width=2.2, arrow=True
        )
        self.explain(
            ["In a & b, a wire runs from an output of b", "to the input of a with the same name."],
            y,
            a.h,
        )
        y += a.h + 26

        stroke, fill = KIND_COLOURS["kernel"]
        canvas.rect(20, y, 270, 64, rx=14, stroke=BOUNDARY, width=1.5, dash="6 4")
        for box_x in (40, 170):
            canvas.rect(box_x, y + 10, 70, 44, rx=8, fill=fill, stroke=stroke, width=1.6)
            canvas.circle(box_x + 70, y + 32, PORT_RADIUS, fill=stroke, stroke=stroke)
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
        self.explain(
            [
                "The result is a dashed boundary with its",
                "ports on its edges. A port that fans out",
                "to both operands is a shared input.",
            ],
            y,
            64,
        )
        y += 90

        placed = draw_obj(
            canvas,
            Obj("a & c", "law", inputs=[Port("sigma", optional=True, factor=True)]),
            40,
            y,
        )
        self.explain(
            [
                "A dashed port is a factor input: an input a",
                "factor of a law still takes, which a later",
                "merge can feed or share.",
            ],
            y,
            placed.h,
        )
        y += placed.h + 26

        canvas.rect(20, y, 270, 40, rx=14, stroke=FAULT, width=2.2, dash="6 4")
        canvas.cross(46, y + 20)
        canvas.text(66, y + 24, "refused", size=12.5, mono=False, fill=FAULT, weight="bold")
        canvas.circle(150, y + 20, PORT_RADIUS, fill=FAULT, stroke=FAULT)
        canvas.text(160, y + 24.5, "the fault", fill=FAULT)
        self.explain(
            ["A refused merge has a red boundary and a", "red cross; what breaks the rule is red."],
            y,
            40,
        )
        self.y = y + 40


def legend_figure(*, objects: bool = True, merges: bool = True) -> Canvas:
    title = {
        (True, True): "Legend of the figures",
        (True, False): "How a figure shows an object",
        (False, True): "How a figure shows a merge",
    }[(objects, merges)]
    legend = Legend(title)
    if objects:
        legend.object_rows()
    if merges:
        legend.merge_rows()
    return legend.canvas


# ----------------------------------------------------------------------------
# The figures of the document
# ----------------------------------------------------------------------------


def opt_kernel(name: str = "a", faults: set[str] = frozenset()) -> Obj:
    """A kernel a(y | mu, sigma) whose sigma is optional."""
    return Obj(
        name,
        "kernel",
        root_output="y",
        inputs=[Port("mu"), Port("sigma", optional=True)],
        faults=set(faults),
    )


def structure_figure() -> Canvas:
    """Draw the overview of Part I: a law, its parts, its produced object, and its outputs.

    The produced object stands to the left of the law, since data flows from
    right to left, and a dashed arrow from the law's header marks what it
    produces.
    """
    canvas = Canvas("A law over {a, g: {c, e}}: its parts, its produced object, and its outputs")
    sample = Obj(
        "a sample",
        "record",
        outputs=[
            Part("a", label="a: 0.4"),
            Part("g", [Part("c", label="c: 1.2"), Part("e", label="e: -0.3")]),
        ],
        show_ports=False,
    )
    law = Obj("d", "law", outputs=[Part("a"), Part("g", [Part("c"), Part("e")])])
    sample_w, sample_h = obj_size(sample)
    law_w, law_h = obj_size(law)
    gap = 150
    law_x = sample_w + gap
    placed_sample = draw_obj(canvas, sample, 0, 0)
    placed_law = draw_obj(canvas, law, law_x, 0)
    arrow_y = HEADER / 2
    canvas.line(
        [(law_x - 2, arrow_y), (placed_sample.right + 4, arrow_y)],
        width=1.6,
        arrow=True,
        dash="5 4",
    )
    canvas.text(
        placed_sample.right + gap / 2,
        arrow_y - 7,
        "produces",
        size=12,
        mono=False,
        anchor="middle",
        fill=MUTED,
        italic=True,
    )
    canvas.text(  # the output ports' names are inside the box, as in every figure
        law_x - 12,
        (placed_law.output_y["a"] + placed_law.output_y["g"]) / 2 + 4.5,
        "outputs",
        size=12,
        mono=False,
        anchor="end",
        fill=MUTED,
        italic=True,
    )
    top = max(sample_h, law_h) + 26
    caption_lines(
        canvas,
        placed_sample.w / 2,
        top,
        ["a sample: one value of its produced", "object, a record of type {a, g: {c, e}}"],
    )
    caption_lines(
        canvas,
        law_x + law_w / 2,
        top,
        [
            "paths: a, g, g/c, g/e",
            "parts: d[a], d[g], d[g/c], d[g/e]",
            "outputs: a and g, on the left edge",
        ],
    )
    return canvas


def figures() -> dict[str, Canvas]:
    group = [Part("a"), Part("g", [Part("b"), Part("c")])]
    return {
        "legend": legend_figure(),
        "legend-objects": legend_figure(merges=False),
        "legend-merge": legend_figure(objects=False),
        "object-structure": structure_figure(),
        "produced-paths": row_figure(
            "A law, a sample from it, its mean, and a batch of its samples share one set of paths",
            [
                (
                    Obj(
                        "d",
                        "law",
                        outputs=[Part("a"), Part("g", [Part("c"), Part("e")])],
                        show_ports=False,
                    ),
                    ["a law over {a, g: {c, e}}"],
                ),
                (
                    Obj(
                        "a sample",
                        "record",
                        outputs=[
                            Part("a", label="a: 0.4"),
                            Part("g", [Part("c", label="c: 1.2"), Part("e", label="e: -0.3")]),
                        ],
                        show_ports=False,
                    ),
                    ["one sample from d"],
                ),
                (
                    Obj(
                        "the mean",
                        "record",
                        outputs=[
                            Part("a", label="a: 0.0"),
                            Part("g", [Part("c", label="c: 1.0"), Part("e", label="e: 0.0")]),
                        ],
                        show_ports=False,
                    ),
                    ["the mean of d"],
                ),
                (
                    Obj(
                        "samples",
                        "batch",
                        outputs=[Part("a"), Part("g", [Part("c"), Part("e")])],
                        levels="sample",
                        show_ports=False,
                    ),
                    ["100 samples from d"],
                ),
            ],
            gap=44,
        ),
        "record-tree": single_figure(
            "The record {a: 1.0, g: {b: 2.0, c: 3.0}} as a tree",
            Obj("r", "record", outputs=group, show_ports=False),
            notes=["paths: a, g, g/b, g/c", "fields: a, g/b, g/c · group: g"],
        ),
        "record-parts": single_figure(
            "The record {a, g: {b, c}} and its outputs",
            Obj("r", "record", outputs=group),
            notes=["its outputs are its children, a and g"],
        ),
        "signature": single_figure(
            "The signature of the kernel k(y | mu, sigma), whose sigma is optional",
            Obj(
                "k",
                "kernel",
                root_output="y",
                inputs=[Port("mu"), Port("sigma", optional=True)],
                body=["over an array"],
            ),
            annotate_sides=True,
        ),
        "output-naming": row_figure(
            "The outputs of a base value, a record, and a wrapped record",
            [
                (
                    Obj(
                        "k",
                        "kernel",
                        root_output="y",
                        inputs=[Port("mu")],
                        body=["over an array"],
                    ),
                    ["a base value: one output,", "named outside it"],
                ),
                (
                    Obj("d", "law", outputs=[Part("mu"), Part("sigma")]),
                    ["a record: one output per child"],
                ),
                (
                    Obj("d", "law", outputs=[Part("params", [Part("mu"), Part("sigma")])]),
                    ["the record wrapped under params:", "one output"],
                ),
            ],
        ),
        "produced-outputs-paths": row_figure(
            "Four laws: their produced types, outputs, and paths",
            [
                (
                    Obj("d1", "law", root_output="x", body=["over an array"]),
                    ["over an array x:", "one output, x, at the root;", "no paths"],
                ),
                (
                    Obj("d2", "law", outputs=[Part("mu"), Part("sigma")]),
                    ["over {mu, sigma}:", "outputs mu, sigma;", "paths mu, sigma"],
                ),
                (
                    Obj("d3", "law", outputs=[Part("a"), Part("g", [Part("c"), Part("e")])]),
                    ["over {a, g: {c, e}}:", "outputs a, g;", "paths a, g, g/c, g/e"],
                ),
                (
                    Obj(
                        "d4",
                        "law",
                        outputs=[Part("params", [Part("a"), Part("g", [Part("c"), Part("e")])])],
                    ),
                    ["d3's record wrapped:", "one output, params;", "every path starts params/"],
                ),
            ],
            gap=44,
        ),
        "batch-functions": row_figure(
            "A batch of functions: its inputs and its outputs",
            [
                (
                    Obj("f_j", "function", outputs=[Part("y"), Part("z")], inputs=[Port("x")]),
                    ["one element: f_j maps x to {y, z};", "outputs y and z"],
                ),
                (
                    Obj(
                        "fs",
                        "batch",
                        outputs=[Part("y"), Part("z")],
                        inputs=[Port("x")],
                        body=["elements:", "functions"],
                        levels="j",
                    ),
                    [
                        "the batch: input x, bound once",
                        "for every element; outputs y, z,",
                        "each a batch of functions",
                    ],
                ),
            ],
        ),
        "batch-kernels": row_figure(
            "A batch of kernels: its inputs, and no outputs",
            [
                (
                    Obj("k_j", "kernel", outputs=[Part("y"), Part("w")], inputs=[Port("mu")]),
                    ["one element: k_j, a kernel over {y, w}", "given mu; outputs y and w"],
                ),
                (
                    Obj(
                        "ks",
                        "batch",
                        outputs=[Part("y"), Part("w")],
                        inputs=[Port("mu")],
                        body=["elements: kernels", "no outputs"],
                        levels="j",
                        show_ports=False,
                        paths_only=True,
                    ),
                    [
                        "the batch: input mu, bound once",
                        "for every element; no outputs,",
                        "since its elements are kernels;",
                        "paths y and w",
                    ],
                ),
            ],
        ),
        "merge-general": merge_figure(
            "The merge a & b of two kernels",
            Obj("a", "kernel", outputs=[Part("u"), Part("v")], inputs=[Port("x"), Port("w")]),
            Obj("b", "kernel", outputs=[Part("x"), Part("z")], inputs=[Port("w"), Port("q")]),
            result_name="a & b",
            result_kind="kernel",
        ),
        "merge-kernel-law": merge_figure(
            "The merge lik & prior of a likelihood and a prior",
            Obj("lik", "kernel", root_output="y", inputs=[Port("mu")], body=["over an array"]),
            Obj("prior", "law", outputs=[Part("mu"), Part("tau")]),
            result_name="lik & prior",
            result_kind="law",
        ),
        "merge-shared-input": merge_figure(
            "The merge a & b of two kernels that share the input theta",
            Obj("a", "kernel", root_output="x", inputs=[Port("theta")]),
            Obj("b", "kernel", root_output="y", inputs=[Port("theta")]),
            result_name="a & b",
            result_kind="kernel",
        ),
        "merge-optional-open": merge_figure(
            "An optional input that stays open in the result",
            opt_kernel(),
            Obj("b", "kernel", root_output="mu", inputs=[Port("theta")]),
            result_name="a & b",
            result_kind="kernel",
        ),
        "merge-optional-fed": merge_figure(
            "An optional input that the right operand feeds",
            opt_kernel(),
            Obj("b", "law", outputs=[Part("mu"), Part("sigma")]),
            result_name="a & b",
            result_kind="law",
        ),
        "merge-records": merge_figure(
            "The merge of two records",
            Obj("r1", "record", outputs=[Part("a")]),
            Obj("r2", "record", outputs=[Part("b"), Part("g", [Part("c")])]),
            result_name="r1 & r2",
            result_kind="record",
        ),
        "held-field": single_figure(
            "A record whose field g holds a law over {mu, sigma}",
            Obj(
                "r",
                "record",
                outputs=[Part("x"), Part("g", [Part("mu"), Part("sigma")], label="g: law d")],
            ),
            notes=["paths: x, g, g/mu, g/sigma", "fields: x, g · leaf paths: x, g/mu, g/sigma"],
        ),
        "select-one": row_figure(
            "A law, its selection at one path, and its selection at a list of one path",
            [
                (
                    Obj("d", "law", outputs=[Part("a"), Part("g", [Part("b"), Part("c")])]),
                    ["a law over {a, g: {b, c}}"],
                ),
                (
                    Obj("d[g]", "law", outputs=[Part("b"), Part("c")]),
                    ["one path: the node g;", "a law over {b, c}"],
                ),
                (
                    Obj("d[[g]]", "law", outputs=[Part("g", [Part("b"), Part("c")])]),
                    ["a list of one path:", "a law over {g: {b, c}}"],
                ),
                (
                    Obj("d[g/b]", "law", root_output="b", body=["over an array"]),
                    ["a base node: one output at", "the root, named by its last name"],
                ),
            ],
            gap=50,
        ),
        "law-of-laws": row_figure(
            "A law over laws, unwrapped and wrapped",
            [
                (
                    Obj("d", "law", body=["over laws", "no outputs"]),
                    ["over laws: no outputs"],
                ),
                (
                    Obj("d", "law", outputs=[Part("params", label="params: a law")]),
                    ["over {params: law}:", "one output, params"],
                ),
            ],
        ),
        "function-outputs": row_figure(
            "A function whose return value is a record, and its selection at one output",
            [
                (
                    Obj("f", "function", outputs=[Part("y"), Part("z")], inputs=[Port("x")]),
                    ["f maps x to {y, z}:", "its outputs are y and z"],
                ),
                (
                    Obj(
                        "f[y]",
                        "function",
                        root_output="y",
                        inputs=[Port("x")],
                        body=["returns an array"],
                    ),
                    ["f[y] maps x to f(x)'s y;", "the same input"],
                ),
            ],
        ),
        "batch-columns": row_figure(
            "A batch of samples and its column at mu",
            [
                (
                    Obj("b", "batch", outputs=[Part("mu"), Part("sigma")], levels="sample"),
                    ["100 samples from a law", "over {mu, sigma}"],
                ),
                (
                    Obj("b[mu]", "batch", root_output="mu", body=["arrays"], levels="sample"),
                    ["its column at mu:", "one output at the root"],
                ),
            ],
        ),
        "factor-inputs": single_figure(
            "The law a & c, whose factor a still takes the optional input sigma",
            Obj(
                "a & c",
                "law",
                outputs=[Part("y"), Part("mu")],
                inputs=[Port("sigma", optional=True, factor=True, label="sigma")],
                body=["factors: a, c"],
            ),
            notes=["no inputs; one factor input, sigma (dashed),", "left at its default 1"],
        ),
        "merge-factor-input": merge_figure(
            "A later merge feeds the factor input sigma of the law a & c",
            Obj(
                "a & c",
                "law",
                outputs=[Part("y"), Part("mu")],
                inputs=[Port("sigma", optional=True, factor=True)],
            ),
            Obj("e", "law", root_output="sigma", body=["over an array"]),
            result_name="(a & c) & e",
            result_kind="law",
        ),
        "merge-invalid-output": merge_figure(
            "Refused: x is an output of both operands",
            Obj("a", "law", outputs=[Part("x"), Part("y")], faults={"x"}),
            Obj("b", "law", outputs=[Part("x"), Part("z")], faults={"x"}),
            result_name="a & b",
            result_kind="law",
            invalid=True,
            fault_results={"x"},
        ),
        "merge-invalid-order": merge_figure(
            "Refused: the right operand takes x, which the left operand outputs",
            Obj("a", "law", outputs=[Part("x")], faults={"x"}),
            Obj("b", "kernel", root_output="y", inputs=[Port("x")], faults={"x"}),
            result_name="a & b",
            result_kind="kernel",
            invalid=True,
            fault_results={"x"},
            fault_wires={"x"},
        ),
        "merge-invalid-conform": merge_figure(
            "Refused: the output x of shape (4,) does not conform to the input x of shape (3,)",
            Obj("a", "kernel", root_output="y", inputs=[Port("x", label="x: (3,)")], faults={"x"}),
            Obj("b", "law", root_output="x", body=["x: (4,)"], faults={"x"}),
            result_name="a & b",
            result_kind="law",
            invalid=True,
            fault_wires={"x"},
        ),
        "merge-invalid-unify": merge_figure(
            "Refused: the shared input theta has shapes (2,) and (3,), which do not unify",
            Obj(
                "a",
                "kernel",
                root_output="x",
                inputs=[Port("theta", label="theta: (2,)")],
                faults={"theta"},
            ),
            Obj(
                "b",
                "kernel",
                root_output="y",
                inputs=[Port("theta", label="theta: (3,)")],
                faults={"theta"},
            ),
            result_name="a & b",
            result_kind="kernel",
            invalid=True,
            fault_wires={"theta"},
            fault_results={"theta"},
        ),
        "merge-invalid-law": merge_figure(
            "Refused: the right operand produces a law, which a record cannot hold unnamed",
            Obj("a", "function", root_output="y", inputs=[Port("params")]),
            Obj(
                "b",
                "function",
                inputs=[Port("data")],
                body=["returns a law", "no outputs"],
                faulty=True,
            ),
            result_name="a & b",
            result_kind="function",
            invalid=True,
        ),
    }


def main() -> None:
    for name, canvas in figures().items():
        (FIGURE_DIR / f"{name}.svg").write_text(canvas.render(), encoding="utf-8")
        print(f"wrote plans/figures/{name}.svg")


if __name__ == "__main__":
    main()
