"""The expression a tracked term carries, and its rendering as a label and a notation.

Every tracked term carries one immutable **expression**: a tree that states
what the term is, for a reader. The term's label is the name part of its
rendering and the notation of a law, a kernel, or a function is the whole
rendering, so the label, the notation, ``str()``, and every derived label are
read from this one tree (design II.4). An operation builds its result's
expression from its operands' expressions, and no operation reads one.

The nodes are frozen dataclasses that hold strings, tuples of strings, and
child nodes, and no node holds a reference to a term:

1. :class:`Named`: a label and a signature, as ``prior(mu)``;
2. :class:`Product`: the factors of a product without a label, as
   ``lik(y | mu)·prior(mu)``;
3. :class:`Conditioned`: a law at given values of some paths, as
   ``model(mu; y)``;
4. :class:`Selected`: a law at some of its paths, as ``model(y)``;
5. :class:`Draw`: a draw from a law, as ``(y, mu) ~ model``;
6. :class:`Applied`: a function applied to draws and values, as
   ``f(beta ~ model; y)``;
7. :class:`Summary`: a summary of a law or a draw, as ``E[mu ~ prior]`` or
   ``log prior(mu)``;
8. :class:`Operator`: an operator applied to values, as ``2 * effect``;
9. :class:`Indexed`: a selection of a batch, as ``(mu ~ prior)[sample=0]``.

A law's own signature is read from its declaration when it is rendered, so a
term's expression records the signature of a law only where the law is a child
of another node, which :func:`embedded` records when an operation builds the
node. The paths a law holds fixed are read from the tree.

A rendering shows at most :attr:`~probpipe.core.config.NotationConfig.max_depth`
nested levels. A node deeper than that renders as its label, the name of a law
or a function, or as ``…`` for a value, and the rendering warns.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from ._repr import (
    PRODUCT_SYMBOL,
    format_notation,
    format_signature,
    format_value,
    grouped_label,
    is_product,
)

__all__ = [
    "Applied",
    "Conditioned",
    "Draw",
    "Expression",
    "Indexed",
    "Named",
    "Operator",
    "Product",
    "Selected",
    "Signature",
    "Summary",
    "constant",
    "core_of",
    "draw_of",
    "embedded",
    "expression_of",
    "fixed_paths_of",
    "label_of",
    "notation_of",
    "with_fixed",
]

#: The text a collapsed value renders as.
ELLIPSIS = "…"

#: The number of nested nodes a stored expression keeps. A node built over a
#: deeper child stores the child's collapsed form, so a long derivation, such
#: as a loop that adds to a value, stores a tree of bounded depth that copies
#: and pickles without deep recursion.
_STORED_DEPTH = 64


@dataclass(frozen=True, slots=True)
class Signature:
    """What a law, a kernel, or a function is over, as its declaration states it.

    Attributes
    ----------
    components : tuple of str
        A law's or a kernel's event components, or a function's parameters, in
        declaration order.
    given : tuple of str
        A kernel's given slots, in declaration order; empty for a law and a
        function.
    fixed : tuple of str
        The paths a law or a kernel holds fixed at given values, in the order
        they were fixed. Only a :class:`Named` node reads them from its
        signature; every other node reads them from the tree.
    """

    components: tuple[str, ...]
    given: tuple[str, ...] = ()
    fixed: tuple[str, ...] = ()


class Expression:
    """A node of a term's expression.

    Every node renders a label and a notation (:func:`label_of`,
    :func:`notation_of`), states the paths it holds fixed (:func:`fixed_paths_of`),
    and records its nesting depth. A subclass is a frozen dataclass whose
    fields are strings, tuples of strings, and child nodes.
    """

    __slots__ = ()

    def _children(self) -> tuple[Expression, ...]:
        """The child nodes, in rendering order."""
        return ()

    @property
    def depth(self) -> int:
        """The number of nodes on the longest path from this node to a leaf, this node included."""
        return 1 + max((child.depth for child in self._children()), default=0)


def _kept(child: Expression) -> Expression:
    """*child* as a node stores it: its collapsed form when it is deeper than a stored tree keeps."""
    if child.depth < _STORED_DEPTH:
        return child
    return Named(_collapsed_text(child))


def _kept_all(children: Iterable[Expression]) -> tuple[Expression, ...]:
    """Each of *children* as a node stores it."""
    return tuple(_kept(child) for child in children)


@dataclass(frozen=True, slots=True)
class Named(Expression):
    """A term under a label: a law, a kernel, or a function with its signature, or a value.

    Attributes
    ----------
    label : str
        The label, which is the node's label.
    signature : Signature or None
        The signature of a law, a kernel, or a function, which the notation
        follows the label with, and whose fixed paths the node holds. ``None``
        for a value, and for a term's own expression whose signature its
        declaration supplies.
    """

    label: str
    signature: Signature | None = None


@dataclass(frozen=True, slots=True)
class Product(Expression):
    """A product without a label, which displays factor by factor.

    Its label joins the factors' labels with ``·``, a factor whose label is a
    product joining as it is, and its notation joins the factors' notations.
    A factor that is itself a :class:`Product` enters as its factors, so
    products join associatively.

    Attributes
    ----------
    factors : tuple of Expression
        The factors' expressions, in the product's order.
    """

    factors: tuple[Expression, ...]

    def __post_init__(self) -> None:
        flat: list[Expression] = []
        for factor in self.factors:
            flat.extend(factor.factors if isinstance(factor, Product) else (factor,))
        object.__setattr__(self, "factors", _kept_all(flat))

    def _children(self) -> tuple[Expression, ...]:
        return self.factors


@dataclass(frozen=True, slots=True)
class Conditioned(Expression):
    """A law or a kernel at given values of some paths: a conditional, or a kernel at its givens.

    It keeps its base's label, and it holds its base's fixed paths followed by
    the paths of :attr:`fixed` its base does not hold. A conditioned base
    merges into one node, so conditioning again appends the new paths.

    Attributes
    ----------
    base : Expression
        The law or kernel conditioned.
    fixed : tuple of str
        The paths fixed at given values by this conditioning.
    signature : Signature or None
        The components and given slots of the result, recorded where the node
        is a child of another node; ``None`` for a term's own expression.
    """

    base: Expression
    fixed: tuple[str, ...]
    signature: Signature | None = None

    def __post_init__(self) -> None:
        base = self.base
        if isinstance(base, Conditioned):
            object.__setattr__(self, "fixed", _merged(base.fixed, self.fixed))
            base = base.base
        object.__setattr__(self, "base", _kept(base))

    def _children(self) -> tuple[Expression, ...]:
        return (self.base,)


@dataclass(frozen=True, slots=True)
class Selected(Expression):
    """A law at some of its paths: a marginal or a field view, or a law renamed from its base.

    It keeps its base's label and holds its base's fixed paths.

    Attributes
    ----------
    base : Expression
        The law selected from.
    paths : tuple of str
        The paths selected, in the order of the result's components.
    signature : Signature or None
        The components of the result, recorded where the node is a child of
        another node; ``None`` for a term's own expression.
    """

    base: Expression
    paths: tuple[str, ...]
    signature: Signature | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "base", _kept(self.base))

    def _children(self) -> tuple[Expression, ...]:
        return (self.base,)


@dataclass(frozen=True, slots=True)
class Draw(Expression):
    """A draw from a law, labeled ``components ~ label`` and then ``; fixed paths``.

    A draw from a law that :class:`Applied` describes is that law's notation,
    as ``f(beta ~ model; y)``, since the law is the law of the function at
    draws of its inputs.

    Attributes
    ----------
    components : tuple of str
        The components drawn, one written as it is and several in parentheses.
    law : Expression
        The law drawn from.
    """

    components: tuple[str, ...]
    law: Expression

    def __post_init__(self) -> None:
        object.__setattr__(self, "law", _kept(self.law))

    def _children(self) -> tuple[Expression, ...]:
        return (self.law,)


@dataclass(frozen=True, slots=True)
class Applied(Expression):
    """A function applied to its arguments: the law of a function lifted over laws.

    Its label is the function's label, and its notation is the call, as
    ``f(beta ~ model; y)``.

    Attributes
    ----------
    function : str
        The function's output label.
    arguments : tuple of Expression
        The arguments in order: a :class:`Draw` for each law drawn from, and
        a value's expression for any other argument.
    """

    function: str
    arguments: tuple[Expression, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _kept_all(self.arguments))

    def _children(self) -> tuple[Expression, ...]:
        return self.arguments


#: The summaries a :class:`Summary` node writes as ``kind[argument]``.
_BRACKETED_SUMMARIES = frozenset({"E", "Var", "Cov", "Q"})

#: The summary of a law's score, written ``log`` and the law's notation.
SCORE = "log"

#: The summary of a law's density, written as the law's notation.
DENSITY = "density"


@dataclass(frozen=True, slots=True)
class Summary(Expression):
    """A value that summarizes a law or a draw.

    The kinds and their renderings:

    1. ``E``, ``Var``, ``Cov``, and ``Q``: the expectation, the variance, the
       covariance, and the quantile of a draw, as ``E[(y, mu) ~ model]``;
    2. ``log``: a score, ``log`` followed by the law's notation, as
       ``log prior(mu)``;
    3. ``density``: a density, which reads as the law's notation, as
       ``prior(mu)``.

    Attributes
    ----------
    kind : str
        One of ``E``, ``Var``, ``Cov``, ``Q``, ``log``, and ``density``.
    argument : Expression
        A draw, or an applied function, for the bracketed kinds; a law for
        ``log`` and ``density``.
    """

    kind: str
    argument: Expression

    def __post_init__(self) -> None:
        if self.kind not in _BRACKETED_SUMMARIES | {SCORE, DENSITY}:
            raise ValueError(f"unknown summary kind {self.kind!r}")
        object.__setattr__(self, "argument", _kept(self.argument))

    def _children(self) -> tuple[Expression, ...]:
        return (self.argument,)


#: The unary operators written as a call of their operand, as ``abs(x)``.
_CALL_OPERATORS = frozenset({"abs"})


@dataclass(frozen=True, slots=True)
class Operator(Expression):
    """An operator applied to values, as ``2 * effect`` or ``-effect``.

    A binary operator writes its symbol between its two operands, a prefix
    operator before its one operand, and ``abs`` as a call of its operand.
    An operand that is an expression is parenthesized.

    Attributes
    ----------
    symbol : str
        The operator's symbol, such as ``+`` or ``-``, or ``abs``.
    operands : tuple of Expression
        One operand for a unary operator and two for a binary one.
    """

    symbol: str
    operands: tuple[Expression, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operands", _kept_all(self.operands))

    def _children(self) -> tuple[Expression, ...]:
        return self.operands


@dataclass(frozen=True, slots=True)
class Indexed(Expression):
    """A selection of a batch, labeled by the batch's grouped label and the selected levels.

    It holds its base's fixed paths, so an element of a batch of posteriors
    reads as ``model[dataset=0](mu; y)``.

    Attributes
    ----------
    base : Expression
        The batch's expression.
    index : str
        The selected levels, as ``sample=0`` or ``chain=0, draw=7``.
    signature : Signature or None
        The components of an element law, recorded where the node is a child
        of another node; ``None`` for a term's own expression.
    """

    base: Expression
    index: str
    signature: Signature | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "base", _kept(self.base))

    def _children(self) -> tuple[Expression, ...]:
        return (self.base,)


# Each node class caches nothing and holds only its fields, so a node compares
# and hashes by value.
_WRAPPERS = (Conditioned, Selected)


def _merged(held: tuple[str, ...], added: Iterable[str]) -> tuple[str, ...]:
    """*held* followed by the paths of *added* it does not hold, in order."""
    return held + tuple(path for path in dict.fromkeys(added) if path not in held)


# ---------------------------------------------------------------------------
# Reading the tree
# ---------------------------------------------------------------------------


def core_of(expression: Expression) -> Expression:
    """*expression* without the conditionings and selections around it."""
    while isinstance(expression, _WRAPPERS):
        expression = expression.base
    return expression


def fixed_paths_of(expression: Expression) -> tuple[str, ...]:
    """The paths the law or kernel *expression* describes holds fixed, in the order they were fixed.

    A :class:`Named` node holds those of its signature, a :class:`Conditioned`
    node its base's followed by its own, and a selection or an indexed batch
    its base's. Any other node holds none.
    """
    if isinstance(expression, Named):
        return () if expression.signature is None else expression.signature.fixed
    if isinstance(expression, Conditioned):
        return _merged(fixed_paths_of(expression.base), expression.fixed)
    if isinstance(expression, (Selected, Indexed)):
        return fixed_paths_of(expression.base)
    return ()


def with_fixed(expression: Expression, paths: Iterable[str]) -> Expression:
    """*expression* holding *paths* fixed after the paths it holds.

    *expression* is returned as it is when it holds every path already, and
    otherwise conditioned on the paths it does not hold.
    """
    held = fixed_paths_of(expression)
    added = tuple(path for path in dict.fromkeys(paths) if path not in held)
    return Conditioned(expression, added) if added else expression


def with_signature(expression: Expression, signature: Signature | None) -> Expression:
    """*expression* recording *signature*, the components and given slots of the term it describes.

    A :class:`Named` node keeps its fixed paths, and a conditioning, a
    selection, or an indexed batch records the components and given slots.
    Any other node, and a *signature* of ``None``, leaves *expression* as it is.
    """
    if signature is None:
        return expression
    if isinstance(expression, Named):
        return Named(
            expression.label,
            Signature(signature.components, signature.given, fixed_paths_of(expression)),
        )
    if isinstance(expression, (Conditioned, Selected, Indexed)):
        own = Signature(signature.components, signature.given)
        return expression if expression.signature == own else replace(expression, signature=own)
    return expression


def expression_of(term: Any) -> Expression:
    """The expression *term* carries; a tracked term restored without one carries its label."""
    expression = getattr(term, "_expression", None)
    if isinstance(expression, Expression):
        return expression
    return Named(term._label)


def own_signature(term: Any) -> Signature | None:
    """The signature *term*'s declaration states, or ``None`` for a term that has none."""
    signature = getattr(term, "_own_signature", None)
    return signature() if callable(signature) else None


def embedded(term: Any) -> Expression:
    """*term*'s expression as a child of another node, recording the signature its declaration states.

    A node holds no reference to a term, so a law's expression records its
    components and given slots where an operation makes it a child.
    """
    return with_signature(expression_of(term), own_signature(term))


def draw_of(term: Any, components: Iterable[str] | None = None) -> Draw:
    """A draw of *components* from the law *term*, by default every event component."""
    names = tuple(term.event_spec.components) if components is None else tuple(components)
    return Draw(names, embedded(term))


def constant(value: Any) -> Named:
    """The node of a value that is not a tracked term: its formatted value, as ``2.0``."""
    return Named(format_value(value))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _collapsed_text(expression: Expression) -> str:
    """The text a node renders as when it is deeper than a rendering shows: its label or ``…``.

    A law or a function renders as its name, which is the label of a named
    law, the factors' names of a product, and the function of an applied
    function. A value renders as ``…``.
    """
    core = core_of(expression)
    if isinstance(core, Named):
        return core.label
    if isinstance(core, Product):
        return _joined(_collapsed_text(factor) for factor in core.factors)
    if isinstance(core, Applied):
        return core.function
    return ELLIPSIS


def _joined(labels: Iterable[str]) -> str:
    """The labels of factors joined with ``·``, a product's joining as it is and any other grouped."""
    return PRODUCT_SYMBOL.join(
        label if is_product(label) else grouped_label(label) for label in labels
    )


def _components_text(components: tuple[str, ...]) -> str:
    """The components of a draw: one as it is, as ``mu``, and several in parentheses, as ``(y, mu)``."""
    return components[0] if len(components) == 1 else f"({', '.join(components)})"


class _Rendering:
    """One rendering of an expression, which counts the levels it nests and the nodes it collapses.

    Parameters
    ----------
    max_depth : int
        The number of nested levels the rendering shows.
    """

    def __init__(self, max_depth: int) -> None:
        self.max_depth = max_depth
        self.collapsed = False

    def _collapse(self, expression: Expression) -> str:
        """*expression*'s collapsed text, recording that the rendering collapsed a node."""
        self.collapsed = True
        return _collapsed_text(expression)

    def label(self, expression: Expression, level: int) -> str:
        """The label of *expression* at nesting *level*: a law's name, or a value's rendering."""
        match expression:
            case Named():
                return expression.label
            case Conditioned() | Selected():
                return self.label(expression.base, level)
            case Product():
                return _joined(self.label(factor, level + 1) for factor in expression.factors)
            case Applied():
                return expression.function
            case Draw() if isinstance(expression.law, Applied):
                # The law of a lifted function is the function at draws of its
                # inputs, so a draw from it is that call, at the draw's level.
                return self.notation(expression.law, level, None)
        if level > self.max_depth:
            return self._collapse(expression)
        match expression:
            case Draw():
                law = expression.law
                text = f"{_components_text(expression.components)} ~ {self.label(law, level + 1)}"
                fixed = fixed_paths_of(law)
                return f"{text}; {', '.join(fixed)}" if fixed else text
            case Summary():
                if expression.kind == SCORE:
                    law = self.notation(expression.argument, level + 1, None)
                    return f"{SCORE} {grouped_label(law)}"
                if expression.kind == DENSITY:
                    return self.notation(expression.argument, level + 1, None)
                return f"{expression.kind}[{self.label(expression.argument, level + 1)}]"
            case Operator():
                return self._operator(expression, level)
            case Indexed():
                return (
                    f"{grouped_label(self.label(expression.base, level + 1))}[{expression.index}]"
                )
        raise TypeError(f"cannot render {type(expression).__name__}")

    def _operator(self, expression: Operator, level: int) -> str:
        """The rendering of an operator over its operands, each grouped as an operand."""
        operands = [self.label(operand, level + 1) for operand in expression.operands]
        symbol = expression.symbol
        if len(operands) == 2:
            return f"{grouped_label(operands[0])} {symbol} {grouped_label(operands[1])}"
        if symbol in _CALL_OPERATORS:
            return f"{symbol}({operands[0]})"
        return f"{symbol}{grouped_label(operands[0])}"

    def notation(self, expression: Expression, level: int, own: Signature | None) -> str:
        """The notation of *expression* at nesting *level*, with *own* the signature its term declares.

        A law, a kernel, or a function renders as its label followed by its
        signature, a product without a label factor by factor, and an applied
        function as its call. A value renders as its label.
        """
        match expression:
            case Product():
                if level > self.max_depth:
                    return self._collapse(expression)
                return PRODUCT_SYMBOL.join(
                    self.notation(factor, level + 1, None) for factor in expression.factors
                )
            case Applied():
                if level > self.max_depth:
                    return self._collapse(expression)
                arguments = ", ".join(self.label(arg, level + 1) for arg in expression.arguments)
                return f"{expression.function}({arguments})"
            case Named() | Conditioned() | Selected() | Indexed():
                signature = own or expression.signature
                if signature is None:
                    return self.label(expression, level)
                text = format_signature(
                    signature.components, signature.given, fixed_paths_of(expression)
                )
                return format_notation(self.label(expression, level), text)
        return self.label(expression, level)


def _warn_if_collapsed(rendering: _Rendering) -> None:
    """Warn that *rendering* collapsed a node, naming the setting that shows more levels."""
    if rendering.collapsed:
        warnings.warn(
            f"a label or notation nests more than notation_config.max_depth="
            f"{rendering.max_depth} levels, so its deeper parts show as their labels or "
            f"{ELLIPSIS!r}; raise notation_config.max_depth to show them",
            UserWarning,
            stacklevel=3,
        )


def _max_depth() -> int:
    """The number of nested levels a rendering shows, as ``notation_config`` sets it."""
    from .config import notation_config

    return notation_config.max_depth


def label_of(expression: Expression) -> str:
    """The label *expression* renders: the name of a law, a kernel, or a function, and a value's rendering.

    A conditioning and a selection keep their base's label, a product without
    a label joins its factors' labels with ``·``, and an applied function takes
    the function's label. A value renders in full, as ``(y, mu) ~ model`` or
    ``E[mu ~ prior]``, with each part grouped by design II.4.

    Parameters
    ----------
    expression : Expression
        The expression to render.

    Returns
    -------
    str
        The label, which shows at most ``notation_config.max_depth`` nested
        levels.

    Warns
    -----
    UserWarning
        When the rendering nests more levels than ``notation_config.max_depth``.
    """
    rendering = _Rendering(_max_depth())
    text = rendering.label(expression, 1)
    _warn_if_collapsed(rendering)
    return text


def notation_of(expression: Expression, own: Signature | None = None) -> str:
    """The notation *expression* renders, with *own* the signature the term's declaration states.

    A law, a kernel, or a function reads as its grouped label followed by its
    signature, which lists the fixed paths after ``;``, as ``model(mu; y)``; a
    product without a label reads factor by factor, as ``lik(y | mu)·prior(mu)``;
    and a law of a function lifted over laws reads as the function's call, as
    ``f(beta ~ model; y)``.

    Parameters
    ----------
    expression : Expression
        The expression to render.
    own : Signature or None, optional
        The components and given slots the term's declaration states, which
        replace those the expression records at its root.

    Returns
    -------
    str
        The notation, which shows at most ``notation_config.max_depth``
        nested levels.

    Warns
    -----
    UserWarning
        When the rendering nests more levels than ``notation_config.max_depth``.
    """
    rendering = _Rendering(_max_depth())
    text = rendering.notation(expression, 1, own)
    _warn_if_collapsed(rendering)
    return text
