"""The expression a tracked term carries, and its rendering as a label and a notation.

Every tracked term carries one immutable **expression**: a tree that states
what the term is, for a reader. The term's label is the name part of the tree's
rendering, and the notation of a law, a kernel, or a function is the whole
rendering. The label, the notation, ``str()``, and every derived label are
therefore read from this one tree (design II.4). An operation builds its
result's expression from its operands' expressions, and no operation reads one.

The nodes are frozen dataclasses that hold strings, tuples of strings, and
child nodes. No node holds a reference to a term. The node classes are these:

1. :class:`Named`: a label and a signature, as ``prior(mu)``;
2. :class:`Product`: the factors of a product without a label, as
   ``lik(y | mu)·prior(mu)``;
3. :class:`Conditioned`: a law at given values of some paths, as
   ``model(mu; y)``;
4. :class:`Selected`: a law at some of its paths, as ``model(y)``;
5. :class:`Draw`: a draw from a law, as ``(y, mu) ~ model``;
6. :class:`Applied`: a function applied to draws and values, as
   ``f(beta ~ model; y)``;
7. :class:`AppliedValue`: the value a function's call returns, as
   ``f(beta, 2.0)``;
8. :class:`Summary`: a summary of a law or a draw, as ``𝔼[mu ~ prior]`` or
   ``log prior(mu)``;
9. :class:`Operator`: an operator applied to values, as ``2 * effect``;
10. :class:`Indexed`: a selection of a batch, as ``(mu ~ prior)[sample=0]``;
11. :class:`Collection`: the members of a batch of laws or functions, as
    ``[prior(mu), lik(y | mu)]``;
12. :class:`Truncated`: a part that storage truncated, described below.

Each node renders itself: it gives its label, its notation, and the text it
collapses to, and it states the paths it holds fixed. A term reads the
signature of its own expression from its declaration when it renders, so a
stored expression never disagrees with its term. A node that has a law as a
child records the law's signature, because the node holds no term.

A rendering shows at most :attr:`~probpipe.core.config.NotationConfig.max_depth`
nested levels. A node deeper than that renders as its collapsed text, which is
the name of a law or a function, or ``…`` for a value. A stored expression keeps
at most :data:`_STORED_DEPTH` levels, and a part nested deeper is stored as a
:class:`Truncated` node, which renders as the part's collapsed text and holds
the paths the part holds fixed. A term's label is rendered once, when the term
is built, and its notation each time it is shown. A display of a term warns
when what it shows collapsed a level, or shows a truncated part whose text is
less than the part rendered in full. Storing a label renders it silently.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from ._repr import (
    ELLIPSIS,
    PRODUCT_SYMBOL,
    format_components,
    format_notation,
    format_signature,
    format_value,
    grouped_label,
    is_product,
)

__all__ = [
    "DENSITY",
    "ELLIPSIS",
    "SCORE",
    "Applied",
    "AppliedValue",
    "Collapse",
    "Collection",
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
    "Truncated",
    "constant",
    "draw_of",
    "joined_labels",
    "warn_collapsed",
]

#: The package directory, whose frames a collapse warning skips, so the warning
#: names the user's line that displays the term.
_WARNING_SKIP_PREFIXES = (os.path.dirname(os.path.dirname(__file__)) + os.sep,)

#: The greatest depth of a stored expression. A node built over a child that
#: nests this many levels stores a :class:`Truncated` node in its place, so a
#: long derivation, such as a loop that adds to a value, never builds a tree
#: too deep to copy, hash, or pickle. ``notation_config.max_depth``, which bounds
#: what a rendering shows, is at most this bound, since a rendering of more
#: levels would show no more.
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
        A kernel's given slots, in declaration order. A law and a function have
        none.
    defaults : tuple of (str, str)
        Each given slot or parameter that has a default, paired with the
        default as :func:`~probpipe.core._repr.format_default` formats it, in
        declaration order. The signature writes such a name as ``name=value``.
    """

    components: tuple[str, ...]
    given: tuple[str, ...] = ()
    defaults: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Collapse:
    """What a rendering left out, which a display of it warns about.

    At least one of :attr:`beyond_depth` and :attr:`truncated` is true.

    Attributes
    ----------
    max_depth : int
        The ``notation_config.max_depth`` the rendering showed.
    beyond_depth : bool
        Whether the rendering collapsed a part nested deeper than *max_depth*,
        which a greater setting shows.
    truncated : bool
        Whether the rendering showed a part that storage truncated, or
        collapsed a part that holds one, where the truncated text shows less
        than the part rendered in full. No setting shows what such a part
        left out.
    """

    max_depth: int
    beyond_depth: bool
    truncated: bool


class _Rendering:
    """The state of one rendering: the levels it shows and what it left out.

    Parameters
    ----------
    max_depth : int
        The number of nested levels the rendering shows.
    """

    def __init__(self, max_depth: int) -> None:
        self.max_depth = max_depth
        self.collapsed = False
        self.truncated = False

    def beyond(self, level: int) -> bool:
        """Whether a node at nesting *level* is deeper than the rendering shows."""
        return level > self.max_depth

    def collapse(self, node: Expression) -> str:
        """The collapsed text of *node*, recorded so that a rendering for display warns.

        A collapsed node that holds a part storage truncated with a loss is
        recorded too, since a greater setting would not show that part.
        """
        self.collapsed = True
        if not self.truncated:
            self.truncated = _holds_lost_part(node)
        return node._collapsed()

    def result(self) -> Collapse | None:
        """What the rendering left out, or ``None`` when it shows every part in full."""
        if not (self.collapsed or self.truncated):
            return None
        return Collapse(self.max_depth, self.collapsed, self.truncated)


class Expression:
    """A node of a term's expression.

    Every node renders a label and a notation (:meth:`render_label`,
    :meth:`render_notation`), states the paths it holds fixed (:meth:`fixed_paths`),
    and stores its nesting depth in its field ``depth``. A subclass is a frozen
    dataclass whose fields are strings, tuples of strings, and child nodes, and
    its ``__post_init__`` sets ``depth`` once the children are final.
    """

    __slots__ = ()

    def _children(self) -> tuple[Expression, ...]:
        """The child nodes, in rendering order."""
        return ()

    def _set_depth(self) -> None:
        """Store the depth of this node, computed from its children's stored depths.

        The depth is the number of nodes on the longest path from this node to
        a leaf, this node included.
        """
        depth = 1 + max((child.depth for child in self._children()), default=0)
        object.__setattr__(self, "depth", depth)

    # -- reading the tree ----------------------------------------------------

    def fixed_paths(self) -> tuple[str, ...]:
        """The paths the law or kernel this node describes holds fixed, in the order fixed.

        A :class:`Conditioned` node holds its base's followed by its own, a
        selection and an indexed batch hold their base's, and any other node
        holds none.
        """
        return ()

    def core(self) -> Expression:
        """This node without the conditionings and selections around it."""
        return self

    def with_fixed(self, paths: Iterable[str]) -> Expression:
        """This node holding *paths* fixed after the paths it holds.

        The node is returned as it is when it holds every path already, and
        otherwise conditioned on the paths it does not hold.
        """
        held = self.fixed_paths()
        added = tuple(path for path in dict.fromkeys(paths) if path not in held)
        return Conditioned(self, added) if added else self

    def signed(self, signature: Signature) -> Expression:
        """This node recording *signature* as the signature of the term it describes.

        A label, a conditioning, a selection, and an indexed batch record it,
        and any other node is returned as it is.
        """
        return self

    def _defaulted_givens(self) -> tuple[tuple[str, str], ...]:
        """The given slots with a default that this node's law or kernel leaves free.

        Each slot is paired with its formatted default. A kernel's slot that
        has a default stays a given of every law and kernel that conditioning
        makes from the kernel until a conditioning fixes it, so the kernel
        ``counts(y | K, n0=50.0)`` at ``K`` is ``counts(y | n0=50.0; K)``.
        """
        return ()

    def full_signature(self, signature: Signature) -> Signature:
        """*signature* followed by the defaulted given slots this node leaves free.

        A slot that *signature* names already is left as it is.
        """
        free = [
            (name, text)
            for name, text in self._defaulted_givens()
            if name not in signature.given and name not in signature.components
        ]
        if not free:
            return signature
        return Signature(
            signature.components,
            signature.given + tuple(name for name, _ in free),
            signature.defaults + tuple(free),
        )

    # -- rendering -----------------------------------------------------------

    def _collapsed(self) -> str:
        """The text this node renders as when it is deeper than a rendering shows.

        A law or a function renders as its name, and a value as ``…``.
        """
        return ELLIPSIS

    def _label(self, rendering: _Rendering, level: int) -> str:
        """The label of this node at nesting *level*: a law's name, or a value's rendering."""
        raise TypeError(f"cannot render {type(self).__name__}")

    def _notation(self, rendering: _Rendering, level: int, own: Signature | None) -> str:
        """The notation of this node at nesting *level*.

        *own* is the signature that the node's term declares, or ``None``. A
        value renders as its label. A law whose expression is a value's renders
        as that value's label followed by *own*. A sum ``f + g`` of Gaussian
        random functions is such a law: its expression is an
        :class:`Operator`, and it displays as ``(f + g)(f)``.
        """
        label = self._label(rendering, level)
        if own is None:
            return label
        return format_notation(label, self._signature_text(own))

    def _call(self, rendering: _Rendering, level: int) -> str | None:
        """The call that the law of a lifted function renders as.

        Any other node gives ``None``.
        """
        return None

    def _signature_text(self, signature: Signature) -> str:
        """The text of *signature* for this node.

        It lists the node's fixed paths and the defaulted slots the node leaves free.
        """
        signature = self.full_signature(signature)
        return format_signature(
            signature.components,
            signature.given,
            self.fixed_paths(),
            dict(signature.defaults),
        )

    def render_label(self) -> str:
        """The label this node renders, at the current ``notation_config.max_depth``.

        The label of a law, a kernel, or a function is its name, and a value's
        label is its rendering.

        A conditioning and a selection keep their base's label, a product without
        a label joins its factors' labels with ``·``, and an applied function takes
        the function's label. A value renders in full, as ``(y, mu) ~ model`` or
        ``𝔼[mu ~ prior]``, with each part grouped by design II.4.

        Returns
        -------
        str
            The label, which shows at most ``notation_config.max_depth`` nested
            levels.
        """
        return self.label_rendering()[0]

    def label_rendering(self) -> tuple[str, Collapse | None]:
        """The label this node renders, as :meth:`render_label` gives it, and what it left out.

        A term stores both when it is built, and a display of its label warns
        with :func:`warn_collapsed` when the label left a part out.

        Returns
        -------
        tuple of (str, Collapse or None)
            The label, and what its rendering left out, or ``None`` when it shows
            every part in full.
        """
        rendering = _Rendering(_max_depth())
        text = self._label(rendering, 1)
        return text, rendering.result()

    def render_notation(self, own: Signature | None = None, *, warn: bool = False) -> str:
        """The notation this node renders.

        *own* is the signature that the term's declaration states.

        A law, a kernel, or a function reads as its grouped label followed by its
        signature, which lists the fixed paths after ``;``, as ``model(mu; y)``. A
        product without a label reads factor by factor, as ``lik(y | mu)·prior(mu)``,
        and the law of a function lifted over laws reads as the function's call,
        as ``f(beta ~ model; y)``.

        Parameters
        ----------
        own : Signature or None, optional
            The components and given slots that the term's declaration states,
            which replace those the node records.
        warn : bool, optional
            Whether the rendering warns when it collapses a node, as the
            ``notation`` of a term does.

        Returns
        -------
        str
            The notation, which shows at most ``notation_config.max_depth``
            nested levels.

        Warns
        -----
        UserWarning
            When *warn* is true and the rendering collapses a part nested deeper
            than ``notation_config.max_depth``, or shows less of a part that
            storage truncated than the part rendered in full.
        """
        rendering = _Rendering(_max_depth())
        text = self._notation(rendering, 1, own)
        collapse = rendering.result()
        if warn and collapse is not None:
            warn_collapsed(collapse, stored=False)
        return text


def _kept(child: Expression) -> Expression:
    """*child* as a node stores it.

    A child that nests :data:`_STORED_DEPTH` levels is stored as a
    :class:`Truncated` node, which keeps its collapsed text, its signature, the
    paths it holds fixed, and the defaulted given slots it leaves free, and
    records where its text shows less than *child* renders in full.
    """
    if child.depth < _STORED_DEPTH:
        return child
    signature = child.signature if isinstance(child, _Signed) else None
    shown = Truncated(child._collapsed(), child.fixed_paths(), child._defaulted_givens(), signature)
    return replace(
        shown,
        label_lost=_shows_less(child._label, shown._label),
        notation_lost=_shows_less(
            lambda rendering, level: child._notation(rendering, level, None),
            lambda rendering, level: shown._notation(rendering, level, None),
        ),
        call_lost=_shows_less(child._call, shown._call),
    )


def _holds_lost_part(node: Expression) -> bool:
    """Whether *node* is or holds a :class:`Truncated` node that shows less than its part."""
    pending = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, Truncated):
            if current.label_lost or current.notation_lost or current.call_lost:
                return True
        else:
            pending.extend(current._children())
    return False


def _shows_less(full: Any, shown: Any) -> bool:
    """Whether the rendering *shown* gives less than the rendering *full* of a part.

    Each argument renders the part at a nesting level, as ``_label`` does. *full*
    renders every stored level, so it leaves a part out only where it meets a
    part that storage truncated with a loss.
    """
    rendering = _Rendering(_STORED_DEPTH)
    text = full(rendering, 1)
    return rendering.collapsed or rendering.truncated or text != shown(_Rendering(_STORED_DEPTH), 1)


def _kept_all(children: Iterable[Expression]) -> tuple[Expression, ...]:
    """Each of *children* as a node stores it."""
    return tuple(_kept(child) for child in children)


def _depth_field() -> Any:
    """The dataclass field that stores a node's depth.

    A node computes its depth once, when it is built, from its children's
    stored depths. The field takes no part in equality, hashing, or the repr,
    so two nodes compare by their content alone.
    """
    return field(init=False, compare=False, repr=False)


class _Signed(Expression):
    """A node that may record the signature of the law or kernel it describes.

    It renders its notation as its label followed by the signature its term
    declares, or by the one it records where it is a child of another node.
    """

    __slots__ = ()

    #: The recorded signature, which each subclass stores in its field.
    signature: Signature | None

    def signed(self, signature: Signature) -> Expression:
        if self.signature == signature:
            return self
        return replace(self, signature=signature)  # type: ignore[type-var]

    def _notation(self, rendering: _Rendering, level: int, own: Signature | None) -> str:
        signature = own or self.signature
        if signature is None:
            return self._label(rendering, level)
        return format_notation(self._label(rendering, level), self._signature_text(signature))


@dataclass(frozen=True, slots=True)
class Named(_Signed):
    """A term under a label: a law, a kernel, or a function with its signature, or a value.

    Attributes
    ----------
    label : str
        The label the node renders, as ``prior`` or ``2.0``.
    signature : Signature or None
        The signature of a law, a kernel, or a function, which the notation
        follows the label with. It is ``None`` for a value and for the
        expression a term is constructed with, whose declaration supplies the
        signature. The expression that ``with_label`` gives a law or a kernel
        records the signature, so its notation keeps the defaulted given slots
        the law leaves free.
    """

    label: str
    signature: Signature | None = None
    depth: int = _depth_field()

    def __post_init__(self) -> None:
        self._set_depth()

    def _defaulted_givens(self) -> tuple[tuple[str, str], ...]:
        signature = self.signature
        if signature is None:
            return ()
        return tuple((name, text) for name, text in signature.defaults if name in signature.given)

    def _collapsed(self) -> str:
        return self.label

    def _label(self, rendering: _Rendering, level: int) -> str:
        return self.label


@dataclass(frozen=True, slots=True)
class Truncated(_Signed):
    """A part that storage truncated, which renders as the part's collapsed text.

    A node built over a child that nests :data:`_STORED_DEPTH` levels stores
    this node in the child's place. Its label is the name of a law or a
    function, or ``…`` for a value, and its notation is that label followed by
    the part's signature, where the part records one. It holds the paths the
    part holds fixed, so a law selected from a truncated posterior still lists
    them after ``;``.

    A rendering that shows this node records that it left a part out, so a
    display of it warns, where the node's text shows less than the part
    rendered in full. A law selected from a law many times keeps its name, so
    its truncated base shows all it would and a display of it does not warn.

    Attributes
    ----------
    text : str
        The collapsed text of the part.
    fixed : tuple of str
        The paths the part holds fixed, in the order fixed.
    givens : tuple of (str, str)
        The given slots with a default that the part leaves free, each with its
        formatted default.
    signature : Signature or None
        The signature the part records, or ``None``.
    label_lost : bool
        Whether :attr:`text` shows less than the part's label.
    notation_lost : bool
        Whether the node's notation shows less than the part's notation.
    call_lost : bool
        Whether the part renders as a call, which this node does not show.
    """

    text: str
    fixed: tuple[str, ...] = ()
    givens: tuple[tuple[str, str], ...] = ()
    signature: Signature | None = None
    label_lost: bool = True
    notation_lost: bool = True
    call_lost: bool = False
    depth: int = _depth_field()

    def __post_init__(self) -> None:
        self._set_depth()

    def fixed_paths(self) -> tuple[str, ...]:
        return self.fixed

    def _defaulted_givens(self) -> tuple[tuple[str, str], ...]:
        return self.givens

    def _collapsed(self) -> str:
        return self.text

    def _label(self, rendering: _Rendering, level: int) -> str:
        if self.label_lost:
            rendering.truncated = True
        return self.text

    def _notation(self, rendering: _Rendering, level: int, own: Signature | None) -> str:
        if self.notation_lost:
            rendering.truncated = True
        signature = own or self.signature
        if signature is None:
            return self.text
        return format_notation(self.text, self._signature_text(signature))

    def _call(self, rendering: _Rendering, level: int) -> str | None:
        if self.call_lost:
            rendering.truncated = True
        return None


@dataclass(frozen=True, slots=True)
class Product(Expression):
    """A product without a label, which displays factor by factor.

    Its label joins the factors' labels with ``·``. A factor's label that is
    already a product joins without parentheses, and any other compound label
    is parenthesized. Its notation joins the factors' notations. A factor that
    is itself a :class:`Product` enters as its factors, so products join
    associatively.

    Attributes
    ----------
    factors : tuple of Expression
        The factors' expressions, in the product's order.
    """

    factors: tuple[Expression, ...]
    depth: int = _depth_field()

    def __post_init__(self) -> None:
        flat: list[Expression] = []
        for factor in self.factors:
            flat.extend(factor.factors if isinstance(factor, Product) else (factor,))
        object.__setattr__(self, "factors", _kept_all(flat))
        self._set_depth()

    def _children(self) -> tuple[Expression, ...]:
        return self.factors

    def _collapsed(self) -> str:
        return joined_labels(factor._collapsed() for factor in self.factors)

    def _label(self, rendering: _Rendering, level: int) -> str:
        return joined_labels(factor._label(rendering, level + 1) for factor in self.factors)

    def _notation(self, rendering: _Rendering, level: int, own: Signature | None) -> str:
        if rendering.beyond(level):
            return rendering.collapse(self)
        return PRODUCT_SYMBOL.join(
            factor._notation(rendering, level + 1, None) for factor in self.factors
        )


@dataclass(frozen=True, slots=True)
class Collection(Expression):
    """The members of a batch of laws or functions, which displays as a bracketed list.

    Its label lists each element's notation in brackets, separated by commas,
    as ``[prior(mu), lik(y | mu)]``, and ends the list with ``…`` when the batch
    has more members than the node holds. It collapses to ``[…]``.

    Attributes
    ----------
    elements : tuple of Expression
        The expressions of the members the node shows, in the batch's order.
    omitted : bool
        Whether the batch has members after those of :attr:`elements`.
    """

    elements: tuple[Expression, ...]
    omitted: bool = False
    depth: int = _depth_field()

    def __post_init__(self) -> None:
        object.__setattr__(self, "elements", _kept_all(self.elements))
        self._set_depth()

    def _children(self) -> tuple[Expression, ...]:
        return self.elements

    def _collapsed(self) -> str:
        return f"[{ELLIPSIS}]"

    def _label(self, rendering: _Rendering, level: int) -> str:
        if rendering.beyond(level):
            return rendering.collapse(self)
        parts = [child._notation(rendering, level + 1, None) for child in self.elements]
        if self.omitted:
            parts.append(ELLIPSIS)
        return "[" + ", ".join(parts) + "]"


@dataclass(frozen=True, slots=True)
class Conditioned(_Signed):
    """A law or a kernel at given values of some paths.

    It is a conditional law, or a kernel at some of its given slots.

    It keeps its base's label. It holds its base's fixed paths followed by the
    paths of :attr:`fixed` that its base does not hold. A conditioned base
    merges into this node, so conditioning again appends the new paths.

    Attributes
    ----------
    base : Expression
        The law or kernel conditioned.
    fixed : tuple of str
        The paths that this conditioning fixes at given values.
    signature : Signature or None
        The components and given slots of the result, recorded where the node
        is a child of another node. It is ``None`` for a term's own expression.
    """

    base: Expression
    fixed: tuple[str, ...]
    signature: Signature | None = None
    depth: int = _depth_field()

    def __post_init__(self) -> None:
        base = self.base
        if isinstance(base, Conditioned):
            object.__setattr__(self, "fixed", _merged(base.fixed, self.fixed))
            base = base.base
        object.__setattr__(self, "base", _kept(base))
        self._set_depth()

    def _children(self) -> tuple[Expression, ...]:
        return (self.base,)

    def fixed_paths(self) -> tuple[str, ...]:
        return _merged(self.base.fixed_paths(), self.fixed)

    def core(self) -> Expression:
        return self.base.core()

    def _defaulted_givens(self) -> tuple[tuple[str, str], ...]:
        fixed = self.fixed_paths()
        return tuple(
            (name, text) for name, text in self.base._defaulted_givens() if name not in fixed
        )

    def _collapsed(self) -> str:
        return self.base._collapsed()

    def _label(self, rendering: _Rendering, level: int) -> str:
        return self.base._label(rendering, level)


@dataclass(frozen=True, slots=True)
class Selected(_Signed):
    """A law at some of its paths: a marginal, a field view, or a law renamed from its base.

    It keeps its base's label and holds its base's fixed paths.

    Attributes
    ----------
    base : Expression
        The law selected from.
    paths : tuple of str
        The paths selected, in the order of the result's components.
    signature : Signature or None
        The components of the result, recorded where the node is a child of
        another node. It is ``None`` for a term's own expression.
    """

    base: Expression
    paths: tuple[str, ...]
    signature: Signature | None = None
    depth: int = _depth_field()

    def __post_init__(self) -> None:
        object.__setattr__(self, "base", _kept(self.base))
        self._set_depth()

    def _children(self) -> tuple[Expression, ...]:
        return (self.base,)

    def fixed_paths(self) -> tuple[str, ...]:
        return self.base.fixed_paths()

    def core(self) -> Expression:
        return self.base.core()

    def _defaulted_givens(self) -> tuple[tuple[str, str], ...]:
        return self.base._defaulted_givens()

    def _collapsed(self) -> str:
        return self.base._collapsed()

    def _label(self, rendering: _Rendering, level: int) -> str:
        return self.base._label(rendering, level)


@dataclass(frozen=True, slots=True)
class Draw(Expression):
    """A draw from a law.

    Its label is the drawn components, ``~``, and the law's label, as
    ``(y, mu) ~ model``. When the law holds paths fixed, ``;`` and those paths
    follow, as ``mu ~ model; y``. A draw from a law that :class:`Applied`
    describes is that law's call, as ``f(beta ~ model; y)``, since that law is
    the law of the function at draws of its inputs.

    Attributes
    ----------
    components : tuple of str
        The components drawn. One is written as it is, and several are
        parenthesized.
    law : Expression
        The law drawn from.
    """

    components: tuple[str, ...]
    law: Expression
    depth: int = _depth_field()

    def __post_init__(self) -> None:
        object.__setattr__(self, "law", _kept(self.law))
        self._set_depth()

    def _children(self) -> tuple[Expression, ...]:
        return (self.law,)

    def _label(self, rendering: _Rendering, level: int) -> str:
        call = self.law._call(rendering, level)
        if call is not None:
            return call
        if rendering.beyond(level):
            return rendering.collapse(self)
        law = self.law
        text = f"{format_components(self.components)} ~ {law._label(rendering, level + 1)}"
        fixed = law.fixed_paths()
        return f"{text}; {', '.join(fixed)}" if fixed else text


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
    depth: int = _depth_field()

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _kept_all(self.arguments))
        self._set_depth()

    def _children(self) -> tuple[Expression, ...]:
        return self.arguments

    def _collapsed(self) -> str:
        return self.function

    def _label(self, rendering: _Rendering, level: int) -> str:
        return self.function

    def _notation(self, rendering: _Rendering, level: int, own: Signature | None) -> str:
        if rendering.beyond(level):
            return rendering.collapse(self)
        arguments = ", ".join(
            self._argument(argument, rendering, level + 1) for argument in self.arguments
        )
        return f"{self.function}({arguments})"

    def _call(self, rendering: _Rendering, level: int) -> str | None:
        return self._notation(rendering, level, None)

    @staticmethod
    def _argument(argument: Expression, rendering: _Rendering, level: int) -> str:
        """*argument* as the call shows it.

        A law or a kernel shows its notation, and a value shows its label.
        """
        if isinstance(argument, _Signed) and argument.signature is not None:
            return argument._notation(rendering, level, None)
        return argument._label(rendering, level)


@dataclass(frozen=True, slots=True)
class AppliedValue(Applied):
    """The value a function's call returns, labeled by the call.

    Its label and its notation are both the call, as ``f(beta, 2.0)``, where an
    :class:`Applied` node's label is the function's label alone. It collapses
    to the function's label.

    Attributes
    ----------
    function : str
        The function's label.
    arguments : tuple of Expression
        The arguments' expressions, in parameter order. A law or a function
        shows its notation in the call, and a value shows its label.
    """

    def _label(self, rendering: _Rendering, level: int) -> str:
        return self._notation(rendering, level, None)


#: Automatic law and anonymous-function descriptions; explicit labels stay literal.
_GENERIC_LAW_SYMBOL = "ℙ"
_ANONYMOUS_FUNCTION_SYMBOL = "𝒻"

#: Internal summary kinds keep their spelling; only their presentation uses Unicode.
_SUMMARY_SYMBOLS = {"E": "𝔼", "Var": "𝕍", "Cov": "ℂ", "Q": "ℚ"}

#: The summaries a :class:`Summary` node writes as ``kind[argument]``.
_BRACKETED_SUMMARIES = frozenset(_SUMMARY_SYMBOLS)

#: The summary of a law's score, written ``log`` and the law's notation.
SCORE = "log"

#: The summary of a law's density, written as the law's notation.
DENSITY = "density"


@dataclass(frozen=True, slots=True)
class Summary(Expression):
    """A value that summarizes a law or a draw.

    Internal summary kinds keep their ASCII spelling. Their displayed symbols
    are ``𝔼``, ``𝕍``, ``ℂ``, and ``ℚ`` for ``E``, ``Var``, ``Cov``, and ``Q``.
    Explicit labels are rendered literally. The kinds and their renderings:

    1. ``E``, ``Var``, ``Cov``, and ``Q``: the expectation, the variance, the
       covariance, and the quantile of a draw, as ``𝔼[(y, mu) ~ model]``;
    2. ``log``: a score, ``log`` followed by the law's notation, as
       ``log prior(mu)``;
    3. ``density``: a density, which reads as the law's notation, as
       ``prior(mu)``.

    Attributes
    ----------
    kind : str
        One of ``E``, ``Var``, ``Cov``, ``Q``, ``log``, and ``density``.
    argument : Expression
        A draw or an applied function for the bracketed kinds, and a law for
        ``log`` and ``density``.
    """

    kind: str
    argument: Expression
    depth: int = _depth_field()

    def __post_init__(self) -> None:
        if self.kind not in _BRACKETED_SUMMARIES | {SCORE, DENSITY}:
            raise ValueError(f"unknown summary kind {self.kind!r}")
        object.__setattr__(self, "argument", _kept(self.argument))
        self._set_depth()

    def _children(self) -> tuple[Expression, ...]:
        return (self.argument,)

    def _label(self, rendering: _Rendering, level: int) -> str:
        if rendering.beyond(level):
            return rendering.collapse(self)
        argument = self.argument
        if self.kind == SCORE:
            return f"{SCORE} {grouped_label(argument._notation(rendering, level + 1, None))}"
        if self.kind == DENSITY:
            return argument._notation(rendering, level + 1, None)
        if isinstance(argument, Applied):
            # The expectation of a function at draws is the expectation of the call.
            return (
                f"{_SUMMARY_SYMBOLS[self.kind]}[{argument._notation(rendering, level + 1, None)}]"
            )
        return f"{_SUMMARY_SYMBOLS[self.kind]}[{argument._label(rendering, level + 1)}]"


#: The unary operators written as a call of their operand, as ``abs(x)``.
_CALL_OPERATORS = frozenset({"abs"})


@dataclass(frozen=True, slots=True)
class Operator(Expression):
    """An operator applied to values, as ``2 * effect`` or ``-effect``.

    A binary operator writes its symbol between its two operands, a prefix
    operator writes it before its one operand, and ``abs`` reads as a call of
    its operand. An operand whose label is compound is parenthesized.

    Attributes
    ----------
    symbol : str
        The operator's symbol, such as ``+`` or ``-``, or ``abs``.
    operands : tuple of Expression
        One operand for a unary operator and two for a binary one.
    """

    symbol: str
    operands: tuple[Expression, ...]
    depth: int = _depth_field()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operands", _kept_all(self.operands))
        self._set_depth()

    def _children(self) -> tuple[Expression, ...]:
        return self.operands

    def _label(self, rendering: _Rendering, level: int) -> str:
        if rendering.beyond(level):
            return rendering.collapse(self)
        operands = [operand._label(rendering, level + 1) for operand in self.operands]
        if len(operands) == 2:
            return f"{grouped_label(operands[0])} {self.symbol} {grouped_label(operands[1])}"
        if self.symbol in _CALL_OPERATORS:
            return f"{self.symbol}({operands[0]})"
        return f"{self.symbol}{grouped_label(operands[0])}"


@dataclass(frozen=True, slots=True)
class Indexed(_Signed):
    """A selection of a batch, labeled by the batch's grouped label and the selected levels.

    It holds its base's fixed paths, so an element of a batch of posteriors
    reads as ``model[dataset=0](mu; y)``. A selection of a batch of laws that
    a function lifted over laws gives keeps the call in its notation, as
    ``effect_of(mu ~ prior, tau)[tau=1:3]``. An element of that batch reads as
    its own row's call, :attr:`element`, as ``effect_of(mu ~ prior, 4.0)``.

    Attributes
    ----------
    base : Expression
        The batch's expression.
    index : str
        The selected levels, as ``sample=0`` or ``chain=0, draw=7``.
    signature : Signature or None
        The components of an element law, recorded where the node is a child
        of another node. It is ``None`` for a term's own expression.
    element : Expression or None
        The call of the row that an element of a lifted batch holds, which the
        element's notation and its draws read. It is ``None`` for any other
        selection.
    """

    base: Expression
    index: str
    signature: Signature | None = None
    element: Expression | None = None
    depth: int = _depth_field()

    def __post_init__(self) -> None:
        object.__setattr__(self, "base", _kept(self.base))
        if self.element is not None:
            object.__setattr__(self, "element", _kept(self.element))
        self._set_depth()

    def _children(self) -> tuple[Expression, ...]:
        return (self.base,) if self.element is None else (self.base, self.element)

    def fixed_paths(self) -> tuple[str, ...]:
        return self.base.fixed_paths()

    def _defaulted_givens(self) -> tuple[tuple[str, str], ...]:
        return self.base._defaulted_givens()

    def _label(self, rendering: _Rendering, level: int) -> str:
        if rendering.beyond(level):
            return rendering.collapse(self)
        return f"{grouped_label(self.base._label(rendering, level + 1))}[{self.index}]"

    def _notation(self, rendering: _Rendering, level: int, own: Signature | None) -> str:
        call = self._call(rendering, level)
        if call is not None:
            return call
        return _Signed._notation(self, rendering, level, own)

    def _call(self, rendering: _Rendering, level: int) -> str | None:
        """The row's call for an element of a lifted batch, or the batch's call and the levels.

        Any other selection gives ``None``.
        """
        if self.element is not None:
            return self.element._call(rendering, level)
        call = self.base._call(rendering, level + 1)
        if call is not None:
            return f"{grouped_label(call)}[{self.index}]"
        return None


def _merged(held: tuple[str, ...], added: Iterable[str]) -> tuple[str, ...]:
    """*held* followed by the paths of *added* that it does not hold, in order."""
    return held + tuple(path for path in dict.fromkeys(added) if path not in held)


# ---------------------------------------------------------------------------
# Building nodes
# ---------------------------------------------------------------------------


def draw_of(term: Any, components: Iterable[str] | None = None) -> Draw:
    """A draw of *components* from the law *term*, by default every event component."""
    names = tuple(term.event_spec.components) if components is None else tuple(components)
    return Draw(names, term._embedded_expression())


def constant(value: Any) -> Named:
    """The node of a value that is not a tracked term.

    The node is labeled by the formatted value, as ``2.0``.
    """
    return Named(format_value(value))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def joined_labels(labels: Iterable[str]) -> str:
    """The labels of factors joined with ``·``.

    A label that is already a product joins without parentheses, and any other
    compound label is parenthesized.
    """
    return PRODUCT_SYMBOL.join(
        label if is_product(label) else grouped_label(label) for label in labels
    )


def warn_collapsed(collapse: Collapse, *, stored: bool) -> None:
    """Warn that a label or a notation shown left parts out, as *collapse* records.

    Parameters
    ----------
    collapse : Collapse
        What the rendering shown left out.
    stored : bool
        Whether the rendering is a term's label, which the term stores when it
        is built, so a setting raised later does not change it.

    Warns
    -----
    UserWarning
        Always. A rendering that collapsed a part beyond ``notation_config.max_depth``
        names the setting, and for a stored label says to raise it before
        building the term. A rendering that showed a part storage truncated
        says that no setting shows that part.
    """
    depth = f"notation_config.max_depth={collapse.max_depth} levels"
    stored_levels = f"the {_STORED_DEPTH} levels a term stores"
    shown_as = f"so its deeper parts show as their labels or {ELLIPSIS!r}"
    subject = "this label" if stored else "a label or notation"
    if collapse.beyond_depth:
        setting = f"{depth}, the setting when the term was built" if stored else depth
        joint = "," if stored else ""
        limit = f"{joint} and more than {stored_levels}" if collapse.truncated else ""
        fix = "before building the term " if stored else ""
        shows = f"up to {_STORED_DEPTH} levels" if collapse.truncated else "them"
        message = (
            f"{subject} nests more than {setting}{limit}, {shown_as}. Raise "
            f"notation_config.max_depth {fix}to show {shows}."
        )
    else:
        message = (
            f"{subject} nests more than {stored_levels}, {shown_as} at any "
            f"notation_config.max_depth"
        )
    warnings.warn(message, UserWarning, skip_file_prefixes=_WARNING_SKIP_PREFIXES)


def _max_depth() -> int:
    """The number of nested levels a rendering shows, as ``notation_config`` sets it."""
    from .config import notation_config

    return notation_config.max_depth
