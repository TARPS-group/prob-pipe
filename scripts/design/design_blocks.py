"""Read the design reference's code blocks as Python declarations.

Each numbered section of ``design/`` leads with contract code blocks that are
valid Python: class declarations with their bases, member signatures with
``...`` bodies, and dataclass fields, annotated with trailing comments. This
module parses those blocks so that tooling can list what a section declares,
emit stubs from it, and check an implementation against it.

Usage::

    python scripts/design/design_blocks.py list III.7 III.9
    python scripts/design/design_blocks.py stubs III.9
    python scripts/design/design_blocks.py check III.7 III.8 --module probpipe.distributions

``check`` looks each declared name up in the given modules, in order, and
reports every declared class, member, field, and parameter that the
implementation lacks or declares differently.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DESIGN_DIR = Path(__file__).resolve().parents[2] / "design"

_SECTION_HEADING = re.compile(r"^## ([IVX]+\.\d+) — (.+)$", re.M)
_CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.S)


# ---------------------------------------------------------------------------
# The declaration model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Parameter:
    """One declared parameter: its name, kind, annotation, and default, as source."""

    name: str
    kind: inspect._ParameterKind
    annotation: str | None
    default: str | None


@dataclass(frozen=True)
class Member:
    """A method or property declared in a class body."""

    name: str
    decorators: tuple[str, ...]
    parameters: tuple[Parameter, ...]
    returns: str | None
    comment: str

    @property
    def is_property(self) -> bool:
        return "property" in self.decorators


@dataclass(frozen=True)
class Field:
    """An annotated attribute declared in a class body, such as a dataclass field."""

    name: str
    annotation: str
    comment: str


@dataclass(frozen=True)
class ClassDeclaration:
    """A class a section declares, with its bases, members, and fields."""

    section: str
    name: str
    bases: tuple[str, ...]
    decorators: tuple[str, ...]
    members: tuple[Member, ...]
    fields: tuple[Field, ...]
    comment: str


@dataclass(frozen=True)
class FunctionDeclaration:
    """A top-level function a section declares."""

    section: str
    member: Member


@dataclass(frozen=True)
class NameDeclaration:
    """A top-level annotated name a section declares, such as a registry instance."""

    section: str
    name: str
    annotation: str
    comment: str


type Declaration = ClassDeclaration | FunctionDeclaration | NameDeclaration


@dataclass(frozen=True)
class Section:
    """One numbered section of the reference and its Python code blocks."""

    id: str
    title: str
    file: str
    blocks: tuple[str, ...]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def load_sections(design_dir: Path = DESIGN_DIR) -> dict[str, Section]:
    """Every numbered section of the reference, keyed by its id, such as ``"III.7"``."""
    sections: dict[str, Section] = {}
    for path in sorted(design_dir.glob("*.md")):
        text = path.read_text()
        headings = list(_SECTION_HEADING.finditer(text))
        for index, heading in enumerate(headings):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            body = text[heading.start() : end]
            sections[heading.group(1)] = Section(
                id=heading.group(1),
                title=heading.group(2).strip(),
                file=path.name,
                blocks=tuple(_CODE_BLOCK.findall(body)),
            )
    return sections


def _comment_after(lines: Sequence[str], node: ast.AST) -> str:
    """The comments on a node's last line and on the comment-only lines that follow it."""
    comments: list[str] = []
    end = node.end_lineno or node.lineno
    for number in range(node.lineno, end + 1):
        line = lines[number - 1]
        if "#" in line:
            comments.append(line.split("#", 1)[1].strip())
    for line in lines[end:]:
        stripped = line.strip()
        if not stripped.startswith("#"):
            break
        comments.append(stripped[1:].strip())
    return " ".join(comment for comment in comments if comment)


def _parameters(arguments: ast.arguments) -> tuple[Parameter, ...]:
    """The parameters of a signature, with defaults and annotations as source."""
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(arguments.defaults))
    defaults += list(arguments.defaults)
    result: list[Parameter] = []

    def source(node: ast.AST | None) -> str | None:
        return None if node is None else ast.unparse(node)

    for index, argument in enumerate(positional):
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if index < len(arguments.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        result.append(
            Parameter(argument.arg, kind, source(argument.annotation), source(defaults[index]))
        )
    if arguments.vararg is not None:
        result.append(
            Parameter(
                arguments.vararg.arg,
                inspect.Parameter.VAR_POSITIONAL,
                source(arguments.vararg.annotation),
                None,
            )
        )
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
        result.append(
            Parameter(
                argument.arg,
                inspect.Parameter.KEYWORD_ONLY,
                source(argument.annotation),
                source(default),
            )
        )
    if arguments.kwarg is not None:
        result.append(
            Parameter(
                arguments.kwarg.arg,
                inspect.Parameter.VAR_KEYWORD,
                source(arguments.kwarg.annotation),
                None,
            )
        )
    return tuple(result)


def _member(node: ast.FunctionDef, lines: Sequence[str]) -> Member:
    return Member(
        name=node.name,
        decorators=tuple(ast.unparse(decorator) for decorator in node.decorator_list),
        parameters=_parameters(node.args),
        returns=None if node.returns is None else ast.unparse(node.returns),
        comment=_comment_after(lines, node),
    )


def parse_block(section: str, block: str) -> list[Declaration]:
    """The declarations of one code block, in source order.

    Raises
    ------
    SyntaxError
        If the block is not valid Python.
    """
    lines = block.splitlines()
    declarations: list[Declaration] = []
    for node in ast.parse(block).body:
        if isinstance(node, ast.ClassDef):
            members = tuple(
                _member(child, lines) for child in node.body if isinstance(child, ast.FunctionDef)
            )
            fields = tuple(
                Field(child.target.id, ast.unparse(child.annotation), _comment_after(lines, child))
                for child in node.body
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
            )
            header = ast.ClassDef(
                name=node.name, bases=[], keywords=[], body=[], decorator_list=[], type_params=[]
            )
            header.lineno = header.end_lineno = node.lineno
            declarations.append(
                ClassDeclaration(
                    section=section,
                    name=node.name,
                    bases=tuple(ast.unparse(base) for base in node.bases),
                    decorators=tuple(ast.unparse(d) for d in node.decorator_list),
                    members=members,
                    fields=fields,
                    comment=_comment_after(lines, header),
                )
            )
        elif isinstance(node, ast.FunctionDef):
            declarations.append(FunctionDeclaration(section, _member(node, lines)))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            declarations.append(
                NameDeclaration(
                    section,
                    node.target.id,
                    ast.unparse(node.annotation),
                    _comment_after(lines, node),
                )
            )
    return declarations


def declarations(
    section_ids: Iterable[str], sections: dict[str, Section] | None = None
) -> list[Declaration]:
    """The declarations of the given sections, in section and source order.

    Raises
    ------
    KeyError
        If a section id is not a numbered section of the reference.
    """
    sections = load_sections() if sections is None else sections
    result: list[Declaration] = []
    for section_id in section_ids:
        for block in sections[section_id].blocks:
            result.extend(parse_block(section_id, block))
    return result


# ---------------------------------------------------------------------------
# Checking an implementation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One disagreement between a declaration and the implementation."""

    section: str
    name: str
    problem: str


def _compare_signature(
    section: str, owner: str, member: Member, implemented: Any, skip_first: bool
) -> list[Finding]:
    """Parameter names, kinds, and defaults of *member* against *implemented*."""
    where = f"{owner}.{member.name}" if owner else member.name
    try:
        signature = inspect.signature(implemented)
    except (TypeError, ValueError):
        return [Finding(section, where, "has no inspectable signature")]
    declared = list(member.parameters)
    if skip_first and declared and declared[0].name in ("self", "cls"):
        declared = declared[1:]
    actual = list(signature.parameters.values())
    if skip_first and actual and actual[0].name in ("self", "cls"):
        actual = actual[1:]
    findings: list[Finding] = []
    declared_names = [parameter.name for parameter in declared]
    actual_names = [parameter.name for parameter in actual]
    if declared_names != actual_names:
        findings.append(
            Finding(section, where, f"declares parameters {declared_names}, has {actual_names}")
        )
        return findings
    for want, have in zip(declared, actual, strict=True):
        if want.kind != have.kind:
            findings.append(
                Finding(
                    section,
                    where,
                    f"declares {want.name} as {want.kind.name}, has {have.kind.name}",
                )
            )
        if want.default is not None:
            if have.default is inspect.Parameter.empty:
                findings.append(
                    Finding(section, where, f"declares {want.name}={want.default}, has no default")
                )
            elif want.default not in (str(have.default), repr(have.default)):
                findings.append(
                    Finding(
                        section,
                        where,
                        f"declares {want.name}={want.default}, has {have.default!r}",
                    )
                )
    return findings


def check(
    decls: Iterable[Declaration],
    resolve: Callable[[str], Any | None],
    *,
    skip: Iterable[str] = (),
) -> list[Finding]:
    """Every disagreement between *decls* and the objects *resolve* finds.

    Parameters
    ----------
    decls : iterable of Declaration
        What the reference declares.
    resolve : callable
        Maps a declared top-level name to the implemented object, or ``None``.
    skip : iterable of str
        Declared names left unchecked, for example ``"__mul__"``, whose owner a
        section's code block does not name.

    Returns
    -------
    list of Finding
        Empty when the implementation agrees with every declaration.
    """
    skipped = set(skip)
    findings: list[Finding] = []
    for decl in decls:
        if isinstance(decl, ClassDeclaration):
            if decl.name in skipped:
                continue
            implemented = resolve(decl.name)
            if implemented is None:
                findings.append(Finding(decl.section, decl.name, "is not implemented"))
                continue
            if not isinstance(implemented, type):
                findings.append(Finding(decl.section, decl.name, "is not a class"))
                continue
            mro_names = {klass.__name__ for klass in implemented.__mro__}
            for base in decl.bases:
                base_name = re.sub(r"\[.*\]$", "", base).split(".")[-1]
                if base_name not in mro_names:
                    findings.append(
                        Finding(decl.section, decl.name, f"does not derive from {base_name}")
                    )
            for member in decl.members:
                attribute = inspect.getattr_static(implemented, member.name, None)
                if attribute is None:
                    findings.append(
                        Finding(decl.section, f"{decl.name}.{member.name}", "is not implemented")
                    )
                    continue
                if member.is_property:
                    if not isinstance(attribute, property):
                        findings.append(
                            Finding(decl.section, f"{decl.name}.{member.name}", "is not a property")
                        )
                    continue
                target = attribute
                if isinstance(target, (classmethod, staticmethod)):
                    target = target.__func__
                findings.extend(
                    _compare_signature(decl.section, decl.name, member, target, skip_first=True)
                )
            if decl.fields:
                annotations: dict[str, Any] = {}
                for klass in reversed(implemented.__mro__):
                    annotations.update(getattr(klass, "__annotations__", {}))
                for declared_field in decl.fields:
                    if declared_field.name not in annotations and not hasattr(
                        implemented, declared_field.name
                    ):
                        findings.append(
                            Finding(
                                decl.section,
                                f"{decl.name}.{declared_field.name}",
                                "is not declared",
                            )
                        )
        elif isinstance(decl, FunctionDeclaration):
            if decl.member.name in skipped:
                continue
            implemented = resolve(decl.member.name)
            if implemented is None:
                findings.append(Finding(decl.section, decl.member.name, "is not implemented"))
                continue
            findings.extend(_compare_signature(decl.section, "", decl.member, implemented, False))
        elif decl.name not in skipped and resolve(decl.name) is None:
            findings.append(Finding(decl.section, decl.name, "is not implemented"))
    return findings


def resolver(modules: Sequence[str]) -> Callable[[str], Any | None]:
    """A *resolve* for :func:`check` that looks a name up in *modules*, in order."""
    loaded = [importlib.import_module(module) for module in modules]

    def resolve(name: str) -> Any | None:
        for module in loaded:
            if hasattr(module, name):
                return getattr(module, name)
        return None

    return resolve


# ---------------------------------------------------------------------------
# Emitting stubs
# ---------------------------------------------------------------------------


def _signature_source(member: Member) -> str:
    parts: list[str] = []
    seen_keyword_only = False
    for index, parameter in enumerate(member.parameters):
        text = parameter.name
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            text = "*" + text
            seen_keyword_only = True
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            text = "**" + text
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY and not seen_keyword_only:
            parts.append("*")
            seen_keyword_only = True
        if parameter.annotation is not None:
            text += f": {parameter.annotation}"
        if parameter.default is not None:
            text += f" = {parameter.default}" if parameter.annotation else f"={parameter.default}"
        parts.append(text)
        is_last_positional_only = parameter.kind is inspect.Parameter.POSITIONAL_ONLY and (
            index + 1 == len(member.parameters)
            or member.parameters[index + 1].kind is not inspect.Parameter.POSITIONAL_ONLY
        )
        if is_last_positional_only:
            parts.append("/")
    returns = f" -> {member.returns}" if member.returns else ""
    return f"def {member.name}({', '.join(parts)}){returns}:"


def stub_source(decls: Iterable[Declaration]) -> str:
    """Python source declaring every class and function in *decls* as a stub.

    A method body raises ``NotImplementedError`` naming the member, a protocol
    member keeps the ``...`` body, and a declaration's comments become its
    docstring.
    """
    out: list[str] = []
    for decl in decls:
        if isinstance(decl, ClassDeclaration):
            is_protocol = any(base.split("[")[0] == "Protocol" for base in decl.bases)
            out.extend(f"@{decorator}" for decorator in decl.decorators)
            bases = f"({', '.join(decl.bases)})" if decl.bases else ""
            out.append(f"class {decl.name}{bases}:")
            out.append(f'    """{decl.comment or decl.name + "."}"""')
            for declared_field in decl.fields:
                out.append(f"    {declared_field.name}: {declared_field.annotation}")
            for member in decl.members:
                out.append("")
                out.extend(f"    @{decorator}" for decorator in member.decorators)
                out.append("    " + _signature_source(member))
                if member.comment:
                    out.append(f'        """{member.comment}"""')
                if is_protocol:
                    out.append("        ...")
                else:
                    out.append(f'        raise NotImplementedError("{decl.name}.{member.name}")')
            out.extend(["", ""])
        elif isinstance(decl, FunctionDeclaration):
            out.append(_signature_source(decl.member))
            if decl.member.comment:
                out.append(f'    """{decl.member.comment}"""')
            out.append(f'    raise NotImplementedError("{decl.member.name}")')
            out.extend(["", ""])
        else:
            out.append(f"{decl.name}: {decl.annotation}")
            out.extend(["", ""])
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("list", "stubs", "check"))
    parser.add_argument("sections", nargs="+", help="section ids, such as III.7")
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="for check: a module to resolve declared names in; repeatable, first match wins",
    )
    parser.add_argument("--skip", action="append", default=[], help="for check: a name to skip")
    args = parser.parse_args(argv)
    decls = declarations(args.sections)
    if args.command == "list":
        for decl in decls:
            if isinstance(decl, ClassDeclaration):
                members = ", ".join(member.name for member in decl.members)
                print(f"{decl.section}\tclass {decl.name}({', '.join(decl.bases)})\t{members}")
            elif isinstance(decl, FunctionDeclaration):
                print(f"{decl.section}\tdef {decl.member.name}")
            else:
                print(f"{decl.section}\t{decl.name}: {decl.annotation}")
        return 0
    if args.command == "stubs":
        sys.stdout.write(stub_source(decls))
        return 0
    findings = check(decls, resolver(args.module or ["probpipe"]), skip=args.skip)
    for finding in findings:
        print(f"{finding.section}\t{finding.name}\t{finding.problem}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
