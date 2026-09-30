"""Report what the package declares but does not yet implement.

Two lists make up the report:

- **Stubs**: every ``raise NotImplementedError("<Owner.member>")`` whose message is
  a string literal naming the member, optionally followed by ``": detail"``, and
  every body a factory built and marked ``_is_stub`` in a capability table. A
  runtime refusal with a computed message is not a stub.
- **Pending tests**: every test carrying the ``pending`` marker, collected by
  pytest.

Usage::

    python scripts/design/ledger.py              # both lists
    python scripts/design/ledger.py --no-tests   # stubs only, without collecting tests
    python scripts/design/ledger.py --require-empty

``--require-empty`` exits with status 1 while either list is non-empty.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "probpipe"

_STUB_MESSAGE = re.compile(r"^[A-Za-z_]\w*(\.\w+)+(:.*)?$")


@dataclass(frozen=True)
class Stub:
    """One declared member without an implementation."""

    path: str
    line: int
    member: str


def _literal_message(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None


def stubs(package: Path = PACKAGE) -> Iterator[Stub]:
    """Every stub site under *package*, in path and line order."""
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        relative = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            call: ast.Call | None = None
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                function = node.exc.func
                if isinstance(function, ast.Name) and function.id == "NotImplementedError":
                    call = node.exc
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_stub"
            ):
                call = node
            if call is None:
                continue
            message = _literal_message(call)
            if message is not None and _STUB_MESSAGE.match(message):
                yield Stub(relative, node.lineno, message.split(":", 1)[0])


def generated_stubs() -> Iterator[Stub]:
    """Every stub a capability table holds, built by a factory rather than written out.

    A factory marks each body it builds with ``_is_stub``; the capability tables
    of the distribution kinds are where such bodies are stored.
    """
    import probpipe.distributions as distributions

    seen: set[int] = set()
    for name in distributions.__all__:
        cls = getattr(distributions, name)
        table = getattr(cls, "_capability_table", None) if isinstance(cls, type) else None
        for methods in (table or {}).values():
            for method in methods.values():
                if getattr(method, "_is_stub", False) and id(method) not in seen:
                    seen.add(id(method))
                    code = method.__code__
                    path = str(Path(code.co_filename).resolve().relative_to(ROOT))
                    yield Stub(path, code.co_firstlineno, method.__qualname__)


def pending_tests() -> list[str]:
    """The node ids of every test carrying the ``pending`` marker."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "pending",
            "-p",
            "no:cacheprovider",
            "--no-cov",
            "-n",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if "::" in line]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-tests", action="store_true", help="skip collecting pending tests")
    parser.add_argument(
        "--require-empty", action="store_true", help="exit 1 while a stub or pending test remains"
    )
    args = parser.parse_args(argv)
    found = list(stubs()) + list(generated_stubs())
    print(f"stubs: {len(found)}")
    for module, count in sorted(Counter(stub.path for stub in found).items()):
        print(f"  {module}: {count}")
    for stub in found:
        print(f"    {stub.path}:{stub.line}  {stub.member}")
    pending: list[str] = []
    if not args.no_tests:
        pending = pending_tests()
        print(f"pending tests: {len(pending)}")
        for module, count in sorted(Counter(t.split("::")[0] for t in pending).items()):
            print(f"  {module}: {count}")
    if args.require_empty and (found or pending):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
