"""A law or a kernel that holds paths fixed, built without conditioning it.

Conditioning records the paths it fixes in the expression of its result, and the
notation lists them after ``;``. The tests of the notation and the repr build such
a law from any law with :func:`with_fixed_paths`, so they need no kernel that
conditioning applies to.
"""

from __future__ import annotations

from typing import Any


def with_fixed_paths(term: Any, *paths: str) -> Any:
    """*term*, a law or a kernel just built, holding *paths* fixed, as conditioning records them."""
    term._store_expression(term._expression.with_fixed(paths))
    return term
