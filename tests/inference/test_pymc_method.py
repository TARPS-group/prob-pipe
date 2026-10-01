"""The PyMC inference methods, ``pymc_nuts`` and ``pymc_advi``, on the canonical cases.

Each consumes a ``PyMCModel``, so the harness conditions each case's PyMC
representation on the case's observations; the cases' references are shared
with every other backend.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymc")

from tests.inference._harness import validate_method

test_pymc_nuts_canonical = validate_method("pymc_nuts")
test_pymc_advi_canonical = validate_method("pymc_advi")
