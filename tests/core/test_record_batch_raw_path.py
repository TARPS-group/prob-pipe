"""A record batch's raw form at a path, as a record's is."""

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import NumericRecordBatch
from probpipe.core._record_spec import NumericRecordSpec


def _batch():
    return NumericRecordBatch(
        {"g/a": jnp.arange(3.0), "g/b": jnp.ones(3), "c": jnp.zeros((3, 2))},
        "draw",
        element_spec=NumericRecordSpec(g=NumericRecordSpec(a=(), b=()), c=(2,)),
        label="draws",
    )


class TestRawAtAPath:
    def test_a_field_gives_its_column(self):
        np.testing.assert_array_equal(np.asarray(_batch().raw("g/a")), np.arange(3.0))
        np.testing.assert_array_equal(np.asarray(_batch().raw(("g", "a"))), np.arange(3.0))

    def test_an_interior_node_gives_the_columns_beneath_it(self):
        node = _batch().raw("g")
        assert set(node) == {"a", "b"}
        np.testing.assert_array_equal(np.asarray(node["b"]), np.ones(3))

    def test_no_path_gives_the_whole_storage_view(self):
        assert set(_batch().raw()) == {"g", "c"}

    def test_an_unknown_path_names_the_batch_paths(self):
        with pytest.raises(KeyError, match="g/a"):
            _batch().raw("g/z")
