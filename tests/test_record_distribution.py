"""Tests for the ``RecordDistribution`` / ``_RecordDistributionView`` surface.

- Public ``.parent`` / ``.field`` properties on ``_RecordDistributionView``
  (parallel to ``_RecordArrayView``).
- ``.n`` property on ``RecordDistribution`` (STYLE_GUIDE §1.9).
- Single-field ``.shape`` / ``.ndim`` shims on ``RecordDistribution``
  and ``_RecordDistributionView`` — and their multi-field ``TypeError``.
- ``event_shapes`` returns ``dict[str, tuple[int, ...]]`` uniformly,
  including the empty-dict case for untemplated distributions.
"""

from __future__ import annotations

import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    Normal,
)
from probpipe.core._record_distribution import (
    RecordDistribution,
    _RecordDistributionView,
)
from probpipe.distributions.joint import ProductDistribution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_field_dist():
    """Single-field ``ProductDistribution`` — exercises the array-like shim."""
    return ProductDistribution(x=Normal(0.0, 1.0, name="x"))


@pytest.fixture
def multi_field_dist():
    """Two-field ``ProductDistribution`` — exercises the multi-field guard."""
    return ProductDistribution(
        x=Normal(0.0, 1.0, name="x"),
        y=Normal(0.0, 1.0, name="y"),
    )


# ---------------------------------------------------------------------------
# View .parent / .field public surface
# ---------------------------------------------------------------------------


class TestViewParentField:
    """``dist[field]`` returns a view whose ``.parent`` / ``.field`` are
    public and match ``_RecordArrayView``'s surface."""

    def test_view_parent_is_dist(self, multi_field_dist):
        view = multi_field_dist["x"]
        assert isinstance(view, _RecordDistributionView)
        assert view.parent is multi_field_dist

    def test_view_field_is_key(self, multi_field_dist):
        assert multi_field_dist["x"].field == "x"
        assert multi_field_dist["y"].field == "y"

    def test_select_all_views_share_parent(self, multi_field_dist):
        cols = multi_field_dist.select_all()
        assert {v.parent for v in cols.values()} == {multi_field_dist}


# ---------------------------------------------------------------------------
# RecordDistribution.n
# ---------------------------------------------------------------------------


class TestRecordDistributionN:
    """STYLE_GUIDE §1.9 — finite-collection distributions expose ``.n``
    as ``prod(batch_shape)``."""

    def test_scalar_dist_n_is_one(self, multi_field_dist):
        assert multi_field_dist.n == 1

    def test_single_field_dist_n_is_one(self, single_field_dist):
        assert single_field_dist.n == 1


# ---------------------------------------------------------------------------
# Single-field .shape / .ndim shims
# ---------------------------------------------------------------------------


class TestSingleFieldShapeShim:
    """A single-field ``RecordDistribution`` exposes ``.shape`` and
    ``.ndim`` as thin delegates to the sole field's event shape (with
    the ``batch_shape`` prefix). Multi-field raises ``TypeError``."""

    def test_shape_on_single_field_dist(self, single_field_dist):
        # Scalar Normal has event_shape = (), so .shape == ().
        assert single_field_dist.shape == ()
        assert single_field_dist.ndim == 0

    def test_shape_raises_on_multi_field(self, multi_field_dist):
        with pytest.raises(TypeError, match="is not array-like"):
            _ = multi_field_dist.shape
        with pytest.raises(TypeError, match="is not array-like"):
            _ = multi_field_dist.ndim

    def test_error_message_mentions_event_shapes_not_dtypes(
        self, multi_field_dist,
    ):
        """The error message points users at ``.event_shapes`` only.
        ``.dtypes`` is only defined on ``NumericRecordDistribution``
        and shouldn't be mentioned on the generic base."""
        with pytest.raises(TypeError) as excinfo:
            _ = multi_field_dist.shape
        assert ".dtypes" not in str(excinfo.value)
        assert ".event_shapes" in str(excinfo.value)

    def test_view_shape_matches_event_shape(self, multi_field_dist):
        view = multi_field_dist["x"]
        # Scalar Normal component → event_shape = (), batch_shape = ().
        assert view.shape == ()
        assert view.ndim == 0


# ---------------------------------------------------------------------------
# event_shapes always returns a dict
# ---------------------------------------------------------------------------


class TestEventShapesUniformDict:
    """``event_shapes`` returns ``dict[str, tuple[int, ...]]`` on every
    ``RecordDistribution``. Untemplated distributions get ``{}`` —
    matching their empty ``.fields``."""

    def test_templated_returns_dict(self, multi_field_dist):
        es = multi_field_dist.event_shapes
        assert isinstance(es, dict)
        assert es == {"x": (), "y": ()}

    def test_array_empirical_auto_wraps_single_field(self):
        """An ``EmpiricalDistribution`` built from a raw array auto-wraps
        as a single-field Record keyed by ``name=``; the template,
        fields, and event shapes reflect that single field."""
        samples = np.random.randn(100, 3)
        dist = EmpiricalDistribution(samples, name="x")
        assert dist.record_template is not None
        assert dist.fields == ("x",)
        assert dist.event_shapes == {"x": (3,)}
        # Single-field shortcut.
        assert dist.event_shape == (3,)
