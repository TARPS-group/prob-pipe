"""Iteration regression tests: distributions are non-iterable; only the
Record family iterates field names.

The rule (codified in STYLE_GUIDE.md §1.11):

* :class:`Record` and :class:`NumericRecord` iterate field names dict-style.
* :class:`RecordBatch` / :class:`NumericRecordBatch` are collections: they
  iterate leading-axis views, and fields are read from ``event_template``.
* :class:`DistributionBatch` is a batch: ``len`` is the leading-axis size,
  ``batch_size`` the total count of laws, and iteration visits views of
  the laws along the leading axis.
* Every other :class:`Distribution` subclass is non-iterable.
  An empirical law exposes its stored atoms on ``.atoms`` with
  ``.num_atoms`` reporting the count, and an inference result its
  draws on ``.draws()``; parametric distributions have neither.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from probpipe import (
    Beta,
    BootstrapDistribution,
    BootstrapReplicateDistribution,
    Distribution,
    EmpiricalDistribution,
    Gamma,
    KDEDistribution,
    MinibatchedDistribution,
    MultivariateNormal,
    Normal,
    NumericRecord,
    NumericRecordBatch,
    Record,
    RecordBatch,
)
from probpipe.families import BijectorTransformedDistribution


def _make_transformed():
    """Build a BijectorTransformedDistribution at parametrise time.

    Importing the bijector here keeps the test parametrisation
    side-effect-free at import — TFP's bijector module is heavy
    enough to warrant deferring.
    """
    import tensorflow_probability.substrates.jax.bijectors as tfb

    return BijectorTransformedDistribution(
        "td",
        Normal(loc=0.0, scale=1.0, label="base"),
        tfb.Exp(),
    )


# User-constructible Distribution subclasses, parametrised here to pin
# the non-iterable rule. A lifted call's result is an EmpiricalDistribution,
# which is parametrised with the others.
DISTRIBUTIONS = [
    pytest.param(lambda: Normal(loc=0.0, scale=1.0, label="x"), id="Normal"),
    pytest.param(lambda: Beta(alpha=1.0, beta=1.0, label="x"), id="Beta"),
    pytest.param(lambda: Gamma(concentration=2.0, rate=1.0, label="x"), id="Gamma"),
    pytest.param(
        lambda: MultivariateNormal(
            loc=jnp.zeros(3),
            cov=jnp.eye(3),
            label="x",
        ),
        id="MultivariateNormal",
    ),
    pytest.param(
        lambda: Normal(loc=0.0, scale=1.0, label="x") * Normal(loc=0.0, scale=1.0, label="y"),
        id="FactoredDistribution",
    ),
    pytest.param(
        lambda: _make_transformed(),
        id="BijectorTransformedDistribution",
    ),
    pytest.param(
        lambda: KDEDistribution("kde", jnp.arange(60.0).reshape(20, 3)),
        id="KDEDistribution",
    ),
    pytest.param(
        lambda: EmpiricalDistribution(
            "theta",
            jnp.zeros((10, 3)),
        ),
        id="EmpiricalDistribution",
    ),
    pytest.param(
        lambda: BootstrapReplicateDistribution(
            "obs",
            EmpiricalDistribution("obs", jnp.zeros((10, 2))),
        ),
        id="BootstrapReplicateDistribution_empirical",
    ),
    pytest.param(
        lambda: BootstrapReplicateDistribution(
            "boot",
            Normal(loc=0.0, scale=1.0, label="x"),
            replicate_size=5,
        ),
        id="BootstrapReplicateDistribution_sampleable",
    ),
    pytest.param(
        lambda: BootstrapDistribution(
            "measure",
            Normal(loc=0.0, scale=1.0, label="x"),
            5,
        ),
        id="BootstrapDistribution",
    ),
    pytest.param(
        lambda: _make_minibatched_distribution(),
        id="MinibatchedDistribution",
    ),
]


def _make_minibatched_distribution():
    """Build a MinibatchedDistribution at parametrise time."""
    from probpipe.families import BernoulliFamily, glm_likelihood

    X = jnp.eye(4)
    y = jnp.array([1.0, 0.0, 1.0, 0.0])
    prior = MultivariateNormal(loc=jnp.zeros(4), cov=jnp.eye(4), label="beta")
    lik = glm_likelihood("y", BernoulliFamily(), X=X)
    return MinibatchedDistribution("measure", prior, lik, y, batch_size=2)


@pytest.mark.parametrize("make_dist", DISTRIBUTIONS)
def test_distribution_is_not_iterable(make_dist):
    """Every Distribution subclass must reject iteration.

    The rule: distributions represent a single random variable, not a
    collection. An empirical law exposes ``.atoms`` and ``.num_atoms``;
    ``DistributionBatch`` covers batched cases.

    Python's iter-via-``__getitem__`` fallback returns a non-empty
    iterator object even on classes without ``__iter__``, so we
    actually iterate (or call ``list``) to confirm the protocol does
    not yield items.

    ``Distribution`` sets ``__iter__`` to ``None``, so ``iter`` itself raises
    ``TypeError`` rather than falling back to ``__getitem__``, which addresses
    components. That fallback would yield nothing useful: a ``KeyError`` for an
    integer key, or, worse, an ``IndexError`` that ends iteration silently.
    """
    d = make_dist()
    assert isinstance(d, Distribution)
    with pytest.raises(TypeError):
        iter(d)


# -- Record family is iterable ---------------------------------------------


def test_record_iterates_field_names():
    r = Record("r", a=1.0, b=2.0)
    assert list(iter(r)) == ["a", "b"]


def test_numeric_record_iterates_field_names():
    nr = NumericRecord("nr", a=jnp.array(1.0), b=jnp.array([2.0, 3.0]))
    assert list(iter(nr)) == ["a", "b"]


def test_a_record_batch_iterates_leading_axis_views():
    from probpipe.core._specs import RecordSpec

    batch = RecordBatch(
        "batch",
        {"a": jnp.zeros((5,)), "b": jnp.zeros((5,))},
        level_names="draw",
        axes_per_level=(1,),
        element_spec=RecordSpec(a=(), b=()),
    )
    rows = list(iter(batch))
    assert len(rows) == 5
    assert all(tuple(row.keys()) == ("a", "b") for row in rows)


def test_a_numeric_record_batch_iterates_leading_axis_views():
    from probpipe.core._specs import NumericRecordSpec

    batch = NumericRecordBatch(
        "batch",
        {"a": jnp.zeros((4,)), "b": jnp.zeros((4,))},
        level_names="draw",
        axes_per_level=(1,),
        element_spec=NumericRecordSpec(a=(), b=()),
    )
    rows = list(iter(batch))
    assert len(rows) == 4
    assert all(tuple(row.keys()) == ("a", "b") for row in rows)
