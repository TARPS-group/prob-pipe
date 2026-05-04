"""Iteration regression tests: distributions are non-iterable; only the
Record family iterates field names (#142).

The rule (codified in STYLE_GUIDE.md §1.11):

* :class:`Record`, :class:`NumericRecord`, :class:`RecordArray`,
  :class:`NumericRecordArray` iterate field names dict-style.
* :class:`DistributionArray` has ``__len__ == prod(batch_shape)`` and
  is positional (access via ``da[i]``); not generally treated as an
  iterable.
* Every other :class:`Distribution` subclass is non-iterable. Stored
  samples live on ``.samples`` / ``.draws()``; ``.n`` reports the
  count.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from probpipe import (
    Beta,
    BootstrapReplicateDistribution,
    Distribution,
    EmpiricalDistribution,
    Gamma,
    JointEmpirical,
    KDEDistribution,
    MultivariateNormal,
    Normal,
    NumericRecord,
    NumericRecordArray,
    ProductDistribution,
    Record,
    RecordArray,
    RecordEmpiricalDistribution,
    RecordBootstrapReplicateDistribution,
    TransformedDistribution,
)


def _make_transformed():
    """Build a TransformedDistribution at parametrise time.

    Importing the bijector here keeps the test parametrisation
    side-effect-free at import — TFP's bijector module is heavy
    enough to warrant deferring.
    """
    import tensorflow_probability.substrates.jax.bijectors as tfb
    return TransformedDistribution(
        Normal(loc=0.0, scale=1.0, name="base"), tfb.Exp(), name="td",
    )


# User-constructible Distribution subclasses, parametrised here to pin
# the non-iterable rule (#142). WF-output classes (BroadcastDistribution,
# _RecordMarginal / _MixtureMarginal / _ListMarginal, BootstrapDistribution
# of an op return) are produced by the WorkflowFunction layer rather than
# user code; they inherit non-iterability from their bases (Distribution
# / RecordEmpiricalDistribution / Distribution) and don't need direct
# parametrisation here.
DISTRIBUTIONS = [
    pytest.param(lambda: Normal(loc=0.0, scale=1.0, name="x"), id="Normal"),
    pytest.param(lambda: Beta(alpha=1.0, beta=1.0, name="x"), id="Beta"),
    pytest.param(lambda: Gamma(concentration=2.0, rate=1.0, name="x"), id="Gamma"),
    pytest.param(
        lambda: MultivariateNormal(
            loc=jnp.zeros(3), cov=jnp.eye(3), name="x",
        ),
        id="MultivariateNormal",
    ),
    pytest.param(
        lambda: ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=0.0, scale=1.0, name="y"),
        ),
        id="ProductDistribution",
    ),
    pytest.param(
        lambda: _make_transformed(),
        id="TransformedDistribution",
    ),
    pytest.param(
        lambda: KDEDistribution(jnp.zeros((20, 3)), name="kde"),
        id="KDEDistribution",
    ),
    pytest.param(
        lambda: EmpiricalDistribution(
            jnp.zeros((10, 3)), name="theta",
        ),
        id="RecordEmpiricalDistribution",
    ),
    pytest.param(
        lambda: BootstrapReplicateDistribution(
            jnp.zeros((10, 2)), name="obs",
        ),
        id="RecordBootstrapReplicateDistribution",
    ),
    pytest.param(
        lambda: BootstrapReplicateDistribution(
            Normal(loc=0.0, scale=1.0, name="x"), n=5,
        ),
        id="BootstrapReplicateDistribution_sampleable",
    ),
    pytest.param(
        lambda: JointEmpirical(
            x=jnp.zeros((10,)),
            y=jnp.zeros((10,)),
        ),
        id="NumericJointEmpirical",
    ),
]


@pytest.mark.parametrize("make_dist", DISTRIBUTIONS)
def test_distribution_is_not_iterable(make_dist):
    """Every Distribution subclass must reject iteration.

    The rule: distributions represent a single random variable, not a
    collection. Use ``.samples`` / ``.draws()`` for stored samples,
    ``.n`` for count, and ``DistributionArray`` for batched
    distributions.

    Python's iter-via-``__getitem__`` fallback returns a non-empty
    iterator object even on classes without ``__iter__``, so we
    actually iterate (or call ``list``) to confirm the protocol does
    not yield items.

    Accepted exceptions are ``TypeError`` (class explicitly forbids
    iteration via ``__iter__`` raising or no ``__getitem__``) and
    ``KeyError`` (Record-family ``__getitem__`` rejects integer keys
    because fields are str-keyed). ``IndexError`` is **not** accepted:
    Python's iter-fallback treats ``IndexError`` on integer access as
    the end-of-iteration signal, and ``list(iter(d))`` would silently
    return ``[]`` rather than surfacing the failure — exactly the
    silent-iteration footgun the rule is meant to forbid.
    """
    d = make_dist()
    assert isinstance(d, Distribution)
    with pytest.raises((TypeError, KeyError)):
        list(iter(d))


# -- Record family is iterable ---------------------------------------------


def test_record_iterates_field_names():
    r = Record(a=1.0, b=2.0)
    assert list(iter(r)) == ["a", "b"]


def test_numeric_record_iterates_field_names():
    nr = NumericRecord(a=jnp.array(1.0), b=jnp.array([2.0, 3.0]))
    assert list(iter(nr)) == ["a", "b"]


def test_record_array_iterates_field_names():
    from probpipe.core.record import RecordTemplate
    ra = RecordArray(
        a=jnp.zeros((5,)),
        b=jnp.zeros((5,)),
        batch_shape=(5,),
        template=RecordTemplate(a=(), b=()),
    )
    assert list(iter(ra)) == ["a", "b"]


def test_numeric_record_array_iterates_field_names():
    from probpipe.core.record import NumericRecordTemplate
    nra = NumericRecordArray(
        a=jnp.zeros((4,)),
        b=jnp.zeros((4,)),
        batch_shape=(4,),
        template=NumericRecordTemplate(a=(), b=()),
    )
    assert list(iter(nra)) == ["a", "b"]
