"""Contract tests of mixture, a derived operation over composition and evaluation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ApplicabilityError,
    DistributionBatch,
    EmpiricalDistribution,
    Normal,
    NumericArrayBatch,
    RecordBatch,
    RecordSpec,
    conditional_distribution,
    log_prob,
    mean,
    sample,
    variance,
    workflow_run,
)
from probpipe.core._dispatch import ResolutionError
from probpipe.core._specs import OutputSpec
from probpipe.distributions._capabilities import (
    SupportsApproximateConditioning,
    SupportsConditionalSampling,
)
from probpipe.distributions._distribution import DistributionSpec
from probpipe.families import MixtureDistribution
from probpipe.operations._mixture import mixture

from ._laws import REAL, Gaussian, Kernel


def _kernel():
    return conditional_distribution(
        lambda mu: Normal("y", mu, 1.0), given_spec={"mu": REAL}, label="likelihood"
    )


def _mixing():
    return EmpiricalDistribution(jnp.array([-1.0, 3.0]), jnp.array([0.25, 0.75]), output_name="mu")


def test_mixture_is_derived_from_its_formula():
    assert mixture.is_derived
    assert mixture.formula == "pushforward(K & mixing, _kernel_output_projection(K))"


def test_the_result_carries_the_kernel_event_spec():
    kernel = Kernel("y", ("mu",))
    assert mixture.check(kernel, Gaussian("mu")).result == OutputSpec(
        DistributionSpec(kernel.event_spec)
    )


class _ApproximateKernel(Kernel, SupportsApproximateConditioning, SupportsConditionalSampling):
    """A kernel whose conditioning capability returns an approximate law."""

    def _conditional_sample(self, given, key, sample_shape=()):
        return self._condition_on(given)._sample(key, sample_shape)


def test_the_mixture_is_the_law_of_the_kernel_output():
    kernel = _kernel()
    with workflow_run(seed=0):
        law = mixture.using(max_evaluations=13)(kernel, Gaussian("mu"))
    assert law.event_spec == kernel.event_spec
    assert isinstance(law, EmpiricalDistribution)
    assert law.num_atoms == 13
    assert law.provenance.metadata["route"] == "derived"
    assert law.provenance.metadata["exact"] is False


def test_finite_mixing_retains_the_weights_density_and_moments():
    kernel, mixing = _kernel(), _mixing()
    report = mixture.check(kernel, mixing)
    assert (report.route, report.exact) == ("finite", True)
    law = mixture.using(exact_only=True)(kernel, mixing)
    assert isinstance(law, MixtureDistribution)
    assert law.event_spec == kernel.event_spec
    np.testing.assert_allclose(law._weights, [0.25, 0.75], rtol=1e-7, atol=0)
    np.testing.assert_allclose(mean(law), 2.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(variance(law), 4.0, rtol=0, atol=1e-6)
    x = jnp.array([-2.0, 0.0, 4.0])
    expected = np.log(
        0.25 * np.exp(-0.5 * (np.asarray(x) + 1) ** 2)
        + 0.75 * np.exp(-0.5 * (np.asarray(x) - 3) ** 2)
    ) - 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(
        log_prob(law, NumericArrayBatch(x, "value", label="values")), expected, rtol=1e-6, atol=1e-6
    )
    assert law.provenance.metadata["route"] == "finite"
    assert law.provenance.metadata["exact"] is True


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("form", ["array", "record", "nested", "named_record"])
def test_the_kernel_event_form_survives_projection(form, finite):
    def build(mu):
        if form == "array":
            return Normal("y", mu, 1.0)
        if form == "nested":
            spec, values = RecordSpec(g=RecordSpec(y=REAL)), {"g": {"y": jnp.stack([mu, mu + 2])}}
        else:
            spec, values = RecordSpec(y=REAL), {"y": jnp.stack([mu, mu + 2])}
        atoms = RecordBatch(values, "atom", label="atoms")
        event = OutputSpec(result=spec) if form == "named_record" else OutputSpec(spec)
        return EmpiricalDistribution(atoms, event_spec=event)

    kernel = conditional_distribution(build, given_spec={"mu": REAL}, label="kernel")
    with workflow_run(seed=1):
        law = mixture.using(max_evaluations=7)(kernel, _mixing() if finite else Gaussian("mu"))
        draw = sample(law)
    assert law.event_spec == kernel.event_spec
    assert draw.spec == kernel.event_spec.term_spec
    assert isinstance(law, MixtureDistribution if finite else EmpiricalDistribution)
    if not finite:
        assert law.num_atoms == 7


@pytest.mark.parametrize("finite", [False, True])
def test_unfed_required_inputs_are_refused(finite):
    kernel = conditional_distribution(
        lambda mu, sigma: Normal("y", mu, sigma), given_spec={"mu": REAL, "sigma": REAL}
    )
    mixing = _mixing() if finite else Gaussian("mu")
    with pytest.raises(ApplicabilityError, match=r"does not supply.*sigma"):
        mixture.check(kernel, mixing)
    with pytest.raises(ApplicabilityError, match=r"does not supply.*sigma"):
        mixture(kernel, mixing)


def test_generic_sampling_is_excluded_by_exact_only():
    with pytest.raises(ResolutionError):
        mixture.using(exact_only=True)(_kernel(), Gaussian("mu"))


def test_the_generic_route_can_be_selected_for_empirical_mixing():
    with workflow_run(seed=0):
        law = mixture.using(method="derived", max_evaluations=7)(_kernel(), _mixing())
    assert isinstance(law, EmpiricalDistribution)
    assert law.num_atoms == 7
    assert law.provenance.metadata["exact"] is False


def test_seeded_workflows_reproduce_generic_atoms():
    kernel, mixing = _kernel(), Gaussian("mu")
    configured = mixture.using(max_evaluations=13)
    with workflow_run(seed=3):
        first = configured(kernel, mixing)
    with workflow_run(seed=3):
        second = configured(kernel, mixing)
    np.testing.assert_array_equal(first.atoms.raw, second.atoms.raw)


def test_record_atoms_feed_inputs_by_name_and_ignore_extra_outputs():
    mixing = EmpiricalDistribution(
        RecordBatch(
            {"b": jnp.array([1.0, 4.0]), "unused": jnp.zeros(2), "a": jnp.array([2.0, -1.0])},
            "atom",
            label="atoms",
        ),
        jnp.array([0.25, 0.75]),
    )
    kernel = conditional_distribution(
        lambda a, b: Normal("y", a + 2 * b, 1.0), given_spec={"a": REAL, "b": REAL}
    )
    law = mixture(kernel, mixing)
    np.testing.assert_allclose(mean(law), 6.25, rtol=0, atol=1e-6)
    np.testing.assert_allclose(variance(law), 2.6875, rtol=2e-6, atol=1e-6)


def test_atom_axes_are_enumerated_in_row_major_order():
    mixing = EmpiricalDistribution(
        NumericArrayBatch(jnp.array([[-1.0, 0.0], [2.0, 3.0]]), ("row", "column"), label="atoms"),
        jnp.array([0.1, 0.2, 0.3, 0.4]),
        output_name="mu",
    )
    law = mixture(_kernel(), mixing)
    np.testing.assert_allclose(mean(law), 1.7, rtol=0, atol=1e-6)
    np.testing.assert_allclose(variance(law), 3.01, rtol=0, atol=1e-6)


@pytest.mark.parametrize("finite", [False, True])
def test_optional_inputs_take_their_defaults(finite):
    kernel = conditional_distribution(
        lambda mu, scale=2.0: Normal("y", mu, scale), given_spec={"mu": REAL}
    )
    with workflow_run(seed=0):
        law = mixture.using(max_evaluations=7)(kernel, _mixing() if finite else Gaussian("mu"))
    assert law.event_spec == kernel.event_spec
    if finite:
        np.testing.assert_allclose(variance(law), 7.0, rtol=0, atol=1e-6)


def test_mixing_overrides_an_optional_input_default():
    kernel = conditional_distribution(
        lambda mu, scale=2.0: Normal("y", mu, scale), given_spec={"mu": REAL}
    )
    mixing = EmpiricalDistribution(
        RecordBatch(
            {"mu": jnp.array([-1.0, 3.0]), "scale": jnp.array([1.0, 3.0])}, "atom", label="atoms"
        ),
        jnp.array([0.25, 0.75]),
    )
    law = mixture(kernel, mixing)
    np.testing.assert_allclose(mean(law), 2.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(variance(law), 10.0, rtol=0, atol=1e-6)


def test_batches_of_mixing_laws_sweep_the_operation():
    laws = DistributionBatch([_mixing(), _mixing()], "model", label="models")
    result = mixture(_kernel(), laws)
    assert isinstance(result, DistributionBatch)
    assert result.level_names == ("model",)
    assert all(isinstance(law, MixtureDistribution) for law in result)


@pytest.mark.parametrize("finite", [False, True])
def test_incompatible_matching_specs_are_rejected(finite):
    mixing = (
        EmpiricalDistribution(jnp.ones((2, 3)), output_name="mu")
        if finite
        else Normal("mu", jnp.ones(3), 1.0)
    )
    with pytest.raises(ValueError, match="shape"):
        mixture(_kernel(), mixing)


@pytest.mark.parametrize("finite", [False, True])
def test_duplicate_outputs_are_rejected(finite):
    mixing = (
        EmpiricalDistribution(
            RecordBatch({"mu": jnp.zeros(2), "y": jnp.ones(2)}, "atom", label="atoms")
        )
        if finite
        else Gaussian("mu") & Gaussian("y")
    )
    with pytest.raises(ValueError, match=r"produced|output|overlap"):
        mixture(_kernel(), mixing)


@pytest.mark.parametrize("weights", [[1.0], [0.0, 1.0], [1.0, 1.0]])
def test_single_zero_and_uniform_weights(weights):
    mixing = EmpiricalDistribution(
        jnp.arange(len(weights), dtype=jnp.float32), jnp.array(weights), output_name="mu"
    )
    law = mixture.using(exact_only=True)(_kernel(), mixing)
    normalized = np.array(weights) / sum(weights)
    expected_mean = normalized @ np.arange(len(weights))
    expected_variance = 1 + normalized @ (np.arange(len(weights)) - expected_mean) ** 2
    np.testing.assert_allclose(mean(law), expected_mean, rtol=0, atol=1e-6)
    np.testing.assert_allclose(variance(law), expected_variance, rtol=0, atol=1e-6)


def test_generic_samples_have_the_analytic_location_and_spread():
    with workflow_run(seed=0):
        law = mixture.using(max_evaluations=2048)(_kernel(), Normal("mu", 2.0, 3.0))
    # Seeds 0 to 3: mean errors <= 0.122; variance errors <= 0.582.
    np.testing.assert_allclose(mean(law), 2.0, rtol=0, atol=0.35)
    np.testing.assert_allclose(variance(law), 10.0, rtol=0, atol=1.5)


def test_finite_check_constructs_no_components(monkeypatch):
    kernel, mixing = _kernel(), _mixing()

    def fail(*args, **kwargs):
        pytest.fail("check must not bind the kernel or draw samples")

    monkeypatch.setattr(type(kernel), "_condition_on", fail)
    assert mixture.check(kernel, mixing).route == "finite"


def test_finite_mixing_records_approximate_conditioning():
    kernel, mixing = _ApproximateKernel(), _mixing()
    report = mixture.check(kernel, mixing)
    assert (report.route, report.exact) == ("finite", False)
    law = mixture(kernel, mixing)
    assert isinstance(law, MixtureDistribution)
    assert law.provenance.metadata["exact"] is False
    with pytest.raises(ResolutionError):
        mixture.using(exact_only=True)(kernel, mixing)


def test_a_generic_kernel_without_sampling_is_infeasible():
    kernel, mixing = Kernel(), Gaussian("mu")
    assert mixture.check(kernel, mixing).feasible is False
    with pytest.raises(ResolutionError, match="SupportsSampling"):
        mixture(kernel, mixing)
