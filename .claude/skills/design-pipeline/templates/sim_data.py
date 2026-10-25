"""Simulated datasets at the real design, drawn from the pipeline's own model.

The smoke stage fits one of these datasets in place of the real data, and the
validation fits many of them, so both exercise the model that the fit to the
real data uses, at the real covariates. A model whose data have missing values
masks each simulated dataset here to the real pattern of missing values, by
the CAPABILITIES.md row "Validate the computation on realistic simulated data".
"""

from __future__ import annotations

from probpipe import RecordBatch, sample


def simulate(model, n: int, response: str) -> tuple[RecordBatch, RecordBatch]:
    """`n` datasets drawn from the joint model, and the parameter values that generated them.

    Both are record batches on the level `sim`, which `condition_on` fits in
    one sweep. The draw takes its key from the enclosing `workflow_run`.
    """
    sims = sample(model, sample_shape=(n,)).with_level_names(sample="sim")
    parameters = [name for name in model.event_spec.outputs if name != response]
    return sims.without(*parameters), sims.without(response)
