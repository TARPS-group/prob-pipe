"""Simulated datasets at the real design, drawn from the pipeline's own model.

The smoke stage fits one of these datasets in place of the real data, and the
validation fits many of them, so both exercise the model that the fit to the
real data uses, at the real covariates. A model whose data have missing values
masks each simulated dataset here to the real pattern of missing values, by
the CAPABILITIES.md row "Validate the computation on realistic simulated data".
"""

from __future__ import annotations

import numpy as np

from probpipe import RecordBatch, sample, workflow_run


def simulate(
    model, n: int, parameters: tuple[str, ...], response: str, seed: int
) -> tuple[RecordBatch, np.ndarray]:
    """`n` datasets drawn from `model`, and the parameter values that generated them.

    The datasets are a `RecordBatch` on the level `sim`, which `condition_on`
    fits in one sweep. The true values are an array of shape (sim, coordinate),
    ordered by the fields of the parameters like the posterior summaries.
    """
    with workflow_run(seed=seed):
        sims = sample(model, sample_shape=(n,)).with_level_names(sample="sim")
    return sims.without(*parameters), np.asarray(sims.without(response).to_vector())
