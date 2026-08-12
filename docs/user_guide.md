# User Guide

The User Guide is a set of focused walkthroughs for specific ProbPipe features.
Use the [tutorials](tutorials/getting_started.ipynb) for a guided first pass, the
[API reference](api/index.md) for symbol-level details, and future examples or
case studies for applied, problem-first notebooks.

## Core Types

<div class="grid cards" markdown>

-   **[Distribution basics](user_guide/01_distributions.ipynb)**

    ---

    The core distribution ops, built-in distribution families, capability
    protocols, and `from_distribution` conversions.

-   **[Records and Record Distributions](user_guide/02_records.ipynb)**

    ---

    Structured values and random records: `Record`, `NumericRecord`,
    `RecordDistribution`, `RecordArray`, and `DistributionArray`.

</div>

## Workflow Mechanics

<div class="grid cards" markdown>

-   **[Broadcasting and Functions](user_guide/03_broadcasting.ipynb)**

    ---

    Automatic uncertainty propagation with `Function`, empirical
    enumeration, cartesian products, vectorized execution, and reproducible
    seeds.

-   **[Joint distributions](user_guide/04_joint_distributions.ipynb)**

    ---

    Product and sequential joints, Gaussian and empirical joint distributions,
    component views, `condition_on`, and flat-vector interop.

-   **[JAX interop](user_guide/08_jax_interop.ipynb)**

    ---

    Gradients through distributions, score functions, sensitivity analysis,
    maximum likelihood estimation, and variational inference.

</div>

## Inference and Representation

<div class="grid cards" markdown>

-   **[External backends](user_guide/05_external_backends.ipynb)**

    ---

    How `condition_on` dispatches to BlackJAX NUTS, Stan, PyMC, and nutpie,
    including method selection and the inference method registry.

-   **[Converting between representations](user_guide/06_converting_representations.ipynb)**

    ---

    Bijectors, transformed distributions, moment matching, and converter
    registry paths for satisfying distribution protocols.

-   **[Sequential updating](user_guide/07_sequential_updating.ipynb)**

    ---

    Batch-wise Bayesian updating with `IncrementalConditioner`, automatic KDE
    conversion, and provenance chains.

</div>

## Advanced Workflows

<div class="grid cards" markdown>

-   **[Bagged posteriors](user_guide/09_bagged_posteriors.ipynb)**

    ---

    Bootstrap replicate distributions, broadcasting `condition_on` over
    resampled data, and posterior spread diagnostics.

-   **[Random functions and Gaussian emulators](user_guide/10_random_functions_and_emulators.ipynb)**

    ---

    `RandomFunction`, Gaussian random functions, linear basis functions,
    algebraic operations, GP emulators, and synthetic-likelihood surrogates.

-   **[Scalability with Prefect](user_guide/11_prefect_scalability.ipynb)**

    ---

    Global Prefect configuration for distributing bagged posterior fits across
    every configured `Function`.

</div>
