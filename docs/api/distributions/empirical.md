# Empirical and bootstrap

Finite-sample distributions backed by stored draws or by an underlying
data source. `num_atoms` reports the count of stored atoms (point
masses); the bootstrap-replicate family instead exposes
`replicate_size` (items per resampled replicate). `.samples`,
`.draws()`, and `.components` access the stored items.

::: probpipe.EmpiricalDistribution

::: probpipe.RecordEmpiricalDistribution

::: probpipe.KDEDistribution

::: probpipe.BootstrapDistribution

::: probpipe.BootstrapReplicateDistribution

::: probpipe.RecordBootstrapReplicateDistribution

::: probpipe.JointEmpirical

::: probpipe.NumericJointEmpirical
