# Empirical and bootstrap

Finite-sample distributions backed by stored draws or by an underlying
data source. `num_atoms` (or `num_observations` for the bootstrap-replicate
family — see [STYLE_GUIDE §1.9](../../STYLE_GUIDE.md)) reports the count;
`.samples`, `.draws()`, and `.components` access the stored items.

::: probpipe.EmpiricalDistribution

::: probpipe.RecordEmpiricalDistribution

::: probpipe.KDEDistribution

::: probpipe.BootstrapDistribution

::: probpipe.BootstrapReplicateDistribution

::: probpipe.RecordBootstrapReplicateDistribution

::: probpipe.JointEmpirical

::: probpipe.NumericJointEmpirical
