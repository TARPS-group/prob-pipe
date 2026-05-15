# Empirical and bootstrap

Finite-sample distributions backed by stored draws or by an underlying data
source. Each finite-sample class exposes `.n` (count of stored items — see
the convention in `STYLE_GUIDE.md` §1.9) and is non-iterable (use `.samples`
/ `.draws()` / `.components` for collection-style access).

::: probpipe.EmpiricalDistribution

::: probpipe.RecordEmpiricalDistribution

::: probpipe.KDEDistribution

::: probpipe.BootstrapDistribution

::: probpipe.BootstrapReplicateDistribution

::: probpipe.RecordBootstrapReplicateDistribution

::: probpipe.JointEmpirical

::: probpipe.NumericJointEmpirical
