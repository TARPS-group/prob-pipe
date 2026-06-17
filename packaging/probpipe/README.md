# probpipe (batteries)

This is the **batteries** distribution of ProbPipe: the friendly `probpipe`
name that pins the lean [`probpipe-core`](https://github.com/TARPS-group/prob-pipe)
base and bundles the inference backends the documentation exercises, so that

```bash
pip install probpipe
```

runs every example and tutorial out of the box — PyMC, nutpie, and BayesFlow on
Python 3.12–3.13 (on 3.14 the BayesFlow neural-SBI backend is omitted until
upstream lifts its `<3.14` cap).

It ships **no code** — the importable `probpipe` package lives in
`probpipe-core`. For a minimal install, depend on `probpipe-core` directly and
add backends as extras (`pip install "probpipe-core[pymc]"`, etc.). Every
user-facing extra is also re-exported here, so `pip install "probpipe[pymc]"`
works too.

See the [project README](https://github.com/TARPS-group/prob-pipe) and
[documentation](https://tarps-group.github.io/prob-pipe/) for details.
