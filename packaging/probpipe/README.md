# probpipe

The `probpipe` distribution pins the minimal
[`probpipe-core`](https://github.com/TARPS-group/prob-pipe) base and adds the
inference backends the documentation uses, so that

```bash
pip install probpipe
```

runs every example and tutorial out of the box — PyMC, nutpie, and BayesFlow on
Python 3.12–3.13 (on 3.14 the BayesFlow neural-SBI backend is omitted until
upstream lifts its `<3.14` cap).

It ships **no code** — the importable `probpipe` package lives in
`probpipe-core`. For a minimal install, depend on `probpipe-core` directly and
add backends as extras (`pip install "probpipe-core[pymc]"`, etc.). The extras
not already bundled here (`prefect`, `viz`, `stan`) are re-exported, so
`pip install "probpipe[stan]"` works too.

See the [project README](https://github.com/TARPS-group/prob-pipe) and
[documentation](https://tarps-group.github.io/prob-pipe/) for details.
