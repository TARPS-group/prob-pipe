# Citing ProbPipe

If you use ProbPipe in your research, we would appreciate a citation.

## Citing the software

ProbPipe is archived on [Zenodo](https://zenodo.org/) under the
version-independent **concept DOI**
[10.5281/zenodo.20683559](https://doi.org/10.5281/zenodo.20683559), which
always resolves to the latest release. Cite the concept DOI unless you need
to pin a specific version (each release also gets its own version DOI).

```bibtex
@software{probpipe,
  author  = {Huggins, Jonathan and Roberts, Andrew and Zhu, Jiaqiang and
             Erozer, Can and Lim, Yongho},
  title   = {{ProbPipe}: Probabilistic pipelines with automated uncertainty quantification},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.20683559},
  url     = {https://github.com/TARPS-group/prob-pipe}
}
```

The repository's `CITATION.cff` carries the same metadata, so GitHub's
**"Cite this repository"** button (top-right of the repo page) produces an
up-to-date APA or BibTeX entry.

## Citing the inference backends

ProbPipe is an orchestration and conversion layer — the actual inference is
performed by established libraries. If your results depend on a particular
backend, please also cite it:

- **BlackJAX** (the default gradient-MCMC backend; NUTS, HMC, SGLD, SGHMC) —
  see the [BlackJAX citation guidance](https://blackjax-devs.github.io/blackjax/).
- **nutpie** (NUTS for Stan / PyMC models).
- **Stan** / **CmdStanPy** / **BridgeStan** (`StanModel` targets).
- **PyMC** (`PyMCModel` targets; ADVI).
- **TensorFlow Probability** (distribution implementations and TFP NUTS/HMC).
- **BayesFlow** (amortized simulation-based inference).

Cite the specific algorithm's paper where one exists (e.g., the NUTS paper
for NUTS sampling); the backends' own documentation lists the canonical
references.

## Citing method papers

As ProbPipe-specific methods are published, their references will be listed
here.
