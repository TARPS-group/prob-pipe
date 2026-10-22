# Minibatch sampling benchmark

Measured on 2026-10-22 against the population-permutation implementation at
`d6c76000dacfd66e27b4f981219f605577b42f53`.

## Algorithm and contract

Without-replacement sampling retains the first occurrences of independent
integer candidates in their original order. If too few distinct values remain,
another batch supplements the retained values. Sorting identifies duplicates;
the selected indices retain their order of first occurrence.

For a requested batch of size `b`, sample `s = min(b, N - b)` distinct indices.
For `b > N / 2`, these indices identify excluded observations. Extract their
complement and independently permute it to preserve uniform random order.
A full batch uses a population permutation; a single requested or excluded
index uses one integer draw. Sampling with replacement is unchanged.

For `s >= 2`, the candidate count is the static integer
`max(s + 16, ceil(-1.25 * N * log1p(-s / N)))`. Under independent uniform draws,
the expected number of draws needed to collect `s` distinct values is
`N * (H_N - H_(N-s))`, where `H` is the harmonic sum.
The logarithm approximates that expectation; the surplus reduces supplementation.
Since `s <= N / 2`, candidate storage is proportional to `s`.
Complement construction and output shuffling require storage proportional to `N`.
The number of supplementation rounds depends on the key and has no fixed cap.

Relabeling the population leaves the selection rule unchanged, so every ordered
size-`s` sample has the same probability under uniform integer draws. Taking a
uniform excluded subset gives a uniform included subset. The independent
permutation makes its order uniform. These arguments inherit the integer
generator's finite-precision properties.

The implementation uses the same algorithm under eager execution, JIT, and on
CPU and GPU. Public signatures and result shapes are preserved. Indices stay
in range and never repeat within a batch sampled without replacement.
Precise seeded indices and subsequent training trajectories can change from
the baseline. Repeating a key with the same arguments and environment reproduces
the result. No public algorithm parameter or ABI version change is introduced.

## Reproduction

Run each device separately from the repository root:

```bash
.venv-gpu/bin/python -m scripts.benchmarks.minibatch_indices --device cpu --repeats 10
.venv-gpu/bin/python -m scripts.benchmarks.minibatch_indices --device gpu --repeats 10
```

Both measurements used Python 3.12.11 and JAX 0.10.1 with 64-bit mode disabled.
Hardware was an Intel Core Ultra 7 255H with 12 CPUs visible to WSL and an
NVIDIA GeForce RTX 5070 Laptop GPU. The CPU run used the same virtual environment
as the GPU run. Other verification processes finished before timing began.

Each measurement synchronizes device results. Initial key generation, model
construction, and device placement precede timing. Ten distinct keys are used
for warm measurements. First-call time includes tracing, compilation, and
execution; an enabled persistent cache may supply the executable. The mapped
measurements use 64 keys per call. Gradient scans split keys internally and
consume all 25 results in a checksum; their timings include those splits.
The benchmark uses the actual minibatch gradient estimator with only the index
sampler replaced for the reference. It does not time an inference runner.

Raw measurements include first calls, warm minima and maxima, and compilation
counts: [CPU](minibatch_cpu.jsonl), [GPU](minibatch_gpu.jsonl).
Times are machine-dependent observations and are not blocking test thresholds.

## Warm index sampling

Each entry is the median synchronized call time in milliseconds.

| N | Batch | CPU permutation | CPU current | GPU permutation | GPU current |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 32 | 0.118 | 0.159 | 1.307 | 0.469 |
| 1,000 | 256 | 0.133 | 0.259 | 0.972 | 0.263 |
| 10,000 | 32 | 3.189 | 0.134 | 1.426 | 0.299 |
| 10,000 | 256 | 3.054 | 0.198 | 1.490 | 0.494 |
| 100,000 | 32 | 33.618 | 0.096 | 1.465 | 0.292 |
| 100,000 | 256 | 33.408 | 0.062 | 1.531 | 0.243 |
| 1,000,000 | 32 | 393.789 | 0.109 | 1.497 | 0.267 |
| 1,000,000 | 256 | 375.592 | 0.086 | 1.588 | 0.270 |
| 65,536 | 256 | 20.728 | 0.229 | 1.441 | 0.247 |
| 66,049 | 257 | 21.324 | 0.216 | 1.468 | 0.210 |
| 1,000,000 | 4,096 | 377.783 | 1.143 | 2.654 | 0.546 |
| 10,000 | 5,000 | 3.322 | 1.744 | 1.414 | 0.815 |
| 10,000 | 5,001 | 2.981 | 3.205 | 1.486 | 2.070 |
| 1,000,000 | 900,000 | 381.643 | 368.390 | 4.927 | 3.303 |
| 1,000,000 | 999,968 | 384.628 | 383.405 | 4.924 | 4.182 |
| 1,024 | 1,024 | 0.183 | 0.141 | 1.369 | 1.409 |

The current algorithm is slower for the two small CPU populations and for the
batch just above half the population on both devices. Dense batches still pay
for materializing and shuffling the output. These results do not establish a
speedup for every size. Dispatch and device scheduling affect short individual
calls, so the scan measurements below provide a separate execution context.

## First calls and compilation

For `N=1,000,000`, `batch_size=256`, index first calls took:

| Device | Permutation, ms | Current, ms |
|---|---:|---:|
| CPU | 494.787 | 137.036 |
| GPU | 335.289 | 561.566 |

Every measured configuration compiled once on its first call and zero times
across its warm calls. This includes the gradient estimator and mapped calls.
The GPU first-call cost increased in this example despite faster warm calls.

## Mapped sampling and actual gradients

Mapped sampling times cover all 64 lanes, in milliseconds:

| N | Batch | CPU permutation | CPU current | GPU permutation | GPU current |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 256 | 157.986 | 2.997 | 1.502 | 0.836 |
| 65,536 | 256 | 1,269.073 | 2.430 | 4.938 | 0.731 |

Actual logistic minibatch gradients for `N=1,000,000`, per synchronized call:

| Batch | CPU permutation | CPU current | GPU permutation | GPU current |
|---:|---:|---:|---:|---:|
| 32 | 384.084 | 0.243 | 2.799 | 1.116 |
| 256 | 387.457 | 0.469 | 1.526 | 1.173 |

For a compiled GPU loop of 25 gradients, the median time divided by 25 was
0.537 ms versus 0.302 ms at batch 32, and 0.567 ms versus 0.293 ms at batch 256,
for permutation and current sampling respectively. The loop holds parameters
fixed and changes the key every step; it measures gradient evaluation rather
than a complete training or inference update.

## Correctness and repository checks

The 65 sampler tests passed on CPU and CUDA. They cover:

- Both key formats and the index shape, range, and uniqueness contracts.
- Eager/JIT equality and mapped/individual equality.
- The half-population boundary, complements, and full batches.
- Warm compilation reuse with new keys.

A forced-collision case verifies that supplementation retains earlier values.
The memory test checks that small batches allocate no population array.

Across seeds 0-3 on both devices, 40,000 draws per seed were compared with
enumerated ordered samples and subsets. Cases `(N,b)=(5,2),(5,3),(4,4)` cover
direct sampling, complements, and full batches. The maximum frequency error
was 2.84 standard deviations; the test bound is seven.

Actual logistic gradient means and covariances were compared with the analytic
finite-population formula at `N=128`, `b=8,64,120`, using 8,192 draws per seed.
Maximum mean errors ranged from 0.20 to 2.04 standard errors, and maximum
normalized covariance errors ranged from 0.007 to 0.040. Bounds are six and
0.08 respectively. [Calibration measurements](minibatch_calibration.jsonl)
record every seed and device.

Import-graph targets passed on the base with 580 passes and 8 skips. The same
targets plus 63 initial sampler tests and 189 documentation tests passed with
832 passes and the same 8 skips. Two subsequently added mapping tests passed
on both devices, bringing the sampler count to 65. There were no new skips or
xfails. The strict docs build, Ruff lint and format, and changed-file docstring
check passed. The full-tree docstring check reports the unchanged
`probpipe/core/_specs.py:704` argument and return documentation defects.
The full repository test suite was not run; no PR was opened.
