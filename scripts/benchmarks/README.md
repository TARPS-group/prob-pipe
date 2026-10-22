# Execution benchmarks

These manual benchmarks measure ProbPipe execution costs. Statistical model evaluation is outside their scope.
Run them from a repository checkout with its dependencies installed.

```bash
uv run python -m scripts.benchmarks list
uv run python -m scripts.benchmarks run
uv run python -m scripts.benchmarks run --group family --group inference
uv run python -m scripts.benchmarks run --suite full --device cuda
uv run python -m scripts.benchmarks compare before.jsonl after.jsonl
```

The runner uses its own Python interpreter for each child process. On macOS,
select `--device cpu` (the default). CUDA requires a CUDA-enabled environment;
invoke that environment's Python with `-m scripts.benchmarks`. The runner does
not install dependencies, support Metal, or fall back to CPU when CUDA fails.

`--case` selects an exact name printed by `list`; it and `--group` may repeat.
`--scenario` selects `same_object`, `new_object`, or `structure_change` and may
repeat. Omitting it runs all three. The first call establishes the original
structure. Later calls reuse the owner, construct a new owner with changed
parameter values, or alternate between two sizes with a retained owner for
each size. Thus structure-change samples include both new and reused shapes;
the raw records identify their sizes. New-object preparation occurs before
timing, so those call times exclude reconstruction costs.

Each case, scenario, and process repetition runs in a fresh subprocess,
serially. Persistent compilation caching is disabled. Setup initializes the
device, prepares keys and inputs, and constructs owners without executing the
measured operation. `setup_ms` records total workload preparation after device
initialization; `owner_preparation_ms` records each owner's construction and
input preparation within it. Neither includes interpreter startup or imports.

Every timed call includes output extraction and device synchronization. Public
operations include the following costs:

- Workflow context: entry, exit and internal key handling.
- Inputs: value adjustment within the workload.
- Outputs: assembling the returned result.
 External key preparation and input
placement are excluded. Minibatch index calls use prepared keys directly;
gradient scans include splitting keys over 25 steps and report the whole scan's
time, not per-call latency. The original permutation remains an algorithm
reference, separately named from the current sampler. Full CPU permutation
scans can be expensive; select cases when investigating a narrower question.

The quick suite uses one process per case/scenario, two warmups and five
measured calls. The full suite uses three processes, three warmups and twenty
measured calls. Override these with `--processes`, `--warmups` and `--repeats`.
First-call and warmup observations remain in the output. Each process uses the
same reproducible seed sequence, controlled by `--seed`. Function lifting
covers sequential and JAX executors; inference fixes RWMH proposal settings,
chain counts and iteration counts with adaptation disabled.

Results default to `.benchmark-results/`, ignored by Git. Use `--output PATH`
to create another file, or `--output -` for JSONL on stdout. Existing files are
never overwritten. Logs go to stderr. Results are local artifacts: do not
commit or upload them. Historical measurements are stored under
`.benchmark-results/legacy/` in the checkout where they were migrated.

Each JSONL row contains:

- Identity: schema and case versions, parameters and scenario.
- Reproduction settings: process index, seed, commit, dirty state and environment.
- Measurements: preparation timings, raw samples, median and quartile summaries.

Each sample identifies its phase and size. It records elapsed milliseconds and
the compilation-event count. Failures carry
an error and cause a nonzero exit status; they have no valid timing summary.
The existing compilation counter observes JAX's backend compile-duration event.
It would also count persistent-cache loads, which these runs disable.

For a commit comparison, run the same commands in both checkouts using the same
machine and dependency versions, then pass the two result files to `compare`.
The tool does not switch branches. It aligns cases and scenarios, pools process
repetitions, and reports per-phase medians, quartiles, time ratios (after/before)
and compilation events per call. Different environments, workload versions,
seeds, repeat settings or timing policies are marked incompatible. Missing and
failed cases are reported explicitly. A dirty checkout is recorded; its commit
alone does not identify the measured code. Timings are observations, not CI
thresholds or claims that one hardware platform is faster than another.

The conditional joint uses response `a` and coefficient `beta`. Its declared
field order agrees with JAX's dictionary order. The current execution layer can
reorder record fields during mapped density calls when those orders differ;
this benchmark does not modify that execution behavior.
