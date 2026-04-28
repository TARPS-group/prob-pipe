# Plan: Fix Pickle Serialization for Ray Compatibility

## Motivation

ProbPipe's `Record` class (and its subclasses) enforces immutability via
a `__setattr__` guard that raises `AttributeError("Record is immutable")`
on any attribute assignment. This breaks pickle deserialization because
pickle's default restore mechanism calls `__setattr__` to rebuild the
object's state.

This was discovered while testing Prefect + Ray distribution of bagged
posteriors. When `RayTaskRunner` dispatches a task, Ray serializes all
arguments with cloudpickle and sends them to worker processes. Any
argument containing a `Record` (e.g., the bootstrap dataset) fails
deserialization.

**Error observed:**
```
ray.exceptions.RaySystemError: System error: Record is immutable
  File ".../ray/_private/serialization.py", line 290, in _deserialize_pickle5_data
    obj = pickle.loads(in_band, buffers=buffers)
  File ".../probpipe/core/record.py", line 226, in __setattr__
    raise AttributeError("Record is immutable")
```

This blocks the primary use case from the Prefect integration PR: using
Ray for true parallelism of bagged posterior MCMC fits across CPU cores
or machines.

## Affected Classes

### Must fix (immutability guard blocks pickle)

| Class | File | Issue |
|-------|------|-------|
| `Record` | `probpipe/core/record.py` | `__setattr__` guard + `__slots__` |
| `RecordTemplate` | `probpipe/core/record.py` | Same pattern |
| `NumericRecord` | `probpipe/core/_numeric_record.py` | Inherits `Record`'s guard, adds `_flat_size` slot |
| `RecordArray` | `probpipe/core/_record_array.py` | Inherits guard, adds `_batch_shape`, `_template` slots |
| `NumericRecordArray` | `probpipe/core/_record_array.py` | Inherits `RecordArray`'s guard |

### Should verify (no `__setattr__` guard, but used in Prefect tasks)

| Class | File | Notes |
|-------|------|-------|
| `EmpiricalDistribution` | `probpipe/core/_empirical.py` | No slots, uses `__dict__` — likely pickles fine, but should verify |
| `BootstrapReplicateDistribution` | `probpipe/core/_empirical.py` | Same — contains `Record` instances which will fail |
| `Distribution` base | `probpipe/core/_distribution_base.py` | No slots — likely fine |

## Root Cause

All five "must fix" classes share the same pattern:

```python
class Record:
    __slots__ = ("_store", "_name", "_source")

    def __init__(self, *, name=None, **fields):
        object.__setattr__(self, "_store", OrderedDict(...))
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_source", None)

    def __setattr__(self, name, value):
        raise AttributeError("Record is immutable")
```

During `__init__`, they bypass the guard with `object.__setattr__()`.
But pickle doesn't call `__init__` — it creates an empty instance and
tries to restore state via `__setattr__`, which hits the guard.

## Proposed Solution: `__reduce__` Method

The cleanest fix is `__reduce__`, which tells pickle exactly how to
reconstruct the object. For immutable objects whose state is fully
determined by their constructor arguments, this is straightforward:

```python
class Record:
    def __reduce__(self):
        # Reconstruct via Record(name=..., **fields)
        return (
            _unpickle_record,
            (dict(self._store), self._name),
        )

def _unpickle_record(store, name):
    """Reconstruct a Record from its pickled state."""
    return Record(name=name, **store)
```

For classes with additional cached state (e.g., `NumericRecord._flat_size`),
the reconstruction function recomputes caches automatically because it
calls `__init__`.

### Why `__reduce__` over `__getstate__`/`__setstate__`

- `__getstate__`/`__setstate__` still requires `__setstate__` to bypass
  the immutability guard with `object.__setattr__()` — more code, same
  effect.
- `__reduce__` delegates to existing constructors, so any validation
  or caching in `__init__` is automatically applied.
- `__reduce__` is the standard approach for immutable objects (used by
  `namedtuple`, frozen `dataclass`, etc.).

## Design Per Class

### `Record`

```python
def __reduce__(self):
    return (_unpickle_record, (dict(self._store), self._name))

def _unpickle_record(store, name):
    rec = Record(name=name, **store)
    return rec
```

Note: `_source` (provenance) is intentionally not preserved through
serialization — provenance is write-once and is set after construction.
Distributed tasks will have their own provenance context.

### `RecordTemplate`

```python
def __reduce__(self):
    return (_unpickle_record_template, (dict(self._specs),))

def _unpickle_record_template(specs):
    return RecordTemplate(**specs)
```

### `NumericRecord`

Inherits from `Record`. The `_flat_size` cache is recomputed in
`__init__`, so we just need to reconstruct with the same fields:

```python
def __reduce__(self):
    return (_unpickle_numeric_record, (dict(self._store), self._name))

def _unpickle_numeric_record(store, name):
    return NumericRecord(name=name, **store)
```

### `RecordArray`

```python
def __reduce__(self):
    return (
        _unpickle_record_array,
        (dict(self._store), self._batch_shape, self._name),
    )

def _unpickle_record_array(store, batch_shape, name):
    return RecordArray(batch_shape=batch_shape, name=name, **store)
```

### `NumericRecordArray`

Same pattern as `RecordArray` — validation is re-run in `__init__`.

## Action Plan

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 1 | Add `__reduce__` to `Record` | `probpipe/core/record.py` | Serialize `_store` and `_name`; `_source` is not preserved |
| 2 | Add `__reduce__` to `RecordTemplate` | `probpipe/core/record.py` | Serialize `_specs` |
| 3 | Add `__reduce__` to `NumericRecord` | `probpipe/core/_numeric_record.py` | Delegates to `NumericRecord(...)` constructor |
| 4 | Add `__reduce__` to `RecordArray` | `probpipe/core/_record_array.py` | Serialize `_store`, `_batch_shape`, `_name` |
| 5 | Add `__reduce__` to `NumericRecordArray` | `probpipe/core/_record_array.py` | Same as `RecordArray` |
| 6 | Write pickle round-trip tests | `tests/test_record_serialization.py` (new) | See test plan |
| 7 | Verify `EmpiricalDistribution` pickles correctly | `tests/test_record_serialization.py` | May already work — verify |
| 8 | Verify `BootstrapReplicateDistribution` pickles correctly | `tests/test_record_serialization.py` | Contains `Record` — will work once Record is fixed |
| 9 | End-to-end: run bagged posterior with `RayTaskRunner` | manual test | Verify the full pipeline works with Ray |
| 10 | Remove `ThreadPoolTaskRunner` workaround from `run_prefect_demo.py` | `docs/examples/run_prefect_demo.py` | Let auto-detection pick up Ray |

## Test Plan

### New tests (`tests/test_record_serialization.py`)

| # | Test | What it verifies |
|---|------|-----------------|
| 1 | `test_record_pickle_roundtrip` | `pickle.loads(pickle.dumps(record))` preserves fields and name |
| 2 | `test_record_template_pickle_roundtrip` | Same for `RecordTemplate` |
| 3 | `test_numeric_record_pickle_roundtrip` | Same for `NumericRecord`, including `_flat_size` recomputation |
| 4 | `test_record_array_pickle_roundtrip` | Same for `RecordArray`, including `_batch_shape` |
| 5 | `test_numeric_record_array_pickle_roundtrip` | Same for `NumericRecordArray` |
| 6 | `test_record_cloudpickle_roundtrip` | Same tests with `cloudpickle` (what Ray uses) |
| 7 | `test_empirical_distribution_pickle` | Verify `EmpiricalDistribution` with Record samples pickles correctly |
| 8 | `test_bootstrap_replicate_pickle` | Verify `BootstrapReplicateDistribution` with Record data pickles correctly |
| 9 | `test_provenance_not_preserved` | Verify `_source` is `None` after unpickling (intentional) |
| 10 | `test_record_immutability_after_unpickle` | Unpickled Record still raises on `__setattr__` |

## Backward Compatibility

**No breaking changes.** Adding `__reduce__` methods only affects pickle
behavior, which was previously broken (raised `AttributeError`). All
existing code that doesn't pickle Records is unaffected.

## Relationship to Prefect Integration

This is a follow-up to the global Prefect config PR. Once merged:

1. `_auto_detect_task_runner()` will return `RayTaskRunner` when
   `prefect-ray` is installed.
2. Tasks dispatched via `task.map()` will serialize arguments with
   cloudpickle — which now works because `Record.__reduce__` handles it.
3. The `run_prefect_demo.py` script can drop the `ThreadPoolTaskRunner`
   workaround and use auto-detected Ray.

## Open Questions

1. **Should `_source` (provenance) survive pickling?** Currently
   proposed: no. Provenance is write-once and typically set by the
   operation that creates the object. Distributed tasks create their
   own provenance. If needed, we can add it later by including
   `self._source` in the `__reduce__` tuple.

2. **Should we also add `__copy__` / `__deepcopy__`?** The
   `Distribution.renamed()` method already uses `copy.copy()` with
   manual `object.__setattr__()`. Records might benefit from the same
   pattern. Proposed: defer unless needed.

3. **Should we test with Ray directly?** The pickle round-trip tests
   cover the serialization layer. A full Ray end-to-end test requires
   Ray to be installed and a cluster running. Proposed: manual
   verification with `run_prefect_demo.py`, not an automated test.
