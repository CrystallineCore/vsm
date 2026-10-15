# Versioned Semantic Memory: minimal prototype

A working, dependency-free prototype of an LLM agent memory store you can **roll back**, **selectively undo**, and **actually erase**.

- **Requirements:** Python 3.10+ and the standard library. No NumPy, no pip installs, no network, no model downloads.
- **Size:** about 630 lines of library code, plus about 1,200 lines of tests, demo and benchmark.

## Run it (one command)

```bash
python3 evaluate.py          # all checks, ~10 s, exits non-zero on any failure
```

Expected scorecard:

```
  PASS  Unit tests                   27 tests
  PASS  Differential vs reference    800,000 checks
  PASS  Mutation check               5 planted bugs
  PASS  Demo                         All 25 demo checks passed.
  PASS  Benchmark laws               horizon + erasure laws

  ALL CHECKS PASSED
```

Reference output from a full run is in `sample_output/`.

## What to show in an evaluation

| Command | What it shows | Time |
|---|---|---|
| `python3 demo.py --pause` | 7-part narrated walkthrough. Every claim printed with ✓ is also asserted. `--pause` waits for Enter between sections. | ~2 min live |
| `python3 shell.py --sample` | Interactive shell for live questions (`compare`, `history`, `revert`, `forget`, `find`, `compact`, `rollback`, …) | any |
| `python3 benchmark.py` | Backend comparison and retention-window sweep on a synthetic workload with exact ground truth | ~1 s |
| `python3 evaluate.py` | Everything above plus the test suites, with a pass/fail scorecard | ~10 s |

### Suggested live shell script

```
vsm> compare Which programming language is my favourite?     # naive RAG: 'C' (stale), layered: 'Rust'
vsm> history user::lang                                      # full audit trail
vsm> get user::lang @0                                       # time travel
vsm> put user::city Atlantis | the user now lives in Atlantis # poisoned memory
vsm> put user::pet cat | I have a cat                        # a later, good memory
vsm> revert 3                                                # undo ONLY the poison, keep the cat
vsm> delete user::pet
vsm> find I have a cat                                       # PRESENT: soft delete is undoable
vsm> forget user::pet
vsm> find I have a cat                                       # absent: physically erased
vsm> compact 2
vsm> rollback 3                                              # refused: horizon is 2
vsm> layers
```

## How it works

Memory is an ordered stack of **immutable layers**, the same structure as an LSM-tree storage engine:

```
[ compacted base ] [ stage k ] [ stage k+1 ] ... [ stage head ]     reads walk right -> left
```

| Operation | Behaviour | Cost |
|---|---|---|
| `ingest(events)` | Seal one new immutable layer (one session or batch = one stage) | O(events) |
| `get(id)` / `query(text)` | Newest version wins. A tombstone hides all older versions. A Bloom filter per layer lets keyed reads skip layers without reading them. | O(layers) |
| `get(id, as_of=s)` | Answer as of any stage still retained | O(layers) |
| `rollback(n)` | **Global undo:** drop the top *n* layers. Exact, with no undo log. | O(n) |
| `revert(stage)` | **Selective undo,** like `git revert`: writes a compensating layer that undoes one stage and keeps later changes. Anything changed again afterwards is reported as a conflict and left alone. | O(stage size × layers) |
| `delete(id)` | **Soft delete:** a tombstone. Hidden immediately, reversible by rollback, physically removed once it falls outside the retention window. | O(1) |
| `forget(id)` | **Hard erasure:** rewrites every layer holding the concept. It is gone from storage immediately and can never be brought back by rollback. The audit log stores only a hash. | O(layers) |
| `compact(W)` | Merge everything except the newest *W* layers into one base. Superseded versions and deleted data are dropped. The rollback horizon becomes exactly *W*. | O(records) |
| `snapshot()` | Free copy (layers are immutable and shared), used for "what if I rolled back?" | O(layers) |
| `save()` / `load()` | Atomic JSON persistence. The horizon survives a restart. | O(bytes) |

## Results (synthetic workload: 120 concepts, 12 stages, 5 seeds)

```
backend                  accuracy  resurf.  erased   old    rb@1    rb@2    rb@4    rb@8      KB
naive RAG                   0.171    1.000   0.000  0.00     n/a     n/a     n/a     n/a    33.9
mutable + soft delete       1.000    0.000   0.000  0.00     n/a     n/a     n/a     n/a    16.0
layered, no compaction      1.000    0.000   0.000  0.00   1.000   1.000   1.000   1.000    38.8
layered, compact W=4        1.000    0.000   0.220  1.00   1.000   1.000   1.000  beyond    21.2
```

- **accuracy:** top-1 answer is the *current* value.
- **resurf.:** a deleted concept is returned.
- **erased:** deleted text is absent from stored bytes. **old** counts only deletions made 4 or more stages ago.
- **rb@d:** exact-state fidelity after rolling back *d* stages.

**What this shows:**

1. Naive RAG finds the right concept but returns the wrong *version* (accuracy 0.17), and it resurfaces every deleted fact.
2. The mutable store is correct at HEAD but cannot undo anything, and its soft delete never erases.
3. The layered store matches the mutable store on accuracy and adds exact rollback.
4. With compaction it also uses less storage than keeping full history, and everything deleted before the window is physically gone.
5. **The retention window W is the single knob.** Exact rollback is possible if and only if depth ≤ W, and data deleted more than W stages ago is physically erased. The benchmark checks both laws automatically.

## How correctness is verified

1. **Unit tests** (`tests/test_engine.py`, 27 tests): reads, tombstones, rollback, revert and conflicts, forget, compaction, as-of, persistence, Bloom filter, baselines.
2. **Differential test** (`tests/test_differential.py`): runs random sequences of all 8 operations against an independent reference model that just replays a list of dicts. They must agree after every step, over 2,000 trials and 800k checks. It already caught one bug, in the reference model itself (`as_of=-1` after compaction).
3. **Mutation check** (`tests/mutation_check.py`): plants 5 realistic bugs (oldest-wins compaction, ignored deletes, tombstones not suppressing, rollback past the horizon, partial forget) and confirms the differential test catches every one. This shows the tests can fail.
4. **Demo assertions:** each of the 25 claims the demo prints is also checked in code.

## Honest limitations

- **Concept identity is supplied by the caller.** Mapping free text such as "I've switched to Rust" to `user::lang` is the hard, unsolved part of the idea. Here the benchmark uses the generator's exact IDs (an oracle), so the accuracy numbers describe the storage layer, not an end-to-end agent.
- **The embedder is lexical** (bag of words), not semantic. It is enough to exercise version resolution, but it does not measure retrieval quality. The engine doesn't depend on it, so a sentence-embedding model could be swapped in.
- **Erasure covers this store only.** Copies in LLM context, logs or backups are out of scope. (Crypto-shredding would be the next step for backups.)
- **Single process, in memory.** There is no concurrency control, WAL or crash recovery beyond atomic whole-file saves.
- **Linear scans.** `query` scores every live record. A real system would use an ANN index per layer.

## Layout

```
vsm_mini/
  engine.py      layered store: layers, tombstones, rollback, revert, forget, compaction, persistence
  baselines.py   naive RAG and mutable-with-soft-delete, same interface
  bloom.py       Bloom filter (no false negatives)
  embed.py       lexical bag-of-words embedder
  dataset.py     synthetic update-heavy workload with exact ground truth
tests/
  test_engine.py       unit tests (python3 -m unittest discover -s tests)
  test_differential.py engine vs independent reference model
  mutation_check.py    planted bugs must be caught
demo.py  shell.py  benchmark.py  evaluate.py
sample_output/         captured output of evaluate.py and demo.py
```
