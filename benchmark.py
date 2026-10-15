#!/usr/bin/env python3
"""Benchmark on a synthetic update-heavy workload with exact ground truth.

    python3 benchmark.py                 # 5 seeds, default sizes
    python3 benchmark.py --seeds 10 --entities 80 --stages 16

Table 1 compares backends. Table 2 sweeps the compaction retention window W
and shows that the exact rollback horizon equals W. Results -> results.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from vsm_mini import (LayeredStore, MutableStore, NaiveRAG, NotSupported, RollbackError)
from vsm_mini.dataset import generate

DEPTHS = (1, 2, 4, 8)


def build(kind: str, wl, window: int | None = None):
    store = {"naive": NaiveRAG, "mutable": MutableStore}.get(kind, LayeredStore)()
    for events in wl.stages:
        store.ingest(events)
        if window is not None:
            store.compact(window)
    return store


def deletion_age(wl, cid) -> int:
    """How many stages ago the concept was (last) deleted."""
    last = len(wl.truth) - 1
    for s in range(last, 0, -1):
        if cid in wl.truth[s - 1] and cid not in wl.truth[s]:
            return last - s
    return 0


def evaluate(store, wl, age: int = 4) -> dict:
    final = wl.truth[-1]
    ever = set(wl.queries)
    deleted = sorted(ever - set(final))

    t0 = time.perf_counter()
    correct = 0
    for cid, val in final.items():
        hits = store.query(wl.queries[cid], k=1)
        correct += bool(hits) and hits[0][1].concept_id == cid and hits[0][1].value == val
    resurfaced = 0
    for cid in deleted:
        hits = store.query(wl.queries[cid], k=1)
        resurfaced += bool(hits) and hits[0][1].concept_id == cid
    n_queries = len(final) + len(deleted)
    latency_us = (time.perf_counter() - t0) / max(1, n_queries) * 1e6

    raw = store.to_bytes()
    def gone(cid):
        return all(t.encode() not in raw for t in wl.texts[cid])
    erased = sum(gone(cid) for cid in deleted)
    old = [cid for cid in deleted if deletion_age(wl, cid) >= age]
    erased_old = sum(gone(cid) for cid in old)

    rollback = {}
    for d in DEPTHS:
        if d >= len(wl.truth):
            continue
        try:
            snap = store.snapshot() if hasattr(store, "snapshot") else store
            snap.rollback(d)
            expect = wl.truth[-1 - d]
            got = {cid: r.value for cid, r in snap.live().items()}
            keys = set(expect) | set(got)
            rollback[d] = sum(expect.get(k) == got.get(k) for k in keys) / len(keys)
        except NotSupported:
            rollback[d] = "n/a"
        except RollbackError:
            rollback[d] = "beyond"

    return {
        "accuracy": correct / max(1, len(final)),
        "resurfacing": resurfaced / max(1, len(deleted)),
        "physically_erased": erased / max(1, len(deleted)),
        "erased_old": erased_old / len(old) if old else float("nan"),
        "bytes": len(raw),
        "latency_us": latency_us,
        "rollback": rollback,
    }


def mean_of(runs: list[dict]) -> dict:
    out = {}
    for key in ("accuracy", "resurfacing", "physically_erased", "erased_old", "bytes",
                "latency_us"):
        vals = [r[key] for r in runs if r[key] == r[key]]   # skip NaN
        out[key] = statistics.fmean(vals) if vals else float("nan")
    out["rollback"] = {}
    for d in runs[0]["rollback"]:
        vals = [r["rollback"][d] for r in runs]
        out["rollback"][d] = (statistics.fmean(vals) if all(isinstance(v, float) for v in vals)
                              else max(set(v for v in vals if isinstance(v, str)), key=str))
    return out


def fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--entities", type=int, default=40)
    ap.add_argument("--stages", type=int, default=12)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    workloads = [generate(n_entities=args.entities, n_stages=args.stages, seed=s)
                 for s in range(args.seeds)]
    wl0 = workloads[0]
    n_deleted = len(set(wl0.queries) - set(wl0.truth[-1]))
    print(f"Workload: {len(wl0.queries)} concepts, {args.stages} stages, "
          f"~{n_deleted} deleted at the end, {args.seeds} seeds (means shown)\n")

    configs = [
        ("naive RAG", "naive", None),
        ("mutable + soft delete", "mutable", None),
        ("layered, no compaction", "layered", None),
        ("layered, compact W=4", "layered", 4),
    ]
    table1 = {}
    for label, kind, w in configs:
        table1[label] = mean_of([evaluate(build(kind, wl, w), wl) for wl in workloads])

    hdr = (f"{'backend':<24}{'accuracy':>9}{'resurf.':>9}{'erased':>8}{'old':>6}"
           + "".join(f"{'rb@' + str(d):>8}" for d in DEPTHS) + f"{'KB':>8}{'us/q':>8}")
    print("Table 1. Backends\n"
          "  accuracy = correct CURRENT value at top-1   resurf. = deleted concept returned\n"
          "  erased   = deleted data absent from stored bytes (all deletions / 'old' ="
          " deleted >= 4 stages ago)")
    print(hdr)
    print("-" * len(hdr))
    base_bytes = table1["layered, no compaction"]["bytes"]
    for label, r in table1.items():
        print(f"{label:<24}{r['accuracy']:>9.3f}{r['resurfacing']:>9.3f}{r['physically_erased']:>8.3f}{r['erased_old']:>6.2f}"
              + "".join(f"{fmt(r['rollback'].get(d, '-')):>8}" for d in DEPTHS)
              + f"{r['bytes'] / 1024:>8.1f}{r['latency_us']:>8.0f}")

    print("\nTable 2. Retention window sweep (layered store, compacted after every stage)")
    hdr2 = f"{'W':<6}" + "".join(f"{'rb@' + str(d):>8}" for d in DEPTHS) + \
           f"{'bytes':>9}{'accuracy':>10}{'resurf.':>9}{'erased>W':>10}"
    print(hdr2)
    print("-" * len(hdr2))
    table2 = {}
    for w in (1, 2, 4, 8, None):
        r = mean_of([evaluate(build("layered", wl, w), wl, age=w if w else 10**9)
                     for wl in workloads])
        table2["none" if w is None else w] = r
        print(f"{('none' if w is None else w):<6}"
              + "".join(f"{fmt(r['rollback'].get(d, '-')):>8}" for d in DEPTHS)
              + f"{r['bytes'] / base_bytes:>8.2f}x{r['accuracy']:>10.3f}{r['resurfacing']:>9.3f}{fmt(r['erased_old']) if r['erased_old'] == r['erased_old'] else '-':>10}")

    horizon_ok = all(
        (table2[w]["rollback"][d] == 1.0) if (w == "none" or d <= w) else
        (table2[w]["rollback"][d] == "beyond")
        for w in table2 for d in table2[w]["rollback"])
    print()
    erasure_ok = all(table2[w]["erased_old"] == 1.0 for w in table2 if w != "none"
                     and table2[w]["erased_old"] == table2[w]["erased_old"])
    print(f"Erasure law (data deleted more than W stages ago is physically gone): "
          f"{'HOLDS' if erasure_ok else 'VIOLATED'}")
    print(f"Horizon law (exact rollback possible iff depth <= W): "
          f"{'HOLDS' if horizon_ok else 'VIOLATED'}")
    print("Note: the embedder is lexical, and concept ids come from the generator (oracle).\n"
          "These numbers show storage-layer behaviour, not end-to-end semantic accuracy.")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "backends": table1,
                   "retention_sweep": {str(k): v for k, v in table2.items()},
                   "horizon_law_holds": horizon_ok, "erasure_law_holds": erasure_ok}, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0 if horizon_ok and erasure_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
