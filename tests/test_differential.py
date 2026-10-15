"""Differential test: the engine vs an independent, deliberately naive reference.

The reference keeps a plain list of per-stage dicts and answers every question
by replaying them from scratch. It shares no code with the engine. Random
sequences of ingest / delete / rollback / revert / compact / forget / save+load
are applied to both, and their answers must agree after every single step.

    python3 tests/test_differential.py [n_trials]
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsm_mini import (Event, HistoryUnavailable, LayeredStore, Op, RollbackError)

TOMB = "<TOMBSTONE>"


class Reference:
    def __init__(self):
        self.stages: list[dict[str, str]] = []   # index == stage id
        self.horizon = 0
        self.compacted = False

    def state(self, upto: int | None = None) -> dict[str, str]:
        s: dict[str, str] = {}
        for d in self.stages[:upto]:
            for k, v in d.items():
                if v == TOMB:
                    s.pop(k, None)
                else:
                    s[k] = v
        return s

    def ingest(self, d):
        self.stages.append(dict(d))
        self.horizon += 1

    def rollback(self, n) -> bool:
        if n > self.horizon:
            return False
        if n:
            del self.stages[-n:]
        self.horizon -= n
        return True

    def compact(self, w):
        n_layers = self.horizon + (1 if self.compacted else 0)
        if n_layers > w:
            self.horizon = w
            self.compacted = True

    def forget(self, cid):
        for d in self.stages:
            d.pop(cid, None)

    def revert(self, stage):
        """None = refused; else list of conflicts."""
        if not (len(self.stages) - self.horizon <= stage < len(self.stages)):
            return None
        target = self.stages[stage]
        before = self.state(stage)
        new, conflicts = {}, []
        for cid in target:
            newest = max(i for i, d in enumerate(self.stages) if cid in d)
            if newest != stage:
                conflicts.append(cid)
            else:
                new[cid] = before.get(cid, TOMB)
        if new:
            self.ingest(new)
        return conflicts


def engine_state(store, as_of=None):
    return {k: r.value for k, r in store.live(as_of=as_of).items()}


def run_trial(seed: int, n_ops: int = 40) -> int:
    rng = random.Random(seed)
    eng, ref = LayeredStore(), Reference()
    cids = [f"<c{i}>" for i in range(6)]
    counter = 0
    checks = 0
    ops = ["ingest"] * 9 + ["delete"] * 2 + ["rollback"] * 3 + ["compact"] * 2 + \
          ["forget", "revert", "revert", "persist", "asof", "asof"]

    for step in range(n_ops):
        op = rng.choice(ops)
        ctx = f"seed={seed} step={step} op={op}"
        if op == "ingest":
            d = {}
            events = []
            for cid in rng.sample(cids, rng.randint(0, 3)):
                if rng.random() < 0.15:
                    events.append(Event(cid, op=Op.DELETE)); d[cid] = TOMB
                else:
                    counter += 1
                    v = f"v{counter}"
                    events.append(Event(cid, f"text {cid} {v}", Op.UPDATE, v)); d[cid] = v
            eng.ingest(events); ref.ingest(d)
        elif op == "delete":
            cid = rng.choice(cids)
            eng.delete(cid); ref.ingest({cid: TOMB})
        elif op == "rollback":
            n = rng.randint(0, 4)
            ok = ref.rollback(n)
            try:
                eng.rollback(n)
                assert ok, f"{ctx}: engine rolled back {n} but reference refused"
            except RollbackError:
                assert not ok, f"{ctx}: engine refused rollback({n}) reference allowed"
        elif op == "compact":
            w = rng.randint(0, 4)
            eng.compact(w); ref.compact(w)
            for layer in eng.layers:
                if layer.compacted:
                    assert not any(r.tombstone for r in layer.records.values()), ctx
        elif op == "forget":
            cid = rng.choice(cids)
            eng.forget(cid); ref.forget(cid)
            raw = eng.to_bytes()
            assert f"text {cid} ".encode() not in raw and f'"{cid}"'.encode() not in raw, \
                f"{ctx}: forgotten data still physically present"
        elif op == "revert":
            stage = rng.randint(0, max(0, len(ref.stages) - 1))
            expected = ref.revert(stage)
            try:
                _, conflicts = eng.revert(stage)
                assert expected is not None, f"{ctx}: engine reverted, reference refused"
                assert sorted(conflicts) == sorted(expected), f"{ctx}: conflicts differ"
            except (HistoryUnavailable, ValueError):
                assert expected is None, f"{ctx}: engine refused revert({stage})"
        elif op == "persist":
            with tempfile.TemporaryDirectory() as tmp:
                p = os.path.join(tmp, "m.json")
                eng.save(p)
                reloaded = LayeredStore.load(p)
            assert reloaded.to_bytes() == eng.to_bytes(), f"{ctx}: persistence not lossless"
            eng = reloaded
        elif op == "asof":
            s = rng.randint(-1, len(ref.stages) - 1)
            boundary = len(ref.stages) - ref.horizon - 1   # hi of compacted layer
            should_work = (not ref.compacted) or s >= boundary or s == -1
            try:
                got = engine_state(eng, as_of=s)
                assert should_work, f"{ctx}: as_of={s} answered from compacted history"
                assert got == ref.state(s + 1), f"{ctx}: as_of={s} mismatch"
            except HistoryUnavailable:
                assert not should_work, f"{ctx}: as_of={s} wrongly refused"

        # ---- invariants after every step
        assert engine_state(eng) == ref.state(), f"{ctx}: state mismatch"
        assert eng.horizon() == ref.horizon, f"{ctx}: horizon {eng.horizon()} != {ref.horizon}"
        assert eng.head_stage == len(ref.stages) - 1, f"{ctx}: head stage mismatch"
        for cid in cids:
            r = eng.get(cid)
            assert (r.value if r else None) == ref.state().get(cid), f"{ctx}: get({cid})"
        checks += 4 + len(cids)
    return checks


class TestDifferential(unittest.TestCase):
    def test_engine_matches_reference(self):
        n = int(os.environ.get("VSM_TRIALS", "500"))
        total = sum(run_trial(seed) for seed in range(n))
        self.assertGreater(total, 0)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    total = sum(run_trial(seed) for seed in range(n))
    print(f"Differential test: {n} random trials, {total} step-level checks, 0 mismatches.")
