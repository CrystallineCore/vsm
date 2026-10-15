"""The layered, versioned memory engine.

Memory is an ordered stack of immutable layers (LSM-tree style):

    [compacted base] [stage k] [stage k+1] ... [stage head]      <- newest wins

* ingest()   seals one new immutable layer (one "stage" = one session / batch)
* get/query  walk newest -> oldest; the first record seen for a concept wins,
             and a tombstone hides every older version of that concept
* rollback() drops the top n layers: exact undo, no undo log needed
* revert()   selective undo of ONE past stage, keeping later changes
* delete()   soft, reversible delete (tombstone); physically removed at compaction
* forget()   hard erasure: physically removed from every layer immediately,
             and never resurrected by rollback
* compact()  merges all but the newest W layers into one base layer, discarding
             superseded versions and tombstoned data. W = the rollback horizon.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Iterator

from .bloom import BloomFilter
from .embed import cosine, embed


class Op(str, Enum):
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"


@dataclass(frozen=True)
class Event:
    """An incoming fact. The caller supplies the concept_id (see README: this is
    the hard, unsolved part of the idea; the prototype takes it as given)."""
    concept_id: str
    text: str = ""
    op: Op = Op.INSERT
    value: str | None = None


@dataclass(frozen=True)
class Record:
    concept_id: str
    text: str
    value: str | None
    stage: int
    tombstone: bool = False


class NotSupported(Exception):
    """Raised by backends that structurally cannot perform an operation."""


class RollbackError(Exception):
    """Requested undo goes further back than the retained history."""


class HistoryUnavailable(Exception):
    """The requested stage was merged away by compaction."""


class Layer:
    """One immutable change-set. Never modified after construction."""
    __slots__ = ("lo", "hi", "kind", "note", "records", "bloom", "vectors")

    def __init__(self, lo: int, hi: int, records: dict[str, Record],
                 kind: str = "stage", note: str = ""):
        self.lo, self.hi, self.kind, self.note = lo, hi, kind, note
        self.records = MappingProxyType(dict(records))
        self.bloom = BloomFilter(len(records))
        for cid in records:
            self.bloom.add(cid)
        self.vectors = MappingProxyType(
            {cid: embed(r.text) for cid, r in records.items() if not r.tombstone})

    @property
    def compacted(self) -> bool:
        return self.kind == "compacted"

    def without(self, cid: str) -> "Layer":
        return Layer(self.lo, self.hi,
                     {k: v for k, v in self.records.items() if k != cid},
                     self.kind, self.note)

    def label(self) -> str:
        return f"stage {self.lo}" if self.lo == self.hi else f"stages {self.lo}-{self.hi}"

    def to_dict(self) -> dict:
        return {"lo": self.lo, "hi": self.hi, "kind": self.kind, "note": self.note,
                "records": [[r.concept_id, r.text, r.value, r.stage, r.tombstone]
                            for r in self.records.values()]}

    @classmethod
    def from_dict(cls, d: dict) -> "Layer":
        recs = {row[0]: Record(*row) for row in d["records"]}
        return cls(d["lo"], d["hi"], recs, d["kind"], d.get("note", ""))


class LayeredStore:
    name = "layered (this project)"

    def __init__(self) -> None:
        self.layers: list[Layer] = []
        self.erasure_log: list[dict] = []
        self.counters = {"layers_visited": 0, "bloom_skips": 0}

    # ------------------------------------------------------------------ writes
    @property
    def head_stage(self) -> int:
        return self.layers[-1].hi if self.layers else -1

    def ingest(self, events: Iterable[Event], note: str = "") -> int:
        """Seal one immutable layer containing these events. Returns its stage id."""
        stage = self.head_stage + 1
        recs: dict[str, Record] = {}
        for e in events:
            if not e.concept_id:
                raise ValueError("event needs a concept_id")
            tomb = Op(e.op) is Op.DELETE
            recs[e.concept_id] = Record(e.concept_id, "" if tomb else e.text,
                                        None if tomb else e.value, stage, tomb)
        self.layers.append(Layer(stage, stage, recs, "stage", note))
        return stage

    def put(self, concept_id: str, value: str, text: str | None = None, note: str = "") -> int:
        return self.ingest([Event(concept_id, text or f"{concept_id} is {value}",
                                  Op.UPDATE, value)], note)

    def delete(self, concept_id: str, note: str = "") -> int:
        """Soft delete: hidden immediately, reversible by rollback until compacted."""
        return self.ingest([Event(concept_id, op=Op.DELETE)], note or f"delete {concept_id}")

    # ------------------------------------------------------------------- reads
    def _visible(self, as_of: int | None) -> Iterator[Layer]:
        for layer in reversed(self.layers):
            if as_of is not None:
                if layer.lo > as_of:
                    continue
                if layer.hi > as_of:
                    raise HistoryUnavailable(
                        f"stage {as_of} was merged into the compacted layer "
                        f"({layer.label()}); history before stage {layer.hi} is gone")
            yield layer

    def _newest(self, cid: str, as_of: int | None = None) -> Record | None:
        """Newest record for cid, *including* tombstones."""
        for layer in self._visible(as_of):
            self.counters["layers_visited"] += 1
            if cid not in layer.bloom:          # cheap: skip layer without reading it
                self.counters["bloom_skips"] += 1
                continue
            rec = layer.records.get(cid)
            if rec is not None:
                return rec
        return None

    def get(self, concept_id: str, as_of: int | None = None) -> Record | None:
        rec = self._newest(concept_id, as_of)
        return None if rec is None or rec.tombstone else rec

    def query(self, text: str, k: int = 3, as_of: int | None = None) -> list[tuple[float, Record]]:
        """Semantic path: newest version of each concept, tombstones suppress older ones."""
        qv = embed(text)
        seen: set[str] = set()
        hits: list[tuple[float, Record]] = []
        for layer in self._visible(as_of):
            for cid, rec in layer.records.items():
                if cid in seen:
                    continue            # an older version: superseded
                seen.add(cid)
                if rec.tombstone:
                    continue            # deleted: suppress all older versions too
                score = cosine(qv, layer.vectors[cid])
                if score > 0:
                    hits.append((score, rec))
        hits.sort(key=lambda h: (-h[0], -h[1].stage, h[1].concept_id))
        return hits[:k]

    def live(self, as_of: int | None = None) -> dict[str, Record]:
        state: dict[str, Record] = {}
        seen: set[str] = set()
        for layer in self._visible(as_of):
            for cid, rec in layer.records.items():
                if cid not in seen:
                    seen.add(cid)
                    if not rec.tombstone:
                        state[cid] = rec
        return state

    def history(self, concept_id: str) -> list[Record]:
        """Every retained version, oldest first (the audit trail)."""
        return [l.records[concept_id] for l in self.layers if concept_id in l.records]

    # -------------------------------------------------------------------- undo
    def horizon(self) -> int:
        """How many stages can be rolled back exactly."""
        n = 0
        for layer in reversed(self.layers):
            if layer.compacted:
                break
            n += 1
        return n

    def rollback(self, n_stages: int = 1) -> list[int]:
        """Global undo: drop the newest n layers. O(n), exact, no undo log."""
        if n_stages < 0:
            raise ValueError("n_stages must be >= 0")
        if n_stages == 0:
            return []
        h = self.horizon()
        if n_stages > h:
            raise RollbackError(f"can roll back at most {h} stage(s); "
                                f"older stages were merged by compaction")
        dropped = self.layers[-n_stages:]
        del self.layers[-n_stages:]
        return [l.lo for l in dropped]

    def revert(self, stage: int) -> tuple[int | None, list[str]]:
        """Selective undo (like `git revert`): undo what ONE stage wrote, keep
        everything written after it. Returns (new_stage_id, conflicts). A concept
        that was changed again after `stage` is a conflict and is left alone."""
        target = next((l for l in self.layers if l.lo <= stage <= l.hi), None)
        if target is None:
            raise ValueError(f"no stage {stage}")
        if target.compacted:
            raise HistoryUnavailable(f"stage {stage} was merged by compaction")
        restore: list[Event] = []
        conflicts: list[str] = []
        for cid in target.records:
            if self._newest(cid).stage != stage:
                conflicts.append(cid)
                continue
            prior = self._newest(cid, as_of=stage - 1)
            if prior is None or prior.tombstone:
                restore.append(Event(cid, op=Op.DELETE))
            else:
                restore.append(Event(cid, prior.text, Op.UPDATE, prior.value))
        new_stage = self.ingest(restore, note=f"revert of stage {stage}") if restore else None
        return new_stage, conflicts

    # ------------------------------------------------------- erasure / space
    def forget(self, concept_id: str) -> dict:
        """Hard erasure. Rewrites every layer that holds the concept, so no
        retained history and no rollback can bring it back. The audit log keeps
        only a hash of the id, never the content."""
        touched = 0
        new_layers = []
        for layer in self.layers:
            if concept_id in layer.records:
                new_layers.append(layer.without(concept_id))
                touched += 1
            else:
                new_layers.append(layer)
        self.layers = new_layers
        entry = {"concept_sha256": hashlib.sha256(concept_id.encode()).hexdigest()[:16],
                 "at_stage": self.head_stage, "layers_rewritten": touched,
                 "unix_time": round(time.time(), 3)}
        self.erasure_log.append(entry)
        return entry

    def compact(self, retention_window: int) -> dict:
        """Merge every layer except the newest W into one base layer.

        Superseded versions are discarded, and because the base is the bottom of
        the stack, tombstones and the data they hide are dropped: deleted data
        is physically gone. Rollback horizon becomes exactly W."""
        if retention_window < 0:
            raise ValueError("retention_window must be >= 0")
        cut = max(0, len(self.layers) - retention_window)
        cold = self.layers[:cut]
        if not cold:
            return {"merged_layers": 0, "records_before": 0, "records_after": 0,
                    "superseded_dropped": 0, "tombstones_dropped": 0}
        before = sum(len(l.records) for l in cold)
        merged: dict[str, Record] = {}
        for layer in cold:                      # oldest -> newest, newest wins
            merged.update(layer.records)
        superseded = before - len(merged)
        tombstones = [cid for cid, r in merged.items() if r.tombstone]
        for cid in tombstones:
            del merged[cid]
        base = Layer(cold[0].lo, cold[-1].hi, merged, "compacted",
                     note=f"compacted {len(cold)} layer(s)")
        self.layers = [base] + self.layers[cut:]
        return {"merged_layers": len(cold), "records_before": before,
                "records_after": len(merged), "superseded_dropped": superseded,
                "tombstones_dropped": len(tombstones)}

    # ------------------------------------------------------------ inspection
    def stats(self) -> dict:
        return {
            "stages": self.head_stage + 1,
            "layers": len(self.layers),
            "compacted_layers": sum(l.compacted for l in self.layers),
            "records_stored": sum(len(l.records) for l in self.layers),
            "tombstones": sum(r.tombstone for l in self.layers for r in l.records.values()),
            "live_concepts": len(self.live()),
            "bytes": len(self.to_bytes()),
            "bloom_bytes": sum(l.bloom.size_bytes for l in self.layers),
            "rollback_horizon": self.horizon(),
        }

    def snapshot(self) -> "LayeredStore":
        """O(#layers) copy: layers are immutable, so they are shared, not copied."""
        s = LayeredStore()
        s.layers = list(self.layers)
        s.erasure_log = list(self.erasure_log)
        return s

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"format": "vsm-mini/1", "layers": [l.to_dict() for l in self.layers],
                "erasure_log": self.erasure_log}

    def to_bytes(self) -> bytes:
        """Everything the store physically holds; used to *verify* erasure."""
        return json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(self.to_bytes())
        os.replace(tmp, path)                   # atomic on POSIX and Windows

    @classmethod
    def from_dict(cls, d: dict) -> "LayeredStore":
        if d.get("format") != "vsm-mini/1":
            raise ValueError("unknown store format")
        s = cls()
        s.layers = [Layer.from_dict(x) for x in d["layers"]]
        s.erasure_log = list(d.get("erasure_log", []))
        return s

    @classmethod
    def load(cls, path: str) -> "LayeredStore":
        with open(path, "rb") as f:
            return cls.from_dict(json.loads(f.read().decode("utf-8")))
