"""Two baselines with the same interface as LayeredStore.

* NaiveRAG      - append every chunk to a flat vector list; no versions, no deletes.
* MutableStore  - one row per concept, updates overwrite, delete sets a flag.

Operations a baseline structurally cannot do raise NotSupported, so the
benchmark records the inability as a result instead of leaving a gap.
"""
from __future__ import annotations

import json
from typing import Iterable

from .embed import cosine, embed
from .engine import Event, NotSupported, Op, Record


class NaiveRAG:
    name = "naive RAG"

    def __init__(self) -> None:
        self.entries: list[tuple[Record, dict]] = []
        self.stage = -1

    def ingest(self, events: Iterable[Event], note: str = "") -> int:
        self.stage += 1
        for e in events:
            # A delete event carries no content, and a flat index has nothing to
            # apply it to: the chunk just stays there.
            if Op(e.op) is not Op.DELETE and e.text:
                self.entries.append((Record(e.concept_id, e.text, e.value, self.stage),
                                     embed(e.text)))
        return self.stage

    def delete(self, concept_id: str, note: str = "") -> int:
        return self.ingest([Event(concept_id, op=Op.DELETE)])

    def query(self, text: str, k: int = 3, as_of=None) -> list[tuple[float, Record]]:
        qv = embed(text)
        scored = [(cosine(qv, v), i, r) for i, (r, v) in enumerate(self.entries)]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: (-s[0], s[1]))       # no notion of recency
        return [(s, r) for s, _, r in scored[:k]]

    def rollback(self, n_stages: int = 1):
        raise NotSupported("naive RAG keeps no version order to roll back")

    def compact(self, retention_window: int):
        raise NotSupported("naive RAG cannot tell superseded chunks from current ones")

    def forget(self, concept_id: str):
        raise NotSupported("naive RAG has no concept identity to erase by")

    def to_bytes(self) -> bytes:
        return json.dumps([[r.concept_id, r.text, r.value, r.stage]
                           for r, _ in self.entries]).encode("utf-8")


class MutableStore:
    name = "mutable + soft delete"

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.stage = -1

    def ingest(self, events: Iterable[Event], note: str = "") -> int:
        self.stage += 1
        for e in events:
            if Op(e.op) is Op.DELETE:
                if e.concept_id in self.rows:
                    self.rows[e.concept_id]["deleted"] = True   # soft delete
            else:
                self.rows[e.concept_id] = {"text": e.text, "value": e.value,
                                           "stage": self.stage, "deleted": False,
                                           "vec": embed(e.text)}  # overwrite: old value lost
        return self.stage

    def delete(self, concept_id: str, note: str = "") -> int:
        return self.ingest([Event(concept_id, op=Op.DELETE)])

    def get(self, concept_id: str, as_of=None):
        if as_of is not None:
            raise NotSupported("mutable store overwrote history")
        row = self.rows.get(concept_id)
        if row is None or row["deleted"]:
            return None
        return Record(concept_id, row["text"], row["value"], row["stage"])

    def query(self, text: str, k: int = 3, as_of=None) -> list[tuple[float, Record]]:
        qv = embed(text)
        hits = []
        for cid, row in self.rows.items():
            if row["deleted"]:
                continue
            s = cosine(qv, row["vec"])
            if s > 0:
                hits.append((s, Record(cid, row["text"], row["value"], row["stage"])))
        hits.sort(key=lambda h: (-h[0], -h[1].stage, h[1].concept_id))
        return hits[:k]

    def rollback(self, n_stages: int = 1):
        raise NotSupported("mutable store overwrote the old values")

    def compact(self, retention_window: int):
        raise NotSupported("mutable store has nothing to compact")

    def forget(self, concept_id: str):
        # A mutable store *can* hard-delete a row; it just loses history to do it.
        self.rows.pop(concept_id, None)

    def to_bytes(self) -> bytes:
        return json.dumps({c: {k: v for k, v in r.items() if k != "vec"}
                           for c, r in self.rows.items()}).encode("utf-8")
