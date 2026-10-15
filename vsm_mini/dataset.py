"""Synthetic update-heavy workload with exact ground truth.

Stage 0 inserts every concept; each later stage updates some live concepts,
deletes a few, and re-inserts a few deleted ones. `truth[s]` is the exact
state (concept_id -> value) after stage s, computed by plain replay.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .engine import Event, Op

ATTRIBUTES = {
    "favourite language": ["python", "rust", "golang", "kotlin", "haskell", "elixir", "zig"],
    "home city": ["chennai", "mumbai", "berlin", "lisbon", "osaka", "nairobi", "toronto"],
    "employer": ["acme", "globex", "initech", "umbrella", "hooli", "vandelay", "wonka"],
    "text editor": ["vim", "emacs", "helix", "vscode", "nano", "sublime", "zed"],
    "pet": ["cat", "dog", "parrot", "tortoise", "hamster", "goldfish", "rabbit"],
    "coffee order": ["espresso", "latte", "cortado", "filter", "mocha", "americano", "flat-white"],
}

UPDATE_TEMPLATES = [
    "The {attr} of {ent} is now {val}.",
    "Update: {ent} changed their {attr} to {val}.",
    "{ent} says the {attr} these days is {val}.",
]


@dataclass
class Workload:
    stages: list[list[Event]]
    truth: list[dict[str, str]]
    queries: dict[str, str]          # concept_id -> natural-language question
    texts: dict[str, list[str]]      # concept_id -> every text ever written for it


def generate(n_entities: int = 40, attrs_per_entity: int = 3, n_stages: int = 12,
             update_frac: float = 0.2, delete_frac: float = 0.03,
             reinsert_frac: float = 0.3, seed: int = 0) -> Workload:
    rng = random.Random(seed)
    concepts = []
    for e in range(n_entities):
        ent = f"person{e:03d}"
        for attr in rng.sample(sorted(ATTRIBUTES), attrs_per_entity):
            concepts.append((f"{ent}::{attr}", ent, attr))
    meta = {cid: (ent, attr) for cid, ent, attr in concepts}
    queries = {cid: f"What is the {attr} of {ent}?" for cid, ent, attr in concepts}
    texts: dict[str, list[str]] = {cid: [] for cid in meta}

    def value_for(cid, avoid=None):
        choices = [v for v in ATTRIBUTES[meta[cid][1]] if v != avoid]
        return rng.choice(choices)

    state: dict[str, str] = {}
    stages: list[list[Event]] = []
    truth: list[dict[str, str]] = []

    first = []
    for cid, ent, attr in concepts:
        val = value_for(cid)
        text = f"The {attr} of {ent} is {val}."
        first.append(Event(cid, text, Op.INSERT, val))
        texts[cid].append(text)
        state[cid] = val
    stages.append(first)
    truth.append(dict(state))

    for _ in range(1, n_stages):
        events = []
        live = sorted(state)
        dead = sorted(set(meta) - set(state))
        n_upd = max(1, int(len(live) * update_frac))
        updated = rng.sample(live, min(n_upd, len(live)))
        rest = [c for c in live if c not in updated]
        deleted = rng.sample(rest, min(len(rest), max(1, int(len(live) * delete_frac))))
        revived = rng.sample(dead, int(len(dead) * reinsert_frac)) if dead else []
        for cid in updated + revived:
            ent, attr = meta[cid]
            val = value_for(cid, avoid=state.get(cid))
            text = rng.choice(UPDATE_TEMPLATES).format(attr=attr, ent=ent, val=val)
            events.append(Event(cid, text, Op.UPDATE, val))
            texts[cid].append(text)
            state[cid] = val
        for cid in deleted:
            events.append(Event(cid, op=Op.DELETE))
            del state[cid]
        stages.append(events)
        truth.append(dict(state))

    return Workload(stages, truth, queries, texts)
