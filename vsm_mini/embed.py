"""Tiny lexical embedder: text -> L2-normalised sparse bag-of-words vector.

This is deliberately *not* semantic. It only captures word overlap, which is
enough to exercise retrieval and version resolution with zero dependencies.
The engine never depends on how vectors are produced, so a real sentence
embedder could be swapped in later without touching storage logic.
"""
from __future__ import annotations

import math
import re

_STOP = frozenset(
    """a an the is are was were be been am i me my mine we our you your it its of to in
    on at for and or but with that this what which who whom where when how do does did
    so as by from have has had will would can could should just about ve m s t""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    out = []
    for word in _TOKEN.findall(text.lower()):
        if word in _STOP:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]  # crude plural folding: "lives" -> "live"
        out.append(word)
    return out


def embed(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for tok in tokenize(text):
        counts[tok] = counts.get(tok, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in counts.values()))
    return {t: v / norm for t, v in counts.items()} if norm else {}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())
