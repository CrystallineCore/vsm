"""Bloom filter: 'is this concept possibly in this layer?' with no false negatives."""
from __future__ import annotations

import hashlib
import math


class BloomFilter:
    __slots__ = ("m", "k", "bits")

    def __init__(self, expected_items: int, fp_rate: float = 0.01):
        n = max(1, expected_items)
        self.m = max(8, math.ceil(-n * math.log(fp_rate) / (math.log(2) ** 2)))
        self.k = max(1, round(self.m / n * math.log(2)))
        self.bits = bytearray((self.m + 7) // 8)

    def _positions(self, key: str):
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:16], "big") | 1  # double hashing
        return ((h1 + i * h2) % self.m for i in range(self.k))

    def add(self, key: str) -> None:
        for p in self._positions(key):
            self.bits[p >> 3] |= 1 << (p & 7)

    def __contains__(self, key: str) -> bool:
        return all(self.bits[p >> 3] & (1 << (p & 7)) for p in self._positions(key))

    @property
    def size_bytes(self) -> int:
        return len(self.bits)
