"""Unit tests for the engine and baselines. Run: python3 -m unittest -v"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsm_mini import (Event, HistoryUnavailable, LayeredStore, MutableStore, NaiveRAG,
                      NotSupported, Op, RollbackError)
from vsm_mini.bloom import BloomFilter


def put(store, cid, value, text=None):
    return store.ingest([Event(cid, text or f"{cid} is {value}", Op.UPDATE, value)])


class TestReads(unittest.TestCase):
    def test_newest_version_wins(self):
        s = LayeredStore()
        put(s, "a", "1"); put(s, "a", "2"); put(s, "b", "x")
        self.assertEqual(s.get("a").value, "2")
        self.assertEqual({k: r.value for k, r in s.live().items()}, {"a": "2", "b": "x"})

    def test_last_event_in_a_stage_wins(self):
        s = LayeredStore()
        s.ingest([Event("a", "a one", Op.INSERT, "1"), Event("a", "a two", Op.UPDATE, "2")])
        self.assertEqual(s.get("a").value, "2")

    def test_query_never_returns_superseded_version(self):
        s = LayeredStore()
        put(s, "lang", "C", "my favourite language is C")
        put(s, "lang", "Rust", "my favourite language is now Rust, definitely")
        hits = s.query("favourite language", k=5)
        self.assertEqual([r.value for _, r in hits], ["Rust"])

    def test_tombstone_suppresses_all_older_versions(self):
        s = LayeredStore()
        put(s, "a", "1", "alpha one"); put(s, "a", "2", "alpha two")
        s.delete("a")
        self.assertIsNone(s.get("a"))
        self.assertEqual(s.query("alpha"), [])

    def test_reinsert_after_delete(self):
        s = LayeredStore()
        put(s, "a", "1"); s.delete("a"); put(s, "a", "3")
        self.assertEqual(s.get("a").value, "3")

    def test_as_of(self):
        s = LayeredStore()
        put(s, "a", "1"); put(s, "a", "2"); s.delete("a")
        self.assertEqual(s.get("a", as_of=0).value, "1")
        self.assertEqual(s.get("a", as_of=1).value, "2")
        self.assertIsNone(s.get("a", as_of=2))
        self.assertEqual(s.live(as_of=-1), {})

    def test_empty_store(self):
        s = LayeredStore()
        self.assertIsNone(s.get("x"))
        self.assertEqual(s.query("anything"), [])
        self.assertEqual(s.horizon(), 0)
        self.assertRaises(RollbackError, s.rollback, 1)

    def test_rejects_missing_concept_id(self):
        self.assertRaises(ValueError, LayeredStore().ingest, [Event("", "text")])


class TestUndo(unittest.TestCase):
    def test_rollback_is_exact(self):
        s = LayeredStore()
        states = []
        for i in range(6):
            put(s, f"k{i % 3}", str(i))
            states.append({k: r.value for k, r in s.live().items()})
        for depth in range(1, 6):
            snap = s.snapshot()
            snap.rollback(depth)
            self.assertEqual({k: r.value for k, r in snap.live().items()}, states[-1 - depth])

    def test_rollback_undoes_soft_delete(self):
        s = LayeredStore()
        put(s, "a", "1"); s.delete("a"); s.rollback(1)
        self.assertEqual(s.get("a").value, "1")

    def test_snapshot_is_independent(self):
        s = LayeredStore()
        put(s, "a", "1"); put(s, "a", "2")
        snap = s.snapshot(); snap.rollback(1)
        self.assertEqual(s.get("a").value, "2")
        self.assertEqual(snap.get("a").value, "1")

    def test_revert_is_selective(self):
        s = LayeredStore()
        put(s, "city", "Chennai"); put(s, "editor", "Vim")
        bad = put(s, "city", "Atlantis")
        put(s, "editor", "Helix")
        new, conflicts = s.revert(bad)
        self.assertEqual((s.get("city").value, s.get("editor").value), ("Chennai", "Helix"))
        self.assertEqual(conflicts, [])
        self.assertEqual(new, 4)

    def test_revert_of_first_write_removes_concept(self):
        s = LayeredStore()
        st = put(s, "a", "1")
        s.revert(st)
        self.assertIsNone(s.get("a"))

    def test_revert_reports_conflict_and_leaves_later_write(self):
        s = LayeredStore()
        put(s, "a", "1"); bad = put(s, "a", "2"); put(s, "a", "3")
        new, conflicts = s.revert(bad)
        self.assertEqual((new, conflicts), (None, ["a"]))
        self.assertEqual(s.get("a").value, "3")

    def test_revert_unknown_stage(self):
        self.assertRaises(ValueError, LayeredStore().revert, 7)


class TestErasureAndCompaction(unittest.TestCase):
    def test_forget_is_physical_and_survives_rollback_and_reload(self):
        s = LayeredStore()
        put(s, "phone", "555-0101", "my phone is 555-0101")
        put(s, "phone", "555-0199", "new phone is 555-0199")
        put(s, "x", "1")
        s.forget("phone")
        raw = s.to_bytes()
        self.assertNotIn(b"555-01", raw)
        self.assertNotIn(b'"phone"', raw)
        s.rollback(3)
        self.assertIsNone(s.get("phone"))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            s.save(p)
            with open(p, "rb") as f:
                self.assertNotIn(b"555-01", f.read())

    def test_soft_delete_keeps_bytes_until_compaction(self):
        s = LayeredStore()
        put(s, "addr", "12 Beach Rd", "address is 12 Beach Rd")
        s.delete("addr")
        for i in range(5):
            put(s, "n", str(i))
        self.assertIn(b"12 Beach Rd", s.to_bytes())
        s.compact(retention_window=2)
        self.assertNotIn(b"12 Beach Rd", s.to_bytes())

    def test_delete_inside_window_is_kept_until_it_leaves_the_window(self):
        s = LayeredStore()
        put(s, "addr", "12 Beach Rd"); s.delete("addr")
        s.compact(retention_window=2)
        self.assertIn(b"12 Beach Rd", s.to_bytes())   # still undoable
        s.rollback(1)
        self.assertEqual(s.get("addr").value, "12 Beach Rd")

    def test_compaction_preserves_current_state_and_sets_horizon(self):
        s = LayeredStore()
        for i in range(10):
            put(s, f"k{i % 4}", str(i))
            if i == 5:
                s.delete("k1")
        before = {k: r.value for k, r in s.live().items()}
        stats = s.compact(retention_window=3)
        self.assertEqual({k: r.value for k, r in s.live().items()}, before)
        self.assertEqual(s.horizon(), 3)
        self.assertGreater(stats["superseded_dropped"], 0)
        self.assertRaises(RollbackError, s.rollback, 4)
        self.assertTrue(all(not r.tombstone for r in s.layers[0].records.values()))

    def test_compact_with_window_larger_than_stack_is_noop(self):
        s = LayeredStore()
        put(s, "a", "1")
        self.assertEqual(s.compact(10)["merged_layers"], 0)
        self.assertEqual(s.horizon(), 1)

    def test_as_of_into_compacted_history_refused(self):
        s = LayeredStore()
        for i in range(5):
            put(s, "a", str(i))
        s.compact(2)
        self.assertRaises(HistoryUnavailable, s.get, "a", 0)
        self.assertEqual(s.get("a", as_of=2).value, "2")   # boundary is still exact
        self.assertRaises(HistoryUnavailable, s.revert, 1)

    def test_erasure_log_has_no_content(self):
        s = LayeredStore()
        put(s, "secret::id", "hunter2")
        entry = s.forget("secret::id")
        self.assertNotIn("secret", str(entry))


class TestPersistenceAndSketches(unittest.TestCase):
    def test_roundtrip(self):
        s = LayeredStore()
        for i in range(8):
            put(s, f"k{i % 3}", str(i), f"value number {i}")
        s.delete("k2"); s.compact(3)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            s.save(p)
            r = LayeredStore.load(p)
        self.assertEqual(r.to_bytes(), s.to_bytes())
        self.assertEqual(r.horizon(), 3)
        self.assertEqual(r.query("value number"), s.query("value number"))

    def test_bloom_has_no_false_negatives(self):
        keys = [f"user{i}::attr{i % 7}" for i in range(2000)]
        bf = BloomFilter(len(keys))
        for k in keys:
            bf.add(k)
        self.assertTrue(all(k in bf for k in keys))
        fp = sum(f"other{i}" in bf for i in range(5000)) / 5000
        self.assertLess(fp, 0.03)

    def test_bloom_lets_keyed_reads_skip_layers(self):
        s = LayeredStore()
        put(s, "target", "1")
        for i in range(50):
            put(s, f"noise{i}", str(i))
        s.get("target")
        self.assertGreater(s.counters["bloom_skips"], 40)


class TestBaselines(unittest.TestCase):
    def test_naive_rag_returns_stale_and_deleted(self):
        r = NaiveRAG()
        r.ingest([Event("lang", "favourite language is C", Op.INSERT, "C")])
        r.ingest([Event("lang", "favourite language is now Rust instead", Op.UPDATE, "Rust")])
        self.assertEqual(r.query("favourite language", 1)[0][1].value, "C")
        r.delete("lang")
        self.assertTrue(r.query("favourite language"))
        for op in (lambda: r.rollback(1), lambda: r.compact(1), lambda: r.forget("lang")):
            self.assertRaises(NotSupported, op)

    def test_mutable_soft_delete_keeps_bytes(self):
        m = MutableStore()
        m.ingest([Event("p", "phone 555-0101", Op.INSERT, "555-0101")])
        m.delete("p")
        self.assertIsNone(m.get("p"))
        self.assertIn(b"555-0101", m.to_bytes())
        self.assertRaises(NotSupported, m.rollback, 1)


if __name__ == "__main__":
    unittest.main()
