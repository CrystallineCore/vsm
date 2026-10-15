#!/usr/bin/env python3
"""Interactive shell for poking at the memory store live.

    python3 shell.py              # empty store
    python3 shell.py --sample     # preloaded with a few facts

Type `help` for commands. Each write becomes its own immutable stage.
"""
from __future__ import annotations

import cmd
import shlex
import sys

from vsm_mini import (Event, HistoryUnavailable, LayeredStore, NaiveRAG, Op, RollbackError)


class MemoryShell(cmd.Cmd):
    intro = ("Versioned Semantic Memory shell. Type `help` or `help <command>`.\n"
             "Example:  put user::lang Rust | I switched to Rust\n")
    prompt = "vsm> "

    def __init__(self, sample: bool = False):
        super().__init__()
        self.store = LayeredStore()
        self.rag = NaiveRAG()           # shadow baseline, for `compare`
        if sample:
            for cid, val, text in [
                ("user::lang", "C", "My favourite programming language is C."),
                ("user::city", "Chennai", "I live in Chennai."),
                ("user::lang", "Rust", "These days my favourite programming language is Rust."),
            ]:
                self._write([Event(cid, text, Op.UPDATE, val)])
            print("Loaded 3 sample stages (user::lang C -> Rust, user::city Chennai).")

    # ---------------------------------------------------------------- helpers
    def _write(self, events, note=""):
        self.rag.ingest(events)
        return self.store.ingest(events, note=note)

    def onecmd(self, line):
        try:
            return super().onecmd(line)
        except (RollbackError, HistoryUnavailable, ValueError) as e:
            print(f"  refused: {e}")
        except Exception as e:  # keep the shell alive during a live demo
            print(f"  error: {type(e).__name__}: {e}")

    def emptyline(self):
        pass

    # ---------------------------------------------------------------- writes
    def do_put(self, arg):
        """put <concept_id> <value> [| free text]   -- record a fact (new stage)"""
        head, _, text = arg.partition("|")
        parts = head.split(maxsplit=1)
        if len(parts) != 2:
            print("  usage: put <concept_id> <value> [| free text]")
            return
        cid, value = parts[0], parts[1].strip()
        text = text.strip() or f"{cid.split('::')[-1]} is {value}"
        print(f"  stage {self._write([Event(cid, text, Op.UPDATE, value)])} sealed")

    def do_delete(self, arg):
        """delete <concept_id>   -- soft delete (tombstone); undoable until compacted"""
        print(f"  stage {self._write([Event(arg.strip(), op=Op.DELETE)])} sealed (tombstone)")

    def do_forget(self, arg):
        """forget <concept_id>   -- hard erasure from every layer; not undoable"""
        print(f"  {self.store.forget(arg.strip())}")

    # ----------------------------------------------------------------- reads
    def do_get(self, arg):
        """get <concept_id> [@stage]   -- current (or as-of) value"""
        parts = arg.split()
        as_of = int(parts[1].lstrip("@")) if len(parts) > 1 else None
        rec = self.store.get(parts[0], as_of=as_of)
        print(f"  {rec.value!r}  (stage {rec.stage}: \"{rec.text}\")" if rec else "  (none)")

    def do_ask(self, arg):
        """ask <question>   -- top-3 semantic matches, newest version wins"""
        for score, rec in self.store.query(arg, k=3) or []:
            print(f"  {score:.3f}  {rec.concept_id:<20} = {rec.value!r:<12} (stage {rec.stage})")
        if not self.store.query(arg, k=1):
            print("  (no match)")

    def do_compare(self, arg):
        """compare <question>   -- top-1 answer: layered store vs naive RAG"""
        for name, b in (("layered", self.store), ("naive RAG", self.rag)):
            hits = b.query(arg, k=1)
            print(f"  {name:<10} -> {hits[0][1].value!r} (stage {hits[0][1].stage})" if hits
                  else f"  {name:<10} -> (no match)")

    def do_history(self, arg):
        """history <concept_id>   -- every retained version, oldest first"""
        for r in self.store.history(arg.strip()) or []:
            print(f"  stage {r.stage:>3}: " + ("DELETED" if r.tombstone else f"{r.value!r}  \"{r.text}\""))

    def do_layers(self, arg):
        """layers   -- show the layer stack, oldest at top"""
        for l in self.store.layers:
            items = ", ".join(("DEL " + r.concept_id) if r.tombstone else f"{r.concept_id}={r.value}"
                              for r in l.records.values())
            print(f"  [{'compacted' if l.compacted else 'stage':<9}] {l.label():<13} {items[:70]}")
        print(f"  rollback horizon: {self.store.horizon()}")

    def do_stats(self, arg):
        """stats   -- storage and horizon statistics"""
        for k, v in self.store.stats().items():
            print(f"  {k:<18} {v}")

    # ------------------------------------------------------------------ undo
    def do_rollback(self, arg):
        """rollback [n]   -- drop the newest n stages (default 1)"""
        print(f"  dropped stages {self.store.rollback(int(arg or 1))}")

    def do_revert(self, arg):
        """revert <stage>   -- selectively undo one stage, keep later changes"""
        new, conflicts = self.store.revert(int(arg))
        print(f"  compensating stage: {new}; conflicts (changed later, left alone): {conflicts}")

    def do_compact(self, arg):
        """compact <W>   -- merge all but the newest W layers; W becomes the horizon"""
        print(f"  {self.store.compact(int(arg or 0))}")

    # ----------------------------------------------------------- persistence
    def do_save(self, arg):
        """save <path>   -- write the store to a JSON file"""
        self.store.save(arg.strip() or "memory.json")
        print("  saved")

    def do_load(self, arg):
        """load <path>   -- replace the store with one loaded from disk"""
        self.store = LayeredStore.load(arg.strip() or "memory.json")
        print(f"  loaded {len(self.store.layers)} layers")

    def do_find(self, arg):
        """find <text>   -- is this exact text physically present in storage?"""
        present = arg.encode() in self.store.to_bytes()
        print(f"  {'PRESENT in stored bytes' if present else 'absent from stored bytes'}")

    def do_quit(self, arg):
        """quit   -- exit"""
        return True

    do_exit = do_quit
    do_EOF = do_quit


if __name__ == "__main__":
    MemoryShell(sample="--sample" in sys.argv).cmdloop()
