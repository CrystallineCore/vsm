"""Mutation check: plant known bugs in the engine and confirm the differential
test catches every one. A test suite that cannot fail proves nothing.

    python3 tests/mutation_check.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vsm_mini.engine as E
from test_differential import run_trial


def _oldest_wins_compaction(self, w):
    cut = max(0, len(self.layers) - w)
    cold = self.layers[:cut]
    if not cold:
        return {}
    merged = {}
    for layer in reversed(cold):                 # BUG: oldest version wins
        merged.update(layer.records)
    merged = {k: v for k, v in merged.items() if not v.tombstone}
    self.layers = [E.Layer(cold[0].lo, cold[-1].hi, merged, "compacted")] + self.layers[cut:]
    return {}


def _compaction_keeps_tombstones_as_values(self, w):
    cut = max(0, len(self.layers) - w)
    cold = self.layers[:cut]
    if not cold:
        return {}
    merged = {}
    for layer in cold:
        for k, r in layer.records.items():
            if not r.tombstone:                  # BUG: deletes ignored during merge
                merged[k] = r
    self.layers = [E.Layer(cold[0].lo, cold[-1].hi, merged, "compacted")] + self.layers[cut:]
    return {}


def _tombstones_do_not_suppress(self, as_of=None):
    state = {}
    for layer in self._visible(as_of):
        for cid, r in layer.records.items():
            if cid not in state and not r.tombstone:   # BUG: falls through to older version
                state[cid] = r
    return state


def _rollback_ignores_horizon(self, n=1):
    if n:
        del self.layers[-n:]                     # BUG: can drop the compacted base
    return []


def _forget_only_top_layer(self, cid):
    for i in range(len(self.layers) - 1, -1, -1):
        if cid in self.layers[i].records:        # BUG: older copies survive
            self.layers[i] = self.layers[i].without(cid)
            break
    return {}


MUTANTS = [
    ("compaction keeps the OLDEST version", "compact", _oldest_wins_compaction),
    ("compaction ignores deletes", "compact", _compaction_keeps_tombstones_as_values),
    ("tombstones don't hide older versions", "live", _tombstones_do_not_suppress),
    ("rollback ignores the horizon", "rollback", _rollback_ignores_horizon),
    ("forget() erases only the newest copy", "forget", _forget_only_top_layer),
]


def main(trials: int = 300) -> bool:
    all_caught = True
    for label, attr, bad in MUTANTS:
        original = getattr(E.LayeredStore, attr)
        setattr(E.LayeredStore, attr, bad)
        try:
            for seed in range(trials):
                run_trial(seed)
            caught, where = False, ""
        except AssertionError as e:
            caught, where = True, str(e).split(":")[0]
        except Exception as e:                   # crash also counts as detection
            caught, where = True, f"crash ({type(e).__name__})"
        finally:
            setattr(E.LayeredStore, attr, original)
        all_caught &= caught
        print(f"  {'CAUGHT ' if caught else 'MISSED '} {label:<40} {where}")
    return all_caught


if __name__ == "__main__":
    ok = main()
    print("All planted bugs detected." if ok else "Some planted bugs were NOT detected.")
    sys.exit(0 if ok else 1)
