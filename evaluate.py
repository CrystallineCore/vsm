#!/usr/bin/env python3
"""One command to verify the whole prototype.

    python3 evaluate.py            # everything (~20 s)
    python3 evaluate.py --quick    # fewer differential trials

Runs, in order: unit tests, differential test vs an independent reference,
mutation check (planted bugs must be caught), the narrated demo, and the
benchmark. Prints a scorecard and exits non-zero if anything fails.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))


def banner(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main() -> int:
    try:
        sys.stdout.reconfigure(errors="replace")   # never crash on a console encoding
    except AttributeError:
        pass
    quick = "--quick" in sys.argv
    trials = 300 if quick else 2000
    results = []

    banner("1/5  Unit tests")
    os.environ["VSM_TRIALS"] = "50"            # the differential gets its own step
    suite = unittest.defaultTestLoader.loadTestsFromName("test_engine")
    buf = io.StringIO()
    r = unittest.TextTestRunner(stream=buf, verbosity=1).run(suite)
    print(buf.getvalue().strip().splitlines()[-1], f"- {r.testsRun} tests")
    results.append(("Unit tests", r.wasSuccessful(), f"{r.testsRun} tests"))

    banner(f"2/5  Differential test ({trials} random operation sequences)")
    from test_differential import run_trial
    t0 = time.time()
    try:
        checks = sum(run_trial(s) for s in range(trials))
        print(f"{trials} trials, {checks:,} step-level checks, 0 mismatches "
              f"({time.time() - t0:.1f}s)")
        results.append(("Differential vs reference", True, f"{checks:,} checks"))
    except AssertionError as e:
        print("MISMATCH:", e)
        results.append(("Differential vs reference", False, str(e)[:60]))

    banner("3/5  Mutation check (planted bugs must be detected)")
    import mutation_check
    ok = mutation_check.main(trials=300)
    results.append(("Mutation check", ok, f"{len(mutation_check.MUTANTS)} planted bugs"))

    banner("4/5  Narrated demo")
    p = subprocess.run([sys.executable, os.path.join(HERE, "demo.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "NO_COLOR": "1", "PYTHONIOENCODING": "utf-8"})
    print(p.stdout)
    last = (p.stdout.strip().splitlines() or ["(no output)"])[-1].strip()
    results.append(("Demo", p.returncode == 0, last))

    banner("5/5  Benchmark")
    p = subprocess.run([sys.executable, os.path.join(HERE, "benchmark.py")],
                       capture_output=True, text=True, encoding="utf-8", cwd=HERE,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(p.stdout, p.stderr)
    results.append(("Benchmark laws", p.returncode == 0, "horizon + erasure laws"))

    banner("Scorecard")
    for name, passed, detail in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<28} {detail}")
    all_ok = all(p for _, p, _ in results)
    print(f"\n  {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
