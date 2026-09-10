"""ol_gate.py -- gates for the free-phase pattern instrument of ol_pattern.py.

G1 (m23, complete): the machine {5..23} has period 37,182,145 and is sieved directly, so D_3(m23)
is known exactly.  Every triple (a, b, c) of realised gap VALUES of m23 is put to the instrument
and the verdict compared with the scan: 0 disagreements is the gate.

G2 (m29, complete on the realised side, sampled on the other): D_3(m29) is exact off the closure
from D_9(m23) (three new gaps fuse at most 3 * J_max(23->29) = 9 old ones, so the fixed-depth step
at m = 3 has loss = 0), and the instrument is asked the same question.

Usage: uv run python research/anchor235/r70/ol_gate.py [nsample]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "r66")))

from mo_core import base_gaps, closure_step, dict_from_gaps  # noqa: E402
from ol_pattern import realised_word  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def triples_of_period(g):
    n = g.size
    w = np.stack([g, np.roll(g, -1), np.roll(g, -2)], axis=1)
    return set(map(tuple, np.unique(w, axis=0).astype(int)))


def main():
    nsample = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    rng = np.random.default_rng(20260911)
    report = {}

    # ---------------- G1: m23, complete
    P, g23 = base_gaps(23)
    D3 = triples_of_period(g23)
    vals = sorted(set(int(v) for v in np.unique(g23)))
    print(f"m23: P = {P:,} N = {g23.size:,} |D_1| = {len(vals)} |D_3| = {len(D3):,}", flush=True)
    gears23 = [5, 7, 11, 13, 17, 19, 23]
    t0 = time.time()
    bad = []
    tested = 0
    for a in vals:
        for b in vals:
            for c in vals:
                want = (a, b, c) in D3
                got = realised_word((a, b, c), gears23)
                tested += 1
                if want != got:
                    bad.append((a, b, c, want, got))
    print(f"G1 m23: {tested:,} triples tested, {len(bad)} disagreements "
          f"[{time.time()-t0:.1f}s]", flush=True)
    report["G1_m23"] = {"tested": tested, "disagreements": len(bad),
                        "examples": bad[:5], "D3": len(D3), "D1": len(vals),
                        "secs": round(time.time() - t0, 1)}

    # ---------------- G2: m29 from the closure, D_3 exact
    win, mult = dict_from_gaps(g23, 9)
    w2, m2, st = closure_step(win, mult, 29, m=3, mode="fixed")
    print(f"m29: D_3 from D_9(m23) -- {w2.shape[0]:,} rows, mass {int(m2.sum()):,}, "
          f"loss {st['loss']}", flush=True)
    D3_29 = set(map(tuple, w2[:, :3].astype(int)))
    v29 = sorted(set(int(v) for v in np.unique(w2[:, 0])))
    gears29 = [5, 7, 11, 13, 17, 19, 23, 29]
    t0 = time.time()
    bad2 = []
    for t in sorted(D3_29):
        if not realised_word(t, gears29):
            bad2.append((t, True, False))
    print(f"G2a m29: {len(D3_29):,} realised triples, {len(bad2)} missed "
          f"[{time.time()-t0:.1f}s]", flush=True)
    grid = [(int(a), int(b), int(c)) for a in v29 for b in v29 for c in v29]
    non = [t for t in grid if t not in D3_29]
    idx = rng.choice(len(non), size=min(nsample, len(non)), replace=False)
    t0 = time.time()
    bad3 = []
    for i in idx:
        t = non[int(i)]
        if realised_word(t, gears29):
            bad3.append((t, False, True))
    print(f"G2b m29: {len(idx):,} unrealised triples sampled, {len(bad3)} false positives "
          f"[{time.time()-t0:.1f}s]", flush=True)
    report["G2_m29"] = {"D3": len(D3_29), "D1": len(v29), "loss": int(st["loss"]),
                        "realised_missed": len(bad2), "sampled_unrealised": int(len(idx)),
                        "false_positives": len(bad3),
                        "examples": [list(map(int, x[0])) for x in (bad2 + bad3)[:5]]}
    with open(os.path.join(OUT, "gate.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
