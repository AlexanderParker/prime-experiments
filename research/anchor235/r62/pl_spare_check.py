"""pl_spare_check.py -- exercise the spare-gear lemma on ORDINARY 2-runs (not only the attaining
ones), where free gears do exist: every run with a free gear must have span <= F(M), and every run
of span > F(M) must have no free gear.  A single counterexample refutes the lemma.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
from mf_core import build_levels, u_of                    # noqa: E402
from pl_spare import busy_and_free                        # noqa: E402

OUT = os.path.join(HERE, "results")
lines = []
W = lines.append
W("=== the spare-gear lemma on ordinary 2-runs ===")
W("machine | runs tested | runs with a free gear | of those, span > F (must be 0) | "
  "runs with span > F | of those, free > 0 (must be 0) | span distribution of free runs")
L = build_levels()
rng = np.random.default_rng(20260906)
bad = 0
for n, qn in ((2, 13), (3, 17), (4, 19), (5, 23), (6, 29)):
    lv = L[n]
    gears = list(lv.gears)
    us = [u_of(g) for g in gears]
    F = int(lv.size.max())
    N = lv.size.size
    idx = rng.choice(N - 2, size=min(4000, N - 2), replace=False)
    nfree = nbig = nfreebig = nbigfree = 0
    spans = []
    t0 = time.time()
    for i in idx:
        i = int(i)
        a, v = int(lv.size[i]), int(lv.size[i + 1])
        fr, ob, bu = busy_and_free(gears, us, int(lv.O[i]), a, v)
        S = a + v
        if fr:
            nfree += 1
            spans.append(S)
            if S > F:
                nfreebig += 1
                bad += 1
        if S > F:
            nbig += 1
            if fr:
                nbigfree += 1
    hist = {}
    for s in spans:
        hist[s] = hist.get(s, 0) + 1
    W(f"m{gears[-1]} | {len(idx)} | {nfree} | {nfreebig} | {nbig} | {nbigfree} | "
      f"max span with a free gear {max(spans) if spans else '-'} (F = {F}) "
      f"[{time.time()-t0:.1f}s]")
W(f"\ncounterexamples to the lemma: {bad}")
txt = "\n".join(lines)
open(os.path.join(OUT, "pl_spare_check.txt"), "w").write(txt)
print(txt)
