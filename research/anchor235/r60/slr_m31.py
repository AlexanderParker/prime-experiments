"""slr_m31.py -- the adjacent-pair dictionary of M = {5..31}, streamed, for the rung 31->37.

Out-of-sample test of the branch's mechanism: a_L(37) = 12, 3*12 - 1 = 35 = 5*7, so
Leg(12) cap M = {5, 7} and c_5(12) = 3.  The pre-registered prediction is r(12) >= 43.

The period is 33,426,748,355 columns.  It is sieved in chunks by `nproc` processes, each owning a
contiguous range plus a margin on both sides; a gap is attributed to the chunk containing its LEFT
endpoint, so every gap of the period is counted exactly once and every adjacent pair is complete.
Same construction as research/anchor235/r45/deep_profile.py, which returned F(m31) = 58.

Outputs results/slr_m31.txt / .json.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31]
VMAX = 96
MARGIN = 512
CHUNK = 3 * 10 ** 7


def worker(args):
    lo, hi = args
    u = [pow(6, -1, g) for g in GEARS]
    D = np.zeros(VMAX * VMAX, dtype=np.int64)
    spec = np.zeros(VMAX, dtype=np.int64)
    c0 = lo
    while c0 < hi:
        c1 = min(c0 + CHUNK, hi)
        s, e = c0 - MARGIN, c1 + MARGIN
        blocked = np.zeros(e - s, dtype=bool)
        for g, ug in zip(GEARS, u):
            for t in (ug, g - ug):
                blocked[(t - s) % g::g] = True
        opens = np.flatnonzero(~blocked).astype(np.int64) + s
        del blocked
        gaps = np.diff(opens).astype(np.int64)
        own = np.flatnonzero((opens[:-1] >= c0) & (opens[:-1] < c1))
        if own.size:
            spec += np.bincount(gaps[own], minlength=VMAX)[:VMAX]
            oi = own[own + 1 < gaps.size]
            D += np.bincount(gaps[oi] * VMAX + gaps[oi + 1],
                             minlength=VMAX * VMAX)[:VMAX * VMAX]
        del opens, gaps
        c0 = c1
    return spec, D


def cost_closed(p, v):
    if v % p == 0:
        return 2
    if (3 * v - 1) % p == 0 or (3 * v + 1) % p == 0:
        return 3
    return 4


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    t0 = time.time()
    P = 1
    for g in GEARS:
        P *= g
    bounds = [P * i // nproc for i in range(nproc + 1)]
    jobs = [(bounds[i], bounds[i + 1]) for i in range(nproc)]
    with Pool(nproc) as pool:
        parts = pool.map(worker, jobs)
    spec = sum(p[0] for p in parts)
    D = sum(p[1] for p in parts).reshape(VMAX, VMAX)
    dt = time.time() - t0
    F = int(np.max(np.flatnonzero(spec)))
    realised = np.flatnonzero(spec).tolist()
    F2 = max(a + v for a in realised for v in np.flatnonzero(D[a]).tolist())
    sym = bool(np.array_equal(D[:F + 1, :F + 1], D[:F + 1, :F + 1].T))
    r, rows = {}, {}
    for v in realised:
        s = set(np.flatnonzero(D[:, v]).tolist()) | set(np.flatnonzero(D[v]).tolist())
        rows[v] = sorted(s)
        r[v] = max(s) if s else 0
    d = {v: min(F, F2 - v) - r[v] for v in realised}
    c5 = {v: cost_closed(5, v) for v in realised}
    aL, bL, qn = 12, 25, 37
    lines = []
    W = lines.append
    W(f"=== M = {{5..31}} streamed, {nproc} processes, {dt:.1f}s ===")
    W(f"  period {P:,}  openings {int(spec.sum()):,}  F = {F}  F_2 = {F2}  "
      f"dictionary symmetric: {sym}")
    W(f"  q' = 37, letters (a_L, b_L) = ({aL}, {bL}); 3a_L = 36 = q' - 1, neighbours 35 = 5*7 "
      f"and 37 = q'")
    for tag, v in (("short letter", aL), ("long letter", bL), ("padded q'", qn)):
        if v not in r:
            W(f"  row {tag} = {v}: not realised")
            continue
        holes = [a for a in realised if a < r[v] and a not in set(rows[v])]
        W(f"  row {tag} = {v}: r = {r[v]} ({r[v]/F:.3f} F, {r[v]/F2:.3f} F_2), "
          f"cap min(F, F_2-v) = {min(F, F2-v)}, d = {d[v]}, c_5 = {c5[v]}, "
          f"Leg cap M = {[p for p in GEARS if cost_closed(p, v) <= 3]}, "
          f"mult = {int(spec[v]):,}, holes = {holes}")
        W("      R(%d) with counts: " % v
          + " ".join(f"{a}:{int(D[a][v] + D[v][a])}" for a in rows[v]))
    W("  profile v : r(v) [cap] {d} c5 :")
    W("    " + "  ".join(f"{v}:{r[v]}[{min(F,F2-v)}]{{{d[v]}}}c{c5[v]}" for v in realised))
    for cc in (2, 3, 4):
        vs = [v for v in realised if c5[v] == cc]
        ds = sorted(d[v] for v in vs)
        W(f"  c_5 = {cc}: {len(vs)} sizes, d min/median/max = "
          f"{ds[0]}/{np.median(ds):.1f}/{ds[-1]}, median r/F = "
          f"{np.median([r[v]/F for v in vs]):.3f}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "slr_m31.txt"), "w").write(txt)
    json.dump(dict(F=F, F2=F2, realised=realised, r={int(k): int(v) for k, v in r.items()},
                   d={int(k): int(v) for k, v in d.items()},
                   c5={int(k): int(v) for k, v in c5.items()},
                   mult={int(v): int(spec[v]) for v in realised},
                   rows={int(k): v for k, v in rows.items()}, secs=dt),
              open(os.path.join(OUT, "slr_m31.json"), "w"))
    print(txt[:2000])


if __name__ == "__main__":
    main()
