"""pl_depth2.py -- item 4(c): does the record of M + q' come out of M's dictionary at depth 2?

Two candidate formulas, both read off the OLD machine only:

    2-run form:  F2run(M, q') = max over letters l in {a_L, b_L, q'} of ( l + r(l) )
    3-run form:  F3run(M, q') = max over letters l of ( l + N(l) ),  N(l) = largest neighbour SUM

The 3-run form is the J = 2 fusion of the merge law (file 05, cited): a fusion that removes two
openings needs the gap between them to be a letter, and its length is L + l + R.  The measured
F(M + q') ladder is 2, 5, 7, 11, 18, 25, 34, 43, 58 at M_0 .. M_8.

Part B (optional, argv "m31"): the same at M = {5..31}, q' = 37, streamed over the full period
33,426,748,355, to get N(12) and hence the depth-2 prediction for F({5..37}) = 88 (recorded).

Outputs results/pl_depth2.txt / .json.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
sys.path.insert(0, os.path.join(HERE, "..", "r58"))
from mf_core import build_levels, u_of                    # noqa: E402
from ag_gate import build_m29_gaps                        # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
GEARS31 = [5, 7, 11, 13, 17, 19, 23, 29, 31]
VMAX = 96
MARGIN = 512
CHUNK = 3 * 10 ** 7


def letters(q):
    a = (2 * u_of(q)) % q
    return (min(a, q - a), max(a, q - a))


def rows_and_N(size, chunk=20_000_000):
    """r(v) (largest single neighbour) and N(v) (largest neighbour sum) for every realised v."""
    n = size.size
    mx = int(size.max())
    r = np.zeros(mx + 1, dtype=np.int64)
    N = np.zeros(mx + 1, dtype=np.int64)
    mult = np.zeros(mx + 1, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        # a 3-run (L, v, R) starting at index i:  L = size[i], v = size[i+1], R = size[i+2]
        i0, i1 = s, min(e, n - 2)
        if i1 > i0:
            Lg = size[i0:i1].astype(np.int64)
            V = size[i0 + 1:i1 + 1].astype(np.int64)
            R = size[i0 + 2:i1 + 2].astype(np.int64)
            mult += np.bincount(V, minlength=mx + 1)
            np.maximum.at(r, V, np.maximum(Lg, R))
            np.maximum.at(N, V, Lg + R)
            del Lg, V, R
    # the wrap-around triples
    for i in (n - 2, n - 1):
        Lg, V, R = int(size[i]), int(size[(i + 1) % n]), int(size[(i + 2) % n])
        mult[V] += 1
        r[V] = max(r[V], Lg, R)
        N[V] = max(N[V], Lg + R)
    realised = np.flatnonzero(mult).tolist()
    return realised, {v: int(r[v]) for v in realised}, {v: int(N[v]) for v in realised}


def worker31(args):
    lo, hi = args
    u = [pow(6, -1, g) for g in GEARS31]
    r = np.zeros(VMAX, dtype=np.int64)
    N = np.zeros(VMAX, dtype=np.int64)
    spec = np.zeros(VMAX, dtype=np.int64)
    c0 = lo
    while c0 < hi:
        c1 = min(c0 + CHUNK, hi)
        s, e = c0 - MARGIN, c1 + MARGIN
        blocked = np.zeros(e - s, dtype=bool)
        for g, ug in zip(GEARS31, u):
            for t in (ug, g - ug):
                blocked[(t - s) % g::g] = True
        opens = np.flatnonzero(~blocked).astype(np.int64) + s
        del blocked
        gaps = np.diff(opens).astype(np.int64)
        own = np.flatnonzero((opens[:-1] >= c0) & (opens[:-1] < c1))
        own = own[(own >= 1) & (own + 1 < gaps.size)]
        if own.size:
            V = gaps[own]
            Lg = gaps[own - 1]
            R = gaps[own + 1]
            spec += np.bincount(V, minlength=VMAX)[:VMAX]
            np.maximum.at(r, V, np.maximum(Lg, R))
            np.maximum.at(N, V, Lg + R)
        del opens, gaps
        c0 = c1
    return spec, r, N


def main():
    t0 = time.time()
    lines = []
    W = lines.append
    W("=== ITEM 4(c): the record of M + q' from M's dictionary at depth 2 ===")
    W("F_measured = the recorded ladder F(M_0..M_8) = 2, 5, 7, 11, 18, 25, 34, 43, 58")
    W("2-run form = max over l in {a_L, b_L, q'} of l + r(l);  3-run form = max of l + N(l)")
    W("rung | F(M) | F(M+q') | a_L | b_L | 2-run form | 3-run form | 3-run = F(M+q')? | "
      "the maximising letter")
    LAD = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}
    L = build_levels()
    res = {}
    for n in range(0, 7):
        qn = PRIMES[n + 1]
        realised, r, N = rows_and_N(L[n].size)
        aL, bL = letters(qn)
        F = max(realised)
        cand2 = {l: (l + r[l]) for l in (aL, bL, qn) if l in r}
        cand3 = {l: (l + N[l]) for l in (aL, bL, qn) if l in N}
        f2 = max(cand2.values()) if cand2 else 0
        f3 = max(cand3.values()) if cand3 else 0
        arg3 = max(cand3, key=cand3.get) if cand3 else None
        W(f"{PRIMES[n]}->{qn} | {F} | {LAD[qn]} | {aL} | {bL} | {f2} | {f3} | "
          f"{'YES' if f3 == LAD[qn] else ('over' if f3 > LAD[qn] else 'under')} | l = {arg3}")
        res[str(qn)] = dict(F=F, Fnext=LAD[qn], aL=aL, bL=bL, f2=f2, f3=f3, arg3=arg3,
                            r={int(k): int(v) for k, v in r.items()},
                            N={int(k): int(v) for k, v in N.items()})
    g29 = build_m29_gaps(L[6])
    realised, r, N = rows_and_N(g29)
    aL, bL = letters(31)
    F = max(realised)
    cand2 = {l: (l + r[l]) for l in (aL, bL, 31) if l in r}
    cand3 = {l: (l + N[l]) for l in (aL, bL, 31) if l in N}
    f2, f3 = max(cand2.values()), max(cand3.values())
    arg3 = max(cand3, key=cand3.get)
    W(f"29->31 | {F} | 58 | {aL} | {bL} | {f2} | {f3} | "
      f"{'YES' if f3 == 58 else ('over' if f3 > 58 else 'under')} | l = {arg3}")
    W(f"  m29 detail: r(a_L)={r[aL]} N(a_L)={N[aL]}  r(b_L)={r[bL]} N(b_L)={N[bL]}  "
      f"r(31)={r.get(31)} N(31)={N.get(31)}")
    res["31"] = dict(F=F, Fnext=58, aL=aL, bL=bL, f2=f2, f3=f3, arg3=arg3,
                     r={int(k): int(v) for k, v in r.items()},
                     N={int(k): int(v) for k, v in N.items()})
    del g29, L
    W(f"[{time.time()-t0:.1f}s]")

    if "m31" in sys.argv:
        P = 1
        for g in GEARS31:
            P *= g
        nproc = 3
        bounds = [(i * P // nproc, (i + 1) * P // nproc) for i in range(nproc)]
        with Pool(nproc) as pool:
            parts = pool.map(worker31, bounds)
        spec = sum(p[0] for p in parts)
        r31 = np.max([p[1] for p in parts], axis=0)
        N31 = np.max([p[2] for p in parts], axis=0)
        realised = np.flatnonzero(spec).tolist()
        F = max(realised)
        aL, bL = letters(37)
        cand2 = {l: int(l + r31[l]) for l in (aL, bL, 37) if spec[l]}
        cand3 = {l: int(l + N31[l]) for l in (aL, bL, 37) if spec[l]}
        W(f"\n=== M = {{5..31}} streamed ({time.time()-t0:.1f}s): F = {F}, "
          f"openings {int(spec.sum()):,} ===")
        W(f"  a_L = {aL}: r = {int(r31[aL])}, N = {int(N31[aL])}; "
          f"b_L = {bL}: r = {int(r31[bL])}, N = {int(N31[bL])}; "
          f"37: r = {int(r31[37])}, N = {int(N31[37])}")
        W(f"  2-run form = {max(cand2.values())}, 3-run form = {max(cand3.values())}, "
          f"recorded F({{5..37}}) = 88 -> "
          f"{'YES' if max(cand3.values()) == 88 else 'the depth-2 form FALLS SHORT'}")
        res["37"] = dict(F=F, Fnext=88, aL=aL, bL=bL, f2=max(cand2.values()),
                         f3=max(cand3.values()),
                         r={int(k): int(r31[k]) for k in realised},
                         N={int(k): int(N31[k]) for k in realised})

    json.dump(res, open(os.path.join(OUT, "pl_depth2.json"), "w"))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "pl_depth2.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
