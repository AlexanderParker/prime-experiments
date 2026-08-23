"""Round 20 (mechanic): Q_j(M; a) for EVERY threshold a, in one pass.

qualifying_spectrum.py answers one step at a time because the qualifying floor
a = 2*round(q'/6) depends on q'.  This computes the whole table

    Q[j][a] = max sum of j consecutive gaps whose j-2 MIDDLE gaps are all >= a

in a single stream pass, so the word-free criterion  max_j Q_j <= F + q'  can
be evaluated for ANY q' at that machine - which is what decides whether the
criterion's margin tracks the MACHINE (scale) or the DEPTH it has to reach
(litcap(q'), a function of q' mod 35 that does not grow).

Usage: uv run python research/qspec_table.py y [--limit N]
"""
import os
import sys
import time
import numpy as np
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")
from flank_envelope import primes_upto, literal_cap

V = 128
S = 512
JMAX = 8


def run(y, limit=None, seg=64_000_000):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    K = P if limit is None else min(P, limit)
    uvals = [pow(6, -1, g) for g in gears]
    Q = np.zeros((JMAX + 1, V), np.int64)      # Q[j][a]
    F = 0
    tail = np.array([], dtype=np.int64)
    t0 = time.time()
    for lo in range(0, K, seg):
        hi = min(K, lo + seg)
        ex = np.zeros(hi - lo, bool)
        for g, u in zip(gears, uvals):
            ex[(u - lo) % g::g] = True
            ex[(-u - lo) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64) + lo
        ops = np.concatenate([tail, op])
        if len(ops) > JMAX + 3:
            d = np.diff(ops).astype(np.int64)
            F = max(F, int(d.max()))
            c = np.concatenate([[0], np.cumsum(d)])
            n = len(d)
            for j in range(3, JMAX + 1):
                if n < j:
                    break
                tot = c[j:] - c[:-j]
                mid = d[1:n - j + 2].copy()
                for t in range(2, j - 1):
                    np.minimum(mid, d[t:n - j + 1 + t], out=mid)
                sel = (tot < S) & (mid < V)
                if not sel.any():
                    continue
                cnt = np.bincount(mid[sel] * S + tot[sel],
                                  minlength=V * S).reshape(V, S)
                has = cnt > 0
                rowmax = np.where(has.any(1),
                                  S - 1 - np.argmax(has[:, ::-1], axis=1), 0)
                # Q[j][a] = max over mid >= a  ->  suffix max
                suf = np.maximum.accumulate(rowmax[::-1])[::-1]
                np.maximum(Q[j], suf, out=Q[j])
        tail = ops[-(JMAX + 4):].copy()
    return dict(y=y, P=P, K=K, F=F, Q=Q, secs=time.time() - t0)


def main():
    args = sys.argv[1:]
    limit = None
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(float(args[i + 1]))
        del args[i:i + 2]
    y = int(args[0])
    r = run(y, limit=limit)
    F, Q = r["F"], r["Q"]
    print(f"machine {y}: F = {F}, period {r['P']:.4g}, coverage "
          f"{r['K']/r['P']:.4f}, {r['secs']:.0f}s")
    print("  the criterion for every next-prime candidate q':")
    print("   q'   a=2u'  litcap  ell_max  max_j<=ell+2 Q_j   F+q'   margin"
          "   /q'")
    rows = []
    for q in (13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
              73, 79, 83, 89, 97, 101, 103):
        if q <= y:
            continue
        a = 2 * round(q / 6)
        if a >= V:
            continue
        lc = literal_cap(q)
        L = lc - 1
        top = max([int(Q[j][a]) for j in range(3, min(L + 2, JMAX) + 1)]
                  or [0])
        marg = F + q - top
        rows.append((y, q, a, lc, L, top, F + q, marg, marg / q))
        print(f"  {q:4d}  {a:4d}   {lc:4d}   {L:5d}     {top:10d}"
              f"   {F+q:6d}   {marg:+6d}   {marg/q:6.3f}")
    p = os.path.join(DDIR, "qspec_table.csv")
    new = not os.path.exists(p) or os.path.getsize(p) == 0
    with open(p, "a") as f:
        if new:
            f.write("y,qp,a,litcap,ell_max,maxQ,Fplusq,margin,margin_over_q,"
                    "coverage\n")
        for t in rows:
            f.write(",".join(str(x) for x in t[:8]) +
                    f",{t[8]:.4f},{r['K']/r['P']:.6f}\n")
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
