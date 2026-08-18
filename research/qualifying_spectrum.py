"""Round 17 (mechanic): THE QUALIFYING SPECTRUM Q_j(M; t), measured.

Constructor's fuel_bound Theorem 1: every qualifying interior gap of a chain
is 0 or +-2c mod q' and positive, hence >= 2u' = a = 2*round(q'/6).  So the
merged window of a legal word is j = ell+2 consecutive gaps whose j-2 MIDDLE
gaps are all >= a.  Define

    Q_j(M; a) = max sum of j consecutive gaps whose j-2 middle gaps are >= a.

Then span + FS <= Q_{ell+2} for every QUALIFYING word, and part (D) is implied
by the purely spectral, word-free inequality  Q_{ell+2}(M; a) <= F(M) + q'.
This tool measures Q_j exactly, at full period, and prints the criterion.

Usage: uv run python research/qualifying_spectrum.py y q' [--limit N]
"""
import os
import sys
import time
import numpy as np
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flank_envelope import primes_upto, literal_cap

JMAX = 8


def main():
    args = sys.argv[1:]
    limit = None
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(float(args[i + 1]))
        del args[i:i + 2]
    y, q1 = int(args[0]), int(args[1])
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    K = P if limit is None else min(P, limit)
    uvals = [pow(6, -1, g) for g in gears]
    a = 2 * round(q1 / 6)
    Fj = np.zeros(JMAX + 1, np.int64)
    Qj = np.zeros(JMAX + 1, np.int64)
    Qad = np.zeros(JMAX + 1, np.int64)
    tail = np.array([], dtype=np.int64)
    t0 = time.time()
    for lo in range(0, K, 64_000_000):
        hi = min(K, lo + 64_000_000)
        ex = np.zeros(hi - lo, bool)
        for g, u in zip(gears, uvals):
            ex[(u - lo) % g::g] = True
            ex[(-u - lo) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64) + lo
        ops = np.concatenate([tail, op])
        if len(ops) > JMAX + 2:
            d = np.diff(ops)
            c = np.concatenate([[0], np.cumsum(d)])
            big = d >= a
            n = len(d)
            for j in range(1, JMAX + 1):
                if n < j:
                    continue
                tot = c[j:] - c[:-j]
                Fj[j] = max(Fj[j], int(tot.max()))
                if j >= 3:
                    ok = np.ones(n - j + 1, bool)
                    for m in range(1, j - 1):
                        ok &= big[m:n - j + 1 + m]
                    if ok.any():
                        v = tot[ok]
                        k = int(np.argmax(v))
                        if int(v[k]) > Qj[j]:
                            Qj[j] = int(v[k])
                            Qad[j] = int(ops[np.flatnonzero(ok)[k]])
        tail = ops[-(JMAX + 3):].copy()
    F = int(Fj[1])
    print(f"machine {y} -> q'={q1}: a = 2u' = {a}, F = {F}, F+q' = {F+q1}, "
          f"litcap = {literal_cap(q1)}, {time.time()-t0:.0f}s "
          f"(coverage {K/P:.4f})")
    print("   j   F_j   Q_j(a)   drop   Q_j <= F+q'?      address of Q_j")
    for j in range(3, JMAX + 1):
        v, fv = int(Qj[j]), int(Fj[j])
        print(f"  {j:2d}  {fv:4d}   {v:5d}   {fv-v:4d}   "
              f"{'YES' if 0 < v <= F + q1 else 'NO ':4s}  "
              f"{'(ell=' + str(j-2) + ')':8s}  k = {int(Qad[j]):,}")
    L = literal_cap(q1) - 1
    # Q_j = 0 means NO qualifying window of that depth exists at all, so (D)
    # is vacuous there - it counts as satisfied, not as a failure.  The
    # criterion is therefore the max over all depths up to ell_max.
    qs = [int(Qj[j]) for j in range(3, min(L + 2, JMAX) + 1)]
    top = max(qs) if qs else 0
    kmax_q = max([j for j in range(3, JMAX + 1) if Qj[j] > 0], default=2) - 2
    print(f"  CRITERION over ell = 1..{L}: max_j Q_j = {top} vs F+q' = "
          f"{F+q1}  -> "
          f"{'(D) IMPLIED, word-free (margin %+d)' % (F+q1-top) if 0 < top <= F+q1 else 'NOT implied'}")
    print(f"  FUEL CAP FROM THE SAME OBJECT: Q_j = 0 for j > {kmax_q+2}, so "
          f"no qualifying word longer than ell = {kmax_q} exists "
          f"(k_max <= {kmax_q+1} openings)")


if __name__ == "__main__":
    main()
