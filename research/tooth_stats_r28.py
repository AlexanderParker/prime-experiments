"""
LATERAL round 28 - THE COUNTERFACTUAL FAMILY'S *OTHER* STATISTICS (brief item a).

Round 27 established: in the exhaustive family

    V(y) = prod_{q <= y} {1, .., (q-1)/2},   gear q's teeth at +-v_q,

every member has the same gears, the same mirror symmetry, the same period P and
the same survivor count prod(q-2) - only the tooth POSITIONS move - and the real
twin machine's record gap F sits at the 20.0 / 18.1 / 26.4 / 17.1 percentile at
m11 / m13 / m17 / m19.  That is one statistic.  The live route does not use F on
its own; it uses F_2 and the INCREMENT F(M+q') - F_2(M) (the increment law,
F(M+q') - F_2(M) <= s_min(q') = min(2v_q', q'-2v_q')), and the budget slack
F(M+q') - F(M) - q'.

This script places the twin machine in the counterfactual distribution of each,
exhaustively and exactly.  Two nested families are reported for every step
M -> M+q':
    (A) FULL      - both the old teeth and the new gear's tooth vary  (= V(y'));
    (B) v_q' PINNED to the twin value - only the OLD machine's teeth vary,
        which is the cleaner null model for "given the new gear, is the old
        machine's arithmetic favourable?".

It also asks a question that only the counterfactual frame can ask: IS THE
INCREMENT LAW A GENERIC PROPERTY OF SYMMETRIC-TEETH SIEVES, or is it arithmetic?
i.e. what fraction of the family violates it?

Usage: python tooth_stats_r28.py [--upto 19]
"""
import argparse
import itertools
import sys

import numpy as np

GEARS = [5, 7, 11, 13, 17, 19]

NGATE = 0


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def stats(gears, vs, P):
    """(F, F_2, F_3, #distinct gap values, #openings) for one symmetric-teeth sieve."""
    blocked = np.zeros(P, dtype=bool)
    for q, v in zip(gears, vs):
        blocked[v % q::q] = True
        blocked[(-v) % q::q] = True
    op = np.flatnonzero(~blocked).astype(np.int32)
    n = op.size
    # cyclic j-gaps: append the wrapped copies so every index has a successor
    ext = np.empty(n + 3, dtype=np.int32)
    ext[:n] = op
    ext[n:] = op[:3] + np.int32(P)
    g1 = ext[1:n + 1] - op
    g2 = ext[2:n + 2] - op
    g3 = ext[3:n + 3] - op
    # #distinct gap values by bincount, NOT np.unique - the values are bounded by
    # F (tens), so this is O(n) instead of an O(n log n) sort, and the whole
    # m19 level drops from tens of minutes to minutes.
    return (int(g1.max()), int(g2.max()), int(g3.max()),
            int(np.count_nonzero(np.bincount(g1))), int(n))


def pct(vals, x):
    vals = np.asarray(vals)
    return 100.0 * ((vals < x).sum() + 0.5 * (vals == x).sum()) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=19)
    a = ap.parse_args()

    twin_v = {q: min(pow(6, -1, q) % q, (-pow(6, -1, q)) % q) for q in GEARS}
    gate([twin_v[q] for q in GEARS] == [1, 1, 2, 2, 3, 3],
         "twin tooth vector is (1,1,2,2,3,3) = round(q/6)")

    F, F2, F3, NV, IDX, TIDX, PER = {}, {}, {}, {}, {}, {}, {}
    levels = []
    for n in range(1, len(GEARS) + 1):
        gears = GEARS[:n]
        y = gears[-1]
        if y > a.upto:
            break
        levels.append(n)
        P = int(np.prod(gears))
        PER[n] = P
        space = [list(range(1, (q - 1) // 2 + 1)) for q in gears]
        vecs = list(itertools.product(*space))
        f, f2, f3, nv = [], [], [], []
        nref = None
        for vs in vecs:
            s = stats(gears, list(vs), P)
            if nref is None:
                nref = s[4]
            elif s[4] != nref:
                raise AssertionError("sharing law broken: opening count moved")
            f.append(s[0]); f2.append(s[1]); f3.append(s[2]); nv.append(s[3])
        F[n] = np.array(f); F2[n] = np.array(f2); F3[n] = np.array(f3)
        NV[n] = np.array(nv); IDX[n] = vecs
        TIDX[n] = vecs.index(tuple(twin_v[q] for q in gears))
        gate(int(nref) == int(np.prod([q - 2 for q in gears])),
             "m%-2d: every one of %d counterfactuals has prod(q-2) = %d openings"
             % (y, len(vecs), nref))

    # ---- BLOCK 1: per-machine placement of the twin in F, F_2, F_3, spectrum ----
    print("\n=== 1. THE TWIN'S PERCENTILE IN EACH STATISTIC (exhaustive family) ===")
    print("  y   |V|     stat    twin  min  median   max   percentile")
    for n in levels:
        if n < 2:
            continue
        y = GEARS[n - 1]
        ti = TIDX[n]
        for nm, arr in (("F", F[n]), ("F_2", F2[n]), ("F_3", F3[n]),
                        ("#gapvals", NV[n])):
            print("  %-3d %-7d %-7s %-5d %-4d %-8.1f %-5d %5.1f%%"
                  % (y, len(arr), nm, arr[ti], arr.min(), float(np.median(arr)),
                     arr.max(), pct(arr, arr[ti])))
        print("")
    gate(F[3][TIDX[3]] == 7 and F[4][TIDX[4]] == 11 and F[5][TIDX[5]] == 18,
         "reproduces round 27's F(twin) = 7, 11, 18 at m11, m13, m17")
    if 6 in levels:
        gate(F[6][TIDX[6]] == 25, "reproduces round 27's F(twin) = 25 at m19")

    # ---- BLOCK 2: the STEP statistics ----
    print("=== 2. THE STEP STATISTICS: increment, budget slack, increment law ===")
    print("   (A) = full family V(y'); (B) = new gear's tooth PINNED to the twin's)")
    for n in levels:
        if n < 2:
            continue
        q = GEARS[n - 1]
        L = (q - 1) // 2
        y = GEARS[n - 2] if n >= 2 else None
        # member i of level n has prefix i//L at level n-1
        pre = np.arange(len(F[n])) // L
        Fnew = F[n]
        Fold = F[n - 1][pre]
        F2old = F2[n - 1][pre]
        vq = (np.arange(len(F[n])) % L) + 1
        inc = Fnew - F2old
        slack = Fnew - Fold - q
        smin = np.minimum(2 * vq, q - 2 * vq)
        viol = inc > smin
        ti = TIDX[n]
        pin = vq == twin_v[q]
        gate(bool(pin[ti]), "step %d->%d: the twin member has v_q' = %d"
             % (GEARS[n - 2] if n >= 2 else 0, q, twin_v[q]))
        gate(int(pre[ti]) == TIDX[n - 1],
             "step ->%d: the twin member's prefix is the twin at the previous level" % q)
        print("\n  STEP %s -> %d   (|V| = %d, pinned sub-family %d)"
              % ("{5..%d}" % y if n >= 2 else "{}", q, len(Fnew), int(pin.sum())))
        print("     twin: F(M)=%d F_2(M)=%d F(M+q')=%d  increment=%d  s_min=%d  slack=%d"
              % (Fold[ti], F2old[ti], Fnew[ti], inc[ti], smin[ti], slack[ti]))
        for nm, arr in (("F(M+q')", Fnew), ("increment F(M+q')-F_2(M)", inc),
                        ("budget slack F(M+q')-F(M)-q'", slack)):
            pa = pct(arr, arr[ti])
            pb = pct(arr[pin], arr[ti])
            print("     %-28s twin %-5d  min %-5d med %-7.1f max %-5d   "
                  "pct(A) %5.1f%%  pct(B) %5.1f%%"
                  % (nm, arr[ti], arr.min(), float(np.median(arr)), arr.max(), pa, pb))
        print("     INCREMENT LAW  inc <= s_min : violated by %d/%d = %.2f%% of the "
              "family  (pinned sub-family %d/%d = %.2f%%)"
              % (int(viol.sum()), len(viol), 100.0 * viol.mean(),
                 int(viol[pin].sum()), int(pin.sum()), 100.0 * viol[pin].mean()))
        print("     twin verdict: increment %d vs s_min %d -> %s"
              % (inc[ti], smin[ti], "HOLDS" if not viol[ti] else "FAILS"))
        # how far into the law's own margin does the family reach?
        print("     law margin s_min - inc : twin %+d ; family min %+d, median %+.1f, "
              "max %+d ; twin percentile %.1f%%"
              % (smin[ti] - inc[ti], int((smin - inc).min()),
                 float(np.median(smin - inc)), int((smin - inc).max()),
                 pct(smin - inc, smin[ti] - inc[ti])))

    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
