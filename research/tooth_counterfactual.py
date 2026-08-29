"""
LATERAL round 27 - IS THE TWIN MACHINE'S TOOTH VECTOR SPECIAL FOR F?

New object, built from a RELATIONSHIP between parts (measurement directive):
the machine is (gears) x (a tooth half-width per gear).  The gears are given;
the half-widths v_q = 6^{-1} mod q are FORCED by the twin constellation.  Vary
them.  F is invariant under k -> +-k + b but NOT under k -> ck (scaling is not
an isometry of Z_P), so the counterfactual family

    V(y) = prod_{q <= y} {1, .., (q-1)/2},   teeth of gear q at +-v_q

is a genuine parameter space of symmetric-teeth sieves with the SAME gears, the
same period P and the SAME opening count prod(q-2) (the sharing law, result 2),
in which only the POSITIONS move.  |V| = 180, 1440, 12960 at m13, m17, m19 -
small enough to enumerate EXHAUSTIVELY.  So: where does the real machine sit in
the exact distribution of F over its own counterfactuals?

Relation to Refuted 3 (r2): that enumeration scored the real phase vector on
WASTE metrics (top 10-25%, no variational handle).  This scores it on F itself,
which is the quantity the project actually needs, and it is a different question.

Usage: python tooth_counterfactual.py [--upto 19]
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


def maxgap(gears, vs, P):
    blocked = np.zeros(P, dtype=bool)
    for q, v in zip(gears, vs):
        blocked[v % q::q] = True
        blocked[(-v) % q::q] = True
    op = np.flatnonzero(~blocked)
    if op.size < 2:
        return P, op.size
    d = np.diff(op)
    wrap = P - op[-1] + op[0]
    return int(max(d.max(), wrap)), int(op.size)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


def mech(gears, Fs, space, true_v):
    """P11: is F explained by ANGULAR COHERENCE of the teeth?"""
    vecs = list(itertools.product(*space))
    th = np.array([[v / q for v, q in zip(vs, gears)] for vs in vecs])
    disp = th.max(axis=1) - th.min(axis=1)
    sd = th.std(axis=1)
    print("       P11 mechanism: spearman(F, angular dispersion) = %+.4f ; "
          "spearman(F, sd theta) = %+.4f"
          % (spearman(Fs, disp), spearman(Fs, sd)))
    order = np.argsort(Fs, kind="stable")
    tv = tuple(true_v)
    ti = vecs.index(tv)
    print("       theta of the TWIN vector %-22s disp %.4f  sd %.4f  F %d"
          % (str(tv), disp[ti], sd[ti], Fs[ti]))
    print("       5 LOWEST-F vectors : %s"
          % "  ".join("%s F=%d d=%.3f" % (vecs[i], Fs[i], disp[i]) for i in order[:5]))
    print("       5 HIGHEST-F vectors: %s"
          % "  ".join("%s F=%d d=%.3f" % (vecs[i], Fs[i], disp[i]) for i in order[-5:]))
    # mean F by dispersion quartile
    qs = np.quantile(disp, [0.25, 0.5, 0.75])
    bins = np.digitize(disp, qs)
    print("       mean F by dispersion quartile: %s"
          % ", ".join("Q%d %.2f" % (b + 1, Fs[bins == b].mean()) for b in range(4)))
    own = Fs[bins == bins[ti]]
    pct = 100.0 * ((own < Fs[ti]).sum() + 0.5 * (own == Fs[ti]).sum()) / len(own)
    print("       TWIN sits in dispersion quartile Q%d (mean F %.2f) at percentile "
          "%.1f%% of that quartile alone (%d vectors)"
          % (bins[ti] + 1, own.mean(), pct, len(own)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=19)
    a = ap.parse_args()

    print("=== F over the exhaustive symmetric-teeth counterfactual family ===")
    print("   y    |V|     N=prod(q-2)  F(twin)  min  median  max   rank of twin"
          "        percentile")
    for n in range(2, len(GEARS) + 1):
        gears = GEARS[:n]
        y = gears[-1]
        if y > a.upto:
            break
        P = 1
        for q in gears:
            P *= q
        true_v = [pow(6, -1, q) for q in gears]
        true_v = [min(v % q, (-v) % q) for v, q in zip(true_v, gears)]
        space = [list(range(1, (q - 1) // 2 + 1)) for q in gears]
        sizes = 1
        for s in space:
            sizes *= len(s)
        Fs = []
        Ftrue = None
        Nref = None
        for vs in itertools.product(*space):
            F, nop = maxgap(gears, list(vs), P)
            if Nref is None:
                Nref = nop
            else:
                gate_ok = (nop == Nref)
                if not gate_ok:
                    raise AssertionError("opening count not constant - sharing law broken")
            Fs.append(F)
            if list(vs) == true_v:
                Ftrue = F
        Fs = np.array(Fs)
        gate(Ftrue is not None, "m%-2d: the true tooth vector %s is in the family"
             % (y, true_v))
        gate(int(Nref) == int(np.prod([q - 2 for q in gears])),
             "m%-2d: every counterfactual has exactly prod(q-2) = %d openings"
             % (y, Nref))
        rank = int((Fs < Ftrue).sum())
        ties = int((Fs == Ftrue).sum())
        pct = 100.0 * (rank + 0.5 * ties) / len(Fs)
        print("  %-4d %-7d %-12d %-8d %-4d %-7.1f %-5d %d..%d of %d      %.1f%%"
              % (y, sizes, Nref, Ftrue, Fs.min(), float(np.median(Fs)), Fs.max(),
                 rank + 1, rank + ties, len(Fs), pct))
        # the distribution itself
        vals, cnts = np.unique(Fs, return_counts=True)
        print("       F distribution: %s"
              % ", ".join("%d:%d" % (v, c) for v, c in zip(vals, cnts)))
        if n >= 4:
            mech(gears, Fs, space, true_v)
    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
