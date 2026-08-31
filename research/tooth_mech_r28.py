"""
LATERAL round 28 - U12(ii): WHICH GEAR'S TOOTH ACTUALLY DRIVES F, AND IS THE
TWIN'S LOW-F OUTLIER LOCALISED IN THE SMALL GEARS?

Two mechanisms for the round-27 finding are dead (angular coherence, refuted in
the sign; "the teeth are the reciprocal of a small integer", refuted by the
m-sweep).  U12(ii) names the next probe: gears 5 and 7 decide every <= 5-point
shape (the completeness lemma), and the twin vector's (v_5, v_7) = (1,1) is one
of only six (v_5, v_7) classes.  So decompose.

For each machine, over the EXHAUSTIVE family V(y):

  (a) ONE-WAY VARIANCE DECOMPOSITION.  eta^2(q) = the fraction of Var(F)
      explained by gear q's tooth position alone.  Which gear moves F most?
  (b) THE MARGINAL PROFILE.  mean F as a function of v_q, per gear - and
      whether the twin's own v_q is the argmin of that profile.
  (c) THE CONDITIONAL PERCENTILE.  The twin's percentile of F WITHIN its own
      (v_5, v_7) class.  If the effect is the small gears, this goes to ~50%;
      if it survives, the effect is spread over the whole vector.

Usage: python tooth_mech_r28.py [--upto 19]
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
    d = np.diff(op)
    return int(max(int(d.max()), P - int(op[-1]) + int(op[0]))), int(op.size)


def pct(vals, x):
    vals = np.asarray(vals)
    return 100.0 * ((vals < x).sum() + 0.5 * (vals == x).sum()) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=19)
    a = ap.parse_args()

    for n in range(3, len(GEARS) + 1):
        gears = GEARS[:n]
        y = gears[-1]
        if y > a.upto:
            break
        P = int(np.prod(gears))
        twin = [min(pow(6, -1, q) % q, (-pow(6, -1, q)) % q) for q in gears]
        space = [list(range(1, (q - 1) // 2 + 1)) for q in gears]
        vecs = np.array(list(itertools.product(*space)), dtype=np.int64)
        Fs = np.empty(len(vecs), dtype=np.int64)
        nref = None
        for i, vs in enumerate(vecs):
            F, nop = maxgap(gears, list(vs), P)
            Fs[i] = F
            if nref is None:
                nref = nop
            elif nop != nref:
                raise AssertionError("sharing law broken")
        gate(nref == int(np.prod([q - 2 for q in gears])),
             "m%-2d: all %d counterfactuals have prod(q-2) = %d openings"
             % (y, len(vecs), nref))
        ti = int(np.flatnonzero((vecs == np.array(twin)).all(axis=1))[0])
        tot = float(((Fs - Fs.mean()) ** 2).sum())

        print("\n=== m%d   |V| = %d   F(twin) = %d   overall percentile %.1f%% ==="
              % (y, len(vecs), Fs[ti], pct(Fs, Fs[ti])))
        print("  (a) ONE-WAY VARIANCE DECOMPOSITION - which gear's tooth moves F")
        print("      gear  #choices  eta^2      mean F by v_q "
              "(twin's own value marked *)")
        etas = []
        argmin_hits = 0
        for k, q in enumerate(gears):
            between = 0.0
            prof = []
            for v in space[k]:
                sel = vecs[:, k] == v
                m = Fs[sel].mean()
                prof.append((v, m, int(sel.sum())))
                between += sel.sum() * (m - Fs.mean()) ** 2
            eta = between / tot
            etas.append((q, eta))
            best = min(prof, key=lambda t: t[1])[0]
            if best == twin[k]:
                argmin_hits += 1
            s = "  ".join(("%d:%.2f%s" % (v, m, "*" if v == twin[k] else ""))
                          for v, m, _ in prof)
            print("      %-5d %-9d %-10.4f %s" % (q, len(space[k]), eta, s))
        print("      -> gear with the LARGEST eta^2: %d ; monotone decreasing "
              "in q: %s" % (max(etas, key=lambda t: t[1])[0],
                            all(etas[i][1] >= etas[i + 1][1]
                                for i in range(len(etas) - 1))))
        print("      -> the twin's own v_q is the argmin of the marginal profile "
              "for %d of %d gears" % (argmin_hits, len(gears)))

        print("  (c) THE CONDITIONAL PERCENTILE - twin's rank inside its own "
              "(v_5,v_7) class")
        cls = (vecs[:, 0] == twin[0]) & (vecs[:, 1] == twin[1])
        pin = pct(Fs[cls], Fs[ti])
        print("      class (v_5,v_7) = (%d,%d) has %d members, mean F %.2f "
              "(family mean %.2f)" % (twin[0], twin[1], int(cls.sum()),
                                      Fs[cls].mean(), Fs.mean()))
        print("      twin's percentile WITHIN the class: %.1f%%   "
              "(overall: %.1f%%)" % (pin, pct(Fs, Fs[ti])))
        # all six classes ranked
        rows = []
        for v5 in space[0]:
            for v7 in space[1]:
                sel = (vecs[:, 0] == v5) & (vecs[:, 1] == v7)
                rows.append(((v5, v7), Fs[sel].mean(), int(sel.sum())))
        rows.sort(key=lambda r: r[1])
        print("      all (v_5,v_7) classes by mean F: %s"
              % ", ".join("%s %.2f%s" % (r[0], r[1],
                                         "*" if r[0] == (twin[0], twin[1]) else "")
                          for r in rows))

        print("  (d) THE CONDITIONING LADDER - is the effect a MAIN EFFECT of a few")
        print("      gears, or an INTERACTION spread over the whole vector?  Pin the")
        print("      twin's own value on a growing set of coordinates and re-rank it")
        print("      inside the surviving sub-family.  A main effect of the pinned")
        print("      gears would push the percentile UP toward 50%; an interaction")
        print("      pushes it DOWN.")
        print("      pinned          #members  mean F   twin percentile")
        for k in range(0, n + 1):
            sel = np.ones(len(vecs), dtype=bool)
            for c in range(k):
                sel &= vecs[:, c] == twin[c]
            lab = "small gears " + (str(tuple(gears[:k])) if k else "(none)")
            print("      %-15s %-9d %-8.2f %5.1f%%"
                  % (lab, int(sel.sum()), Fs[sel].mean(), pct(Fs[sel], Fs[ti])))
        for k in range(1, n):
            sel = np.ones(len(vecs), dtype=bool)
            for c in range(n - k, n):
                sel &= vecs[:, c] == twin[c]
            lab = "large gears " + str(tuple(gears[n - k:]))
            print("      %-15s %-9d %-8.2f %5.1f%%"
                  % (lab, int(sel.sum()), Fs[sel].mean(), pct(Fs[sel], Fs[ti])))

    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
