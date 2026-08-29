"""
LATERAL round 27 - U5 CLOSED: the cosine near-collisions are numerical crowding,
and the exact-degeneracy law is a THEOREM at every machine.

BACKGROUND (round 21, item 33).  The machine's Hermitian circulant C_M has the
closed-form spectrum: eigenvalue at CRT frequency vector (j_q) is

    lambda(j) = prod_q  f_q(j_q),    f_q(0) = q-2,  f_q(j) = -2 cos(2 pi j u_q / q)

and u_q is invertible, so per gear the factor set is exactly
    S_q = {q-2} u { -2 cos(2 pi r / q) : r = 1..q-1 }.
Round 21 measured the FULL-spectrum tie count as P - prod (q+1)/2 exactly at
m11/13/17, attributed every tie to the mirror, and recorded 6 (m29) / 613 (m31)
DESYMMETRIZED near-collisions at 1e-12 as UNRESOLVED (backlog U5).

THE THEOREM (proved here, verified below).
  Suppose prod_q f_q = prod_q f'_q with f_q, f'_q in S_q.  No element of S_q is
  zero (-2cos(2 pi r/q) = 0 needs 4 | q).  So prod_q (f_q / f'_q) = 1 with
  a_q := f_q/f'_q in K_q := Q(zeta_q)^+.  The K_q have pairwise coprime
  conductors, so each K_q is linearly disjoint from the compositum of the
  others and K_q ^ (compositum of the rest) = Q; hence every a_q lies in Q.
  Now a_q in Q with f_q, f'_q in S_q forces f_q = f'_q:
    - both rational  => both equal q-2;
    - one rational, one not => the ratio is irrational, contradiction;
    - both irrational => they are Galois conjugates over Q, so they have the
      same norm, so a_q^{(q-1)/2} = 1, so a_q = +-1; a_q = -1 needs
      cos(2 pi r/q) = -cos(2 pi r'/q), i.e. 2(r+r') = q or 2(r'-r) = q, both
      impossible for odd q.  So a_q = 1.
  So lambda(j) = lambda(j') iff f_q(j_q) = f_q(j'_q) for every gear, i.e. iff
  j'_q = +-j_q for every gear.  THE DEGENERACY IS EXACTLY THE PER-GEAR SIGN
  GROUP: #distinct = prod (q+1)/2 and tie count = P - prod (q+1)/2, at EVERY
  machine, and NO accidental collision exists at any machine.
  Consequently every reported near-collision is a crowding artifact.

Usage: python u5_collisions.py [--y 29] [--tol 1e-12]
"""
import argparse
import sys

import numpy as np

GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31]

NGATE = 0


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def factor_classes(q):
    """desymmetrized per-gear factor list: r = 0 (value q-2) and r = 1..(q-1)/2."""
    out = [(0, float(q - 2))]
    for r in range(1, (q - 1) // 2 + 1):
        out.append((r, -2.0 * np.cos(2.0 * np.pi * r / q)))
    return out


def build(gears):
    """all desymmetrized levels, with their (r_q) labels."""
    vals = np.array([1.0])
    labels = [()]
    for q in gears:
        fc = factor_classes(q)
        newv = np.empty(vals.size * len(fc))
        newl = []
        for i, (r, f) in enumerate(fc):
            newv[i * vals.size:(i + 1) * vals.size] = vals * f
            newl.extend([lab + (r,) for lab in labels])
        vals, labels = newv, newl
    return vals, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", type=int, default=29)
    ap.add_argument("--tol", type=float, default=1e-12)
    a = ap.parse_args()
    gears = [q for q in GEARS if q <= a.y]

    # --- the counting half of the theorem, checked by brute force at small y
    print("=== A: #distinct levels = prod (q+1)/2 (the theorem's count) ===")
    for n in range(2, 6):
        gs = GEARS[:n]
        P = 1
        for q in gs:
            P *= q
        pred = 1
        for q in gs:
            pred *= (q + 1) // 2
        v, _ = build(gs)
        gate(v.size == pred, "m%-2d: desym level count %d = prod (q+1)/2" % (gs[-1], pred))
        # full spectrum by brute force, distinct count at high tolerance
        import itertools
        full = []
        for js in itertools.product(*[range(q) for q in gs]):
            x = 1.0
            for q, j in zip(gs, js):
                x *= (q - 2) if j == 0 else -2.0 * np.cos(2.0 * np.pi * j / q)
            full.append(x)
        full = np.sort(np.array(full))
        d = np.sum(np.abs(np.diff(full)) > 1e-9) + 1
        gate(d == pred, "m%-2d: full spectrum has exactly %d distinct values, "
             "ties = P - prod (q+1)/2 = %d" % (gs[-1], pred, P - pred))

    # --- the near-collision census at the requested machine
    print("\n=== B: near-collision census at m%d, tol %g ===" % (a.y, a.tol))
    vals, labels = build(gears)
    print("    desymmetrized levels: %d" % vals.size)
    order = np.argsort(vals, kind="stable")
    sv = vals[order]
    dif = np.abs(np.diff(sv))
    idx = np.flatnonzero(dif < a.tol)
    print("    pairs closer than %g in float64: %d" % (a.tol, idx.size))

    # --- recompute those pairs at 60 digits
    print("\n=== C: do they survive 60-digit arithmetic? ===")
    try:
        from mpmath import mp, cos, pi, mpf
    except ImportError:
        print("    mpmath not installed - cannot run the decisive test")
        return 1
    mp.dps = 60

    def exact(lab):
        x = mpf(1)
        for q, r in zip(gears, lab):
            x *= (q - 2) if r == 0 else -2 * cos(2 * pi * r / q)
        return x

    worst = None
    nexact = 0
    for i in idx:
        l1 = labels[order[i]]
        l2 = labels[order[i + 1]]
        d = abs(exact(l1) - exact(l2))
        if d == 0:
            nexact += 1
        if worst is None or d < worst[0]:
            worst = (d, l1, l2)
    if idx.size:
        print("    smallest 60-digit separation over all %d pairs: %s" % (idx.size, worst[0]))
        print("      between labels %s and %s" % (worst[1], worst[2]))
    gate(nexact == 0,
         "m%d: NONE of the %d float64 near-collisions is an exact tie at 60 digits"
         % (a.y, idx.size))
    if idx.size:
        gate(float(worst[0]) > 1e-40,
             "m%d: every near-collision separates by more than 1e-40" % a.y)

    # --- crowding is the explanation: measure the local density
    print("\n=== D: crowding, measured ===")
    med = np.median(dif[dif > 0])
    print("    median adjacent spacing of the desym spectrum: %.3e" % med)
    print("    smallest 1%% of spacings below: %.3e" % np.quantile(dif, 0.01))
    print("    fraction of levels with |lambda| < 1: %.4f" % np.mean(np.abs(vals) < 1))
    print("    expected pairs within %g if the %d levels were locally uniform at"
          % (a.tol, vals.size))
    print("    the observed density near the crowded region: order %d"
          % int(np.sum(dif < a.tol * 1000) * a.tol / (a.tol * 1000)))

    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
