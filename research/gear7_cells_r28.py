"""
LATERAL round 28 - BACKLOG U7: WHICH CELL ORBIT CARRIES THE GEAR-7 DRIFT?

U7 was posed in round 25 and RE-POSED in round 26 after item 56 answered its
parity half for every gear at once.  The surviving question is not "is gear 7
parity-obstructed" but:

    WHICH mirror ORBIT of gear 7's cell matrix carries the measured asymmetry,
    and why does its magnitude decay so much more slowly than gear 5's?

THE OBJECT, stated correctly (round 25 indexed it by (start phase, exposed-step
count); the equivalent and cleaner indexing is by ENDPOINTS).  Fix a gear p with
teeth at +-v_p, so openings live on the exposed set A_p = Z_p \\ {+-v_p},
|A_p| = p-2, and A_p is closed under negation.  For consecutive openings o -> o'
define

    C[a][b] = # { gaps with o = a, o' = b  (mod p) },   a, b in A_p,

a (p-2) x (p-2) integer matrix.  Its row and column sums are EXACTLY N/(p-2) by
CRT.  The mirror k -> -k sends the gap (o, o') to (-o', -o), so

    C[a][b] = C[-b][-a]      exactly, at every machine,

an involution of the (p-2)^2 cells whose fixed cells are the anti-diagonal
b = -a (there are p-2 of them, one per row, because A_p is negation-closed).
Hence (p-2)(p-1)/2 orbits and, after the row sums, (p-2)(p-3)/2 free integers:
THREE at p = 5, TEN at p = 7 - the "10 free integers" U7 names.

THE DRIFT.  The gap-length classes are N_v = sum over the diagonal b - a = v of
C[a][b], and the asymmetries are alpha_v = N_v - N_{-v}.  The mirror maps the
diagonal {b - a = v} to itself, so it does NOT force alpha_v = 0 - it only forces
parities.  This script therefore measures, exactly:

  (A) the cell matrix at gear 5 and gear 7, m11..m23, with the mirror relation
      and the row sums asserted;
  (B) the DEVIATION D = C - N/(p-2)^2 decomposed into mirror orbits, ranked -
      which orbit carries the drift;
  (C) the decay of each orbit's deviation with the machine, gear 5 against
      gear 7 - the "why so much slower" half.

Usage: python gear7_cells_r28.py [--upto 23]
"""
import argparse
import sys
from fractions import Fraction

import numpy as np

GEARS = [5, 7, 11, 13, 17, 19, 23]

NGATE = 0


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def openings(gears, P):
    blocked = np.zeros(P, dtype=bool)
    for q in gears:
        v = pow(6, -1, q)
        blocked[v % q::q] = True
        blocked[(-v) % q::q] = True
    return np.flatnonzero(~blocked).astype(np.int64)


def cell_matrix(op, P, p):
    """C[a][b] over the exposed set of gear p, cyclically closed."""
    v = pow(6, -1, p)
    A = sorted(r for r in range(p) if r % p != v % p and r % p != (-v) % p)
    idx = {r: i for i, r in enumerate(A)}
    m = len(A)
    start = op % p
    end = np.roll(op, -1) % p          # cyclic: last gap wraps to o_0 = 0
    C = np.zeros((m, m), dtype=np.int64)
    np.add.at(C, (np.array([idx[int(x)] for x in start]),
                  np.array([idx[int(x)] for x in end])), 1)
    return A, idx, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=23)
    a = ap.parse_args()

    hist = {5: [], 7: []}
    for n in range(3, len(GEARS) + 1):
        gears = GEARS[:n]
        y = gears[-1]
        if y > a.upto:
            break
        P = int(np.prod(gears))
        op = openings(gears, P)
        N = op.size
        for p in (5, 7):
            A, idx, C = cell_matrix(op, P, p)
            m = len(A)
            gate(int(C.sum()) == N,
                 "m%-2d gear %d: the %d cells total N = %d (period closed cyclically)"
                 % (y, p, m * m, N))
            gate(bool((C.sum(axis=1) == N // m).all()) and
                 bool((C.sum(axis=0) == N // m).all()),
                 "m%-2d gear %d: every row and column sum is exactly N/(p-2) = %d"
                 % (y, p, N // m))
            neg = np.array([idx[(-r) % p] for r in A])
            gate(bool(np.array_equal(C, C[neg][:, neg].T)),
                 "m%-2d gear %d: the mirror relation C[a][b] = C[-b][-a] holds "
                 "cell for cell" % (y, p))
            # orbits
            orbits = {}
            for i in range(m):
                for j in range(m):
                    k = tuple(sorted([(i, j), (neg[j], neg[i])]))
                    orbits.setdefault(k, []).append((i, j))
            fixed = [k for k in orbits if len(set(k)) == 1]
            gate(len(orbits) == m * (m + 1) // 2 and len(fixed) == m,
                 "m%-2d gear %d: %d mirror orbits, %d of them fixed cells "
                 "(the anti-diagonal b = -a)" % (y, p, len(orbits), len(fixed)))
            flat = Fraction(N, m * m)
            devs = []
            for k, cells in orbits.items():
                (i, j) = k[0]
                d = Fraction(int(C[i][j])) - flat
                devs.append((abs(d), d, (A[i], A[j]), len(set(k)) == 1))
            devs.sort(key=lambda t: -t[0])
            hist[p].append((y, N, devs, C, A, idx))

    for p in (5, 7):
        print("\n=== GEAR %d: THE ORBIT THAT CARRIES THE DRIFT ===" % p)
        print("  the deviation of each mirror orbit from the CRT-flat value")
        print("  N/(p-2)^2, as a fraction of N.  'fix' marks an anti-diagonal")
        print("  (mirror-fixed) cell.")
        for (y, N, devs, C, A, idx) in hist[p]:
            top = devs[:3]
            print("    m%-2d N=%-9d top orbits: %s"
                  % (y, N, "   ".join("(%d,%d)%s %+.4f" % (t[2][0], t[2][1],
                                                           "fix" if t[3] else "",
                                                           float(t[1]) / N)
                                      for t in top)))
        # is the leading orbit STABLE across machines?
        lead = [devs[0][2] for (_, _, devs, _, _, _) in hist[p]]
        print("  leading orbit by machine: %s  -> %s"
              % (lead, "STABLE" if len(set(lead)) == 1 else "MOVES"))
        print("  max |deviation| / N by machine: %s"
              % ", ".join("m%d %.4f" % (y, float(devs[0][0]) / N)
                          for (y, N, devs, _, _, _) in hist[p]))
        # the length-class asymmetries alpha_v
        print("  gap-length class asymmetries alpha_v = N_v - N_{-v}, /N:")
        for (y, N, devs, C, A, idx) in hist[p]:
            Nv = np.zeros(p, dtype=np.int64)
            for i, ai in enumerate(A):
                for j, bj in enumerate(A):
                    Nv[(bj - ai) % p] += C[i][j]
            al = [(v, int(Nv[v] - Nv[(-v) % p])) for v in range(1, p // 2 + 1)]
            print("    m%-2d %s   max|alpha|/N = %.4f"
                  % (y, "  ".join("a_%d %+d" % t for t in al),
                     max(abs(t[1]) for t in al) / N))

    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
