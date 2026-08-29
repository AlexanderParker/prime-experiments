"""Round 27 (constructor): THE INCREMENT LAW AND THE MANAGER'S TRIPLE
INEQUALITY, tested exactly.

THE INCREMENT LAW (manager, round-26 derivation probe, a HYPOTHESIS):

    F(M + q') - F_2(M)  <=  s_min(q') = min(2u', q' - 2u')

at every LITERAL step (it fails at the padded 31->37 by +8).

THE TRIPLE INEQUALITY (manager, derivation pass 2): at depth 3 the increment
law is equivalent to a statement about M ALONE - for every adjacent gap
triple (g_L, w, g_R) of M whose middle w is a LETTER (w = +-s mod q'),

    g_L + w + g_R  <=  F_2(M) + s_min(q').

Their probe measured this on the RELAXED superset (letter condition only, no
survivor constraint) at the five steps 11->13 .. 23->29 and found slack ~6.
This script carries it to every step a dictionary reaches, separates the
LITERAL middles from the PADDED ones (w = 0 mod q'), and tests the per-depth
generalisation Q*_J <= F_2 + s_min against R68's exact Q* table.

DATA, all full-period and exact unless marked:
  m11..m23   direct cyclic scan here (numpy, the seam included)
  m29,31,37  Mechanic's exact 4-tuple censuses (gap_tuples_{29,31,37}_4.csv)
  m41        Mechanic's TRANSFER SUPERSET (gap_tuples_41_4_transfer.csv) -
             a superset, so its maximum is an UPPER bound on the true lhs,
             which is the sound direction for testing an upper-bound law.

Gates: F(M) and F_2(M) recovered from each data source and asserted against
the corpus chain before any comparison is made.

Usage:  .venv/Scripts/python.exe research/increment_law.py
"""
import os
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")

KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91, 43: 103, 47: 118, 53: 145}
KNOWN_F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90,
            41: 103, 47: 134, 53: 159}
# R68's exact Q*_J table (research/data/r26_qstar.log)
QSTAR = {11: [11, 8], 13: [16, 18], 17: [25, 25], 19: [31, 33, 34],
         23: [39, 43], 29: [55, 58, 55, 55], 31: [68, 85, 88, 68],
         37: [90, 90, 91]}


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def next_prime(y):
    p = y + 1
    while not is_prime(p):
        p += 1
    return p


def letters(q1):
    u1 = round(q1 / 6)
    return 2 * u1, q1 - 2 * u1


def gaps_of(y):
    gears = [p for p in range(5, y + 1) if is_prime(p)]
    P = prod(gears)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    op = np.flatnonzero(~ex).astype(np.int64)
    return np.diff(np.concatenate([op, [op[0] + P]]))


def triples_from_scan(y):
    d = gaps_of(y)
    T = np.stack([d, np.roll(d, -1), np.roll(d, -2)], axis=1)
    P2 = d + np.roll(d, -1)
    return T, int(d.max()), int(P2.max())


def triples_from_csv(path):
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    T = np.unique(np.concatenate([arr[:, 0:3], arr[:, 1:4]]), axis=0)
    P2 = np.unique(np.concatenate([arr[:, 0:2], arr[:, 1:3], arr[:, 2:4]]),
                   axis=0)
    return T, int(arr.max()), int(P2.sum(axis=1).max())


def main():
    steps = [11, 13, 17, 19, 23, 29, 31, 37, 41]
    print("=" * 78)
    print("PART A  THE TRIPLE INEQUALITY, exact, at every step a dictionary "
          "reaches")
    print("=" * 78)
    print("  lhs = max over adjacent gap triples (g_L, w, g_R) of M with the")
    print("  stated middle condition, of g_L + w + g_R.  rhs = F_2(M) + "
          "s_min(q').")
    print()
    print("   M    q'  s_min   F_2   rhs |  LITERAL middle   |  PADDED "
          "middle    | ANY letter")
    print("                              |  max  slack  wit  |  max  slack"
          "     wit | max  slack")
    rows = []
    for y in steps:
        q1 = next_prime(y)
        a, b = letters(q1)
        smin = min(a, b)
        if y <= 23:
            T, F, F2 = triples_from_scan(y)
            src = "scan"
        elif y == 41:
            T, F, F2 = triples_from_csv(
                os.path.join(DDIR, "gap_tuples_41_4_transfer.csv"))
            src = "superset"
        else:
            T, F, F2 = triples_from_csv(
                os.path.join(DDIR, "gap_tuples_%d_4.csv" % y))
            src = "exact"
        assert F == KNOWN_F[y], (y, F, "F gate")
        if src != "superset":
            assert F2 == KNOWN_F2[y], (y, F2, "F2 gate")
        F2 = KNOWN_F2[y]
        rhs = F2 + smin
        w = T[:, 1]
        r = w % q1
        lit = (r == a) | (r == b)
        pad = (r == 0)
        tot = T.sum(axis=1)
        out = {}
        for name, mask in (("lit", lit), ("pad", pad), ("any", lit | pad)):
            if mask.any():
                i = int(np.argmax(np.where(mask, tot, -1)))
                out[name] = (int(tot[i]), tuple(int(v) for v in T[i]))
            else:
                out[name] = (None, None)
        rows.append((y, q1, smin, F2, rhs, out, src))
        def fmt(k):
            v, wit = out[k]
            if v is None:
                return "  none            "
            return "%4d %+5d  %-12s" % (v, rhs - v, "%d,%d,%d" % wit)
        print("  %3d %4d %5d %5d %5d |  %s|  %s| %4d %+5d   [%s]"
              % (y, q1, smin, F2, rhs, fmt("lit"), fmt("pad"),
                 out["any"][0], rhs - out["any"][0], src))

    print()
    print("  VERDICT, literal middles: %s"
          % ("HOLDS at every step"
             if all(r[5]["lit"][0] is None or r[5]["lit"][0] <= r[4]
                    for r in rows) else "FAILS somewhere - see the table"))
    bad = [r[0] for r in rows
           if r[5]["any"][0] is not None and r[5]["any"][0] > r[4]]
    print("  VERDICT, any legal middle (literal or padded): %s"
          % ("HOLDS at every step" if not bad
             else "FAILS at m%s" % ", m".join(map(str, bad))))
    print("  slack sequence (literal middles), m11..m41: %s"
          % ", ".join("%+d" % (r[4] - r[5]["lit"][0])
                      if r[5]["lit"][0] is not None else "n/a" for r in rows))

    print()
    print("=" * 78)
    print("PART D  A FREE REDUCTION OF THE DEPTH-3 OBLIGATION")
    print("=" * 78)
    print("  For any triple of consecutive gaps, (g_L + w) and (w + g_R) are")
    print("  2-gap windows of M, so both are <= F_2(M).  Hence, with NO")
    print("  hypothesis at all,")
    print()
    print("      g_L + w + g_R  <=  F_2(M) + min(g_L, g_R).")
    print()
    print("  So the triple inequality HOLDS AUTOMATICALLY at any triple whose")
    print("  smaller flank is <= s_min(q'), and the whole depth-3 obligation")
    print("  reduces to the triples whose BOTH flanks exceed s_min.  Below:")
    print("  Delta_3 = max over legal triples of (span - F_2) - the quantity")
    print("  the law must cap by s_min - and the min-flank at the argmax.")
    print()
    print("   M    q'  s_min  Delta_3  minflank@argmax  free?  #legal triples"
          "  #with both flanks > s_min")
    for (y, q1, smin, F2, rhs, out, src) in rows:
        if src == "superset":
            continue
        if y <= 23:
            T, _, _ = triples_from_scan(y)
        else:
            T, _, _ = triples_from_csv(
                os.path.join(DDIR, "gap_tuples_%d_4.csv" % y))
        a, b = letters(q1)
        r = T[:, 1] % q1
        lit = (r == a) | (r == b)
        sel = T[lit]
        tot = sel.sum(axis=1)
        i = int(np.argmax(tot))
        d3 = int(tot[i]) - F2
        mf = int(min(sel[i][0], sel[i][2]))
        both = int(((sel[:, 0] > smin) & (sel[:, 2] > smin)).sum())
        print("  %3d %4d %5d %8d %14d   %-6s %13d %14d"
              % (y, q1, smin, d3, mf, "yes" if mf <= smin else "no",
                 len(sel), both))
    print()
    print("  READING (for the manager's derivation): the free bound already")
    print("  proves the depth-3 case at every step whose extremal legal")
    print("  triple has a small flank; where it does not, Delta_3 is still")
    print("  far under s_min, and Delta_3 itself is BOUNDED BY A CONSTANT")
    print("  (max 4 over all eight steps) while s_min grows linearly in q'.")

    print()
    print("=" * 78)
    print("PART B  the per-depth generalisation  Q*_J <= F_2(M) + s_min(q')")
    print("=" * 78)
    print("  (R68's exact Q* table; Q*_2 = F_2(M) identically, so the J = 2")
    print("  case is trivial and the law is a statement about J >= 3.)")
    print("   M    q'  s_min  F_2  rhs |  Q*_3  Q*_4  Q*_5 | max_J Q*_J = "
          "F(M+q')  excess")
    for y in sorted(QSTAR):
        q1 = next_prime(y)
        a, b = letters(q1)
        smin = min(a, b)
        F2 = KNOWN_F2[y]
        rhs = F2 + smin
        qs = QSTAR[y]
        assert qs[0] == F2, (y, qs[0], F2)
        mx = max(qs)
        assert mx == KNOWN_F[q1], (y, mx, KNOWN_F[q1])
        cells = "".join("%6s" % (qs[j] if j < len(qs) else "-")
                        for j in (1, 2, 3))
        print("  %3d %4d %5d %5d %4d | %s | %10d          %+d   %s"
              % (y, q1, smin, F2, rhs, cells, mx, mx - rhs,
                 "OK" if mx <= rhs else "*** FAILS"))

    print()
    print("=" * 78)
    print("PART C  the increment law at the three DEEP steps (corpus anchors)")
    print("=" * 78)
    print("   step        F(M+q')   F_2(M)   diff   s_min(q')   verdict")
    deep = [(41, 43), (43, 47), (47, 53)]
    for y, q1 in deep:
        a, b = letters(q1)
        smin = min(a, b)
        F2 = KNOWN_F2.get(y)
        Fn = KNOWN_F[q1]
        if F2 is None:
            lo = KNOWN_F2[max(k for k in KNOWN_F2 if k < y)]
            print("   %2d->%-2d      %5d    [%d,%d]  <=%3d   %7d       %s"
                  % (y, q1, Fn, lo, Fn, Fn - lo, smin,
                     "HOLDS (F_2 monotone: F_2(%d) >= F_2(%d) = %d)"
                     % (y, max(k for k in KNOWN_F2 if k < y), lo)
                     if Fn - lo <= smin else "UNDECIDED"))
        else:
            print("   %2d->%-2d      %5d    %5d  %5d   %7d       %s"
                  % (y, q1, Fn, F2, Fn - F2, smin,
                     "HOLDS" if Fn - F2 <= smin else "*** FAILS"))
    print()
    print("  (F_2(43) is not on record; the corpus has 103 <= F_2(43) <= 118.")
    print("   F_2 is non-decreasing in the machine - adding a gear deletes")
    print("   openings and can only merge gaps - so F_2(43) >= F_2(41) = 103,")
    print("   which is enough: 118 - 103 = 15 <= 16 = s_min(47).)")
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
