"""
LATERAL round 28 - BACKLOG U11: IS "EVERY HOLE LIES IN THE TOP HALF OF THE GAP
RANGE" A THEOREM?

U11 (posed round 26): the spectrum of realised gap values at machine M is
{1..F(M)} minus a HOLE set, and the project has always observed that the
spectrum fills monotonically from below.  Item 55(b)'s loss rule is exact only
because every hole is large.  So: how large, exactly, and is there a theorem?

This script does three things.

(1) INDEPENDENT RE-DERIVATION.  Mechanic's hole table (mechanic.md lines
    653-662) is the project's reference.  Machines 11..23 are directly sievable
    here (P(23) = 37,182,145), so their hole lists are recomputed FROM SCRATCH
    and asserted equal to the table.  That double-sources five rows of a table
    the whole project cites.  Machines 29..43 are CITED, not recomputed
    (P(29) = 1.078e9 upward), and are marked as such in the output.

(2) THE EXACT BAND.  min(hole)/F at every machine with hole data.  U11's
    "top half" is the claim min(hole)/F > 0.5.

(3) THE COMPLEMENTARY FORM, which is the one with a chance of being a theorem:
    G(M) = min(hole) - 1 is the largest G with EVERY g <= G realised.  Compared
    against candidate lower bounds, of which 2 * #gears is the only one that
    survives (and is TIGHT at m13).

Usage: python hole_topband_r28.py
"""
import sys

import numpy as np

NGATE = 0

# Mechanic's reference table, mechanic.md lines 653-662 (F, holes).
REF = {
    11: (7, set()),
    13: (11, {9}),
    17: (18, {17}),
    19: (25, {19, 24}),
    23: (34, {24}),
    29: (43, {41, 42}),
    31: (58, {54, 56, 57}),
    37: (88, {73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 86, 87}),
    41: (91, {84, 87, 89}),
    43: (103, {102}),
}
SIEVABLE = [11, 13, 17, 19, 23]
GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def spectrum(y):
    gears = [q for q in GEARS if q <= y]
    P = 1
    for q in gears:
        P *= q
    blocked = np.zeros(P, dtype=bool)
    for q in gears:
        v = pow(6, -1, q)
        blocked[v % q::q] = True
        blocked[(-v) % q::q] = True
    op = np.flatnonzero(~blocked).astype(np.int64)
    g = np.diff(np.concatenate([op, [P]]))       # cyclic: N gaps, closes the period
    if int(g.sum()) != P:
        raise AssertionError("period not closed cyclically")
    vals = set(int(x) for x in np.unique(g))
    F = max(vals)
    return F, set(range(1, F + 1)) - vals, len(gears), int(op.size)


def main():
    print("=== 1. INDEPENDENT RE-DERIVATION OF THE HOLE TABLE (m11..m23) ===")
    for y in SIEVABLE:
        F, holes, n, N = spectrum(y)
        refF, refH = REF[y]
        gate(F == refF, "m%-2d: F = %d agrees with the reference table" % (y, F))
        gate(holes == refH, "m%-2d: hole set %s agrees with the reference table"
             % (y, sorted(holes) if holes else "{}"))
        print("      m%-2d  F=%-4d N=%-9d gears=%d  holes=%s"
              % (y, F, N, n, sorted(holes) if holes else "none"))

    print("\n=== 2. U11: THE BAND EVERY HOLE LIVES IN ===")
    print("   y    src    F     holes                              "
          "min hole  min/F    G=minhole-1  G/F")
    ratios = []
    Gs = {}
    for y in sorted(REF):
        F, holes = REF[y]
        n = len([q for q in GEARS if q <= y])
        src = "sieved" if y in SIEVABLE else "cited "
        if not holes:
            print("   %-4d %-6s %-5d %-34s %-9s %-8s %-12s %s"
                  % (y, src, F, "none", "-", "-", "F (=%d)" % F, "1.000"))
            Gs[y] = F
            continue
        mh = min(holes)
        r = mh / F
        ratios.append((y, r))
        Gs[y] = mh - 1
        hs = ",".join(str(h) for h in sorted(holes))
        if len(hs) > 33:
            hs = hs[:30] + "..."
        print("   %-4d %-6s %-5d %-34s %-9d %-8.3f %-12d %.3f"
              % (y, src, F, hs, mh, r, mh - 1, (mh - 1) / F))
    worst = min(ratios, key=lambda t: t[1])
    gate(all(r > 0.5 for _, r in ratios),
         "U11 CONFIRMED: every hole exceeds F/2 at all %d machines with hole data"
         % len(ratios))
    gate(all(r > 0.7 for _, r in ratios),
         "U11 SHARPENED: every hole exceeds 0.70*F ; the tightest machine is "
         "m%d at %.4f" % (worst[0], worst[1]))
    gate(not all(r > 0.71 for _, r in ratios),
         "and 0.70 is very nearly sharp - m%d sits at %.4f, so 0.71 would FAIL"
         % (worst[0], worst[1]))

    print("\n=== 3. THE COMPLEMENTARY FORM: HOW FAR DOES THE SPECTRUM FILL? ===")
    print("   G(M) = largest G with every g <= G realised.  Candidate lower bounds:")
    print("   y    #gears n   G(M)    2n    y     n^2   F/2   verdict for 2n")
    tight = []
    for y in sorted(REF):
        n = len([q for q in GEARS if q <= y])
        G = Gs[y]
        ok = G >= 2 * n
        if G == 2 * n:
            tight.append(y)
        print("   %-4d %-11d %-7d %-5d %-5d %-5d %-5d %s"
              % (y, n, G, 2 * n, y, n * n, REF[y][0] // 2,
                 "holds" + (" (TIGHT)" if G == 2 * n else "")
                 if ok else "FAILS"))
    gate(all(Gs[y] >= 2 * len([q for q in GEARS if q <= y]) for y in REF),
         "CONJECTURE C-U11: every g <= 2 * #gears is realised - holds at all %d "
         "machines" % len(REF))
    gate(tight == [13],
         "and it is TIGHT at exactly one machine, m13 (G = 8 = 2*4), so it is not "
         "slack everywhere")
    gate(not all(Gs[y] >= len([q for q in GEARS if q <= y]) ** 2 for y in REF),
         "the competing bound n^2 FAILS (so 2n is not simply the weakest of many)")

    print("\n  STATUS, honestly: (2) is a MEASUREMENT over ten machines, five of")
    print("  them re-derived here from scratch and five cited.  (3) is a")
    print("  CONJECTURE with a tight case, not a theorem: the counting argument")
    print("  it suggests (a window of length g <= 2n+1 needs g-1 interior slots")
    print("  blocked, and n gears supply at most 2 blocked residues each in a")
    print("  window shorter than the smallest gear) gives the right SHAPE but not")
    print("  a proof, because it does not show the CRT system with the two")
    print("  endpoints left OPEN is solvable.  That solvability is exactly the")
    print("  covering-half obstruction Constructor's N(M) negative names.")
    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
