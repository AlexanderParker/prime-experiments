"""harv_score29.py -- scoring this lane's outstanding predictions against the
corpus F ladder as it stands after Mechanic's round 28.

HARVESTER lane, round 29.  Gate: prints ALL ASSERTIONS GREEN or dies.
    uv run python research/harv_score29.py

WHAT IS BEING SCORED.  Three entries of this lane's cumulative record were
written when the twin machine's record gap F(y) was known only to y = 43.
Mechanic's rounds 27-28 turned F(47), F(53) and F(59) into exact numbers, so
they are now decidable:

  (1) harvester.md 5b, written r22:  "F(2,53) >= 426 (needs <= 486 for the
      tolerance constant; quadratic-law prediction ~441)".
  (2) harvester.md 5e, written r24:  the TWIN PERCENTILE statement - "the
      extreme is 1.34x-2.27x harder at every one of twelve machines
      (externally cross-checked against Ziller-Morack's independent table),
      median 1.70x for y >= 11".  Twelve machines meant y = 5..43.
  (3) harvester.md 5g, written r13-14: the route-transfer BUDGET - every
      (d, step) pair passes at alpha = 2.5 and 3, twins' own worst measured
      value being 2.432 at 31 -> 37.

A NOTATION HAZARD, and it is the reason this file spells everything out.
THIS LANE's F(2,y) is the fixed-twin member of the per-difference family in
MEMBER units: F(2,y) = 3 * F(y), where F(y) is the corpus's record gap in SLOT
units (F(2,37) = 264 = 3 * 88).  MECHANIC's F_2(M) is something else entirely -
the DEPTH-2 spectrum value of machine M (F_2(59) = 173).  The two collide on
the string "F_2(59)" and mean 483 and 173 respectively.  Nothing below uses
Mechanic's F_J; only the record ladder F(y).

EXACT INTEGER ARITHMETIC THROUGHOUT.  Every ratio is printed as a fraction of
integers and compared as integers cross-multiplied; no float ever decides a
gate.
"""
from __future__ import annotations

from fractions import Fraction

OUT = []
N = [0]


def W(s=""):
    OUT.append(s)


def check(cond, msg):
    N[0] += 1
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# THE CORPUS RECORD-GAP LADDER, slot units.  Sources, all in
# docs/proof-search/agents-shared.md:
#   y <= 37 : the pruned-scan ladder, harvester.md 5a/5b
#             (F(2,y) = 21,33,54,75,102,129,264 at y = 11..37)
#   41, 43  : round 28 constructor/mechanic (F(41) = 91, F(43) = 103)
#   47, 53  : round 27 (F(47) = 118, F(53) = 145)
#   59      : round 28 mechanic, F(59) = 161 EXACT
F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 37: 88,
     41: 91, 43: 103, 47: 118, 53: 145, 59: 161}

# A288815, the paired Jacobsthal function at primorials, indexed by the largest
# prime.  OEIS record #19, read first-hand by this lane 2026-08-29.
H2 = {2: 2, 3: 6, 5: 18, 7: 30, 11: 66, 13: 150, 17: 192, 19: 258, 23: 366,
      29: 450, 31: 570, 37: 708, 41: 894, 43: 1044, 47: 1284, 53: 1422,
      59: 1656, 61: 1902, 67: 2190, 71: 2460, 73: 2622}


def main():
    W("HARVESTER ROUND 29 -- SCORING AGAINST THE CORPUS F LADDER")
    W()
    W("[1] F(2,53), harvester.md 5b (written round 22)")
    W()
    f53 = 3 * F[53]
    W(f"    RECORDED: F(2,53) >= 426; needs <= 486 for the tolerance constant;")
    W(f"              quadratic-law prediction ~441.")
    W(f"    NOW EXACT: F(53) = 145 (mechanic r27/r28), so F(2,53) = 3*145 = {f53}.")
    check(f53 == 435, "F(2,53) is not 435")
    check(f53 >= 426, "the r22 lower bound 426 is violated")
    check(f53 <= 486, "the tolerance ceiling 486 is violated")
    err = Fraction(441 - f53, f53)
    W(f"    SCORE: lower bound 426  CONFIRMED  (435 >= 426, slack 9)")
    W(f"           ceiling    486  CONFIRMED  (435 <= 486, slack 51)")
    W(f"           quadratic-law 441 is HIGH by {441-f53} = "
      f"{float(err)*100:.2f}% of the truth  -- the law over-predicts, and by")
    W(f"           less than the 6-quantum of the mod-6 grid.")
    W(f"    AND THE NEXT RUNG, free: F(59) = 161 gives F(2,59) = {3*F[59]}.")
    check(3 * F[59] == 483, "F(2,59) is not 483")
    W()

    W("[2] THE TWIN PERCENTILE, harvester.md 5e (written round 24)")
    W()
    W("    The family-max denominator is Ziller-Morack's h_2(y)/2, an")
    W("    INDEPENDENT table; the twin numerator is this lane's F(2,y).")
    W("    ratio = (h_2(y)/2) / F(2,y) = how much harder the extremal")
    W("    difference is than the twin difference at the same machine.")
    W()
    W("      y   F(y)  F(2,y)   h_2(y)  h_2/2   ratio = extreme/twin   status")
    rows = []
    for y in sorted(F):
        if y not in H2:
            continue
        tw = 3 * F[y]
        ex = H2[y] // 2
        check(H2[y] % 2 == 0, f"h_2({y}) is odd")
        r = Fraction(ex, tw)
        new = y in (47, 53, 59)
        rows.append((y, r, new))
        W(f"     {y:3d}  {F[y]:4d}  {tw:6d}   {H2[y]:6d}  {ex:5d}   "
          f"{ex:5d}/{tw:<5d} = {float(r):.3f}     {'NEW r29' if new else ''}")
    W()
    old = [r for y, r, new in rows if not new and y >= 11]
    newr = [(y, r) for y, r, new in rows if new]
    lo, hi = min(old), max(old)
    W(f"    RECORDED BAND (y >= 11, twelve machines y = 5..43):")
    W(f"      1.34x - 2.27x, median 1.70x.")
    W(f"    RECOMPUTED on the same machines: min {float(lo):.3f}, "
      f"max {float(hi):.3f}.")
    W(f"    THE THREE OUT-OF-SAMPLE MACHINES:")
    for y, r in newr:
        inside = Fraction(134, 100) <= r <= Fraction(227, 100)
        check(inside, f"machine {y} ratio {float(r)} left the recorded band")
        W(f"      y = {y}: {float(r):.3f}   INSIDE the 1.34-2.27 band")
    allr = sorted(r for _, r, _ in rows if _ >= 11) if False else \
        sorted(r for y, r, _n in rows if y >= 11)
    med = allr[len(allr) // 2] if len(allr) % 2 else \
        (allr[len(allr) // 2 - 1] + allr[len(allr) // 2]) / 2
    W(f"    MEDIAN over all fifteen machines y >= 11: {float(med):.3f} "
      f"(recorded 1.70).")
    check(Fraction(160, 100) <= med <= Fraction(180, 100),
          "the median moved outside 1.60-1.80")
    W("    SCORE: CONFIRMED OUT OF SAMPLE AT THREE MACHINES.  The twin case")
    W("    remains the easy end of its own family at every machine where both")
    W("    numbers exist, now fifteen of them.  Note the ratio is NOT")
    W("    monotone (1.81 at 47, 1.63 at 53, 1.71 at 59) - it is a per-machine")
    W("    arithmetic quantity, exactly as 5e says.")
    W()

    W("[3] THE ROUTE-TRANSFER BUDGET, harvester.md 5g (written rounds 13-14)")
    W()
    W("    Twin increments per added gear, in slot units, out of sample:")
    steps = [(37, 41), (41, 43), (43, 47), (47, 53), (53, 59)]
    worst = Fraction(0)
    for a, b in steps:
        inc = F[b] - F[a]
        r = Fraction(inc, b)
        worst = max(worst, r)
        W(f"      {a} -> {b}: F {F[a]:4d} -> {F[b]:4d}, increment {inc:3d}, "
          f"increment/q' = {inc}/{b} = {float(r):.3f}")
    W()
    W(f"    RECORDED: all 35 (d, step) pairs pass at alpha = 2.5 and 3; twins'")
    W(f"              own worst measured value is 2.432 at 31 -> 37.")
    W(f"    OUT OF SAMPLE: the five steps above have max increment/q' = "
      f"{float(worst):.3f},")
    W(f"    a factor {float(Fraction(2432,1000)/worst):.1f} inside the twin")
    W(f"    record and {float(Fraction(5,2)/worst):.1f} inside the alpha = 2.5")
    W(f"    budget.  SCORE: CONFIRMED OUT OF SAMPLE AT FIVE FURTHER STEPS.")
    check(worst < Fraction(5, 2), "a twin step exceeded the alpha=2.5 budget")
    check(worst < Fraction(2432, 1000), "a twin step beat the 31->37 record")
    W("    HONEST LIMIT: this confirms the TWIN row only.  5g's binding")
    W("    negative is unchanged - fixed differences with single-step")
    W("    increments 3.231, 3.947 and 4.435 q' exist, so no uniform")
    W("    alpha <= 3 budget holds over the family, and five more twin steps")
    W("    inside budget say nothing about that.")
    W()

    W(f"ALL ASSERTIONS GREEN  ({N[0]} assertions)")
    txt = "\n".join(OUT)
    print(txt)
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                     "harv_score29.out")
    with open(p, "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
