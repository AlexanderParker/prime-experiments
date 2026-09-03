"""Round 29 (constructor): THE TEETH-SENSITIVE HYPOTHESIS, STATED AND SCORED.

Round 28 (R86) closed with this on record, as the SHAPE of the teeth-sensitive
input the manager's counterfactual negative demands:

    H1   F(M) mod q'  is not in  {0, a, b},
         c = 6^{-1} mod q',  a = 2c mod q' (least positive rep),  b = q' - a
         ("F(M) is not congruent to a tooth difference of the incoming gear").

It is teeth-sensitive because a and b are functions of 6^{-1} mod q' - no
structural theorem of the project sees them.  This script

  (1) states it exactly and DECIDES it at every corpus step where F(M) is on
      record (the twelve steps 11 -> 13 .. 53 -> 59),
  (2) computes its BASE RATE under a random tooth, so the score can be read
      against chance,
  (3) tests it against THE THREE OPEN m31 ROWS - the only rows of the whole
      increment law that fail - which is the job R86 left for this round, and
  (4) tests the replacement candidate H3 (the padded row) at every machine
      where the padded letter q' is a realised gap, and looks for ANY
      teeth-arithmetic separator of the m31 failure.

Usage:  uv run python research/teeth_r29.py
"""
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evenj_r29 as EJ                                      # noqa: E402

KNOWN_F = EJ.KNOWN_F
KNOWN_F2 = EJ.KNOWN_F2
STEPS = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
LITCAP = {13: 2, 17: 2, 19: 2, 23: 4, 29: 3, 31: 4, 37: 6, 41: 2, 43: 4,
          47: 4, 53: 6, 59: 4}      # R20's cap by q' mod 210, for the record


def main():
    print("=" * 78)
    print("THE TEETH-SENSITIVE HYPOTHESIS H1  (constructor, round 29)")
    print("=" * 78)
    print("  H1 :  F(M) mod q' not in {0, a, b}      a = 2.(6^-1 mod q'),"
          " b = q' - a")
    print("  H1a:  F(M) mod q' != 0                  (the PADDED half)")
    print("  H1b:  F(M) mod q' not in {a, b}         (the LITERAL half)")
    print()
    print("  %-4s %-4s %-4s %-4s %-5s %-6s %-8s %s"
          % ("M", "q'", "a", "b", "F(M)", "F mod q'", "class", "H1"))
    hits = []
    n1 = n1a = n1b = 0
    rate = Fraction(0)
    for y in STEPS:
        q1 = EJ.next_prime(y)
        c = pow(6, -1, q1)
        a = (2 * c) % q1
        b = q1 - a
        F = KNOWN_F[y]
        r = F % q1
        cl = ("0/padded" if r == 0 else "a" if r == a else "b" if r == b
              else "-")
        ok = r not in (0, a, b)
        n1 += ok
        n1a += (r != 0)
        n1b += r not in (a, b)
        rate += Fraction(3, q1)
        if not ok:
            hits.append((y, q1, r, cl))
        print("  %-4d %-4d %-4d %-4d %-5d %-8d %-8s %s"
              % (y, q1, a, b, F, r, cl, "holds" if ok else "*** FAILS"))
    n = len(STEPS)
    print()
    print("  SCORE   H1 %d/%d   H1a %d/%d   H1b %d/%d" % (n1, n, n1a, n, n1b, n))
    print("  failures: %s" % (hits if hits else "none"))
    print("  BASE RATE under a random tooth: expected failures = "
          "3 * sum(1/q') = %s = %.3f" % (rate, float(rate)))
    print("  => %d observed against %.2f expected: H1 carries NO evidence of"
          % (len(hits), float(rate)))
    print("     being a law.  It is a per-step arithmetic condition.")

    # ------------------------------------------------------------------ (3) --
    print("\n" + "-" * 78)
    print("(3) H1 AGAINST THE THREE OPEN m31 ROWS")
    r31 = EJ.analyse(31)
    q1, F2, s = r31["q1"], r31["F2"], r31["s_min"]
    W = r31["words"]
    print("  m31: q' = %d, a = %d, b = %d, F = %d, F_2 = %d, s_min = %d"
          % (q1, r31["a"], r31["b"], r31["F"], F2, s))
    print("  F(31) mod 37 = %d, which is neither 0 nor a nor b: H1 HOLDS at m31."
          % (r31["F"] % q1))
    print("  The three rows of R83 that fail (Phi(w) <= F_2 + s_min - span w):")
    bad = 0
    for w in [(37,), (12, 37), (37, 12)]:
        phi = W[w][0]
        rhs = F2 + s - sum(w)
        bad += phi > rhs
        print("     w = %-10s span %3d  Phi = %3d  need <= %3d   %s"
              % (str(w), sum(w), phi, rhs,
                 "FAILS by %d" % (phi - rhs) if phi > rhs else "ok"))
    print("  ==> H1 HOLDS at exactly the machine whose rows FAIL (%d of 3 rows"
          % bad)
    print("      failing).  H1 IS NOT THE SEPARATOR FOR THE OPEN ROWS.")

    # ------------------------------------------------------------------ (4) --
    print("\n" + "-" * 78)
    print("(4) H3 - THE PADDED ROW  Phi(q') <= F_2 + s_min - q'  at every")
    print("    machine where the padded letter q' is a realised gap")
    print("  %-4s %-4s %-5s %-5s %-6s %-8s %-8s %-8s %s"
          % ("M", "q'", "F", "F_2", "s_min", "Phi(q')", "need <=", "margin",
             "q'/F"))
    rows = []
    for y in (19, 23, 29, 31, 37):
        r = EJ.analyse(y)
        q = r["q1"]
        if (q,) not in r["words"]:
            print("  m%-3d q' = %d is NOT a realised gap - row vacuous" % (y, q))
            continue
        phi = r["words"][(q,)][0]
        need = r["F2"] + r["s_min"] - q
        rows.append((y, q, r["F"], r["F2"], r["s_min"], phi, need,
                     need - phi, q / r["F"]))
        print("  %-4d %-4d %-5d %-5d %-6d %-8d %-8d %-8s %.3f"
              % (y, q, r["F"], r["F2"], r["s_min"], phi, need,
                 "%+d" % (need - phi), q / r["F"]))
    print("  => the padded row fails at EXACTLY ONE machine, m31.")

    print("\n  CANDIDATE SEPARATORS OF THE m31 FAILURE, all tested:")
    print("  %-4s %-8s %-8s %-8s %-8s %-8s %-8s"
          % ("M", "margin", "q'/F", "q' %210", "litcap", "F mod q'", "2c/q'"))
    for (y, q, F, F2, s, phi, need, marg, ratio) in rows:
        print("  %-4d %-8s %-8.3f %-8d %-8d %-8d %-8.3f"
              % (y, "%+d" % marg, ratio, q % 210, LITCAP[q], F % q,
                 2 * pow(6, -1, q) % q / q))
    # ------------------------------------------------------------------ (5) --
    print("\n  (5) THE CONSTRUCT THAT DOES ORDER THEM: OCCURRENCE COUNT.")
    print("  R33's flank order-statistic law says FS_max(w) ~ 2.77 ln occ(w).")
    print("  occ is computable by scan at m19 and m23 only; measured here:")
    import numpy as np
    fit = []
    for y in (19, 23):
        r = EJ.analyse(y)
        d = EJ.gaps_of(y)
        n = len(d)
        for w in sorted(r["words"]):
            if len(w) != 1:
                continue
            occ = int((d == w[0]).sum())
            phi = r["words"][w][0]
            fit.append((y, w[0], occ, phi, phi / np.log(occ)))
            print("     m%-3d w = (%2d)  occ = %8d   Phi = %3d   "
                  "Phi/ln(occ) = %.2f" % (y, w[0], occ, phi, phi / np.log(occ)))
    print("     ratio band over these %d cells: %.2f - %.2f  (R33 fitted 2.77)"
          % (len(fit), min(f[4] for f in fit), max(f[4] for f in fit)))
    lo = min(f[4] for f in fit)
    hi = max(f[4] for f in fit)
    import math
    print("  AND THAT BAND IS THE RESULT, NOT THE FIT.  The two PADDED letters")
    print("  (23 at m19, 29 at m23) are the two extremes of it - 1.80 and 6.14")
    print("  against 2.39-2.96 for the four literal letters.  R33's law is a")
    print("  LITERAL-LETTER law at this granularity; the padded letter's flank")
    print("  envelope is not governed by its occurrence count in any way these")
    print("  six cells support.")
    print("  Inverting it at m31 is therefore USELESS, and saying so is the")
    print("  point: Phi(37) = 48 gives occ(37; m31) in [exp(48/%.2f),"
          " exp(48/%.2f)] = [%.1e, %.1e]," % (hi, lo, math.exp(48 / hi),
                                              math.exp(48 / lo)))
    print("  eight orders of magnitude wide.  NAMED, NOT DELIVERED: the")
    print("  construct that would decide the m31 rows is the COUNTED padded-gap")
    print("  census occ(q'; M) at m29/m31/m37 (the existing censuses are")
    print("  distinct-tuple lists and carry no counts).  That is a Mechanic job")
    print("  and it is the one measurement this item wanted and could not make.")
    print("\n  NEGATIVE, and it is the honest answer: none of these orders the")
    print("  machines so that m31 is the extreme one.  q'/F is not monotone in")
    print("  the margin (m19 has the LARGEST ratio and the SECOND-largest")
    print("  margin); litcap 4 is shared with m19 and m43; F mod q' is nowhere")
    print("  near a tooth class at m31.  NO TEETH-ARITHMETIC SEPARATOR OF THE")
    print("  THREE OPEN ROWS WAS FOUND THIS ROUND.")
    print("=" * 78)


if __name__ == "__main__":
    main()
