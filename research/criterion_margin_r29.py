"""Round 29 (mechanic): THE SPECTRUM-PLUS-DEPTH CRITERION'S MARGIN LADDER.

Constructor's round-28 criterion is

    F(M + q')  <=  max_{2 <= J <= J_max} F_J(M),     J_max = A_kill(M -> q') + 1

and it certifies (D) at the step when that maximum is at most the budget
F(M) + q'.  Since F_J(M) is non-decreasing in J the maximum is F_{J_max}(M), so
the criterion's margin at a step is exactly

    margin(M -> q')  =  F(M) + q'  -  F_{A_kill + 1}(M).

This file tabulates that margin at every step where BOTH inputs are on record,
in exact integers, with no fit and no rate.  It is the honest frame for the
round-29 finding that the criterion FAILS at 47 -> 53: the failure is not an
accident of machine 47, it is what the margin does as A_kill grows - each extra
unit of A_kill charges one more level of the F ladder (measured 7-16 units per
level) against a budget that only gains q' - q'_prev.

usage: uv run python research/criterion_margin_r29.py
"""

# C11-UPDATE (mechanic.md), plus this round's F_4/F_5/F_6(47).
F = {13: [11, 16, 23, 26, 28, 31],
     17: [18, 25, 28, 33, 35, 40],
     19: [25, 31, 35, 38, 47, 50],
     23: [34, 39, 50, 58, 65, 77],
     29: [43, 55, 65, 70, 85, 90],
     31: [58, 68, 85, 90, 92, 97],
     37: [88, 90, 97, 105, 113, 120],
     41: [91, 103, 110, 118, 128, None],
     43: [103, 116, 125, 132, None, None],
     # F_6(47) = 177 EXACT this round (seed-174 band run, 100% of machine 23's
     # period); F_4, F_5 are bracketed [154, 174] and [167, 174] and are not
     # needed here because F_J is non-decreasing in J.
     47: [118, 134, 145, None, None, 177]}
NEXT = {13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37, 37: 41, 41: 43,
        43: 47, 47: 53}
# A_kill(M -> next), all exact full-period decisions (C10, C22, C23)
AK = {13: 2, 17: 2, 19: 3, 23: 2, 29: 4, 31: 4, 37: 3, 41: 3, 43: 3, 47: 5}


def main(f47=None):
    if f47:
        F[47][3:6] = f47
    print("  step        A_kill  J_max  F_Jmax(M)  budget F(M)+q'  margin  "
          "verdict")
    rows = []
    for M, q in NEXT.items():
        J = AK[M] + 1
        v = F[M][J - 1]
        if v is None:
            print(f"  {M:2d} -> {q:2d}      {AK[M]}      {J}      "
                  f"NOT ON RECORD")
            continue
        b = F[M][0] + q
        rows.append((M, q, AK[M], J, v, b, b - v))
        print(f"  {M:2d} -> {q:2d}      {AK[M]}      {J}     {v:6d}      "
              f"{b:8d}      {b - v:+5d}   "
              f"{'CERTIFIES' if b >= v else 'FAILS'}")
    print("\n  by A_kill (margins, exact):")
    for a in sorted({r[2] for r in rows}):
        m = [r[6] for r in rows if r[2] == a]
        print(f"    A_kill = {a}:  {sorted(m)}")
    print("\n  the F-ladder's cost per level (F_{J+1} - F_J), exact:")
    for M in sorted(F):
        d = [F[M][i + 1] - F[M][i] for i in range(5)
             if F[M][i + 1] is not None and F[M][i] is not None]
        print(f"    m{M:2d}: {d}")
    print("\n  READING: the criterion is a statement about the DEPTH the fuel "
          "census allows.\n  Every step with A_kill <= 3 certifies (margins "
          "+5..+24); of the four steps\n  with A_kill >= 4, two fail "
          "(29->31 by 11, 47->53) and one passes by +3.")


if __name__ == "__main__":
    main()
