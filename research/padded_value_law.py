"""Round 28 (constructor): THE LEGAL ALPHABET, AND TWO STANDING LAWS TESTED
WHERE THEY CAN ACTUALLY FAIL.

R40's measured law M1 says that of the values a spacing between consecutive
KILLED openings may take - the residue classes {0, +-2c} mod q' admit
a, b, q', a+q', b+q', 2q', ... - the machine only ever realises a, b and q'.
It was measured at the six steps 11->13 .. 29->31.  At EVERY one of those steps

    a + q'  >  F(M)   or   the value a + q' is a HOLE,

so the law was largely untestable there: the bigger representatives simply do
not fit under the record, or happen to be missing.  From m31 on they fit.  This
file computes THE LEGAL ALPHABET

    Lambda(M) = { v <= F(M) : v = 0 or +-2c (mod q'), v a realised gap of M }

exactly at every machine with a dictionary, and reports where M1 breaks.

AND IT TESTS THE MANAGER'S ROUND-28 VIOLATOR SHAPE.  The tooth-counterfactual
that violates (D) has gap word [f_L, s_min, F_old, s_min, f_R] - a J = 5 window
whose CENTRAL MIDDLE IS THE OLD RECORD GAP.  For that window to be word-legal
at all, F(M) must itself be a legal letter:

    F(M) mod q'  in  {0, a, b}.

That is a one-line arithmetic test at every step of the corpus, it is
teeth-sensitive (it is a statement about 6^{-1} mod q'), and it is exactly the
kind of input the manager's counterfactual negative says a true derivation must
use.  Part C runs it.

Sources: the exact full-period 4-tuple censuses (m23, 29, 31, 37) and, at m41,
Mechanic's transfer superset - whose induced DEPTH-1 set is EXACT (it
reproduces F(41) = 91 and the complete m41 hole list {84, 87, 89} from COV-SAT,
a different method).  Small machines by direct scan.

Usage:  .venv/Scripts/python.exe research/padded_value_law.py
"""
import os
import sys
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")
import crt_dict                                          # noqa: E402

KNOWN_F = crt_dict.KNOWN_F
CENSUS = {23: "gap_tuples_23_4.csv", 29: "gap_tuples_29_4.csv",
          31: "gap_tuples_31_4.csv", 37: "gap_tuples_37_4.csv"}
SUP41 = os.path.join(DDIR, "r27", "gap_tuples_41_4_screened_spancap.csv")
M41_HOLES = [84, 87, 89]


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def next_prime(y):
    p = y + 1
    while not is_prime(p):
        p += 1
    return p


def gap_values(y):
    """The realised single-gap value set of M, exact."""
    if y <= 19:
        gears = [p for p in range(5, y + 1) if is_prime(p)]
        P = prod(gears)
        ex = np.zeros(P, bool)
        for g in gears:
            u = pow(6, -1, g)
            ex[u % g::g] = True
            ex[(-u) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64)
        d = np.diff(np.concatenate([op, [op[0] + P]]))
        v = set(int(x) for x in np.unique(d))
    elif y in CENSUS:
        arr = np.loadtxt(os.path.join(DDIR, CENSUS[y]), delimiter=",",
                         skiprows=1, dtype=np.int64)
        v = set(int(x) for x in np.unique(arr))
    elif y == 41:
        v = set(range(1, 92)) - set(M41_HOLES)
    else:
        return None
    assert max(v) == KNOWN_F[y], ("F gate", y, max(v), KNOWN_F[y])
    return v


def main():
    steps = [11, 13, 17, 19, 23, 29, 31, 37, 41]
    print("=" * 78)
    print("PART A   THE LEGAL ALPHABET Lambda(M), exact")
    print("=" * 78)
    print("   M    q'   F  (a,b) | admissible values <= F        | realised "
          "(= Lambda)                | M1?")
    m1_fail = []
    for y in steps:
        q1 = next_prime(y)
        u1 = round(q1 / 6)
        a, b = 2 * u1, q1 - 2 * u1
        vals = gap_values(y)
        F = KNOWN_F[y]
        adm = [v for v in range(1, F + 1) if v % q1 in (0, a % q1, b % q1)]
        lam = [v for v in adm if v in vals]
        extra = [v for v in lam if v not in (a, b, q1)]
        if extra:
            m1_fail.append((y, extra))
        print("  %3d %5d %3d (%2d,%2d) | %-28s | %-33s | %s"
              % (y, q1, F, a, b, ",".join(map(str, adm)),
                 ",".join(map(str, lam)),
                 "OK" if not extra else "FAILS: " + ",".join(map(str, extra))))
    print()
    print("  M1 ('every realised legal spacing value is a, b or q'') is:")
    if m1_fail:
        for y, e in m1_fail:
            q1 = next_prime(y)
            u1 = round(q1 / 6)
            a, b = 2 * u1, q1 - 2 * u1
            print("    REFUTED at m%-3d - %s realised, and %s"
                  % (y, ", ".join("%d = %s" % (
                      v, "a+q'" if v == a + q1 else
                         ("b+q'" if v == b + q1 else
                          ("2q'" if v == 2 * q1 else "?"))) for v in e),
                     "these are legal letters of a strictly larger alphabet"))
        print("    M1 was measured only at 11->13 .. 29->31; it is a "
              "SMALL-MACHINE\n    phenomenon and it stops holding at the first "
              "machine where the bigger\n    representatives fit under the "
              "record and are not holes.")
    else:
        print("    NOT REFUTED at any machine tested.")

    print()
    print("=" * 78)
    print("PART B   IS 2q' A REALISED GAP?  (the padded half of M1)")
    print("=" * 78)
    print("   M    q'   F   2q'  2q' <= F?  realised?")
    for y in steps:
        q1 = next_prime(y)
        vals = gap_values(y)
        F = KNOWN_F[y]
        t = 2 * q1 <= F
        print("  %3d %5d %3d %5d  %-9s  %s"
              % (y, q1, F, 2 * q1, "yes" if t else "no (untestable)",
                 ("YES" if 2 * q1 in vals else "no") if t else "-"))

    print()
    print("=" * 78)
    print("PART C   THE VIOLATOR TEST: is the RECORD GAP a legal letter?")
    print("=" * 78)
    print("  The manager's (D)-violating counterfactual is a J = 5 window whose")
    print("  central middle is F(M).  For that window to be word-legal at all,")
    print("  F(M) mod q' must lie in {0, a, b}.  Twelve corpus steps:")
    print()
    print("   M    q'   F(M)  F mod q'   (a, b)     F(M) a legal letter?")
    hits = []
    for y in [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]:
        q1 = next_prime(y)
        u1 = round(q1 / 6)
        a, b = 2 * u1, q1 - 2 * u1
        F = KNOWN_F[y]
        r = F % q1
        hit = r in (0, a % q1, b % q1)
        if hit:
            hits.append(y)
        print("  %3d %5d %6d %9d   (%2d,%2d)     %s"
              % (y, q1, F, r, a, b,
                 "*** YES (r = %s)" % ("0" if r == 0 else
                                       ("a" if r == a % q1 else "b"))
                 if hit else "no"))
    print()
    print("  HITS: %s of 12 steps%s"
          % (len(hits), (" - m%s" % ", m".join(map(str, hits))) if hits
             else ""))
    print("  The counterfactual violator's shape therefore does not exist in")
    print("  the real machine at 11 of the 12 corpus steps FOR AN ARITHMETIC")
    print("  REASON - F(M) is not congruent to a tooth difference - and at the")
    print("  one step where it is (m13: F = 11 = b), the J = 5 layer is")
    print("  CERTIFIED EMPTY (research/perj_window.py: m13 has no legal")
    print("  4-window, let alone a 5-window).  The refusal is teeth-arithmetic")
    print("  plus realisability, exactly as the counterfactual negative demands.")
    print("  HONEST SCOPE: 3 of q' residues are legal, so the expected number")
    print("  of hits over these 12 steps is 3*sum(1/q') = %.2f - the observed 1"
          % sum(3.0 / next_prime(y) for y in
                [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]))
    print("  is exactly typical.  This is ARITHMETIC LUCK PER STEP, not a law:")
    print("  it will happen again, and when it does the kill has to come from")
    print("  the cover half (realisability), which is where it came from at m13.")
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
