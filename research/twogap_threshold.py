"""Round 25 (constructor): WHAT THE TWO-GAP OBLIGATION LOOKS LIKE ONCE IT IS
A COVERING PROBLEM - three exact machine-free measurements.

R58/R59: (D) at a step reduces EXACTLY to the two-gap statement
F_2(M) <= F(M) + q', zero further slack; R55 proved no rearrangement
invariant and no corridor congruence supplies it (both saturate at 2F).
research/crt_dict.py puts the statement in COVERING form, with no period and
no max-plus layer anywhere:

    a pair (g1,g2) is realised  <=>  the CSP "one phase a_q per gear; the 3
    points 0, g1, g1+g2 OPEN; all g1+g2-2 interior points COVERED" is feasible

so the two-gap law at machine M is exactly the finite statement

    EVERY pair (g1,g2) with g1 + g2 > F(M) + q' has an INFEASIBLE cover CSP.

This script measures the three machine-free instruments that act on THAT form.

(1) CAPACITY.  Gear q at its best allowed phase covers at most
    max_a |cover(q,a) ∩ Y| interior points; if the gears cannot between them
    reach |Y| the pair is refuted with no search at all.  Measured: how many
    of the over-budget pairs does capacity alone kill?

(2) THE FIRST MOMENT.  Independence model, each slot open with probability
    rho = prod (1 - 2/q):

        E_1(S) = P rho^2 (1-rho)^(S-1)          one gap of span S
        E_2(S) = P (S-1) rho^3 (1-rho)^(S-2)    two adjacent gaps summing to S

    F_model = max{S : E_1 >= 1}, F_2model = max{S : E_2 >= 1}.  The model's
    OWN two-gap statement is F_2model <= F_model + q'.

(3) THE ASYMPTOTIC SHAPE of the model increment.  Setting both expectations
    to 1 gives  F_2model - F_model  ~  ln((S-1) rho/(1-rho)) / ln(1/(1-rho)),
    i.e. of order log^2(y) * log(F) - a POLYLOG quantity - against a budget
    q' of order y.  Extending the model to large y measures the ratio.

Usage: python research/twogap_threshold.py [--ymax 20000]
"""
import os
import sys
from math import log, prod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crt_dict import KNOWN_F, KNOWN_F2, gears_of, _inv6, primes  # noqa: E402

NEXTP = {11: 13, 13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37, 37: 41,
         41: 43}


def capacity(y, X, S):
    """(need, capacity) for the pattern with open points X inside [0,S]."""
    qs = gears_of(y)
    xs = set(X)
    Y = [t for t in range(1, S) if t not in xs]
    need = len(Y)
    cap = 0
    for q in qs:
        u = _inv6(q)
        forb = set()
        for x in X:
            forb.add((u - x) % q)
            forb.add((-u - x) % q)
        best = 0
        for a in range(q):
            if a in forb:
                continue
            c = sum(1 for t in Y if (a + t - u) % q == 0 or
                    (a + t + u) % q == 0)
            best = max(best, c)
        cap += best
    return need, cap


def model(y):
    qs = primes(5, y)
    P = prod(qs)
    rho = prod(1 - 2 / q for q in qs)
    lp = log(P)
    lr = log(rho)
    l1 = -log(1 - rho)
    # E_1(S) >= 1  <=>  S <= 1 + (lnP + 2 ln rho)/(-ln(1-rho))     closed form
    f1 = int(1 + (lp + 2 * lr) / l1)
    # E_2(S) >= 1  <=>  lnP + ln(S-1) + 3 ln rho - (S-2)(-ln(1-rho)) >= 0
    lo, hi = 2, f1 * 4 + 1000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if lp + log(mid - 1) + 3 * lr - (mid - 2) * l1 >= 0:
            lo = mid
        else:
            hi = mid - 1
    return f1, lo, rho


def next_prime(y):
    n = y + 1
    while any(n % d == 0 for d in range(2, int(n ** 0.5) + 1)):
        n += 1
    return n


def main():
    ymax = (int(sys.argv[sys.argv.index("--ymax") + 1])
            if "--ymax" in sys.argv else 4001)
    print("THE TWO-GAP OBLIGATION IN COVERING FORM\n")

    print("(1) CAPACITY over ALL over-budget pairs  (a pair is 'killed by "
          "capacity' when the gears\n    cannot cover its interior even at "
          "their individually best phases)")
    print("  M    q'  budget  over-budget pairs  killed by capacity  worst "
          "cap/need   best cap/need")
    for y in sorted(NEXTP):
        if y not in KNOWN_F:
            continue
        q1, F = NEXTP[y], KNOWN_F[y]
        B = F + q1
        pairs = [(g1, S - g1) for S in range(B + 1, 2 * F + 1)
                 for g1 in range(max(1, S - F), min(F, S - 1) + 1)]
        if not pairs:
            print("  %-4d %-3d %6d  none (budget %d > 2F = %d): the two-gap "
                  "statement is FREE" % (y, q1, B, B, 2 * F))
            continue
        killed = 0
        lo = hi = None
        for g1, g2 in pairs:
            need, cap = capacity(y, [0, g1, g1 + g2], g1 + g2)
            r = cap / need
            killed += (r < 1)
            lo = r if lo is None else min(lo, r)
            hi = r if hi is None else max(hi, r)
        print("  %-4d %-3d %6d %18d %19d  %11.3f %15.3f"
              % (y, q1, B, len(pairs), killed, lo, hi))
    print("\n  Capacity kills only the pairs where BOTH gaps are near F; the "
          "asymmetric\n  over-budget pairs (one big, one small) survive it "
          "everywhere.  Since the two-gap\n  law must refute ALL of them, "
          "capacity is not a supplier - but it is not vacuous\n  either, which "
          "X12 (local form) did not distinguish.\n")

    print("(2) FIRST-MOMENT MODEL vs the machine, at the measured machines")
    print("  M     rho      F_model  F_true   F2_model  F2_true   model "
          "incr  true incr   q'   model incr/q'  true incr/q'")
    for y in sorted(NEXTP):
        if y not in KNOWN_F2:
            continue
        q1 = NEXTP[y]
        f1, f2, rho = model(y)
        mi, ti = f2 - f1, KNOWN_F2[y] - KNOWN_F[y]
        print("  %-4d %.5f %8d %7d %10d %8d %11d %10d %5d %13.3f %13.3f"
              % (y, rho, f1, KNOWN_F[y], f2, KNOWN_F2[y], mi, ti, q1,
                 mi / q1, ti / q1))
    print("\n  In the model the two-gap statement HOLDS at every machine "
          "(model incr < q' always),\n  and the true increment is smaller "
          "still.  So independence - unlike the histogram\n  (X35) and the "
          "corridor (X34) - does NOT saturate: it is the first machine-free\n"
          "  instrument that gets the two-gap law RIGHT.\n")

    print("(3) ASYMPTOTIC SHAPE of the model increment (model only - no "
          "machine data)")
    print("     y      rho       F_model  F2_model  incr   q'   incr/q'")
    ys = [p for p in primes(11, ymax)]
    pick = [ys[i] for i in range(0, len(ys), max(1, len(ys) // 14))]
    for y in pick:
        f1, f2, rho = model(y)
        q1 = next_prime(y)
        print("  %6d  %.6f %9d %9d %6d %5d %9.4f"
              % (y, rho, f1, f2, f2 - f1, q1, (f2 - f1) / q1))
    print("\n  incr/q' DECAYS: the model's two-gap increment is a POLYLOG "
          "quantity\n  (~ log(F)/log(1/(1-rho)) ~ log^2(y) log(y)) against a "
          "budget q' ~ y.  The two-gap\n  statement therefore gets EASIER with "
          "y in the model, by an unbounded factor -\n  R31's 'deeper cases are "
          "the easier ones', now for layer 0 and in closed form.")


if __name__ == "__main__":
    main()
