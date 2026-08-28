"""Round 25 (constructor), cross-lane: LATERAL'S MIRROR PARITY LAW MET WITH THE
SCAN-FREE COUNTER - and the F_2 = 2F endpoint decided.

LATERAL (round 25, proved, research/mirror_cells.py): the opening set of a
machine is exactly closed under k -> -k, 0 is the only fixed slot, and
N = prod(q-2) is odd, so at each depth there is EXACTLY ONE self-mirror window,
at t_j = -j/2 mod N.  At depth 2 the unique self-mirror adjacent pair is forced
to (k_1, k_1) with k_1 < F.  CONSEQUENCE: every adjacent EQUAL pair (g,g) with
g != k_1 occurs an EVEN number of times - in particular (F,F) does.

THE LEVER: to prove an equal pair does not occur it is enough to cap its count
at ONE.  This script supplies both halves from the CRT side, with no period:

  (1) VERIFICATION.  Exact occurrence counts of every equal adjacent pair (g,g)
      at machines 11..29, by the enumerating CRT counter (crt_dict.
      count_solutions, uncapped).  Lateral's law predicts every count is EVEN
      except at the single self-mirror value k_1, where it is ODD.  This is an
      independent check of the law by a method that never touches the period.

  (2) THE ENDPOINT.  Is (F,F) realised at all?  R55 showed the histogram bound
      F + G_2 = 2F because maximal gaps are mirror-paired, and that 2F exceeds
      the budget from 19->23 on.  Deciding (F,F) directly settles whether the
      wall the machine-free instruments hit is even attainable.

Usage: python research/mirror_parity_lever.py [11 13 17 19 23 29 31]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crt_dict                                        # noqa: E402
from crt_dict import KNOWN_F, KNOWN_F2, gears_of       # noqa: E402

NEXTP = {11: 13, 13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37, 37: 41,
         41: 43}
BIG = 20_000_000   # enumeration cap; a capped count is reported, not asserted


def pattern(g1, g2):
    X = [0, g1, g1 + g2]
    Y = [t for t in range(1, g1 + g2) if t not in set(X)]
    return X, Y


def main():
    ys = [int(a) for a in sys.argv[1:] if a.isdigit()] or [11, 13, 17, 19, 23]
    print("MIRROR PARITY MET WITH THE SCAN-FREE COUNTER\n")
    print("(1) exact counts of every adjacent EQUAL pair (g,g), by CRT "
          "enumeration - no period")
    print("  M    F    odd-count values (Lateral: exactly one, = k_1)   "
          "max g with (g,g) realised   time")
    for y in ys:
        qs = gears_of(y)
        F = KNOWN_F[y]
        t0 = time.time()
        odd, maxg, counts, ncap = [], None, {}, 0
        for g in range(1, F + 1):
            X, Y = pattern(g, g)
            c, capped = crt_dict.count_solutions(qs, X, Y, cap=BIG)
            counts[g] = (c, capped)
            if c:
                maxg = g
                if capped:
                    ncap += 1
                elif c % 2:
                    odd.append((g, c))
        print("  %-4d %-4d %-52s %-27s %5.1f s%s"
              % (y, F, str(odd), maxg, time.time() - t0,
                 "  (%d values capped at %d - not asserted)" % (ncap, BIG)
                 if ncap else ""), flush=True)
        assert len(odd) <= 1, ("mirror parity law violated at machine %d: "
                               "odd-count equal pairs %s" % (y, odd))
        if ncap == 0:
            assert len(odd) == 1, ("no self-mirror equal pair found at m%d"
                                   % y)
            assert odd[0][0] < F, ("the self-mirror pair is (F,F) at m%d" % y)
        else:
            assert odd == [] or odd[0][0] < F

    print("\n(2) THE F_2 = 2F ENDPOINT - is (F,F) realised?")
    print("  M    F    2F   budget F+q'   (F,F) count   verdict")
    for y in ys:
        qs = gears_of(y)
        F = KNOWN_F[y]
        q1 = NEXTP[y]
        X, Y = pattern(F, F)
        c, _ = crt_dict.count_solutions(qs, X, Y, cap=BIG)
        print("  %-4d %-4d %-4d %11d %13d   %s"
              % (y, F, 2 * F, F + q1, c,
                 "NOT realised (parity: even, and = 0)" if c == 0
                 else "REALISED %d times" % c))
        assert c % 2 == 0, ("(F,F) has odd count at m%d" % y)

    print("\n(3) the same question one step up the pair ladder: the largest "
          "EQUAL pair sum vs the budget")
    print("  M    max g with (g,g)   2g    budget F+q'   slack   F_2(M)")
    for y in ys:
        qs = gears_of(y)
        F, q1 = KNOWN_F[y], NEXTP[y]
        best = 0
        for g in range(F, 0, -1):
            X, Y = pattern(g, g)
            c, _ = crt_dict.count_solutions(qs, X, Y, cap=1)
            if c:
                best = g
                break
        print("  %-4d %16d %5d %13d %7d %7s"
              % (y, best, 2 * best, F + q1, F + q1 - 2 * best,
                 KNOWN_F2.get(y)))
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
