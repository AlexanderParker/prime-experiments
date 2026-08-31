"""Round 28 (mechanic): THE ANATOMY OF THE INFLATION ONSET, and the free trim
it hands the certificate.

THE DEPTH-0 LEMMA (proof in three lines, no scan).  A transfer emission at
depth 0 - no deleted interiors - is exactly a realised M 4-tuple w, occurring
at some M-opening y_0.  Its 5 exposed points give at most 10 forbidden values
for the new gear's phase A, so for every q' >= 11 there is an admissible A;
and y_0 mod q' runs over ALL residues as y_0 runs over the q' laps of the
M-period (CRT, P(M) invertible mod q').  Hence

        D_4(M)  SUBSET  D_4(M + q')    for every q' >= 11

- the realised 4-tuple dictionary is MONOTONE ALONG THE MACHINE LADDER.  More
generally D_m(M) subset D_m(M+q') whenever q' > 2(m+1).

WHY IT MATTERS.  Constructor's rung-nine oracle has to decide a superset of
m41 4-tuples.  Every candidate that is already in the exact m37 dictionary is
YES BY THEOREM - no solver, no scan.  This script measures that trim, and then
asks whether the inflation onset is simply "the span at which the transfer
starts emitting tuples the old machine did not already have".

Usage:  <venv>/python research/onset_anatomy_r28.py
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)

from dict_transfer import load_dict, transfer          # noqa: E402
from onset_r28 import screen                           # noqa: E402
from onset_ladder_r28 import gaps_cyclic, ktuples      # noqa: E402

F1 = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91}
F4 = {13: 26, 17: 33, 19: 38, 23: 58, 29: 70, 31: 90, 37: 105, 41: 118}
ONSET = {(13, 17): 15, (17, 19): 17, (19, 23): 25, (23, 29): 31,
         (29, 31): 41, (31, 37): 53, (37, 41): 68}


def exact(y):
    if y in (13, 17, 19, 23):
        return ktuples(gaps_cyclic(y), 4)
    return set(load_dict(os.path.join(DATA, "gap_tuples_%d_4.csv" % y)))


def main():
    print("PART A - THE DEPTH-0 LEMMA, CHECKED AGAINST EVERY EXACT PAIR\n")
    D = {y: exact(y) for y in (13, 17, 19, 23, 29, 31, 37)}
    for M, qp in [(13, 17), (17, 19), (19, 23), (23, 29), (29, 31), (31, 37)]:
        miss = D[M] - D[qp]
        print("    D_4(%2d) (%6d)  subset  D_4(%2d) (%6d) : %s"
              % (M, len(D[M]), qp, len(D[qp]),
                 "YES" if not miss else "NO - %d missing %s"
                 % (len(miss), sorted(miss)[:3])))
        assert not miss, "DEPTH-0 LEMMA VIOLATED"
    # the round-27 m41 shard: exact at span <= 77, so the lemma is checkable
    # there for every m37 tuple of span <= 77.
    shard = set(load_dict(os.path.join(DATA, "r27",
                                       "gap_tuples_41_4_exact_le77.csv")))
    sub = {t for t in D[37] if sum(t) <= 77}
    miss = sub - shard
    print("    D_4(37) restricted to span <= 77 (%6d)  subset  the exact m41 "
          "shard (%6d) : %s" % (len(sub), len(shard),
                                "YES" if not miss else "NO (%d)" % len(miss)))
    assert not miss, "DEPTH-0 LEMMA VIOLATED AT 37 -> 41"

    print("\nPART B - THE FREE TRIM IT GIVES THE ORACLE\n")
    for src, out in [("r27/gap_tuples_41_4_screened_spancap.csv", 37),
                     ("r28/gap_tuples_41_4_walkscreened.csv", 37)]:
        p = os.path.join(DATA, src)
        if not os.path.exists(p):
            continue
        cand = set(load_dict(p))
        hit = cand & D[out]
        print("    %-46s %8d candidates, %7d (%.1f%%) are YES BY THEOREM "
              "(already in D_4(%d))" % (src, len(cand), len(hit),
                                        100.0 * len(hit) / len(cand), out))

    print("\nPART C - IS THE ONSET 'THE FIRST NEW TUPLE THAT FAILS'?\n")
    print("    step    onset  | below the onset: candidates  of them OLD "
          "(in D_4(M))  NEW   | min span of a NEW candidate")
    for (M, qp), o in sorted(ONSET.items()):
        if qp == 41:
            cand = [t for t in load_dict(os.path.join(
                DATA, "r27", "gap_tuples_41_4_screened_spancap.csv"))
                if sum(t) <= 77]
        else:
            sup, _, _ = transfer(sorted(D[M]), qp, F4[qp], F1[qp],
                                 verbose=False)
            cand, _ = screen(sup, qp)
        below = [t for t in cand if sum(t) < o]
        old = [t for t in below if t in D[M]]
        newmin = min((sum(t) for t in cand if t not in D[M]), default=None)
        print("    %2d->%2d   %4d  |               %8d         %8d      %6d"
              "   |  %s" % (M, qp, o, len(below), len(old),
                            len(below) - len(old), newmin))


if __name__ == "__main__":
    main()
