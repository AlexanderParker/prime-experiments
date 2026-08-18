"""Round 13 (corrected tests): marginal and joint flank feasibility at tier A.

leftF : exists r in S_m(w) with r - F in E_m           (a maximal gap may end
                                                        at the word's start)
rightF: exists r in S_m(w) with r + span + F in E_m
jointFF: exists r with both.
Also: the largest T such that a T-gap is feasible on the left, i.e. whether
tier A caps flank SIZE at all (expect: no - escape).
"""
import numpy as np
import sys
sys.path.insert(0, "research")
from flank_tierA import EM, carrier, LADDER
from word_ceiling import words, valid_starts, FK, STEPS


def tests(w, F, m):
    E, s = EM[m], carrier(w, m)
    span = sum(w)
    L = s & np.roll(E, F % m)                       # r-F in E
    R = s & np.roll(E, -(span + F) % m)             # r+span+F in E
    return int(L.sum()), int(R.sum()), int((L & R).sum())


def size_scan(w, m, upto=60):
    """which left-flank sizes are tier-A feasible at all?"""
    E, s = EM[m], carrier(w, m)
    return [g for g in range(1, upto + 1)
            if bool((s & np.roll(E, g % m)).any())]


if __name__ == "__main__":
    for y, q1 in STEPS:
        F = FK[y]
        print(f"\n=== {y}->{q1}  F={F}")
        for w in words(q1):
            if not valid_starts(w, q1):
                continue
            row = []
            for m, _ in LADDER:
                a, b, c = tests(w, F, m)
                row.append(f"m={m}: L{a} R{b} both{c}")
            print(f"  w={w}: " + " | ".join(row))
        # size feasibility for the binding single-letter word
        w0 = (2 * round(q1 / 6),)
        feas = size_scan(w0, 35)
        print(f"  left-flank sizes tier-A-feasible for w={w0} (mod 35): "
              f"{len(feas)} of 60; infeasible: "
              f"{[g for g in range(1,61) if g not in feas][:12]}")
