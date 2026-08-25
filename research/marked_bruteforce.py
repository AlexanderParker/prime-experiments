"""Round 23 (constructor): BRUTE-FORCE AUDIT of the marked qualifying
spectrum Q^[J] at the smallest step, 11 -> 13.

Machine 11 has 191 openings in a period of 385, so the definition can be
checked by literal enumeration - every window, every phase, every marked
subset - with no pruning, no DP and no bookkeeping to get wrong.

    Q^[J](old) = max span x_m - x_0 over windows of OLD openings such that,
    for SOME phase phi of gear q', there is a set M of exactly J-1 interior
    openings with (i) every interior NOT in M killed by q' at phi, and
    (ii) consecutive members of M at distance >= a = 2u''.

Compared against research/marked_qspec.py's pruned/DP implementation and
against the exact Q_J(new; a) of the new machine.

Usage: uv run python research/marked_bruteforce.py
"""
from itertools import combinations
from math import prod

import numpy as np


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def openings(y):
    gears = primes(5, y)
    P = prod(gears)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    return np.flatnonzero(~ex).astype(np.int64), P


def brute(old, qp, qpp, Jmax=5, maxm=30):
    op, P = openings(old)
    n = len(op)
    a = 2 * round(qpp / 6)
    c = pow(6, -1, qp)
    ext = np.concatenate([op, op[:maxm + 2] + P])
    r = (ext % qp).astype(int)
    best = {J: 0 for J in range(2, Jmax + 1)}
    wit = {J: None for J in range(2, Jmax + 1)}
    for i in range(n):
        for m in range(1, maxm + 1):
            span = int(ext[i + m] - ext[i])
            interior = list(range(i + 1, i + m))
            n_int = len(interior)
            for J in range(2, Jmax + 1):
                if n_int < J - 1 or span <= best[J]:
                    continue
                found = None
                for phi in range(qp):
                    kill = {(c + phi) % qp, (-c + phi) % qp}
                    surv = [t for t in interior if r[t] not in kill]
                    if len(surv) > J - 1:
                        continue
                    for M in combinations(interior, J - 1):
                        if not set(surv) <= set(M):
                            continue
                        pos = [int(ext[t]) for t in M]
                        if all(pos[k + 1] - pos[k] >= a
                               for k in range(len(pos) - 1)):
                            found = (phi, tuple(pos),
                                     tuple(int(ext[t]) for t in surv))
                            break
                    if found:
                        break
                if found:
                    best[J] = span
                    wit[J] = (int(ext[i]), int(ext[i + m]), span) + found
    return best, wit, a


def main():
    old, qp, qpp = 11, 13, 17
    best, wit, a = brute(old, qp, qpp)
    print("BRUTE-FORCE Q^[J](%d) for the step %d -> %d, floor a = %d"
          % (old, old, qp, a))
    for J in sorted(best):
        print("   J = %d : Q^[J] = %2d   witness (x0, xm, span, phase, "
              "marked, surviving interiors) = %s" % (J, best[J], wit[J]))
    print("\n   exact  Q_J(13; 6)          = [16, 18, 23, 0]  (J = 2..5, "
          "full-period machine-13 scan)")
    print("   reported research/marked_qspec.py Q^[J](11) = [16, 23, 23, 0]")
    got = [best[J] for J in range(2, 6)]
    print("   brute force                 = %s" % got)
    exact = [16, 18, 23, 0]
    for J in range(2, 6):
        assert got[J - 2] >= exact[J - 2], (J, got, exact)
    print("\n   sound-relaxation check  Q_J(new) <= Q^[J](old) : PASSES")
    ub = [max([34 if False else 0] + exact[:J - 1]) for J in range(2, 6)]
    print("   sandwich upper bound max_{j<=J} Q_j(new)       = %s" % ub)
    for J in range(2, 6):
        assert got[J - 2] <= ub[J - 2], (
            "SANDWICH LEMMA VIOLATED at J=%d: %d > %d" % (J, got[J - 2],
                                                          ub[J - 2]))
    print("   sandwich check          Q^[J](old) <= max_j Q_j : PASSES")
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
