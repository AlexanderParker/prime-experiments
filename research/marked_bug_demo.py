"""Round 23 (mechanic): EXHIBIT the round-22 marked-spectrum bug.

research/marked_qspec.py (round 22) reported Q^[3](19) = 50 where the
round-23 tool research/j5_census.py reports 43 (= the exact Q_3(23)).
This script finds a window the OLD code accepts and the definition rejects,
and prints it with the offending interior named, so the disagreement is
settled by exhibition rather than by argument.

The bug: marked_qspec.feasible() returns True as soon as J-1 marks have been
placed, WITHOUT checking that the interiors after the last mark are killed.
Its recursion refuses to SKIP a forced (unkilled) interior, but it never
looks at the ones beyond the final mark.
"""
import sys
from math import prod
import numpy as np

OLD, QP, QPP = 19, 23, 29
A = 2 * round(QPP / 6)          # 10
J = 3                           # marks = J-1 = 2


def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


def openings(y):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    return [int(x) for x in np.flatnonzero(~ex)], P


# ---- the round-22 predicate, transcribed verbatim from marked_qspec.py ----
def r22_feasible(interior, marked_needed, floor_a, forced_flag):
    n = len(interior)
    if marked_needed == 0:
        return n == 0
    if n < marked_needed:
        return False
    iv = interior
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def rec(idx, cnt, last):
        if cnt == marked_needed:
            return True                      # <-- THE BUG: no tail check
        if idx >= n:
            return False
        res = False
        if not forced_flag[idx]:
            res = rec(idx + 1, cnt, last)
        if res:
            return True
        if cnt == 0 or iv[idx] - last >= floor_a:
            if rec(idx + 1, cnt + 1, iv[idx]):
                return True
        return False

    return rec(0, 0, -10 ** 18)


def r23_feasible(interior, marked_needed, floor_a, forced_flag):
    """literal definition: choose exactly marked_needed marks, containing
    every forced interior, consecutive marks >= floor_a apart."""
    import itertools
    n = len(interior)
    forced = {t for t in range(n) if forced_flag[t]}
    if len(forced) > marked_needed or n < marked_needed:
        return None
    for S in itertools.combinations(range(n), marked_needed):
        if not forced <= set(S):
            continue
        if all(interior[S[t + 1]] - interior[S[t]] >= floor_a
               for t in range(marked_needed - 1)):
            return S
    return None


def main():
    op, P = openings(OLD)
    n = len(op)
    u = pow(6, -1, QP)
    LOOK = 40
    ext = op + [x + P for x in op[:LOOK]]
    print(f"machine {OLD}, q'={QP}, u'={u}, floor a={A}, J={J} "
          f"(marks = {J-1});  disputed value: r22 said 50, r23 says 43")
    shown = 0
    best22 = best23 = 0
    for i in range(n):
        x0 = ext[i]
        for m in range(1, LOOK):
            span = ext[i + m] - x0
            if span > 55:
                break
            pos = ext[i + 1:i + m]
            ni = len(pos)
            if ni < J - 1:
                continue
            rp = [x % QP for x in pos]
            for c in range(QP):
                kill = {(c - u) % QP, (c + u) % QP}
                ff = [rp[t] not in kill for t in range(ni)]
                if sum(ff) > J - 1:
                    continue
                ok22 = r22_feasible(tuple(pos), J - 1, A, tuple(ff))
                w23 = r23_feasible(pos, J - 1, A, ff)
                if ok22:
                    best22 = max(best22, span)
                if w23 is not None:
                    best23 = max(best23, span)
                if ok22 and w23 is None and shown < 3 and span >= 44:
                    shown += 1
                    print(f"\n  DISAGREEMENT  window k={x0}, span {span}, "
                          f"phase c={c} (gear {QP} kills residues "
                          f"{sorted(kill)})")
                    print(f"    interiors (offset, residue mod {QP}, killed?):")
                    for t in range(ni):
                        print(f"      +{pos[t]-x0:3d}   r={rp[t]:2d}   "
                              f"{'KILLED' if not ff[t] else 'ALIVE  <-- must be marked'}")
                    print(f"    marks needed: {J-1}; alive interiors: "
                          f"{sum(ff)}; the r22 recursion stops after placing "
                          f"{J-1} marks and never inspects the tail.")
                    # name the offending interior
                    alive = [t for t in range(ni) if ff[t]]
                    print(f"    alive at offsets "
                          f"{[pos[t]-x0 for t in alive]}; any admissible mark "
                          f"set must contain all of them AND have consecutive "
                          f"marks >= {A} apart - impossible here.")
    print(f"\n  max span accepted by the r22 predicate at J={J}: {best22}")
    print(f"  max span accepted by the literal definition at J={J}: {best23}")
    print(f"  exact Q_3(23; 10) = 43  (full-period machine-23 scan, r21)")
    assert best23 == 43, best23
    print("  -> the literal definition reproduces the exact value; "
          "the r22 predicate over-accepts.")


if __name__ == '__main__':
    main()
