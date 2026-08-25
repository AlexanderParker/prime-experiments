"""Round 23 (mechanic): independent validation of the marked-spectrum predicate.

Two controls on research/j5_census.py, both by LITERAL subset enumeration
(itertools.combinations) with no DP, no memoisation and no pruning:

  (A) PREDICATE CONTROL.  Sample (window, phase, J) triples at the real
      machine and assert that the census DP's admissibility verdict equals the
      literal one, triple by triple.  This is the control the R2 anchor cannot
      give (R2 never calls the DP), and it is what adjudicated the round-22
      bug: the round-22 predicate fails it, the round-23 one passes.
  (B) SPECTRUM CONTROL.  Recompute the whole Q^[J] table by brute force on the
      two smallest steps and compare with the census values.

usage: uv run python research/j5_verify.py [nsample]
"""
import sys, itertools, random
from math import prod
import numpy as np

sys.path.insert(0, __file__.rsplit('research', 1)[0] + 'research')
_ARGV = sys.argv
sys.argv = ['x']
from j5_census import feasible_marks  # noqa: E402
sys.argv = _ARGV


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


def literal(pos, forced, need, a):
    """the definition, verbatim: some size-`need` subset of pos containing all
    of `forced` with consecutive members >= a apart."""
    if need == 0:
        return not forced
    for S in itertools.combinations(range(len(pos)), need):
        if not forced <= set(S):
            continue
        if all(pos[S[t + 1]] - pos[S[t]] >= a for t in range(need - 1)):
            return True
    return False


def r22_predicate(pos, forced, need, a):
    """marked_qspec.feasible transcribed (the round-22 code under test)."""
    n = len(pos)
    if need == 0:
        return n == 0
    if n < need:
        return False
    ff = [t in forced for t in range(n)]
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def rec(idx, cnt, last):
        if cnt == need:
            return True
        if idx >= n:
            return False
        if not ff[idx] and rec(idx + 1, cnt, last):
            return True
        if cnt == 0 or pos[idx] - last >= a:
            if rec(idx + 1, cnt + 1, pos[idx]):
                return True
        return False

    return rec(0, 0, -10 ** 18)


def control_A(old, qp, qpp, nsample):

    op, P = openings(old)
    n = len(op)
    u = pow(6, -1, qp)
    a = 2 * round(qpp / 6)
    LOOK = 40
    ext = op + [x + P for x in op[:LOOK]]
    rng = random.Random(20260825)
    checked = agree23 = agree22 = 0
    r22_over = 0
    for _ in range(nsample):
        i = rng.randrange(n)
        m = rng.randrange(2, 16)
        pos = ext[i + 1:i + m]
        ni = len(pos)
        if ni < 1:
            continue
        rp = [x % qp for x in pos]
        c = rng.randrange(qp)
        kill = {(c - u) % qp, (c + u) % qp}
        forced = {t for t in range(ni) if rp[t] not in kill}
        for J in range(2, 8):
            need = J - 1
            if ni < need:
                continue
            lit = literal(pos, forced, need, a)
            dp = feasible_marks(pos, forced, need, a) is not None
            old22 = r22_predicate(pos, forced, need, a)
            checked += 1
            agree23 += (lit == dp)
            agree22 += (lit == old22)
            if old22 and not lit:
                r22_over += 1
            assert lit == dp, (i, m, c, J, pos, sorted(forced), lit, dp)
    print(f"  CONTROL A ({old}->{qp}): {checked:,} (window,phase,J) triples")
    print(f"    round-23 predicate agrees with the literal definition: "
          f"{agree23:,}/{checked:,}   (asserted)")
    print(f"    round-22 predicate agrees: {agree22:,}/{checked:,}  "
          f"- OVER-ACCEPTS {r22_over:,}")


def control_B(old, qp, qpp, span_cap):
    op, P = openings(old)
    n = len(op)
    u = pow(6, -1, qp)
    a = 2 * round(qpp / 6)
    LOOK = 40
    ext = op + [x + P for x in op[:LOOK]]
    best = {J: 0 for J in range(2, 8)}
    for i in range(n):
        x0 = ext[i]
        for m in range(1, LOOK):
            span = ext[i + m] - x0
            if span > span_cap:
                break
            pos = ext[i + 1:i + m]
            ni = len(pos)
            rp = [x % qp for x in pos]
            for c in range(qp):
                kill = {(c - u) % qp, (c + u) % qp}
                forced = {t for t in range(ni) if rp[t] not in kill}
                for J in range(2, 8):
                    need = J - 1
                    if ni < need or span <= best[J] or len(forced) > need:
                        continue
                    if literal(pos, forced, need, a):
                        best[J] = span
    print(f"  CONTROL B ({old}->{qp}, span_cap {span_cap}): "
          f"Q^[J] = {[best[J] for J in range(2, 8)]}")
    return best


if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    print("CONTROL A - the predicate, triple by triple")
    control_A(19, 23, 29, ns)
    control_A(23, 29, 31, ns)
    print("\nCONTROL B - the whole spectrum by literal enumeration")
    b1 = control_B(11, 13, 17, 40)
    assert [b1[J] for J in range(2, 6)] == [16, 18, 23, 0], b1
    b2 = control_B(13, 17, 19, 45)
    assert [b2[J] for J in range(2, 7)] == [25, 28, 31, 32, 34], b2
    print("  both match research/j5_census.py exactly (asserted).")
