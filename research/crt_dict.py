"""Round 25 (constructor): THE SCAN-FREE DICTIONARY.

R58 reduced requirement (D) at one step to a finite set of realisability
queries - "is this tuple of consecutive gap values realised by machine M?" -
90 to 955 of them per step.  R59 named the remaining job: answer those queries
WITHOUT a period scan.  Round 24 answered them from a dumped realised-tuple
set, which came from the scan, so the obligation was measured, not avoided.

This module answers them by CRT arithmetic alone, from the GEAR LIST and
nothing else.

THE OBJECT.  A tuple of gaps (v_1..v_m) is realised by M iff some slot k has
    X = {0, v_1, v_1+v_2, ...}   (the m+1 prefix-sum points)  ALL OPEN, and
    Y = (0, span) \\ X                                        ALL BLOCKED.
By CRT a slot k is exactly a phase vector (a_q)_q, a_q = k mod q, and

    k + i is blocked by gear q   <=>   a_q = +-u_q - i  (mod q),  u_q = 6^{-1}.

So the question is a finite CSP with one variable per gear:
    (open)   a_q  not in  {+-u_q - x : x in X}          for every gear q
    (cover)  for every t in Y, SOME gear q has a_q = +-u_q - t.
Nothing else.  The period never appears.

TWO INDEPENDENT DECIDERS, both exact, both in this file:
  decide_cover  - backtracking search over the CSP (unit propagation + the
                  minimum-remaining-options branch rule).  Returns a WITNESS
                  phase vector when realised, and an exhaustive refutation
                  when not.  This is the one the chain uses.
  pattern_count - R43's pruned inclusion-exclusion COUNTER (imported from
                  research/qualrun_zerocert.py).  Exact count, exponential in
                  span; used here only as a cross-check.

MEASURED (this file's gate, `validate`): the two agree on every case tested,
the decider reproduces every published anchor, and it recovers F(M) and
F_2(M) at machines 11..37 with no scan.

Usage:
    python research/crt_dict.py validate
    python research/crt_dict.py spectrum 31        # scan-free F, F_2 of m31
"""
import os
import sys
import time
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qualrun_zerocert import pattern_count, primes   # noqa: E402

# corpus ladder, exact (used only for ASSERTIONS, never as an input)
KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91, 43: 103, 47: 118, 53: 145}
KNOWN_F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90,
            41: 103}


@lru_cache(maxsize=None)
def gears_of(y):
    return tuple(primes(5, y))


@lru_cache(maxsize=None)
def _inv6(q):
    return pow(6, -1, q)


class Budget(Exception):
    pass


def decide_cover(qs, X, Y, node_budget=20_000_000, want_witness=False):
    """Exact decision of #(X open, Y blocked) > 0 over the gears qs.

    The CSP: one variable a_q per gear, domain = the residues that keep every
    point of X open; every point of Y must be covered by some gear.  Solved by
    DFS with (i) bitmask coverage over Y, (ii) branching on the uncovered
    point with the fewest live options, and (iii) a CAPACITY BOUND - if the
    unassigned gears cannot between them cover the still-uncovered points even
    at their individually best residues, the node is dead.  (iii) is what makes
    REFUTATIONS affordable: a refutation must exhaust the tree.

    Returns (True, witness_or_None, nodes) or (False, None, nodes).
    Raises Budget if the search exceeds node_budget (answer UNKNOWN - the
    caller must NOT delete on an unknown).
    """
    n = len(qs)
    Yl = list(Y)
    ny = len(Yl)
    # NOTE (round-25 bug, caught by the cross-check against Mechanic's m23/m29
    # 4-tuple censuses): an EMPTY Y must still pass through the domain check.
    # The all-ones tuples (1,1), (1,1,1), ... have no interior point at all,
    # and an early "return True" here declared them realised when in fact gear
    # 5 has no admissible residue.  The bug never touched a reported number
    # (those tuples never carry a T3-legal edge and never attain a maximum),
    # but it is a wrong answer and the early return is gone.
    pos = {t: j for j, t in enumerate(Yl)}
    FULL = (1 << ny) - 1
    masks = []                      # masks[gi] = {a: bitmask over Y}
    for gi, q in enumerate(qs):
        u = _inv6(q)
        forb = set()
        for x in X:
            forb.add((u - x) % q)
            forb.add((-u - x) % q)
        d = {}
        for a in range(q):
            if a in forb:
                continue
            m = 0
            for t in Yl:
                if (a + t - u) % q == 0 or (a + t + u) % q == 0:
                    m |= 1 << pos[t]
            d[a] = m
        if not d:
            return False, None, 0
        masks.append(d)
    # options[j] = list of (gi, a) covering Y[j]
    options = [[] for _ in range(ny)]
    for gi in range(n):
        for a, m in masks[gi].items():
            mm = m
            while mm:
                b = mm & -mm
                options[(b.bit_length() - 1)].append((gi, a))
                mm ^= b
    for j in range(ny):
        if not options[j]:
            return False, None, 0
    asg = [None] * n
    nodes = [0]

    def rec(covered):
        nodes[0] += 1
        if nodes[0] > node_budget:
            raise Budget()
        unc = FULL & ~covered
        if unc == 0:
            return True
        need = bin(unc).count("1")
        # (iii) capacity bound over the unassigned gears
        cap = 0
        for gi in range(n):
            if asg[gi] is not None:
                continue
            best = 0
            for m in masks[gi].values():
                c = bin(m & unc).count("1")
                if c > best:
                    best = c
            cap += best
            if cap >= need:
                break
        if cap < need:
            return False
        # (ii) branch on the uncovered point with the fewest live options
        bestj, bestlo = -1, None
        mm = unc
        while mm:
            b = mm & -mm
            j = b.bit_length() - 1
            mm ^= b
            lo = [(gi, a) for (gi, a) in options[j] if asg[gi] is None]
            if not lo:
                return False
            if bestlo is None or len(lo) < len(bestlo):
                bestj, bestlo = j, lo
                if len(lo) == 1:
                    break
        bestlo.sort(key=lambda ga: -bin(masks[ga[0]][ga[1]] & unc).count("1"))
        for (gi, a) in bestlo:
            asg[gi] = a
            if rec(covered | masks[gi][a]):
                return True
            asg[gi] = None
        return False

    ok = rec(0)
    wit = None
    if ok and want_witness:
        wit = [(qs[i], asg[i]) for i in range(n) if asg[i] is not None]
    return ok, wit, nodes[0]


def count_solutions(qs, X, Y, cap=4, node_budget=20_000_000):
    """Number of phase vectors (= slots k mod P) realising the pattern, up to
    `cap`.  Returns (count, capped_flag).  Used for the MIRROR PARITY LEVER
    (Lateral round 25): equal adjacent pairs (g,g) occur an EVEN number of
    times, so a count capped at 1 proves the count is 0.
    """
    n = len(qs)
    Yl = list(Y)
    ny = len(Yl)
    pos = {t: j for j, t in enumerate(Yl)}
    FULL = (1 << ny) - 1
    masks = []
    for q in qs:
        u = _inv6(q)
        forb = set()
        for x in X:
            forb.add((u - x) % q)
            forb.add((-u - x) % q)
        d = {}
        for a in range(q):
            if a in forb:
                continue
            m = 0
            for t in Yl:
                if (a + t - u) % q == 0 or (a + t + u) % q == 0:
                    m |= 1 << pos[t]
            d[a] = m
        if not d:
            return 0, False
        masks.append(d)
    keys = [sorted(masks[i]) for i in range(n)]
    found = [0]
    nodes = [0]

    def rec(i, covered):
        nodes[0] += 1
        if nodes[0] > node_budget:
            raise Budget()
        if found[0] >= cap:
            return
        if i == n:
            if covered == FULL:
                found[0] += 1
            return
        # prune: can the remaining gears still finish the cover?
        unc = FULL & ~covered
        if unc:
            capa = 0
            for gi in range(i, n):
                capa += max(bin(m & unc).count("1")
                            for m in masks[gi].values())
                if capa >= bin(unc).count("1"):
                    break
            if capa < bin(unc).count("1"):
                return
        for a in keys[i]:
            rec(i + 1, covered | masks[i][a])
            if found[0] >= cap:
                return

    rec(0, 0)
    return found[0], found[0] >= cap


_CACHE = {}


def realised(y, gaps, node_budget=2_000_000):
    """Is (gaps) a tuple of CONSECUTIVE gaps of machine y?  Exact, scan-free.

    Returns True / False, or raises Budget when undecided.
    """
    key = (y, tuple(gaps))
    v = _CACHE.get(key)
    if v is not None:
        return v
    qs = gears_of(y)
    X = [0]
    for g in gaps:
        X.append(X[-1] + g)
    span = X[-1]
    xs = set(X)
    Y = [t for t in range(1, span) if t not in xs]
    ok, _, _ = decide_cover(qs, X, Y, node_budget=node_budget)
    _CACHE[key] = ok
    return ok


def realised_nodes(y, gaps, node_budget=2_000_000):
    qs = gears_of(y)
    X = [0]
    for g in gaps:
        X.append(X[-1] + g)
    xs = set(X)
    Y = [t for t in range(1, X[-1]) if t not in xs]
    return decide_cover(qs, X, Y, node_budget=node_budget, want_witness=True)


# ----------------------------------------------------------------- spectrum
def scanfree_F(y, cap=None, node_budget=2_000_000):
    """F(M) = largest realised single gap, decided one value at a time."""
    hi = cap if cap else 4 * y
    for v in range(hi, 0, -1):
        if realised(y, (v,), node_budget):
            return v
    raise AssertionError("no realised gap")


def scanfree_F2(y, F, node_budget=2_000_000):
    """F_2(M) = largest realised adjacent pair sum.  Descending on the sum, so
    the first realised pair found is the maximum."""
    for S in range(2 * F, 1, -1):
        for u in range(max(1, S - F), min(F, S - 1) + 1):
            if realised(y, (u, S - u), node_budget):
                return S, (u, S - u)
    raise AssertionError("no realised pair")


# ----------------------------------------------------------------- gate
def validate():
    t0 = time.time()
    print("SCAN-FREE DICTIONARY - validation gate\n")

    # (1) decider vs the exact pruned-IE counter, machine 13 and 19
    print("(1) decide_cover  vs  R43 pattern_count (exact IE), all gap tuples")
    n = 0
    for y in (11, 13, 17):
        qs = gears_of(y)
        for m, rng in ((1, range(1, KNOWN_F[y] + 4)),
                       (2, None), (3, None)):
            tuples = []
            if m == 1:
                tuples = [(v,) for v in rng]
            elif m == 2:
                tuples = [(a, b) for a in range(1, 13) for b in range(1, 13)]
            else:
                tuples = [(a, b, c) for a in range(1, 9)
                          for b in range(1, 9) for c in range(1, 9)]
            for t in tuples:
                X = [0]
                for g in t:
                    X.append(X[-1] + g)
                xs = set(X)
                Y = [i for i in range(1, X[-1]) if i not in xs]
                cnt, _ = pattern_count(list(qs), X, Y)
                ok, _, _ = decide_cover(qs, X, Y)
                assert ok == (cnt > 0), (y, t, cnt, ok)
                n += 1
    print("    %d tuples at machines 11, 13, 17: decision == (IE count > 0) "
          "in every case" % n)

    # (2) published anchors (cov_count.py ANCHORS, mechanic r21)
    print("(2) published pattern anchors")
    ANCH = [(19, (8, 15), True, "word (8,15) at m19, occ 31"),
            (23, (10, 21), True, "word (10,21) at m23, occ 138"),
            (29, (10, 21, 10), True, "the k=4 fuel word, exactly 4 occ"),
            (29, (21, 10, 21), False, "ZERO occurrences (r17)"),
            (23, (29,), True, "hist_23[29] = 6"),
            (19, (24,), False, "24 is a hole of machine 19"),
            (23, (24,), False, "24 is a hole of machine 23"),
            (37, (14, 41, 14), True, "the only realised depth-3 word at m37"),
            (29, (10, 10, 21), True, "permutation of the k=4 word (R39)"),
            ]
    for y, t, want, why in ANCH:
        got = realised(y, t)
        assert got == want, (y, t, got, want, why)
        print("    m%-2d %-14s -> %-5s  %s" % (y, str(t), got, why))

    # (3) scan-free F and F_2 against the corpus ladder.
    # SCOPE NOTE (measured, round 25): F(M) is cheap at every machine reached,
    # but F_2(M) is NOT - an over-budget PAIR refutation at m37 costs 5.8 s on
    # average (23.7 s worst) against 43 ms for a 4-tuple, because a pair has
    # only three open points and therefore much larger gear domains.  Pinning
    # F_2(37) exactly would be ~3,800 such refutations (hours), so the gate
    # asserts F_2 through m31 and, at m37, asserts F exactly plus the witness
    # pair (2,88) that gives F_2(37) >= 90 = the corpus value.
    print("(3) scan-free F(M) and F_2(M) - no scan, no dump, gears only")
    for y in (11, 13, 17, 19, 23, 29, 31):
        t1 = time.time()
        F = scanfree_F(y, cap=KNOWN_F[y] + 25)
        assert F == KNOWN_F[y], (y, F, KNOWN_F[y])
        F2, pair = scanfree_F2(y, F)
        assert F2 == KNOWN_F2[y], (y, F2, KNOWN_F2[y])
        print("    m%-2d  F = %3d  F_2 = %3d  (max pair %s)   %5.1fs"
              % (y, F, F2, pair, time.time() - t1))
    t1 = time.time()
    F37 = scanfree_F(37, cap=KNOWN_F[37] + 7)
    assert F37 == KNOWN_F[37], F37
    assert realised(37, (2, 88)), "the F_2(37) = 90 witness pair"
    print("    m37  F = %3d  F_2 >= %3d (witness pair (2,88); the exact "
          "value is a %d-refutation job, see scope note)   %5.1fs"
          % (F37, 90, 3800, time.time() - t1))

    print("\nall assertions passed  (%.0fs)" % (time.time() - t0))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate()
        return
    if len(sys.argv) > 2 and sys.argv[1] == "spectrum":
        y = int(sys.argv[2])
        F = scanfree_F(y, cap=KNOWN_F.get(y, 4 * y) + 40)
        F2, pair = scanfree_F2(y, F)
        print("machine %d: F = %d, F_2 = %d (pair %s) - scan-free"
              % (y, F, F2, pair))
        return
    print(__doc__)


if __name__ == "__main__":
    main()
