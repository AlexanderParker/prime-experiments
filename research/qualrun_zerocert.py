"""Round 21 (constructor): EXACT deep qualifying-run counts at unscannable
machines - Lateral's pruned inclusion-exclusion adapted to pattern events.

The renewal-ladder identity (R38):
    #(k mod P : X exposed, Y blocked) = sum_{T subseteq Y} (-1)^|T|
                                        prod_q c_q(X u T)
was blocked from Y = all interiors by the 2^|Y| IE cost.  Lateral's round-21
cross-lane offer (psd_bite.bonferroni_runs): enumerate only subsets with
NONZERO CRT count - a zero N(T) zeroes the whole subtree (hereditary-zero
pruning) - and seed the per-gear masks with the required-open points X.
With that, Y = ALL interiors is affordable and the count is EXACT: the zero
certificate / exact pattern counter named as R38's blocker.

run_m(M; V(q')) = sum over tuples (v_1..v_m) in V^m of
    #(X(v) = prefix sums exposed, all strict interiors blocked).

Round-21 outcome (log research/data/qualrun_zerocert.log): validated exact
against every census row it reached - m19 run2/3/4 (zero certificates),
m23 and m29 all rows (run3(29) = 8 reproduced in 14 s, no scan), m31
run2 = 502,708 EXACT (period 3.34e10).  Partial run3(31): the six nonzero
tuples found, summing 508 = the census value; padded tuples exceed the
node budget (cost ~exponential in span; dead at span 99).  Machine 37 NOT
reached this round - named next job (Mechanic's COV-SAT the likely
supplier).  The memoized variant is REFUTED as a speedup (test_memo2.py).

Usage: uv run python research/qualrun_zerocert.py [y ...]   (default
19 23 29 31 37; per-tuple progress printed - redirect to a log)
"""
import csv
import os
import sys
import time
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")

KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}
NEXTP = {11: 13, 13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37, 37: 41}
NODE_BUDGET = 300_000_000


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def exposed_mask(q):
    u = pow(6, -1, q)
    m = (1 << q) - 1
    m &= ~(1 << (u % q))
    m &= ~(1 << ((-u) % q))
    return m


def rotate(mask, t, q):
    """mask of r such that r + t is exposed: shift the exposed set by -t."""
    t %= q
    return ((mask >> t) | (mask << (q - t))) & ((1 << q) - 1)


def pattern_count(gears, X, Y):
    """Exact #(k mod prod(gears): all X exposed, all Y blocked), or None if
    the node budget is exceeded.  Returns (count, nodes)."""
    base = []
    for q in gears:
        m = exposed_mask(q)
        mm = (1 << q) - 1
        for x in X:
            mm &= rotate(m, x, q)
        if mm == 0:
            return 0, 0
        base.append((q, m, mm))
    Ys = sorted(Y)
    masks0 = [mm for (_, _, mm) in base]
    # order Y elements by depth-1 surviving count (ascending): the
    # strongest killers first - subset enumeration is order-free, the
    # pruning is not
    def d1(t):
        n = 1
        for (q, m, _), mm in zip(base, masks0):
            c = (mm & rotate(m, t, q)).bit_count()
            if c == 0:
                return 0
            n *= c
        return n
    Ys.sort(key=d1)
    rotc = [[rotate(m, t, q) for (q, m, _) in base] for t in Ys]
    n0 = prod(mm.bit_count() for mm in masks0)
    total = n0
    nodes = 0
    L = len(Ys)
    stack = [(0, masks0, 0)]          # (start index, masks, depth)
    while stack:
        start, masks, depth = stack.pop()
        for i in range(start, L):
            nm = []
            n = 1
            for m, r in zip(masks, rotc[i]):
                mr = m & r
                c = mr.bit_count()
                if c == 0:
                    n = 0
                    break
                n *= c
                nm.append(mr)
            if n == 0:
                continue
            nodes += 1
            if nodes > NODE_BUDGET:
                return None, nodes
            total += n if (depth + 1) % 2 == 0 else -n
            stack.append((i + 1, nm, depth + 1))
    return total, nodes


def pattern_count_memo(gears, X, Y):
    """Same exact count via the memoized alternating recursion
        f(i, masks) = f(i+1, masks) - f(i+1, masks & rot_i)
    with base f(L, masks) = prod bit_count(masks): mask states collapse
    (intersections saturate), so shared subtrees are computed once.
    Returns (count, states)."""
    base = []
    for q in gears:
        m = exposed_mask(q)
        mm = (1 << q) - 1
        for x in X:
            mm &= rotate(m, x, q)
        if mm == 0:
            return 0, 0
        base.append((q, m, mm))
    Ys = sorted(Y)
    L = len(Ys)
    rotc = [tuple(rotate(m, t, q) for (q, m, _) in base) for t in Ys]
    memo = {}

    def f(i, masks):
        if i == L:
            n = 1
            for m in masks:
                n *= m.bit_count()
            return n
        if len(memo) > 20_000_000:
            raise MemoryError("memo state budget exceeded")
        key = (i, masks)
        v = memo.get(key)
        if v is not None:
            return v
        # skip ahead over positions that cannot cut (mask unchanged) -
        # they contribute factor (1 - 1) = 0 only if identical; careful:
        # masks & rot == masks means the blocked term equals the free term
        # -> f = f(i+1, masks) - f(i+1, masks) = 0
        nm = []
        dead = False
        for m, r in zip(masks, rotc[i]):
            mr = m & r
            if mr == 0:
                dead = True
                break
            nm.append(mr)
        if dead:
            v = f(i + 1, masks)          # blocked-branch N = 0 always
        else:
            nm = tuple(nm)
            if nm == masks:
                v = 0                     # the two branches cancel exactly
            else:
                v = f(i + 1, masks) - f(i + 1, nm)
        memo[key] = v
        return v

    sys.setrecursionlimit(100000)
    masks0 = tuple(mm for (_, _, mm) in base)
    try:
        v = f(0, masks0)
    except MemoryError:
        return None, len(memo)
    return v, len(memo)


def qual_values(q1, F):
    c = pow(6, -1, q1)
    R = {0, (2 * c) % q1, (-2 * c) % q1}
    return [v for v in range(1, F + 1) if v % q1 in R]


def brute_check():
    """pattern_count vs direct sieve at machine 13 for several patterns."""
    import numpy as np
    gears = primes(5, 13)
    P = prod(gears)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    open_ = ~ex
    for X, Y in (([0, 6], range(1, 6)), ([0, 11], range(1, 11)),
                 ([0, 6, 17], [t for t in range(1, 17) if t != 6]),
                 ([0, 5], [2, 3]), ([0, 4, 8], [1, 2, 3, 5, 6, 7])):
        cnt = 0
        for k in range(P):
            if all(open_[(k + x) % P] for x in X) and \
               all(ex[(k + t) % P] for t in Y):
                cnt += 1
        got, _ = pattern_count(gears, list(X), list(Y))
        assert got == cnt, (X, list(Y), got, cnt)
    print("brute-force validation at machine 13: 5/5 patterns exact")


def run_m(y, m, verbose=True):
    gears = primes(5, y)
    q1 = NEXTP[y]
    F = KNOWN_F[y]
    V = qual_values(q1, F)
    total = 0
    nodes_tot = 0
    t0 = time.time()
    incomplete = False
    from itertools import product as iproduct
    for tup in iproduct(V, repeat=m):
        span = sum(tup)
        X = [0]
        for v in tup:
            X.append(X[-1] + v)
        Y = [t for t in range(1, span) if t not in set(X)]
        cnt, nodes = pattern_count(gears, X, Y)
        nodes_tot += nodes
        if cnt is None:
            print(f"    tuple {tup}: NODE BUDGET EXCEEDED ({nodes:,}) - "
                  f"run_{m}({y}) INCOMPLETE", flush=True)
            incomplete = True
            continue
        if verbose and (cnt or nodes > 1_000_000):
            print(f"    tuple {tup}: count {cnt:,}  ({nodes:,} nodes, "
                  f"{time.time() - t0:.0f}s)", flush=True)
        total += cnt
    status = "INCOMPLETE" if incomplete else "EXACT"
    print(f"  run_{m}({y}; V({q1})={V}) = {total:,}  [{status}]  "
          f"({nodes_tot:,} nodes, {time.time() - t0:.0f}s)", flush=True)
    return total if not incomplete else None


def main():
    ys = [int(x) for x in sys.argv[1:]] or [19, 23, 29, 31, 37]
    brute_check()
    known = {}
    p = os.path.join(DDIR, "tm_resid_runs.csv")
    with open(p) as f:
        for row in csv.DictReader(f):
            known[int(row["y"])] = row
    for y in ys:
        print(f"\n=== machine {y} (q' = {NEXTP[y]}, F = {KNOWN_F[y]}, "
              f"period {prod(primes(5, y)):,})", flush=True)
        for m in (2, 3):
            r = run_m(y, m)
            if y in known and r is not None:
                exact = int(known[y][f"run{m}"])
                assert r == exact, (y, m, r, exact)
                print(f"    == census row EXACT MATCH ({exact:,})",
                      flush=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
