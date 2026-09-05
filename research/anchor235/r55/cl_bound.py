"""r55 cl_bound - item 5: the collision bound, and how far it reaches.

THE VALID FORM.  Partition the gear set S into blocks.  The union of all the gears' strikes is
contained in the union of the blocks' unions, so

    L = |union|  <=  sum_{blocks B} joint_max(B; L)
                  =  sum_{g in S} max_g(L)  -  sum_{blocks B} c_B(L) .

If the right-hand side is below L for EVERY K-set S (with the best partition for each S), then
no K primes cover a run of L.  Block size 1 is the counting filter of file 20; block size 2 is
a maximum-weight MATCHING of the gears; block size K is the exact adversarial computation.

THE INVALID FORM.  Subtracting every pairwise deficit, sum over ALL pairs, is not a bound:
Bonferroni runs the other way (|union| >= sum - sum over pairs).  Part A refutes it with the
recorded explicit covers.

Gears with g >= L strike at most 2 columns (file 20 Lemma 2), so they are granted 2 each with
no deficit - the safe direction for a "no cover exists" conclusion, exactly as file 20 does.
"""
import itertools
import os
from math import comb

from cl_core import (RESULTS, PRIMES, block_c_at, block_jm_at, dump, maxstrike, pair_c_at,
                     real_sep, say_factory)

LINES = []
say = say_factory(LINES)

# the recorded covers: K -> (L covered, gears)   (file 20 / small_K_theorem.md 1.2)
COVERS = {
    4: (15, [7, 5, 11, 17]),
    5: (21, [5, 11, 29, 7, 23]),
    6: (27, [5, 23, 11, 37, 7, 17]),
    7: (36, [5, 17, 31, 11, 19, 7, 13]),
    8: (44, [5, 7, 13, 29, 19, 83, 31, 11]),
    9: (67, [5, 23, 37, 11, 17, 7, 13, 31, 47]),
    10: (87, [5, 17, 7, 11, 13, 19, 23, 37, 79, 29]),
}
A = {1: 2, 2: 5, 3: 7, 4: 16, 5: 22, 6: 28, 7: 37, 8: 45, 9: 68, 10: 88, 11: 101, 12: 115}
W = {1: 8, 2: 20, 3: 28, 4: 48, 5: 60, 6: 88, 7: 140, 8: 160, 9: 228, 10: 280}
PERIOD_CAP = 9_000_000


def best_partition_value(gears, L, bmax, cache):
    """min over partitions into blocks of size <= bmax of sum_B joint_max(B; L).

    Exact DP over subsets.  Returns the value (an upper bound on the columns the set can
    strike together)."""
    n = len(gears)
    full = (1 << n) - 1
    memo = {}

    def jm(mask):
        key = (tuple(gears), mask, L)
        v = cache.get(key)
        if v is None:
            gs = [gears[i] for i in range(n) if mask >> i & 1]
            P = 1
            for g in gs:
                P *= g
            if P > PERIOD_CAP:
                # too big to evaluate exactly: drop this block option.  Dropping options can
                # only RAISE the min over partitions, i.e. weaken the bound - the safe
                # direction for a "no cover exists" conclusion.
                v = None
            else:
                v = block_jm_at(gs, [real_sep(g) for g in gs], L)
            cache[key] = v
        return v

    def rec(rem):
        if rem == 0:
            return 0
        if rem in memo:
            return memo[rem]
        low = (rem & -rem).bit_length() - 1
        rest = [i for i in range(n) if (rem >> i & 1) and i != low]
        best = None
        for size in range(1, bmax + 1):
            for combo in itertools.combinations(rest, size - 1):
                mask = 1 << low
                for i in combo:
                    mask |= 1 << i
                j = jm(mask)
                if j is None:
                    continue
                v = j + rec(rem & ~mask)
                if best is None or v < best:
                    best = v
        memo[rem] = best
        return best

    return rec(full)


def part_a():
    say("=" * 100)
    say("ITEM 5A - the all-pairs form is NOT a bound (refuted by the recorded covers)")
    say("=" * 100)
    say("  a cover of L columns exists, so any valid bound must be >= L")
    say(f"  {'K':>3} {'L covered':>10} {'sum max_g(L)':>13} {'sum ALL pairs c':>16} "
        f"{'all-pairs bound':>16} {'verdict':>10}")
    for K in sorted(COVERS):
        L, gs = COVERS[K]
        sm = sum(maxstrike(g, real_sep(g), L) for g in gs)
        cp = sum(pair_c_at(g, real_sep(g), h, real_sep(h), L)
                 for g, h in itertools.combinations(sorted(gs), 2))
        say(f"  {K:>3} {L:>10} {sm:>13} {cp:>16} {sm - cp:>16} "
            f"{('FALSE' if sm - cp < L else 'ok'):>10}")
    say()
    say("  and the same covers against the VALID block form (blocks of size <= 2, best "
        "matching):")
    say(f"  {'K':>3} {'L covered':>10} {'block-2 bound':>14} {'verdict':>10}")
    cache = {}
    for K in sorted(COVERS):
        L, gs = COVERS[K]
        v = best_partition_value(sorted(gs), L, 2, cache)
        say(f"  {K:>3} {L:>10} {v:>14} {('VIOLATED' if v < L else 'ok'):>10}")


def bound_value(small, L, bmax, K, cache):
    """the adversary's value for a K-set whose gears below L are `small`."""
    v = best_partition_value(list(small), L, bmax, cache) if small else 0
    return v + 2 * (K - len(small))


def part_b():
    say()
    say("=" * 100)
    say("ITEM 5B - does the block bound prove the adversarial lemma?  L = W(K), the window")
    say("=" * 100)
    say("  bound_b(K, L) = max over K-sets of [ min over partitions into blocks of size <= b")
    say("                  of sum_B joint_max(B; L) ], gears >= L granted 2 each.")
    say("  The lemma at K follows if bound_b(K, W(K)) < W(K).")
    say()
    cache = {}
    for K in range(3, 11):
        L = W[K]
        pool = [p for p in PRIMES if 5 <= p < L]
        say(f"  K = {K}, L = W({K}) = {L}, A({K}) = {A[K]}; "
            f"pool of gears below L: {len(pool)} primes {pool[0]}..{pool[-1]}")
        for b in range(1, min(K, 4) + 1):
            # cheap first: a witness set that already beats L makes the bound fail
            witness = None
            cands = [pool[:K]]
            if K <= len(pool):
                cands += [pool[:K - 1] + [pool[K - 1 + j]] for j in range(1, 4)
                          if K - 1 + j < len(pool)]
                cands += [pool[:K - 2] + [pool[K - 1], pool[K]]] if K >= 2 else []
            for cand in cands:
                v = bound_value(cand, L, b, K, cache)
                if v >= L:
                    witness = (cand, v)
                    break
            if witness:
                say(f"     b = {b}: FAILS - witness {witness[0]} gives {witness[1]} >= {L}")
                continue
            nsub = sum(comb(len(pool), m) for m in range(0, K + 1))
            if nsub > 3_000_000:
                say(f"     b = {b}: every witness tried is below {L}, but the exhaustive max "
                    f"over {nsub:.3g} subsets was not run - NOT DECIDED")
                continue
            # exhaustive over the pool
            best, arg = -1, None
            for m in range(0, K + 1):
                for S in itertools.combinations(pool, m):
                    ub = sum(maxstrike(g, real_sep(g), L) for g in S) + 2 * (K - m)
                    if ub <= best:
                        continue
                    v = bound_value(S, L, b, K, cache)
                    if v > best:
                        best, arg = v, S
            say(f"     b = {b}: max over all K-sets = {best} at {arg} "
                f"-> {'PROVES the lemma at K = %d' % K if best < L else 'fails'}")
    say()


def part_b2():
    say("=" * 100)
    say("ITEM 5B' - the block-b value of the K SMALLEST gears at L = W(K), block size to 7")
    say("=" * 100)
    say(f"  (a value >= W(K) refutes the bound at that block size outright; a value < W(K)")
    say(f"   leaves the max over all K-sets still to be taken.  Blocks whose period exceeds")
    say(f"   {PERIOD_CAP:,} are not evaluated, which only weakens the bound.)")
    say(f"  {'K':>3} {'W(K)':>5} " + " ".join(f"{('b=%d' % b):>6}" for b in range(1, 8)))
    cache = {}
    for K in range(4, 11):
        L = W[K]
        gs = [p for p in PRIMES if p >= 5][:K]
        row = []
        for b in range(1, 8):
            row.append(best_partition_value(gs, L, min(b, K), cache) if b <= K else None)
        say(f"  {K:>3} {L:>5} " + " ".join(
            f"{(str(v) if v is not None else '-'):>6}" for v in row))
    say()


def part_c():
    say("=" * 100)
    say("ITEM 5C - the asymptotic rate: which block size can bite at all")
    say("=" * 100)
    say("  exact growth law for a block: c_B(L + P_B) = c_B(L) + sum_g 2P/g - "
        "(P - prod_g (g-2)),")
    say("  so the deficit rate of a block is  r_B = sum_{g in B} 2/g - (1 - prod (1 - 2/g)),")
    say("  the inclusion-exclusion tail of order >= 2.  The block bound can bite only if")
    say("      rho_b(S) = sum_g 2/g - max over partitions (blocks <= b) of sum_B r_B  <  1.")
    say()
    say(f"  {'K':>3} {'gears':>34} " + " ".join(f"{('rho_%d' % b):>8} " for b in range(1, 7)))
    for K in range(3, 13):
        gs = [p for p in PRIMES if p >= 5][:K]
        row = []
        for b in range(1, 7):
            row.append(rho(gs, b))
        say(f"  {K:>3} {str(gs):>34} " + " ".join(f"{v:>8.4f} " for v in row))
    say()
    say("  the least block size with rho_b < 1 (the bound is vacuous below it):")
    out = []
    for K in range(3, 13):
        gs = [p for p in PRIMES if p >= 5][:K]
        b = next((b for b in range(1, 9) if rho(gs, b) < 1), None)
        out.append((K, b))
    say("    " + ", ".join(f"K={K}: b={b}" for K, b in out))


def rho(gears, bmax):
    """sum 2/g minus the best partition's total deficit rate, blocks of size <= bmax."""
    n = len(gears)
    memo = {}

    def rate(idxs):
        s = sum(2.0 / gears[i] for i in idxs)
        p = 1.0
        for i in idxs:
            p *= (1 - 2.0 / gears[i])
        return s - (1 - p)

    def rec(rem):
        if rem == 0:
            return 0.0
        if rem in memo:
            return memo[rem]
        low = (rem & -rem).bit_length() - 1
        rest = [i for i in range(n) if (rem >> i & 1) and i != low]
        best = -1.0
        for size in range(1, bmax + 1):
            for combo in itertools.combinations(rest, size - 1):
                idxs = (low,) + combo
                mask = 0
                for i in idxs:
                    mask |= 1 << i
                v = rate(idxs) + rec(rem & ~mask)
                if v > best:
                    best = v
        memo[rem] = best
        return best

    return sum(2.0 / g for g in gears) - rec((1 << n) - 1)


def part_d():
    say()
    say("=" * 100)
    say("ITEM 5D - the induction increment: what a new gear q' adds")
    say("=" * 100)
    say("  a new gear q' brings capacity max_{q'}(L) and collides with each old gear g;")
    say("  predicted rate of the collision with g is 4/(g q') per column.")
    say(f"  {'q_new':>6} {'L':>5} {'max_q(L)':>9} {'sum_g c(g,q;L)':>15} "
        f"{'predicted 4L sum 1/(g q)':>25} {'net = max - sum':>16}")
    M = [5, 7, 11, 13, 17, 19, 23]
    for qn in (29, 31, 37, 41, 43):
        for L in (280, 560):
            mq = maxstrike(qn, real_sep(qn), L)
            s = sum(pair_c_at(g, real_sep(g), qn, real_sep(qn), L) for g in M)
            pred = 4.0 * L * sum(1.0 / (g * qn) for g in M)
            say(f"  {qn:>6} {L:>5} {mq:>9} {s:>15} {pred:>25.2f} {mq - s:>16}")
    say()
    say("  and the rule in rates: the new gear's net contribution per column is")
    say("      2/q' - sum_{g in M} 4/(g q')  =  (2/q') (1 - 2 sum_{g in M} 1/g),")
    say("  negative once sum_{g in M} 1/g > 1/2:")
    run = 0.0
    for g in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        run += 1.0 / g
        say(f"    M up to {g:>3}: sum 1/g = {run:.4f}, 1 - 2 sum = {1 - 2 * run:+.4f}")
    say("  This is exactly why the ALL-PAIRS form is invalid (item 5A): from the fifth gear on")
    say("  it subtracts more than the gear brings, and it would 'prove' what is false.")
    say("  Under the block form each gear sits in ONE block, so a new gear collides with at")
    say("  most b-1 old gears and the increment stays positive.")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    part_a()
    part_b()
    part_b2()
    part_c()
    part_d()
    dump(LINES, "cl_bound.txt")


if __name__ == "__main__":
    main()
