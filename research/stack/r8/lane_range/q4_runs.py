"""Q4: closing a run of k consecutive openings of q=13 with gap word w = (d_1..d_{k-1}).

A block B of positions is g-alignable iff its positions occupy at most two residue classes mod g
and, if two, these are at distance d_g (chain rule).  A gear can take (strike) a set of openings
of the run only if it is alignable for that gear.  Hence
   mu_T(w) = least number of blocks in a partition of the positions {0, s_1, .., s_{k-1}} into
             alignable blocks assigned to DISTINCT gears of T
is a lower bound for the number of gears of T striking all k openings of any run with word w.
   k=2: mu = 1 iff a prime of T divides 3d-1 or 3d+1 (leg rule); else 2.
   |B| >= 3: two positions share a class => their distance (a contiguous sub-sum) is a multiple of g
             => g <= span; so above the span only pairs (at distance d_g or g-d_g) or singletons.
Exhaustive check: actual minimum cover (set cover over the residue-sense strike sets) >= mu for
every run of length 2..6 in one period; tally of equality flagged as a check count only.
"""
import numpy as np
from itertools import combinations
from collections import Counter, defaultdict
from lane_common import period, c, d, openings, T_primes, record


def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


def alignable(g, pos):
    res = {p % g for p in pos}
    if len(res) == 1:
        return True
    if len(res) == 2:
        a, b = sorted(res)
        return (b - a) % g in (d(g), (g - d(g)) % g)
    return False


def sdr(blocks_takers):
    """system of distinct representatives exists? (small backtracking)"""
    blocks_takers = sorted(blocks_takers, key=len)
    used = set()

    def rec(i):
        if i == len(blocks_takers):
            return True
        for g in blocks_takers[i]:
            if g not in used:
                used.add(g)
                if rec(i + 1):
                    return True
                used.discard(g)
        return False
    return rec(0)


def mu(word, T):
    pos = [0]
    for dd in word:
        pos.append(pos[-1] + dd)
    k = len(pos)
    best = k
    for part in set_partitions(list(range(k))):
        if len(part) >= best:
            continue
        takers = []
        ok = True
        for B in part:
            tk = [g for g in T if alignable(g, [pos[i] for i in B])]
            if not tk:
                ok = False
                break
            takers.append(tk)
        if ok and sdr(takers):
            best = len(part)
    return best


def chain_rule(word, T):
    """some g in T kills the whole run (partial sums all in {0,+d_g} or all in {0,-d_g})"""
    s = np.cumsum([0] + list(word))
    for g in T:
        dg = d(g)
        r = set((s % g).tolist())
        if r <= {0, dg} or r <= {0, (g - dg) % g}:
            return g
    return None


def min_cover(strike_sets, k):
    """exact minimum number of gears whose strike sets cover range(k); None if impossible."""
    full = frozenset(range(k))
    covered_any = set().union(*strike_sets.values()) if strike_sets else set()
    if covered_any != set(full):
        return None
    gears_list = list(strike_sets)
    best = [k + 1]

    def rec(uncovered, used, depth):
        if not uncovered:
            best[0] = min(best[0], depth)
            return
        if depth + 1 >= best[0]:
            return
        p = min(uncovered)
        for g in gears_list:
            if g not in used and p in strike_sets[g]:
                rec(uncovered - strike_sets[g], used | {g}, depth + 1)
    rec(full, frozenset(), 0)
    return best[0]


def main(q=13):
    P = period(q)
    O = openings(q)
    F = record(q)
    T = T_primes(q)
    gaps = np.diff(O)
    print(f"q={q} P={P} |O|={len(O)} F={F} T={T[0]}..{T[-1]} ({len(T)}), gaps present: {sorted(set(gaps.tolist()))}")
    leg = {dd: [g for g in T if (3 * dd - 1) % g == 0 or (3 * dd + 1) % g == 0] for dd in sorted(set(gaps.tolist()))}
    print(f"  leg primes in T per gap: {leg}")
    # precompute strikes per gear on O
    S = {g: ((O % g) == c(g)) | ((O % g) == (g - c(g)) % g) for g in T}
    mu_cache = {}
    for k in range(2, 7):
        tally = Counter()
        words_by_mu = defaultdict(set)
        twins_runs = 0
        for i in range(len(O) - k + 1):
            word = tuple(int(x) for x in gaps[i:i + k - 1])
            if word not in mu_cache:
                mu_cache[word] = mu(word, T)
                # mu = 1 iff chain rule
                assert (mu_cache[word] == 1) == (chain_rule(word, T) is not None), word
                if k == 2:
                    assert (mu_cache[word] == 1) == bool(leg[word[0]])
            m = mu_cache[word]
            words_by_mu[m].add(word)
            strike_sets = {}
            for g in T:
                sset = frozenset(j for j in range(k) if S[g][i + j])
                if sset:
                    strike_sets[g] = sset
            actual = min_cover(strike_sets, k)
            if actual is None:
                twins_runs += 1
                tally[(m, 'twin')] += 1
                continue
            assert actual >= m, (word, O[i:i + k].tolist(), actual, m)
            tally[(m, actual)] += 1
        print(f" k={k}: words={len(set().union(*words_by_mu.values()))}; mu values -> #words: "
              f"{ {m: len(w) for m, w in sorted(words_by_mu.items())} }")
        for m in sorted(words_by_mu):
            ex = sorted(words_by_mu[m])[:6]
            print(f"     mu={m}: e.g. {ex}")
        print(f"     (check counts) (mu, actual min cover) tally over all runs: {dict(sorted(tally.items(), key=str))}; "
              f"runs containing a twin (uncoverable): {twins_runs}")
    # alignable blocks of size >= 3 need g <= span: show the 3-takes that occur
    three = set()
    for word in mu_cache:
        pos = np.cumsum([0] + list(word))
        for B in combinations(range(len(pos)), 3):
            for g in T:
                if g > pos[B[-1]] - pos[B[0]]:
                    break
                if alignable(g, [int(pos[b]) for b in B]):
                    three.add((g, tuple(int(pos[b]) for b in B)))
    print(f"  alignable triples (g, positions) with g <= span: {sorted(three)[:12]} ... ({len(three)} total, check count)")
    print(f"  every triple has two positions congruent mod g (distance = g) and the third at +-d_g: "
          f"{all(any((pos[a]-pos[b]) % g == 0 for a, b in combinations(range(3), 2)) for g, pos in three)}")


if __name__ == "__main__":
    main()
