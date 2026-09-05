"""r53 sk_cases - the case analysis that proves A(K) for K = 1..6, in the sharpest split.

THE SPLIT.  Fix the run length L.  A gear g >= L meets each of its two residue classes at
most once in the run, so it strikes at most TWO of the L columns, and if it strikes two, at
a distance t with t = a_g or t = g - a_g (the span lemma).  A gear g < L may strike more.
So write

    T(L) = {primes 5 <= g < L}        (the gears that can strike three or more)
    aux  = {primes g >= L}            (each strikes at most two)

and split a K-set by S = (the set) intersect T(L).  The K - |S| aux gears contribute at most
2 columns each, which gives the counting filter

    sum_{g in S} maxstrike(g, L) + 2 (K - |S|)  >=  L .

For every S that survives the filter, every phase vector of S is enumerated; the holes H that
S leaves must be finished by the K - |S| aux gears, each taking one hole, or two holes at a
distance t in D(g) = {a_g, g - a_g} intersect [1, L-1], with the primes distinct.  So the
number of aux gears needed is |H| - (largest number of disjoint hole pairs that can be
assigned injectively to distinct aux primes able to span them).

THE SPAN LEMMA supplies D(g) in closed form: t even forces t = a_g, i.e. g = 3t -+ 1;
t odd forces t = g - a_g, i.e. g = (3t -+ 1)/2.  So at most two primes can span any given
distance, and the whole aux supply at level L is a short table.

Run:  uv run python research/anchor235/r53/sk_cases.py
"""
import os
from itertools import combinations

from sk_core import RESULTS, arc, is_prime, primes_upto, strike_mask

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


A_K = {1: 2, 2: 5, 3: 7, 4: 16, 5: 22, 6: 28}


def maxstrike(g, L):
    q, r = divmod(L, g)
    return 2 * q + (2 if r > arc(g) else (1 if r >= 1 else 0))


def T_of(L):
    return [p for p in primes_upto(max(5, L)) if 5 <= p < L]


def aux_supply(L):
    """primes g >= L that can span a pair, with the distances they can span."""
    out = {}
    for g in primes_upto(3 * L + 4):
        if g < max(5, L):
            continue
        a = arc(g)
        ds = sorted({t for t in (a, g - a) if 1 <= t <= L - 1})
        if ds:
            out[g] = ds
    return out


def span_table(L):
    """for each distance t, the primes >= L that can span it (by the span lemma)."""
    tab = {}
    for t in range(1, L):
        cand = []
        if t % 2 == 0:
            cand = [3 * t - 1, 3 * t + 1]
        else:
            cand = [(3 * t - 1) // 2, (3 * t + 1) // 2]
        gs = [g for g in cand if g >= max(5, L) and is_prime(g)
              and (arc(g) == t or g - arc(g) == t)]
        if gs:
            tab[t] = gs
    return tab


def bipartite(pairs, supply):
    """can each pair be given a distinct aux prime able to span its distance?"""
    prs = list(pairs)
    match = {}

    def try_assign(i, seen):
        for g, ds in supply.items():
            d = prs[i][1] - prs[i][0]
            if d in ds and g not in seen:
                seen.add(g)
                if g not in match or try_assign(match[g], seen):
                    match[g] = i
                    return True
        return False

    for i in range(len(prs)):
        if not try_assign(i, set()):
            return False
    return True


def max_pairs(H, supply):
    best = 0

    def rec(rem, chosen):
        nonlocal best
        if len(chosen) > best and bipartite(chosen, supply):
            best = len(chosen)
        if len(rem) < 2:
            return
        i = rem[0]
        rec(rem[1:], chosen)
        for jx in range(1, len(rem)):
            j = rem[jx]
            if any((j - i) in ds for ds in supply.values()):
                rec([x for k, x in enumerate(rem) if k not in (0, jx)],
                    chosen + [(i, j)])

    rec(list(H), [])
    return best


def analyse(K, L, cap=8_000_000, show=True):
    T = T_of(L)
    supply = aux_supply(L)
    rows = []
    total_phases = 0
    for m in range(0, min(K, len(T)) + 1):
        for S in combinations(T, m):
            capacity = sum(maxstrike(g, L) for g in S) + 2 * (K - m)
            if capacity < L:
                continue
            prod = 1
            for g in S:
                prod *= g
            if prod > cap:
                rows.append((S, K - m, capacity, prod, None, None))
                continue
            total_phases += prod
            best = (10 ** 9, None)

            def rec(i, cov):
                nonlocal best
                if i == len(S):
                    H = [j for j in range(L) if not (cov >> j & 1)]
                    if len(H) > 2 * (K - m):
                        return
                    need = len(H) - max_pairs(H, supply)
                    if need < best[0]:
                        best = (need, tuple(H))
                    return
                g = S[i]
                for ph in range(g):
                    rec(i + 1, cov | strike_mask(g, ph, L))

            rec(0, 0)
            rows.append((S, K - m, capacity, prod, best[0], best[1]))
    return T, supply, rows, total_phases


def main():
    os.makedirs(RESULTS, exist_ok=True)
    for K in range(2, 7):
        L = A_K[K]
        say("=" * 96)
        say(f"K = {K},  L = A(K) = {L}:  no K primes >= 5 cover {L} consecutive columns")
        say("=" * 96)
        T = T_of(L)
        say(f"  T(L) = gears below L = {T}")
        say(f"  maxstrike in the run: " +
            ", ".join(f"{g}->{maxstrike(g, L)}" for g in T) +
            f"; every gear >= {L} strikes at most 2")
        st = span_table(L)
        say(f"  span table (distance -> the primes >= {L} that can span it):")
        say("    " + "; ".join(f"{t}:{gs}" for t, gs in sorted(st.items())))
        T2, supply, rows, tot = analyse(K, L)
        say(f"  aux primes with a usable pair distance: "
            f"{ {g: ds for g, ds in supply.items()} }")
        say(f"  cases surviving the counting filter: {len(rows)}"
            f"   (total phase vectors enumerated: {tot})")
        say(f"    {'S = chosen gears below L':>30} {'aux':>4} {'cap':>4} {'phases':>9} "
            f"{'aux needed':>10} {'verdict':>10}")
        allclosed = True
        for S, naux, capacity, prod, need, H in rows:
            if need is None:
                say(f"    {str(list(S)):>30} {naux:>4} {capacity:>4} {prod:>9} "
                    f"{'not run':>10} {'SKIPPED':>10}")
                allclosed = False
                continue
            ok = need > naux
            allclosed &= ok
            say(f"    {str(list(S)):>30} {naux:>4} {capacity:>4} {prod:>9} "
                f"{need:>10} {'no cover' if ok else 'COVER':>10}"
                f"   best holes {list(H) if H else '-'}")
        say(f"  ==> {'every case closed: A(%d) <= %d' % (K, L) if allclosed else 'NOT CLOSED'}")
        say()
    with open(os.path.join(RESULTS, "sk_cases.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
