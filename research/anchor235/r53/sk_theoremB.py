"""r53 sk_theoremB - the mechanism and the certificates for A(K), K = 1..6.

Three things.

(1) THE CAPACITY TABLE.  maxstrike(g, L) = the largest number of an L-run's columns one gear
    can strike = 2*floor(L/g) + (2 if L mod g > a_g else 1 if L mod g >= 1 else 0), and the
    per-case bound "b big gears contribute at most 2 each".  Cases killed by counting alone
    are marked; they need no enumeration.

(2) THE ARC SUPPLY.  At level L a gear is BIG if g - a_g >= L; a big gear realises exactly
    {}, {i, i+a} (i+a <= L-1), {i} (i < a or i >= L-a).  So a big gear can join two holes only
    at an EVEN distance a with 3a -+ 1 prime and that prime big at L, and only if a <= L-1.
    The table of usable pair-distances is printed for each L.

(3) THE SPLIT ENUMERATION (a complete certificate, independent of the MILP).
    Let Sm(L) = {p >= 5 prime : p - a_p <= L-1} be the small pool.  Any K-set's small part is
    a subset of Sm(L) of size <= K, so it is contained in some K-subset S of Sm(L).  For each
    such S run an exhaustive search whose concrete gears are restricted to S, with the domino
    and single types of the type lemma also available, and a budget of K gears in all.  If
    every S says NO COVER then no K primes cover L.  Each sub-search is small.

    For the small parts whose phase count is manageable the HOLE-DISTANCE DICTIONARY is also
    printed: the least number of holes over phases and, when it is 2, the distances realisable.

Run:  uv run python research/anchor235/r53/sk_theoremB.py
"""
import os
from itertools import combinations

from sk_core import (RESULTS, Level, arc, is_prime, masks_for, primes_upto, sep,
                     strike_mask)

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


A_K = {1: 2, 2: 5, 3: 7, 4: 16, 5: 22, 6: 28}


def maxstrike(g, L):
    q, r = divmod(L, g)
    extra = 2 if r > arc(g) else (1 if r >= 1 else 0)
    return 2 * q + extra


def small_pool(L):
    return [p for p in primes_upto(3 * L + 4) if p >= 5 and p - arc(p) <= L - 1]


def pair_arcs(L):
    """Even a <= L-1 such that some prime 3a -+ 1 is big at L; with its multiplicity."""
    out = []
    a = 2
    while a <= L - 1:
        gs = [g for g in (3 * a - 1, 3 * a + 1)
              if g >= 5 and is_prime(g) and arc(g) == a and g - a >= L]
        if gs:
            out.append((a, gs))
        a += 2
    return out


# ---------------------------------------------------------------- restricted search

class RestrictedLevel(Level):
    """Engine (2) with the concrete primes restricted to the set S."""

    def __init__(self, L, K, S):
        self.S = set(S)
        Level.__init__(self, L, K)

    def _filter(self):
        pass


def build_restricted(L, K, S):
    lv = Level.__new__(Level)
    lv.L = L
    lv.K = K
    lv.full = (1 << L) - 1
    items = [('p', p, 1) for p in sorted(S)]
    a = 2
    while a < L:
        m = 0
        for g in (3 * a - 1, 3 * a + 1):
            if g >= 5 and is_prime(g) and arc(g) == a and g - a >= L:
                m += 1
        if m:
            items.append(('d', a, m))
        a += 2
    items.append(('s', None, K))
    lv.items = items
    lv.cap = []
    lv.wins = [w for w in (8, 16, 24, 32) if w < L]
    lv.wcap = {w: [] for w in lv.wins}
    for kind, key, _m in items:
        ms = masks_for(kind, key, L)
        lv.cap.append(max(bin(m).count("1") for m in ms))
        for w in lv.wins:
            wm = (1 << w) - 1
            best = 0
            for m in ms:
                for s in range(0, L):
                    c = bin((m >> s) & wm).count("1")
                    if c > best:
                        best = c
            lv.wcap[w].append(best)
    return lv


def split_certificate(K, L, verbose=False):
    """Complete: for every K-subset S of the small pool, is S + (big items) a cover?"""
    pool = small_pool(L)
    covers = []
    n = 0
    for S in combinations(pool, min(K, len(pool))):
        lv = build_restricted(L, K, S)
        if lv.coverable():
            covers.append((S, lv.witness))
        n += 1
    return n, covers, pool


# ------------------------------------------------- the matching route, case by case

def max_pairs(H, supply):
    """Largest number of disjoint pairs of holes whose distances are available arcs,
    respecting each arc's multiplicity (how many big primes carry it)."""
    H = list(H)
    best = 0

    def rec(rem, used, npair):
        nonlocal best
        if npair > best:
            best = npair
        if len(rem) < 2:
            return
        i = rem[0]
        rec(rem[1:], used, npair)          # leave hole i to a singleton
        for jx in range(1, len(rem)):
            j = rem[jx]
            d = j - i
            if supply.get(d, 0) > used.get(d, 0):
                u = dict(used)
                u[d] = u.get(d, 0) + 1
                rec([x for k, x in enumerate(rem) if k not in (0, jx)], u, npair + 1)

    rec(H, {}, 0)
    return best


def case_report(S, L, b, cap=4_000_000):
    """Exhaustive over the phases of S: the least number of BIG gears needed to finish.

    cost(phase) = |H| - maxpairs(H); the K-set covers L iff some phase has cost <= b."""
    prod = 1
    for g in S:
        prod *= g
    if prod > cap:
        return None
    supply = {a: len(gs) for a, gs in pair_arcs(L)}
    best = (10 ** 9, None, None)
    hist = {}

    def rec(i, cov):
        nonlocal best
        if i == len(S):
            H = [j for j in range(L) if not (cov >> j & 1)]
            if len(H) > 2 * b + 4:
                return
            c = len(H) - max_pairs(H, supply)
            hist[c] = hist.get(c, 0) + 1
            if c < best[0]:
                best = (c, tuple(H), None)
            return
        g = S[i]
        for ph in range(g):
            rec(i + 1, cov | strike_mask(g, ph, L))

    rec(0, 0)
    return best[0], best[1], hist, prod


# ---------------------------------------------------------------- hole dictionary

def dictionary(S, L, cap=4_000_000):
    """min holes over phases, and the realisable hole sets at that minimum."""
    prod = 1
    for g in S:
        prod *= g
    if prod > cap:
        return None
    full = (1 << L) - 1
    best = L + 1
    dist = set()
    sets = []

    def rec(i, cov):
        nonlocal best
        if i == len(S):
            h = [j for j in range(L) if not (cov >> j & 1)]
            if len(h) < best:
                best = len(h)
                dist.clear()
                sets.clear()
            if len(h) == best:
                sets.append(tuple(h))
                if len(h) == 2:
                    dist.add(h[1] - h[0])
            return
        g = S[i]
        for ph in range(g):
            rec(i + 1, cov | strike_mask(g, ph, L))

    rec(0, 0)
    return best, sorted(dist), sorted(set(sets))


def main():
    os.makedirs(RESULTS, exist_ok=True)

    say("=" * 92)
    say("1.  CAPACITY: maxstrike(g, L), and the per-case counting bound")
    say("=" * 92)
    for K in range(3, 7):
        L = A_K[K]
        pool = small_pool(L)
        say(f"  K = {K}, L = A(K) = {L}.  small pool Sm(L) = {pool}")
        say("     maxstrike: " + ", ".join(f"{g}->{maxstrike(g, L)}" for g in pool))
        say(f"     {'b':>3} {'k=K-b':>5} {'best cap':>8}   small parts with capacity + 2b >= L")
        for b in range(0, K + 1):
            k = K - b
            best = sorted(pool, key=lambda g: -maxstrike(g, L))[:k]
            capp = sum(maxstrike(g, L) for g in best) + 2 * b
            surv = [S for S in combinations(pool, k)
                    if sum(maxstrike(g, L) for g in S) + 2 * b >= L]
            txt = ("closed by counting" if capp < L
                   else f"{len(surv)}: " + "; ".join(str(list(S)) for S in surv[:8])
                   + (" ..." if len(surv) > 8 else ""))
            say(f"     {b:>3} {k:>5} {capp:>8}   {txt}")
        say()

    say("=" * 92)
    say("2.  ARC SUPPLY: the pair-distances a big gear can span at level L")
    say("=" * 92)
    for K in range(3, 7):
        L = A_K[K]
        pa = pair_arcs(L)
        say(f"  L = {L}: " + ("; ".join(f"a={a} by {gs}" for a, gs in pa) if pa else "none"))
    say()

    say("=" * 92)
    say("3.  HOLE-DISTANCE DICTIONARIES of the small parts that matter")
    say("=" * 92)
    for S, L in [((5, 7), 7), ((5,), 7), ((5, 7), 16), ((5, 7, 11), 16),
                 ((5, 7, 11), 15), ((5, 7, 13), 16), ((5, 7, 17), 16),
                 ((5, 7, 11, 13), 16), ((5, 7, 11), 22), ((5, 7, 11, 23), 22),
                 ((5, 7, 11, 13), 22), ((5, 7, 11, 17), 22)]:
        d = dictionary(list(S), L)
        if d is None:
            say(f"  S = {S}, L = {L}: phase count too large")
            continue
        best, dist, sets = d
        extra = (f"  distances {dist}" if best == 2 else "")
        say(f"  S = {str(list(S)):>18}, L = {L:>2}: min holes = {best}{extra}"
            f"   ({len(sets)} hole sets at the minimum)")
    say()

    say("=" * 92)
    say("3b. THE MATCHING ROUTE, case by case: for every (b, small part S) that survives the")
    say("     counting bound, the least number of big gears the holes of S ever need.")
    say("     A cover needs that number to be <= b.")
    say("=" * 92)
    for K in range(3, 7):
        L = A_K[K]
        pool = small_pool(L)
        say(f"  K = {K}, L = {L}")
        for b in range(0, K + 1):
            k = K - b
            surv = [S for S in combinations(pool, k)
                    if sum(maxstrike(g, L) for g in S) + 2 * b >= L]
            for S in surv:
                r = case_report(list(S), L, b, cap=(4_000_000 if K <= 5 else 300_000))
                if r is None:
                    say(f"     b={b} S={list(S)}: phase count too large "
                        f"(left to the split certificate)")
                    continue
                c, H, hist, prod = r
                say(f"     b={b} S={str(list(S)):>24}: {prod:>9} phases, "
                    f"least big gears needed = {c:>2} "
                    f"{'<= b  COVER' if c <= b else '>  b  no cover'}"
                    f"   (best hole set {list(H) if H else '-'})")
        say()

    say("=" * 92)
    say("4.  THE SPLIT CERTIFICATE: every K-subset of the small pool, at L = A(K)")
    say("     (complete: any K-set's small part sits inside one of these)")
    say("=" * 92)
    for K in range(1, 7):
        L = A_K[K]
        n, covers, pool = split_certificate(K, L)
        say(f"  K = {K}, L = {L}: {n} subsets of Sm(L) (|Sm| = {len(pool)}), "
            f"{len(covers)} covering  ->  {'NO K-SET COVERS L' if not covers else 'COVER FOUND'}")
        for S, w in covers[:3]:
            say(f"      cover with small part {S}: {w}")
    say()

    say("     and the same one level down, at L = A(K) - 1 (a cover must exist there):")
    for K in range(2, 7):
        L = A_K[K] - 1
        n, covers, pool = split_certificate(K, L)
        say(f"  K = {K}, L = {L}: {len(covers)} of {n} subsets cover"
            f"   {'OK' if covers else 'UNEXPECTED'}")

    with open(os.path.join(RESULTS, "sk_theoremB.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
