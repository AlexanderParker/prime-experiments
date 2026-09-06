"""R7 item 1: THE LOADED RECORD RULE, proved and verified.

The rule (top_machine_4.md L53, here stated exactly and proved in top_machine_7.md):

    core(L) = {g in G : g <= L + 1}          t(L) = #{g in G : g > L + 1}
    a core phase vector leaves the uncovered set U = [0,L) \\ (core traces)
    D(U)   = sum over the maximal step-2 runs of U inside each parity class of ceil(run/2)

    [0, L) coverable  <=>  min over core phase vectors of D(U) <= t(L)
    F_top(G)          =    max { L : that holds }

This script checks:
  A. the matching lemma  min #pieces = D(U)  exhaustively on all subsets of [0,L), L <= 14
  B. the rule against a full-period scan on every pairwise-coprime odd gear set with
     period below the cap
  C. the rule against the 13 independently known records
  D. the rule against the exhaustive triple / quadruple tables of top_machine_2.md L30
  E. the SHARPNESS of the tail hypothesis: the same engine with core = {g <= L}

usage: uv run python research/topmachine/r7/rule.py
"""

import itertools
import json
import os
from functools import lru_cache
from math import gcd, prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = []
SCAN_CAP = 24_000_000


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


# ---------------------------------------------------------------- the domino cost

def domino_cost_set(cells):
    """D(U) for a sorted iterable of cells."""
    cost = 0
    for par in (0, 1):
        cs = [c for c in cells if c % 2 == par]
        if not cs:
            continue
        run = 1
        for a, b in zip(cs, cs[1:]):
            if b - a == 2:
                run += 1
            else:
                cost += (run + 1) // 2
                run = 1
        cost += (run + 1) // 2
    return cost


def domino_cost_mask(mask, L):
    return domino_cost_set([c for c in range(L) if (mask >> c) & 1])


def D_empty(L):
    """the empty-core cost of [0, L): closed form and check."""
    a, b = (L + 1) // 2, L // 2
    return (a + 1) // 2 + (b + 1) // 2


# ---------------------------------------------------------------- the matching lemma (A)

def min_pieces_bruteforce(cells):
    """Minimum number of pieces (each a distance-2 pair or a single cell) whose union is U.

    Pure search: repeatedly either take a lone cell or pair the smallest with smallest+2.
    """
    cs = tuple(sorted(cells))
    if not cs:
        return 0

    @lru_cache(maxsize=None)
    def rec(rest):
        if not rest:
            return 0
        c = rest[0]
        best = 1 + rec(rest[1:])                      # c alone
        if c + 2 in rest:                             # c paired with c + 2
            r = tuple(x for x in rest[1:] if x != c + 2)
            best = min(best, 1 + rec(r))
        return best

    v = rec(cs)
    rec.cache_clear()
    return v


# ---------------------------------------------------------------- the rule

def traces(g, L):
    """the distinct traces of gear g inside [0, L)"""
    seen, out = set(), []
    for a in range(g):
        cells = 0
        for c in range(a % g, L, g):
            cells |= 1 << c
        for c in range((a + g - 2) % g, L, g):
            cells |= 1 << c
        if cells not in seen:
            seen.add(cells)
            out.append(cells)
    return out


def coverable(gears, L, boundary=1):
    """Is [0, L) coverable?  boundary = 1 is the rule (core = {g <= L + 1});
    boundary = 0 is the WRONG rule (core = {g <= L}), kept to show the hypothesis is sharp."""
    if L <= 0:
        return True
    core = [g for g in gears if g <= L + boundary]
    t = len(gears) - len(core)
    full = (1 << L) - 1
    if not core:
        return D_empty(L) <= t
    tr = [traces(g, L) for g in core]
    order = sorted(range(len(core)), key=lambda i: -max(bin(c).count("1") for c in tr[i]))
    tr = [tr[i] for i in order]
    sizes = [max(bin(c).count("1") for c in t_) for t_ in tr]
    suffix = [0] * (len(tr) + 1)
    for i in range(len(tr) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + sizes[i]
    seen = set()

    def dfs(i, covered):
        if covered == full:
            return True
        key = (i, covered)
        if key in seen:
            return False
        unc = full & ~covered
        n_unc = bin(unc).count("1")
        if i == len(tr):
            if domino_cost_mask(unc, L) <= t:
                return True
            seen.add(key)
            return False
        # valid bound: the best the remaining core gears can do is cover suffix[i] more cells,
        # and whatever is left needs at least ceil(rest/2) tail dominoes
        if (max(0, n_unc - suffix[i]) + 1) // 2 > t:
            seen.add(key)
            return False
        for cells in tr[i]:
            if dfs(i + 1, covered | cells):
                return True
        seen.add(key)
        return False

    return dfs(0, 0)


def F_rule(gears, boundary=1, hi=None):
    """max L that is coverable (coverability is monotone in L: restrict the same phases)."""
    if hi is None:
        hi = 3 * len(gears) + 40
    best = 0
    for L in range(1, hi + 1):
        if coverable(gears, L, boundary):
            best = L
        elif L > best + 8:
            break
    return best


# ---------------------------------------------------------------- the scan

def F_scan(gears):
    """exact record by a full-period scan (cyclic)"""
    W = prod(gears)
    a = np.ones(W, dtype=bool)
    for g in gears:
        a[0::g] = False
        a[(g - 2) % g::g] = False
    pos = np.flatnonzero(a)
    if pos.size == 0:
        return None
    d = np.diff(pos)
    wrap = pos[0] + W - pos[-1]
    return int(max(d.max(initial=0), wrap)) - 1


def pairwise_coprime(s):
    return all(gcd(a, b) == 1 for a, b in itertools.combinations(s, 2))


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


KNOWN = {
    (7, 11, 13): 6, (11, 13, 17): 5, (13, 17, 19): 5, (17, 19, 23): 5, (19, 23, 29): 5,
    (7, 11, 13, 17): 9, (11, 13, 17, 19): 8, (13, 17, 19, 23): 8, (17, 19, 23, 29): 8,
    (11, 13, 17, 19, 23): 10,
    (7, 11, 13, 17, 19, 23, 29, 31): 32,
    (13, 17, 19, 23, 29, 31, 37, 41): 18,
    (19, 23, 29, 31, 37, 41, 43, 47): 16,
}


def main():
    summary = {}
    say("# R7.1  The loaded record rule")
    say()

    # ---------------------------------------------------------------- A
    say("## A. The matching lemma: min #pieces = D(U), exhaustive")
    say()
    say("| L | subsets | mismatches |")
    say("|---|---|---|")
    badA = 0
    for L in range(1, 15):
        bad = 0
        for mask in range(1 << L):
            cells = [c for c in range(L) if (mask >> c) & 1]
            if min_pieces_bruteforce(cells) != domino_cost_set(cells):
                bad += 1
        badA += bad
        say(f"| {L} | {1 << L} | {bad} |")
    say()
    say(f"**{badA} mismatches** over {sum(1 << L for L in range(1, 15)):,} subsets: the domino "
        "cost IS the minimum number of pieces.")
    say()
    summary["matching_lemma_mismatches"] = badA

    # the closed form of the empty-core cost
    say("Empty-core cost `D(L)` against the closed form `2 floor(L/4) + min(L mod 4, 2)`:")
    bad = sum(1 for L in range(0, 400)
              if D_empty(L) != 2 * (L // 4) + min(L % 4, 2))
    say(f"`L = 0..399`: **{bad} mismatches**.  `D(0..12) = "
        f"{', '.join(str(D_empty(L)) for L in range(13))}`.")
    say()
    summary["D_closed_form_mismatches"] = bad

    # ---------------------------------------------------------------- B
    say("## B. The rule against a full-period scan")
    say()
    pool = [5, 7, 9, 11, 13, 17, 19, 23, 25, 29, 31, 37, 41, 43, 47, 49]
    sets = []
    for m in range(2, 7):
        for s in itertools.combinations(pool, m):
            if prod(s) <= SCAN_CAP and pairwise_coprime(s):
                sets.append(s)
    say(f"All pairwise-coprime subsets of `{pool}` of size 2..6 with period <= "
        f"{SCAN_CAP:,}: **{len(sets)} gear sets**.")
    say()
    badB = 0
    rowsB = []
    loaded = 0
    for s in sets:
        fs = F_scan(list(s))
        fr = F_rule(list(s))
        core = [g for g in s if g <= fr + 1]
        if core:
            loaded += 1
        rowsB.append({"gears": list(s), "scan": fs, "rule": fr, "core": core})
        if fs != fr:
            badB += 1
    say(f"**{badB} mismatches of {len(sets)}**; {loaded} of them are LOADED wheels "
        f"(nonempty core), {len(sets) - loaded} free.")
    say()
    say("A sample of the loaded ones (the cases the free parity law cannot reach):")
    say()
    say("| gears | m | period | scan | rule | core | t |")
    say("|---|---|---|---|---|---|---|")
    shown = 0
    for r in rowsB:
        if r["core"] and shown < 18:
            g = r["gears"]
            say(f"| {','.join(map(str, g))} | {len(g)} | {prod(g):,} | {r['scan']} | "
                f"{r['rule']} | {{{','.join(map(str, r['core']))}}} | "
                f"{len(g) - len(r['core'])} |")
            shown += 1
    say()
    summary["scan_sets"] = len(sets)
    summary["scan_mismatches"] = badB
    summary["scan_loaded"] = loaded

    # ---------------------------------------------------------------- C
    say("## C. The rule against the 13 independently known records")
    say()
    say("| gears | m | known | rule | core | t |")
    say("|---|---|---|---|---|---|")
    badC = 0
    for gs, known in KNOWN.items():
        f = F_rule(list(gs))
        core = [g for g in gs if g <= f + 1]
        say(f"| {','.join(map(str, gs))} | {len(gs)} | {known} | {f} | "
            f"{('{' + ','.join(map(str, core)) + '}') if core else 'empty'} | "
            f"{len(gs) - len(core)} |")
        if f != known:
            badC += 1
    say()
    say(f"**{badC} mismatches of {len(KNOWN)}**.")
    say()
    summary["known_mismatches"] = badC

    # ---------------------------------------------------------------- D
    say("## D. The rule against the exhaustive triple and quadruple tables (L30)")
    say()
    ps = [p for p in primes_upto(97) if p >= 7]
    badD = 0
    counts = {}
    for m, expect in ((3, (5, 6)), (4, (8, 9))):
        tally = {}
        for s in itertools.combinations(ps, m):
            f = F_rule(list(s))
            tally[f] = tally.get(f, 0) + 1
            want = expect[1] if 7 in s else expect[0]
            if f != want:
                badD += 1
        counts[m] = tally
        say(f"`m = {m}`: {sum(tally.values()):,} sets, rule gives "
            f"{ {k: v for k, v in sorted(tally.items())} }")
    say()
    say(f"Expected from L30: triples 5 (1,330 sets) / 6 (210 with 7); quadruples 8 (5,985) / "
        f"9 (1,330 with 7).  **{badD} mismatches of {1540 + 7315:,}**.")
    say()
    summary["L30_mismatches"] = badD
    summary["L30_counts"] = {str(k): v for k, v in counts.items()}

    # ---------------------------------------------------------------- E
    say("## E. The tail hypothesis is sharp: `g > L + 1`, not `g > L`")
    say()
    say("Running the same engine with the boundary gear `g = L + 1` counted as a TAIL gear "
        "(i.e. core = `{g <= L}`):")
    say()
    say("| gears | m | scan | rule (`g <= L+1` core) | wrong (`g <= L` core) |")
    say("|---|---|---|---|---|")
    badE = 0
    testE = [(9, 11, 13, 17), (13, 17, 19, 23, 29, 31), (5, 7), (7, 11, 13),
             (11, 13, 17, 19, 23), (5, 7, 11, 13), (9, 11, 13), (7, 11, 13, 17)]
    for s in testE:
        fs = F_scan(list(s)) if prod(s) <= SCAN_CAP else None
        fr = F_rule(list(s), boundary=1)
        fw = F_rule(list(s), boundary=0)
        if fw != fr:
            badE += 1
        say(f"| {','.join(map(str, s))} | {len(s)} | {fs} | {fr} | "
            f"{fw}{' **wrong**' if fw != fr else ''} |")
    say()
    say(f"The wrong boundary differs at **{badE} of {len(testE)}** sets: the gear `g = L + 1` "
        "shows the END PAIR `{0, L-1}`, which crosses parity and is not a domino, so it cannot "
        "be counted as a tail gear.")
    say()
    summary["sharp_boundary_differences"] = badE

    json.dump(summary, open(os.path.join(RES, "rule.json"), "w"), indent=1)
    with open(os.path.join(RES, "rule.out"), "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
