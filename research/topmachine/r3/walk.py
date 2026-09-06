"""Q1-Q5, Q17, Q18: the walk, its closed form, its distribution, the correlation."""

import sys
from itertools import combinations
from math import comb, prod

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
from core import (all_struck_counts, gaps_of, open_mask_pair,
                  open_mask_triple, walk_lengths_fast)

OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


# ---------------------------------------------------------------- Q1, Q2, Q3

def mex_walk_pair(gears, W):
    """L(x) = mex{(-x) mod g, (-x-2) mod g}, vectorised over x in [0, W)."""
    m = len(gears)
    top = 2 * m
    x = np.arange(W)
    cov = np.zeros((W, top + 2), dtype=bool)
    for g in gears:
        a = (-x) % g
        b = (-x - 2) % g
        cov[np.arange(W), np.minimum(a, top + 1)] |= (a <= top)
        cov[np.arange(W), np.minimum(b, top + 1)] |= (b <= top)
    cov[:, top + 1] = True  # sentinel
    return cov.argmin(axis=1)


def mex_walk_triple(gears, W):
    m = len(gears)
    top = 3 * m
    x = np.arange(W)
    cov = np.zeros((W, top + 2), dtype=bool)
    for g in gears:
        for i in (0, 1, 2):
            a = (-x - i) % g
            cov[np.arange(W), np.minimum(a, top + 1)] |= (a <= top)
    cov[:, top + 1] = True
    return cov.argmin(axis=1)


def q1q2(sets):
    say("### Q1/Q2  the mex form and the location bound (pair view)")
    say()
    say("| gears | m | W | every gear > 2m | mex == scan | max L | 2m-(m mod 2) |")
    say("|---|---|---|---|---|---|---|")
    for gears in sets:
        W = prod(gears)
        m = len(gears)
        mask = open_mask_pair(gears)
        L = walk_lengths_fast(mask)
        big = min(gears) > 2 * m
        mx = mex_walk_pair(gears, W)
        agree = "0 mismatches" if np.array_equal(L, mx) else f"MISMATCH {int((L != mx).sum())}"
        bound = 2 * m - (m % 2)
        say(f"| {','.join(map(str,gears))} | {m} | {W:,} | {big} | {agree} | "
            f"{int(L.max())} | {bound} |")
    say()


def q3(sets):
    say("### Q3  the twin-candidate walk (single-number view, teeth {0,-1,-2})")
    say()
    say("| gears | m | W | every gear >= 3m+3 | starts prod(g-3) | mex == scan | max R | 3m |")
    say("|---|---|---|---|---|---|---|---|")
    for gears in sets:
        W = prod(gears)
        m = len(gears)
        mask = open_mask_triple(gears)
        R = walk_lengths_fast(mask)
        cnt = int(mask.sum())
        pred = prod(g - 3 for g in gears)
        big = min(gears) >= 3 * m + 3
        mx = mex_walk_triple(gears, W)
        agree = "0 mismatches" if np.array_equal(R, mx) else f"MISMATCH {int((R != mx).sum())}"
        say(f"| {','.join(map(str,gears))} | {m} | {W:,} | {big} | "
            f"{cnt:,} = {pred:,} {'yes' if cnt==pred else 'NO'} | {agree} | "
            f"{int(R.max())} | {3*m} |")
    say()


# ---------------------------------------------------------------- Q4, Q5

def T_brute(j):
    """T(j,k,e) by brute force over subsets of [0,j)."""
    t = {}
    for k in range(j + 1):
        for S in combinations(range(j), k):
            Sset = set(S)
            e = sum(1 for s in S if s - 2 in Sset)
            t[(k, e)] = t.get((k, e), 0) + 1
    return t


def T_closed(j):
    """T(j,k,e) by the path convolution: [0,j) splits into evens and odds."""
    n1 = (j + 1) // 2
    n2 = j // 2

    def path(n, k, e):
        if k == 0:
            return 1 if e == 0 else 0
        return comb(k - 1, e) * comb(n - k + 1, k - e)

    t = {}
    for k1 in range(n1 + 1):
        for e1 in range(k1 + 1):
            p1 = path(n1, k1, e1)
            if not p1:
                continue
            for k2 in range(n2 + 1):
                for e2 in range(k2 + 1):
                    p2 = path(n2, k2, e2)
                    if not p2:
                        continue
                    key = (k1 + k2, e1 + e2)
                    t[key] = t.get(key, 0) + p1 * p2
    return {k: v for k, v in t.items() if v}


def C_closed(gears, j):
    t = T_closed(j)
    tot = 0
    for (k, e), cnt in t.items():
        tot += (-1) ** k * cnt * prod(g - 2 * k + e for g in gears)
    return tot


def q4q5(sets, jmax=12):
    say("### Q4/Q5  the all-struck count C(j): distribution, gap census, closed form")
    say()
    # T table identity first
    ok = all(T_brute(j) == {k: v for k, v in T_closed(j).items()}
             for j in range(0, 13))
    say(f"T(j,k,e) brute force == path-convolution closed form for j = 0..12: "
        f"{'0 mismatches' if ok else 'MISMATCH'}")
    say()
    for gears in sets:
        W = prod(gears)
        m = len(gears)
        mask = open_mask_pair(gears)
        L = walk_lengths_fast(mask)
        F = int(L.max())
        jm = min(jmax, F + 2)
        C = all_struck_counts(mask, jm + 1)
        Cc = [C_closed(gears, j) for j in range(jm + 2)]
        hyp = min(gears) >= jm + 2
        # distribution
        dist = np.bincount(L, minlength=F + 2)
        d1 = [C[j] - C[j + 1] for j in range(jm + 1)]
        ok_dist = all(int(dist[j]) == d1[j] for j in range(min(len(dist), jm + 1)))
        # gap census
        g_arr = gaps_of(mask)
        maxd = int(g_arr.max())
        N = np.bincount(g_arr, minlength=maxd + 2)
        ok_gap = all(int(N[d]) == C[d - 1] - 2 * C[d] + C[d + 1]
                     for d in range(1, min(maxd, jm - 1) + 1))
        ok_cl = all(C[j] == Cc[j] for j in range(jm + 2)) if hyp else None
        # tail: L = j equals gaps >= j+1
        ok_tail = all(int(dist[j]) == int(N[j + 1:].sum()) for j in range(1, F + 1))
        say(f"**{','.join(map(str,gears))}**  W = {W:,}, m = {m}, F_top = {F}, "
            f"max gap = {maxd}, hypothesis (every gear >= j+2 to j = {jm}): {hyp}")
        say(f"- C(j), j = 0..{jm}: {', '.join(f'{C[j]:,}' for j in range(jm+1))}")
        if hyp:
            say(f"- closed form agrees: {'0 mismatches' if ok_cl else 'MISMATCH'}")
        say(f"- #{{L = j}} = C(j) - C(j+1): {'0 mismatches' if ok_dist else 'MISMATCH'}")
        say(f"- #{{L = j}} = #{{gaps >= j+1}} (j >= 1): {'0 mismatches' if ok_tail else 'MISMATCH'}")
        say(f"- N_d = C(d-1) - 2C(d) + C(d+1): {'0 mismatches' if ok_gap else 'MISMATCH'}")
        say(f"- max{{j : C(j) > 0}} = {max(j for j in range(len(C)) if C[j] > 0)}  (F_top = {F})")
        say()


# ---------------------------------------------------------------- Q17, Q18

def q17q18(sets, dmax=40):
    say("### Q17/Q18  the correlation product and the holes")
    say()
    say("| gears | B(d) = prod c_g(d) for d = 1..%d | B(1) = prod(g-4) | B(2) = prod(g-3) | gap holes below the record |" % dmax)
    say("|---|---|---|---|---|")
    for gears in sets:
        W = prod(gears)
        mask = open_mask_pair(gears)
        ok = True
        for d in range(1, dmax + 1):
            meas = int((mask & np.roll(mask, -d)).sum())
            pred = 1
            for g in gears:
                if d % g == 0:
                    pred *= g - 2
                elif d % g == 2 % g or d % g == (-2) % g:
                    pred *= g - 3
                else:
                    pred *= g - 4
            if meas != pred:
                ok = False
                break
        g_arr = gaps_of(mask)
        maxd = int(g_arr.max())
        present = set(int(v) for v in np.unique(g_arr))
        holes = [d for d in range(1, maxd) if d not in present]
        b1 = int((mask & np.roll(mask, -1)).sum())
        b2 = int((mask & np.roll(mask, -2)).sum())
        say(f"| {','.join(map(str,gears))} | {'0 mismatches' if ok else 'MISMATCH'} | "
            f"{b1:,} = {prod(g-4 for g in gears):,} | {b2:,} = {prod(g-3 for g in gears):,} | "
            f"{holes} |")
    say()
    say("Triple view, holes:")
    say()
    say("| gears | gaps present (to the record) | holes |")
    say("|---|---|---|")
    for gears in sets:
        mask = open_mask_triple(gears)
        g_arr = gaps_of(mask)
        maxd = int(g_arr.max())
        present = sorted(set(int(v) for v in np.unique(g_arr)))
        holes = [d for d in range(1, maxd) if d not in present]
        say(f"| {','.join(map(str,gears))} | {present[:10]}{' ...' if len(present)>10 else ''} | {holes} |")
    say()


if __name__ == "__main__":
    SETS = [
        (7, 11, 13), (11, 13, 17), (13, 17, 19), (17, 19, 23), (19, 23, 29),
        (7, 11, 13, 17), (11, 13, 17, 19), (13, 17, 19, 23), (17, 19, 23, 29),
        (11, 13, 17, 19, 23),
    ]
    q1q2(SETS)
    q3([(13, 17, 19), (17, 19, 23), (19, 23, 29), (17, 19, 23, 29),
        (19, 23, 29, 31), (23, 29, 31, 37), (7, 11, 13), (11, 13, 17)])
    q4q5(SETS)
    q17q18(SETS)
    with open("research/topmachine/r3/results/walk.out", "w") as f:
        f.write("\n".join(OUT))
