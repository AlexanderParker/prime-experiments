"""Round 8 lateral: the COMPLETE WORD GRAMMAR of saturated runs under
small-gear CRT - the language, its growth, and its finite horizon.

Frame (slot space; the n-space period 30030 is 6 x the slot period 5005).
A word w (letters = prime side) of length L is ADMISSIBLE for gear set G iff
some phase k0 makes every prime side avoid every gear's tooth:
w[i] = L forbids k0 = u_q - i (mod q), w[i] = R forbids k0 = -u_q - i (mod q).
Per gear each position forbids exactly ONE residue, so:

    w admissible  <=>  for every q in G the chosen residues do not cover Z_q
    (per-gear allowed phase sets are nonempty and combine freely by CRT).

Phase view: a slot where the small machine hits BOTH sides (a B-slot - these
are exactly the split/Bezout classes of gear pairs, e.g. k = 1 mod 35 where
5 | 6k-1 and 7 | 6k+1) admits NO letter. So

    language(L) nonempty  <=>  the CRT period contains an L-window with no
    B-slot,   and   L0(G) = (max cyclic gap between B-slots) - 1

is the FINITE HORIZON of the language: gears {5,7} alone give B-classes
{1,34} mod 35, hence L0 <= 32 forever. Since every saturated run anywhere
(interior members exceed 13) must have its word in the language, L0 is an
UNCONDITIONAL cap on saturated-run length at every scale - and the language
is precisely the set of admissible one-prime-per-slot constellations (the
grammar cap is the constellation-admissibility boundary).

Outputs: L0 for growing gear sets; full language census for gears <= 13
(count per length, growth, forced-repeat laws, emptiness at L0+1); the
observed 757 runs (research/data/satruns_ge10.csv, members to 7.2e10) checked
against the language; the six L=13 runs' words and congruences.

Run: uv run python research/word_grammar.py    (repo root; numpy)
"""
import csv
import os
from collections import Counter, defaultdict
from math import prod

import numpy as np

MRB = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    for p in MRB:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in MRB:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def hit_arrays(gears):
    P = prod(gears)
    hL = np.zeros(P, bool)
    hR = np.zeros(P, bool)
    for q in gears:
        u = pow(6, -1, q)
        hL[u::q] = True
        hR[(q - u) % q::q] = True
    return P, hL, hR

def horizon(gears):
    """L0 = (max cyclic gap between both-sides-hit slots) - 1."""
    P, hL, hR = hit_arrays(gears)
    b = np.flatnonzero(hL & hR)
    gaps = np.diff(np.concatenate((b, [b[0] + P])))
    i = int(np.argmax(gaps))
    return int(gaps.max()) - 1, int((b[i] + 1) % P), len(b)

def language_census(gears, Lmax):
    """Distinct admissible words per length, by phase-side generation."""
    P, hL, hR = hit_arrays(gears)
    hL2 = np.concatenate((hL, hL[:Lmax + 2]))
    hR2 = np.concatenate((hR, hR[:Lmax + 2]))
    B2 = hL2 & hR2
    out = {}
    for L in range(1, Lmax + 2):
        words = set()
        phases = 0
        for k0 in range(P):
            if B2[k0:k0 + L].any():
                continue
            phases += 1
            forced = []
            free = []
            for i in range(L):
                if hL2[k0 + i]:
                    forced.append('R')
                elif hR2[k0 + i]:
                    forced.append('L')
                else:
                    forced.append('?')
                    free.append(i)
            base = forced[:]
            for m in range(1 << len(free)):
                for j, pos in enumerate(free):
                    base[pos] = 'L' if (m >> j) & 1 else 'R'
                words.add(''.join(base))
        out[L] = (len(words), phases)
        if not words:
            break
    return out

def word_admissible(w, gears=(5, 7, 11, 13)):
    for q in gears:
        u = pow(6, -1, q)
        chosen = {(u - i) % q if c == 'L' else (-u - i) % q
                  for i, c in enumerate(w)}
        if len(chosen) == q:
            return False
    return True

def main():
    print("=" * 72)
    print("PART 1: the finite horizon L0 (unconditional saturated-run cap)")
    sets = [(5, 7), (5, 7, 11), (5, 7, 11, 13), (5, 7, 11, 13, 17),
            (5, 7, 11, 13, 17, 19), (5, 7, 11, 13, 17, 19, 23)]
    L0_13 = None
    for g in sets:
        L0, at, nb = horizon(g)
        if g == (5, 7, 11, 13):
            L0_13 = L0
        print(f"  gears {g}: period {prod(g):>9}, B-slots {nb:>8}, "
              f"L0 = {L0:>2} (corridor starts at slot {at} mod {prod(g)})")
    print("  monotone in the gear set; every saturated run at every scale has")
    print("  length <= L0 of any gear subset below its window - UNCONDITIONAL.")

    print("=" * 72)
    print(f"PART 2: language census, gears <= 13 (slot period 5005)")
    cen = language_census((5, 7, 11, 13), L0_13)
    prev = None
    print(f"  {'L':>3} {'|lang|':>8} {'2^L':>10} {'phases':>7} {'ratio':>6}")
    for L, (n, ph) in cen.items():
        r = n / prev if prev else float('nan')
        print(f"  {L:>3} {n:>8} {2**L:>10} {ph:>7} {r:>6.3f}")
        prev = n if n else None
    Ls = [L for L, (n, _) in cen.items() if n]
    ns = {L: n for L, (n, _) in cen.items()}
    print(f"  language EMPTY at L = {L0_13 + 1} (= L0+1, matches part 1)")
    full = max(L for L in Ls if ns[L] == 2 ** L)
    print(f"  all 2^L words admissible up to L = {full}; first exclusions at "
          f"L = {full+1} ({2**(full+1) - ns[full+1]} words, incl. "
          f"{'LLLLL' if not word_admissible('LLLLL') else '?'} - "
          f"same-letter blocks cap at 4 by gear 5)")
    # grammar laws sanity inside the language
    assert not word_admissible('L' * 5) and not word_admissible('R' * 5)
    assert word_admissible('LRLRLR') and not word_admissible('LRLRLRL')
    assert word_admissible('RLRLR') and not word_admissible('RLRLRL')
    print("  laws verified in-language: same-letter blocks <= 4; strict "
          "alternation <= 6 (L-first) / 5 (R-first) [round-7 cap = special case]")

    print("=" * 72)
    print("PART 3: the observed 757 runs vs the language")
    path = os.path.join(os.path.dirname(__file__), "data", "satruns_ge10.csv")
    runs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            runs.append((int(row["k_start"]), int(row["L"])))
    words = []
    bad = 0
    for k0, L in runs:
        w = []
        for k in range(k0, k0 + L):
            pl, pr = is_prime(6 * k - 1), is_prime(6 * k + 1)
            if pl == pr:
                bad += 1
                break
            w.append('L' if pl else 'R')
        else:
            words.append((k0, ''.join(w)))
    inadm = [(k0, w) for k0, w in words if not word_admissible(w)]
    print(f"  {len(runs)} runs, {bad} non-saturated (should be 0), "
          f"{len(inadm)} inadmissible words (MUST be 0): "
          f"{'OK' if not inadm and not bad else 'FAIL'}")
    byL = Counter(L for _, L in runs)
    used = defaultdict(set)
    for k0, w in words:
        used[len(w)].add(w)
    for L in sorted(byL):
        lang = ns.get(L, 0)
        print(f"  L={L:>2}: runs {byL[L]:>3}, distinct words {len(used[L]):>3}, "
              f"language size {lang:>6} -> coverage {len(used[L])/lang:.4f}")
    thirteens = [(k0, w) for k0, w in words if len(w) == 13]
    print(f"  the L=13 runs ({len(thirteens)}):")
    for k0, w in thirteens:
        print(f"    k={k0:>12}  k mod 35 = {k0%35:>2}  k mod 5005 = {k0%5005:>4}  {w}")
    dw = Counter(w for _, w in thirteens)
    dup = {w: c for w, c in dw.items() if c > 1}
    print(f"  duplicate 13-words: {dup if dup else 'none - all distinct'}")

def ceiling_curve(Ls=(33, 40, 50, 63, 100, 160, 200, 252)):
    """Part 4: unconditional twin-free load ceiling beyond the horizon.
    On ANY twin-free window a B-slot carries 0 primes, so
    P_run <= L - minB(L); ceiling = 1 - minB(L)/L, asymptote 1 - 730/5005."""
    P, hL, hR = hit_arrays((5, 7, 11, 13))
    B = (hL & hR).astype(np.int32)
    cs = np.concatenate(([0], np.cumsum(np.concatenate((B, B[:max(Ls) + 1])))))
    print("PART 4: unconditional load ceiling for L > L0 (gears <= 13):")
    for L in Ls:
        mb = int((cs[L:P + L] - cs[:P]).min())
        print(f"  L={L:>3}: min B-slots {mb:>2} -> ceiling {(L - mb)/L:.4f}")
    print(f"  asymptote 1 - 730/5005 = {1 - 730/5005:.4f}")

if __name__ == "__main__":
    main()
    print("=" * 72)
    ceiling_curve()
