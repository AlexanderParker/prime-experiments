"""Round 7 lateral: alternation words of saturated runs vs the mirror laws.

A saturated slot (exactly one prime member) has its letter DETERMINED by the
machine: in the open interior, prime = unhit, so letter(k) = the side NOT hit
by any gear. Saturated runs are the one-sided stretches of the machine's hit
pattern; their words are machine words, and the positional mirror law
(k -> -k reverses order and swaps L/R) makes exact predictions:

  P1 (parity theorem): an odd-length word can never equal its own
     reverse-complement (middle letter would equal its own complement), so
     odd-length saturated runs are NEVER self-mirror. Even lengths can be.
  P2 (mirror statistics): the word distribution at each length should be
     symmetric under reverse-complement (the mirror is a bijection of the
     machine pattern), but NOT under reverse alone or complement alone -
     no machine symmetry produces those.
  P3 (forced skeleton): letters at slots whose composite side is hit by a
     gear <= 13 are CRT-forced; predicted fraction 1 - prod(1-2/q) = 0.703.

Tests below: run census by length, palindrome counts, total-variation
asymmetry under revcomp/reverse/complement, letter marginals, forced-letter
fraction, duplicate words and their position congruences, recurrence of the
L*=13 landmark word, and max strict-alternation stretch.

Run: uv run python research/alternation_words.py   (repo root; numpy)
"""
from collections import Counter, defaultdict

import numpy as np

from derivative_scan import sieve

LANDMARK13 = "RLLRRLLLLRLRL"          # slots 2452..2464, primes 14713..14783

def revcomp(w):
    return ''.join('L' if c == 'R' else 'R' for c in reversed(w))

def comp(w):
    return ''.join('L' if c == 'R' else 'R' for c in w)

def tvd(words, f):
    """Total variation distance between dist(w) and dist(f(w))."""
    c = Counter(words)
    n = len(words)
    keys = set(c) | {f(w) for w in c}
    return 0.5 * sum(abs(c.get(w, 0) - c.get(f(w), 0)) for w in keys) / n

def analyze(y, Lmin=8):
    print(f"--- y = {y} ---")
    K, gears, oml, omr, gvl, gvr = sieve(y)
    s0 = (y + 1) // 6 + 1
    lo = oml == 0
    ro = omr == 0
    sat = lo ^ ro
    sat[:s0] = False
    # small-gear hit masks (forced-skeleton test)
    smL = np.zeros(K + 1, bool)
    smR = np.zeros(K + 1, bool)
    for g in (5, 7, 11, 13):
        u = pow(6, -1, g)
        smL[u::g] = True
        smR[(g - u) % g::g] = True
    d = np.diff(np.concatenate(([0], sat.astype(np.int8), [0])))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    lens = en - st
    runs = [(int(a), ''.join('L' if lo[k] else 'R' for k in range(a, b)))
            for a, b in zip(st, en) if b - a >= Lmin]
    bylen = Counter(len(w) for _, w in runs)
    print(f"  maximal saturated runs (len >= {Lmin}): {len(runs)}; by length: "
          f"{dict(sorted(bylen.items()))}")
    # letter marginal + forced fraction
    nl = sum(w.count('L') for _, w in runs)
    tot = sum(len(w) for _, w in runs)
    forced = 0
    for a, w in runs:
        for i, c in enumerate(w):
            k = a + i
            forced += smR[k] if c == 'L' else smL[k]   # composite side hit by small gear
    print(f"  letter marginal: L {nl/tot:.4f} (mirror predicts 0.5); "
          f"forced-letter fraction (gears <= 13): {forced/tot:.4f} "
          f"(CRT prediction 0.703)")
    # per-length tests
    print(f"  {'L':>3} {'runs':>6} {'distinct':>8} {'dups':>5} {'palin':>6} "
          f"{'TV(revcomp)':>12} {'TV(reverse)':>12} {'TV(comp)':>9}")
    for L in sorted(bylen):
        if bylen[L] < 20 and L > Lmin + 4:
            continue
        pop = [w for _, w in runs if len(w) == L]
        c = Counter(pop)
        dup = sum(v - 1 for v in c.values() if v > 1)
        pal = sum(1 for w in pop if w == revcomp(w))
        print(f"  {L:>3} {len(pop):>6} {len(c):>8} {dup:>5} {pal:>6} "
              f"{tvd(pop, revcomp):>12.4f} "
              f"{tvd(pop, lambda w: w[::-1]):>12.4f} {tvd(pop, comp):>9.4f}")
        if L % 2 == 1:
            assert pal == 0, "parity theorem violated"
    # duplicate congruence structure (largest population length)
    Lpop = min(bylen)
    pos = defaultdict(list)
    for a, w in runs:
        if len(w) == Lpop:
            pos[w].append(a)
    diffs = [b - a for v in pos.values() if len(v) > 1
             for a, b in zip(v, v[1:])]
    if diffs:
        da = np.array(diffs)
        base = np.diff(np.array(sorted(a for a, w in runs if len(w) == Lpop)))
        print(f"  duplicate-word position differences (L={Lpop}, {len(da)} pairs): "
              f"divisible by 5: {np.mean(da%5==0):.2f}, 7: {np.mean(da%7==0):.2f}, "
              f"11: {np.mean(da%11==0):.2f}, 13: {np.mean(da%13==0):.2f}, "
              f"35: {np.mean(da%35==0):.2f}  "
              f"(baseline all-pairs: 5: {np.mean(base%5==0):.2f}, "
              f"7: {np.mean(base%7==0):.2f})")
    # landmark recurrence
    hits = [(a, len(w)) for a, w in runs if LANDMARK13 in w]
    print(f"  landmark word {LANDMARK13} occurrences (len>= {Lmin} runs): "
          f"{hits if hits else 'only at slot 2452' if y*y//6 < 2452 else hits}")
    # strict alternation: longest stretch of saturated slots with alternating letters
    letter = np.where(lo, 1, np.where(ro, 2, 0))
    letter[~sat] = 0
    ok = (letter[:-1] != 0) & (letter[1:] != 0) & (letter[:-1] != letter[1:])
    dd = np.diff(np.concatenate(([0], ok.astype(np.int8), [0])))
    s2, e2 = np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)
    maxalt = int((e2 - s2).max()) + 1 if len(s2) else 1
    i = int(s2[np.argmax(e2 - s2)])
    print(f"  longest STRICT alternation (LRLR...): {maxalt} slots at slot {i} "
          f"(depth {i/K:.4f})")
    return runs

if __name__ == "__main__":
    print("=" * 72)
    print("ALTERNATION WORDS of saturated runs vs the mirror laws")
    for y in (3163, 10007):
        analyze(y)
