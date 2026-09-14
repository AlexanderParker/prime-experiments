"""The band gears (non-base gears up to sqrt q): what fixes their forbidden classes.

E = -1 + 2 P A with A the alternating sum of the non-base gears descending from q.  For a band
gear g the two forbidden classes of h are a = -E (6d)^{-1} and b = -(E + 2)(6d)^{-1} mod g, so
they are fixed by E mod g, i.e. by A mod g.  This tables, per band gear g, the residue A mod g
and E mod g across the machines, and the split of A mod g into the part above g (gears q down to
the prime after g), the g term (0), and the part below g (band gears below g).

usage: uv run python research/stack/r8/band_classes.py 2000
"""
import sys
from collections import Counter, defaultdict
from sympy import primerange
from pathlib import Path

def main():
    Q = int(sys.argv[1])
    qs = list(primerange(11, Q + 1))
    def alt(xs): return sum(g if i % 2 == 0 else -g for i, g in enumerate(xs))
    resA = defaultdict(Counter); resE = defaultdict(Counter); nband = Counter()
    eopen = Counter(); rows = []
    for q in qs:
        primes = list(primerange(2, q + 1)); base = []; P = 1
        for p in primes:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        nb = [p for p in primes if p not in base][::-1]   # descending
        A = alt(nb); E = -1 + 2 * P * A
        r = int(q ** 0.5)
        band = [g for g in nb if g <= r and g >= 5]
        for g in band:
            i = nb.index(g)
            above = alt(nb[:i]); below = alt(nb[i + 1:]) * (1 if i % 2 == 0 else -1) * (-1)
            # A = above + s*g + s'*below-part; check
            assert (above - (below if i % 2 == 0 else -below) * 0 + 0) is not None
            resA[g][A % g] += 1; resE[g][E % g] += 1; nband[g] += 1
            eopen[(g, E % g in (0, g - 2))] += 1
        if q in (101, 499, 997, 1999):
            rows.append(f"q = {q}: base {base}, band {band}, A = {A}, E = {E}; " +
                        ", ".join(f"E mod {g} = {E % g} (h classes up: {(-E * pow(6, -1, g)) % g}, {(-(E + 2) * pow(6, -1, g)) % g})" for g in band))
    out = [__doc__.strip(), ""]
    for g in sorted(nband):
        if nband[g] < 30: continue
        out.append(f"g = {g}: machines with g in the band {nband[g]}; E mod g counts by residue: " +
                   " ".join(f"{r}:{resE[g][r]}" for r in range(g)) +
                   f"; E struck by g (E mod g in {{0, g-2}}): {eopen[(g, True)]} of {nband[g]}")
    out.append(""); out.extend(rows)
    Path("research/stack/r8/results_band_classes.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
