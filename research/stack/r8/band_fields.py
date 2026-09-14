"""For each band gear g at a machine q: which of g's fields exist in the window (q, q^2] and which
of them strike the final step's landings E +- 6h.

Fields of row g, keyed by the cofactor of g:
  square      g^2 (and g^2 times more) -- g^2 in the window iff g^2 > q; for a band gear g <= sqrt q it is not
  powers      g^j alone, in the window iff g^j <= q^2
  higher:g,j  g times a product of j-1 primes all above g (smallest factor g), order j; exists iff g^j <= q^2
  lower       g times something with a smaller gear: charged to the smaller gear's field, not g's
The table gives, per band gear: the largest order possible, which pure powers sit in the window,
and the orders of the landings' struck members whose smallest gear is g (both directions, every
high gear in reach), plus whether g^2 ever divides a struck member.

usage: uv run python research/stack/r8/band_fields.py 101 499 997 1999
"""
import sys
from collections import Counter
from sympy import primerange, factorint
from pathlib import Path

def main():
    out = [__doc__.strip(), ""]
    def alt(xs): return sum(g if i % 2 == 0 else -g for i, g in enumerate(xs))
    for q in map(int, sys.argv[1:]):
        primes = list(primerange(2, q + 1)); base = []; P = 1
        for p in primes:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        nb = [p for p in primes if p not in base][::-1]
        E = -1 + 2 * P * alt(nb)
        r = int(q ** 0.5)
        band = [g for g in nb if 5 <= g <= r][::-1]
        high = [g for g in primes if g > r]
        gears = [g for g in primes if g >= 5]
        out.append(f"q = {q}: window ({q}, {q*q}], base {base}, band {band}, high gears {len(high)}, E = {E}")
        kills = {g: Counter() for g in band}; sq = Counter(); mem = {g: [] for g in band}
        for d in (1, -1):
            for h in high:
                L = E + 6 * d * h
                if not (q < L and L + 2 <= q * q): continue
                for m in (L, L + 2):
                    f = factorint(m)
                    small = min((p for p in f if p >= 5), default=None)
                    if small in kills:
                        j = sum(f.values()); kills[small][j] += 1
                        if f[small] >= 2: sq[small] += 1
                        if len(mem[small]) < 3: mem[small].append(f"{m} = " + "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(f.items())))
        for g in band:
            jmax = 0
            while g ** (jmax + 1) <= q * q: jmax += 1
            pw = [j for j in range(2, jmax + 1) if q < g ** j <= q * q]
            out.append(f"   band gear {g}: g^2 = {g*g} below the window; pure powers in the window: {['g^%d = %d' % (j, g**j) for j in pw]}; "
                       f"largest order with smallest gear g: {jmax}; struck landing members with smallest gear {g} by order: {dict(sorted(kills[g].items()))}; "
                       f"g^2 divides a struck member: {sq[g]}; examples: {mem[g]}")
        out.append("")
    Path("research/stack/r8/results_band_fields.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
