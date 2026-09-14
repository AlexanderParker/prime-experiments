"""Each field type in the band as its own object on the landing line.

Machine q, landing E, final step up: L = E + 6h, h a high gear (sqrt q < h <= q).  Band gear g
strikes the left member when h is in the class a_g = -E 6^{-1} (mod g), the right member when h
is in b_g = -(E + 2) 6^{-1} (mod g).  Inside a class h = a + g t and the struck member is
  g * c,   c = c_0 + 6 t   (c_0 = (E + 6 a) / g for the left member, (E + 2 + 6 a) / g for the right)
so the cofactors run along an arithmetic progression of step 6: a column line one level down.
The field type higher:g of order j is the set of t at which c is a product of j - 1 primes, all
at least g.  A cofactor with a prime factor below g is not g's kill (the smaller gear's field).
The table lists, per band gear, per member, the class, the cofactor line, and every high gear h
in the class with its cofactor, factorisation, order and field owner.

usage: uv run python research/stack/r8/band_field_traces.py 499
"""
import sys
from sympy import primerange, factorint
from pathlib import Path

def main():
    q = int(sys.argv[1]); d = 1
    primes = list(primerange(2, q + 1)); base = []; P = 1
    for p in primes:
        if P * p <= q // 2: P *= p; base.append(p)
        else: break
    def alt(xs): return sum(g if i % 2 == 0 else -g for i, g in enumerate(xs))
    nb = [p for p in primes if p not in base][::-1]
    E = -1 + 2 * P * alt(nb)
    r = int(q ** 0.5)
    band = [g for g in nb if 5 <= g <= r][::-1]
    high = [g for g in primes if g > r]
    out = [__doc__.strip(), "", f"q = {q}, window ({q}, {q*q}], base {base} (P = {P}), band {band}, E = {E}, step up L = E + 6h", ""]
    for g in band:
        inv = pow(6, -1, g)
        for name, off in (("left", 0), ("right", 2)):
            a = (-(E + off) * inv) % g
            c0 = (E + off + 6 * a) // g
            assert (E + off + 6 * a) % g == 0
            out.append(f"band gear {g}, {name} member: class h = {a} (mod {g}); h = {a} + {g} t; member = {g} * c, c = {c0} + 6 t")
            for h in high:
                if h % g != a: continue
                L = E + 6 * h
                if not (q < L and L + 2 <= q * q): continue
                t = (h - a) // g; m = L + off; c = m // g
                f = factorint(c); j = 1 + sum(f.values())
                small = min(f)
                owner = f"higher:{g} order {j}" if small >= g else f"lower gear {small}'s field (higher:{small})"
                fs = "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(f.items()))
                out.append(f"   h = {h:>4} (t = {t:>2}): member {m} = {g} * {c}, c = {fs}; {owner}")
            out.append("")
    Path(f"research/stack/r8/results_band_field_traces_{q}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
