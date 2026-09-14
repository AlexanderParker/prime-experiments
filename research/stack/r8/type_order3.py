"""One field type alone: (g, left member, order 3) on the landing line, step up L = E + 6h.

Class h = a (mod g), h = a + g t, cofactor c = c_0 + 6 t.  Order 3: c = p_1 p_2, primes >= g,
p_1 the smaller.  Then t sits in p_1's class 6 t = -c_0 (mod p_1), t = t_1 + p_1 s, and the
quotient runs on its own column line d_0 + 6 s, prime at the kill.  The table shows, for one gear
g at one machine, every order-3 kill of the type with its p_1, class t_1, s, and quotient line.

usage: uv run python research/stack/r8/type_order3.py 499 11
"""
import sys
from sympy import primerange, factorint, isprime
from pathlib import Path

def main():
    q, g = int(sys.argv[1]), int(sys.argv[2])
    primes = list(primerange(2, q + 1)); base = []; P = 1
    for p in primes:
        if P * p <= q // 2: P *= p; base.append(p)
        else: break
    def alt(xs): return sum(x if i % 2 == 0 else -x for i, x in enumerate(xs))
    E = -1 + 2 * P * alt([p for p in primes if p not in base][::-1])
    r = int(q ** 0.5); high = [p for p in primes if p > r]
    inv = pow(6, -1, g); a = (-E * inv) % g; c0 = (E + 6 * a) // g
    out = [__doc__.strip(), "", f"q = {q}, E = {E}, gear {g}: class h = {a} (mod {g}), cofactor line c = {c0} + 6t", ""]
    out.append("type (g, left, order 3) alone:")
    for h in high:
        if h % g != a: continue
        L = E + 6 * h
        if not (q < L and L + 2 <= q * q): continue
        t = (h - a) // g; c = c0 + 6 * t; f = factorint(c)
        if sum(f.values()) != 2 or min(f) < g: continue
        p1 = min(f); p2 = c // p1
        t1 = (-c0 * pow(6, -1, p1)) % p1; s = (t - t1) // p1; d0 = (c0 + 6 * t1) // p1
        assert t % p1 == t1 and d0 + 6 * s == p2 and isprime(p2)
        out.append(f"   h = {h} (t = {t}): c = {c} = {p1} * {p2}; p1's class t = {t1} (mod {p1}), s = {s}; quotient line d = {d0} + 6s, d = {p2} prime")
    out.append("")
    out.append("for comparison, the other t of the class (not this type):")
    for h in high:
        if h % g != a: continue
        L = E + 6 * h
        if not (q < L and L + 2 <= q * q): continue
        t = (h - a) // g; c = c0 + 6 * t; f = factorint(c)
        if sum(f.values()) == 2 and min(f) >= g: continue
        fs = "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(f.items()))
        what = "order 2 (type of the last round)" if len(f) == 1 and sum(f.values()) == 1 else (f"gear {min(f)}'s kill" if min(f) < g else f"order {1 + sum(f.values())}")
        out.append(f"   h = {h} (t = {t}): c = {fs}; {what}")
    Path(f"research/stack/r8/results_type_order3_{q}_{g}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
