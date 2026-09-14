"""The right-member type and the down-step type of one gear, alone.

Gear g at machine q, landing E of the primorial spiral.
  right member, step up:  class h = b (mod g), 6b = -(E + 2); h = b + g t; E + 6h + 2 = g (c_0 + 6t)
  left member, step down: class h = a' (mod g), 6a' = E;       h = a' + g t; E - 6h = g (c_0 - 6t)
  right member, step down: class h = b' (mod g), 6b' = E + 2;  h = b' + g t; E - 6h + 2 = g (c_0 - 6t)
Each row of the class: the cofactor, its factorisation, and the type that takes it.

usage: uv run python research/stack/r8/type_right_down.py 499 11
"""
import sys
from sympy import primerange, factorint
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
    inv = pow(6, -1, g)
    out = [__doc__.strip(), "", f"q = {q}, E = {E}, gear {g}"]
    for name, d, off in (("right member, step up", 1, 2), ("left member, step down", -1, 0), ("right member, step down", -1, 2)):
        cls = ((-(E + off) if d > 0 else (E + off)) * inv) % g
        c0 = (E + off + 6 * d * cls) // g
        assert (E + off + 6 * d * cls) % g == 0
        out.append(""); out.append(f"{name}: class h = {cls} (mod {g}); h = {cls} + {g} t; member = {g} * c, c = {c0} {'+' if d > 0 else '-'} 6t")
        for h in high:
            if h % g != cls: continue
            L = E + 6 * d * h
            if not (q < L and L + 2 <= q * q): continue
            t = (h - cls) // g; m = L + off; c = m // g
            assert c == c0 + 6 * d * t
            f = factorint(c); fs = "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(f.items()))
            j = 1 + sum(f.values())
            what = f"type order {j}" if min(f) >= g else f"gear {min(f)}'s kill"
            out.append(f"   h = {h} (t = {t}): c = {fs}; {what}")
    Path(f"research/stack/r8/results_type_right_down_{q}_{g}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
