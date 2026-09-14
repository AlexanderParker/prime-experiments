"""Triple: base gears 5 and 7 (fixed teeth on column h) with one E-anchored gear g, step up.

Survivors of the triple are the high gears h with h not 1, 4 mod 5; not 1, 6 mod 7; not a, b
mod g (6a = -E, 6b = -(E + 2) mod g).  One line per machine: g's teeth, the survivors, their
classes mod 35g, and the next single type that takes each survivor (smallest gear dividing a
member, or twin).

usage: uv run python research/stack/r8/triple_5_7_g.py 11 499 997 1999
"""
import sys
from sympy import primerange, factorint
from pathlib import Path

def spiral(q):
    primes = list(primerange(2, q + 1)); base = []; P = 1
    for p in primes:
        if P * p <= q // 2: P *= p; base.append(p)
        else: break
    def alt(xs): return sum(x if i % 2 == 0 else -x for i, x in enumerate(xs))
    return primes, base, -1 + 2 * P * alt([p for p in primes if p not in base][::-1])

def main():
    g = int(sys.argv[1]); out = [__doc__.strip(), ""]
    for q in map(int, sys.argv[2:]):
        primes, base, E = spiral(q); assert 5 in base and 7 in base
        r = int(q ** 0.5); high = [p for p in primes if p > r]
        inv = pow(6, -1, g); a, b = (-E * inv) % g, (-(E + 2) * inv) % g
        out.append(f"q = {q}, E = {E}: gear {g} teeth h = {a} (left), {b} (right) mod {g}; base teeth fixed (5: 1, 4; 7: 6, 1)")
        surv = []
        for h in high:
            L = E + 6 * h
            if not (q < L and L + 2 <= q * q): continue
            if h % 5 in (1, 4) or h % 7 in (1, 6) or (h != g and h % g in (a, b)): continue
            assert L % 5 and (L + 2) % 5 and L % 7 and (L + 2) % 7 and (h == g or (L % g and (L + 2) % g))
            f1, f2 = factorint(L), factorint(L + 2)
            nxt = "twin" if (len(f1) == 1 and f1[L] == 1 and len(f2) == 1 and f2[L + 2] == 1) else \
                  f"next: gear {min([p for p in list(f1) + list(f2) if p * p <= L + 2 and p >= 5] or [max(list(f1) + list(f2))])}"
            surv.append((h, nxt))
        out.append(f"   triple leaves {len(surv)} high gears; classes mod {35 * g}: {sorted(set(h % (35 * g) for h, _ in surv))}")
        for h, nxt in surv: out.append(f"      h = {h:>4} (mod 5: {h % 5}, mod 7: {h % 7}, mod {g}: {h % g}): {nxt}")
        out.append("")
    Path(f"research/stack/r8/results_triple_5_7_{g}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
