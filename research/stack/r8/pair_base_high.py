"""Fourth pair: base gear 5 (anchored on column h, fixed teeth) with gear 13 (anchored on E),
across machines; and the walk's own gear h against its own landing.

Gear 5's teeth on the h-line: h = 1 (left), 4 (right) mod 5 (5 in the base, q >= 61).  Gear 13's
teeth: h = a, b mod 13 with 6a = -E, 6b = -(E + 2) mod 13, placed by E.  13 is a high gear for
q < 169 and a band gear after; the pair's rule is the same.  Per machine: 13's teeth, the high
gears both miss, and their classes mod 65.  Then h against itself: h strikes its own landing
E + 6h iff h | E (left) or h | E + 2 (right).

usage: uv run python research/stack/r8/pair_base_high.py 61 167 499 997
"""
import sys
from sympy import primerange
from pathlib import Path

def spiral(q):
    primes = list(primerange(2, q + 1)); base = []; P = 1
    for p in primes:
        if P * p <= q // 2: P *= p; base.append(p)
        else: break
    def alt(xs): return sum(x if i % 2 == 0 else -x for i, x in enumerate(xs))
    return primes, base, -1 + 2 * P * alt([p for p in primes if p not in base][::-1])

def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); extra = list(map(int, sys.argv[3:]))
    out = [__doc__.strip(), ""]
    selfhits = []
    for q in [p for p in primerange(lo, hi + 1)] + extra:
        primes, base, E = spiral(q)
        if 5 not in base: continue
        r = int(q ** 0.5); high = [p for p in primes if p > r]
        inv = pow(6, -1, 13); a, b = (-E * inv) % 13, (-(E + 2) * inv) % 13
        role = "high" if 13 > r else "band"
        surv = []; own = []
        for h in high:
            L = E + 6 * h
            if not (q < L and L + 2 <= q * q): continue
            miss5 = h % 5 not in (1, 4); miss13 = h % 13 not in (a, b) or h == 13
            assert miss5 == bool(L % 5 and (L + 2) % 5)
            if h != 13: assert miss13 == bool(L % 13 and (L + 2) % 13)
            if miss5 and miss13: surv.append(h)
            if L % h == 0 or (L + 2) % h == 0:
                own.append(h); assert E % h == 0 or (E + 2) % h == 0
        selfhits += [(q, h) for h in own]
        out.append(f"q = {q}: E = {E}, E mod 13 = {E % 13}, gear 13 ({role}) teeth h = {a} (left), {b} (right) mod 13; "
                   f"pair (5, 13) leaves {len(surv)} of the high gears in reach: {surv}; classes mod 65: {sorted(set(h % 65 for h in surv))}; "
                   f"gears striking their own landing: {own}")
    out.append("")
    out.append(f"own-landing strikes (q, h), each with h | E or h | E + 2: {selfhits}")
    Path("research/stack/r8/results_pair_base_high.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
