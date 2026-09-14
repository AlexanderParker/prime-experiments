"""First pair of types on the landing line: base gear 5 and one band gear g, step up.

Two anchors.  Base gear 5 sees column h itself (E = -1 mod 5): its teeth on the h-line sit at
h = 6^{-1} = 1 (left member) and h = -1 = 4 (right member) mod 5, for every machine with 5 in
the base.  Band gear g sees the landing: its teeth sit at h = a (left) and h = b (right) mod g,
a, b fixed by E mod g.  The pair's joint pattern is the alignment of the two anchors, listed
here as one row per high gear h: where h stands against gear 5's teeth (column h), where the
landing stands against gear g's teeth, and what remains when both are missed.

usage: uv run python research/stack/r8/pair_base_band.py 499 11
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
    assert 5 in base
    def alt(xs): return sum(x if i % 2 == 0 else -x for i, x in enumerate(xs))
    E = -1 + 2 * P * alt([p for p in primes if p not in base][::-1])
    r = int(q ** 0.5); high = [p for p in primes if p > r]
    inv = pow(6, -1, g); a = (-E * inv) % g; b = (-(E + 2) * inv) % g
    out = [__doc__.strip(), "", f"q = {q}, E = {E}, base {base}; gear 5 teeth on the h-line: h = 1 (left), 4 (right) mod 5; gear {g} teeth: h = {a} (left), {b} (right) mod {g}", ""]
    surv = []
    for h in high:
        L = E + 6 * h
        if not (q < L and L + 2 <= q * q): continue
        five = "5 left" if h % 5 == 1 else ("5 right" if h % 5 == 4 else "misses 5")
        gg = f"{g} left" if h % g == a else (f"{g} right" if h % g == b else f"misses {g}")
        assert (five.startswith("5 l")) == (L % 5 == 0) and (five.startswith("5 r")) == ((L + 2) % 5 == 0)
        assert (gg.startswith(f"{g} l")) == (L % g == 0) and (gg.startswith(f"{g} r")) == ((L + 2) % g == 0)
        rest = ""
        if five.startswith("m") and gg.startswith("m"):
            surv.append(h)
            f1, f2 = factorint(L), factorint(L + 2)
            rest = "  pair survivor; " + ("twin" if len(f1) == 1 == len(f2) and max(f1.values()) == 1 == max(f2.values()) else
                    f"then taken by gear {min([p for p in list(f1) + list(f2) if p not in (2, 3) and (len(f1) > 1 or f1.get(p, 0) > 1 or L != p) and (len(f2) > 1 or f2.get(p, 0) > 1 or L + 2 != p)] or [0])}")
        out.append(f"   h = {h:>4}: h mod 5 = {h % 5} ({five}); h mod {g} = {h % g} ({gg}){rest}")
    out.append("")
    out.append(f"pair survivors: {len(surv)} of the high gears in reach; the classes they occupy mod {5 * g}: {sorted(set(h % (5 * g) for h in surv))}")
    Path(f"research/stack/r8/results_pair_base_band_{q}_{g}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
