"""Second pair: base gear 5 with base gear 7, same anchor (column h), step up.

Both gears see column h = (6h - 1, 6h + 1) itself (E = -1 mod 35 whenever 5 and 7 are in the
base).  Gear 5's teeth on the h-line: h = 1 (left), 4 (right) mod 5.  Gear 7's teeth: h = 6
(left), 1 (right) mod 7.  So the pair's pattern is fixed, the same at every machine with 5 and 7
in the base: 15 open classes of h mod 35.  Checked here at several machines by listing the high
gears the pair leaves and the classes mod 35 they occupy.  Then the cross pair band 11 with
high 13 at q = 499, both anchored on E.

usage: uv run python research/stack/r8/pair_same_anchor.py 499 997 1999
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
    out = [__doc__.strip(), ""]
    fixed = sorted(h for h in range(35) if h % 5 not in (1, 4) and h % 7 not in (1, 6))
    out.append(f"open classes of h mod 35 for the pair (5, 7) on column h: {fixed} ({len(fixed)} of 35)")
    for q in map(int, sys.argv[1:]):
        primes, base, E = spiral(q)
        assert 5 in base and 7 in base
        r = int(q ** 0.5); high = [p for p in primes if p > r]
        left = []; seen = set()
        for h in high:
            L = E + 6 * h
            if not (q < L and L + 2 <= q * q): continue
            open5 = L % 5 and (L + 2) % 5; open7 = L % 7 and (L + 2) % 7
            assert bool(open5) == (h % 5 not in (1, 4)) and bool(open7) == (h % 7 not in (1, 6))
            if open5 and open7: left.append(h); seen.add(h % 35)
        out.append(f"q = {q}, E = {E} (E mod 35 = {E % 35}): pair (5, 7) leaves {len(left)} high gears, classes mod 35 seen {sorted(seen)}; inside the fixed set: {set(seen) <= set(fixed)}; the fixed classes never seen are the multiples of 5 or 7: {sorted(set(fixed) - seen)}")
    # cross pair 11 (band) with 13 (high) at 499, both anchored on E
    q = 499; primes, base, E = spiral(q); r = int(q ** 0.5); high = [p for p in primes if p > r]
    cls = {}
    for g in (11, 13):
        inv = pow(6, -1, g); cls[g] = ((-E * inv) % g, (-(E + 2) * inv) % g)
    out.append(""); out.append(f"cross pair at q = {q}: gear 11 teeth h = {cls[11]} mod 11, gear 13 teeth h = {cls[13]} mod 13, both from E = {E}")
    surv = []
    for h in high:
        L = E + 6 * h
        if not (q < L and L + 2 <= q * q): continue
        s11 = "11 left" if h % 11 == cls[11][0] else ("11 right" if h % 11 == cls[11][1] else "misses")
        s13 = "13 left" if h % 13 == cls[13][0] else ("13 right" if h % 13 == cls[13][1] else "misses")
        if h == 13: s13 = "is gear 13 (never strikes its own landing unless 13 | E or E + 2)"
        if s11 == "misses" and s13.startswith("m"): surv.append(h)
        out.append(f"   h = {h:>4}: mod 11 = {h % 11:>2} ({s11}); mod 13 = {h % 13:>2} ({s13})")
    out.append(f"cross pair survivors: {len(surv)}; classes mod 143 they occupy: {sorted(set(h % 143 for h in surv))}")
    Path("research/stack/r8/results_pair_same_anchor.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:7])); print("...", len(surv), "cross survivors")

if __name__ == "__main__":
    main()
