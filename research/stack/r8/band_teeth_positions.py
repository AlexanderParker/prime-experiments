"""The band gears' teeth as positions on the h-line, step up, one machine.  Gear g in the band
(above the base, at most sqrt q) has teeth a + g t (left) and b + g t (right).  For each band
gear, in the machine's order: every tooth position in (sqrt q, q], marked not a gear / gear
taken earlier (by which gear) / gear standing -> taken now.  The base gears come first, their
teeth fixed (5: 1, 4; 7: 6, 1), listed the same way.  Then the whole line stands written as
positions from E, gear by gear.

usage: uv run python research/stack/r8/band_teeth_positions.py 499
"""
import sys
from sympy import primerange, isprime
from pathlib import Path

def main():
    q = int(sys.argv[1])
    primes = list(primerange(2, q + 1)); base = []; P = 1
    for p in primes:
        if P * p <= q // 2: P *= p; base.append(p)
        else: break
    def alt(xs): return sum(x if i % 2 == 0 else -x for i, x in enumerate(xs))
    E = -1 + 2 * P * alt([p for p in primes if p not in base][::-1])
    r = int(q ** 0.5); gears = [p for p in primes if p >= 5]
    high = [p for p in primes if p > r]
    live = {h: None for h in high if q < E + 6 * h and E + 6 * h + 2 <= q * q}   # gear -> taker
    out = [__doc__.strip(), "", f"q = {q}, E = {E}, base {base}, band {[g for g in gears if g <= r and g not in base]}, high gears from {high[0]}, in reach {len(live)}", ""]
    for g in gears:
        if g > r: break
        inv = pow(6, -1, g); a, b = (-E * inv) % g, (-(E + 2) * inv) % g
        role = "base" if g in base else "band"
        out.append(f"gear {g} ({role}), teeth h = {a} (left), {b} (right) mod {g}:")
        for name, c in (("left", a), ("right", b)):
            marks = []
            for x in range(c, q + 1, g):
                if x <= r: continue
                if not isprime(x): marks.append(f"{x}")
                elif x not in live: marks.append(f"{x} out of reach")
                elif live[x] is not None: marks.append(f"{x} gear, taken by {live[x]}")
                else: live[x] = g; marks.append(f"{x} GEAR -> taken")
            out.append(f"   {name} tooth {c} + {g}t: " + ", ".join(marks))
        out.append(f"   standing after gear {g}: {[h for h, t in live.items() if t is None]}")
        out.append("")
    standing = [h for h, t in live.items() if t is None]
    out.append(f"after the sub-machine: {len(standing)} standing: {standing}")
    Path(f"research/stack/r8/results_band_teeth_positions_{q}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
