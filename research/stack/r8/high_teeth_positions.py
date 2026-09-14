"""A high gear's teeth as positions on the h-line, step up.  Gear g above sqrt q has teeth at
a + g t (left member) and b + g t (right member), 6a = -E, 6b = -(E + 2) mod g.  The h-line in
reach is the gears in (sqrt q, q] with the landing inside the window.  For every high gear:
its tooth positions inside (sqrt q, q], which of them are gears, and which of those were still
standing when g's turn came (the machine's order).  Gears above q/2 hold at most two positions
per tooth (kernel tooth_at_most_two).

usage: uv run python research/stack/r8/high_teeth_positions.py 499
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
    r = int(q ** 0.5); gears = [p for p in primes if p >= 5]; high = [p for p in primes if p > r]
    live = set(h for h in high if q < E + 6 * h and E + 6 * h + 2 <= q * q)
    out = [__doc__.strip(), "", f"q = {q}, E = {E}, high gears from {high[0]}, in reach {len(live)}", ""]
    for g in gears:
        inv = pow(6, -1, g); a, b = (-E * inv) % g, (-(E + 2) * inv) % g
        if g <= r:
            live -= set(h for h in live if h % g in (a, b) and h != g); continue
        rows = []
        for name, c in (("left", a), ("right", b)):
            pos = [x for x in range(c, q + 1, g) if x > r]
            marks = []
            for x in pos:
                if x == g: marks.append(f"{x} (g itself)")
                elif not isprime(x): marks.append(f"{x}")
                elif x in live: marks.append(f"{x} GEAR standing -> taken")
                else: marks.append(f"{x} gear, already taken")
            rows.append(f"{name} tooth {c}: " + (", ".join(marks) if marks else "no position in reach"))
        taken = [h for h in live if h % g in (a, b) and h != g]
        live -= set(taken)
        out.append(f"gear {g:>3}{' (above q/2)' if 2 * g > q else '':<12}: " + " | ".join(rows) + (f"   => takes {taken}, {len(live)} remain" if taken else ""))
    out.append(""); out.append(f"passing: {sorted(live)}")
    Path(f"research/stack/r8/results_high_teeth_positions_{q}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
