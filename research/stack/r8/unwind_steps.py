"""Unwinding the machine one gear at a time: how each field advances into the new range.

Step g -> g' (consecutive primes >= 5): the new range is the numbers in (g^2, g'^2), i.e. the
columns between the square columns. Origin of the step: the square g^2 (column (g^2-1)/6).
Every composite member n of the new range is n = h * m with h its least factor, h <= g, and
m >= h a survivor of the gears below h; m lies in an EARLIER range (m < g'^2 / h), so each
kill in the new range is an element of an earlier range carried up by the gear h. For each
step the script prints: the columns of the new range with each killed member written as
h x m, m's own range (the pair of squares m sits between), and the offset of the kill from the
origin in columns; the openings (twins); then the tally: kills per gear h, kills per source
range of m, and the gear fields' offsets from the origin (first member of each gear field in
the new range and its distance from g^2).
Usage: uv run python unwind_steps.py gmax
"""
import sys
from collections import Counter
from sympy import factorint, isprime, nextprime, primerange


def sq_range(m):
    """the consecutive-prime squares p^2 <= m < p'^2 that contain m (p = 2 allowed)."""
    p = 2
    while nextprime(p) ** 2 <= m: p = nextprime(p)
    return f"[{p}^2,{nextprime(p)}^2)"


def main():
    gmax = int(sys.argv[1])
    for g in primerange(5, gmax + 1):
        gp = nextprime(g); a = (g * g - 1) // 6; top = gp * gp
        cols = [k for k in range(a + 1, top) if 6 * k + 1 < top]
        print(f"\nstep {g} -> {gp}: new range ({g}^2, {gp}^2) = ({g*g}, {top}), origin column {a}, {len(cols)} columns")
        per_gear = Counter(); per_src = Counter(); first_of = {}; twins = []
        for k in cols:
            parts = []
            tw = True
            for n in (6 * k - 1, 6 * k + 1):
                if isprime(n): parts.append(f"{n}=P"); continue
                tw = False
                f = factorint(n); h = min(f); m = n // h
                per_gear[h] += 1; per_src[sq_range(m)] += 1
                first_of.setdefault(h, (k - a, n))
                parts.append(f"{n}={h}x{m}{'(m prime)' if isprime(m) else '(m=' + 'x'.join(f'{p}^{e}' if e > 1 else str(p) for p, e in factorint(m).items()) + ')'} m in {sq_range(m)}")
            if tw: twins.append(k - a)
            if g <= 13:
                print(f"  +{k-a:3d}: " + " | ".join(parts) + ("   TWIN" if tw else ""))
        print(f"  twins at offsets {twins} from the origin")
        print(f"  kills per gear h: {dict(sorted(per_gear.items()))}")
        print(f"  kills per source range of the cofactor m: {dict(sorted(per_src.items(), key=lambda x: int(x[0][1:].split('^')[0])))}")
        print(f"  first strike of each gear field in the new range (offset from the origin, member): {dict(sorted(first_of.items()))}")


if __name__ == "__main__":
    main()
