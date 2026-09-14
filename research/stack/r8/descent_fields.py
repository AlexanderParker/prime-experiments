"""The t-line of the primorial descent as the fields construction: P_s is the machine's carried
base, rows are the gears above the base, columns are t (the landing 2 t P_s - 1).  Row g is
painted L at t where g divides the left member and R where it divides the right member (the
multiples field of g on the t-line).  Columns with no paint in any row up to sqrt(2 t P_s + 1)
are twins.  Text grid, one character per column: L, R, or . ; column headers every 10.

usage: uv run python research/stack/r8/descent_fields.py 210 60
"""
import sys
from sympy import primerange, isprime
from pathlib import Path

def main():
    Ps, T = int(sys.argv[1]), int(sys.argv[2])
    base = [p for p in primerange(2, 100) if Ps % p == 0]
    top = 2 * T * Ps + 1
    gears = [g for g in primerange(base[-1] + 1, int(top ** 0.5) + 1)]
    out = [__doc__.strip(), "", f"P_s = {Ps}, base {base}; columns t = 1..{T}; rows = gears {gears[0]}..{gears[-1]} that paint at least once", ""]
    hdr = "".join(str(t // 10) if t % 10 == 0 else " " for t in range(1, T + 1))
    hdr2 = "".join(str(t % 10) for t in range(1, T + 1))
    out.append(f"{'':>6} {hdr}"); out.append(f"{'t':>6} {hdr2}")
    painted = set()
    for g in gears:
        inv = pow(2 * Ps, -1, g); a, b = inv % g, (-inv) % g
        row = ""
        for t in range(1, T + 1):
            if t % g == a: row += "L"; painted.add(t)
            elif t % g == b: row += "R"; painted.add(t)
            else: row += "."
        if "L" in row or "R" in row: out.append(f"{g:>6} {row}")
    twins = "".join("T" if t not in painted else " " for t in range(1, T + 1))
    out.append(f"{'twin':>6} {twins}")
    for t in range(1, T + 1):
        assert (t not in painted) == (isprime(2 * t * Ps - 1) and isprime(2 * t * Ps + 1)), t
    out.append("")
    out.append("teeth (left, right) per row: " + ", ".join(f"{g}: ({pow(2 * Ps, -1, g) % g}, {(-pow(2 * Ps, -1, g)) % g})" for g in gears[:20]))
    Path(f"research/stack/r8/results_descent_fields_{Ps}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
