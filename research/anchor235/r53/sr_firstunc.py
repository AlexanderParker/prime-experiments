"""sr_firstunc.py -- the first uncoupled size, and what it is arithmetically.

"v is uncoupled in {5..y}" is a statement about the GEARS alone: v is missed by all three
residue classes {0, +d_q, -d_q} mod q for every q <= y.  By H6 (half_column.md), for v < y^2/3
that is the same as: v is y-rough (only 2, 3 and primes > y divide it) and, for even v, the
column v/2 is a twin column above y.

So "the machine has an uncoupled size below y^2/3" -- the object the sum-rule branch would have
to force -- is exactly "there is a rough twin column in the window".  This script measures the
first uncoupled size v_1(y) at every prime rung, against y^2/3 (the window) and against
2 d_0(y) (twice the first twin column above y, the object node 1e.i already measured).

Writes results/sr_firstunc.txt
"""
import os, sys
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
XMAX = 2_000_000
YMAX = 20000


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = False
    return s


def main():
    isp = primes_upto(6 * XMAX + 2)
    gears = [int(p) for p in np.flatnonzero(primes_upto(YMAX)) if p >= 5]
    coupled = np.zeros(XMAX + 1, dtype=bool)
    lines = []
    W = lines.append
    W("  y | first uncoupled v_1(y) | v_1 vs y^2/3 | v_1/2 = c | c factorisation | (6c-1,6c+1) twin? | 2 d_0(y) | v_1 = 2 d_0?")
    ptr = 2
    ptre = 2
    rows = 0
    agree = 0
    even = 0
    twin_ok = 0
    inside = 0
    even_inside = 0
    ratios = []
    wratios = []
    checkpoints = set([5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                       101, 199, 401, 601, 997, 1999, 4999, 9973, 19997])
    for y in gears:
        u = pow(6, -1, y)
        d = (2 * u) % y
        for t in (0, d, (-d) % y):
            coupled[t % y:: y] = True
        while ptr <= XMAX and coupled[ptr]:
            ptr += 1
        v1 = ptr
        while ptre <= XMAX and coupled[ptre]:
            ptre += 2
        ve = ptre
        rows += 1
        w = y * y / 3.0
        if v1 < w:
            inside += 1
        if v1 % 2 == 0:
            even += 1
        c = ve // 2
        tw = bool(isp[6 * c - 1] and isp[6 * c + 1]) and (6 * c - 1 > y)
        if tw:
            twin_ok += 1
        if ve < w:
            even_inside += 1
        ratios.append((ve / y, y, ve))
        wratios.append((ve / w, y, ve))
        # d_0 : first twin column above y
        c0 = y // 6 + 1
        while not (isp[6 * c0 - 1] and isp[6 * c0 + 1] and 6 * c0 - 1 > y):
            c0 += 1
        if v1 == 2 * c0:
            agree += 1
        if y in checkpoints:
            fac = []
            if c:
                n = c
                for p in [2, 3, 5, 7, 11, 13]:
                    e = 0
                    while n % p == 0:
                        n //= p
                        e += 1
                    if e:
                        fac.append(f"{p}^{e}" if e > 1 else f"{p}")
                if n > 1:
                    fac.append(str(n))
            W(f"  {y} | {v1} | {'<' if v1 < w else '>='} {w:.0f} | ve={ve} c={c} | {'.'.join(fac)} | "
              f"{tw} | {2*c0} | {v1 == 2*c0}")
    W("")
    W(f"  rungs tested: {rows} primes 5..{gears[-1]}")
    W(f"  first uncoupled size is EVEN: {even} of {rows}")
    W(f"  the first uncoupled EVEN size: its half-column is a twin column above y: {twin_ok} of {rows}")
    W(f"  the first uncoupled EVEN size lies below y^2/3: {even_inside} of {rows}")
    W(f"  first uncoupled size lies below y^2/3 (inside the window): {inside} of {rows}")
    ratios.sort(); wratios.sort()
    W(f"  v_e(y)/y : min {ratios[0][0]:.3f} at y={ratios[0][1]} (v_e={ratios[0][2]}), max {ratios[-1][0]:.3f} at y={ratios[-1][1]} (v_e={ratios[-1][2]}), median {ratios[len(ratios)//2][0]:.3f}")
    W(f"  v_e(y)/(y^2/3) : max {wratios[-1][0]:.5f} at y={wratios[-1][1]} (v_e={wratios[-1][2]})")
    W(f"  first uncoupled size equals 2 d_0(y) (twice the FIRST twin column above y): {agree} of {rows}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "sr_firstunc.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
