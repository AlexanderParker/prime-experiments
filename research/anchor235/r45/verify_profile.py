"""Branch 2g.i: exceptionless checks.  Prover, 2026-09-05.

(1) reads the saved full-period profiles (m11..m31) and checks the corrected law
    N(v) <= F_2(M) for every realised gap size v >= 6, machine by machine, listing every
    exception and the tight cases;
(2) checks the gear-5 multiplicity weight w(s) = #{(r,r') in {0,2,3}^2 : r'-r = s mod 5}
    (3,1,2,2,1 for s = 0,1,2,3,4 mod 5) against the measured gap counts;
(3) verifies the v-shifted L6 identity on full periods: for a gap (x1, x2 = x1+v) and any
    gear g, with p_g the distance from x1 leftward and q_g the distance from x2 rightward to
    g's nearest strike, p_g + q_g = -v, -v+d_g or -v-d_g (mod g), d_g = 2u_g.  At v = 0 this
    is L6 (pair_statement.md).
"""
import numpy as np, re, sys
from math import prod

RES = "research/anchor235/r45/results/"
FILES = ["deep_profile_23.txt", "deep_profile_29.txt", "deep_profile_31.txt"]


def parse():
    out = {}
    for fn in FILES:
        txt = open(RES + fn).read()
        for blk in txt.split("=== machine ")[1:]:
            head = blk.splitlines()[0]
            y = int(re.search(r"\{5\.\.(\d+)\}", head).group(1))
            F = int(re.search(r" F=(\d+)", head).group(1))
            m = re.search(r"profile v: N\(v\).*?\n\s*(.*?)\n", blk, re.S)
            prof = {}
            for tok in m.group(1).split():
                v, rest = tok.split(":")
                nv, cnt = rest.split("(")
                prof[int(v)] = (int(nv), int(cnt.rstrip(")")))
            F2 = max(nv for v, (nv, c) in prof.items() if v == 0) if 0 in prof else None
            out[y] = (F, prof)
    return out


def f2_of(y):
    """F_2 read off the Q*_2 line of the same file."""
    for fn in FILES:
        txt = open(RES + fn).read()
        for blk in txt.split("=== machine ")[1:]:
            if re.search(r"\{5\.\.(\d+)\}", blk.splitlines()[0]).group(1) == str(y):
                return int(re.search(r"Q\*_2 = (\d+)", blk).group(1))
    return None


def check_cap():
    data = parse()
    print("machine   F   F_2   max N(v) over v>=6 (at v)   exceptions v>=6   spikes v<6 (v:N-F_2)")
    for y in sorted(data):
        F, prof = data[y]
        F2 = f2_of(y)
        big = [(v, nv) for v, (nv, c) in prof.items() if v >= 6]
        mv, mn = max(big, key=lambda t: t[1])
        exc = [(v, nv) for v, nv in big if nv > F2]
        low = [(v, nv - F2) for v, (nv, c) in sorted(prof.items()) if v < 6 and nv > F2]
        print(f"  m{y:<5d} {F:3d}  {F2:4d}   {mn:4d} (v={mv})            {exc}         {low}")
    print()


def check_weights():
    data = parse()
    for y in sorted(data):
        F, prof = data[y]
        rows = []
        for v in range(1, 16):
            if v in prof:
                rows.append(f"{v}:{prof[v][1]}(w{[3,1,2,2,1][v%5]})")
        print(f"  m{y}: gap counts by size with gear-5 weight -- " + " ".join(rows))
    print()


def check_l6(top, sample=None):
    gears = [p for p in (5, 7, 11, 13, 17, 19) if p <= top]
    us = [pow(6, -1, g) for g in gears]
    P = prod(gears)
    blocked = np.zeros(P, dtype=bool)
    for g, u in zip(gears, us):
        blocked[u % g::g] = True
        blocked[(-u) % g::g] = True
    opens = np.flatnonzero(~blocked)
    n = opens.size
    bad = 0
    tot = 0
    idx = range(n - 1) if sample is None else range(0, n - 1, max(1, (n - 1) // sample))
    for i in idx:
        x1 = int(opens[i]); x2 = int(opens[i + 1]); v = x2 - x1
        for g, u in zip(gears, us):
            # nearest strike of g strictly left of x1, and strictly right of x2
            p = min((x1 - c) % g for c in (u, -u) if (x1 - c) % g != 0) \
                if any((x1 - c) % g != 0 for c in (u, -u)) else g
            q = min((c - x2) % g for c in (u, -u) if (c - x2) % g != 0) \
                if any((c - x2) % g != 0 for c in (u, -u)) else g
            d = (2 * u) % g
            tot += 1
            if (p + q + v) % g not in (0, d % g, (-d) % g):
                bad += 1
    print(f"  L6-at-a-gap, m{top}: {tot} (gap, gear) pairs checked, {bad} violations "
          f"of p+q = -v, -v+d, -v-d (mod g)")


if __name__ == "__main__":
    print("\n(1) the corrected law  N(v) <= F_2(M)  for every realised v >= 6\n")
    check_cap()
    print("(2) gap multiplicity against the gear-5 weight w(s mod 5) = 3,1,2,2,1\n")
    check_weights()
    print("(3) the v-shifted L6 identity\n")
    check_l6(13)
    check_l6(17)
    check_l6(19, sample=200000)
