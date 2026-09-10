"""u45_census.py -- the EXACT full-period gap census c(d) of the engine {5..y} with no period.

c(d) = the number of gaps of size >= d per period = #{columns x : x is an opening and
x+1 .. x+d-1 are all blocked}.  W32 (research/proof/law_register.md, top_machine_2.md 3.7) needs
exactly this object, and for y >= 37 the period has never been scanned (m37 is 1.24e12 columns,
m53 is 1.7e19).

The count is a covering count.  Gear g strikes column k iff k = +-u_g (mod g), u_g = 6^{-1} mod g.
With r = x mod g, offset o of the window [0, d-1] is struck by g iff o = +-u_g - r (mod g), so the
struck set of g depends only on r, and by CRT the vector (x mod g)_g runs over the product of the
Z_g exactly once per period.  Hence

    c(d) = # { (r_g)_g : no gear strikes offset 0, and every offset 1..d-1 is struck by some gear }

and that is computed by a dynamic programme over the gears whose state is the SET of offsets not
yet struck.  The state set collapses hard (the small gears strike arithmetic progressions), so the
programme is cheap where a period is impossible.  Exact integer arithmetic; no sampling.

Usage: uv run python research/anchor235/r72/u45_census.py <y> [dmax]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

def primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = 0
    sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def gears_upto(y):
    return [p for p in primes_upto(int(y)) if p >= 5]


def strike_masks(g, d, teeth=None):
    """{mask: multiplicity} over the residues r = x mod g that leave offset 0 unstruck, where the
    mask is the set of offsets in [1, d-1] that g strikes.  Collapsing equal masks is what makes
    a machine with many gears cheap: a gear with g >> d misses the window entirely at most of its
    phases, and those phases share the empty mask.

    `teeth` is the pair of struck residue classes: by default the column coordinate's
    +-6^{-1} mod g; the manifold wheels of top_machine_2.md use {0, -2} on the raw line instead.
    """
    if teeth is None:
        u = pow(6, -1, g)
        t1, t2 = u % g, (-u) % g
    else:
        t1, t2 = teeth[0] % g, teeth[1] % g
    out = {}
    for r in range(g):
        if r == t1 or r == t2:
            continue                       # offset 0 would be struck: x is not an opening
        m = 0
        o1 = (t1 - r) % g
        while o1 < d:
            if o1:
                m |= 1 << o1
            o1 += g
        o2 = (t2 - r) % g
        while o2 < d:
            if o2:
                m |= 1 << o2
            o2 += g
        out[m] = out.get(m, 0) + 1
    return out


def c_of_gears(gears, d, cap=40_000_000, teeth=None):
    """c(d) for an arbitrary gear set: exact Python int.  `teeth` = None is the column
    coordinate (+-6^{-1} mod g); teeth=(0, -2) is the raw-line pair wheel."""
    if d <= 1:
        v = 1
        for g in gears:
            v *= g - 2
        return v, 1
    full = (1 << d) - 2
    dp = {full: 1}
    peak = 1
    for g in gears:
        ms = strike_masks(g, d, teeth)
        nd = {}
        for st, c in dp.items():
            for m, mu in ms.items():
                k = st & ~m
                nd[k] = nd.get(k, 0) + c * mu
        dp = nd
        peak = max(peak, len(dp))
        if len(dp) > cap:
            raise MemoryError(f"census DP at {gears}, d={d}: {len(dp):,} states")
    return dp.get(0, 0), peak


def c_of(y, d, cap=40_000_000):
    """c(d) for the machine {5..y}: exact, as a Python int.  Returns (value, peak states)."""
    gears = gears_upto(y)
    if d <= 1:
        v = 1
        for g in gears:
            v *= g - 2
        return v, 1
    full = (1 << d) - 2                    # bits 1 .. d-1
    dp = {full: 1}
    peak = 1
    for g in gears:
        ms = strike_masks(g, d)
        nd = {}
        for st, c in dp.items():
            for m, mu in ms.items():
                k = st & ~m
                nd[k] = nd.get(k, 0) + c * mu
        dp = nd
        peak = max(peak, len(dp))
        if len(dp) > cap:
            raise MemoryError(f"census DP at y={y}, d={d}: {len(dp):,} states")
    return dp.get(0, 0), peak


def census(y, dmax, cap=40_000_000):
    """c(d) for d = 1 .. dmax (stops early if c(d) = 0)."""
    out = {}
    peak = 0
    for d in range(1, dmax + 1):
        v, p = c_of(y, d, cap)
        peak = max(peak, p)
        out[d] = v
        if v == 0:
            break
    return out, peak


def main():
    y = int(sys.argv[1])
    dmax = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    t0 = time.time()
    c, peak = census(y, dmax)
    P = 1
    for g in gears_upto(y):
        P *= g
    spec = {d: c[d] - c.get(d + 1, 0) for d in c if c[d]}
    rep = {"y": y, "gears": gears_upto(y), "P": P, "dmax_reached": max(c),
           "peak_states": peak, "secs": round(time.time() - t0, 1),
           "c": {str(k): str(v) for k, v in c.items()},
           "m": {str(k): str(v) for k, v in spec.items()}}
    print(json.dumps({k: rep[k] for k in ("y", "P", "dmax_reached", "peak_states", "secs")},
                     indent=1))
    for d in sorted(c):
        print(f"  d={d:3d}  c(d)={c[d]:,}   m(d)={spec.get(d, 0):,}")
    with open(os.path.join(OUT, f"census_m{y}.json"), "w") as f:
        json.dump(rep, f)


if __name__ == "__main__":
    main()
