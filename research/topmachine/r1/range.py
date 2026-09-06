"""The top machine on a RANGE shorter than its wheel (its real use).

Machine: gears = primes in (q, Z].  Pair n = (n, n+2) struck iff n = 0 or -2 (mod g).
Two regimes:
  (a) Z = floor(sqrt(N+2))  - the machine as used on [1, N]
  (b) a fixed small gear set whose wheel is far above N - to see non-periodicity
Measured: density against the CRT product, block densities, the longest pair-free run
against the wheel's record, the structure at the origin, and the neighbourhood of the
multiples of the smallest gears' product.
"""

import json
import sys
from math import prod

import numpy as np


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if s[i]:
            s[i * i :: i] = False
    return np.flatnonzero(s)


def build(gears, N):
    a = np.ones(N, dtype=bool)
    for g in gears:
        a[0::g] = False
        a[(g - 2) % g :: g] = False
    return a


def longest_struck_run(a):
    idx = np.flatnonzero(a)
    if len(idx) == 0:
        return len(a), 0
    best = int(idx[0])
    at = 0
    if len(idx) > 1:
        d = np.diff(idx) - 1
        j = int(np.argmax(d))
        if d[j] > best:
            best = int(d[j])
            at = int(idx[j]) + 1
    tail = len(a) - 1 - int(idx[-1])
    if tail > best:
        best, at = tail, int(idx[-1]) + 1
    return best, at


def run_hist(a):
    idx = np.flatnonzero(a)
    d = np.diff(idx)
    h = {}
    for L, c in zip(*np.unique(d, return_counts=True)):
        h[int(L)] = int(c)
    return h


def analyse(q, N, Zmode="sqrt", nfixed=8, allp=None):
    Zsq = int((N + 2) ** 0.5)
    if Zmode == "sqrt":
        gears = [int(p) for p in allp if q < p <= Zsq]
    else:
        gears = [int(p) for p in allp if p > q][:nfixed]
    a = build(gears, N)
    cnt = int(a.sum())
    dens = cnt / N
    prodv = 1.0
    for g in gears:
        prodv *= 1 - 2 / g
    blocks = 20
    bs = N // blocks
    bd = [float(a[i * bs : (i + 1) * bs].mean()) for i in range(blocks)]
    F, at = longest_struck_run(a)
    # origin: first 4*q pairs
    origin = [int(x) for x in np.flatnonzero(a[: 6 * q + 12])]
    # neighbourhood of multiples of the product of the three smallest gears
    sm = gears[:3]
    Wsm = prod(sm)
    hits = []
    if Wsm < N:
        L = 4 * gears[0]
        for t in range(1, min(200, N // Wsm)):
            c = t * Wsm
            if c - L >= 0 and c + L < N:
                hits.append(float(a[c - L : c + L].mean()))
    out = {
        "q": q,
        "N": N,
        "mode": Zmode,
        "Z": gears[-1],
        "gear_count": len(gears),
        "open_pairs": cnt,
        "density": dens,
        "crt_product": prodv,
        "ratio_density_over_product": dens / prodv if prodv else None,
        "block_densities": bd,
        "block_first_over_last": bd[0] / bd[-1],
        "longest_pairfree_run": F,
        "at": at,
        "two_m": 2 * len(gears),
        "origin_open_pairs": origin[:40],
        "small_wheel": Wsm,
        "mean_density_near_small_wheel_multiples": (
            float(np.mean(hits)) if hits else None
        ),
        "n_small_wheel_multiples": len(hits),
        "gap_hist_head": {k: v for k, v in sorted(run_hist(a).items())[:8]},
        "gap4_count": run_hist(a).get(4, 0),
    }
    return out


def main():
    allp = primes_upto(40000)
    res = []
    for N in (10**5, 10**6, 10**7):
        for q in (5, 7, 11, 13, 17, 19):
            r = analyse(q, N, "sqrt", allp=allp)
            res.append(r)
            print(
                "sqrt q'>%d N=%.0e Z=%d m=%d dens=%.6f prod=%.6f ratio=%.4f F=%d(2m=%d) "
                "blk1/blk20=%.3f gap4=%d"
                % (q, N, r["Z"], r["gear_count"], r["density"], r["crt_product"],
                   r["ratio_density_over_product"], r["longest_pairfree_run"],
                   r["two_m"], r["block_first_over_last"], r["gap4_count"])
            )
    for N in (10**5, 10**6, 10**7):
        for q in (5, 11, 17):
            r = analyse(q, N, "fixed", nfixed=8, allp=allp)
            res.append(r)
            print(
                "fixed q'>%d N=%.0e Z=%d m=%d dens=%.6f prod=%.6f ratio=%.4f F=%d "
                "blk1/blk20=%.3f nearwheel=%.4f"
                % (q, N, r["Z"], r["gear_count"], r["density"], r["crt_product"],
                   r["ratio_density_over_product"], r["longest_pairfree_run"],
                   r["block_first_over_last"],
                   r["mean_density_near_small_wheel_multiples"] or -1)
            )
    with open(sys.argv[1], "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
