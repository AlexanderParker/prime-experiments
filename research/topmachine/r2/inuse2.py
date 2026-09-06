"""Section 3, second half: the IN-USE machine (gears (q, Z], Z = sqrt(N)) at larger N,
to place its record exactly as a fraction of N, of Z and of the origin clump.

usage: uv run python research/topmachine/r2/inuse2.py results/inuse2.json
"""

import json
import sys
import time

import numpy as np

CHUNK = 1 << 25


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if s[i]:
            s[i * i :: i] = False
    return np.flatnonzero(s)


def scan(gears, N):
    """Longest pair-free run on [0, N) and its position, plus the count of open pairs."""
    best = 0
    at = -1
    last_open = -1
    total = 0
    base = 0
    while base < N:
        n = min(CHUNK, N - base)
        a = np.ones(n, dtype=bool)
        for g in gears:
            a[(-base) % g :: g] = False
            a[(-2 - base) % g :: g] = False
        idx = np.flatnonzero(a)
        total += len(idx)
        if len(idx):
            pos = idx.astype(np.int64) + base
            prev = np.empty(len(pos), dtype=np.int64)
            prev[0] = last_open
            prev[1:] = pos[:-1]
            gaps = pos - prev
            j = int(np.argmax(gaps))
            if int(gaps[j]) - 1 > best:
                best = int(gaps[j]) - 1
                at = int(prev[j]) + 1
            last_open = int(pos[-1])
        base += n
    return best, at, total


def main():
    allp = primes_upto(200000)
    res = []
    for N in (10**7, 10**8, 10**9):
        Z = int((N + 2) ** 0.5)
        for q in (5, 7, 11, 13, 17, 19):
            gears = [int(p) for p in allp if q < p <= Z]
            t = time.time()
            F, at, tot = scan(gears, N)
            qp = gears[0]
            r = {
                "q": q,
                "N": N,
                "Z": Z,
                "gear_count": len(gears),
                "q_prime": qp,
                "F_range": F,
                "at": at,
                "at_over_N": at / N,
                "at_over_Z": at / Z,
                "at_over_clump": at / (qp - 3) if qp > 3 else None,
                "F_over_Z": F / Z,
                "inside_gear_zone": at + F <= Z,
                "density": tot / N,
                "two_m": 2 * len(gears),
                "secs": round(time.time() - t, 1),
            }
            res.append(r)
            print(
                "q=%2d N=%.0e Z=%6d gears=%5d  F=%6d at=%8d  at/N=%.2e at/Z=%.3f  F/Z=%.3f "
                " inside gear zone=%s  2m=%d  %.0fs"
                % (q, N, Z, len(gears), F, at, at / N, at / Z, F / Z, r["inside_gear_zone"],
                   2 * len(gears), r["secs"]),
                flush=True,
            )
    with open(sys.argv[1], "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
