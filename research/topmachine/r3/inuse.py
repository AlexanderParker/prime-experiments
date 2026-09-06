"""Q6: the in-use machine (gears (q, sqrt(N)]) - the general mex form and the counting bound."""

import sys
from math import isqrt

import numpy as np

OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, isqrt(n) + 1):
        if s[p]:
            s[p * p::p] = False
    return np.flatnonzero(s)


def open_pair_range(gears, N):
    """mask[n] True iff the pair (n, n+2) is open, for n in [0, N]."""
    mask = np.ones(N + 3, dtype=bool)
    for g in gears:
        mask[0::g] = False
        idx = (g - 2) % g
        mask[idx::g] = False
    return mask[:N + 1]


def walk_from(mask):
    """L(x) = min{j >= 0 : mask[x+j]} on a prefix; last positions truncated."""
    N = len(mask)
    L = np.zeros(N, dtype=np.int64)
    cur = 0
    for x in range(N - 1, -1, -1):
        cur = 0 if mask[x] else cur + 1
        L[x] = cur
    return L


def mex_general(gears, x, B):
    """L(x) = mex of the union of the two arithmetic progressions per gear."""
    cov = np.zeros(B + 2, dtype=bool)
    terms = 0
    for g in gears:
        for t0 in ((-x) % g, (-x - 2) % g):
            j = t0
            while j <= B:
                cov[j] = True
                terms += 1
                j += g
    cov[B + 1] = True
    return int(cov.argmin()), terms


def run(qs, N):
    Z = isqrt(N)
    allp = primes_upto(Z)
    say(f"**N = {N:,}, Z = floor(sqrt(N)) = {Z}**")
    say()
    say("| q | gears m | density | record F | mean walk | median | 99th pct | "
        "mex form on 2,000 x | terms used | 2m | H_S = sum_{g <= F} 1/g | "
        "counting bound 2m/(1-2H_S) |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    rng = np.random.default_rng(11)
    for q in qs:
        gears = [int(p) for p in allp if p > q]
        m = len(gears)
        mask = open_pair_range(gears, N)
        L = walk_from(mask)
        # ignore the tail where truncation could bite
        Luse = L[: N - 5000]
        F = int(Luse.max())
        mean = float(Luse.mean())
        med = float(np.median(Luse))
        p99 = float(np.percentile(Luse, 99))
        dens = float(mask[:N - 5000].mean())
        xs = rng.integers(0, N - 5000, size=2000)
        ok = True
        terms = 0
        for x in xs:
            v, t = mex_general(gears, int(x), F)
            terms += t
            if v != int(L[int(x)]):
                ok = False
        H = sum(1.0 / g for g in gears if g <= F)
        bound = (2 * m / (1 - 2 * H)) if H < 0.5 else float("inf")
        say(f"| {q} | {m} | {dens:.6f} | {F} | {mean:.2f} | {med:.0f} | {p99:.0f} | "
            f"{'0 mismatches' if ok else 'FAIL'} | {terms/len(xs):.0f} | {2*m} | "
            f"{H:.3f} | {'vacuous' if H >= 0.5 else f'{bound:.0f}'} |")
    say()


def threshold(qs, N):
    """The length L* at which sum_{q < g <= L} 1/g reaches 1/2."""
    Z = isqrt(N)
    allp = primes_upto(max(Z, 100000))
    say("| q | L* where sum_{q<g<=L} 1/g = 1/2 | measured record F |")
    say("|---|---|---|")
    return allp


if __name__ == "__main__":
    for N in (10 ** 6, 10 ** 7):
        run([5, 7, 11, 13, 17, 19], N)
    with open("research/topmachine/r3/results/inuse.out", "w") as f:
        f.write("\n".join(OUT))
