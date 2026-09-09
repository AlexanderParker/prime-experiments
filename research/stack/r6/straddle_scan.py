"""The straddling run across every prime square to PMAX: x_s, L_s, the ratio, and the top-run share.

Vocabulary (lengthen_never_precede.md).  Column k is the pair (6k - 1, 6k + 1).  At the ladder step
q -> p = nextprime(q) the machine M + p = {5..p} acts on the prefix [1, W(p')] and the SQUARE COLUMN
is W = (p^2 - 1)/6 (its upper member is p^2).  The STRADDLING RUN is the maximal blocked run of
{5..p} containing W.  By the reduction (R) the openings of {5..p} in the prefix are the twin columns
with lower member > p, so with

    t0 = the last twin lower member below p^2,     k0 = (t0 + 1)/6
    t1 = the first twin lower member above p^2,    k1 = (t1 + 1)/6

the straddling run is (x_s, L_s) = (k0 + 1, k1 - k0 - 1).  d_0(p) is the first opening of {5..p},
i.e. the column of the first twin with lower member > p (the first-twin distance); the frontier's
hypothesis reaches the straddling run only when L_s >= d_0(p).  The TOP-RUN SHARE at the rung
q = prevprime(p) is tau = L_top / (W - q//6) with L_top = W - k0 the part of the straddling run
inside the prefix of {5..q} (valve_existence.md table 2's convention: the window of {5..q} is the
columns k with 6k - 1 > q up to W).

Method: for each prime p a small sieve of the columns around W by the primes below SIEVE_B (column k
is struck by g iff k = +-6^{-1} mod g), then gmpy2.is_prime on the survivors outward from W in both
directions.  The window widens until a twin is found on each side.  Exact: the sieve only prefilters.

Usage: uv run python research/stack/r6/straddle_scan.py [PMAX] [SHARD] [NSHARD]   (one core, < 300 MB)
Output: results/straddle_<PMAX>[_<SHARD>].npz with one row per prime p >= 7.
"""
import os
import sys
import time

import gmpy2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
SIEVE_B = 1000          # columns are prefiltered by the primes 5 <= g < SIEVE_B
WIDTH0 = 512            # initial half-window in columns; doubled until a twin is found


def small_primes(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.flatnonzero(s).astype(np.int64)


def twin_columns(limit):
    """columns k with (6k-1, 6k+1) both prime and 6k+1 <= limit."""
    s = np.ones(limit + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    k = np.arange(1, (limit - 1) // 6 + 1, dtype=np.int64)
    return k[s[6 * k - 1] & s[6 * k + 1]]


def sieve_residues(bound):
    """(g, r1, r2) for every prime 5 <= g < bound: the two column classes g strikes."""
    out = []
    for g in small_primes(bound).tolist():
        if g < 5:
            continue
        u = pow(6, -1, g)
        out.append((g, u % g, (-u) % g))
    return out


def straddle(W, res, isp):
    """(k0, k1): last twin column < W and first twin column > W.  Exact."""
    width = WIDTH0
    while True:
        lo = max(1, W - width)
        hi = W + width
        n = hi - lo + 1
        blocked = bytearray(n)
        one = b"\x01"
        for g, r1, r2 in res:
            st = (r1 - lo) % g
            blocked[st::g] = one * ((n - st + g - 1) // g)
            st = (r2 - lo) % g
            blocked[st::g] = one * ((n - st + g - 1) // g)
            # a gear never strikes the member that IS the gear (only proper multiples);
            # unmarking is safe in any case, the survivors are tested exactly
            kk = (g + 1) // 6 if (g + 1) % 6 == 0 else ((g - 1) // 6 if (g - 1) % 6 == 0 else -1)
            if lo <= kk <= hi:
                blocked[kk - lo] = 0
        k0 = k1 = None
        k = W - 1
        while k >= lo:
            if not blocked[k - lo]:
                m = 6 * k - 1
                if isp(m) and isp(m + 2):
                    k0 = k
                    break
            k -= 1
        k = W + 1
        while k <= hi:
            if not blocked[k - lo]:
                m = 6 * k - 1
                if isp(m) and isp(m + 2):
                    k1 = k
                    break
            k += 1
        if k0 is not None and k1 is not None:
            return k0, k1
        if lo == 1 and k0 is None:
            k0 = 0                      # no twin column below W at all (only p = 5, 7 territory)
            if k1 is not None:
                return k0, k1
        width *= 2
        if width > 1 << 22:
            raise RuntimeError(f"no straddling run found at W={W}")


def main(pmax, shard, nshard):
    t0 = time.time()
    isp = gmpy2.is_prime
    res = sieve_residues(SIEVE_B)
    primes = small_primes(pmax)
    primes = primes[primes >= 5]
    # d_0(p): the column of the first twin with lower member > p
    tw = twin_columns(int(pmax * 1.4) + 10 ** 5)
    tlow = 6 * tw - 1
    plist = primes[primes >= 7].tolist()
    prevp = {}
    for i, p in enumerate(primes.tolist()):
        if p >= 7:
            prevp[p] = int(primes[i - 1])
    d0 = tw[np.searchsorted(tlow, np.array(plist, dtype=np.int64), side="right")]
    rows = []
    todo = [(i, p) for i, p in enumerate(plist) if i % nshard == shard]
    for n, (i, p) in enumerate(todo):
        W = (p * p - 1) // 6
        k0, k1 = straddle(W, res, isp)
        rows.append((p, W, k0, k1, int(d0[i]), prevp[p]))
        if n and n % 20000 == 0:
            print(f"  ... p={p} n={n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    A = np.array(rows, dtype=np.int64)
    tag = f"{pmax}" if nshard == 1 else f"{pmax}_{shard}"
    np.savez_compressed(os.path.join(RES, f"straddle_{tag}.npz"), rows=A)
    print(f"{len(rows)} primes, {time.time() - t0:.0f}s -> straddle_{tag}.npz", flush=True)


if __name__ == "__main__":
    pmax = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 5
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nshard = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    main(pmax, shard, nshard)
