"""Manifold census at large Q, precomputed overnight (2026-09-07) for the valves work.

For the manifold (the primes in (q, Q]) acting on [1, Q^2]: segmented sieve by the manifold's
primes only, so an integer is open iff it has no prime factor in (q, Q] (the quiet-zone rule:
open = q-smooth times at most one prime above Q).  Reports, per (q, Q):
  - the open pairs (n, n+2) count, and the record gap between consecutive open pairs with its
    position (whole range, and the quiet zone (Q, Q^2] alone);
  - the gap spectrum (top 25 gap lengths by count, and the 10 largest gaps with positions);
  - the family census: (s, s') = the q-smooth cofactors of the two members, counts of the top
    40 families, the number of distinct families, and family (1, 1) against pi_2(Q^2) - pi_2(Q).
usage: uv run python research/topmachine/r9/manifold_census.py q Q [segment]
"""
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0]


def smooth_cofactor(arr, small):
    """q-smooth part of each entry (vectorised trial division by the engine's primes)."""
    a = arr.copy()
    s = np.ones_like(a)
    for p in small:
        while True:
            m = (a % p == 0)
            if not m.any():
                break
            a[m] //= p
            s[m] *= p
    return s


def main():
    q, Q = int(sys.argv[1]), int(sys.argv[2])
    seg = int(sys.argv[3]) if len(sys.argv) > 3 else 50_000_000
    t0 = time.time()
    X = Q * Q
    ps = primes_upto(Q)
    small = [int(p) for p in ps if p <= q]
    mani = ps[ps > q]
    out = [f"# manifold census q={q} Q={Q} range [1,{X}] manifold gears {len(mani)} (from {mani[0]} to {mani[-1]})"]

    gaps = Counter()
    families = Counter()
    n_open_pairs = 0
    big = []  # (gap, position) largest gaps
    rec_quiet = (0, 0)
    last_open = None  # position of the previous open pair (global)
    # twin-prime count in (Q, X] for the family (1,1) check: primes above Q with p+2 prime
    twin_count = 0
    prev_prime_open = None

    lo = 1
    while lo <= X:
        hi = min(lo + seg - 1, X)
        L = hi - lo + 1
        blk = np.ones(L + 2, dtype=bool)  # positions lo..hi+2 (need n+2)
        for p in mani:
            p = int(p)
            start = (-lo) % p
            blk[start::p] = False
        opn = blk  # opn[i] True iff lo+i open
        pair = opn[:L] & opn[2:L + 2]
        idx = np.nonzero(pair)[0]
        if idx.size:
            pos = idx + lo
            n_open_pairs += idx.size
            # gaps between consecutive open pairs (including across segment boundary)
            if last_open is not None:
                g = int(pos[0] - last_open)
                gaps[g] += 1
                big.append((g, int(last_open)))
                if last_open > Q and g > rec_quiet[0]:
                    rec_quiet = (g, int(last_open))
            d = np.diff(pos)
            if d.size:
                u, c = np.unique(d, return_counts=True)
                for a, b in zip(u, c):
                    gaps[int(a)] += int(b)
                k = min(10, d.size)
                top = np.argpartition(d, -k)[-k:]
                for j in top:
                    big.append((int(d[j]), int(pos[j])))
                m = (pos[:-1] > Q)
                if m.any():
                    dq = d[m]
                    j = int(np.argmax(dq))
                    if dq[j] > rec_quiet[0]:
                        rec_quiet = (int(dq[j]), int(pos[:-1][m][j]))
            last_open = int(pos[-1])
            # families: smooth cofactors of both members
            s1 = smooth_cofactor(pos.astype(np.int64), small)
            s2 = smooth_cofactor((pos + 2).astype(np.int64), small)
            key = s1 * (1 << 32) + s2
            u, c = np.unique(key, return_counts=True)
            for a, b in zip(u, c):
                families[(int(a) >> 32, int(a) & 0xFFFFFFFF)] += int(b)
            big.sort(reverse=True)
            big = big[:10]
        lo = hi + 1
        if (lo // seg) % 20 == 0:
            print(f"  ... {lo}/{X} ({time.time() - t0:.0f}s)", flush=True)

    out.append(f"open pairs in [1,{X}]: {n_open_pairs}")
    out.append(f"record gap in the quiet zone (Q, Q^2]: {rec_quiet[0]} starting after position {rec_quiet[1]} (fraction of Q: {rec_quiet[1] / Q:.3f})")
    out.append("largest 10 gaps overall (gap, after position): " + ", ".join(f"({g},{p})" for g, p in big))
    out.append("gap spectrum (top 25 by count): " + ", ".join(f"{g}:{c}" for g, c in sorted(gaps.items(), key=lambda kv: -kv[1])[:25]))
    out.append(f"gap lengths present: {len(gaps)}; gap 4 count: {gaps.get(4, 0)}; gap 3: {gaps.get(3, 0)}; gap 5: {gaps.get(5, 0)}")
    out.append(f"distinct families: {len(families)}; family (1,1): {families.get((1, 1), 0)}")
    out.append("top 40 families (s, s'): count: " + ", ".join(f"({a},{b}):{c}" for (a, b), c in families.most_common(40)))
    out.append(f"elapsed {time.time() - t0:.0f}s")
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(RES, f"census_q{q}_Q{Q}.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
