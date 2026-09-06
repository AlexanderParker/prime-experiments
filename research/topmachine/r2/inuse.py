"""W2: the fixed-gear top machine on a prefix [0, N), scanned in chunks.

For a fixed gear set G the wheel W = prod g is far above any range in use.  This script
scans a prefix exactly, in chunks, and records

  * the exact gap census (gap = distance between consecutive open pairs) over the prefix;
  * the FIRST OCCURRENCE position of every gap length (equivalently of every pair-free run
    length), which is what the range record is made of;
  * the running record F_range(N) at a ladder of N.

Run with the whole period as the prefix and the census is the wheel's exact census.

Scan convention: start with last_open = -1 (the shield, open for every gear), so a scan of
[0, W) yields exactly prod (g - 2) gaps summing to W - the exact cyclic census.

usage: uv run python research/topmachine/r2/inuse.py results/inuse.json
"""

import json
import sys
import time
from math import prod

import numpy as np

CHUNK = 1 << 25  # 33.5 M positions, ~34 MB of bool


def scan(gears, M, checkpoints):
    """Scan [0, M).  Returns (gap counts, first occurrence of each gap length, record ladder)."""
    counts = {}
    first = {}
    ladder = []
    ck = sorted(checkpoints)
    ci = 0
    best = 0
    last_open = -1
    t0 = time.time()
    base = 0
    nchunk = 0
    while base < M:
        n = min(CHUNK, M - base)
        a = np.ones(n, dtype=bool)
        for g in gears:
            a[(-base) % g :: g] = False
            a[(-2 - base) % g :: g] = False
        idx = np.flatnonzero(a)
        if len(idx):
            pos = idx.astype(np.int64) + base
            prev = np.empty(len(pos), dtype=np.int64)
            prev[0] = last_open
            prev[1:] = pos[:-1]
            gaps = pos - prev
            for L, c in zip(*np.unique(gaps, return_counts=True)):
                L = int(L)
                counts[L] = counts.get(L, 0) + int(c)
                if L not in first:
                    j = int(np.argmax(gaps == L))
                    first[L] = int(prev[j]) + 1  # start of the pair-free block
            last_open = int(pos[-1])
        base += n
        nchunk += 1
        while ci < len(ck) and ck[ci] <= base:
            N = ck[ci]
            f = max((L - 1 for L in first if first[L] + L - 2 < N), default=0)
            ladder.append({"N": N, "F_range": f})
            ci += 1
        best = max(counts)
        if nchunk % 20 == 0 or base >= M:
            print(
                "    q'=%d  %.1f%%  t=%.0fs  maxgap=%d"
                % (gears[0], 100 * base / M, time.time() - t0, best),
                flush=True,
            )
    return counts, first, ladder


def main():
    out = []
    jobs = [
        # gears, prefix length, label
        ([7, 11, 13, 17, 19, 23, 29, 31], None, "full period"),
        ([13, 17, 19, 23, 29, 31, 37, 41], 2 * 10**10, "prefix"),
        ([19, 23, 29, 31, 37, 41, 43, 47], 10**10, "prefix"),
    ]
    for gears, M, label in jobs:
        W = prod(gears)
        if M is None:
            M = W
        cks = [10**k for k in range(4, 11)] + [W]
        cks = [c for c in cks if c <= M]
        t = time.time()
        counts, first, ladder = scan(gears, M, cks)
        rec = max(counts)
        out.append(
            {
                "gears": gears,
                "W": W,
                "scanned": M,
                "label": label,
                "fraction_of_period": M / W,
                "gap_counts": {str(k): v for k, v in sorted(counts.items())},
                "first_occurrence": {str(k): v for k, v in sorted(first.items())},
                "record_ladder": ladder,
                "max_gap_seen": rec,
                "F_seen": rec - 1,
                "open_pairs": sum(counts.values()),
                "prod_g_minus_2": prod(g - 2 for g in gears),
                "secs": round(time.time() - t, 1),
            }
        )
        print(gears, "W=%d" % W, "F=%d" % (rec - 1), "ladder", ladder, flush=True)
        with open(sys.argv[1], "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
