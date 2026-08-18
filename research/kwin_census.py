"""Round 18 ask (mechanic): k_win vs k_max, and the par-trading test.

For a step M -> q' a run of k co-deletable openings merges k+1 consecutive
gaps.  Legality (fuel_census frame): consecutive-opening gaps are classified
mod q' as letters +1 (= s), -1 (= q'-s), 0 (padded, gap divisible by q'),
anything else breaks the run; a (k-1)-word of letters is window-valid iff its
prefix-sum range is <= 1.

This tool reports, per depth k, the MAXIMUM FLANKED MERGED SPAN
    merged(i, k) = ops[i+k] - ops[i-1]
over all window-valid k-tuples, with its address and letter word, plus the
count.  Then:
    k_max = deepest k with any valid tuple      (the fuel cap)
    k_win = the k attaining the overall maximum (the depth that wins)
and the spread of max-merged across depths IS the par-trading test: if the
merged maximum is nearly depth-independent, deep chains cannot beat shallow.

Usage: uv run python research/kwin_census.py y q' [--kmax K] [--limit N]
"""
import os
import sys
import time
import numpy as np
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flank_envelope import primes_upto

KW = 8


def main():
    args = sys.argv[1:]
    limit = None
    kw = KW
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(float(args[i + 1]))
        del args[i:i + 2]
    if "--kmax" in args:
        i = args.index("--kmax")
        kw = int(args[i + 1])
        del args[i:i + 2]
    y, q1 = int(args[0]), int(args[1])
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    K = P if limit is None else min(P, limit)
    uvals = [pow(6, -1, g) for g in gears]
    s = (2 * pow(6, -1, q1)) % q1
    best = {k: (0, -1, None) for k in range(1, kw + 1)}
    cnt = np.zeros(kw + 1, np.int64)
    F = 0
    tail = np.array([], dtype=np.int64)
    t0 = time.time()
    for lo in range(0, K, 64_000_000):
        hi = min(K, lo + 64_000_000)
        ex = np.zeros(hi - lo, bool)
        for g, u in zip(gears, uvals):
            ex[(u - lo) % g::g] = True
            ex[(-u - lo) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64) + lo
        ops = np.concatenate([tail, op])
        if len(ops) > kw + 3:
            d = np.diff(ops)
            F = max(F, int(d.max()))
            n = len(d)
            dm = d % q1
            letter = np.full(n, 9, np.int8)
            letter[dm == 0] = 0
            letter[dm == s] = 1
            letter[dm == (q1 - s) % q1] = -1
            valid = letter != 9
            for k in range(1, kw + 1):
                # k openings at index i..i+k-1 ; letters d[i..i+k-2]
                L = n - k - 1
                if L <= 1:
                    break
                if k == 1:
                    ok = np.ones(L, bool)
                else:
                    ok = np.ones(L, bool)
                    run = np.zeros(L, np.int16)
                    lo_ = np.zeros(L, np.int16)
                    hi_ = np.zeros(L, np.int16)
                    for m in range(k - 1):
                        seg = letter[1 + m:1 + m + L]
                        ok &= valid[1 + m:1 + m + L]
                        run = run + seg.astype(np.int16)
                        lo_ = np.minimum(lo_, run)
                        hi_ = np.maximum(hi_, run)
                    ok &= (hi_ - lo_) <= 1
                if not ok.any():
                    continue
                i = np.arange(1, L + 1)[ok[:L]]
                new = ops[i + k] >= lo
                i = i[new]
                if len(i) == 0:
                    continue
                merged = ops[i + k] - ops[i - 1]
                cnt[k] += len(i)
                j = int(np.argmax(merged))
                if int(merged[j]) > best[k][0]:
                    w = tuple(int(x) for x in d[i[j]:i[j] + k - 1]) \
                        if k > 1 else ()
                    best[k] = (int(merged[j]), int(ops[i[j] - 1]), w)
        tail = ops[-(kw + 4):].copy()
    print(f"machine {y} -> q'={q1}: F = {F}, F+q' = {F+q1}, "
          f"s = {s} mod {q1}, coverage {K/P:.4f}, {time.time()-t0:.0f}s")
    print("   k   valid k-tuples      max merged   -F    /q'    "
          "address (flank opening)   interior gap word")
    tot = 0
    for k in range(1, kw + 1):
        m, ad, w = best[k]
        if cnt[k] == 0:
            continue
        tot = max(tot, m)
        print(f"  {k:2d}   {int(cnt[k]):>14,}   {m:9d}  {m-F:4d}  "
              f"{(m-F)/q1:5.2f}   k = {ad:>14,}   {w}")
    kmax = max(k for k in range(1, kw + 1) if cnt[k] > 0)
    kwin = max((k for k in range(1, kw + 1) if cnt[k] > 0),
               key=lambda k: best[k][0])
    vals = [best[k][0] for k in range(1, kmax + 1) if cnt[k] > 0]
    print(f"  k_max = {kmax}   k_win = {kwin}   record merged = {tot} "
          f"(= F(M+q') if the census is complete)")
    print(f"  PAR TRADING: max-merged across depths = {vals}, "
          f"spread = {100*(max(vals)-min(vals))/max(vals):.1f}%")


if __name__ == "__main__":
    main()
