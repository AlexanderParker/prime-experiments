"""Inside the record plug runs of two_layer_census.py: which top gears do the killing?  For each
stretch, the longest plug run (consecutive base-open columns all killed by gears in (B, p]) is
listed with, per killed column, the least top gear that strikes it and its cofactor.  Tests
whether the clustering (K(p) above the independent-random 2.3 ln nb) comes from the smallest top
gears (near B), whose kills are densest.

usage: uv run python plug_run_gears.py p1 p2 ...
"""
import sys, math
import numpy as np

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

def lpf_in(n, lo, hi, P):
    for h in P:
        h = int(h)
        if h > hi: return None
        if h >= lo and n % h == 0: return h
    return None

for p in map(int, sys.argv[1:]):
    P = primes_upto(p + 200)
    q = int(P[np.searchsorted(P, p) + 1])
    c0 = (p * p + 1) // 6 + 1
    c1 = (q * q - 1) // 6 - 1
    L = c1 - c0 + 1
    B = int(round(q ** (2.0 / 3.0)))
    base = np.zeros(L, dtype=bool); top = np.zeros(L, dtype=bool)
    for h in P:
        h = int(h)
        if h < 5: continue
        if h > p: break
        u = pow(6, -1, h)
        arr = base if h <= B else top
        for tooth in (u, h - u):
            arr[(tooth - c0) % h::h] = True
    bopen_idx = np.nonzero(~base)[0]
    killed = top[bopen_idx]
    # longest run
    best = (0, 0); run = 0; start = 0
    for i, k in enumerate(killed):
        if k:
            if run == 0: start = i
            run += 1
            if run > best[0]: best = (run, start)
        else: run = 0
    K, s = best
    cols = bopen_idx[s:s + K] + c0
    print(f"\np={p} q={q} B={B} base_open={len(bopen_idx)} K={K} columns {cols[0]}..{cols[-1]} (span {cols[-1]-cols[0]+1})")
    small = 0; gears_used = []
    for c in cols:
        lo, hi = 6 * c - 1, 6 * c + 1
        gl = lpf_in(lo, B + 1, p, P); gh = lpf_in(hi, B + 1, p, P)
        g = min([x for x in (gl, gh) if x is not None])
        side = '-' if g == gl else '+'
        m = (lo if side == '-' else hi) // g
        gears_used.append(g)
        if g < 2 * B: small += 1
        print(f"   c={c:8d}  killer {g:6d} x {m:8d} on 6c{side}1   (g/B = {g/B:.2f})")
    print(f"   kills by gears below 2B: {small}/{K};  distinct gears: {len(set(gears_used))}/{K};  gear span {min(gears_used)}..{max(gears_used)}")
