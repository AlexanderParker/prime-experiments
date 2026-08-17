"""Complete N(L)/kappa(L) profile over the full period at a given y, k-frame,
by segmented numpy sieve. Reproduces F_k(29) = 43 and F_k(31) = 58, and shows
the tail hazard rate sits at 2.2-3.7 times d at these y (the extreme-value
regime the small-L expansion cannot see). Usage: python kappa_profile.py 29
"""
import sys
import numpy as np
from math import prod

def primes_upto(n):
    s = bytearray([1])*(n+1); s[0:2] = b'\x00\x00'
    for i in range(2, int(n**0.5)+1):
        if s[i]: s[i*i::i] = bytearray(len(s[i*i::i]))
    return [i for i in range(n+1) if s[i]]

def gap_histogram(y, seg=10**8):
    qs = [q for q in primes_upto(y) if q >= 5]
    P = prod(qs)
    us = [(q, pow(6, -1, q)) for q in qs]
    hist = {}
    run = 0
    start = 0
    while start < P:
        n = min(seg, P - start)
        ex = np.ones(n, dtype=bool)
        for q, u in us:
            for r in (u % q, (-u) % q):
                ex[(r - start) % q::q] = False
        idx = np.flatnonzero(ex)
        if len(idx) == 0:
            run += n
        else:
            first = run + int(idx[0])
            if first: hist[first] = hist.get(first, 0) + 1
            d = np.diff(idx) - 1
            d = d[d > 0]
            vals, cnts = np.unique(d, return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                hist[v] = hist.get(v, 0) + c
            run = n - 1 - int(idx[-1])
        start += n
    if run: hist[run] = hist.get(run, 0) + 1  # cyclic close: slot 0 is exposed
    return hist, P, qs

if __name__ == "__main__":
    y = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    hist, P, qs = gap_histogram(y)
    d = prod((q-2)/q for q in qs)
    Rmax = max(hist)
    N = [0]*(Rmax+2)
    for g, c in hist.items():
        for L in range(1, g+1):
            N[L] += (g - L + 1)*c
    print(f"y={y}  P={P}  d={d:.6f}  F_k={Rmax+1}  1/d={1/d:.2f}")
    kmin, amin = None, None
    for L in range(1, Rmax+1):
        kap = ((1 - N[L+1]/N[L])/d - 1)/d
        if kmin is None or kap < kmin: kmin, amin = kap, L
        print(f"{L:3d} {kap:9.4f}  {N[L]}")
    print(f"min kappa = {kmin:.4f} at L={amin}")
