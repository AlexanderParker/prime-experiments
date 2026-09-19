"""Task 4 to 10^7: first rung index among 61-rough offsets (rank by |j|, j<0 first on ties).
Members near s^2 <= 10^14: primality by gmpy2 (strong BPSW), Omega by sympy.factorint when needed.
Usage: python task4_index.py MODE [N_SAMPLE] [S_LO]
  MODE = P_P (P-centres, P-rungs), Q_P (Q-centres, P-rungs), Q_Q (Q-centres, Q-rungs),
         P_Q (P-centres, Q-rungs), M_P (M'-centres, P-rungs). N_SAMPLE > 0 takes a seeded random
  sample of that many centres in [S_LO, 10^7]. Default S_LO = 10^5 (below it task2_laws.py is exact)."""
import math
import os
import sys
import time

import gmpy2
import numpy as np
from sympy import factorint

HERE = os.path.dirname(os.path.abspath(__file__))
SMALL = (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61)


def is_prime(n):
    return bool(gmpy2.is_strong_bpsw_prp(gmpy2.mpz(int(n))))


def omega(n):
    n = int(n)
    if is_prime(n):
        return 1
    return sum(factorint(n).values())


def rough_candidates(s, W):
    c = s // 6
    J = 2 * c - 1
    W = min(W, J)
    j = np.arange(-W, W + 1, dtype=np.int64)
    s2 = s * s
    A = s2 + 6 * j - 1
    B = s2 + 6 * j + 1
    rough = np.ones(len(j), dtype=bool)
    for p in SMALL:
        rough &= (A % p != 0) & (B % p != 0)
    order = np.lexsort((j, np.abs(j)))
    order = order[rough[order]]
    return j[order], A[order], B[order], W == J


def first_index(s, rung_test, W0=3000):
    W = W0
    while True:
        j, A, B, full = rough_candidates(s, W)
        for i in range(len(j)):
            if rung_test(A[i], B[i]):
                return i + 1, int(j[i])
        if full:
            return None, None
        W *= 4


def test_P(a, b):
    return is_prime(a) and is_prime(b)


def test_Q(a, b):
    return omega(a) % 2 == 0 and omega(b) % 2 == 0


def main():
    mode = sys.argv[1]
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    s_lo = int(float(sys.argv[3])) if len(sys.argv) > 3 else 100000
    t0 = time.time()
    d = np.load(os.path.join(HERE, "sieve.npz"))
    flags = {"P": d["P"], "Q": d["Q"], "M": d["Mp"]}[mode[0]]
    K = len(flags)
    s_all = 6 * np.arange(1, K + 1, dtype=np.int64)[flags]
    s_all = s_all[s_all >= s_lo]
    if n_sample and n_sample < len(s_all):
        rng = np.random.default_rng(20260920)
        s_all = np.sort(rng.choice(s_all, size=n_sample, replace=False))
    test = test_P if mode[2] == "P" else test_Q
    print(f"mode {mode}: {len(s_all)} centres in [{s_lo}, {int(s_all.max())}]")
    idx = np.zeros(len(s_all), dtype=np.int64)
    jj = np.zeros(len(s_all), dtype=np.int64)
    for i, s in enumerate(s_all):
        r, j0 = first_index(int(s), test)
        idx[i] = -1 if r is None else r
        jj[i] = 0 if j0 is None else j0
        if (i + 1) % 5000 == 0:
            print(f"  {i+1} centres, {time.time()-t0:.0f}s", flush=True)
    ok = idx > 0
    ls2 = np.log(s_all.astype(float)) ** 2
    ratio = idx[ok] / ls2[ok]
    am = int(np.argmax(ratio))
    print(f"\n{mode}: n = {len(s_all)}, no rung found = {int((~ok).sum())}, mean index = {idx[ok].mean():.2f}, "
          f"max index = {int(idx[ok].max())}")
    print(f"max index/ln^2 s = {ratio.max():.4f} at s = {int(s_all[ok][am])} (index {int(idx[ok][am])}, j = {int(jj[ok][am])}); "
          f"mean index/ln^2 s = {ratio.mean():.4f}")
    print(f"centres with index/ln^2 s > 0.6: {int((ratio > 0.6).sum())}; list: "
          + ", ".join(f"s={int(s_all[ok][i])} idx={int(idx[ok][i])} ratio={ratio[i]:.3f}" for i in np.nonzero(ratio > 0.6)[0][:20]))
    # per-decade table
    print(f"{'s range':>20} {'n':>7} {'mean idx':>9} {'mean idx/ln^2 s':>16} {'max idx/ln^2 s':>15} {'idx ln^2 s fit c':>16}")
    edges = [1e5, 3e5, 1e6, 3e6, 1e7 + 1]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = ok & (s_all >= lo) & (s_all < hi)
        if sel.sum() == 0:
            continue
        r = idx[sel] / ls2[sel]
        # geometric: mean index = 1/d, d ln^2 s = c  ->  c = ln^2 s / mean index
        c = (ls2[sel] / idx[sel].mean()).mean()
        print(f"{int(lo):>9}-{int(hi-1):>10} {int(sel.sum()):>7} {idx[sel].mean():>9.2f} {r.mean():>16.4f} {r.max():>15.4f} {c:>16.2f}")
    # tail check against geometric with the fitted mean
    dens = 1.0 / idx[ok].mean()
    print("tail: " + "; ".join(f">{n}: {int((idx[ok] > n).sum())} obs / {len(idx[ok]) * (1-dens)**n:.2f} geo" for n in (20, 40, 80, 120, 160, 200)))
    np.savez(os.path.join(HERE, f"task4_{mode}_{n_sample}_{s_lo}.npz"), s=s_all, idx=idx, j=jj)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
