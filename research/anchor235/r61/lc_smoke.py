"""lc_smoke.py -- smoke test of the closure machine on machines small enough to sieve directly.

Gate 1: D_1(M + q') from D_K(M) reproduces the directly built spectrum of M + q' value by value.
Gate 2: the order distribution (tagged column) reproduces the direct order histogram.
Gate 3: ITERATION -- D_1(M + q' + q'') from D_K(M) via two closure steps, no period in between.
Gate 4: the span-bounded form has loss 0 and reproduces the same record.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_core import (PRIMES, base_gaps, dict_from_gaps, closure_step, spectrum, order_hist,
                     fj, word_stats)


def direct_spec(top):
    P, g = base_gaps(top)
    return np.bincount(g.astype(np.int64))


def main():
    print("== gate 1/2: one closure step against the directly built machine ==")
    for k in range(len(PRIMES) - 1):
        top, q = PRIMES[k], PRIMES[k + 1]
        if q > 23:
            break
        P, g = base_gaps(top)
        for K in range(2, 10):
            win, mult = dict_from_gaps(g, K)
            w2, m2, st = closure_step(win, mult, q, m=1, mode='fixed', tag_order=True)
            if st["loss"] == 0:
                break
        sp = spectrum(w2, m2)
        oh = order_hist(w2, m2, 1)
        ds = direct_spec(q)
        n = max(sp.size, ds.size)
        sp2 = np.zeros(n, dtype=np.int64); sp2[:sp.size] = sp
        ds2 = np.zeros(n, dtype=np.int64); ds2[:ds.size] = ds
        ok = np.array_equal(sp2, ds2)
        print(f"m{top} -> m{q}: K={K} |D_K|={win.shape[0]} loss={st['loss']} kmax={st['kmax']} "
              f"F={int(np.flatnonzero(sp).max())} spectrum {'EXACT' if ok else 'MISMATCH'} "
              f"orders={list(oh[1:])} sum={int(sp.sum())}")

    print("\n== gate 3: iteration, m13 -> m17 -> m19 -> m23, dictionaries only ==")
    P, g = base_gaps(13)
    win, mult = dict_from_gaps(g, 12)
    print(f"  base m13: |D_12| = {win.shape[0]}, N = {int(mult.sum())}")
    for q, want in ((17, 18), (19, 25), (23, 34)):
        t0 = time.time()
        w2, m2, st = closure_step(win, mult, q, m=win.shape[1] - 3, mode='fixed')
        sp = spectrum(w2, m2)
        ds = direct_spec(q)
        n = max(sp.size, ds.size)
        a = np.zeros(n, dtype=np.int64); a[:sp.size] = sp
        b = np.zeros(n, dtype=np.int64); b[:ds.size] = ds
        print(f"  -> m{q}: depth {w2.shape[1]} |D| = {w2.shape[0]} N = {int(m2.sum())} "
              f"F = {int(np.flatnonzero(sp).max())} (want {want}) loss = {st['loss']} "
              f"kmax = {st['kmax']} spectrum {'EXACT' if np.array_equal(a, b) else 'MISMATCH'} "
              f"[{time.time()-t0:.1f}s]")
        win, mult = w2, m2

    print("\n== gate 4: the span-bounded form, m13 -> m17 -> m19 -> m23, S = 40 ==")
    S = 40
    P, g = base_gaps(13)
    win, mult = dict_from_gaps(g, 24, span_cap=S)
    print(f"  base m13: |V_{S}| = {win.shape[0]}, rows with no terminating zero = "
          f"{int((win[:, -1] != 0).sum())}")
    for q, want in ((17, 18), (19, 25), (23, 34)):
        w2, m2, st = closure_step(win, mult, q, m=win.shape[1], mode='span')
        sp = spectrum(w2, m2)
        ds = direct_spec(q)
        n = max(sp.size, ds.size)
        a = np.zeros(n, dtype=np.int64); a[:sp.size] = sp
        b = np.zeros(n, dtype=np.int64); b[:ds.size] = ds
        L, Lp, Wc, Zc, Lb, Lz = word_stats(w2, m2, PRIMES[PRIMES.index(q) + 1])
        print(f"  -> m{q}: |V_{S}| = {w2.shape[0]} N = {int(m2.sum())} F = "
              f"{int(np.flatnonzero(sp).max())} (want {want}) loss = {st['loss']} "
              f"depth_full_in = {st['depth_full']} F_2 = {fj(w2, m2, 2)} "
              f"F_3 = {fj(w2, m2, 3)} L = {L} L_pad = {Lp} "
              f"spectrum {'EXACT' if np.array_equal(a, b) else 'MISMATCH'}")
        win, mult = w2, m2


if __name__ == "__main__":
    main()
