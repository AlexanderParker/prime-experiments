"""lc_size.py -- how big is the span-bounded dictionary, and how fast does it grow per rung.

V_S(M) is the multiset of opening patterns of M inside [x, x + S] over all openings x, written
as the gap word truncated at span S.  It is exactly closed under the rung step (lc_core), so the
whole ladder is one parameter S and one number: |V_S(M)|.

Usage: uv run python research/anchor235/r61/lc_size.py [S] [maxrung]
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_core import PRIMES, base_gaps, dict_from_gaps, closure_step, spectrum, fj, OUT


def width_for(gaps, S):
    """The largest number of gaps a span-<=S window ever uses (so the array width is exact)."""
    N = gaps.size
    g = np.concatenate([gaps, gaps[:2 * S + 4]]).astype(np.int32)
    best = 0
    step = 4_000_000
    for lo in range(0, N, step):
        hi = min(lo + step, N)
        off = np.zeros(hi - lo, dtype=np.int32)
        cnt = np.zeros(hi - lo, dtype=np.int32)
        for i in range(2 * S + 4):
            off += g[lo + i:hi + i]
            live = off <= S
            if not live.any():
                break
            cnt += live
        best = max(best, int(cnt.max()))
    return best


def main():
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 95
    maxrung = int(sys.argv[2]) if len(sys.argv) > 2 else 41
    base = 23
    P, g = base_gaps(base)
    W = width_for(g, S) + 1
    print(f"S = {S}; base m{base}: N = {g.size}, max gaps in a span-{S} window = {W-1}")
    t0 = time.time()
    win, mult = dict_from_gaps(g, W, span_cap=S)
    print(f"m{base}: |V_S| = {win.shape[0]:,}  N = {int(mult.sum()):,}  "
          f"width = {W}  overfull = {int((win[:, -1] != 0).sum())}  [{time.time()-t0:.1f}s]")
    rows = [(base, win.shape[0], int(mult.sum()), None, None, None, 0.0)]
    for k in range(PRIMES.index(base) + 1, len(PRIMES)):
        q = PRIMES[k]
        if q > maxrung:
            break
        t0 = time.time()
        win, mult, st = closure_step(win, mult, q, m=win.shape[1], mode='span')
        sp = spectrum(win, mult)
        F = int(np.flatnonzero(sp).max())
        secs = time.time() - t0
        print(f"m{q}: |V_S| = {win.shape[0]:,}  N = {int(mult.sum()):,}  F = {F}  "
              f"F_2 = {fj(win, mult, 2)}  loss = {st['loss']}  over0 = {st['over0']}  "
              f"kmax = {st['kmax']}  overfull_in = {st['depth_full']}  [{secs:.1f}s]",
              flush=True)
        rows.append((q, win.shape[0], int(mult.sum()), F, fj(win, mult, 2), st['over0'], secs))
    with open(os.path.join(OUT, f"size_S{S}.txt"), "w") as f:
        f.write(f"S = {S}\nrung | |V_S| | N | F | F_2 | over0 | secs\n")
        for r in rows:
            f.write(" | ".join(str(x) for x in r) + "\n")


if __name__ == "__main__":
    main()
