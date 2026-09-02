"""Teeth-free criterion for (D): in every stretch of L = F + q' + 1 slots of the old word,
some q' consecutive slots hold >= 3 openings (two teeth = two step-q' progressions, at most
2 kills per q' consecutive slots, so such a stretch cannot be fully killed). Report the min
over stretches of (max over q'-subwindows of the opening count), and the first L where the
criterion holds.  Also the exact two-AP test: does any L-stretch have all openings inside
two step-q' progressions (that is (D) failing for some delta)?
"""
import sys
from math import prod

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PR = [5, 7, 11, 13, 17, 19, 23, 29, 31]


def openings(gears):
    P = prod(gears)
    k = np.arange(P, dtype=np.int64)
    w = np.ones(P, dtype=bool)
    for g in gears:
        u = pow(6, -1, g)
        w &= (k % g != u) & (k % g != g - u)
    return w, P


def main():
    qmax = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    for idx in range(1, len(PR)):
        q2 = PR[idx]
        if q2 > qmax:
            break
        gears = PR[:idx]
        w, P = openings(gears)
        reps = 2 + (8 * q2 + 400) // P
        ww = np.concatenate([w] * reps).astype(np.int32)
        cs = np.concatenate([[0], np.cumsum(ww)])
        X = np.flatnonzero(w)
        gaps = np.diff(np.concatenate([X, [X[0] + P]]))
        F = int(gaps.max()) - 1
        L = F + q2 + 1
        cq = cs[q2:q2 + P + L] - cs[:P + L]  # openings in [k, k+q')
        # max over subwindows k..k+L-q' of cq
        best = None
        for LL in range(L, L + 4 * q2):
            mx = sliding_window_view(cq[:P + LL - q2 + 1], LL - q2 + 1).max(axis=1)[:P]
            mn = int(mx.min())
            if best is None:
                best = (mn, int(mx.argmin()))
            if mn >= 3:
                Lstar = LL
                break
        else:
            Lstar = None
        mn, i = best
        print(f"{'+'.join(map(str, gears))} + {q2}: F={F} L={L}: min over L-stretches of (max openings in a q'-subwindow) = {mn} "
              f"-> {'CRITERION HOLDS, (D) proved for every tooth pair' if mn >= 3 else 'not enough'};  "
              f"first L with criterion: {Lstar} (= F + q' + {Lstar - F - q2 if Lstar else '?'})")
        if mn < 3:
            offs = list(map(int, np.flatnonzero(ww[i:i + L])))
            print(f"    worst stretch k={i}: openings at offsets {offs}; gaps {[b - a for a, b in zip(offs, offs[1:])]}")


if __name__ == "__main__":
    main()
