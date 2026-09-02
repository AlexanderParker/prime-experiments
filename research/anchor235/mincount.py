"""Min number of openings of the old word {5..q} in any stretch of L = F + q' + 1 slots,
against the two-class capacity 2*ceil(L/q'). If min > capacity, (D) at that rung follows by
pigeonhole for ANY two teeth. Also: the exact two-class maximum (from both.py) for reference,
and the smallest L at which min count exceeds capacity.
"""
import sys
from math import prod

import numpy as np

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
        reps = 1 + (8 * q2 + 400) // P
        ww = np.concatenate([w] * (reps + 1)).astype(np.int32)
        cs = np.concatenate([[0], np.cumsum(ww)])
        X = np.flatnonzero(w)
        gaps = np.diff(np.concatenate([X, [X[0] + P]]))
        F = int(gaps.max()) - 1
        L = F + q2 + 1
        cnt = cs[L:L + P] - cs[:P]
        mn = int(cnt.min())
        cap = 2 * ((L + q2 - 1) // q2)
        # stretch lengths where pigeonhole starts to work
        Lstar = None
        for LL in range(L, L + 3 * q2):
            c = cs[LL:LL + P] - cs[:P]
            if int(c.min()) > 2 * ((LL + q2 - 1) // q2):
                Lstar = LL
                break
        # distribution of openings in the worst stretches
        i = int(cnt.argmin())
        print(f"{'+'.join(map(str, gears))} + {q2}: F={F} L=F+q'+1={L}  min openings in any L-stretch={mn}  "
              f"two-class capacity 2*ceil(L/q')={cap}  pigeonhole {'PROVES (D)' if mn > cap else 'FAILS'}  "
              f"first L with min>cap: {Lstar} (= F + q' + {Lstar - F - q2 if Lstar else '?'})")
        k = np.arange(i, i + L)
        print(f"    worst stretch starts k={i}: openings at offsets {list(np.flatnonzero(ww[i:i + L]))}")


if __name__ == "__main__":
    main()
