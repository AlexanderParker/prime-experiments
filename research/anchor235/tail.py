"""Gap tail of the old word: number of gaps >= L per period, its log-slope lambda near the
record, and the heuristic F_m - F ~ (m-1) ln(F)/lambda (each allowed hole multiplies the
number of configurations by about the number of hole positions, buying ln/lambda slots).
"""
import sys
from math import prod, log

import numpy as np

PR = [5, 7, 11, 13, 17, 19, 23, 29]


def main():
    for idx in range(3, len(PR)):
        gears = PR[:idx]
        P = prod(gears)
        k = np.arange(P, dtype=np.int64)
        w = np.ones(P, dtype=bool)
        for g in gears:
            u = pow(6, -1, g)
            w &= (k % g != u) & (k % g != g - u)
        X = np.flatnonzero(w)
        gaps = np.diff(np.concatenate([X, [X[0] + P]]))
        F = int(gaps.max()) - 1
        F2 = int((gaps + np.roll(gaps, -1)).max()) - 1
        F3 = int((gaps + np.roll(gaps, -1) + np.roll(gaps, -2)).max()) - 1
        Ls = np.arange(1, F + 2)
        tail = np.array([(gaps >= L).sum() for L in Ls])
        # slope of log tail over the top half of the range
        lo = F // 2
        sel = (Ls >= lo) & (tail > 0)
        lam = -np.polyfit(Ls[sel], np.log(tail[sel]), 1)[0]
        pred2 = log(F) / lam
        print(f"{'+'.join(map(str, gears))}: N={len(X)} mean gap {P / len(X):.2f}  F={F} F2={F2} F3={F3}  "
              f"gaps>=F+1: {int(tail[-1])}, >=F/2: {int(tail[lo - 1])}  lambda={lam:.3f} (1/lambda={1 / lam:.2f})  "
              f"heuristic F2-F ~ lnF/lambda = {pred2:.1f} (measured {F2 - F}), F3-F ~ 2lnF/lambda = {2 * pred2:.1f} (measured {F3 - F})")


if __name__ == "__main__":
    main()
