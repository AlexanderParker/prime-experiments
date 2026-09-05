"""R2.a.i.a.1.a.2 - second moment over q.  The twin-certified sub-object and the error terms.

1) the TWIN-CERTIFIED object: q coprime to 210 with q = 5 (mod 6), so the arc is d = (q+1)/3,
   6i <= 2q - 4 and every open island is a genuine twin prime pair (q^2+6i-2, q^2+6i) inside
   (q^2, (q+1)^2).  E[N], Var, failures and the normalised Chebyshev bound B(X) on that object.

2) the contiguous failure threshold of the one-class witness over [1000, 128000].

3) the equidistribution budget: the gear pairs (g, h) with g h <= 2X (the only pairs whose joint
   classes are equidistributed over a band of length X) against all gear pairs; and the
   equidistributed fraction of the sifting, ln(2X) / theta(2X).

4) the measured variance against the CRT prediction, band by band.

Usage: uv run python research/anchor235/r50/mom_final.py
"""
import os
from math import exp, isqrt, log

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
XS = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
S2 = 4.0 * exp(-2.0 * 0.5772156649015329)
# Var_model / mu measured exactly at 42 sampled q (mom_pair part D)
DISP = {"short": 0.8130, "long": 0.7890}


def primes_upto(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return [i for i in range(2, n + 1) if fl[i]]


def main():
    L = []
    L.append("1) the TWIN-CERTIFIED object: q = 5 (mod 6), coprime to 7, arc d = (q+1)/3")
    L.append("   every open island is a twin prime pair in (q^2, (q+1)^2)")
    L.append("   band     |A''|     E[N]      Var     Var/E    fails   frac        B(X)     B*X/(lnX)^2")
    rows = []
    for X in XS:
        z = np.load(os.path.join(OUT, "mom_scan_X%d.npz" % X))
        q, N, mh = z["q"], z["N"].astype(float), z["mu_hat"]
        s = ((q % 6) == 5) & ((q % 7) != 0)
        qq, NN, mu = q[s], N[s], mh[s] / S2
        E, V = NN.mean(), NN.var()
        f = int((NN == 0).sum())
        B = (((NN - mu) / mu) ** 2).mean()
        rows.append((X, len(NN), E, V, f, B))
        L.append("   %6d %7d %9.4f %9.3f %8.4f %6d  %.3e %10.5f %11.4f"
                 % (X, len(NN), E, V, V / E, f, f / len(NN), B, B * X / log(X) ** 2))
    L.append("")
    L.append("2) the contiguous failure threshold of the one-class witness over [1000, 128000]")
    allq, allN = [], []
    for X in XS:
        z = np.load(os.path.join(OUT, "mom_scan_X%d.npz" % X))
        allq.append(z["q"]); allN.append(z["N"])
    q = np.concatenate(allq); N = np.concatenate(allN)
    keep = np.ones(len(q), dtype=bool)
    keep[1:] = q[1:] != q[:-1]
    q, N = q[keep], N[keep]
    fails = q[N == 0]
    last = int(fails.max())
    above = (q > last)
    L.append("   q coprime to 30 in [1000, 128000]: %d values, %d failures, last at q = %d"
             % (len(q), len(fails), last))
    L.append("   0 failures over the %d values of q coprime to 30 in (%d, 128000]"
             % (int(above.sum()), last))
    c210 = (q % 7) != 0
    f210 = q[c210 & (N == 0)]
    L.append("   restricted to q coprime to 210: %d failures, last at q = %d; 0 failures over the "
             "%d values above it" % (len(f210), int(f210.max()),
                                     int(((q > f210.max()) & c210).sum())))
    s56 = ((q % 6) == 5) & c210
    f56 = q[s56 & (N == 0)]
    L.append("   twin-certified object (q = 5 mod 6, coprime to 7): %d failures, last at q = %d; "
             "0 failures over the %d values above it"
             % (len(f56), int(f56.max()), int(((q > f56.max()) & s56).sum())))
    L.append("   all failures above 5,000: %s" % str([int(x) for x in fails[fails > 5000]]))
    L.append("")
    L.append("3) the equidistribution budget over a band of length X")
    L.append("   band   gears above 7 up to 2X   all gear pairs   pairs with g h <= 2X   fraction"
             "     ln(2X)/theta(2X)")
    for X in XS:
        pr = [p for p in primes_upto(2 * X) if p > 7]
        G = len(pr)
        tot = G * (G - 1) // 2
        cnt = 0
        arr = np.array(pr, dtype=np.int64)
        for a in pr:
            if a * a > 2 * X:
                break
            cnt += int(np.searchsorted(arr, (2 * X) // a, side="right")) - int(
                np.searchsorted(arr, a, side="right"))
        theta = float(np.log(arr.astype(np.float64)).sum())
        L.append("   %6d %20d %16d %22d %10.3e %14.3e"
                 % (X, G, tot, cnt, cnt / tot, log(2 * X) / theta))
    L.append("")
    L.append("4) the measured variance against the CRT prediction (all q coprime to 30)")
    L.append("   band    Var[N]  Var[mu*]  Var[N-mu*]  predicted E[Var_model]  ratio   "
             "residual dispersion Var[N-mu*]/E[N]")
    for X in XS:
        z = np.load(os.path.join(OUT, "mom_scan_X%d.npz" % X))
        q, N, mh = z["q"], z["N"].astype(float), z["mu_hat"]
        mu = mh / S2
        muex = mu * S2 * 0.995            # mu_exact ~ 0.995 mu_hat (part F)
        disp = np.where((q % 6) == 5, DISP["short"], DISP["long"])
        pred = float((disp * muex).mean())
        vres = float((N - mu).var())
        L.append("   %6d %9.3f %9.3f %10.3f %22.3f %8.4f %12.4f"
                 % (X, N.var(), mu.var(), vres, pred, vres / pred, vres / N.mean()))
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(OUT, "mom_final.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
