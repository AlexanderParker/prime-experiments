"""R2.a.i.a.1.a.2 - second moment over q.  Aggregate analysis of the scan.

Reads results/mom_scan_X*.npz and reports, per band X:

  E[N], Var[N], Var/E, Var/E^2 (raw Chebyshev), the actual failing fraction;
  the same conditioned on the arc (q = 5 mod 6 short, q = 1 mod 6 long);
  the model mean mu_hat = m * prod_{11<=g<=q}(1-2/g) and the s=2 corrected
  mu_star = mu_hat / (4 e^{-2 gamma});
  the variance decomposition Var[N] = Var[mu_star] + Var[N-mu_star] + 2 Cov;
  the normalised Chebyshev bound B(X) = mean_q (N - mu_star)^2 / mu_star^2;
  the same over q coprime to 210 only;
  the equidistributed core (the largest set of gears above 7 whose product is <= X).

Usage: uv run python research/anchor235/r50/mom_bound.py
"""
import os
from math import exp, log

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
XS = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
GAMMA = 0.5772156649015329
S2 = 4.0 * exp(-2.0 * GAMMA)          # 1.2618962 - the s = 2 opening handicap


def core(X):
    """largest initial run of gears above 7 whose product is <= X."""
    gs, P = [], 1
    for g in (11, 13, 17, 19, 23, 29, 31, 37, 41, 43):
        if P * g > X:
            break
        P *= g
        gs.append(g)
    return gs, P


def block(q, N, mu):
    E = N.mean()
    V = N.var()
    f = int((N == 0).sum())
    R = ((N - mu) ** 2).mean()
    B = (((N - mu) / mu) ** 2).mean()
    return E, V, f, len(N), R, B


def main():
    rows = []
    for X in XS:
        z = np.load(os.path.join(OUT, "mom_scan_X%d.npz" % X))
        q, d, m, N, mu_hat = z["q"], z["d"], z["m"], z["N"].astype(float), z["mu_hat"]
        mu = mu_hat / S2
        E, V, f, n, R, B = block(q, N, mu)
        short = (q % 6) == 5
        Es, Vs = N[short].mean(), N[short].var()
        El, Vl = N[~short].mean(), N[~short].var()
        c210 = (q % 7) != 0
        E2, V2, f2, n2, R2, B2 = block(q[c210], N[c210], mu[c210])
        vmu = mu.var()
        vres = (N - mu).var()
        cov = np.cov(mu, N - mu, ddof=0)[0, 1]
        gs, P = core(X)
        rows.append(dict(X=X, n=n, E=E, V=V, f=f, R=R, B=B, Es=Es, Vs=Vs, El=El, Vl=Vl,
                         Emu=mu.mean(), vmu=vmu, vres=vres, cov=cov,
                         n2=n2, E2=E2, V2=V2, f2=f2, B2=B2, core=gs, P=P,
                         mmean=m.mean()))

    L = []
    L.append("band   |A|      E[N]     Var      Var/E   Var/E^2   fails  frac       E[mu*]   E/E[mu*]")
    for r in rows:
        L.append("%6d %6d %9.4f %9.3f %7.3f %9.5f %5d  %.3e %8.4f %8.4f" %
                 (r["X"], r["n"], r["E"], r["V"], r["V"] / r["E"], r["V"] / r["E"] ** 2,
                  r["f"], r["f"] / r["n"], r["Emu"], r["E"] / r["Emu"]))
    L.append("")
    L.append("arc split            short arc (q=5 mod 6)        long arc (q=1 mod 6)")
    L.append("band      E_short   Var_short  V/E     E_long   Var_long   V/E")
    for r in rows:
        L.append("%6d %9.4f %10.3f %7.3f %9.4f %10.3f %7.3f" %
                 (r["X"], r["Es"], r["Vs"], r["Vs"] / r["Es"], r["El"], r["Vl"], r["Vl"] / r["El"]))
    L.append("")
    L.append("variance decomposition  Var[N] = Var[mu*] + Var[N-mu*] + 2 Cov")
    L.append("band      Var[N]   Var[mu*]  Var[N-mu*]  2Cov    resid/E[mu*]")
    for r in rows:
        L.append("%6d %9.3f %9.3f %10.3f %8.3f %8.4f" %
                 (r["X"], r["V"], r["vmu"], r["vres"], 2 * r["cov"], r["R"] / r["Emu"]))
    L.append("")
    L.append("the normalised bound B(X) = mean_q (N-mu*)^2/mu*^2")
    L.append("band        B(X)     B ratio   raw Var/E^2   actual frac   B * (X/(ln X)^2)")
    prev = None
    for r in rows:
        rat = (r["B"] / prev) if prev else float("nan")
        prev = r["B"]
        L.append("%6d %11.5f %9.4f %13.5f %13.3e %12.4f" %
                 (r["X"], r["B"], rat, r["V"] / r["E"] ** 2, r["f"] / r["n"],
                  r["B"] * r["X"] / log(r["X"]) ** 2))
    L.append("")
    L.append("restricted to q coprime to 210 (gears 5 and 7 provably never strike an island)")
    L.append("band     |A'|      E[N]     Var      fails    B(X)")
    for r in rows:
        L.append("%6d %6d %9.4f %9.3f %6d %10.5f" %
                 (r["X"], r["n2"], r["E2"], r["V2"], r["f2"], r["B2"]))
    L.append("")
    L.append("the equidistributed core: gears above 7 whose product is at most X")
    L.append("band    core gears                  product   gears above 7 up to 2X")
    from sympy import primepi
    for r in rows:
        L.append("%6d  %-26s %8d   %6d" %
                 (r["X"], str(r["core"]), r["P"], int(primepi(2 * r["X"])) - 4))
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(OUT, "mom_bound.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
