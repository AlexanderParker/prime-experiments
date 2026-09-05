"""R2.a.i.a.1.b item S8 - does the locally-square model predict the REAL object's own failures?

For every prime q in [11, QMAX] the real object is its own machine {5..q} and its own arc
d = 2*6^-1 mod q.  Here, for each such q:

  * REAL     : the actual outcome (does some island of [1,d) survive the real phases q^2 mod g)
  * LSI      : the failure rate over N locally-square phase vectors with the top gear inert
               (exactly the real machine's shape: gear q has q^2 = 0 mod q and strikes nothing
               in [1,d)), gears 5 and 7 barred from the islands as always
  * RNDI     : the same with unrestricted phases
  * INDEP    : the parent's first-moment model, prod_i (1 - p(i)) with the exact rates
               p(i) = prod_g (1 - 2 chi_g(i)/(g-1))

and the predicted number of failures per band is compared with the observed number.

Usage: uv run python research/anchor235/r42/sv_object.py [--QMAX 6000] [--N 4000]
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def sieve_np(n):
    fl = np.ones(n + 1, dtype=bool)
    fl[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if fl[i]:
            fl[i * i:: i] = False
    return fl


def mark(struck, posl, base, g, d, rows):
    k = 0
    while True:
        vals = base + k * g
        sel = vals < d
        if not sel.any():
            return
        v = vals[sel]
        j = posl[v]
        ok = j >= 0
        if ok.any():
            struck[rows[sel][ok], j[ok]] = True
        k += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--QMAX", type=int, default=6000)
    ap.add_argument("--N", type=int, default=4000)
    a = ap.parse_args()
    fl = sieve_np(a.QMAX)
    allp = np.flatnonzero(fl).astype(np.int64)
    gears = allp[allp >= 11]
    us = {int(g): pow(6, -1, int(g)) for g in gears}
    isqr = {}
    for g in gears:
        g = int(g)
        t = np.zeros(g, dtype=bool)
        t[(np.arange(1, g) ** 2) % g] = True
        isqr[g] = t
    rng = np.random.default_rng(777)
    rows_out = []
    lines = []
    for q in allp[allp >= 11]:
        q = int(q)
        d = (2 * pow(6, -1, q)) % q
        if d < 2:
            continue
        isl = np.array([i for i in range(1, d) if i % 35 in (5, 10, 12, 17)], dtype=np.int64)
        m = len(isl)
        if m == 0:
            continue
        posl = np.full(d, -1, dtype=np.int64)
        posl[isl] = np.arange(m)
        gl = gears[gears < q]          # top gear q is inert on [1,d)
        # real outcome
        sr = np.zeros((1, m), dtype=bool)
        r1 = np.arange(1)
        lam = np.zeros(m)              # depth, exact rates, for the INDEP model
        logp = np.zeros(m)
        for g in gl:
            g = int(g)
            u = us[g]
            r = (q * q) % g
            b = np.array([(-r * u) % g])
            aa = np.array([(b[0] + 2 * u) % g])
            mark(sr, posl, aa, g, d, r1)
            mark(sr, posl, b, g, d, r1)
            chi = isqr[g][(-6 * isl) % g].astype(np.int64) + isqr[g][(2 - 6 * isl) % g].astype(np.int64)
            rate = 2.0 * chi / (g - 1)
            lam += rate
            logp += np.log1p(-rate)
        real_open = int(m - sr.sum())
        pfree = np.exp(logp)
        indep = float(np.prod(1.0 - pfree))
        # simulations
        sim = {}
        for kind in ("LSI", "RNDI"):
            n = a.N
            struck = np.zeros((n, m), dtype=bool)
            rws = np.arange(n)
            for g in gl:
                g = int(g)
                u = us[g]
                s = rng.integers(1, g, n)
                r = (s * s) % g if kind == "LSI" else s
                b = (-r * u) % g
                aa = (b + 2 * u) % g
                mark(struck, posl, aa, g, d, rws)
                mark(struck, posl, b, g, d, rws)
            nopen = m - struck.sum(axis=1)
            sim[kind] = (float((nopen == 0).mean()), float(nopen.mean()))
        rows_out.append(dict(q=q, d=int(d), m=int(m), real_open=real_open,
                             real_fail=int(real_open == 0),
                             lsi=sim["LSI"][0], lsi_open=sim["LSI"][1],
                             rndi=sim["RNDI"][0], rndi_open=sim["RNDI"][1],
                             indep=indep, indep_open=float(pfree.sum())))
        if real_open == 0:
            lines.append("REAL FAILURE q=%d d=%d m=%d  P(fail) LSI %.4g RNDI %.4g INDEP %.4g"
                         % (q, d, m, sim["LSI"][0], sim["RNDI"][0], indep))
            print(lines[-1], flush=True)
    # band summary
    bands = [(11, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 6000)]
    lines.append("")
    lines.append("band            primes  observed   LSI pred   RNDI pred  INDEP pred   open(real) open(LSI) open(INDEP)")
    for lo, hi in bands:
        sel = [r for r in rows_out if lo <= r["q"] < hi]
        if not sel:
            continue
        lines.append("[%5d,%5d)  %5d   %6d   %9.3f  %9.3f  %10.3f   %8.3f %8.3f %8.3f"
                     % (lo, hi, len(sel), sum(r["real_fail"] for r in sel),
                        sum(r["lsi"] for r in sel), sum(r["rndi"] for r in sel),
                        sum(r["indep"] for r in sel),
                        np.mean([r["real_open"] for r in sel]),
                        np.mean([r["lsi_open"] for r in sel]),
                        np.mean([r["indep_open"] for r in sel])))
        print(lines[-1], flush=True)
    open(os.path.join(OUT, "sv_object.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(rows_out, open(os.path.join(OUT, "sv_object.json"), "w"))


if __name__ == "__main__":
    main()
