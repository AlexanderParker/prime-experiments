"""R2.a.i.a.1.b item S9 - how rare is a cover among ALL phase vectors, as a function of the arc.

A ladder of short-arc primes q0 (so d = (q0+1)/3 runs through a chosen list).  At each arc the
failure rate of the island witness is estimated for locally-square (LS), unrestricted (RND) and
real (REAL, q^2 mod g for primes q) phase vectors, with the top gear inert in every kind (as in
the real walk).  Adaptive sample size: batches until FAILTARGET failures or NMAX draws.

Usage: uv run python research/anchor235/r42/sv_rate.py
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

DTARGETS = [60, 130, 200, 330, 500, 670, 800, 950, 1100]


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
    ap.add_argument("--NMAX", type=int, default=3000000)
    ap.add_argument("--BATCH", type=int, default=200000)
    ap.add_argument("--FAILTARGET", type=int, default=300)
    ap.add_argument("--QP", type=int, default=10000000)
    a = ap.parse_args()
    fl = sieve_np(a.QP)
    allp = np.flatnonzero(fl).astype(np.int64)
    realpool = allp[allp > 12000]
    lines, out = [], []
    lines.append("  d    q0    m   gears |  kind   draws     fails      rate        95%% CI          INDEP")
    for dt in DTARGETS:
        q0 = None
        for q in range(3 * dt - 1, 3 * dt + 400, 6):
            if q <= a.QP and fl[q] and q % 6 == 5:
                q0 = q
                break
        d = (2 * pow(6, -1, q0)) % q0
        isl = np.array([i for i in range(1, d) if i % 35 in (5, 10, 12, 17)], dtype=np.int64)
        m = len(isl)
        posl = np.full(d, -1, dtype=np.int64)
        posl[isl] = np.arange(m)
        gl = [int(g) for g in allp if 7 < g < q0]
        us = {g: pow(6, -1, g) for g in gl}
        # independent first-moment reference
        logp = np.zeros(m)
        for g in gl:
            t = np.zeros(g, dtype=bool)
            t[(np.arange(1, g) ** 2) % g] = True
            chi = t[(-6 * isl) % g].astype(np.int64) + t[(2 - 6 * isl) % g].astype(np.int64)
            logp += np.log1p(-2.0 * chi / (g - 1))
        indep = float(np.prod(1.0 - np.exp(logp)))
        for kind in ("LS", "RND", "REAL"):
            rng = np.random.default_rng(9090 + dt)
            draws = fails = 0
            nmax = min(a.NMAX, len(realpool)) if kind == "REAL" else a.NMAX
            while draws < nmax and fails < a.FAILTARGET:
                n = min(a.BATCH, nmax - draws)
                struck = np.zeros((n, m), dtype=bool)
                rws = np.arange(n)
                qsq = (realpool[draws:draws + n] ** 2) if kind == "REAL" else None
                for g in gl:
                    u = us[g]
                    if kind == "REAL":
                        r = qsq % g
                    elif kind == "LS":
                        s = rng.integers(1, g, n)
                        r = (s * s) % g
                    else:
                        r = rng.integers(1, g, n)
                    b = (-r * u) % g
                    aa = (b + 2 * u) % g
                    mark(struck, posl, aa, g, d, rws)
                    mark(struck, posl, b, g, d, rws)
                nopen = m - struck.sum(axis=1)
                fails += int((nopen == 0).sum())
                draws += n
            rate = fails / draws
            lo = max(0.0, rate - 1.96 * (rate * (1 - rate) / draws) ** 0.5)
            hi = rate + 1.96 * (rate * (1 - rate) / draws) ** 0.5
            lines.append("%5d %5d %4d %6d | %-5s %9d %6d   %.3e  [%.2e, %.2e]   %.3e"
                         % (d, q0, m, len(gl), kind, draws, fails, rate, lo, hi, indep))
            print(lines[-1], flush=True)
            out.append(dict(d=int(d), q0=int(q0), m=int(m), kind=kind, draws=draws, fails=fails,
                            rate=rate, indep=indep))
    open(os.path.join(OUT, "sv_rate.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(OUT, "sv_rate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
