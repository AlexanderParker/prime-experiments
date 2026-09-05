"""R2.a.i.a.1.b - accounting for the parent's factor-14 miss of the first-moment model.

For every prime q up to QMAX the exact per-island rates give

    p(i)      = prod_{7 < g < q} (1 - 2 chi_g(i)/(g-1))              the model's opening chance
    INDEP     = prod_i (1 - p(i))                                    the parent's C10 model
    INDEP-HL  = prod_i (1 - p(i)/C)  with C = 4 e^{-2 gamma}         the same after the classical
                                                                     Mertens-vs-Hardy-Littlewood
                                                                     correction at sifting level
                                                                     z = sqrt(x)

and both are summed over each band and compared with the observed number of real failures and
with the real open-island count.

Usage: uv run python research/anchor235/r42/sv_pred.py [--QMAX 6000]
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

C = 4 * np.exp(-2 * 0.5772156649015329)


def sieve_np(n):
    fl = np.ones(n + 1, dtype=bool)
    fl[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if fl[i]:
            fl[i * i:: i] = False
    return fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--QMAX", type=int, default=6000)
    a = ap.parse_args()
    fl = sieve_np(a.QMAX + 10)
    allp = np.flatnonzero(fl).astype(np.int64)
    gears = allp[allp > 7]
    qr = {}
    for g in gears:
        g = int(g)
        t = np.zeros(g, dtype=bool)
        t[(np.arange(1, g) ** 2) % g] = True
        qr[g] = t
    us = {int(g): pow(6, -1, int(g)) for g in gears}
    rows = []
    for q in allp[allp >= 11]:
        q = int(q)
        d = (2 * pow(6, -1, q)) % q
        if d < 2:
            continue
        isl = np.array([i for i in range(1, d) if i % 35 in (5, 10, 12, 17)], dtype=np.int64)
        m = len(isl)
        if m == 0:
            continue
        struck = np.zeros(m, dtype=bool)
        logp = np.zeros(m)
        qq = q * q
        for g in gears[gears < q]:
            g = int(g)
            u = us[g]
            r = qq % g
            b = (-r * u) % g
            aa = (b + 2 * u) % g
            mo = isl % g
            struck |= (mo == aa) | (mo == b)
            chi = qr[g][(-6 * isl) % g].astype(np.int64) + qr[g][(2 - 6 * isl) % g].astype(np.int64)
            logp += np.log1p(-2.0 * chi / (g - 1))
        p = np.exp(logp)
        rows.append(dict(q=q, d=int(d), m=int(m), real_open=int(m - struck.sum()),
                         real_fail=int(struck.all()),
                         model_open=float(p.sum()),
                         indep=float(np.prod(1 - p)),
                         indep_hl=float(np.prod(1 - p / C))))
    bands = [(11, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 6000), (6000, 20000)]
    lines = ["band            primes  observed   INDEP    INDEP-HL   real open  model open  ratio"]
    for lo, hi in bands:
        s = [r for r in rows if lo <= r["q"] < hi]
        if not s:
            continue
        ro = np.mean([r["real_open"] for r in s])
        mo = np.mean([r["model_open"] for r in s])
        lines.append("[%5d,%5d)  %5d   %6d  %8.3f  %9.3f   %8.3f  %10.3f  %6.4f"
                     % (lo, hi, len(s), sum(r["real_fail"] for r in s),
                        sum(r["indep"] for r in s), sum(r["indep_hl"] for r in s), ro, mo,
                        mo / ro if ro else float("nan")))
        print(lines[-1], flush=True)
    lines.append("C = 4 e^{-2 gamma} = %.5f" % C)
    print(lines[-1])
    open(os.path.join(OUT, "sv_pred.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(rows, open(os.path.join(OUT, "sv_pred.json"), "w"))


if __name__ == "__main__":
    main()
