"""R2.a.i.a.1.b - the global integer measured: the real open-island count against the model's.

For a prime q the real machine {5..q} sifts the columns of (q^2, q^2 + 6d) up to q, i.e. up to the
SQUARE ROOT of the numbers being sifted.  The independent-gear model of the phase vector predicts

    E[#open islands] = sum_{i island} prod_{7 < g <= q} (1 - 2 chi_g(i)/(g - 1))     (exact rates)

while the real count is the number of island columns that are actually twin prime pairs.  This
script computes both, exactly, for a sample of primes per band, plus the Hardy-Littlewood
prediction 12 C_2 / (ln q^2)^2 divided by the island's own local factors at 5 and 7, and reports
the ratio model / real against the classical constant 4 e^{-2 gamma} = 1.26190.

Usage: uv run python research/anchor235/r42/sv_hl.py [--PER 20]
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

BANDS = [(200, 400), (400, 900), (900, 2000), (2000, 5000), (5000, 12000),
         (12000, 30000), (30000, 80000), (80000, 200000)]
C2 = 0.6601618158468695
GAMMA = 0.5772156649015329


def sieve_np(n):
    fl = np.ones(n + 1, dtype=bool)
    fl[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if fl[i]:
            fl[i * i:: i] = False
    return fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--PER", type=int, default=20)
    ap.add_argument("--QMAX", type=int, default=200000)
    a = ap.parse_args()
    fl = sieve_np(a.QMAX + 10)
    allp = np.flatnonzero(fl).astype(np.int64)
    lines, out = [], []
    lines.append("  band          q      d      m  real open  model open  HL open   model/real  HL/real")
    for lo, hi in BANDS:
        cand = allp[(allp >= lo) & (allp < hi)]
        if len(cand) > a.PER:
            cand = cand[np.linspace(0, len(cand) - 1, a.PER).astype(np.int64)]
        R = M = H = 0.0
        nq = 0
        for q in cand:
            q = int(q)
            d = (2 * pow(6, -1, q)) % q
            isl = np.array([i for i in range(1, d) if i % 35 in (5, 10, 12, 17)], dtype=np.int64)
            m = len(isl)
            if m == 0:
                continue
            gl = allp[(allp > 7) & (allp < q)]
            struck = np.zeros(m, dtype=bool)
            logp = np.zeros(m)
            qq = q * q
            for g in gl:
                g = int(g)
                u = pow(6, -1, g)
                r = qq % g
                b = (-r * u) % g
                aa = (b + 2 * u) % g
                mo = isl % g
                struck |= (mo == aa) | (mo == b)
                t = np.zeros(g, dtype=bool)
                t[(np.arange(1, g) ** 2) % g] = True
                chi = t[(-6 * isl) % g].astype(np.int64) + t[(2 - 6 * isl) % g].astype(np.int64)
                logp += np.log1p(-2.0 * chi / (g - 1))
            ro = int(m - struck.sum())
            mo_ = float(np.exp(logp).sum())
            # Hardy-Littlewood for the island columns: twin density at q^2, with the island's
            # own local factors at 5 and 7 divided out (an island avoids both by construction)
            hl = m * 12.0 * C2 / (np.log(float(qq)) ** 2) / ((3.0 / 5.0) * (5.0 / 7.0))
            R += ro
            M += mo_
            H += hl
            nq += 1
            out.append(dict(q=q, d=int(d), m=int(m), real_open=ro, model_open=mo_, hl_open=hl))
        lines.append("[%6d,%6d) %6d %6s %6s  %9.2f  %10.2f %8.2f   %8.4f  %7.4f"
                     % (lo, hi, nq, "-", "-", R / nq, M / nq, H / nq, M / R if R else float("nan"),
                        H / R if R else float("nan")))
        print(lines[-1], flush=True)
    lines.append("4 e^{-2 gamma} = %.5f" % (4 * np.exp(-2 * GAMMA)))
    print(lines[-1])
    open(os.path.join(OUT, "sv_hl.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(OUT, "sv_hl.json"), "w"))


if __name__ == "__main__":
    main()
