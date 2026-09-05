"""R2.a.i.a.1.b - the global integer, isolated: the deficit is a function of the SIFTING LEVEL.

Fix a gear set G = {primes 7 < g <= z} and a fixed offset window [1, D) with its island set.
For a prime q, island i is open iff q^2 + 6i - 2 and q^2 + 6i have no prime factor <= z.  The
model - which is exact if q were uniform modulo prod(G) - predicts

    E[#open] = sum_i prod_{g in G} (1 - 2 chi_g(i)/(g - 1)) .

The ratio model / real is measured for q in several ranges, i.e. at several values of the
sifting ratio s = log(q^2)/log(z).  s = 2 is the object's own configuration (machine {5..q},
columns just above q^2).

Usage: uv run python research/anchor235/r42/sv_s.py [--Z 5009] [--N 20000]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Z", type=int, default=5009)
    ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--QP", type=int, default=20000000)
    a = ap.parse_args()
    z = a.Z
    D = (2 * pow(6, -1, z)) % z
    fl = sieve_np(a.QP)
    allp = np.flatnonzero(fl).astype(np.int64)
    gl = allp[(allp > 7) & (allp <= z)]
    isl = np.array([i for i in range(1, D) if i % 35 in (5, 10, 12, 17)], dtype=np.int64)
    m = len(isl)
    logp = np.zeros(m)
    for g in gl:
        g = int(g)
        t = np.zeros(g, dtype=bool)
        t[(np.arange(1, g) ** 2) % g] = True
        chi = t[(-6 * isl) % g].astype(np.int64) + t[(2 - 6 * isl) % g].astype(np.int64)
        logp += np.log1p(-2.0 * chi / (g - 1))
    model = float(np.exp(logp).sum())
    lines = ["z = %d, window [1,%d), %d islands, %d gears; model E[#open] = %.4f" % (z, D, m, len(gl), model)]
    lines.append("   q range              s      primes   real open mean    model/real")
    print("\n".join(lines), flush=True)
    out = []
    for lo, hi in [(z, 3 * z), (10 * z, 30 * z), (100 * z, 300 * z), (1000 * z, 3000 * z)]:
        if hi > a.QP:
            break
        qs = allp[(allp >= lo) & (allp < hi)]
        if len(qs) > a.N:
            qs = qs[np.linspace(0, len(qs) - 1, a.N).astype(np.int64)]
        n = len(qs)
        qsq = qs.astype(np.int64) ** 2
        struck = np.zeros((n, m), dtype=bool)
        for g in gl:
            g = int(g)
            u = pow(6, -1, g)
            r = qsq % g
            b = (-r * u) % g
            aa = (b + 2 * u) % g
            mo = isl % g
            struck |= (mo[None, :] == aa[:, None]) | (mo[None, :] == b[:, None])
        ro = float((m - struck.sum(axis=1)).mean())
        s = 2 * np.log(np.sqrt(float(lo) * float(hi))) / np.log(z)
        lines.append("   [%10d,%10d) %6.3f %7d   %12.4f    %10.4f" % (lo, hi, s, n, ro, model / ro))
        print(lines[-1], flush=True)
        out.append(dict(lo=int(lo), hi=int(hi), s=s, n=n, real_open=ro, model=model))
    lines.append("4 e^{-2 gamma} = %.5f" % (4 * np.exp(-2 * 0.5772156649015329)))
    print(lines[-1])
    open(os.path.join(OUT, "sv_s.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(OUT, "sv_s.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
