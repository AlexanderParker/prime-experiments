"""R2.a.i.a.1.a.2 - the second moment over q.  Scan 1: the exact open-island count.

For EVERY integer q coprime to 30 in [X, 2X], with X = 1000, 2000, ..., 64000:

  d(q) = 2 * 6^-1 (mod q)                       the top gear's forward tooth arc
  I(q) = { i = 12 (mod 35) : 1 <= i < d(q) }    the one-class island set (N-I2), full arc
  island i is OPEN iff no gear g prime, 7 < g <= q, has
        q^2 = -6i (mod g)  or  q^2 = 2 - 6i (mod g)
  N(q) = # open islands.

Also records m(q) = |I(q)| and the mean-rate proxy for the first moment
  mu_hat(q) = m(q) * prod_{7 < g <= q} (1 - 2/g).

Writes results/mom_scan_X<X>.npz with columns q, d, m, N, mu_hat and a text summary.
Usage: uv run python research/anchor235/r50/mom_scan.py [--WORKERS 4]
"""
import argparse
import os
from math import isqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

XS = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
QTOP = 2 * XS[-1] + 10


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


G = {}


def init():
    if "gears" in G:
        return
    fl = sieve(QTOP)
    gears = np.array([p for p in range(11, QTOP + 1) if fl[p]], dtype=np.int64)
    G["gears"] = gears
    G["u"] = np.array([pow(6, -1, int(g)) for g in gears], dtype=np.int64)
    G["i35"] = np.array([pow(35, -1, int(g)) for g in gears], dtype=np.int64)
    # cumulative prod_{11 <= g <= gears[k]} (1 - 2/g)
    G["rate"] = np.cumprod(1.0 - 2.0 / gears)


def one_q(q):
    gears = G["gears"]
    ng = int(np.searchsorted(gears, q, side="right"))
    d = (2 * pow(6, -1, q)) % q
    if d < 13:
        return (q, d, 0, 0, 0.0)
    m = (d - 13) // 35 + 1                       # islands i = 12 + 35t, t = 0..m-1
    gl = gears[:ng]
    ul = G["u"][:ng]
    i35 = G["i35"][:ng]
    r = (q * q) % gl
    c1 = ((-r) * ul) % gl                        # i = -q^2 u   (upper member)
    c2 = ((2 - r) * ul) % gl                     # i = (2-q^2) u (lower member)
    t1 = ((c1 - 12) * i35) % gl
    t2 = ((c2 - 12) * i35) % gl
    struck = np.zeros(m, dtype=bool)
    k0 = int(np.searchsorted(gl, m, side="left"))   # gears < m need slices
    for j in range(k0):
        g = int(gl[j])
        a = int(t1[j])
        b = int(t2[j])
        struck[a::g] = True
        struck[b::g] = True
    if k0 < ng:
        for t in (t1[k0:], t2[k0:]):
            tt = t[t < m]
            if tt.size:
                struck[tt] = True
    N = int(m - int(struck.sum()))
    mu_hat = m * float(G["rate"][ng - 1])
    return (q, d, m, N, mu_hat)


def run_chunk(qs):
    init()
    return [one_q(int(q)) for q in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--WORKERS", type=int, default=4)
    args = ap.parse_args()
    from multiprocessing import Pool

    lines = []
    for X in XS:
        qs = [q for q in range(X, 2 * X + 1) if q % 2 and q % 3 and q % 5]
        chunks = [qs[i::args.WORKERS * 4] for i in range(args.WORKERS * 4)]
        with Pool(args.WORKERS) as pool:
            res = pool.map(run_chunk, chunks)
        rows = sorted([r for c in res for r in c])
        arr = np.array(rows, dtype=np.float64)
        np.savez_compressed(os.path.join(OUT, "mom_scan_X%d.npz" % X),
                            q=arr[:, 0].astype(np.int64), d=arr[:, 1].astype(np.int64),
                            m=arr[:, 2].astype(np.int64), N=arr[:, 3].astype(np.int64),
                            mu_hat=arr[:, 4])
        N = arr[:, 3]
        E = N.mean()
        V = N.var(ddof=0)
        fails = int((N == 0).sum())
        lines.append("X=%6d  |A|=%6d  E[N]=%10.4f  Var=%12.4f  Var/E=%8.3f  Var/E^2=%9.5f"
                     "  fails=%3d  frac=%.3e" %
                     (X, len(N), E, V, V / E, V / E ** 2, fails, fails / len(N)))
        print(lines[-1], flush=True)
    with open(os.path.join(OUT, "mom_scan.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
