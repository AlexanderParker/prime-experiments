"""R2.a.i.a.1.b - the minimum blocking set of the STRUCK islands, by kind of phase vector,
and the per-offset open probability that drives the walk length L.

Part 1 (MBS): at a fixed (d, gear set) take NS vectors of each kind and solve, exactly (HiGHS),
the minimum number of gears of (7, q0] that account for every island of [1, d) that is struck.

Part 2 (walk): the probability that offset i is open, i = 1..IMAX, for each kind, at the same
machines - the quantity that decides L.

Usage: uv run python research/anchor235/r42/sv_mbs.py [--NS 200] [--NW 200000]
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

Q0S = [491, 991, 1571]
KINDS = ("REAL", "LS", "RND")
IMAX = 40


def sieve_np(n):
    fl = np.ones(n + 1, dtype=bool)
    fl[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if fl[i]:
            fl[i * i:: i] = False
    return fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--NS", type=int, default=200)
    ap.add_argument("--NW", type=int, default=200000)
    a = ap.parse_args()
    fl = sieve_np(2000000)
    allp = np.flatnonzero(fl).astype(np.int64)
    pool = allp[allp > 12000]
    rng = np.random.default_rng(5150)
    lines, out = [], []
    for q0 in Q0S:
        d = (2 * pow(6, -1, q0)) % q0
        isl = [i for i in range(1, d) if i % 35 in (5, 10, 12, 17)]
        m = len(isl)
        gl = [int(g) for g in allp if 7 < g < q0]
        us = {g: pow(6, -1, g) for g in gl}
        g57 = [5, 7]
        lines.append("=== q0 %d  d %d  m %d  gears %d ===" % (q0, d, m, len(gl)))
        print(lines[-1], flush=True)
        for kind in KINDS:
            # --- part 1: MBS on NS vectors
            mbs, struckn = [], []
            for t in range(a.NS):
                q = int(pool[t]) if kind == "REAL" else None
                rows, cols = [], []
                covered = set()
                for gi, g in enumerate(gl):
                    u = us[g]
                    if kind == "REAL":
                        r = (q * q) % g
                    elif kind == "LS":
                        s = int(rng.integers(1, g))
                        r = (s * s) % g
                    else:
                        r = int(rng.integers(1, g))
                    b = (-r * u) % g
                    aa = (b + 2 * u) % g
                    for j, i in enumerate(isl):
                        if i % g == aa or i % g == b:
                            rows.append(j)
                            cols.append(gi)
                            covered.add(j)
                cv = sorted(covered)
                idx = {j: k for k, j in enumerate(cv)}
                A = csr_matrix((np.ones(len(rows)), ([idx[j] for j in rows], cols)),
                               shape=(len(cv), len(gl)))
                res = milp(c=np.ones(len(gl)), constraints=LinearConstraint(A, lb=1, ub=np.inf),
                           integrality=np.ones(len(gl)), bounds=Bounds(0, 1))
                mbs.append(int(round(res.fun)))
                struckn.append(len(cv))
            lines.append("   %-5s MBS mean %6.2f min %3d max %3d | struck islands mean %6.2f of %d"
                         % (kind, float(np.mean(mbs)), min(mbs), max(mbs), float(np.mean(struckn)), m))
            print(lines[-1], flush=True)
            out.append(dict(q0=q0, d=d, m=m, kind=kind, part="mbs", mbs_mean=float(np.mean(mbs)),
                            mbs_min=min(mbs), mbs_max=max(mbs), struck_mean=float(np.mean(struckn)), ns=a.NS))
            # --- part 2: per-offset open probability
            n = min(a.NW, len(pool)) if kind == "REAL" else a.NW
            W = IMAX + 1
            soff = np.zeros((n, W), dtype=bool)
            rws = np.arange(n)
            qsq = (pool[:n] ** 2) if kind == "REAL" else None
            for g in g57 + gl:
                u = pow(6, -1, g)
                if kind == "REAL":
                    r = qsq % g
                elif kind == "LS":
                    s = rng.integers(1, g, n)
                    r = (s * s) % g
                else:
                    r = rng.integers(1, g, n)
                b = (-r * u) % g
                aa = (b + 2 * u) % g
                for base in (aa, b):
                    k = 0
                    while True:
                        vals = base + k * g
                        sel = vals < W
                        if not sel.any():
                            break
                        soff[rws[sel], vals[sel]] = True
                        k += 1
            popen = 1.0 - soff.mean(axis=0)
            free = ~soff[:, 1:]
            has = free.any(axis=1)
            L = np.where(has, 1 + free.argmax(axis=1), W)
            lines.append("   %-5s P(offset open) i=1..20: %s" % (kind, " ".join("%.4f" % p for p in popen[1:21])))
            lines.append("   %-5s L (censored at %d): mean %.3f median %.1f  P(L<=10) %.4f"
                         % (kind, W, float(L.mean()), float(np.median(L)), float((L <= 10).mean())))
            print(lines[-2], flush=True)
            print(lines[-1], flush=True)
            out.append(dict(q0=q0, d=d, kind=kind, part="walk", popen=[float(x) for x in popen],
                            L_mean=float(L.mean()), L_med=float(np.median(L)), n=n))
    open(os.path.join(OUT, "sv_mbs.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(OUT, "sv_mbs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
