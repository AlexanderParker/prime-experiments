"""W3 item 6 - the frontier the root statement needs, measured directly.

The window statement needs  K(d) > pi(sqrt(6 d)) - 3.  Write  Kneed(d) = pi(sqrt(6 d)) - 3.
The strongest thing an adversary with Kneed gears can do is bounded by two numbers, both measured
here exactly for the REAL separation and for random separations:

  B(d) = max budget:  sum over the Kneed CHEAPEST gears of  max_phase |S_g|   (the counting bound);
  C(d) = max coverage: the largest number of islands the Kneed cheapest gears can cover, one phase
         each (ILP, HiGHS-certified), and the total pairwise overlap T at that optimum.

A cover needs C(d) = m.  The elementary route from a PAIRWISE overlap bound to a contradiction is

     m = |union| = sum_j |S_j| - E,   E = sum_i (mult_i - 1),   T = sum_i C(mult_i, 2) ,
     E >= 2T / max_mult   (and E = T exactly when no island is struck three times),

so with multiplicities at most two a cover by K gears forces  T <= B - m: the adversary must hold
the TOTAL PAIRWISE OVERLAP below the counting excess.  This script measures how far below the
CRT-independent total  T_crt = sum_{j<k} 4m/(g_j g_k)  the adversary actually gets.

Usage: uv run python research/anchor235/r43/sep_frontier.py --DS 140,280,560,840,1120,1330 --nrand 10
"""
import argparse
import json
import os
import random
from math import isqrt, sqrt

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


def islands(d):
    return [i for i in range(1, d) if i % 35 in (5, 10, 12, 17)]


def u_for(g, fam, rng):
    if fam == "real":
        return pow(6, -1, g)
    return rng.randrange(1, g)


def phase_sets(g, u, isl, idx):
    v = pow(u, -1, g)
    h = (g - 1) // 2
    buck = {}
    for i in isl:
        buck.setdefault(i % g, []).append(idx[i])
    rs = set()
    for i in isl:
        for r in (((-i * v) % g), ((2 - i * v) % g)):
            if r and pow(r, h, g) == 1:
                rs.add(r)
    out = []
    for r in rs:
        s = frozenset(buck.get(((2 - r) * u) % g, []) + buck.get(((-r) * u) % g, []))
        if s:
            out.append(s)
    return out


def max_coverage(m, gearsets, tl):
    """One phase per gear; maximise the number of islands covered.  y_i <= sum of x over phases."""
    cols = []
    for g, S in gearsets:
        for s in S:
            cols.append((g, s))
    n = len(cols)
    N = n + m                       # x variables then y variables
    rows, cc, vv = [], [], []
    nr = 0
    # y_i - sum_{j: i in s_j} x_j <= 0
    for i in range(m):
        rows.append(nr)
        cc.append(n + i)
        vv.append(1.0)
        for j, (g, s) in enumerate(cols):
            if i in s:
                rows.append(nr)
                cc.append(j)
                vv.append(-1.0)
        nr += 1
    A1 = coo_matrix((vv, (rows, cc)), shape=(nr, N))
    cons = [LinearConstraint(A1, lb=-np.inf, ub=0)]
    bg = {}
    for j, (g, s) in enumerate(cols):
        bg.setdefault(g, []).append(j)
    rr, ccc, nr2 = [], [], 0
    for g, js in bg.items():
        for j in js:
            rr.append(nr2)
            ccc.append(j)
        nr2 += 1
    cons.append(LinearConstraint(coo_matrix((np.ones(len(rr)), (rr, ccc)), shape=(nr2, N)),
                                 lb=-np.inf, ub=1))
    c = np.zeros(N)
    c[n:] = -1.0
    out = milp(c=c, constraints=cons, integrality=np.ones(N), bounds=Bounds(0, 1),
               options={"time_limit": float(tl), "mip_rel_gap": 0.0, "presolve": True})
    if out.x is None:
        return None
    pick = [cols[j] for j in range(n) if out.x[j] > 0.5]
    cov = int(round(-out.fun))
    ub = -out.mip_dual_bound if out.mip_dual_bound is not None else None
    return cov, pick, (int(np.floor(ub + 1e-6)) if ub is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--DS", type=str, default="140,280,560,840,1120,1330")
    ap.add_argument("--nrand", type=int, default=10)
    ap.add_argument("--tl", type=float, default=600.0)
    ap.add_argument("--tag", type=str, default="fr")
    args = ap.parse_args()
    DS = [int(v) for v in args.DS.split(",")]
    FL = sieve(20000)
    PR = [p for p in range(2, 20001) if FL[p]]
    G = [p for p in PR if p >= 11]
    p = os.path.join(OUT, "sep_frontier_%s.jsonl" % args.tag)
    LOG = open(p, "w")
    for d in DS:
        isl = islands(d)
        m = len(isl)
        idx = {i: k for k, i in enumerate(isl)}
        q = sqrt(6 * d)
        Kneed = sum(1 for x in PR if x <= q) - 3
        gears = G[:Kneed]
        Tcrt = sum(4.0 * m / (gears[a] * gears[b])
                   for a in range(Kneed) for b in range(a + 1, Kneed))
        for fam, seed in [("real", 0)] + [("rand", 700 + k) for k in range(args.nrand)]:
            rng = random.Random(seed)
            gs = []
            B = 0
            for g in gears:
                S = phase_sets(g, u_for(g, fam, rng), isl, idx)
                gs.append((g, S))
                B += max((len(s) for s in S), default=0)
            r = max_coverage(m, gs, args.tl)
            if r is None:
                continue
            cov, pick, ub = r
            mult = {}
            for g, s in pick:
                for e in s:
                    mult[e] = mult.get(e, 0) + 1
            T = sum(v * (v - 1) // 2 for v in mult.values())
            E = sum(v - 1 for v in mult.values())
            rec = dict(d=d, m=m, fam=fam, seed=seed, q=round(q, 1), Kneed=Kneed,
                       gears=[gears[0], gears[-1]], budget=B, budget_m=B / m,
                       max_coverage=cov, cov_ub=ub, exact=(ub == cov),
                       uncovered=m - cov, T_at_opt=T, E_at_opt=E, T_crt=round(Tcrt, 2),
                       max_mult=max(mult.values()) if mult else 0,
                       X_needed=round((B - m) / (Kneed * (Kneed - 1) / 2), 4) if Kneed > 1 else 0)
            LOG.write(json.dumps(rec) + "\n")
            LOG.flush()
            print("d=%-5d m=%-4d Kneed=%-3d (11..%d) %-5s budget=%-4d (%.3f m) maxcov=%-4d%s "
                  "uncov=%-3d T=%-4d E=%-4d T_crt=%.1f maxmult=%d  X_needed=%.3f"
                  % (d, m, Kneed, gears[-1], fam, B, B / m, cov,
                     "" if ub == cov else "(ub %s)" % ub, m - cov, T, E, Tcrt,
                     rec["max_mult"], rec["X_needed"]), flush=True)
    LOG.close()
    print("written", p)


main()
