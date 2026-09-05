"""W3 - K(d) under different separations, one parameterisation for every family.

A gear is a prime g > 7 with a nonzero u_g mod g.  Its two classes at phase r (r a nonzero
quadratic residue mod g, i.e. r = q^2 mod g for the real machine) are

        (2 - r) u_g   and   (-r) u_g   (mod g),

so the separation is s_g = 2 u_g and the phase a_g = -r u_g runs over the coset -u_g QR(g).
u_g = 6^{-1} is the real machine (s_g = 3^{-1}, the same rational one third at every gear).
Everything else - the one-phase-per-gear rule, the reachable-phase coset, the island set, the gear
pool - is identical across families, so the ONLY thing that varies is the separation.

Families:  real | coh:c/r  (s_g = c r^{-1}) | rand:SEED | free (any two classes, parent's K_free).

Gear pool: 11 <= g <= 3d+2 plus one generic singleton per island (r41's complete play list for the
real separation; a RESTRICTION for other separations, which can only raise K - the conservative
direction).  --gmul X uses 11 <= g <= X*d + 2 instead.

ILP: binary per (gear, phase-set) and per island singleton; cover every island; at most one phase
per gear.  Only same-gear dominance.  HiGHS, mip_rel_gap 0, so "exact" = dual bound meets incumbent.

Usage:
  uv run python research/anchor235/r43/sep_cover.py --DS 140,280 --fams real,free,coh:2/5 --tag x
  uv run python research/anchor235/r43/sep_cover.py --DS 140 --rand 30 --tag r140
"""
import argparse
import json
import os
import random
import time
from math import isqrt

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
    """u_g for a gear; separation s_g = 2 u_g.  None = gear unusable in this family."""
    if fam == "real":
        return pow(6, -1, g)
    if fam.startswith("coh:"):
        c, r = fam[4:].split("/")
        c, r = int(c), int(r)
        if g == r or c % g == 0:
            return None
        return (c * pow(r, -1, g) * pow(2, -1, g)) % g
    if fam == "rand":
        return rng.randrange(1, g)
    raise ValueError(fam)


def gear_sets(g, u, buck, isl_idx):
    """Sets of islands a gear can strike, one per reachable phase (size >= 2 kept)."""
    v = pow(u, -1, g)          # class1 = -r u = i  <=>  r = -i v ;  class2 = (2-r) u = i <=> r = 2 - i v
    h = (g - 1) // 2
    rs = set()
    for i in isl_idx:
        for r in (((-i * v) % g), ((2 - i * v) % g)):
            if r and pow(r, h, g) == 1:
                rs.add(r)
    out = set()
    for r in rs:
        s = buck.get(((2 - r) * u) % g, []) + buck.get(((-r) * u) % g, [])
        if len(s) >= 2:
            out.add(frozenset(s))
    return out


def free_sets(buck):
    ks = sorted(buck)
    out = set()
    big = [k for k in ks if len(buck[k]) >= 2]
    for k in big:
        if len(buck[k]) >= 3:
            out.add(frozenset(buck[k]))
        for k2 in ks:
            if k2 == k:
                continue
            s = frozenset(buck[k] + buck[k2])
            if len(s) >= 3:
                out.add(s)
    return out


def maximal_only(S):
    L = sorted(S, key=lambda s: -len(s))
    keep = []
    for s in L:
        if not any(s < t for t in keep):
            keep.append(s)
    return keep


def build(d, PR, fam, rng, gmul=3.0):
    isl = islands(d)
    m = len(isl)
    idx = {i: k for k, i in enumerate(isl)}
    free = fam == "free"
    GMAX = d if free else int(gmul * d) + 2
    cand = []
    maxsize = []
    for g in PR:
        if g > GMAX:
            break
        buck = {}
        for i in isl:
            buck.setdefault(i % g, []).append(idx[i])
        if free:
            S = free_sets(buck)
        else:
            u = u_for(g, fam, rng)
            S = set() if u is None else gear_sets(g, u, buck, isl)
        if not S:
            maxsize.append(1)
            continue
        maxsize.append(max(len(s) for s in S))
        for s in maximal_only(S):
            cand.append((g, s))
    if free:
        for a in range(m):
            for b in range(a + 1, m):
                cand.append((0, frozenset((a, b))))
    for k in range(m):
        cand.append((0, frozenset((k,))))
    return isl, m, cand, maxsize


def solve(m, cand, tl):
    n = len(cand)
    rows, cols = [], []
    for j, (g, s) in enumerate(cand):
        for e in s:
            rows.append(e)
            cols.append(j)
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, n))
    cons = [LinearConstraint(A, lb=1, ub=np.inf)]
    bygear = {}
    for j, (g, s) in enumerate(cand):
        if g > 0:
            bygear.setdefault(g, []).append(j)
    rr, cc, nr = [], [], 0
    for g, js in bygear.items():
        if len(js) > 1:
            for j in js:
                rr.append(nr)
                cc.append(j)
            nr += 1
    if nr:
        cons.append(LinearConstraint(coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(nr, n)),
                                     lb=-np.inf, ub=1))
    out = milp(c=np.ones(n), constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": float(tl), "mip_rel_gap": 0.0, "presolve": True})
    if out.x is None:
        return None, -1
    pick = [j for j in range(n) if out.x[j] > 0.5]
    lb = out.mip_dual_bound
    return pick, (int(np.ceil(lb - 1e-6)) if lb is not None else -1)


def counting_lb(maxsize, m):
    tot, clb = 0, 0
    for c in sorted(maxsize, reverse=True):
        if tot >= m:
            break
        tot += c
        clb += 1
    return clb


def run_one(d, PR, fam, seed, tl, gmul):
    rng = random.Random(seed)
    t0 = time.time()
    isl, m, cand, maxsize = build(d, PR, fam, rng, gmul=gmul)
    pick, lb = solve(m, cand, tl)
    if pick is None:
        return dict(d=d, fam=fam, seed=seed, K=None, lb=-1, secs=time.time() - t0)
    K = len(pick)
    gears = sorted(cand[j][0] for j in pick if cand[j][0] > 0)
    budget = sum(len(cand[j][1]) for j in pick)
    return dict(d=d, fam=fam, seed=seed, m=m, ncand=len(cand), K=K, lb=lb,
                exact=bool(lb >= K), countlb=counting_lb(maxsize, m),
                budget=budget, budget_m=budget / m, gears=gears,
                sizes=sorted((len(cand[j][1]) for j in pick), reverse=True),
                secs=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--DS", type=str, default="140,280")
    ap.add_argument("--fams", type=str, default="real")
    ap.add_argument("--rand", type=int, default=0, help="number of random-separation draws per d")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--tl", type=float, default=900.0)
    ap.add_argument("--gmul", type=float, default=3.0)
    ap.add_argument("--tag", type=str, default="cov")
    args = ap.parse_args()
    DS = [int(v) for v in args.DS.split(",")]
    fams = [f for f in args.fams.split(",") if f]
    NMAX = int(args.gmul * max(DS)) + 10
    FL = sieve(NMAX)
    PR = [p for p in range(11, NMAX + 1) if FL[p]]
    path = os.path.join(OUT, "sep_%s.jsonl" % args.tag)
    LOG = open(path, "w")
    jobs = []
    for d in DS:
        for f in fams:
            jobs.append((d, f, 0))
        for k in range(args.rand):
            jobs.append((d, "rand", args.seed0 + k))
    for d, f, sd in jobs:
        rec = run_one(d, PR, f, sd, args.tl, args.gmul)
        LOG.write(json.dumps(rec) + "\n")
        LOG.flush()
        print("d=%-5d %-10s seed=%-5d K=%-4s lb=%-4s %s  budget/m=%.3f  %ss"
              % (d, f, sd, rec.get("K"), rec.get("lb"),
                 "EXACT" if rec.get("exact") else "ub",
                 rec.get("budget_m", 0.0), rec.get("secs")), flush=True)
    LOG.close()
    print("written", path)


main()
