"""W3 item 5 - gear-by-gear anatomy of the optimal covers, per separation family.

For each family and arc: solve the same ILP as sep_cover.py, then report
  - islands per gear (the size of each chosen strike set),
  - the multiset of coverage multiplicities (islands struck once / twice / 3+),
  - the pairwise overlap matrix of the chosen sets: which pairs overlap most, the total
    pairwise overlap sum, and the CHAIN quantity  sum_j max_{k<j} |S_j ∩ S_k|  (the quantity a
    pairwise overlap lower bound would control: |union| <= sum_j |S_j| - chain),
  - the counting comparison  sum_j |S_j| / m.

Usage: uv run python research/anchor235/r43/sep_anatomy.py --DS 280,560,1120 --nrand 5 --tag a
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


def gear_sets(g, u, buck, isl):
    v = pow(u, -1, g)
    h = (g - 1) // 2
    rs = set()
    for i in isl:
        for r in (((-i * v) % g), ((2 - i * v) % g)):
            if r and pow(r, h, g) == 1:
                rs.add(r)
    out = set()
    for r in rs:
        s = buck.get(((2 - r) * u) % g, []) + buck.get(((-r) * u) % g, [])
        if len(s) >= 2:
            out.add(frozenset(s))
    return out


def maximal_only(S):
    L = sorted(S, key=lambda s: -len(s))
    keep = []
    for s in L:
        if not any(s < t for t in keep):
            keep.append(s)
    return keep


def build(d, PR, fam, rng):
    isl = islands(d)
    m = len(isl)
    idx = {i: k for k, i in enumerate(isl)}
    cand = []
    for g in PR:
        if g > 3 * d + 2:
            break
        buck = {}
        for i in isl:
            buck.setdefault(i % g, []).append(idx[i])
        u = u_for(g, fam, rng)
        if u is None:
            continue
        for s in maximal_only(gear_sets(g, u, buck, isl)):
            cand.append((g, s))
    for k in range(m):
        cand.append((0, frozenset((k,))))
    return isl, m, cand


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


def anatomy(d, PR, fam, seed, tl):
    rng = random.Random(seed)
    t0 = time.time()
    isl, m, cand = build(d, PR, fam, rng)
    pick, lb = solve(m, cand, tl)
    if pick is None:
        return None
    sets = [cand[j] for j in pick]
    sets.sort(key=lambda t: -len(t[1]))
    K = len(sets)
    mult = {}
    for g, s in sets:
        for e in s:
            mult[e] = mult.get(e, 0) + 1
    cover_mult = {1: 0, 2: 0, 3: 0}
    for e, c in mult.items():
        cover_mult[min(c, 3)] += 1
    ov = []
    tot_ov = 0
    for a in range(K):
        for b in range(a + 1, K):
            o = len(sets[a][1] & sets[b][1])
            tot_ov += o
            if o:
                ov.append((sets[a][0], sets[b][0], o))
    ov.sort(key=lambda t: -t[2])
    chain = 0
    for j in range(1, K):
        chain += max(len(sets[j][1] & sets[k][1]) for k in range(j))
    budget = sum(len(s) for _, s in sets)
    npos = sum(1 for a in range(K) for b in range(a + 1, K) if sets[a][1] & sets[b][1])
    return dict(d=d, m=m, fam=fam, seed=seed, K=K, lb=lb, exact=bool(lb >= K),
                gears=[g for g, _ in sets], sizes=[len(s) for _, s in sets],
                budget=budget, budget_m=budget / m,
                struck_once=cover_mult[1], struck_twice=cover_mult[2], struck_3plus=cover_mult[3],
                total_pair_overlap=tot_ov, pairs_with_overlap=npos,
                pairs_total=K * (K - 1) // 2, chain=chain, top_overlaps=ov[:6],
                secs=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--DS", type=str, default="280,560")
    ap.add_argument("--fams", type=str, default="real,coh:2/5,coh:2/7,coh:2/11")
    ap.add_argument("--nrand", type=int, default=5)
    ap.add_argument("--tl", type=float, default=900.0)
    ap.add_argument("--tag", type=str, default="anat")
    args = ap.parse_args()
    DS = [int(v) for v in args.DS.split(",")]
    FL = sieve(3 * max(DS) + 10)
    PR = [p for p in range(11, 3 * max(DS) + 3) if FL[p]]
    p = os.path.join(OUT, "sep_anat_%s.jsonl" % args.tag)
    LOG = open(p, "w")
    for d in DS:
        jobs = [(f, 0) for f in args.fams.split(",") if f] + \
               [("rand", 5000 + k) for k in range(args.nrand)]
        for fam, sd in jobs:
            r = anatomy(d, PR, fam, sd, args.tl)
            if r is None:
                continue
            LOG.write(json.dumps(r) + "\n")
            LOG.flush()
            print("d=%-5d %-9s K=%-3d %s budget/m=%.3f  1x/2x/3x=%d/%d/%d  "
                  "pairOv=%d over %d/%d pairs  chain=%d  %ss"
                  % (d, fam, r["K"], "EXACT" if r["exact"] else "ub", r["budget_m"],
                     r["struck_once"], r["struck_twice"], r["struck_3plus"],
                     r["total_pair_overlap"], r["pairs_with_overlap"], r["pairs_total"],
                     r["chain"], r["secs"]), flush=True)
            print("     gears %s" % (r["gears"],), flush=True)
    LOG.close()
    print("written", p)


main()
