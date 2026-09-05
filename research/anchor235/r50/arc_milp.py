"""Branch 5d.ii.i.a, item 1: A(K) by exact MILP (HiGHS through scipy.optimize.milp).

Feasibility model at level L (columns 0..L-1), over the type-reduced item list of
arc_core (so the model quantifies over ALL primes >= 5, not a truncated pool):

    binary x[i,o] = "one gear of item type i is used at option o"
    (o ranges over the realisable column-subsets of that type: the g phases of a
     concrete gear, the pairs {j, j+a} and the legal singletons of a domino of arc a,
     the single columns of the 'arc >= L' type)

    sum_o x[i,o] <= mult_i        (a type is offered by mult_i primes)
    sum_{i,o} x[i,o] <= K         (K gears)
    sum_{(i,o) : c in mask(i,o)} x[i,o] >= 1   for every column c

Feasible iff K gears can block L consecutive columns.  A(K) = the least infeasible L,
and HiGHS's infeasibility proof is the certificate.

Usage: uv run python research/anchor235/r50/arc_milp.py K L0 [LMAX]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import LinearConstraint, milp, Bounds
from scipy.sparse import csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arc_core import Level, masks_for, RESULTS  # noqa: E402


def model(L, K):
    lv = Level(L, K)
    cols, types = [], []
    for idx, (kind, key, m) in enumerate(lv.items):
        for msk in masks_for(kind, key, L):
            if msk:
                cols.append(msk)
                types.append(idx)
    n = len(cols)
    ntypes = len(lv.items)
    rows, colsA, vals = [], [], []
    # coverage rows 0..L-1  (>= 1)
    for j, msk in enumerate(cols):
        mm = msk
        while mm:
            b = mm & -mm
            rows.append(b.bit_length() - 1)
            colsA.append(j)
            vals.append(1.0)
            mm ^= b
    # multiplicity rows L..L+ntypes-1  (<= mult)
    for j, t in enumerate(types):
        rows.append(L + t)
        colsA.append(j)
        vals.append(1.0)
    # cardinality row L+ntypes  (<= K)
    for j in range(n):
        rows.append(L + ntypes)
        colsA.append(j)
        vals.append(1.0)
    A = csr_matrix((vals, (rows, colsA)), shape=(L + ntypes + 1, n))
    lb = np.concatenate([np.ones(L), np.full(ntypes, -np.inf), [-np.inf]])
    ub = np.concatenate([np.full(L, np.inf),
                         np.array([min(m, K) for _k, _key, m in lv.items],
                                  dtype=float), [K]])
    return A, lb, ub, n, lv


def feasible(L, K, timeout=None):
    A, lb, ub, n, lv = model(L, K)
    res = milp(c=np.zeros(n), constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": timeout} if timeout else None)
    sol = None
    if res.success and res.x is not None:
        pick = [j for j in range(n) if res.x[j] > 0.5]
        cols, types = [], []
        for idx, (kind, key, m) in enumerate(lv.items):
            for msk in masks_for(kind, key, L):
                if msk:
                    cols.append(msk)
                    types.append(idx)
        sol = [(lv.items[types[j]][0], lv.items[types[j]][1],
                [b for b in range(L) if cols[j] >> b & 1]) for j in pick]
    return res.success, res.status, sol, n


def main():
    K = int(sys.argv[1])
    L = int(sys.argv[2])
    LMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    os.makedirs(RESULTS, exist_ok=True)
    out = open(os.path.join(RESULTS, f"arc_milp_K{K}.txt"), "w")
    while L <= LMAX:
        t = time.time()
        ok, status, sol, nvar = feasible(L, K)
        line = (f"K={K} L={L}: {'cover' if ok else 'NO COVER'}  "
                f"(status {status}, {nvar} binaries, {time.time()-t:.1f}s)")
        print(line, flush=True)
        out.write(line + "\n")
        if ok:
            out.write("   witness: " + json.dumps(
                [[k, key, cs] for k, key, cs in sol]) + "\n")
        else:
            out.write(f"*** A({K}) = {L}\n")
        out.flush()
        if not ok:
            break
        L += 1
    out.close()


if __name__ == "__main__":
    main()
