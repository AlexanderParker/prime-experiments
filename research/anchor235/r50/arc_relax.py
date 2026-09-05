"""Branch 5d.ii.i.a, item 4: the refined matching bound on A(K) as a ladder of exact
relaxations, computed by the same MILP as arc_milp.

Keep the concrete (small) gears real; relax what the adversary may buy for a gear whose
long arc does not fit in the run (a bare domino of size a_g):

  B1  hole count only   : any two columns of the run, or one
  B2  + parity          : any two columns at an EVEN distance (a_g is always even,
                          because 3 a_g = g -+ 1 = 0 mod 6), or one
  B3  + realisable arcs : any two columns at a distance a with 3a-1 or 3a+1 prime,
                          UNLIMITED multiplicity
  A   the truth         : arcs of primes actually big at this L, multiplicity 1 or 2
                          (2 exactly at a twin pair)

B1 >= B2 >= B3 >= A, all exact.  The gaps say how much of A(K) is decided by the hole
COUNT alone (a counting quantity, which face A forbids using), by parity, by which arcs
the primes offer, and by how often they offer them.

Usage: uv run python research/anchor235/r50/arc_relax.py [KMAX]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arc_core import Level, arc, is_prime, masks_for, primes_upto, RESULTS  # noqa: E402


def realisable_arcs(amax):
    return [a for a in range(2, amax + 1, 2)
            if is_prime(3 * a - 1) or is_prime(3 * a + 1)]


def build(L, K, mode):
    lv = Level(L, K)
    cols, types, mult = [], [], []
    conc = [(k, key, m) for k, key, m in lv.items if k == 'p']
    for idx, (kind, key, m) in enumerate(conc):
        for msk in masks_for(kind, key, L):
            if msk:
                cols.append(msk)
                types.append(idx)
        mult.append(1)
    if mode is None:
        for kind, key, m in lv.items:
            if kind == 'p':
                continue
            idx = len(mult)
            for msk in masks_for(kind, key, L):
                if msk:
                    cols.append(msk)
                    types.append(idx)
            mult.append(min(m, K))
    else:
        idx = len(mult)
        arcs = set(realisable_arcs(2 * L + 4))
        for i in range(L):
            cols.append(1 << i)
            types.append(idx)
            for j in range(i + 1, L):
                d = j - i
                if mode == 'B1' or (mode == 'B2' and d % 2 == 0) or \
                   (mode == 'B3' and d in arcs):
                    cols.append((1 << i) | (1 << j))
                    types.append(idx)
        mult.append(K)
    nt = len(mult)
    n = len(cols)
    rows, ca, vals = [], [], []
    for j, msk in enumerate(cols):
        mm = msk
        while mm:
            b = mm & -mm
            rows.append(b.bit_length() - 1)
            ca.append(j)
            vals.append(1.0)
            mm ^= b
    for j, t in enumerate(types):
        rows.append(L + t)
        ca.append(j)
        vals.append(1.0)
    for j in range(n):
        rows.append(L + nt)
        ca.append(j)
        vals.append(1.0)
    A = csr_matrix((vals, (rows, ca)), shape=(L + nt + 1, n))
    lb = np.concatenate([np.ones(L), np.full(nt, -np.inf), [-np.inf]])
    ub = np.concatenate([np.full(L, np.inf), np.array(mult, dtype=float), [K]])
    return A, lb, ub, n


def feasible(L, K, mode):
    A, lb, ub, n = build(L, K, mode)
    res = milp(c=np.zeros(n), constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    return res.success


def top(K, mode, L0):
    L = L0
    while feasible(L, K, mode):
        L += 1
    return L


def main():
    KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.makedirs(RESULTS, exist_ok=True)
    log = open(os.path.join(RESULTS, "arc_relax.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    A = {2: 5, 3: 7, 4: 16, 5: 22, 6: 28, 7: 37, 8: 45, 9: 68, 10: 88}
    ps = primes_upto(500)
    say("  K   A(K)   B3 (+prime arcs)  B2 (+parity)  B1 (hole count)   "
        "B3/A   B2/A   B1/A    W(p_{K+1})   A/W")
    out = []
    for K in range(3, KMAX + 1):
        a = A[K]
        row = {}
        for mode in ("B3", "B2", "B1"):
            t = time.time()
            row[mode] = top(K, mode, a)
            say(f"      K={K} {mode} = {row[mode]}  ({time.time()-t:.1f}s)")
        qn = ps[K + 2]
        W = (qn * qn - 1) // 6
        say(f"  {K:2d}  {a:5d}   {row['B3']:9d}  {row['B2']:12d}  {row['B1']:13d}   "
            f"{row['B3']/a:5.2f}  {row['B2']/a:5.2f}  {row['B1']/a:5.2f}   "
            f"{W:8d}   {a/W:5.3f}")
        out.append({"K": K, "A": a, "W": W, **row})
    json.dump(out, open(os.path.join(RESULTS, "arc_relax.json"), "w"))
    log.close()


if __name__ == "__main__":
    main()
