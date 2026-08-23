"""Round 20 (constructor): THE MAX-PLUS (TROPICAL) SIDE OF THE TRANSFER MATRIX.

The spectrum F_j = max sum of j consecutive gaps is a TROPICAL power: with
edge set E = {(u,v) : gap value u is immediately followed by v somewhere in
the period} (the SUPPORT of the lag-1 pair census) and node weights = gap
values,

    F_j  <=  trop_j  :=  max over j-node paths in E of the node sum,

and qualmax_j <= tropQ_j (paths whose j-2 interior nodes lie in V(q')).
The bound drops all memory beyond one step; its TIGHTNESS measures how much
of the extremal structure the pair table alone pins.  Also computed: the max
cycle mean (tropical eigenvalue) of E and of the V-interior subgraph - the
growth rates lim F_j/j and lim qualmax_j/j the pair table permits, and
whether the V-graph even has a cycle (if not, qualifying depth is capped by
pure pair-support nilpotency).

Inputs: gap_pair_joint.csv (lag 1), flank_envelope_spectra.csv (exact F_j),
tm_resid_runs.csv (exact qualmax_j).
"""
import csv
import os
import numpy as np
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")
VMAX = 128
NEXTP = {11: 13, 13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37}
NEG = -10 ** 9


def load_pair1():
    pair = {}
    seen = set()
    with open(os.path.join(DDIR, "gap_pair_joint.csv")) as f:
        for r in csv.DictReader(f):
            if r["lag"] != "1":
                continue
            key = (r["y"], r["gu"], r["gv"])
            if key in seen:
                continue
            seen.add(key)
            pair.setdefault(int(r["y"]), np.zeros((VMAX, VMAX), np.int64))[
                int(r["gu"]), int(r["gv"])] = int(r["count"])
    return pair


def load_spectra():
    sp = {}
    with open(os.path.join(DDIR, "flank_envelope_spectra.csv")) as f:
        for r in csv.DictReader(f):
            if float(r["coverage"]) < 1.0:
                continue
            sp[int(r["y"])] = [int(r[f"F{j}"]) for j in range(1, 9)]
    return sp


def load_runs():
    runs = {}
    with open(os.path.join(DDIR, "tm_resid_runs.csv")) as f:
        for r in csv.DictReader(f):
            runs[int(r["y"])] = r
    return runs


def qual_set(q1, upto):
    c = pow(6, -1, q1)
    Q = {0, (2 * c) % q1, (-2 * c) % q1}
    return [v for v in range(1, upto + 1) if v % q1 in Q]


def max_cycle_mean(nodes, adj, weight):
    """Karp: max mean weight over cycles of (nodes, adj); weight per node.
    Returns None if acyclic."""
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    d = np.full((n + 1, n), float(NEG))
    d[0] = 0.0
    for k in range(1, n + 1):
        for vi, v in enumerate(nodes):
            best = float(NEG)
            for u in adj.get(v, ()):  # predecessors
                if d[k - 1][idx[u]] > NEG / 2:
                    best = max(best, d[k - 1][idx[u]] + weight[v])
            d[k][vi] = best
    best = None
    for vi in range(n):
        if d[n][vi] <= NEG / 2:
            continue
        worst = None
        for k in range(n):
            if d[k][vi] <= NEG / 2:
                continue
            val = (d[n][vi] - d[k][vi]) / (n - k)
            worst = val if worst is None else min(worst, val)
        if worst is not None:
            best = worst if best is None else max(best, worst)
    return best


def main():
    pair = load_pair1()
    spectra = load_spectra()
    runs = load_runs()
    for y in sorted(pair):
        q1 = NEXTP[y]
        C = pair[y]
        nodes = sorted(set(np.flatnonzero(C.sum(1))) |
                       set(np.flatnonzero(C.sum(0))))
        F = max(nodes)
        V = set(qual_set(q1, F))
        edges = {(u, v) for u in nodes for v in nodes if C[u, v] > 0}
        sp = spectra.get(y)
        r = runs.get(y)
        print(f"\n=== machine {y}  q'={q1}  nodes {len(nodes)}  "
              f"edges {len(edges)}  (density {len(edges)/len(nodes)**2:.2f})")
        # longest path with j nodes = window of j gaps
        f = {v: float(v) for v in nodes}          # paths of length 1
        fq = {v: float(v) for v in nodes}         # interior-restricted
        rowF, rowT, rowQ, rowTQ = [], [], [], []
        for j in range(2, 9):
            nf = {}
            for (u, v) in edges:
                cand = f[u] + v
                if cand > nf.get(v, NEG):
                    nf[v] = cand
            f = {v: nf.get(v, float(NEG)) for v in nodes}
            trop = int(max(f.values()))
            # interior-restricted: at step t (2..j-1) node must be in V;
            # recompute fresh for this j
            g = {v: float(v) for v in nodes}
            for t in range(2, j + 1):
                ng = {}
                for (u, v) in edges:
                    if t <= j - 1 and v not in V:
                        continue
                    if g[u] <= NEG / 2:
                        continue
                    cand = g[u] + v
                    if cand > ng.get(v, NEG):
                        ng[v] = cand
                g = {v: ng.get(v, float(NEG)) for v in nodes}
            tropQ = int(max(g.values())) if max(g.values()) > NEG / 2 else 0
            Fj = sp[j - 1] if sp and j <= 8 else None
            qm = int(r[f"qm{j}"]) if r and 3 <= j <= 6 else None
            rowF.append(Fj)
            rowT.append(trop)
            rowQ.append(qm)
            rowTQ.append(tropQ)
        print("   j:        " + "".join(f"{j:>7}" for j in range(2, 9)))
        print("   F_j:      " + "".join(f"{x if x is not None else '?':>7}"
                                        for x in rowF))
        print("   trop_j:   " + "".join(f"{x:>7}" for x in rowT))
        print("   qualmax:  " + "".join(f"{x if x is not None else '?':>7}"
                                        for x in rowQ))
        print("   tropQ_j:  " + "".join(f"{x:>7}" for x in rowTQ))
        # tropical eigenvalues (max cycle means)
        pred = {}
        for (u, v) in edges:
            pred.setdefault(v, []).append(u)
        wt = {v: v for v in nodes}
        mcm = max_cycle_mean(nodes, pred, wt)
        # V-interior cycles: all nodes of the cycle must be in V
        nodesV = [v for v in nodes if v in V]
        predV = {v: [u for u in pred.get(v, ()) if u in V] for v in nodesV}
        mcmV = max_cycle_mean(nodesV, predV, wt) if nodesV else None
        print(f"   max cycle mean (full graph): {mcm:.2f}   "
              f"(V-graph): {mcmV if mcmV is not None else 'ACYCLIC'}")


if __name__ == "__main__":
    main()
