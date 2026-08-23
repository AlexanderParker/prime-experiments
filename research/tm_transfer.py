"""Round 20 (constructor): THE TRANSFER-MATRIX FORMULATION OF p_j.

Frame (exact operators on C^{Z_P}): S = slot shift, D = exposure projector
(D = tensor product over gears of D_q by CRT), B = I - D.  Gap operator
G_v = D (S B)^{v-1} S D; renewal operator R = sum_v G_v (the successor
permutation on openings).  Every census quantity is a matrix element:
N(v) = 1'G_v 1, joint N_j(u,v) = 1'G_u R^{j-1} G_v 1.  In THIS frame there is
no spectral gap (R is a permutation); the spectral content appears after
AGGREGATION to the gap-value chain - measured here.

THE AGGREGATED TRANSFER MATRIX. States = gap values; T[u,v] = P(next gap =
v | this gap = u), built from Mechanic's exact full-period pair census
(lag 1).  Questions answered exactly against the censuses:

 (A) MARKOV CLOSURE: does the one-step matrix reproduce the measured run
     deficits at depths 3-6 (both the residue-qualifying set V(q') and the
     size floors), the lag-2 rebound, and the return to independence by
     lag 4-5?  Predictions vs exact counts, no fits.
 (B) THE SPECTRAL STATEMENT: per-link anti-correlation constant =
     rho(T_VV) (Perron value of the V-restricted substochastic block),
     to be compared with p_1V (independence).  Decorrelation rate =
     |lambda_2(T)| (spectral gap of the full chain).

Inputs: research/data/gap_pair_{joint,hist}.csv (Mechanic, full period,
machines 11..31), research/data/tm_resid_runs.csv (this round, exact).
"""
import csv
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")
VMAX = 128
MACHINES = [11, 13, 17, 19, 23, 29, 31]
NEXTP = {11: 13, 13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37}


def load():
    ghist = {}
    minh = {}
    seen = set()
    dup = 0
    with open(os.path.join(DDIR, "gap_pair_hist.csv")) as f:
        for r in csv.DictReader(f):
            key = (r["y"], r["kind"], r["index"], r["value"])
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            y, v, c = int(r["y"]), int(r["value"]), int(r["count"])
            if r["kind"] == "ghist":
                ghist.setdefault(y, np.zeros(VMAX, np.int64))[v] = c
            else:
                m = int(r["index"])
                minh.setdefault(y, {}).setdefault(
                    m, np.zeros(VMAX, np.int64))[v] = c
    pair = {}
    seen = set()
    with open(os.path.join(DDIR, "gap_pair_joint.csv")) as f:
        for r in csv.DictReader(f):
            key = (r["y"], r["lag"], r["gu"], r["gv"])
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            y, j = int(r["y"]), int(r["lag"])
            pair.setdefault(y, {}).setdefault(
                j, np.zeros((VMAX, VMAX), np.int64))[int(r["gu"]),
                                                     int(r["gv"])] = int(r["count"])
    runs = {}
    with open(os.path.join(DDIR, "tm_resid_runs.csv")) as f:
        for r in csv.DictReader(f):
            runs[int(r["y"])] = r
    if dup:
        print(f"[load] skipped {dup} exact duplicate csv rows")
    return ghist, minh, pair, runs


def qual_set(q1, upto):
    c = pow(6, -1, q1)
    Q = {0, (2 * c) % q1, (-2 * c) % q1}
    return np.array([v for v in range(1, upto + 1) if v % q1 in Q])


def main():
    ghist, minh, pair, runs = load()
    for y in MACHINES:
        if y not in pair:
            continue
        q1 = NEXTP[y]
        gh = ghist[y].astype(float)
        ngaps = gh.sum()
        C1 = pair[y][1].astype(float)
        # marginal consistency (mechanic's census has an uncounted seam gap)
        dmarg = np.abs(C1.sum(1) - gh)
        print(f"\n=== machine {y}  q'={q1}  ngaps {int(ngaps):,}  "
              f"max|rowsum-ghist| = {int(dmarg.max())}")
        F = int(np.flatnonzero(gh)[-1])
        V = qual_set(q1, F)
        supp = np.flatnonzero(gh)
        # row-normalised transfer matrix on the support
        T = np.zeros_like(C1)
        T[supp] = C1[supp] / C1[supp].sum(1, keepdims=True)
        pi = gh / ngaps
        p1V = pi[V].sum()

        # ---------- (A) Markov closure: residue-qualifying runs ----------
        r = runs.get(y)
        print(f"  V(q') = {V.tolist()}   p_1V = {p1V:.6g}")
        if r:
            assert abs(int(r["run1"]) - gh[V].sum()) <= 2, "run1 mismatch"
        x = gh[V].copy()          # counts entering a qualifying run
        TVV = T[np.ix_(V, V)]
        print("   m (j)   exact run count      Markov pred     indep pred"
              "      pred/exact")
        for m in range(2, 6):
            x = x @ TVV
            predm = x.sum()
            ind = ngaps * p1V ** m
            exact = int(r[f"run{m}"]) if r and m <= 4 else None
            tag = (f"{predm / exact:10.3f}" if exact
                   else "     (n/a)")
            print(f"   {m} ({m + 2})  "
                  f"{exact if exact is not None else '?':>12}   "
                  f"{predm:>14.2f}  {ind:>14.2f}  {tag}")

        # ---------- (A') Markov closure: size floors vs minhist ----------
        a = 2 * round(q1 / 6)
        Q = supp[supp >= a]
        pQ = pi[Q].sum()
        TQQ = T[np.ix_(Q, Q)]
        x = gh[Q].copy()
        mh = minh.get(y, {})
        print(f"  size floor a = {a}  (P(g>=a) = {pQ:.5g})  runs vs minhist:")
        line = "   m:      "
        le, lp, li, lr = "   exact:  ", "   pred:   ", "   indep:  ", "   p/e:    "
        for m in range(2, 7):
            if m > 2:
                x = x @ TQQ
            if m == 2:
                x = gh[Q] @ TQQ
            predm = x.sum()
            exact = int(mh[m][a:].sum()) if m in mh else None
            ind = ngaps * pQ ** m
            line += f"{m:>11}"
            le += f"{exact if exact is not None else '?':>11}"
            lp += f"{predm:>11.1f}"
            li += f"{ind:>11.1f}"
            lr += (f"{predm / exact:>11.3f}" if exact else f"{'n/a':>11}")
        print("\n".join([line, le, lp, li, lr]))

        # ---------- (B) spectral constants ----------
        evV = np.linalg.eigvals(TVV) if len(V) else np.array([0.0])
        rhoV = float(np.abs(evV).max())
        evQ = np.linalg.eigvals(TQQ) if len(Q) else np.array([0.0])
        rhoQ = float(np.abs(evQ).max())
        Ts = T[np.ix_(supp, supp)]
        ev = np.linalg.eigvals(Ts)
        ev = ev[np.argsort(-np.abs(ev))]
        lam2 = float(np.abs(ev[1]))
        print(f"  PERRON: rho(T_VV) = {rhoV:.5f} vs p_1V = {p1V:.5f}  "
              f"(per-link anti-corr x{p1V / rhoV if rhoV else float('inf'):.2f})"
              f"   rho(T_QQ) = {rhoQ:.5f} vs pQ = {pQ:.5f} "
              f"(x{pQ / rhoQ if rhoQ else float('inf'):.2f})")
        print(f"  CHAIN SPECTRUM: lambda_1 = {np.abs(ev[0]):.4f} "
              f"|lambda_2| = {lam2:.4f}  |l2|^4 = {lam2**4:.2e}  "
              f"(decorrelation rate)")

        # ---------- lag structure: rebound and decay ----------
        print("   lag j   sizefloor obs R(j)  pred R(j)   residueV obs R(j)"
              "  pred R(j)")
        Tj = np.eye(len(supp))
        idxQ = np.searchsorted(supp, Q)
        idxV = np.searchsorted(supp, V)
        piS = pi[supp]
        for j in range(1, 6):
            Tj = Tj @ Ts
            predQ = piS[idxQ] @ Tj[np.ix_(idxQ, idxQ)].sum(1)
            predV = piS[idxV] @ Tj[np.ix_(idxV, idxV)].sum(1)
            Cj = pair[y][j].astype(float)
            obsQ = Cj[np.ix_(Q, Q)].sum() / Cj.sum()
            obsV = Cj[np.ix_(V, V)].sum() / Cj.sum()
            print(f"     {j}       {obsQ / pQ**2:8.4f}     {predQ / pQ**2:8.4f}"
                  f"        {obsV / p1V**2:8.4f}      {predV / p1V**2:8.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
