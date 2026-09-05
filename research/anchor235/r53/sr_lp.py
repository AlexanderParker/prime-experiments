"""sr_lp.py -- what the identities actually pin down.

The counterfactual of the branch: if twins were finite then for y > Y no even y-rough size below
y^2/3 would be depleted.  The test is whether the identities the multiplicity function satisfies
can be met by a spectrum with NO depleted sizes.  Rather than test one filled-in spectrum, this
computes the exact feasible INTERVAL of m(v) at each size under every identity we have:

  (a) sum_v m(v) = prod (q-2)        (b) sum_v v m(v) = prod q
  (d) m(1) = prod (q-4)              (e) m(2) = 8 prod_{q>=11} (q-4)
  (g) 0 <= m(v) <= A(v) = prod_q c_q(v)      (the autocorrelation cap, W_1 <= sum_j W_j)
  (h) m(v) = 0 for v > F   -- NOT imposed: F is what the root is trying to bound.

Minimising and maximising m(v) over that polytope says exactly how much the identities know.
Also reports the budget share of the uncoupled sizes, and rho(v) = m(v)/A(v), the multiplicity
normalised by the exact pair count, which is what the identities cannot explain.

Writes results/sr_lp.txt
"""
import os, json
import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
PR = [5, 7, 11, 13, 17, 19, 23, 29, 31]


def u_of(g):
    return pow(6, -1, g)


def c_local(q, v):
    u = u_of(q)
    s = {u % q, (-u) % q, (u - v) % q, ((-u) - v) % q}
    return q - len(s)


def role(q, v):
    if v % q == 0:
        return "pad"
    d = (2 * u_of(q)) % q
    return "letter" if v % q in (d % q, (-d) % q) else None


def load(y):
    f1 = os.path.join(OUT, f"spec_m{y}.json")
    f2 = os.path.join(OUT, f"spec_rec_m{y}.json")
    if os.path.exists(f1):
        return {int(k): v for k, v in json.load(open(f1)).items()}
    return {int(k): v for k, v in json.load(open(f2))["m"].items()}


def main():
    lines = []
    W = lines.append
    for iy, y in enumerate([19, 23, 29, 31]):
        gears = [g for g in PR if g <= y]
        m = load(y)
        F = max(m)
        P, N = 1, 1
        for g in gears:
            P *= g
            N *= g - 2
        A = {v: int(np.prod([c_local(g, v) for g in gears], dtype=object)) for v in range(1, F + 1)}
        unc = [v for v in range(2, F + 1) if not any(role(g, v) for g in gears)]
        W(f"=== machine m{y}   F={F}  N={N}  P={P}   uncoupled sizes <= F: {unc}")
        # rho
        W("  v : m(v) : A(v)=prod c_q(v) : rho = m/A : (coupling)")
        for v in range(1, F + 1):
            rl = [f"{g}:{role(g,v)}" for g in gears if role(g, v)]
            W(f"   {v:3d} : {m.get(v,0):>12d} : {A[v]:>12d} : {m.get(v,0)/A[v]:.6e} : "
              f"{','.join(rl) if rl else 'UNCOUPLED'}")
        # rho of the uncoupled sizes against a local median of rho
        for v in unc:
            nb = [m.get(w, 0) / A[w] for w in range(max(2, v - 4), min(F, v + 4) + 1)
                  if w != v and any(role(g, w) for g in gears)]
            if nb:
                med = float(np.median(nb))
                rv = m.get(v, 0) / A[v]
                W(f"  residual after dividing out the pair count: v={v}  rho={rv:.4e} "
                  f"vs local median rho {med:.4e}  ratio {rv/med if med else 0:.4f}")
        # budget shares
        sn = sum(m.get(v, 0) for v in unc) / N
        sp = sum(v * m.get(v, 0) for v in unc) / P
        W(f"  budget share of the uncoupled sizes: count {sn:.3e}, length {sp:.3e}")
        # LP: feasible interval of m(v) for each v, under (a),(b),(d),(e),(g)
        n = F
        Aeq = [[1.0] * n, [float(v) for v in range(1, n + 1)]]
        beq = [float(N), float(P)]
        e1 = [0.0] * n; e1[0] = 1.0
        e2 = [0.0] * n; e2[1] = 1.0
        pq4 = 1
        for g in gears:
            pq4 *= g - 4
        m2 = 8
        for g in gears:
            if g >= 11:
                m2 *= g - 4
        Aeq += [e1, e2]
        beq += [float(pq4), float(m2)]
        bounds = [(0.0, float(A[v])) for v in range(1, n + 1)]
        out = []
        for v in unc + [v + 1 for v in unc if v + 1 <= F]:
            c = [0.0] * n
            c[v - 1] = 1.0
            lo = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
            hi = linprog([-x for x in c], A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
            out.append((v, lo.status, lo.fun if lo.success else None,
                        hi.status, -hi.fun if hi.success else None, A[v], m.get(v, 0)))
        for v, s1, f1, s2, f2, cap, meas in out:
            W(f"  LP interval for m({v}): [{f1:.6g}, {f2:.6g}]  (cap A(v) = {cap}, "
              f"measured {meas})   {'FEASIBLE both ends' if s1 == 0 and s2 == 0 else 'INFEASIBLE'}")
        W("")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "sr_lp.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
