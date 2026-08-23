"""Round 20 lateral: THE RENEWAL FACTOR - the interior disjunction measured in
isolation, after dividing out ALL closed-form endpoint arithmetic.

Objects (full-period, exact):
    W1(g)  = gap histogram of machine y (count of gaps of size exactly g)
    N2(g)  = prod_q c_q(g)                      (pair correlation, closed form)
    rho(g) = W1(g)/N2(g) = P(no opening strictly between | endpoints open)
    pred(g)= N2(g) * prod_{t=1}^{g-1} (1 - N3(0,t,g)/N2(g))
             - the ZERO-PARAMETER closed-form predictor: endpoint arithmetic
               times interior blocking treated as independent given endpoints
               (N3 = 3-point correlation, closed form via c_q).
    kappa(g) = W1(g)/pred(g)
             - the IRREDUCIBLE remainder: exactly the failure of interior
               independence, i.e. the disjunction obstruction, isolated.

Exactness anchors: kappa(1) = kappa(2) = 1 identically (0 or 1 interior
points - no disjunction yet); g = 3 is the first place kappa can differ
from 1.

Data: exact cyclic W1 from depth_identity_<y>.csv (machines 11-29);
machine 31 from the Mechanic's full-period census gap_pair_joint.csv
(marginal; boundary error <= 2 counts in 6.2e9).

Also: the padded-lag cell g = q' (the padding supply) re-tested against the
full predictor - round 19's enhanced-lag law explained ~1/10 of the supply
erraticity; how much does pred() explain?
"""
import csv, os
from math import prod, log
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))]

def cq_set(q, offs):
    """#{r mod q : r+d exposed for all d in offs} - brute force, exact."""
    u = pow(6, -1, q)
    t = {u % q, (-u) % q}
    return sum(1 for r in range(q) if all((r + d) % q not in t for d in offs))

def N(gears, offs):
    return prod(cq_set(q, offs) for q in gears)

def load_W1():
    """machine -> {g: count}, exact where possible."""
    W = {}
    for y in (11, 13, 17, 19, 23, 29):
        p = os.path.join(DATA, f"depth_identity_{y}.csv")
        if os.path.exists(p):
            h = {}
            for r in csv.DictReader(open(p)):
                if int(r["W1"]):
                    h[int(r["g"])] = int(r["W1"])
            W[y] = h
    # machine 31 from the Mechanic's census (lag-1 marginal)
    h = defaultdict(int)
    for r in csv.DictReader(open(os.path.join(DATA, "gap_pair_joint.csv"))):
        if int(r["y"]) == 31 and int(r["lag"]) == 1:
            h[int(r["gu"])] += int(r["count"])
    W[31] = dict(h)
    return W

def analyse(y, W1, qp):
    gears = primes(5, y)
    P = prod(gears)
    gs = sorted(W1)
    F = max(gs)
    rows = []
    for g in gs:
        N2 = N(gears, [0, g])
        f = 1.0
        for t in range(1, g):
            f *= 1.0 - N(gears, [0, t, g]) / N2
        pred = N2 * f
        rows.append((g, W1[g], N2, pred, W1[g] / pred if pred > 0 else float("inf")))
    # exactness anchors
    for g, w, n2, pred, k in rows:
        if g == 1:
            assert abs(w - n2) <= 2, (y, g, w, n2)
        if g == 2:
            assert abs(k - 1.0) < 5e-3 or abs(w - pred) <= 2, (y, g, w, pred)
    # variance decomposition of log W1 over all populated g
    lw = np.array([log(r[1]) for r in rows])
    ln2 = np.array([log(r[2]) for r in rows])
    lpred = np.array([log(r[3]) for r in rows])
    gg = np.array([float(r[0]) for r in rows])
    tot = np.var(lw)
    # M1: arithmetic only (shift-fitted)
    r1 = lw - ln2; r1 -= r1.mean()
    # M2: arithmetic + fitted geometric decay
    A = np.vstack([np.ones_like(gg), gg]).T
    coef2, *_ = np.linalg.lstsq(A, lw - ln2, rcond=None)
    r2 = lw - ln2 - A @ coef2
    # M0g: geometric only (no arithmetic) - the r18 baseline
    coef0, *_ = np.linalg.lstsq(A, lw, rcond=None)
    r0 = lw - A @ coef0
    # M3: full closed-form predictor, ZERO fitted parameters
    r3 = lw - lpred
    # M4: predictor + fitted linear kappa law
    coef4, *_ = np.linalg.lstsq(A, r3, rcond=None)
    r4 = r3 - A @ coef4
    print(f"--- machine {y} (P={P}, {len(rows)} gap values, F={F}) ---")
    print(f"  var(log W1) = {tot:.3f}; residual var: arithmetic-only "
          f"{np.var(r1):.3f} ({100*(1-np.var(r1)/tot):.1f}%), "
          f"arith+geom {np.var(r2):.3f} ({100*(1-np.var(r2)/tot):.1f}%)")
    print(f"  WIGGLE TEST (r18 comparison): geometric-only resid var "
          f"{np.var(r0):.3f} -> with full N2: {np.var(r2):.3f}  "
          f"(N2 removes {100*(1-np.var(r2)/np.var(r0)):.1f}% of the "
          f"post-trend residual; r18's c5*c7 removed 24-28%)")
    print(f"  ZERO-PARAM closed form: rms(log W1 - log pred) = "
          f"{np.sqrt(np.mean(r3**2)):.3f}  (bias {r3.mean():+.3f}); "
          f"+linear kappa fit: resid var {np.var(r4):.3f} "
          f"({100*(1-np.var(r4)/tot):.1f}% of var explained), "
          f"kappa slope {coef4[1]:+.4f}/slot")
    # kappa table at small g and at the padded lag
    ks = {r[0]: r[4] for r in rows}
    show = [g for g in (1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, F) if g in ks]
    print("  kappa(g): " + "  ".join(f"{g}:{ks[g]:.3f}" for g in show))
    if qp is not None and qp in ks:
        i = [r for r in rows if r[0] == qp][0]
        print(f"  PADDED LAG g = q' = {qp}: measured {i[1]}, closed-form pred "
              f"{i[3]:.1f}, kappa {i[4]:.3f}")
    return rows, ks

if __name__ == "__main__":
    W = load_W1()
    NEXTP = {11: 13, 13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37}
    pad = []
    for y in sorted(W):
        rows, ks = analyse(y, W[y], NEXTP[y])
        qp = NEXTP[y]
        if qp in ks:
            i = [r for r in rows if r[0] == qp][0]
            pad.append((y, qp, i[1], i[3], i[4]))
    print("=" * 74)
    print("PADDED-LAG SUPPLY vs THE FULL CLOSED-FORM PREDICTOR")
    print(f"  {'step':>9} {'measured':>10} {'pred':>12} {'meas/pred':>10}")
    for y, qp, w, p, k in pad:
        print(f"  {y:>4}->{qp:<4} {w:>10} {p:>12.1f} {k:>10.3f}")
    if len(pad) >= 2:
        k = np.array([p[4] for p in pad], float)
        print(f"  kappa(q') spread across steps (max/min): "
              f"{k.max()/k.min():.2f}x - this is the residual erraticity "
              f"after the FULL closed-form predictor")
