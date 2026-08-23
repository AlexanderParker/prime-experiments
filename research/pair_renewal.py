"""Round 20 lateral: DOES THE GAP-PAIR INTERACTION REDUCE TO CLOSED-FORM
ENDPOINT ARITHMETIC x FACTORISING RENEWAL?

Renewal factorisation hypothesis (the object Constructor's p_j needs):

    J(g1,g2)  ~  N3(0, g1, g1+g2) * rho(g1) * rho(g2)

where J = full-period count of ADJACENT gap pairs (g1, g2),
      N3 = prod_q c_q({0, g1, g1+g2})   (3-point correlation, closed form),
      rho(g) = W1(g)/N2(g)              (the single-gap renewal factor).

Equivalently K/Lambda ~ 1 with K = J*N1/(W1(g1)W1(g2)) the measured
interaction and Lambda = N3*N1/(N2(g1)N2(g2)) the closed-form one.

Then the aggregate the route cares about: R(lag 1) over QUALIFYING gaps
(g >= 2u', g = 0 or +-2u' mod q') - measured vs predicted. This is the
anti-correlation deficit of (D), predicted from arithmetic + singles only.

Also: the adjacent-gap exclusion law re-checked at machine 31 (6.2e9 pairs).
"""
import csv, os
from math import prod
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))]

def cq_set(q, offs):
    u = pow(6, -1, q)
    t = {u % q, (-u) % q}
    return sum(1 for r in range(q) if all((r + d) % q not in t for d in offs))

# forbidden adjacent classes mod 5 (round-19 exclusion law)
FORBID5 = {(1, 1), (1, 3), (2, 4), (3, 1), (4, 2), (4, 4)}

def load():
    J = defaultdict(lambda: defaultdict(int))   # (y, lag) -> (g1,g2) -> n
    for r in csv.DictReader(open(os.path.join(DATA, "gap_pair_joint.csv"))):
        J[(int(r["y"]), int(r["lag"]))][(int(r["gu"]), int(r["gv"]))] \
            += int(r["count"])
    W = {}
    for y in (13, 17, 19, 23, 29):
        p = os.path.join(DATA, f"depth_identity_{y}.csv")
        if os.path.exists(p):
            W[y] = {int(r["g"]): int(r["W1"])
                    for r in csv.DictReader(open(p)) if int(r["W1"])}
    W[31] = defaultdict(int)
    for (g1, g2), n in J[(31, 1)].items():
        W[31][g1] += n
    W[31] = dict(W[31])
    return J, W

NEXTP = {19: 23, 23: 29, 29: 31, 31: 37}

def qualifying(y, F):
    qp = NEXTP[y]
    up = round(qp / 6)
    Q = [g for g in range(2 * up, F + 1)
         if g % qp in (0, (2 * up) % qp, (-2 * up) % qp)]
    return qp, 2 * up, Q

if __name__ == "__main__":
    J, W = load()
    print("=" * 78)
    print("PART 1: exclusion law at machine 31 (new scale: 6.2e9 adjacent pairs)")
    bad = sum(n for (g1, g2), n in J[(31, 1)].items()
              if (g1 % 5, g2 % 5) in FORBID5)
    pop = sum(1 for (g1, g2), n in J[(31, 1)].items()
              if (g1 % 5, g2 % 5) in FORBID5 and n > 0)
    print(f"  counts in forbidden mod-5 classes: {bad} (populated cells {pop})")
    assert bad == 0

    print("=" * 78)
    print("PART 2: renewal factorisation J vs N3*rho*rho (lag-1 cells)")
    summary = []
    for y in (19, 23, 29, 31):
        gears = primes(5, y)
        N1 = prod(q - 2 for q in gears)
        W1 = W[y]
        F = max(W1)
        N2 = {g: prod(cq_set(q, [0, g]) for q in gears) for g in W1}
        rho = {g: W1[g] / N2[g] for g in W1}
        cells = []
        zeros_sel = 0
        for (g1, g2), n in sorted(J[(y, 1)].items()):
            if g1 not in W1 or g2 not in W1:
                continue
            N3 = prod(cq_set(q, [0, g1, g1 + g2]) for q in gears)
            pred = N3 * rho[g1] * rho[g2]
            if n == 0:
                continue
            if pred == 0:
                print(f"  !! y={y} observed {n} at ({g1},{g2}) with pred 0")
                continue
            cells.append((g1, g2, n, pred))
        lr = np.array([np.log(c[2] / c[3]) for c in cells])
        wts = np.array([c[2] for c in cells], float)
        med = np.exp(np.median(lr))
        print(f"  machine {y}: {len(cells)} populated cells; J/pred: "
              f"median {med:.3f}, log-sd {lr.std():.3f}, "
              f"count-weighted mean {np.exp(np.average(lr, weights=wts)):.3f}, "
              f"5-95% [{np.exp(np.percentile(lr,5)):.2f}, "
              f"{np.exp(np.percentile(lr,95)):.2f}]")
        # the aggregate (D) cares about
        qp, thr, Q = qualifying(y, F)
        Qs = [g for g in Q if g in W1]
        sumW = sum(W1[g] for g in Qs)
        meas = sum(J[(y, 1)].get((a, b), 0) for a in Qs for b in Qs)
        predR = 0.0
        for a in Qs:
            for b in Qs:
                N3 = prod(cq_set(q, [0, a, a + b]) for q in gears)
                predR += N3 * rho[a] * rho[b]
        Rm = meas * N1 / sumW**2 if sumW else float("nan")
        Rp = predR * N1 / sumW**2 if sumW else float("nan")
        print(f"    qualifying set (q'={qp}, g >= {thr}): {Qs}; "
              f"p_1 = {sumW/N1:.3e}")
        print(f"    R(1) measured = {Rm:.4f}   R(1) predicted = {Rp:.4f}   "
              f"ratio {Rm/Rp if Rp else float('nan'):.3f}")
        # context: measured R at lags 1..5
        rl = []
        for L in range(1, 6):
            m = sum(J[(y, L)].get((a, b), 0) for a in Qs for b in Qs)
            rl.append(m * N1 / sumW**2)
        print(f"    measured R(lag 1..5) = " +
              " ".join(f"{v:.3f}" for v in rl))
        summary.append((y, Rm, Rp))
    print("=" * 78)
    print("SUMMARY: the anti-correlation deficit predicted from closed form")
    print(f"  {'y':>3} {'R_meas':>8} {'R_pred':>8} {'meas/pred':>10}")
    for y, Rm, Rp in summary:
        print(f"  {y:>3} {Rm:>8.4f} {Rp:>8.4f} {Rm/Rp:>10.3f}")

    print("=" * 78)
    print("PART 3: full closed-form JOINT predictor (4-point interiors) and")
    print("        does the irreducible correction FACTORISE: kappa2 ~ k1*k1?")
    for y in (23, 29, 31):
        gears = primes(5, y)
        N1 = prod(q - 2 for q in gears)
        W1 = W[y]
        F = max(W1)
        N2 = {g: prod(cq_set(q, [0, g]) for q in gears) for g in W1}
        # single-gap zero-param predictor and kappa1
        k1 = {}
        for g in W1:
            f = 1.0
            for t in range(1, g):
                f *= 1.0 - prod(cq_set(q, [0, t, g]) for q in gears) / N2[g]
            k1[g] = W1[g] / (N2[g] * f) if f > 0 else float("inf")
        lr2 = []
        wts = []
        qcells = []
        qp, thr, Q = qualifying(y, F)
        Rm2_num = Rp2_num = 0.0
        Qs = [g for g in Q if g in W1]
        sumW = sum(W1[g] for g in Qs)
        for (g1, g2), n in sorted(J[(y, 1)].items()):
            if n == 0 or g1 not in W1 or g2 not in W1:
                continue
            pts = [0, g1, g1 + g2]
            N3 = prod(cq_set(q, pts) for q in gears)
            if N3 == 0:
                continue
            f = 1.0
            for t in list(range(1, g1)) + list(range(g1 + 1, g1 + g2)):
                f *= 1.0 - prod(cq_set(q, pts + [t]) for q in gears) / N3
            pred2 = N3 * f
            if pred2 <= 0:
                continue
            kap2 = n / pred2
            ratio = kap2 / (k1[g1] * k1[g2])
            lr2.append(np.log(ratio)); wts.append(n)
            if g1 in Qs and g2 in Qs:
                qcells.append((g1, g2, n, pred2, kap2, ratio))
                Rm2_num += n; Rp2_num += pred2 * k1[g1] * k1[g2]
        lr2 = np.array(lr2); wts = np.array(wts, float)
        print(f"  machine {y}: kappa2/(k1*k1) over {len(lr2)} cells: "
              f"median {np.exp(np.median(lr2)):.3f}, log-sd {lr2.std():.3f}, "
              f"weighted mean {np.exp(np.average(lr2, weights=wts)):.3f}")
        if Rp2_num:
            print(f"    qualifying cells: R meas/pred(full joint x k1k1) = "
                  f"{Rm2_num/Rp2_num:.3f}   (was "
                  f"{[s for s in summary if s[0]==y][0][1]/[s for s in summary if s[0]==y][0][2]:.3f} with N3*rho*rho)")
        for c in qcells[:6]:
            print(f"      ({c[0]:>2},{c[1]:>2}): obs {c[2]:>8}  predfull "
                  f"{c[3]:>10.1f}  kappa2 {c[4]:.3f}  /k1k1 {c[5]:.3f}")
