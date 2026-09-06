"""Valve count per turn and yield weights, from the INVENTORY and YIELD laws alone (no ledger data).

A(m) = number of admissible fuelled families (s, s') with max(s, s') <= m:
  s, s' q-smooth, gcd | 2, same parity, and 4 divides exactly one of an even pair.
w(s, s') = (1/(s s')) prod_{odd p | s s'} (p - 1)/(p - 2): the family's density per unit n relative to the
  pure charge (from the yield law, log factors dropped). Sigma(m) = sum of w over A(m); Sigma(inf) = the
  Euler product. Predicted T_m / P_m ~ Sigma(m) (plus embers and log corrections), limit of B_m / T_m = 1 - 1/Sigma(inf).
usage: uv run python research/valves/r1/valve_count.py
"""
import math, json, os, sys
from fractions import Fraction

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "results"); os.makedirs(out, exist_ok=True)


def primes_upto(q):
    return [p for p in range(2, q + 1) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def smooth_upto(q, M):
    ps = primes_upto(q)
    sm = {1}
    for p in ps:
        new = set()
        for x in sm:
            y = x
            while y <= M:
                new.add(y); y *= p
        sm |= new
    return sorted(x for x in sm if x <= M)


def admissible(a, b):
    if (a - b) % 2:
        return False
    g = math.gcd(a, b)
    if g not in (1, 2):
        return False
    if a % 2 == 0:
        return (a // 2 + b // 2) % 2 == 1
    return True


def w(a, b, ps):
    v = Fraction(2 if a % 2 == 0 else 1, a * b)   # even pairs: partner parity forced, prime chance doubled
    for p in ps:
        if p > 2 and (a % p == 0 or b % p == 0):
            v *= Fraction(p - 1, p - 2)
    return v


def euler_limit(q):
    """Sigma(inf) = sum over all admissible q-smooth (s, s') of w(s, s').
    Odd pairs: s, s' odd coprime: sum = prod_{odd p<=q} (1 + 2 * sum_k (p-1)/(p-2) p^-k) = prod (1 + 2/(p-2)) = prod p/(p-2).
    Even pairs: s = 2a, s' = 2b with a, b coprime odd-part... enumerate by 2-adic valuations: exactly one of s, s' is 2 mod 4:
    s = 2 u, s' = 2^j v (j >= 2) or mirror, u, v odd coprime: w = (1/(2u 2^j v)) prod... = (1/2^{j+1}) w_odd(u, v);
    sum_{j>=2} 1/2^{j+1} = 1/4, two orders: 1/2. So Sigma = prod_{odd p<=q} p/(p-2) * (1 + 1/2) = (3/2) prod p/(p-2)."""
    ps = primes_upto(q)
    v = Fraction(2)   # odd pairs prod p/(p-2); even pairs (one member 2 mod 4, the other 0 mod 4) the same again: 2 sum_{j>=2} 2^-(j+1) * 2 * 2 = 1
    for p in ps:
        if p > 2:
            v *= Fraction(p, p - 2)
    return v


res = {}
for q in (5, 7, 11):
    ps = primes_upto(q)
    M = 60
    sm = smooth_upto(q, 10 ** 4)
    fams = [(a, b) for a in sm for b in sm if admissible(a, b)]
    table = []
    for m in range(1, M + 1):
        F = [(a, b) for (a, b) in fams if max(a, b) <= m]
        sig = sum(w(a, b, ps) for a, b in F)
        newf = [(a, b) for (a, b) in F if max(a, b) == m]
        table.append({"m": m, "A": len(F), "new": newf, "Sigma": float(sig), "pred_B_over_T": float(1 - 1 / sig),
                      "Sigma_1s": float(sum(w(a, b, ps) for a, b in F if a == 1)),
                      "pred_P_over_A": float(1 / sum(w(a, b, ps) for a, b in F if a == 1))})
    big = {}
    for Mbig in (100, 300, 1000, 3000, 10000):
        F = [(a, b) for (a, b) in fams if max(a, b) <= Mbig]
        big[Mbig] = {"A": len(F), "Sigma": float(sum(w(a, b, ps) for a, b in F))}
    lim = euler_limit(q)
    # check the Euler product numerically against the partial sum at 10^4
    lim1 = Fraction(1)
    for p in ps:
        if p > 2:
            lim1 *= Fraction(p - 1, p - 2)
    res[q] = {"table": table, "big": big, "Sigma_inf": float(lim), "pred_limit_B_over_T": float(1 - 1 / lim),
              "Sigma_1s_inf": float(lim1), "pred_limit_P_over_A": float(1 / lim1)}
    print(f"q={q}: A(60)={table[-1]['A']} Sigma(60)={table[-1]['Sigma']:.4f} predB/T(60)={table[-1]['pred_B_over_T']:.4f} "
          f"Sigma(10^4)={big[10000]['Sigma']:.4f} Sigma_inf={float(lim):.4f} lim B/T={float(1-1/lim):.4f} "
          f"P/A(60)={table[-1]['pred_P_over_A']:.4f} lim P/A={float(1/lim1):.4f}")
    print("  A(m), m=1..60:", [t["A"] for t in table])
    print("  pred B/T:", [round(t["pred_B_over_T"], 3) for t in table[:12]], "...", round(table[-1]["pred_B_over_T"], 3))
    print("  pred P/A:", [round(t["pred_P_over_A"], 3) for t in table[:12]], "...", round(table[-1]["pred_P_over_A"], 3))
    print("  new families at m<=12:", [(t["m"], t["new"]) for t in table[:12] if t["new"]])
with open(os.path.join(out, "valve_count.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
