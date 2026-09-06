"""Log-corrected yield prediction per turn, from the YIELD law's integrand: the family (s, s') in turn m has its
fuel at scale mQ/s and its partner's fuel at mQ/s', so its count relative to the pure charge's is
w(s, s') L_m(s, s') with L_m = int_{mQ}^{(m+1)Q} dn / (log(n/s) log((n+2)/s')) / int dn / log^2 n.
Compares T_m / P_m with Sigma^log(m) = sum over admissible max <= m of w L, and P_m / A_m with 1 / Sigma_1^log(m),
per turn and aggregated over turns 30..60.
usage: uv run python research/valves/r1/yield_check.py
"""
import json, os, glob, math

here = os.path.dirname(os.path.abspath(__file__))
res = os.path.join(here, "results")


def primes_upto(q):
    return [p for p in range(2, q + 1) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def smooth_upto(q, M):
    sm = {1}
    for p in primes_upto(q):
        sm |= {x * p ** e for x in sm for e in range(1, 40) if x * p ** e <= M}
    return sorted(x for x in sm if x <= M)


def admissible(a, b):
    return (a - b) % 2 == 0 and math.gcd(a, b) in (1, 2) and (a % 2 == 1 or (a // 2 + b // 2) % 2 == 1)


def w(a, b, ps):
    v = (2.0 if a % 2 == 0 else 1.0) / (a * b)   # even pairs: the partner's parity is forced by the class of P, doubling its prime chance
    for p in ps:
        if p > 2 and (a % p == 0 or b % p == 0):
            v *= (p - 1) / (p - 2)
    return v


def L(m, Q, a, b, k=8):
    """ratio of integrals over the turn, Simpson-free midpoint rule with k panels (smooth integrands)."""
    num = 0.0; den = 0.0
    for i in range(k):
        n = m * Q + (i + 0.5) * Q / k
        num += 1.0 / (math.log(n / a) * math.log((n + 2) / b))
        den += 1.0 / math.log(n) ** 2
    return num / den


out = []
for fn in sorted(glob.glob(os.path.join(res, "ledger_q*_Q*.json"))):
    d = json.load(open(fn)); q, Q, rows = d["q"], d["Q"], d["rows"]
    ps = primes_upto(q)
    sm = smooth_upto(q, 60)
    fams = [(a, b) for a in sm for b in sm if admissible(a, b)]
    per = []
    sumT = sumP = sumA = 0; predT = predA = 0.0
    for r in rows:
        m = r["m"]
        F = [(a, b) for (a, b) in fams if max(a, b) <= m]
        sig = sum(w(a, b, ps) * L(m, Q, a, b) for a, b in F)
        sig1 = sum(w(a, b, ps) * L(m, Q, a, b) for a, b in F if a == 1)
        per.append((m, r["T"] / r["P"], sig, r["P_over_A"], 1 / sig1))
        if m >= 30:
            sumT += r["T"]; sumP += r["P"]; sumA += r["A"]; predT += r["P"] * sig; predA += r["P"] * sig1
    ratios = [t / s for (_, t, s, _, _) in per[2:]]
    ratios1 = [pa * s1 for (_, _, _, pa, s1) in per[2:] if pa]
    ratios1 = [pa / (1 / s1) for (_, _, _, pa, s1) in per[2:]]
    line = (f"q={q} Q={Q}: T/P over Sigma^log(m), m>=3: mean {sum(ratios)/len(ratios):.3f} min {min(ratios):.3f} max {max(ratios):.3f}; "
            f"aggregate m=30..60: T/P = {sumT/sumP:.3f} vs pred {predT/sumP:.3f} (ratio {sumT/predT:.3f}); "
            f"P/A over 1/Sigma_1^log: mean {sum(ratios1)/len(ratios1):.3f} min {min(ratios1):.3f} max {max(ratios1):.3f}; "
            f"aggregate P/A = {sumP/sumA:.4f} vs pred {sumP/predA:.4f} (ratio {(sumP/sumA)/(sumP/predA):.3f}); "
            f"m=60: T/P {per[-1][1]:.3f} pred {per[-1][2]:.3f}, P/A {per[-1][3]:.4f} pred {per[-1][4]:.4f}; "
            f"m=3: T/P {per[2][1]:.3f} pred {per[2][2]:.3f}; m=5: {per[4][1]:.3f} pred {per[4][2]:.3f}; m=9: {per[8][1]:.3f} pred {per[8][2]:.3f}; m=30: {per[29][1]:.3f} pred {per[29][2]:.3f}")
    print(line); out.append(line)
    # the pure share bound inside the zone: L <= (log(mQ)/log Q)^2 ... report max L at m = 60 for the largest family
    Lmax = max(L(60, Q, a, b) for a, b in fams)
    print(f"   max L_60 over families with max<=60: {Lmax:.3f}; (log(61Q)/log(Q))^2 = {(math.log(61*Q)/math.log(Q))**2:.3f}")
open(os.path.join(res, "yield_check.txt"), "w").write("\n".join(out) + "\n")
