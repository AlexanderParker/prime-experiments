"""Round 20 lateral: does the lag-pair correlation predict the measured joint
distribution of ADJACENT GAPS - the object Constructor's p_1 rests on?

CORRELATION RATIO (closed form, from the n-point formula):

    Lambda(g1,g2) = prod_q [ c_q(0,g1,g1+g2) * (q-2) / ( c_q(0,g1) * c_q(0,g2) ) ]

- the exposure correlation of two adjacent gaps RELATIVE TO INDEPENDENCE
(chained through the shared middle opening). Lambda = 1 means the two gaps'
endpoint arithmetic is independent; Lambda = 0 means the pair is structurally
impossible; Lambda < 1 is a deficit, which is what Constructor measures.

Measured against full-period joint histograms of consecutive gap pairs.
"""
from itertools import combinations
from math import prod
import numpy as np
from split_gap_law import primes

def c_q(q, offs):
    u = pow(6, -1, q); t = {u % q, (-u) % q}
    return sum(1 for r in range(q) if all((r + d) % q not in t for d in offs))

def lam(gears, g1, g2):
    v = 1.0
    for q in gears:
        num = c_q(q, [0, g1, g1 + g2]) * (q - 2)
        den = c_q(q, [0, g1]) * c_q(q, [0, g2])
        if den == 0:
            return float('nan')
        v *= num / den
    return v

def joint(y, chunk=40_000_000, cap=40):
    gears = primes(5, y)
    P = prod(gears)
    J = np.zeros((cap + 1, cap + 1), np.int64)
    M = np.zeros(cap + 1, np.int64)
    carry = None
    a = 0
    while a < P:
        S = min(chunk, P - a)
        killed = np.zeros(S, bool)
        for q in gears:
            u = pow(6, -1, q)
            for t in (u, q - u):
                killed[(t - a) % q::q] = True
        o = np.flatnonzero(~killed).astype(np.int64) + a
        if carry is not None:
            o = np.concatenate((carry, o))
        d = np.diff(o)
        M += np.bincount(d[d <= cap], minlength=cap + 1)
        m = (d[:-1] <= cap) & (d[1:] <= cap)
        np.add.at(J, (d[:-1][m], d[1:][m]), 1)
        carry = o[-2:]
        a += S
    return J, M, gears

print("=" * 78)
print("PART 4: predicted vs measured adjacent-gap correlation")
for y in (19, 23):
    J, M, gears = joint(y)
    N = M.sum()
    print(f"  --- machine {y} (gears {gears}), {N} gaps ---")
    print(f"  {'g1':>3} {'g2':>3} {'obs':>7} {'indep':>9} {'obs/indep':>10} "
          f"{'Lambda':>8} {'ratio':>7}")
    rows = []
    for g1 in range(4, 26):
        for g2 in range(4, 26):
            if M[g1] < 50 or M[g2] < 50:
                continue
            exp = M[g1] * M[g2] / N
            if exp < 2:
                continue
            L = lam(gears, g1, g2)
            obs = int(J[g1, g2])
            rows.append((g1, g2, obs, exp, obs / exp if exp else 0, L))
    rows.sort(key=lambda r: r[4])
    for g1, g2, obs, exp, oi, L in rows[:6] + rows[-4:]:
        r = oi / L if L and L == L and L > 0 else float('nan')
        print(f"  {g1:>3} {g2:>3} {obs:>7} {exp:>9.1f} {oi:>10.3f} "
              f"{L:>8.3f} {r:>7.2f}")
    oi = np.array([r[4] for r in rows]); La = np.array([r[5] for r in rows])
    ok = (La > 0) & (La == La)
    print(f"  over {ok.sum()} (g1,g2) cells: corr(log obs/indep, log Lambda) = "
          f"{np.corrcoef(np.log(oi[ok] + 1e-9), np.log(La[ok]))[0,1]:.3f}")
    zero = [(r[0], r[1], r[2]) for r in rows if r[5] == 0 or r[5] != r[5]]
    print(f"  cells where Lambda predicts IMPOSSIBLE: {len(zero)}; "
          f"observed counts there: {sorted({z[2] for z in zero})}")
