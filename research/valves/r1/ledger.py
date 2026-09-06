"""The turn ledger: per turn m = (mQ, (m+1)Q], m = 1..M, the total charges T_m, the burnt charges B_m
(split into fuelled-both and ember-carrying), the pure charge P_m, the embers E_m, the prime-led charges
A_m, and the families present. Checks the ledger identities and the family bookkeeping against the
inventory (admissible pairs with max(s, s') <= m).

usage: uv run python research/valves/r1/ledger.py q Q [M]
Writes results/ledger_q{q}_Q{Q}.json and results/ledger_q{q}_Q{Q}.txt. Prints a bounded summary.
"""
import sys, json, math, os
from collections import Counter, defaultdict
import numpy as np

q = int(sys.argv[1]); Q = int(sys.argv[2]); M = int(sys.argv[3]) if len(sys.argv) > 3 else 60
M = min(M, Q)
N = (M + 1) * Q + 2
here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "results"); os.makedirs(outdir, exist_ok=True)


def sieve(n):
    is_p = np.ones(n + 1, dtype=bool); is_p[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i::i] = False
    return is_p


isprime = sieve(N + 2)
primes_all = np.flatnonzero(isprime)
engine = [int(p) for p in primes_all if p <= q]
gears_mid = primes_all[(primes_all > q) & (primes_all <= Q)]

mopen = np.ones(N + 3, dtype=bool)
for p in gears_mid:
    mopen[p::p] = False
mopen[0] = False

lo = Q + 1; hi = (M + 1) * Q  # lower members n in (Q, (M+1)Q]
pair = (np.flatnonzero(mopen[lo:hi + 1] & mopen[lo + 2:hi + 3]) + lo).astype(np.int64)
n1 = pair; n2 = pair + 2


def air_fuel(x):
    r = x.copy()
    for p in engine:
        while True:
            m = (r % p == 0)
            if not m.any():
                break
            r[m] //= p
    return x // r, r


s1, f1 = air_fuel(n1); s2, f2 = air_fuel(n2)
bad_fuel = int(np.sum(~((f1 == 1) | ((f1 > Q) & isprime[f1])))) + int(np.sum(~((f2 == 1) | ((f2 > Q) & isprime[f2]))))
pure = (s1 == 1) & (s2 == 1)
turn = (n1 - 1) // Q
ember_member = (f1 == 1) | (f2 == 1)
burnt = ~pure
bfuel = burnt & ~ember_member
bember = burnt & ember_member

# all numbers in (Q, (M+1)Q]: embers and primes per turn, prime-led charges
nn = np.arange(lo, hi + 1, dtype=np.int64)
tn = (nn - 1) // Q
sm_air, sm_fuel = air_fuel(nn[mopen[lo:hi + 1]])
ember_pos = nn[mopen[lo:hi + 1]][sm_fuel == 1]
E = np.bincount((ember_pos - 1) // Q, minlength=M + 2)
prime_pos = nn[isprime[lo:hi + 1]]
PI = np.bincount((prime_pos - 1) // Q, minlength=M + 2)
A = np.bincount((prime_pos[mopen[prime_pos + 2]] - 1) // Q, minlength=M + 2)       # P with P+2 open
Ap = np.bincount((prime_pos[mopen[prime_pos - 2]] - 1) // Q, minlength=M + 2)      # P with P-2 open

# inventory: admissible pairs with max <= M
sm = [1]
for p in engine:
    sm = sorted({x * p ** e for x in sm for e in range(0, 40) if x * p ** e <= M})
adm = set()
for a in sm:
    for b in sm:
        if (a - b) % 2 == 0 and math.gcd(a, b) in (1, 2) and (a % 2 == 1 or (a // 2 + b // 2) % 2 == 1):
            adm.add((a, b))


def wfam(a, b):
    v = (2.0 if a % 2 == 0 else 1.0) / (a * b)   # even pairs: the partner's parity is forced by the class of P, doubling its prime chance
    for p in engine:
        if p > 2 and (a % p == 0 or b % p == 0):
            v *= (p - 1) / (p - 2)
    return v


rows = []
viol_cap = 0; viol_ember2 = 0; fam_missing_total = 0; fam_extra_total = 0
for m in range(1, M + 1):
    sel = turn == m
    T = int(sel.sum()); P = int((sel & pure).sum()); Bf = int((sel & bfuel).sum()); Be = int((sel & bember).sum())
    B = Bf + Be
    fams = Counter(zip(s1[sel & bfuel].tolist(), s2[sel & bfuel].tolist()))
    # cap check: every fuelled burnt pair has max(s, s') <= m
    cap_bad = sum(c for (a, b), c in fams.items() if max(a, b) > m)
    viol_cap += cap_bad
    adm_m = {k for k in adm if max(k) <= m}
    present = set(fams) | ({(1, 1)} if P > 0 else set())
    missing = sorted(adm_m - present); extra = sorted(present - adm_m)
    fam_missing_total += len(missing); fam_extra_total += len(extra)
    # ember-carrying pairs per ember: at most 2
    if Be > 2 * E[m]:
        viol_ember2 += 1
    # ember families in this turn (label = the ember side)
    ef = Counter()
    for a, b, fa, fb in zip(s1[sel & bember].tolist(), s2[sel & bember].tolist(), f1[sel & bember].tolist(), f2[sel & bember].tolist()):
        ef[("e", b) if fa == 1 else (a, "e")] += 1
    top = fams.most_common(3)
    maxburnt = top[0] if top else None
    n13 = fams.get((1, 3), 0); n31 = fams.get((3, 1), 0)
    # prime-led split: primes P in turn m with P+2 open = pure + (1, s') fuelled + (1, ember)
    lead1 = sum(c for (a, b), c in fams.items() if a == 1) + sum(c for k, c in ef.items() if k[0] == 1)
    sigma_m = sum(wfam(a, b) for (a, b) in adm_m)
    sigma1_m = sum(wfam(a, b) for (a, b) in adm_m if a == 1)
    rows.append({"m": m, "T": T, "B": B, "Bfuel": Bf, "Bember": Be, "P": P, "E": int(E[m]), "pi": int(PI[m]),
                 "A": int(A[m]), "Aprime": int(Ap[m]), "lead1": lead1, "families": len(present), "A_adm": len(adm_m),
                 "missing": missing[:6], "extra": extra[:6], "ember_families": len(ef),
                 "top_burnt": [[list(k), c] for k, c in top], "N13": n13, "N31": n31,
                 "B_over_T": (B / T if T else None), "T_over_P": (T / P if P else None),
                 "P_over_A": (P / A[m] if A[m] else None), "Sigma_m": sigma_m, "Sigma1_m": sigma1_m,
                 "fams_m_le12": ({f"{a},{b}": c for (a, b), c in sorted(fams.items())} if m <= 12 else None)})

summary = {"q": q, "Q": Q, "M": M, "range_hi": hi + 2, "bad_fuel": bad_fuel, "charges": int(len(pair)),
           "pure_total": int(pure.sum()), "viol_cap": viol_cap, "viol_ember2": viol_ember2,
           "fam_missing_total": fam_missing_total, "fam_extra_total": fam_extra_total, "rows": rows}
with open(os.path.join(outdir, f"ledger_q{q}_Q{Q}.json"), "w") as f:
    json.dump(summary, f, indent=1, default=int)
with open(os.path.join(outdir, f"ledger_q{q}_Q{Q}.txt"), "w") as f:
    f.write(f"q={q} Q={Q} M={M}\n m | T_m | B_m | B_fuel | B_ember | P_m | E_m | pi_m | A_m | fam | adm | B/T | T/P | P/A | Sigma_m | 1/Sigma1\n")
    for r in rows:
        f.write(f"{r['m']:3d} | {r['T']:7d} | {r['B']:7d} | {r['Bfuel']:7d} | {r['Bember']:4d} | {r['P']:6d} | {r['E']:4d} | {r['pi']:6d} | {r['A']:6d} | {r['families']:3d} | {r['A_adm']:3d} | "
                f"{(r['B_over_T'] or 0):.4f} | {(r['T_over_P'] or 0):.3f} | {(r['P_over_A'] or 0):.4f} | {r['Sigma_m']:.3f} | {1/r['Sigma1_m']:.4f}\n")
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
for r in rows[:12] + rows[-1:]:
    print(f"m={r['m']:2d} T={r['T']} B={r['B']} (fuel {r['Bfuel']}, ember {r['Bember']}) P={r['P']} E={r['E']} pi={r['pi']} A={r['A']} lead1={r['lead1']} "
          f"fam={r['families']}/{r['A_adm']} miss={r['missing'][:3]} B/T={(r['B_over_T'] or 0):.4f} T/P={(r['T_over_P'] or 0):.3f} Sig={r['Sigma_m']:.3f} P/A={(r['P_over_A'] or 0):.4f} pred={1/r['Sigma1_m']:.4f}")
