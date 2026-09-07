"""Exact turn scan: P_m(Q) = twins with lower member in (mQ, (m+1)Q], for every integer Q <= QMAX and
every turn m <= MMAX. Reports the empty cells (Q, m) with P_m(Q) = 0, the least Q_m from which turn m is
never empty, and the frontier-certified turns against the truth.

usage: uv run python research/valves/r2/turn_scan.py [QMAX] [MMAX]
"""
import sys, os, json
import numpy as np

QMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 5
MMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 100
here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "results"); os.makedirs(outdir, exist_ok=True)

N = (MMAX + 1) * QMAX + 4
s = np.ones(N + 1, dtype=bool); s[:2] = False
for i in range(2, int(N ** 0.5) + 1):
    if s[i]:
        s[i * i::i] = False
tw = s[:-2] & s[2:]                        # tw[n] = n and n+2 prime
C = np.concatenate([[0], np.cumsum(tw.astype(np.int64))])   # C[x] = twins with lower member <= x - 1
def count(a, b):                            # twins with lower member in (a, b]
    return C[b + 1] - C[a + 1]

Qs = np.arange(1, QMAX + 1, dtype=np.int64)
empties = {}
Qm = {}
minP = {}
for m in range(1, MMAX + 1):
    P = count(m * Qs, (m + 1) * Qs)
    z = Qs[P == 0]
    empties[m] = z.tolist()
    Qm[m] = int(z.max()) + 1 if len(z) else 1
    big = Qs >= 1000
    minP[m] = (int(P[big].min()), int(Qs[big][P[big].argmin()]))
print(f"turn scan: Q <= {QMAX}, m <= {MMAX}, twins sieved to {N}")
print("turn m: the Q with P_m(Q) = 0 (first ten shown), and Q_m = the least Q from which turn m is never empty")
for m in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100]:
    if m > MMAX:
        break
    z = empties[m]
    print(f"  m={m:3d}: {len(z):4d} empty Q, first {z[:8]}, last {z[-1] if z else None}, Q_m={Qm[m]}, "
          f"min P_m over Q>=1000: {minP[m][0]} at Q={minP[m][1]}")
# the largest Q at which SOME turn m <= MMAX is empty, and the largest Q where turn 1..3 are empty
last_any = max((max(v) if v else 0) for v in empties.values())
print(f"  largest Q with an empty turn among m <= {MMAX}: {last_any}; Q_m sequence m=1..{min(MMAX,30)}: {[Qm[m] for m in range(1, min(MMAX, 30) + 1)]}")
# existence in the valve itself: for Q <= 3000 check a twin in (Q, Q^2] directly
ex_bad = [int(Q) for Q in range(1, 3001) if count(Q, Q * Q) == 0] if N >= 3001 * 3001 else None
print(f"  existence in the valve (a twin in (Q, Q^2]) fails for Q in {ex_bad} (checked Q <= 3000)" if ex_bad is not None
      else "  existence check skipped (sieve too small)")
with open(os.path.join(outdir, f"turn_scan_{QMAX}_{MMAX}.json"), "w") as f:
    json.dump(dict(empties={m: v for m, v in empties.items()}, Qm=Qm, minP=minP), f)
