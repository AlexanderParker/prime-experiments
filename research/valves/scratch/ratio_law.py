"""Family counts against the local-density prediction built from the imprint:
N_pred(s, s') = (1 / phi(b)) * C(s, s') * integral_{P_lo}^{Q^2/s} dP / (log P * log((sP+2)/s'))
with b = s' (odd) or s'/2 (even), P_lo = max(Q, (s'Q - 2)/s), and
C = 2 C_2 * prod_{odd p | s} (p-1)/(p-2) * prod_{odd p | s'} (p-1)^2 / (p (p-2)) * (1/2 if s = 2 mod 4 else 1).

usage: uv run python research/valves/scratch/ratio_law.py q Q
"""
import sys, os, math, csv
from sympy import factorint, totient, primerange

q = int(sys.argv[1]); Q = int(sys.argv[2])
here = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(here, "results", f"families_q{q}_Q{Q}.csv"))))
C2 = 1.0
for p in primerange(3, 10 ** 6):
    C2 *= 1 - 1 / (p - 1) ** 2
twoC2 = 2 * C2


def pred(s, sp):
    if sp % 2 == 1:
        b = sp
    else:
        b = sp // 2
    C = twoC2
    for p in factorint(s):
        if p > 2:
            C *= (p - 1) / (p - 2)
    for p in factorint(sp):
        if p > 2:
            C *= (p - 1) ** 2 / (p * (p - 2))
    if s % 4 == 2:
        C *= 0.5
    lo = max(Q, (sp * Q - 2) / s); hi = Q * Q / s
    if hi <= lo:
        return 0.0
    # Simpson on a log grid
    n = 2000
    xs = [lo * (hi / lo) ** (i / n) for i in range(n + 1)]
    f = [1 / (math.log(x) * math.log((s * x + 2) / sp)) for x in xs]
    tot = 0.0
    for i in range(n):
        tot += 0.5 * (f[i] + f[i + 1]) * (xs[i + 1] - xs[i])
    return C * tot / int(totient(b))


print(f"q={q} Q={Q}  (s, s')  N_measured  N_pred  ratio")
out = []
for r in rows:
    s, sp, N = int(r["s"]), int(r["s2"]), int(r["count"])
    if max(s, sp) > Q or N < 5000:
        continue
    pv = pred(s, sp)
    out.append((s, sp, N, pv, N / pv if pv else float("nan")))
for s, sp, N, pv, ratio in sorted(out, key=lambda t: -t[2])[:40]:
    print(f"({s},{sp}) {N} {pv:.0f} {ratio:.4f}")
ratios = [t[4] for t in out]
print("families", len(out), "ratio mean %.4f min %.4f max %.4f" % (sum(ratios) / len(ratios), min(ratios), max(ratios)))
