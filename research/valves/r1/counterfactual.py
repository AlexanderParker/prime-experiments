"""Counterfactual fuel: every proved valve law (imprint, port, inventory, onset, ember) uses only that the fuel
is an integer above Q, odd, coprime to q#. Replace the exhaust by F = {n > Q : gcd(n, q#) = 1, n = 1 (mod 3)}
(charges = s f with s q-smooth and f in F or f = 1). Then no pure charge can exist (f and f + 2 cannot both be
1 mod 3) while burnt families such as (1, 3) survive: the ledger has P_m = 0 and B_m = T_m > 0 in every turn,
and every structural law still holds (checked: imprint containment, onset, port, inventory, ember bound).
usage: uv run python research/valves/r1/counterfactual.py q Q [M]
"""
import sys, math, os, json
from collections import Counter
import numpy as np

q = int(sys.argv[1]); Q = int(sys.argv[2]); M = int(sys.argv[3]) if len(sys.argv) > 3 else 12
N = (M + 1) * Q + 2
engine = [p for p in range(2, q + 1) if all(p % d for d in range(2, int(p ** 0.5) + 1))]
qsharp = math.prod(engine)
nn = np.arange(0, N + 3, dtype=np.int64)
rest = nn.copy()
for p in engine:
    while True:
        m = (rest % p == 0); m[0] = False
        if not m.any():
            break
        rest[m] //= p
air = np.where(nn > 0, nn // np.maximum(rest, 1), 0); fuel = rest
# counterfactual open: fuel = 1 or (fuel > Q and fuel = 1 mod 3)
F = (fuel > Q) & (fuel % 3 == 1)
opn = (fuel == 1) | F
opn[0] = False
lo, hi = Q + 1, (M + 1) * Q
pair = np.flatnonzero(opn[lo:hi + 1] & opn[lo + 2:hi + 3]) + lo
s1 = air[pair]; s2 = air[pair + 2]; f1 = fuel[pair]; f2 = fuel[pair + 2]
turn = (pair - 1) // Q
pure = (s1 == 1) & (s2 == 1)
ember_member = (f1 == 1) | (f2 == 1)
# structural checks
viol = Counter()
for a, b, n in zip(s1.tolist(), s2.tolist(), pair.tolist()):
    m = (n - 1) // Q
    if max(a, b) <= Q and max(a, b) > m:                      # onset / cap
        viol["onset"] += 1
    if (a - b) % 2 or math.gcd(a, b) not in (1, 2) or (a % 2 == 0 and (a // 2 + b // 2) % 2 == 0):
        if max(a, b) <= Q:
            viol["inventory"] += 1
    for p in engine:                                          # imprint
        r = n % p
        if a % p == 0 and r != 0: viol["imprint"] += 1
        elif b % p == 0 and (r + 2) % p != 0: viol["imprint"] += 1
        elif a % p and b % p and (r == 0 or (r + 2) % p == 0): viol["imprint"] += 1
    port = 0 if n % 2 == 0 else n % 6
    pp = 0 if a % 2 == 0 else (3 if a % 3 == 0 else (1 if b % 3 == 0 else 5))
    if port != pp: viol["port"] += 1
rows = []
for m in range(1, M + 1):
    sel = turn == m
    T = int(sel.sum()); P = int((sel & pure).sum()); Be = int((sel & ~pure & ember_member).sum())
    E = int(((nn > m * Q) & (nn <= (m + 1) * Q) & (fuel == 1)).sum())
    fams = Counter(zip(s1[sel & ~pure & ~ember_member].tolist(), s2[sel & ~pure & ~ember_member].tolist()))
    rows.append((m, T, T - P, P, Be, E, len(fams), fams.most_common(4)))
    if m <= 2 and (T - P) != Be: viol["ember_law"] += 1
    if Be > 2 * E: viol["ember_bound"] += 1
print(f"counterfactual fuel F = {{n > Q, gcd(n, q#) = 1, n = 1 mod 3}}, q={q} Q={Q}: structural violations {dict(viol) or 0}")
for m, T, B, P, Be, E, nf, top in rows:
    print(f"m={m:2d} T={T} B={B} P={P} B_ember={Be} E={E} fuelled burnt families={nf} top={top}")
