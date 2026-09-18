"""Single-gear killers of a stretch (the p = 29 mechanism, origin_mechanic.md section 3).

For the stretch (p^2, q^2) with twins T (columns open to every gear <= p): gear g is a
single-gear killer if re-phasing its rigid tooth pair {s+u, s-u} (u = 6^-1 mod g) by some shift s
lands a tooth on every twin AND every column that only g strikes at its real phase is still
struck at the new phase (else the shift opens a column).  Reports, per p, the killers, and the
largest p with one.

usage: uv run python single_gear_killers.py [PMAX]
"""
import sys
import numpy as np

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(PMAX + 200)
plist = [int(x) for x in P if 7 <= x <= PMAX]
last_p = None; total = 0
for p in plist:
    q = int(P[np.searchsorted(P, p) + 1])
    c0 = (p * p + 1) // 6 + 1; c1 = (q * q - 1) // 6 - 1
    if c1 < c0: continue
    L = c1 - c0 + 1
    gears = [int(h) for h in P if 5 <= h <= p]
    count = np.zeros(L, dtype=np.int32)       # number of gears striking each column
    strikes = {}
    for g in gears:
        u = pow(6, -1, g)
        m = np.zeros(L, dtype=bool)
        m[(u - c0) % g::g] = True; m[(g - u - c0) % g::g] = True
        strikes[g] = m
        count += m
    twins = np.nonzero(count == 0)[0]
    if len(twins) == 0:
        print(f"p={p}: DEAD stretch"); continue
    killers = []
    for g in gears:
        u = pow(6, -1, g)
        tw = (twins + c0) % g
        # need a shift s with tw subset of {s+u, s-u} mod g
        cands = set(((tw[0] - u) % g, (tw[0] + u) % g))
        ok_shifts = []
        for s in cands:
            cls = {(s + u) % g, (s - u) % g}
            if all(int(t) in cls for t in tw):
                # lone kills of g at the real phase must remain struck
                lone = np.nonzero(strikes[g] & (count == 1))[0]
                lone_cls = set(((lone + c0) % g).tolist())
                if lone_cls <= cls:
                    ok_shifts.append(s)
        if ok_shifts:
            killers.append((g, ok_shifts, len(tw), len(set(tw.tolist()))))
    if killers:
        last_p = p; total += 1
        print(f"p={p:5d} q={q:5d} cols={L:6d} twins={len(twins):4d}: killers " +
              ", ".join(f"g={g}(shifts {s}, twin classes {nc})" for g, s, nt, nc in killers))
print(f"\nstretches with a single-gear killer: {total} of {len(plist)}; largest p: {last_p}")
