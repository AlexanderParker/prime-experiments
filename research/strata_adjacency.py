"""Constructor round 10, chunk 1: stratified adjacency - can two top-stratum
address classes ever be adjacent, and the per-machine alpha1 finite check.

Frame: k-frame. alpha1 (adjacent/halved frame) target: F2 - F <= alpha1 * q'
<=> F2_k <= F_k + alpha1*q'/3. Dangerous pairs at level alpha1: adjacent gap
pairs (s1, s2) with s1 + s2 > F_k + alpha1*q'/3.

Certificate tiers for each dangerous (s1, s2):
  tier A (gears 5,7 only, machine-free): A3(s1 mod 35, s2 mod 35) = empty.
  tier B (class census mod 385): L_{s1} + s1 and L_{s2} disjoint mod 385,
         where L_s = left-endpoint classes of s-gaps (empirical census,
         verifiable by one period scan).
  tier C: residual - direct in-period check.
Top-stratum question (Lateral's live target): are the maximal-gap address
classes mod 385 ever adjacency-compatible (L_top + s meets L_top)?
"""
import numpy as np
from math import prod

GEARS = {13: [5, 7, 11, 13], 17: [5, 7, 11, 13, 17],
         19: [5, 7, 11, 13, 17, 19], 23: [5, 7, 11, 13, 17, 19, 23]}
QNEXT = {13: 17, 17: 19, 19: 23, 23: 29}


def exposed(gears, m):
    arr = np.ones(m, bool)
    for q in gears:
        c = pow(6, -1, q)
        arr[c::q] = False
        arr[(q - c) % q::q] = False
    return arr


E35 = set(np.flatnonzero(exposed([5, 7], 35)).tolist())


def A3(s1, s2):
    return [r for r in E35 if (r + s1) % 35 in E35
            and (r + s1 + s2) % 35 in E35]


def machine(y, alpha1):
    gears = GEARS[y]
    P = prod(gears)
    idx = np.flatnonzero(exposed(gears, P))
    gaps = np.diff(np.append(idx, idx[0] + P))
    F = int(gaps.max())
    q1 = QNEXT[y]
    B = F + alpha1 * q1 / 3
    pair = gaps + np.roll(gaps, -1)
    F2 = int(pair.max())
    print(f"\n=== gears<={y} (P={P:,})  F_k={F}  F2_k={F2}  q'={q1}  "
          f"alpha1={alpha1}: budget F2_k <= {B:.2f}  "
          f"{'HOLDS' if F2 <= B else 'FAILS'}")

    # left-endpoint classes mod 385 per size (the strata census)
    L = {}
    for s in range(1, F + 1):
        sel = np.flatnonzero(gaps == s)
        if len(sel):
            L[s] = set((idx[sel] % 385).tolist())

    # top-stratum adjacency at class level mod 385
    top = L[F]
    compat = {s: sorted(top & {(r - s) % 385 for r in top})
              for s in range(1, F + 1) if s in L}
    tt = compat.get(F, [])
    print(f"  top stratum: {len(top)} classes mod 385 {sorted(top)}; "
          f"top-top adjacency classes (r and r+F both top): {tt if tt else 'NONE'}")

    # dangerous pairs and certificate tiers
    dang = [(s1, s2) for s1 in range(1, F + 1) for s2 in range(1, F + 1)
            if s1 + s2 > B and s1 in L and s2 in L]
    tierA = [p for p in dang if not A3(p[0] % 35, p[1] % 35)]
    rest = [p for p in dang if p not in tierA]
    tierB = []
    tierC = []
    for s1, s2 in rest:
        inter = {(r + s1) % 385 for r in L[s1]} & L[s2]
        (tierB if not inter else tierC).append((s1, s2))
    # residual: check in-period whether any dangerous adjacency exists
    realized = []
    for s1, s2 in tierC:
        sel = np.flatnonzero((gaps == s1) & (np.roll(gaps, -1) == s2))
        if len(sel):
            realized.append((s1, s2, len(sel)))
    print(f"  dangerous pairs (sum > {B:.2f}): {len(dang)}; "
          f"tier A (A3 empty): {len(tierA)}; tier B (mod-385 class "
          f"disjoint): {len(tierB)}; tier C residual: {len(tierC)}")
    if tierC:
        print(f"    tier C pairs: {tierC}; realized in period: "
              f"{realized if realized else 'NONE - certificate closes'}")
    ok = not realized
    print(f"  ALPHA1 = {alpha1} CERTIFICATE: "
          f"{'CLOSES' if ok and F2 <= B else 'FAILS (budget or realization)'}")
    return F, F2


if __name__ == "__main__":
    # explicit proof-of-concept at y=13 with alpha1 = 1; others at 4/3
    machine(13, 1.0)
    for y in (17, 19, 23):
        machine(y, 4 / 3)
