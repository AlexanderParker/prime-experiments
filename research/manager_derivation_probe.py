"""Manager derivation probe - round 26. Tests pre-registered P-M2, P-M3, P-M4
(research/data/r26/manager_derivation_prereg.md) exactly at the four direct-scannable
steps. P-M1 needs the Q* machinery and is deferred to the lanes' results.

All exact integers; asserts throughout; read-only over the corpus.
"""
import numpy as np

def tooth(q): return pow(6, -1, q)

def blocked_mask(gears):
    P = int(np.prod([int(g) for g in gears]))
    idx = np.arange(P); m = np.zeros(P, dtype=bool)
    for q in gears:
        u = tooth(q)
        m |= (idx % q == u % q) | (idx % q == (q - u) % q)
    return m, P

def gaps_of(mask):
    o = np.flatnonzero(~mask)
    d = np.diff(o)
    wrap = int(o[0]) + len(mask) - int(o[-1])
    return o, np.append(d, wrap)  # cyclic: gap i is o[i] -> o[i+1] (last wraps)

STEPS = [([5,7,11], 13), ([5,7,11,13], 17), ([5,7,11,13,17], 19), ([5,7,11,13,17,19], 23)]

print(f"{'step':>16} {'F_old':>5} {'F2_old':>6} {'F_new':>5} {'incr':>4} {'K':>2} "
      f"{'s_max':>5} {'PM4_rhs':>7} {'PM4':>4} {'PM3_tight':>9} depths")

pm2_ok = pm3_tight_steps = pm4_ok = 0
for old, qp in STEPS:
    mO, PO = blocked_mask(old)
    oO, gO = gaps_of(mO)
    F_old = int(gO.max())
    # F_2(M): max cyclic adjacent pair sum
    F2_old = int((gO + np.roll(gO, -1)).max())
    mN, PN = blocked_mask(old + [qp])
    oN, gN = gaps_of(mN)
    F_new = int(gN.max())
    incr = F_new - F2_old
    u = tooth(qp); s = (2 * u) % qp; s_max = max(s, qp - s)

    # decompose EVERY record window of the new machine
    rec_idx = np.flatnonzero(gN == F_new)
    depths, pm3_any_tight, pm4_all = [], False, True
    for ri in rec_idx:
        a = int(oN[ri]); b = a + F_new  # window in absolute coords (may exceed PN via wrap)
        inside = [k for k in range(a + 1, b) if not mO[k % PO]]
        K = len(inside)  # kills
        depths.append(K + 1)  # gaps in window
        pts = [a] + inside + [b]
        gaps_w = [pts[i+1] - pts[i] for i in range(len(pts) - 1)]
        if K >= 1:
            # P-M3: flank + first interior <= F_2(M), tight?
            left_pair = gaps_w[0] + gaps_w[1]
            right_pair = gaps_w[-2] + gaps_w[-1]
            assert left_pair <= F2_old and right_pair <= F2_old, "adjacency bound violated?!"
            if left_pair == F2_old or right_pair == F2_old:
                pm3_any_tight = True
            # P-M4: incr <= (K-1) * s_max  (only meaningful for K>=1)
            if not (incr <= max(0, (K - 1)) * s_max or incr <= 0):
                pm4_all = False
        else:
            # K=0: pure old gap survived as record - then F_new <= F_old <= F2_old, incr <= 0
            pass
    ok4 = (incr <= 0) or all(
        incr <= max(0, (d - 2)) * s_max for d in depths)  # d-1 kills = d-... K = d-1
    # recompute cleanly: K = depth-1
    ok4 = (incr <= 0) or all(incr <= max(0, (d - 1 - 1)) * s_max for d in depths)
    pm4_ok += ok4
    pm2 = (incr == 0 and max(depths) <= 2) or (incr > 0 and max(depths) >= 3)
    pm2_ok += pm2
    pm3_tight_steps += pm3_any_tight
    print(f"{str(old[-1])+'->'+str(qp):>16} {F_old:>5} {F2_old:>6} {F_new:>5} {incr:>4} "
          f"{max(d-1 for d in depths):>2} {s_max:>5} "
          f"{max(0,(max(depths)-2))*s_max:>7} {'OK' if ok4 else 'FAIL':>4} "
          f"{'YES' if pm3_any_tight else 'no':>9} {sorted(set(depths))}")

print(f"\nP-M2 (incr>0 iff depth>=3): {pm2_ok}/4 steps")
print(f"P-M3 (adjacency tight at some record window): {pm3_tight_steps}/4 steps")
print(f"P-M4 (incr <= (K-1)*s_max): {pm4_ok}/4 steps")
print("done - exact, no fits")
