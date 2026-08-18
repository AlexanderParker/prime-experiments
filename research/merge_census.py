"""Constructor round 10, chunk 2: lemma 2 (fuel-merge control) - precise
statement and full merge census per consecutive step.

DEFINITION (as the tolerance ledger uses it): at step M -> M + q'
(consecutive chain), excess(M, q') = F(M+q') - F2(M). It is positive iff some
k >= 2 chain merge (k consecutive M-openings all deleted by q') beats the best
k = 1 merge (F2 = best adjacent pair). Census here, per step: every maximal
run of deleted openings, its k, merged value g_L + h_1 + ... + h_{k-1} + g_R,
interior gap residues mod q' (chain condition: h = 0 or +-2c), and the
anatomy at the maximum.

Steps 11->13 .. 19->23 in one full array; 23->29 chunked (P = 1.078e9).
Verification anchors: F_k(M+q') must equal 11, 18, 25, 34, 43; F2_k(M) from
round 9: 11, 16, 25, 31, 39.
"""
import numpy as np
from math import prod

STEPS = [(11, 13), (13, 17), (17, 19), (19, 23), (23, 29)]
GEARS = {y: [g for g in [5, 7, 11, 13, 17, 19, 23] if g <= y]
         for y in (11, 13, 17, 19, 23)}
F2K = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39}
FNEW = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43}


def exposed_chunk(gears, lo, hi):
    n = hi - lo
    arr = np.ones(n, bool)
    for q in gears:
        c = pow(6, -1, q)
        for a in (c, (q - c) % q):
            start = (a - lo) % q
            arr[start::q] = False
    return arr


def census(y, q1, chunk=40_000_000):
    gears = GEARS[y]
    P = prod(gears) * q1
    c1 = pow(6, -1, q1)
    kmax_hist = {}
    best = (0, None)                  # merged value, anatomy position
    prev_surv = None                  # last surviving opening position
    run_del = 0                       # deleted openings since prev_surv
    n_chains = 0
    for lo in range(0, P, chunk):
        hi = min(lo + chunk, P)
        pat = exposed_chunk(gears, lo, hi)
        pos = np.flatnonzero(pat).astype(np.int64) + lo
        if len(pos) == 0:
            continue
        r = pos % q1
        deleted = (r == c1) | (r == (q1 - c1) % q1)
        surv = np.flatnonzero(~deleted)
        if len(surv) == 0:            # whole chunk deleted (impossible here)
            run_del += len(pos)
            continue
        # boundary run: prev_surv .. first surviving of this chunk
        k0 = run_del + int(surv[0])
        if prev_surv is not None and k0 > 0:
            merged = int(pos[surv[0]]) - prev_surv
            n_chains += 1
            kmax_hist[k0] = kmax_hist.get(k0, 0) + 1
            if merged > best[0]:
                best = (merged, (prev_surv, int(pos[surv[0]]), k0))
        # interior runs, vectorised
        ks = np.diff(surv) - 1
        merges = np.diff(pos[surv])
        nz = np.flatnonzero(ks > 0)
        n_chains += len(nz)
        for k, cnt in zip(*np.unique(ks[nz], return_counts=True)):
            kmax_hist[int(k)] = kmax_hist.get(int(k), 0) + int(cnt)
        if len(nz):
            j = nz[np.argmax(merges[nz])]
            if int(merges[j]) > best[0]:
                best = (int(merges[j]), (int(pos[surv[j]]),
                                         int(pos[surv[j + 1]]), int(ks[j])))
        prev_surv = int(pos[surv[-1]])
        run_del = int(len(pos) - 1 - surv[-1])
    F_new = best[0]
    excess = F_new - F2K[y]
    print(f"\n=== step {y} -> {q1}: P={P:,}  chains={n_chains:,}  "
          f"k-hist {dict(sorted(kmax_hist.items()))}")
    print(f"  max merged = {F_new} (expect F_k({q1}) = {FNEW[q1]} "
          f"{'OK' if F_new == FNEW[q1] else 'MISMATCH'}); "
          f"F2_k(M) = {F2K[y]}; excess_k = {excess} "
          f"({3*excess}/{q1} = {3*excess/q1:.3f} adjacent-frame /q)")
    # anatomy of the argmax merge
    a, b, k = best[1]
    gs = GEARS[y]
    span = exposed_chunk(gs, a, b + 1)
    ops = np.flatnonzero(span) + a
    hs = np.diff(ops).tolist()
    res = [(int(o) % q1) for o in ops[1:-1]]
    print(f"  argmax anatomy: k={k}, gaps {hs} (g_L={hs[0]}, g_R={hs[-1]}, "
          f"interior {hs[1:-1]}), interior residues mod {q1}: {res} "
          f"(teeth at {c1}, {q1-c1})")
    gL, gR = hs[0], hs[-1]
    print(f"  g_L + g_R = {gL+gR} vs F2 = {F2K[y]} "
          f"({'<=' if gL+gR <= F2K[y] else '>'}); "
          f"interior sum = {sum(hs[1:-1])}; "
          f"decomposition: excess = interior_sum - (F2 - g_L - g_R) = "
          f"{sum(hs[1:-1]) - (F2K[y]-gL-gR)}")


if __name__ == "__main__":
    for y, q1 in STEPS:
        census(y, q1)
