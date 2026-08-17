"""Round 5 lateral: the supply machinery at scale (y = 1009, 10007), and the
derivative scan - the map of WHERE reality is nearest to X-behaviour.

PART 1 - scale. The round-4 term enumeration was O(#products^2); pruned here by
splitting the labour along what each piece is good at:
  - PAIRSPLIT: evaluated directly by the gap law (split_gap_law.split_rep),
    O(pi(y)^2) closed-form class reps, no product enumeration at all;
    verified EXACTLY against the sieve incidence sum_k omega_l * omega_r
    (that equality is the at-scale test of the gap law: ~754k pairs at 10007).
  - U: pure arithmetic (u'(q) slots, partner-gearful test), verified == sieve.
  - everything else (SAME, B, hubs): vectorized sieve (one slice pair per gear),
    spot-verified against independent trial division on random slots.

PART 2 - the reality form of the flagship identity. With n_j = # slots with j
composite members: P = 2 n0 + n1, n2 = B - U, n0 = T_win (twin slots), so

    P(t) = t + T_win(t) - B(t) + U(t)      exactly, at every t
    (per-slot form:  dP = 1 + dT - dB + dU, verified elementwise)

Under X, T_win = 0: the binding defect IS the twin count. Reality is exactly
X-like on twin-free runs, where dP = 1 - d(n2) slot by slot. The scan maps the
long twin-free runs (the near-binding loci) and asks whether they sit on
special ground: prime-load (P-rate) and hub density vs ambient at same depth,
and where they live in the window (bottom band or not).

Run: uv run python research/derivative_scan.py    (from repo root; numpy)
"""
import random
from collections import defaultdict

import numpy as np

from tooth_sharing import isprime, uprime
from split_gap_law import primes, split_rep

# ---------- sieve at scale ----------

def sieve(y):
    K = (y * y - 1) // 6
    gears = primes(5, y)
    oml = np.zeros(K + 1, np.int8)
    omr = np.zeros(K + 1, np.int8)
    for q in gears:
        u = pow(6, -1, q)
        oml[u::q] += 1
        omr[(q - u) % q::q] += 1
    oml[0] = omr[0] = 0
    gvl = np.zeros(K + 1, bool)
    gvr = np.zeros(K + 1, bool)
    for q in gears:
        k = uprime(q)
        (gvl if q % 6 == 5 else gvr)[k] = True
    return K, gears, oml, omr, gvl, gvr

def spot_verify(y, K, gears, oml, omr, n=2000, seed=11):
    rng = random.Random(seed)
    bad = 0
    for _ in range(n):
        k = rng.randrange(1, K + 1)
        wl = sum(1 for q in gears if (6 * k - 1) % q == 0)
        wr = sum(1 for q in gears if (6 * k + 1) % q == 0)
        if wl != oml[k] or wr != omr[k]:
            bad += 1
    return bad

def u_slots_arith(gears, y):
    out = set()
    for q in gears:
        k = uprime(q)
        partner = 6 * k + 1 if q % 6 == 5 else 6 * k - 1
        if partner == q:
            partner = 6 * k - 1 if q % 6 == 5 else 6 * k + 1
        if not (partner > y and isprime(partner)):
            out.add(k)          # partner composite (gearful) or a gear itself
    return np.array(sorted(out))

# ---------- part 1 per scale ----------

def scale_report(y, do_pairs=True):
    print(f"--- y = {y} ---")
    K, gears, oml, omr, gvl, gvr = sieve(y)
    bad = spot_verify(y, K, gears, oml, omr)
    print(f"  K = {K}, gears = {len(gears)}; sieve spot-check vs trial division "
          f"on 2000 random slots: {bad} mismatches")
    cnt = oml.astype(np.int16) + omr
    killed = cnt > 0
    Bm = (oml > 0) & (omr > 0)
    Um = Bm & (gvl | gvr)
    n2m = Bm & ~gvl & ~gvr
    pl = gvl | (oml == 0)
    pr = gvr | (omr == 0)
    pl[0] = pr[0] = False
    Tm = pl & pr
    # per-slot identity dP = 1 + dT - dB + dU  <=>  P(t) = t + T - B + U at all t
    resid = (pl.astype(np.int8) + pr) - 1 - Tm + Bm - Um
    resid[0] = 0
    print(f"  per-slot identity (pl+pr) - 1 - T + B - U == 0: "
          f"max|resid| = {np.abs(resid).max()}  (=> P(t) = t + T(t) - B(t) + U(t) at EVERY t)")
    # U: arithmetic vs sieve
    Ua = u_slots_arith(gears, y)
    Us = np.flatnonzero(Um)
    okU = "OK" if len(Ua) == len(Us) and np.array_equal(Ua, Us) else "MISMATCH"
    # totals
    oc = int(cnt[killed].sum() - killed.sum())
    same = int((oml[oml > 0] - 1).sum() + (omr[omr > 0] - 1).sum())
    B = int(Bm.sum()); U = int(Um.sum()); n2 = int(n2m.sum()); T = int(Tm.sum())
    P = int(pl.sum() + pr.sum())
    hubs = int((cnt >= 3).sum())
    print(f"  totals: B {B}  U {U} [{okU}]  n2 {n2}  T_win {T}  P {P}  "
          f"overcount {oc}  hubs(cnt>=3) {hubs}")
    print(f"  bridge oc = SAME + U + n2: {same} + {U} + {n2} = {same + U + n2} "
          f"[{'OK' if same + U + n2 == oc else 'MISMATCH'}]")
    if do_pairs:
        inc_sieve = int(np.dot(oml.astype(np.int64), omr.astype(np.int64)))
        law_total = 0
        by_gap = defaultdict(int)
        for i in range(len(gears)):
            q = gears[i]
            for j in range(i + 1, len(gears)):
                qp = gears[j]
                Pq = q * qp
                x = split_rep(q, qp)
                c = 0
                for z in (x, Pq - x):
                    if z <= K:
                        c += (K - z) // Pq + 1
                if c:
                    by_gap[qp - q if qp - q <= 6 else 8] += c
                law_total += c
        okL = "OK" if law_total == inc_sieve else "MISMATCH"
        g2 = by_gap.get(2, 0)
        print(f"  PAIRSPLIT gap law {law_total} vs sieve incidence "
              f"sum(omega_l*omega_r) {inc_sieve} [{okL}]  "
              f"({len(gears)*(len(gears)-1)//2} pairs)")
        print(f"  by gap: g=2 {g2} ({100*g2/law_total:.1f}%), g=4 {by_gap.get(4,0)}, "
              f"g=6 {by_gap.get(6,0)}, g>6 {by_gap.get(8,0)}; "
              f"CORR_inc = {law_total - B - (law_total - inc_sieve)} ... "
              f"PAIRSPLIT - B = {law_total - B}")
    return K, cnt, pl, pr, Tm, n2m

# ---------- part 2: the derivative scan ----------

def stride_scan(y, K, cnt, pl, pr, Tm, n2m, top=12, nbins=100):
    print(f"  derivative scan (y={y}): near-binding loci = twin-free runs")
    ts = np.flatnonzero(Tm)
    strides = np.diff(ts)
    prate = (pl.astype(np.int32) + pr)
    hubm = (cnt >= 3).astype(np.int32)
    # ambient depth bins
    edges = np.linspace(1, K + 1, nbins + 1).astype(np.int64)
    Pbin = np.add.reduceat(prate, edges[:-1]) / np.diff(edges)
    Hbin = np.add.reduceat(hubm, edges[:-1]) / np.diff(edges)
    order = np.argsort(strides)[::-1]
    print(f"    twins in window {len(ts)}; max stride {strides.max()}; "
          f"top {top} runs (pos = midpoint/K):")
    print(f"    {'pos':>6} {'len':>6} {'P-rate':>7} {'ambient':>8} {'ratio':>6} "
          f"{'hub-rate':>8} {'ambient':>8} {'ratio':>6}")
    agg = []
    for idx in order[:top]:
        a, b = ts[idx] + 1, ts[idx + 1]          # interior slots a..b-1
        L = b - a
        mid = (a + b) // 2
        bin_i = min(nbins - 1, int(nbins * mid / (K + 1)))
        pr_run = prate[a:b].sum() / L
        hb_run = hubm[a:b].sum() / L
        agg.append((pr_run / Pbin[bin_i], hb_run / max(Hbin[bin_i], 1e-12)))
        print(f"    {mid/K:>6.3f} {L:>6} {pr_run:>7.3f} {Pbin[bin_i]:>8.3f} "
              f"{pr_run/Pbin[bin_i]:>6.3f} {hb_run:>8.3f} {Hbin[bin_i]:>8.3f} "
              f"{hb_run/max(Hbin[bin_i],1e-12):>6.3f}")
    # aggregate over top 1% of strides
    n1pct = max(len(strides) // 100, 10)
    prs, hbs, wts = [], [], []
    for idx in order[:n1pct]:
        a, b = ts[idx] + 1, ts[idx + 1]
        L = b - a
        mid = (a + b) // 2
        bin_i = min(nbins - 1, int(nbins * mid / (K + 1)))
        prs.append(prate[a:b].sum() / L / Pbin[bin_i])
        hbs.append(hubm[a:b].sum() / L / max(Hbin[bin_i], 1e-12))
        wts.append(L)
    w = np.array(wts, float)
    print(f"    top 1% strides ({n1pct} runs, length-weighted): "
          f"P-rate/ambient = {np.average(prs, weights=w):.4f}, "
          f"hub-rate/ambient = {np.average(hbs, weights=w):.4f}")
    # depth geometry: where do the long runs live?
    mids = (ts[order[:n1pct]] + ts[order[:n1pct] + 1]) / 2 / K
    print(f"    depth of top-1% strides: min {mids.min():.3f}, median "
          f"{np.median(mids):.3f}, max {mids.max():.3f} "
          f"(bottom band ends ~{uprime(y)/K:.5f})")
    # bottom band: max stride in first 1% of window vs global
    first = strides[ts[:-1] < K // 100]
    print(f"    max stride inside first 1% of window: {first.max() if len(first) else 0} "
          f"vs global {strides.max()}")

if __name__ == "__main__":
    print("=" * 72)
    print("PART 1+2: supply machinery at scale + derivative scan")
    for y in (1009, 10007):
        K, cnt, pl, pr, Tm, n2m = scale_report(y)
        stride_scan(y, K, cnt, pl, pr, Tm, n2m)
