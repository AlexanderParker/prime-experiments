"""Per-gear cumulative supply trajectories R_q(t) (mechanic round 4).

R_q(t) = root kills of gear q by depth t under lpf attribution: composite
member m of a prefix slot is attributed to lpf(m) (its smallest prime
factor, >= 5 for members). Supply identity sum_q R_q(t) = C(t) = 2t - P(t)
holds by construction and is re-verified per checkpoint.

Band signature: gears q <= sqrt(y) are active from slot 1 (they serve every
prefix - C4's active set); a FRESH gear q in (sqrt(y), y) has R_q = 0 until
its square q^2 enters at t_act = (q^2-1)/6 - k_lo + 1, then climbs the
layer-law staircase: R_q(t) = [q^2 in range] + #primes c in (q, m(t)/q]
+ T_q(t), where T_q counts composite cofactors c with lpf(c) >= q -
EXACTLY zero while m(t) < q^3. Verified here definitionally: R_q(t) is
compared per checkpoint against the spf-table count
    #{c in [ceil(m0/q), floor(m(t)/q)] : spf(c) >= q}
for every gear with y^2/q under the spf-table limit (all gears at y<=2003).

Load metrics per checkpoint (for the X-consistency equation):
    A(t)    = active gears = #{q : q <= sqrt(m(t))}
    C(t)    = 2t - P(t) total kills; mean load = C/A
    rho(t)  = 2(t-P)/(2t-P) - fraction of all kills X forces into doubles
    S_pair(t) = sum over gear pairs of nontrivial (cross) root-class counts
              <= t: the freedom-free pair-coincidence supply schedule
              (upper bound on n2; trivial same-member roots excluded)
    tau(t)  = (t - P(t))/S_pair(t) - X-demand share of the in-principle
              pair supply. In reality t-P <= n2 <= union <= S_pair holds
              identically; tau's peak locates where the schedule is tightest.

Outputs (append mode): research/data/supply_load.csv,
supply_pergear.csv (all gears at y <= 2003, ~24 representatives above).
Usage: uv run python research/supply_trajectory.py [y...]
Default ladder: 503 2003 10007 50021 (pairs skipped above 100k gears^2/2
budget only if --nopairs).
"""
import os
import sys
import time
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragile_census import primes_upto
from prefix_census import is_prime

SPF_LIMIT = 8_000_000


def spf_table(limit):
    """smallest prime factor for odd-and-3-coprime counting; full table."""
    spf = np.zeros(limit + 1, dtype=np.int32)
    for p in primes_upto(int(limit ** 0.5)):
        sl = spf[p * p::p]
        sl[sl == 0] = p
    return spf  # 0 means prime (or 0/1)


def checkpoints_for(W):
    cps = sorted({int(round(10 ** (i / 8))) for i in range(0, 200)
                  if 1 <= round(10 ** (i / 8)) <= W} | {W})
    return cps


def supply(y, seg=16_000_000, do_pairs=True):
    gears = [q for q in primes_upto(y) if q >= 5]
    G = len(gears)
    garr = np.array(gears, dtype=np.int64)
    k_lo = -((-(y - 1)) // 6)
    k_hi = (y * y + 1) // 6
    W = k_hi - k_lo + 1
    cps = checkpoints_for(W)
    uvals = [pow(6, -1, q) for q in gears]
    base = np.zeros(y + 2, dtype=np.int64)  # claims by gear value
    Pb = n0b = n2b = 0
    rec = {}  # t -> (P, n0, n2, Rq array over gears)
    for a in range(k_lo, k_hi + 1, seg):
        b = min(k_hi + 1, a + seg)
        n = b - a
        ownL = np.zeros(n, np.int32)
        ownR = np.zeros(n, np.int32)
        for q, u in zip(gears, uvals):
            v = ownL[(u - a) % q::q]
            v[v == 0] = q
            v = ownR[(-u - a) % q::q]
            v[v == 0] = q
        if a == k_lo:
            for arr, m in ((ownL, 6 * k_lo - 1), (ownR, 6 * k_lo + 1)):
                if m <= y and is_prime(m):
                    arr[0] = 0
        pl = ownL == 0
        pr = ownR == 0
        cP = np.cumsum(pl.astype(np.int64) + pr)
        c0 = np.cumsum(pl & pr)
        c2 = np.cumsum(~pl & ~pr)
        t_a = a - k_lo + 1  # global t of first slot in segment
        for t in cps:
            if t_a <= t <= b - k_lo:
                pos = t - t_a + 1
                cnt = np.bincount(ownL[:pos], minlength=y + 2).astype(np.int64)
                cnt += np.bincount(ownR[:pos], minlength=y + 2)
                Rq = (base + cnt)[garr]
                rec[t] = (Pb + int(cP[pos - 1]), n0b + int(c0[pos - 1]),
                          n2b + int(c2[pos - 1]), Rq)
        base += np.bincount(ownL, minlength=y + 2)
        base += np.bincount(ownR, minlength=y + 2)
        Pb += int(cP[-1])
        n0b += int(c0[-1])
        n2b += int(c2[-1])
    spair = {t: None for t in cps}
    if do_pairs:
        Ms, Rs = [], []
        for j in range(1, G):
            qj, uj = gears[j], uvals[j]
            for i in range(j):
                qi, ui = gears[i], uvals[i]
                M = qi * qj
                # 6k = +1 mod qi, -1 mod qj  ->  k = ui mod qi, -uj mod qj
                r = (ui + qi * (((-uj - ui) * pow(qi, -1, qj)) % qj)) % M
                Ms.append(M)
                Rs.append(r)
        Ms = np.array(Ms, dtype=np.int64)
        Rs = np.array(Rs, dtype=np.int64)
        for t in cps:
            b = k_lo + t - 1
            tot = 0
            for lo in range(0, len(Ms), 4_000_000):
                M = Ms[lo:lo + 4_000_000]
                for r in (Rs[lo:lo + 4_000_000], (-Rs[lo:lo + 4_000_000]) % M):
                    tot += int(((b - r) // M - (k_lo - 1 - r) // M).sum())
            spair[t] = tot
    return dict(y=y, W=W, k_lo=k_lo, gears=garr, cps=cps, rec=rec,
                spair=spair)


def verify_staircase(r):
    """R_q(t) vs spf-count identity + prime/composite cofactor split."""
    y, k_lo, gears = r["y"], r["k_lo"], r["gears"]
    m0 = 6 * k_lo - 1
    qmin = max(5, int(y * y / SPF_LIMIT) + 1)
    limit = min(SPF_LIMIT, y * y // qmin + 1)
    spf = spf_table(limit)
    isc = spf > 0  # composite (for c >= 2)
    # prefix counts of {c : spf(c) >= q} need per-q thresholding; do per gear
    checked = mism = 0
    tq_frac = []  # (q, T_q share at window end)
    vgears = [int(q) for q in gears if q >= qmin]
    # thin the verification set if huge: every gear at y<=2003, else stride
    if len(vgears) > 400:
        vgears = vgears[::len(vgears) // 400]
    cvals = np.arange(limit + 1)
    for q in vgears:
        ok_c = (~isc) | (spf >= q)  # spf>=q or prime
        ok_c[:2] = False
        cum = np.cumsum(ok_c)
        comp_ok = isc & (spf >= q)
        cumc = np.cumsum(comp_ok)
        gi = int(np.searchsorted(gears, q))
        for t in r["cps"]:
            m = 6 * (k_lo + t - 1) + 1
            c_lo = max(-((-m0) // q), q)  # c < q belongs to gear lpf(c) < q
            c_hi = m // q
            if c_hi > limit:
                continue
            want = int(cum[c_hi] - cum[c_lo - 1]) if c_hi >= c_lo else 0
            got = int(r["rec"][t][3][gi])
            checked += 1
            if want != got:
                mism += 1
                if mism <= 5:
                    print(f"  MISMATCH y={y} q={q} t={t}: R={got} vs {want}")
        m = 6 * (k_lo + r["cps"][-1] - 1) + 1
        c_lo = max(-((-m0) // q), q)
        c_hi = m // q
        if c_hi <= limit and c_hi >= c_lo:
            tot = int(cum[c_hi] - cum[c_lo - 1])
            tc = int(cumc[c_hi] - cumc[c_lo - 1])
            # composite cofactors other than c=q (the square is a step, not T)
            if c_lo <= q <= c_hi:
                tc -= 0  # c=q is prime cofactor path? c=q prime, not in comp
            if tot:
                tq_frac.append((q, tc / tot))
    return checked, mism, tq_frac


def main():
    args = [a for a in sys.argv[1:]]
    do_pairs = "--nopairs" not in args
    ys = [int(a) for a in args if not a.startswith("--")] or [
        503, 2003, 10007, 50021]
    ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(ddir, exist_ok=True)

    def opencsv(name, header):
        path = os.path.join(ddir, name)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        f = open(path, "a")
        if new:
            f.write(header + "\n")
        return f
    fl = opencsv("supply_load.csv",
                 "y,t,member,P,n0,n2,C,A_active,mean_load,g5_share,"
                 "rho,S_pair,tau")
    fg = opencsv("supply_pergear.csv", "y,q,t,R")
    for y in ys:
        t0 = time.time()
        pairs = do_pairs
        r = supply(y, do_pairs=pairs)
        gears, k_lo = r["gears"], r["k_lo"]
        G = len(gears)
        # representative gears for the per-gear CSV
        if G <= 350:
            reps = list(range(G))
        else:
            reps = sorted({0, 1, 2, 3} |
                          {int(round(i)) for i in
                           np.geomspace(4, G - 1, 18)} | {G - 2, G - 1})
        peak_tau, peak_t = -1.0, None
        for t in r["cps"]:
            P, n0, n2, Rq = r["rec"][t]
            m = 6 * (k_lo + t - 1) + 1
            C = 2 * t - P
            A = int(np.searchsorted(gears, math.isqrt(m), side="right"))
            assert int(Rq.sum()) == C, (y, t)  # supply identity, exact
            rho = 2 * (t - P) / C if C else 0.0
            sp = r["spair"][t]
            tau = (t - P) / sp if sp else float("nan")
            if sp and tau > peak_tau:
                peak_tau, peak_t = tau, t
            fl.write(f"{y},{t},{m},{P},{n0},{n2},{C},{A},"
                     f"{C / max(A, 1):.3f},{Rq[0] / C if C else 0:.4f},"
                     f"{rho:.4f},{sp if sp is not None else ''},"
                     f"{tau if sp else ''}\n")
            for i in reps:
                fg.write(f"{y},{gears[i]},{t},{Rq[i]}\n")
        checked, mism, tqf = verify_staircase(r)
        big = [f for q, f in tqf if q * q * q >= (y * y)]
        small = [(q, f) for q, f in tqf if q * q * q < (y * y)]
        Wt = r["cps"][-1]
        P, n0, n2, Rq = r["rec"][Wt]
        print(f"y={y} W={r['W']} P={P} n0={n0} n2={n2} "
              f"identity OK at {len(r['cps'])} cps | staircase: {checked} "
              f"checks, {mism} mismatches; T_q share q>y^(2/3): "
              f"max {max(big) if big else 0:.4f}; peak tau "
              f"{peak_tau:.4f} at t={peak_t} "
              f"(member {6*(k_lo+(peak_t or 1)-1)+1}) | {time.time()-t0:.0f}s")
        if small:
            qs, fs = zip(*small[:6])
            print(f"   T_q share below y^(2/3) (first gears verified): "
                  + " ".join(f"{q}:{f:.3f}" for q, f in small[:6]))
        sys.stdout.flush()
    fl.close()
    fg.close()
    print("wrote supply_load.csv, supply_pergear.csv")


if __name__ == "__main__":
    main()
