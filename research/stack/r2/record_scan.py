"""The composite record on every section [q^2, q'^2), q prime, q'^2 <= N (default N = 10^9).

One segmented sieve of [1, N] collects every twin pair (lower member t, t + 2 both prime).  On the
anchor's 30-clock a slot is (30j + 11, 30j + 13), (30j + 17, 30j + 19) or (30j + 29, 30j + 31), slot
index s(n) = 3 j + {11: 0, 17: 1, 29: 2}[n mod 30] for a lower member n; every twin above 5 is a
slot.  The section [q^2, q'^2) holds the slots whose lower member lies in it; by the band theorem
(proof_skeleton.md section 5) a slot there is open under the machines below q' iff it is a twin,
so the longest run of struck slots of the composite machine inside the section is the longest
twin-free run of slots, counting the run from the section's first slot to the first twin and the
run from the last twin to the section's last slot.  Records are in slots; the section's length is
its slot count.

Also: the chain pairs r -> r'' = nextprime(r^2) -> nextprime(r''^2): the record on [r^2, r''^2)
against the record on [r''^2, nextprime(r''^2)^2).

Usage: uv run python research/stack/r2/record_scan.py [N]        (about 300 MB, one core)
"""
import sys, os, json, time, math
import numpy as np
import gmpy2

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
PI2_TABLE = {10**8: 440312, 10**9: 3424506}
SOFF = {11: 0, 17: 1, 29: 2}


def small_primes(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.flatnonzero(s).astype(np.int64)


def slot_index(n):
    """slot index of a lower member n (n = 11, 17, 29 mod 30), vectorised."""
    r = n % 30
    off = np.where(r == 11, 0, np.where(r == 17, 1, 2))
    return 3 * (n // 30) + off


def first_slot_at_or_above(c):
    """index of the first slot whose lower member is >= c."""
    j = c // 30
    for jj in (j, j + 1):
        for e, o in ((11, 0), (17, 1), (29, 2)):
            if 30 * jj + e >= c:
                return 3 * jj + o
    raise RuntimeError


def last_slot_below(c):
    """index of the last slot whose lower member is < c."""
    return first_slot_at_or_above(c) - 1


def collect_twins(N):
    t0 = time.time()
    P = small_primes(int(math.isqrt(N)) + 2)
    SEG = 1 << 24
    parts = []
    cum = 0
    checks = {}
    lo = 0
    while lo < N:
        hi = min(lo + SEG, N)
        L = hi - lo + 2
        a = np.ones(L, dtype=bool)
        if lo == 0:
            a[:2] = False
        for p in P.tolist():
            if p * p > hi + 2:
                break
            start = max(p * p, -(-lo // p) * p)
            if start < lo + L:
                a[start - lo::p] = False
        tw = np.flatnonzero(a[:-2] & a[2:]).astype(np.int64) + lo
        for x in PI2_TABLE:
            if lo <= x < hi:
                checks[x] = cum + int((tw <= x).sum())
        parts.append(tw)
        cum += len(tw)
        lo = hi
    tw = np.concatenate(parts)
    print(f"twins to {N:,}: {len(tw):,} in {time.time() - t0:.0f}s; checks {checks} against {PI2_TABLE}", flush=True)
    return tw, checks


def record_on(tw, lo, hi):
    """longest twin-free slot run inside [lo, hi) (lower members), in slots; also its position."""
    i0 = int(np.searchsorted(tw, lo))
    i1 = int(np.searchsorted(tw, hi))          # twins with lower member in [lo, hi): tw[i0:i1]
    s_first = first_slot_at_or_above(lo)
    s_last = last_slot_below(hi)
    nslots = s_last - s_first + 1
    t = tw[i0:i1]
    if len(t) == 0:
        return nslots, nslots, lo, 0, -1, -1
    si = slot_index(t)
    runs = np.diff(si) - 1                      # struck slots strictly between consecutive twins
    head = int(si[0] - s_first)                 # struck slots before the first twin
    tail = int(s_last - si[-1])
    best, where = head, lo
    if len(runs):
        k = int(runs.argmax())
        if runs[k] > best:
            best, where = int(runs[k]), int(t[k])
    if tail > best:
        best, where = tail, int(t[-1])
    return nslots, best, where, int(len(t)), int(t[0]), int(t[-1])


def main(N):
    tw, checks = collect_twins(N)
    np.save(os.path.join(RES, "twins_1e9.npy"), tw)
    P = small_primes(int(math.isqrt(N)) + 200)
    Pl = P.tolist()
    qs = [q for q in Pl if q >= 7]
    rows = []
    for i, q in enumerate(qs):
        qp = Pl[Pl.index(q) + 1]
        if qp * qp > N:
            break
        lo, hi = q * q, qp * qp
        nslots, best, where, ntw, tfirst, tlast = record_on(tw, lo, hi)
        rows.append((q, qp, lo, hi, nslots, best, where, ntw, tfirst, tlast))
    A = np.array(rows, dtype=np.int64)
    q = A[:, 0]; nslots = A[:, 4]; best = A[:, 5]; ntw = A[:, 7]
    ratio = best / nslots
    rec_numbers = best * 10.0                  # 3 slots per 30 numbers
    ln2 = np.log(A[:, 2].astype(float)) ** 2
    band = (rec_numbers >= 0.5 * ln2) & (rec_numbers <= 4 * ln2)
    out = {"N": N, "checks": {str(k): [v, PI2_TABLE[k]] for k, v in checks.items()},
           "sections": int(len(rows)), "empty_sections": int((ntw == 0).sum()),
           "max_ratio": float(ratio.max()), "argmax_ratio_q": int(q[int(ratio.argmax())]),
           "ratio_at_argmax": [int(best[int(ratio.argmax())]), int(nslots[int(ratio.argmax())])],
           "max_ratio_above": {str(a): [float(ratio[q >= a].max()), int(q[q >= a][int(ratio[q >= a].argmax())])]
                               for a in (100, 1000, 10000) if (q >= a).any()},
           "share_in_band_0.5_4_ln2": float(band.mean()), "n_below_band": int((rec_numbers < 0.5 * ln2).sum()),
           "n_above_band": int((rec_numbers > 4 * ln2).sum()),
           "rec_over_ln2_percentiles": {str(p): float(np.percentile(rec_numbers / ln2, p)) for p in (5, 25, 50, 75, 95, 99, 100)},
           "max_record_slots": int(best.max()), "argmax_record_q": int(q[int(best.argmax())]),
           "record_position_at_max": int(A[int(best.argmax()), 6]),
           "first_twin_offset_max": int((A[:, 8] - A[:, 2]).max()), "first_twin_offset_argmax_q": int(q[int((A[:, 8] - A[:, 2]).argmax())]),
           "last_twin_gap_to_end_max": int((A[:, 3] - A[:, 9]).max()), "last_twin_gap_argmax_q": int(q[int((A[:, 3] - A[:, 9]).argmax())])}
    # the ten largest ratios and the rows for q < 100, and decade summaries
    order = np.argsort(-ratio)[:12]
    out["top_ratio_rows"] = [dict(q=int(A[i, 0]), qp=int(A[i, 1]), slots=int(A[i, 4]), record=int(A[i, 5]),
                                  ratio=round(float(ratio[i]), 4), at=int(A[i, 6]), twins=int(A[i, 7]),
                                  first_twin=int(A[i, 8])) for i in order]
    out["rows_q_lt_100"] = [dict(q=int(r[0]), qp=int(r[1]), section=[int(r[2]), int(r[3])], slots=int(r[4]),
                                 record=int(r[5]), at=int(r[6]), twins=int(r[7]), first_twin=int(r[8]), last_twin=int(r[9]))
                            for r in rows if r[0] < 100]
    dec = []
    for a, b in ((7, 100), (100, 1000), (1000, 10000), (10000, 40000)):
        m = (q >= a) & (q < b)
        if m.any():
            dec.append(dict(q_range=[a, b], n=int(m.sum()), max_ratio=float(ratio[m].max()),
                            median_ratio=float(np.median(ratio[m])),
                            median_record_slots=float(np.median(best[m])), max_record_slots=int(best[m].max()),
                            median_rec_over_ln2=float(np.median((rec_numbers / ln2)[m])),
                            max_rec_over_ln2=float((rec_numbers / ln2)[m].max()),
                            median_slots=float(np.median(nslots[m]))))
    out["decades"] = dec
    # chain pairs
    pairs = []
    Pset = set(Pl)
    for r in Pl:
        if r < 3:
            continue
        r2 = int(gmpy2.next_prime(r * r))
        r3 = int(gmpy2.next_prime(r2 * r2))
        if r3 * r3 > N:
            break
        n1, b1, w1, t1, f1, l1 = record_on(tw, r * r, r2 * r2)
        n2, b2, w2, t2, f2, l2 = record_on(tw, r2 * r2, r3 * r3)
        pairs.append(dict(r=r, r2=r2, r3=r3, prev=[int(r * r), int(r2 * r2)], prev_slots=n1, prev_record=b1, prev_at=w1, prev_twins=t1,
                          next=[int(r2 * r2), int(r3 * r3)], next_slots=n2, next_record=b2, next_at=w2, next_twins=t2,
                          ratio=round(b2 / b1, 3) if b1 else None,
                          ln_ratio_sq=round((math.log(r2 * r2) / math.log(r * r)) ** 2, 3)))
    rat = np.array([p["ratio"] for p in pairs if p["ratio"]])
    out["chain_pairs_n"] = len(pairs)
    out["chain_ratio_median"] = float(np.median(rat))
    out["chain_ratio_min_max"] = [float(rat.min()), float(rat.max())]
    # residual correlation: log record against log r, rank correlation of residuals
    lr = np.log(np.array([p["r"] for p in pairs], float))
    a1 = np.log(np.array([p["prev_record"] for p in pairs], float))
    a2 = np.log(np.array([p["next_record"] for p in pairs], float))
    def resid(y):
        A_ = np.vstack([np.ones_like(lr), lr]).T
        coef, *_ = np.linalg.lstsq(A_, y, rcond=None)
        return y - A_ @ coef, coef
    e1, c1 = resid(a1)
    e2, c2 = resid(a2)
    from scipy.stats import spearmanr
    rho = spearmanr(e1, e2).correlation
    out["chain_resid_spearman"] = float(rho)
    out["chain_fit_prev_log_record_vs_log_r"] = [float(c1[0]), float(c1[1])]
    out["chain_fit_next_log_record_vs_log_r"] = [float(c2[0]), float(c2[1])]
    out["chain_pairs"] = pairs
    with open(os.path.join(RES, "record_scan.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(RES, "record_scan_rows.npz"), rows=A)
    print(json.dumps({k: v for k, v in out.items() if k not in ("chain_pairs", "rows_q_lt_100")}, indent=1))
    print("rows q < 100:")
    for r in out["rows_q_lt_100"]:
        print(r)
    print("chain pairs:")
    for p in pairs:
        print(p)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10**9)
