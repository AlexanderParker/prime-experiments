"""The start-of-band excess and the gear zone, on a segment above each band's start.

A band of machine k is [g_k^2, g_{k+1}^2) with the lower machines 1..k-1 given as prime ranges
[lo_i, hi_i] (all their gears are below g_k) and the anchor 2, 3, 5 as the clock.  On the segment
[A, A + L), A = g_k^2, the script marks for every slot (n, n + 2) with lower member n >= A:
  open_i     : neither member has a gear of machine i (i < k) as a factor,
  rough      : open under every lower machine (both members g_k-rough),
  open_k     : neither member has a factor in [g_k, g_k^2]  (via the cofactor: S4),
  twin       : both members prime.
Per bin (doubling cycle blocks from the start, the zone Z_k = [g_k^2, g_k g_k'), and u-bins of
width 0.25, u = ln x / ln g_k) it reports the counts, the independence product, the ratio
twins / (slots x prod_i P_i(open)), its three factors P(twin | rough pair) x R_lower x 1/P_k, and
the per-number Buchstab check P(prime | rough number) against 1 / (u omega(u)).
It also lists, for the first cycles, every rough composite (machine k's genuine strikes on the
rough set) with its factorisation: the gear zone.

Usage: uv run python research/stack/r2/excess.py     (all bands of q = 7..23 and the base chain)
"""
import sys, os, json, time, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
LMAX = 40_000_000
TWO_C2 = 1.3203236316


def small_primes(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.flatnonzero(s).astype(np.int64)


def nextprime_list(P, x):
    return int(P[np.searchsorted(P, x, side="right")])


def seg_marks(lo, L, primes):
    """True where no prime of the list divides the number lo + i."""
    a = np.ones(L, dtype=bool)
    for p in primes:
        start = -(-lo // p) * p
        if start < lo + L:
            a[start - lo::p] = False
    return a


def omega(u):
    """Buchstab's function on [1, 3]: 1/u on [1, 2]; (1 + ln(u - 1))/u on [2, 3]; beyond 3 by
    numerical integration of (u omega(u))' = omega(u - 1)."""
    if u <= 1:
        return 0.0
    if u <= 2:
        return 1 / u
    if u <= 3:
        return (1 + math.log(u - 1)) / u
    # numerical: tabulate u omega(u) from 3 upward
    h = 1e-3
    grid = [3.0]
    val = [1 + math.log(2)]
    def om(v):
        if v <= 2:
            return 1 / v
        if v <= 3:
            return (1 + math.log(v - 1)) / v
        i = int((v - 3) / h)
        i = min(i, len(val) - 1)
        return val[i] / grid[i]
    while grid[-1] < u:
        g = grid[-1]
        val.append(val[-1] + h * om(g + h / 2 - 1))
        grid.append(g + h)
    return val[-1] / grid[-1]


def band_run(label, P, machines, k, A, B, gk, gkp, Lmax=LMAX, ncyc_list=16):
    """machines: list of (lo, hi) for machines 1..k-1 (index i = 1..k-1); machine k = [gk, gk^2]."""
    t0 = time.time()
    L = int(min(B - A, Lmax))
    L = L + 2
    hi = A + L
    # primality on [A, hi)
    isp = seg_marks(A, L, P[P * P <= hi].tolist())
    # lower machines
    lower_marks = []
    for (lo, hi_m) in machines:
        gears = P[(P >= lo) & (P <= hi_m)].tolist()
        lower_marks.append(seg_marks(A, L, gears))
    # cofactor after dividing out every prime below gk (anchor and all lower machines)
    R = np.arange(A, A + L, dtype=np.int64)
    for p in P[P < gk].tolist():
        pa = p
        while pa < hi:
            start = -(-A // pa) * pa
            if start < hi:
                R[start - A::pa] //= p
            pa *= p
    # open under machine k: R == 1, or R > gk^2 and R prime (R == n: isp; R < n: R lies in (gk^2, n/7], inside the segment)
    n_arr = np.arange(A, A + L, dtype=np.int64)
    Rprime = np.zeros(L, dtype=bool)
    m_eq = R == n_arr
    Rprime[m_eq] = isp[m_eq]
    m_lt = (R < n_arr) & (R > gk * gk) & (R < hi)
    Rprime[m_lt] = isp[R[m_lt] - A]
    open_k_num = (R == 1) | ((R > gk * gk) & Rprime)
    del Rprime, m_eq, m_lt
    # slots: lower member n = A + i with n mod 30 in {11, 17, 29}, n + 2 < hi
    i_all = np.arange(L - 2, dtype=np.int64)
    n_all = A + i_all
    r30 = n_all % 30
    is_slot = (r30 == 11) | (r30 == 17) | (r30 == 29)
    idx = i_all[is_slot]
    n_slot = n_all[is_slot]
    j_slot = n_slot // 30
    twin = isp[idx] & isp[idx + 2]
    opens = [lm[idx] & lm[idx + 2] for lm in lower_marks]
    rough = np.ones(len(idx), dtype=bool)
    for o in opens:
        rough &= o
    open_k = open_k_num[idx] & open_k_num[idx + 2]
    # per-number quantities on rough numbers (for Buchstab): numbers in the segment coprime to 30
    cop = np.ones(L, dtype=bool)
    for d in (2, 3, 5):
        cop[(-A % d)::d] = False
    rough_num = cop.copy()
    for lm in lower_marks:
        rough_num &= lm
    prime_num = isp & cop
    # sanity: a rough number below gk^2 ... (none in the segment); rough composites = rough & ~prime
    rough_comp = rough_num & ~isp
    # the gear zone: the first rough composites above the square, with least prime factor
    rc_idx = np.flatnonzero(rough_comp)[:40]
    gear_zone = []
    for i in rc_idx.tolist():
        n = A + i
        lpf = None
        for p in P[P >= gk].tolist():
            if n % p == 0:
                lpf = int(p)
                break
            if p * p > n:
                break
        gear_zone.append([int(n), lpf, int(n // lpf) if lpf else None])
    # exact check of the two zone statements on the segment: for every rough composite n below x, its
    # least prime factor g satisfies g^2 <= n (new strike) and its gears' rough multiple g*gk <= n.
    rc_all = np.flatnonzero(rough_comp)
    zone_len = gk * (gkp - gk)
    n_rc_in_zone = int(((rc_all + A) < A + zone_len).sum())     # must be exactly 1 (the square) if the square is in the segment
    # zone slots: lower member in [A, A + zone_len - 2)
    inz = n_slot + 2 < A + zone_len
    zone = {"len": int(zone_len), "slots": int(inz.sum()), "rough_pairs": int((rough & inz).sum()),
            "twins": int((twin & inz).sum()), "rough_composites_in_zone": n_rc_in_zone,
            "twins_eq_rough_pairs": bool(np.array_equal(twin & inz, rough & inz)),
            "open_k_slot_fraction": float((open_k & inz).mean()) if inz.any() else None,
            "square_slot_lower": int(A - 2), "square_mod30": int(A % 30)}
    out = {"label": label, "k": k, "gk": int(gk), "gkp": int(gkp), "A": int(A), "B": int(B), "L": int(L - 2),
           "u_max": math.log(A + L) / math.log(gk), "zone": zone, "gear_zone_head": gear_zone,
           "machines_lower": [[int(a), int(b)] for a, b in machines]}

    def bin_stats(mask, ulo, uhi):
        ns = int(mask.sum())
        if ns == 0:
            return None
        tw = int((twin & mask).sum())
        rp = int((rough & mask).sum())
        fr = [float((o & mask).sum() / ns) for o in opens]
        pk = float((open_k & mask).sum() / ns)
        prod_lower = float(np.prod(fr)) if fr else 1.0
        prod = prod_lower * pk
        # per number on the same cycles
        jlo, jhi = int(j_slot[mask].min()), int(j_slot[mask].max())
        nlo, nhi = 30 * jlo, min(30 * (jhi + 1), A + L)
        sl = slice(max(nlo - A, 0), nhi - A)
        rn = int(rough_num[sl].sum())
        pn = int((prime_num[sl]).sum())
        cn = int(cop[sl].sum())
        um = 0.5 * (ulo + uhi)
        bu = 1 / (um * omega(um)) if um > 1 else None
        return {"u": [round(ulo, 3), round(uhi, 3)], "cycles": jhi - jlo + 1, "slots": ns, "twins": tw,
                "rough_pairs": rp, "P_lower": fr, "P_k": pk, "product": prod,
                "ratio": (tw / ns) / prod if prod > 0 else None,
                "P_twin_given_rough": tw / rp if rp else None,
                "R_lower": (rp / ns) / prod_lower if prod_lower > 0 else None,
                "inv_P_k": 1 / pk if pk > 0 else None,
                "numbers_coprime30": cn, "rough_numbers": rn, "primes": pn,
                "P_prime_given_rough_number": pn / rn if rn else None,
                "buchstab": bu, "P_rough_number": rn / cn if cn else None,
                "P_prime_number": pn / cn if cn else None}
    lng = math.log(gk)
    u_of = lambda n: math.log(n) / lng
    # doubling cycle blocks from the first cycle j0 = A // 30 (the cycle containing the square)
    j0 = int(A // 30)
    blocks = []
    a = 0
    w = 1
    while True:
        m = (j_slot >= j0 + a) & (j_slot < j0 + a + w)
        if not m.any():
            break
        st = bin_stats(m, u_of(30 * (j0 + a) + 11), u_of(min(30 * (j0 + a + w) + 11, A + L)))
        if st:
            st["j"] = [j0 + a, j0 + a + w]
            blocks.append(st)
        a += w
        w *= 2
    out["blocks"] = blocks
    # the zone as a bin
    if inz.any():
        out["zone_bin"] = bin_stats(inz, u_of(A), u_of(A + zone_len))
    # u-bins
    ubins = []
    u0 = u_of(A)
    u1 = u_of(A + L - 2)
    edges = [u0] + [x for x in np.arange(math.ceil(u0 * 4) / 4, u1, 0.25)] + [u1]
    for ua, ub in zip(edges[:-1], edges[1:]):
        m = (n_slot >= math.exp(ua * lng)) & (n_slot < math.exp(ub * lng))
        st = bin_stats(m, ua, ub)
        if st:
            ubins.append(st)
    out["ubins"] = ubins
    # the first cycles one by one
    first = []
    for jj in range(j0, j0 + ncyc_list):
        m = j_slot == jj
        if not m.any():
            continue
        rcs = [int(A + i) for i in rc_all.tolist() if 30 * jj <= A + i < 30 * (jj + 1)]
        first.append({"j": jj, "slots": int(m.sum()), "rough_pairs": int((rough & m).sum()),
                      "twins": int((twin & m).sum()), "twin_lower": [int(x) for x in n_slot[twin & m]],
                      "rough_pair_lower": [int(x) for x in n_slot[rough & m]],
                      "rough_composites": rcs})
    out["first_cycles"] = first
    # the first twin and the first rough pair above the square
    tw_idx = np.flatnonzero(twin)
    rp_idx = np.flatnonzero(rough)
    out["first_twin_lower"] = int(n_slot[tw_idx[0]]) if len(tw_idx) else None
    out["first_rough_pair_lower"] = int(n_slot[rp_idx[0]]) if len(rp_idx) else None
    out["first_twin_dist_cycles"] = (int(n_slot[tw_idx[0]]) - A) / 30 if len(tw_idx) else None
    out["seconds"] = time.time() - t0
    print(f"{label}: A={A} L={L-2} u_max={out['u_max']:.2f} zone {zone} first twin {out['first_twin_lower']} ({out['seconds']:.0f}s)", flush=True)
    return out


def main():
    P = small_primes(3_000_000)
    results = []
    # the q-stacks, as in stacked_squares.py: engine 7..q, machine 2 = [q', q'^2], machine k+1 = [nextprime(g_k^2), .^2]
    for q in (7, 11, 13, 17, 19, 23):
        qsharp = 1
        for p in P[P <= q].tolist():
            qsharp *= p
        machines = [(7, q)]
        g = nextprime_list(P, q)
        while g * g <= qsharp:
            machines.append((g, g * g))
            g = nextprime_list(P, g * g)
        # bands: band k = [g_k^2, g_{k+1}^2) capped at q# + 1
        gks = [7] + [m[0] for m in machines[1:]] + [g]
        for k in range(1, len(machines) + 1):
            gk = gks[k - 1]
            A = gk * gk
            B = min(gks[k] * gks[k], qsharp + 1)
            if B <= A:
                continue
            lower = machines[:k - 1]
            results.append(band_run(f"q={q} band {k}", P, lower, k, A, B, gk, nextprime_list(P, gk)))
    # the base chain: g_1 = 3 (machine 1 = {7} on the slots; 3, 5 are the anchor), g_2 = 11, g_3 = 127, g_4 = 16139
    chain_g = [3, 11, 127, 16139, 260467367]
    chain_machines = [(7, 7), (11, 121), (127, 16129), (16139, 16139 * 16139)]
    for k in range(2, 5):
        gk = chain_g[k - 1]
        A = gk * gk
        B = chain_g[k] * chain_g[k]
        lower = chain_machines[:k - 1]
        results.append(band_run(f"chain band {k} (g={gk})", P, lower, k, A, B, gk, nextprime_list(P, gk)))
    with open(os.path.join(RES, "excess.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
