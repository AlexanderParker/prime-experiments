"""The chain from the base: links [g, g^2) along g -> nextprime(g^2), for every base.

Part A: one segmented sieve of [1, N] (N = 10^9) gives, for every prime g with g^2 <= N, the
twin pairs of the link [g, g^2) (lower member p with g <= p < g^2), the first and last of them,
the first twin above g^2, and the cumulative twin count pi_2 at g and at g^2.  The chains from
every base g_1 <= sqrt N are then read off the table.  pi_2(10^8) and pi_2(10^9) are printed as
a check of the sieve against the published values 440,312 and 3,424,506.

Part B: the base chain 3 -> 11 -> 127 -> ... continued with gmpy2 (probable primes above 3.3e24)
to g_10: for each g_k the first twin above g_k (lower member >= g_k) and its distance, and the
Hardy-Littlewood estimate of every link count that the sieve cannot reach, calibrated against
the published pi_2(10^k) table.

Usage: uv run python research/stack/r2/chain.py [N]      (default N = 10^9; about 300 MB)
"""
import sys, os, json, time, math
import numpy as np
import gmpy2
from gmpy2 import mpz

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

TWO_C2 = 1.3203236316
PI2_TABLE = {10**8: 440312, 10**9: 3424506, 10**10: 27412679, 10**11: 224376048,
             10**12: 1870585220, 10**13: 15834664872, 10**14: 135780321665,
             10**15: 1177209242304, 10**16: 10304195697298, 10**17: 90948839353159,
             10**18: 808675888577436}


def small_primes(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.flatnonzero(s).astype(np.int64)


def li2(x):
    """integral_2^x dt / ln^2 t by Simpson on a log grid."""
    if x <= 2:
        return 0.0
    # substitute t = e^s: integral of e^s / s^2 ds from ln 2 to ln x
    a, b = math.log(2), math.log(x)
    n = 20000
    h = (b - a) / n
    tot = 0.0
    for i in range(n + 1):
        s = a + i * h
        w = 1 if i in (0, n) else (4 if i % 2 else 2)
        tot += w * math.exp(s) / (s * s)
    return tot * h / 3


def part_a(N):
    t0 = time.time()
    G = int(math.isqrt(N))
    P = small_primes(G + 2)                       # primes to sqrt N (and a little more)
    gs = P[P * P <= N]                            # the bases / link starts
    # per g: pi_2 just below g (twins with lower member < g), pi_2 at g^2 - 1 (lower member < g^2),
    # first twin >= g, last twin < g^2, first twin > g^2
    n_g = len(gs)
    pi2_at_g = np.full(n_g, -1, np.int64)
    pi2_at_sq = np.full(n_g, -1, np.int64)
    first_ge_g = np.full(n_g, -1, np.int64)
    last_lt_sq = np.full(n_g, -1, np.int64)
    first_gt_sq = np.full(n_g, -1, np.int64)
    sq = gs * gs
    SEG = 1 << 24
    cum = 0
    pending_first_gt_sq = []                      # indices whose first twin above g^2 is not yet seen
    pending_first_ge_g = []
    last_twin_seen = -1
    checks = {}
    lo = 0
    while lo < N:
        hi = min(lo + SEG, N)
        L = hi - lo + 2                           # two extra numbers so that p + 2 is inside
        a = np.ones(L, dtype=bool)
        if lo == 0:
            a[:2] = False
        for p in P.tolist():
            if p * p > hi + 2:
                break
            start = max(p * p, -(-lo // p) * p)
            if start >= lo + L:
                continue
            a[start - lo::p] = False
        tw = np.flatnonzero(a[:-2] & a[2:]).astype(np.int64) + lo    # lower members in [lo, hi)
        # exclude the pair (2, 4)? 2 is prime, 4 is not; (3, 5) is a twin pair with lower member 3.
        # cumulative counts at the checkpoints
        for x in PI2_TABLE:
            if lo <= x < hi:
                checks[x] = cum + int((tw <= x).sum())
        if lo <= 10**8 < hi:
            checks[10**8] = cum + int((tw <= 10**8).sum())
        # per-g quantities whose position falls in this segment
        idx_g = np.flatnonzero((gs >= lo) & (gs < hi))
        for i in idx_g.tolist():
            g = int(gs[i])
            pi2_at_g[i] = cum + int((tw < g).sum())
            k = int(np.searchsorted(tw, g))
            if k < len(tw):
                first_ge_g[i] = int(tw[k])
            else:
                pending_first_ge_g.append(i)
        idx_s = np.flatnonzero((sq >= lo) & (sq < hi))
        for i in idx_s.tolist():
            s = int(sq[i])
            k = int(np.searchsorted(tw, s - 2))   # twins with both members below s: lower member <= s - 3
            pi2_at_sq[i] = cum + k
            # last twin below the square: lower member < s (a twin with lower member s is impossible)
            if k > 0:
                last_lt_sq[i] = int(tw[k - 1])
            else:
                last_lt_sq[i] = last_twin_seen
            k2 = int(np.searchsorted(tw, s, side="right"))
            if k2 < len(tw):
                first_gt_sq[i] = int(tw[k2])
            else:
                pending_first_gt_sq.append(i)
        if len(tw):
            if pending_first_gt_sq:
                for i in pending_first_gt_sq:
                    first_gt_sq[i] = int(tw[0])
                pending_first_gt_sq = []
            if pending_first_ge_g:
                for i in pending_first_ge_g:
                    first_ge_g[i] = int(tw[0])
                pending_first_ge_g = []
            last_twin_seen = int(tw[-1])
        cum += len(tw)
        lo = hi
        if (lo // SEG) % 10 == 0:
            print(f"  sieve at {lo:,} ({time.time() - t0:.0f}s, twins {cum:,})", flush=True)
    print("checks:", {x: (checks.get(x), PI2_TABLE[x]) for x in checks}, flush=True)
    link = pi2_at_sq - pi2_at_g                   # twins with lower member in [g, g^2)
    # the chains
    gset = {int(g): i for i, g in enumerate(gs.tolist())}
    chains = {}
    roots = []
    is_link_target = set()
    for i, g in enumerate(gs.tolist()):
        s = int(sq[i])
        nxt = int(gmpy2.next_prime(s))
        if nxt in gset:
            is_link_target.add(nxt)
    for i, g in enumerate(gs.tolist()):
        g = int(g)
        if g in is_link_target:
            continue
        roots.append(g)
        ch = []
        cur = g
        while cur in gset:
            j = gset[cur]
            ch.append({"g": cur, "square": int(sq[j]), "next": int(gmpy2.next_prime(int(sq[j]))),
                       "twins_in_link": int(link[j]), "first_twin_ge_g": int(first_ge_g[j]),
                       "dist_first_twin_from_g": int(first_ge_g[j] - cur),
                       "last_twin_in_link": int(last_lt_sq[j]),
                       "first_twin_above_square": int(first_gt_sq[j]),
                       "dist_first_twin_above_square": int(first_gt_sq[j] - sq[j])})
            cur = int(gmpy2.next_prime(int(sq[j])))
        chains[g] = ch
    out = {"N": N, "n_bases": int(n_g), "checks": {str(x): [checks.get(x), PI2_TABLE[x]] for x in checks},
           "empty_links": int((link == 0).sum()), "min_link": int(link.min()),
           "min_link_g": int(gs[int(link.argmin())]),
           "n_chains": len(chains), "roots_head": roots[:40],
           "chains_head": {str(g): chains[g] for g in roots if g <= 60},
           "seconds": time.time() - t0}
    # the per-g table, compact
    np.savez_compressed(os.path.join(RES, "chain_table.npz"), g=gs, link=link, first_ge_g=first_ge_g,
                        last_lt_sq=last_lt_sq, first_gt_sq=first_gt_sq, pi2_g=pi2_at_g, pi2_sq=pi2_at_sq)
    # distance statistics: first twin above g^2 in units of ln^2 g^2, and relative to nextprime(g^2)
    d = (first_gt_sq - sq).astype(float)
    l2 = np.log(sq.astype(float)) ** 2
    out["dist_above_square_over_ln2"] = {"mean": float((d / l2).mean()), "max": float((d / l2).max()),
                                         "argmax_g": int(gs[int((d / l2).argmax())]),
                                         "max_dist": int(d.max()), "argmax_dist_g": int(gs[int(d.argmax())])}
    dg = (first_ge_g - gs).astype(float)
    out["dist_from_g_over_ln2"] = {"mean": float((dg / np.log(gs.astype(float)) ** 2).mean()),
                                   "max": float((dg / np.log(gs.astype(float)) ** 2).max()),
                                   "argmax_g": int(gs[int((dg / np.log(gs.astype(float)) ** 2).argmax())]),
                                   "zeros": int((dg == 0).sum())}
    out["link_over_hl"] = {}
    for g in (127, 853, 2819, 16141, 29947):
        if g in gset:
            j = gset[g]
            hl = TWO_C2 * (li2(int(sq[j])) - li2(g))
            out["link_over_hl"][str(g)] = [int(link[j]), round(hl), round(int(link[j]) / hl, 4)]
    return out, chains


def part_b(kmax=10):
    t0 = time.time()
    # calibration of the Hardy-Littlewood integral against the table
    calib = {str(x): round(PI2_TABLE[x] / (TWO_C2 * li2(x)), 5) for x in PI2_TABLE}
    chain = []
    g = mpz(3)
    for k in range(1, kmax + 1):
        sq = g * g
        # first twin >= g
        p = g if gmpy2.is_prime(g) else gmpy2.next_prime(g)
        while not gmpy2.is_prime(p + 2):
            p = gmpy2.next_prime(p)
        t1 = p
        nxt = gmpy2.next_prime(sq)
        lg = float(gmpy2.log(g))
        rec = {"k": k, "g": str(g), "digits": len(str(g)), "log10_g": round(float(gmpy2.log10(g)), 3),
               "first_twin_ge_g": str(t1), "dist": int(t1 - g), "dist_over_ln2g": round(int(t1 - g) / lg ** 2, 3) if lg > 1 else None,
               "next": str(nxt), "gap_above_square": int(nxt - sq),
               "probable": bool(g > 3300000000000000000000000)}
        if float(gmpy2.log10(sq)) <= 18.5:
            x1, x0 = int(sq), int(g)
            rec["hl_link_estimate"] = round(TWO_C2 * (li2(x1) - li2(x0)))
        else:
            rec["hl_link_estimate"] = "beyond the table (g^2 > 10^18): log10 of the HL estimate = %.2f" % (
                math.log10(TWO_C2) + float(gmpy2.log10(sq)) - 2 * math.log10(float(gmpy2.log(sq))))
        chain.append(rec)
        print(f"  k={k} g has {rec['digits']} digits, first twin at distance {rec['dist']} ({time.time() - t0:.0f}s)", flush=True)
        g = nxt
    return {"calibration_table_over_hl": calib, "chain": chain, "seconds": time.time() - t0}


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
    out_a, chains = part_a(N)
    with open(os.path.join(RES, "chain_a.json"), "w") as f:
        json.dump(out_a, f, indent=1)
    with open(os.path.join(RES, "chains_all.json"), "w") as f:
        json.dump({str(k): v for k, v in chains.items()}, f)
    print(json.dumps({k: v for k, v in out_a.items() if k != "chains_head"}, indent=1))
    out_b = part_b(10)
    with open(os.path.join(RES, "chain_b.json"), "w") as f:
        json.dump(out_b, f, indent=1)
    print(json.dumps(out_b, indent=1))
