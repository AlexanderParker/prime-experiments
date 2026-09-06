"""R4.a - clutch facts II: joint moments of the top gears on the bottom machine's
openings, and the Brun (inclusion-exclusion) truncations.

Pair count X_{g,h} = # bottom-open columns in [0,P) struck by both g and h, against
4N/(gh); triples against 8N/(ghk).  Truncations of S(z) at orders 1, 2, 3.

Usage: uv run python research/anchor235/r59/period_moments.py 11 13 17 19 23
"""
import sys, os, json, math, time, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def primes_upto(n):
    if n < 2:
        return np.zeros(0, dtype=np.int64)
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = False
    return np.nonzero(s)[0].astype(np.int64)


def crt2(a1, m1, a2, m2):
    inv = pow(m1, -1, m2)
    t = ((a2 - a1) * inv) % m2
    return (a1 + m1 * t) % (m1 * m2)


def count_class(mask, a, M, P):
    """# bottom-open columns k in [0,P) with k = a (mod M)."""
    if a >= P:
        return 0
    idx = np.arange(a, P, M, dtype=np.int64)
    return int(mask[idx].sum())


def run(q, log, npairs=15000, ntriples=4000, seed=7):
    t0 = time.time()
    rnd = random.Random(seed)
    bottom = [int(p) for p in primes_upto(max(q, 5)) if 5 <= p <= q]
    m = len(bottom)
    P = 1
    N = 1
    for g in bottom:
        P *= g
        N *= (g - 2)
    six_P = 6 * P
    Z = math.isqrt(six_P + 1)
    tops = [int(p) for p in primes_upto(Z) if p > q]
    bound = 2 * 3 ** m
    mask = np.ones(P, dtype=bool)
    for g in bottom:
        u = pow(6, -1, g)
        mask[u::g] = False
        mask[(g - u) % g:: g] = False
    out = dict(q=q, P=P, N=N, Z=Z, n_top=len(tops), bound_3m=bound)
    log(f"=== MOMENTS q={q}  P={P}  N={N}  Z={Z}  top gears {len(tops)}  bound {bound}")

    # ---- pairs ----
    allpairs = [(tops[i], tops[j]) for i in range(len(tops))
                for j in range(i + 1, len(tops))]
    if len(allpairs) > npairs:
        pairs = rnd.sample(allpairs, npairs)
    else:
        pairs = allpairs
    rows = []
    for g, h in pairs:
        ug, uh = pow(6, -1, g), pow(6, -1, h)
        M = g * h
        tot = 0
        for ag in (ug % g, (g - ug) % g):
            for ah in (uh % h, (h - uh) % h):
                tot += count_class(mask, crt2(ag, g, ah, h), M, P)
        pred = 4.0 * N / M
        rows.append((g, h, tot, pred))
    dev_lt = [abs(t - p) for g, h, t, p in rows if g * h < P]
    dev_ge = [abs(t - p) for g, h, t, p in rows if g * h >= P]
    out["pairs"] = dict(n=len(rows), n_lt_P=len(dev_lt),
                        max_dev_lt_P=max(dev_lt) if dev_lt else 0.0,
                        max_dev_ge_P=max(dev_ge) if dev_ge else 0.0,
                        exceptions=sum(1 for d in dev_lt if d >= bound))
    log(f"  [P4] pairs {len(rows)}: max |X - 4N/gh| = "
        f"{max(dev_lt) if dev_lt else 0:.2f} for gh < P ({len(dev_lt)} pairs), "
        f"{max(dev_ge) if dev_ge else 0:.2f} for gh >= P ({len(dev_ge)}); "
        f"exceptions above {bound}: {out['pairs']['exceptions']}")
    # buckets by gh/P
    buck = {}
    for g, h, t, p in rows:
        r = g * h / P
        b = int(math.floor(math.log10(r))) if r > 0 else -99
        buck.setdefault(b, []).append((abs(t - p), p))
    bl = []
    for b in sorted(buck):
        v = buck[b]
        bl.append([b, len(v), sum(x[0] for x in v) / len(v),
                   sum(x[1] for x in v) / len(v)])
    out["pair_buckets"] = bl
    log("      bucket log10(gh/P): n, mean|dev|, mean prediction")
    for b in bl:
        log(f"        {b[0]:>3}: n={b[1]:>6}  mean|dev|={b[2]:.3f}  mean pred={b[3]:.4g}")
    out["cross1_gh"] = 4.0 * N     # gh where the prediction crosses 1
    log(f"      prediction 4N/(gh) crosses 1 at gh = {4*N} = {4*N/P:.3f} P")

    # ---- triples ----
    trows = []
    n = len(tops)
    for _ in range(min(ntriples, 4000)):
        i, j, k = sorted(rnd.sample(range(n), 3))
        g, h, r = tops[i], tops[j], tops[k]
        M = g * h * r
        tot = 0
        ug, uh, ur = pow(6, -1, g), pow(6, -1, h), pow(6, -1, r)
        for ag in (ug % g, (g - ug) % g):
            for ah in (uh % h, (h - uh) % h):
                agh = crt2(ag, g, ah, h)
                for ar in (ur % r, (r - ur) % r):
                    tot += count_class(mask, crt2(agh, g * h, ar, r), M, P)
        trows.append((M, tot, 8.0 * N / M))
    d_lt = [abs(t - p) for M, t, p in trows if M < P]
    d_ge = [abs(t - p) for M, t, p in trows if M >= P]
    out["triples"] = dict(n=len(trows), n_lt_P=len(d_lt),
                          max_dev_lt_P=max(d_lt) if d_lt else 0.0,
                          max_dev_ge_P=max(d_ge) if d_ge else 0.0,
                          exceptions=sum(1 for d in d_lt if d >= bound))
    log(f"  [P4] triples {len(trows)}: max dev {max(d_lt) if d_lt else 0:.2f} for "
        f"ghk < P ({len(d_lt)}), {max(d_ge) if d_ge else 0:.2f} for ghk >= P; "
        f"exceptions {out['triples']['exceptions']}")

    # ---- Brun truncations ----
    kmin = (q + 1) // 6 + 1
    N_range = int(mask[kmin:].sum())
    zs = [z for z in (40, 60, 100, 200, 400, 1000, 2000) if z <= Z]
    if not zs:
        zs = [Z]
    trunc = []
    for z in zs:
        gs = [g for g in tops if g <= z]
        if not gs:
            continue
        # exact S(z)
        alive = mask.copy()
        for g in gs:
            u = pow(6, -1, g)
            alive[u % g:: g] = False
            alive[(g - u) % g:: g] = False
        S = int(alive[kmin:].sum())
        del alive
        # order 1, 2, 3 terms
        t1 = 0.0
        for g in gs:
            u = pow(6, -1, g)
            c = 0
            for a in (u % g, (g - u) % g):
                idx = np.arange(a, P, g, dtype=np.int64)
                c += int(mask[idx].sum())
            t1 += c
        t2 = 0.0
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                g, h = gs[i], gs[j]
                ug, uh = pow(6, -1, g), pow(6, -1, h)
                for ag in (ug % g, (g - ug) % g):
                    for ah in (uh % h, (h - uh) % h):
                        t2 += count_class(mask, crt2(ag, g, ah, h), g * h, P)
        t3 = 0.0
        if len(gs) <= 100:
            for i in range(len(gs)):
                for j in range(i + 1, len(gs)):
                    for k in range(j + 1, len(gs)):
                        g, h, r = gs[i], gs[j], gs[k]
                        ug, uh, ur = pow(6, -1, g), pow(6, -1, h), pow(6, -1, r)
                        M = g * h * r
                        for ag in (ug % g, (g - ug) % g):
                            for ah in (uh % h, (h - uh) % h):
                                agh = crt2(ag, g, ah, h)
                                for ar in (ur % r, (r - ur) % r):
                                    t3 += count_class(mask, crt2(agh, g * h, ar, r),
                                                      M, P)
        o1 = N_range - t1
        o2 = o1 + t2
        o3 = o2 - t3
        trunc.append([z, len(gs), S, o1, o2, o3, t2, t3])
        log(f"  [P6] z={z:>5} gears={len(gs):>3}  S={S:>10}  order1={o1:>13.0f}  "
            f"order2={o2:>13.0f}  order3={o3:>13.0f}  (t2={t2:.0f}, t3={t3:.0f}, "
            f"order2 error {o2 - S:+.0f}, order3 term {t3:.0f})")
    out["truncations"] = trunc
    out["N_range"] = N_range
    out["seconds"] = time.time() - t0
    with open(os.path.join(RES, f"moments_q{q}.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"  done in {out['seconds']:.1f}s")


def main():
    qs = [int(x) for x in sys.argv[1:]] or [11, 13, 17, 19, 23]
    fh = open(os.path.join(RES, "moments.log"), "a", encoding="utf-8")

    def log(msg):
        print(msg)
        fh.write(msg + "\n")
        fh.flush()

    for q in qs:
        run(q, log)
    fh.close()


if __name__ == "__main__":
    main()
