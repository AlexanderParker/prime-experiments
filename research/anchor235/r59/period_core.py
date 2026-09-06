"""R4.a - clutch facts I: moments, twisted copies, survivor curve, switching, placement.

Bottom machine {5..q} over one full period P = prod g (columns), openings N = prod (g-2).
Top machine = the primes in (q, Z], Z = isqrt(6P+1), same construction, no exemption.

  1. first moments  X_g  against 2N/g  on the bottom machine's openings   (P1)
  2. the twisted-copy identity, cofactor sets                             (P2)
  3. the survivor curve S(z) and the twin identity                       (P3)
  4. the bilinear g-m switching                                          (P5)
  5. the placement view                                                  (P8, P9)

Usage:  uv run python research/anchor235/r59/period_core.py 11 13 17 19 23
"""
import sys, os, json, math, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
GAMMA = 0.5772156649015329


def primes_upto(n):
    if n < 2:
        return np.zeros(0, dtype=np.int64)
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = False
    return np.nonzero(s)[0].astype(np.int64)


def home_col(g):
    return (g + 1) // 6 if g % 6 == 5 else (g - 1) // 6


def run(q, log):
    t0 = time.time()
    bottom = [int(p) for p in primes_upto(max(q, 5)) if 5 <= p <= q]
    m = len(bottom)
    P = 1
    N_pred = 1
    for g in bottom:
        P *= g
        N_pred *= (g - 2)
    six_P = 6 * P
    Z = math.isqrt(six_P + 1)
    tops = [int(p) for p in primes_upto(Z) if p > q]
    kmin = (q + 1) // 6 + 1
    bound = 2 * 3 ** m
    out = dict(q=q, m=m, P=P, N_pred=N_pred, six_P=six_P, Z=Z,
               n_top=len(tops), kmin=kmin, bound_3m=bound)
    log(f"=== q={q}  m={m}  P={P}  N={N_pred}  6P={six_P}  Z={Z}  "
        f"top gears={len(tops)}  kmin={kmin}  2*3^m={bound}")

    mask = np.ones(P, dtype=bool)
    for g in bottom:
        u = pow(6, -1, g)
        mask[u::g] = False
        mask[(g - u) % g:: g] = False
    N = int(mask.sum())
    assert N == N_pred
    N_range = int(mask[kmin:].sum())
    out["N"] = N
    out["N_range"] = N_range
    log(f"  bottom openings N={N}; in [kmin,P): {N_range}")

    # ---------- 1 + 2 + 4: one pass per gear ----------
    dev_max, dev_arg, rel_max, rel_arg = 0.0, None, 0.0, None
    ncopies = ncof = nmis = 0
    tw_period_checks = []
    maxm = six_P // tops[0] + 2
    isprime = np.zeros(maxm + 1, dtype=bool)
    isprime[primes_upto(maxm)] = True
    n_inc = n_one = n_prime = n_comp = n_prime_leZ = 0
    onetooth = 1.0
    for h in bottom:
        onetooth *= (1.0 - 1.0 / (h - 1))
    pred_prime = 0.0
    picount = np.cumsum(isprime.astype(np.int64))     # pi(x) for x <= maxm

    strike_counts = []
    for gi_, g in enumerate(tops):
        u = pow(6, -1, g)
        Xtot = 0
        for eps, a in ((+1, u % g), (-1, (g - u) % g)):
            idx = np.arange(a, P, g, dtype=np.int64)
            sel = mask[idx]
            Xtot += int(sel.sum())
            kk = idx[sel]
            mem = 6 * kk - eps
            mact = mem // g
            # twisted machine on t (k = a + g t, m = m0 + 6 t)
            T = len(idx)
            m0 = (6 * a - eps) // g
            alive_t = np.ones(T, dtype=bool)
            for h in bottom:
                gih = pow(g, -1, h)
                uh = pow(6, -1, h)
                for tooth in (0, (-eps * 2 * gih) % h):
                    alive_t[((tooth - m0) * uh) % h:: h] = False
            mpred = m0 + 6 * np.nonzero(alive_t)[0].astype(np.int64)
            if len(mpred) != len(mact) or not np.array_equal(mpred, mact):
                nmis += 1
            ncopies += 1
            ncof += len(mact)
            # switching / cofactor classification, restricted to k >= kmin
            mm = mact[kk >= kmin]
            n_inc += len(mm)
            one = int((mm == 1).sum())
            n_one += one
            pr = isprime[mm]
            npr = int(pr.sum())
            n_prime += npr
            n_comp += len(mm) - one - npr
            n_prime_leZ += int((mm[pr] <= Z).sum())
            # sieve prediction for prime cofactors: primes in (q, X] in one class mod 6,
            # thinned by the twisted tooth (density (h-2)/(h-1) at each bottom gear)
            X = int(mm.max()) if len(mm) else 0
            if X > q:
                pred_prime += (int(picount[X]) - int(picount[q])) * 0.5 * onetooth
            del idx, sel, kk, mem, mact, alive_t, mpred, mm, pr
        strike_counts.append((g, Xtot))
        fair = 2.0 * N / g
        d = abs(Xtot - fair)
        if d > dev_max:
            dev_max, dev_arg = d, g
        if d / fair > rel_max:
            rel_max, rel_arg = d / fair, g

    out["moment1"] = dict(max_abs_dev=dev_max, at_gear=dev_arg, max_rel_dev=rel_max,
                          at_gear_rel=rel_arg, bound=bound, n_cells=len(tops),
                          exceptions=0 if dev_max < bound else 1)
    log(f"  [P1] max |X_g - 2N/g| = {dev_max:.2f} at g={dev_arg} (bound {bound}); "
        f"max rel {rel_max:.4f} at g={rel_arg}")
    out["twisted"] = dict(copies=ncopies, cofactors=ncof, mismatches=nmis)
    log(f"  [P2] twisted copies {ncopies}, cofactors {ncof}, mismatches {nmis}")

    # twisted machine over a full m-period, sample of gears
    for g in (tops[0], tops[len(tops) // 2], tops[-1]):
        for eps in (+1, -1):
            u = pow(6, -1, g)
            a = u % g if eps == +1 else (g - u) % g
            m0 = (6 * a - eps) // g
            al = np.ones(P, dtype=bool)
            for h in bottom:
                gih = pow(g, -1, h)
                uh = pow(6, -1, h)
                for tooth in (0, (-eps * 2 * gih) % h):
                    al[((tooth - m0) * uh) % h:: h] = False
            tw_period_checks.append([g, eps, int(al.sum())])
            del al
    out["twisted_period_counts"] = tw_period_checks
    log(f"  [P2] twisted openings per m-period 6P: "
        f"{[c[2] for c in tw_period_checks]} (N={N})")

    # ---------- 3: survivor curve ----------
    alive = mask.copy()
    curve = []
    surv = N_range
    for g in tops:
        u = pow(6, -1, g)
        i1 = np.arange(u % g, P, g, dtype=np.int64)
        i2 = np.arange((g - u) % g, P, g, dtype=np.int64)
        idx = np.concatenate((i1, i2))
        a = alive[idx]
        kk = idx[a]
        surv -= int((kk >= kmin).sum())
        alive[idx] = False
        curve.append((g, surv))
        del i1, i2, idx, a, kk
    S_Z = surv
    out["S_Z"] = S_Z

    # independent twin count of the range, plain sieve of the members
    Apr = np.ones(P, dtype=bool)
    Bpr = np.ones(P, dtype=bool)
    for p in primes_upto(Z):
        if p < 5:
            continue
        p = int(p)
        u = pow(6, -1, p)
        i1 = np.arange(u % p, P, p, dtype=np.int64)
        i2 = np.arange((p - u) % p, P, p, dtype=np.int64)
        hp = home_col(p)
        if p % 6 == 5:
            Apr[i1[i1 != hp]] = False
            Bpr[i2] = False
        else:
            Apr[i1] = False
            Bpr[i2[i2 != hp]] = False
        del i1, i2
    twin_mask = Apr & Bpr
    del Apr, Bpr
    twin_true = int(twin_mask[kmin:].sum())
    ps = set(int(x) for x in primes_upto(Z + 2))
    small_tw = sum(1 for p in primes_upto(Z) if p > q and (int(p) + 2) in ps)
    out["twin_true"] = twin_true
    out["twins_member_le_Z"] = small_tw
    out["twin_identity_gap"] = twin_true - S_Z
    mism = int(np.count_nonzero((alive[kmin:] | twin_mask[kmin:]) != twin_mask[kmin:]))
    out["survivors_not_twins"] = mism
    log(f"  [P3] S(Z) = {S_Z}; twins in range = {twin_true}; difference "
        f"{twin_true - S_Z} (twins with a member <= Z = {small_tw}); "
        f"survivors that are not twins: {mism}")
    del alive, twin_mask

    prodv = 1.0
    tab = []
    for g, sr in curve:
        prodv *= (1.0 - 2.0 / g)
        tab.append((g, sr, prodv, sr / (N_range * prodv),
                    math.log(six_P) / math.log(g)))
    out["curve_at_s"] = []
    for st in (6.0, 4.2664, 3.0, 2.5, 2.2, 2.0):
        best = min(tab, key=lambda r: abs(r[4] - st))
        out["curve_at_s"].append([st, int(best[0]), int(best[1]), best[2],
                                  best[3], best[4]])
        log(f"      s~{st}: z={best[0]} (s={best[4]:.3f})  S={best[1]}  "
            f"ratio={best[3]:.4f}")
    out["ratio_s2"] = tab[-1][3]
    out["classical"] = 1.0 / (4 * math.exp(-2 * GAMMA))
    log(f"      ratio at s=2: {tab[-1][3]:.5f}  vs 1/(4 e^-2gamma) = "
        f"{out['classical']:.5f}  (of limit: {tab[-1][3]/out['classical']:.4f})")
    out["curve"] = [[int(a), int(b), c, d, e] for a, b, c, d, e in tab]

    # ---------- 4: switching identity ----------
    tarr = np.array(tops, dtype=np.int64)
    D = Q = 0
    for i, g in enumerate(tops):
        mm = tarr[i + 1:]
        mm = mm[g * mm <= six_P + 1]
        if len(mm):
            prod = g * mm
            kc = np.where(prod % 6 == 5, (prod + 1) // 6, (prod - 1) // 6)
            kc = kc[(kc < P) & (kc >= kmin)]
            if len(kc):
                D += int(mask[kc].sum())
        pr2 = g * g
        if pr2 <= six_P + 1:
            kc = (pr2 + 1) // 6 if pr2 % 6 == 5 else (pr2 - 1) // 6
            if kmin <= kc < P and mask[kc]:
                Q += 1
    out["bilinear"] = dict(incidences=n_inc, m_eq_1=n_one, m_prime=n_prime,
                           m_composite=n_comp, frac_prime=n_prime / n_inc,
                           pred_prime=pred_prime, pred_frac=pred_prime / n_inc,
                           E=n_prime_leZ, D=D, Q=Q,
                           identity_ok=int(n_prime_leZ == 2 * D + Q))
    log(f"  [P5] incidences {n_inc}: m=1 {n_one}, m prime {n_prime} "
        f"({n_prime/n_inc:.4f}), composite {n_comp}; sieve prediction for primes "
        f"{pred_prime:.1f} ({pred_prime/n_inc:.4f}), ratio "
        f"{n_prime/pred_prime:.4f}")
    log(f"       switching: E={n_prime_leZ}  2D+Q={2*D+Q}  (D={D}, Q={Q})  "
        f"{'OK' if n_prime_leZ == 2*D+Q else 'MISMATCH'}")

    # ---------- 5: placement ----------
    occ = {}
    bopen_home = 0
    own_side_violation = 0
    for g in tops:
        hg = home_col(g)
        occ.setdefault(hg, []).append(g)
        if mask[hg]:
            bopen_home += 1
        for h in bottom:
            uh = pow(6, -1, h)
            bad = uh % h if g % 6 == 5 else (h - uh) % h
            if hg % h == bad:
                own_side_violation += 1
    n1 = sum(1 for v in occ.values() if len(v) == 1)
    n2 = sum(1 for v in occ.values() if len(v) == 2)
    n3 = sum(1 for v in occ.values() if len(v) > 2)
    n1b = sum(1 for k_, v in occ.items() if len(v) == 1 and mask[k_])
    n2b = sum(1 for k_, v in occ.items() if len(v) == 2 and mask[k_])
    prefix_cols = max(occ) + 1
    bopen_prefix = int(mask[1:prefix_cols].sum())
    twins_in_top = sum(1 for p in tops if (p + 2) in ps and p + 2 <= Z)
    out["placement"] = dict(prefix_cols=int(prefix_cols), placed=len(tops),
                            distinct_columns=len(occ), cols_1=n1, cols_2=n2,
                            cols_3plus=n3, cols_1_bopen=n1b, cols_2_bopen=n2b,
                            bottom_open_prefix=bopen_prefix,
                            home_bottom_open=bopen_home,
                            frac_home_bottom_open=bopen_home / len(tops),
                            onetooth_density=onetooth,
                            own_side_violations=own_side_violation,
                            doubly_occupied=n2, twins_in_top_range=twins_in_top)
    log(f"  [P8] {len(tops)} placements on {len(occ)} columns of the prefix "
        f"[1,{prefix_cols}) of P={P}; home bottom-open {bopen_home} "
        f"({bopen_home/len(tops):.4f}) vs one-tooth density {onetooth:.4f}; "
        f"own-side violations {own_side_violation}")
    log(f"  [P9] doubly occupied {n2} (twins in (q,Z] = {twins_in_top}); triples {n3}; "
        f"single {n1}; bottom-open prefix columns {bopen_prefix}; "
        f"doubly occupied and bottom-open {n2b}")

    hcols = np.array([home_col(g) for g in tops], dtype=np.int64)
    eq = []
    for h in bottom:
        r = hcols % h
        cnt = np.bincount(r, minlength=h)
        nz = cnt[cnt > 0]
        exp = len(tops) / (h - 1)
        eq.append([h, int(np.count_nonzero(cnt == 0)),
                   float(np.max(np.abs(nz - exp)) / math.sqrt(exp))])
    out["home_residue"] = eq
    log(f"  [P8] home residues mod bottom gear (gear, empty classes, max sigma): "
        f"{[(a, b_, round(c,2)) for a, b_, c in eq]}")

    out["seconds"] = time.time() - t0
    with open(os.path.join(RES, f"core_q{q}.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"  done in {out['seconds']:.1f}s")
    return out


def main():
    qs = [int(x) for x in sys.argv[1:]] or [11, 13, 17, 19, 23]
    fh = open(os.path.join(RES, "core.log"), "a", encoding="utf-8")

    def log(msg):
        print(msg)
        fh.write(msg + "\n")
        fh.flush()

    for q in qs:
        run(q, log)
    fh.close()


if __name__ == "__main__":
    main()
