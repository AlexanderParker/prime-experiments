"""R4.a - clutch facts III: the placement law, twin gaps at the period scale, the
shared origin, and who kills what inside a twin-free stretch.

Usage: uv run python research/anchor235/r59/clutch_patterns.py 11 13 17 19 23
"""
import sys, os, json, math, time
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


def home_col(g):
    return (g + 1) // 6 if g % 6 == 5 else (g - 1) // 6


def run(q, log):
    t0 = time.time()
    bottom = [int(p) for p in primes_upto(max(q, 5)) if 5 <= p <= q]
    P = 1
    N = 1
    for g in bottom:
        P *= g
        N *= (g - 2)
    six_P = 6 * P
    Z = math.isqrt(six_P + 1)
    tops = [int(p) for p in primes_upto(Z) if p > q]
    out = dict(q=q, P=P, N=N, Z=Z, n_top=len(tops))
    log(f"=== PATTERNS q={q}  P={P}  Z={Z}  top gears {len(tops)}")

    # ---------- 1. the placement residue law ----------
    # exact statement about ALL integers coprime to 6: as r runs over the residues
    # mod 6h that are coprime to 6 and nonzero mod h, the home column k = (r -+ 1)/6
    # takes each of the h-2 NON-tooth classes twice and each of the two tooth classes
    # of gear h once.
    law_ok = True
    law_rows = []
    for h in bottom:
        uh = pow(6, -1, h)
        teeth = {uh % h, (h - uh) % h}
        cnt = [0] * h
        for r in range(6 * h):
            if r % 6 not in (1, 5):
                continue
            if r % h == 0:
                continue
            k = (r + 1) // 6 if r % 6 == 5 else (r - 1) // 6
            cnt[k % h] += 1
        ok = all((cnt[c] == 1) if c in teeth else (cnt[c] == 2) for c in range(h))
        law_ok &= ok
        # measured on the actual top primes
        obs = [0] * h
        for g in tops:
            obs[home_col(g) % h] += 1
        tooth_tot = sum(obs[c] for c in teeth)
        other_tot = sum(obs[c] for c in range(h) if c not in teeth)
        law_rows.append([h, sorted(teeth), ok, tooth_tot, other_tot,
                         (other_tot / (h - 2)) / (tooth_tot / 2) if tooth_tot else None])
    out["placement_law"] = dict(exact_over_residues=bool(law_ok), rows=law_rows)
    log(f"  [law] placement residue law exact over all residues: {law_ok}")
    for r in law_rows:
        log(f"        gear {r[0]:>2}: teeth {r[1]}  measured tooth-class total {r[3]}, "
            f"other-class total {r[4]}, per-class ratio other/tooth = "
            f"{r[5]:.3f} (law: 2.000)")

    # ---------- 2. masks and the twin set ----------
    b_open = np.ones(P, dtype=bool)
    for g in bottom:
        u = pow(6, -1, g)
        b_open[u % g:: g] = False
        b_open[(g - u) % g:: g] = False
    t_open = np.ones(P, dtype=bool)
    for g in tops:
        u = pow(6, -1, g)
        t_open[u % g:: g] = False
        t_open[(g - u) % g:: g] = False
    # twins = both open, plus the columns whose only top strike is a home strike
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
    twin = Apr & Bpr
    del Apr, Bpr
    kmin = (q + 1) // 6 + 1
    twin[:kmin] = False
    ntw = int(twin.sum())
    out["twins"] = ntw

    # ---------- 3. twin gaps at the period scale ----------
    tidx = np.flatnonzero(twin)
    gaps = np.diff(tidx)
    gmax = int(gaps.max())
    gpos = int(tidx[int(np.argmax(gaps))])
    W_hi = (q * q - 1) // 6            # window top column
    inwin = tidx[(tidx >= kmin) & (tidx <= W_hi)]
    win_gap = int(np.diff(inwin).max()) if len(inwin) > 1 else None
    F_bottom = 0
    bidx = np.flatnonzero(b_open)
    F_bottom = int(np.diff(bidx).max())
    out["twin_gaps"] = dict(n_twins=ntw, max_gap=gmax, at_column=gpos,
                            frac_of_period=gpos / P, window_top=W_hi,
                            twins_in_window=int(len(inwin)),
                            max_gap_in_window=win_gap,
                            F_bottom=F_bottom,
                            window_length=W_hi - kmin)
    log(f"  [gap] twins in [kmin,P): {ntw}; longest twin gap {gmax} columns at "
        f"column {gpos} ({gpos/P:.3f} of the period); window (kmin,{W_hi}] holds "
        f"{len(inwin)} twins, longest gap there {win_gap}; bottom record F = "
        f"{F_bottom}; window length {W_hi - kmin}")

    # gap growth across the period: max twin gap in each of 20 blocks
    NB = 20
    bl = P // NB
    blockmax = []
    for i in range(NB):
        lo, hi = i * bl, (i + 1) * bl if i < NB - 1 else P
        sel = tidx[(tidx >= lo) & (tidx < hi)]
        blockmax.append(int(np.diff(sel).max()) if len(sel) > 1 else 0)
    out["twin_gap_by_block"] = blockmax
    log(f"  [gap] max twin gap by block: {blockmax}")

    # ---------- 4. who kills inside the longest twin-free stretch ----------
    lo = gpos + 1
    hi = gpos + gmax
    seg_b = b_open[lo:hi]
    seg_t = t_open[lo:hi]
    n_bc = int((~seg_b).sum())
    n_bo_tc = int((seg_b & ~seg_t).sum())
    n_bo_to = int((seg_b & seg_t).sum())
    out["longest_gap_anatomy"] = dict(length=gmax, bottom_closed=n_bc,
                                      bottom_open_top_closed=n_bo_tc,
                                      both_open=n_bo_to)
    log(f"  [gap] anatomy of the longest twin-free stretch ({gmax} columns): "
        f"bottom-closed {n_bc} ({n_bc/gmax:.3f}), bottom-open/top-closed {n_bo_tc} "
        f"({n_bo_tc/gmax:.3f}), both-open {n_bo_to}")
    # pooled over the 100 longest stretches
    order = np.argsort(gaps)[::-1][:100]
    tot = bc = botc = 0
    for oi in order:
        a = int(tidx[oi]) + 1
        b_ = int(tidx[oi]) + int(gaps[oi])
        tot += b_ - a
        bc += int((~b_open[a:b_]).sum())
        botc += int((b_open[a:b_] & ~t_open[a:b_]).sum())
    out["top100_gap_anatomy"] = dict(columns=tot, bottom_closed=bc,
                                     bottom_open_top_closed=botc)
    log(f"  [gap] pooled over the 100 longest stretches ({tot} columns): "
        f"bottom-closed {bc/tot:.3f}, bottom-open/top-closed {botc/tot:.3f}")

    # ---------- 5. the first KILL: where the two machines first interact ----------
    # a proper strike of a top gear on a bottom-OPEN column has member g*m with m
    # q-rough and m > 1, hence m >= the least prime above q, hence member > q^2.
    first_kill = None
    fk_gear = None
    for g in tops:
        u = pow(6, -1, g)
        for eps, a in ((+1, u % g), (-1, (g - u) % g)):
            idx = np.arange(a, P, g, dtype=np.int64)
            kk = idx[b_open[idx]]
            kk = kk[kk >= kmin]
            if len(kk):
                mm = (6 * kk - eps) // g
                kk = kk[mm > 1]
            if len(kk):
                c = int(kk[0])
                if first_kill is None or c < first_kill:
                    first_kill, fk_gear = c, g
            del idx, kk
    out["first_kill"] = dict(column=first_kill, gear=fk_gear, window_top=W_hi,
                             above_window=bool(first_kill > W_hi),
                             ratio=first_kill / W_hi)
    log(f"  [win] first proper kill of a bottom-open column: column {first_kill} "
        f"(gear {fk_gear}); window top column {W_hi}; above the window: "
        f"{first_kill > W_hi} (ratio {first_kill / W_hi:.3f})")

    # how much of the bottom machine the top machine eventually eats
    N_range = int(b_open[kmin:].sum())
    out["eaten"] = dict(N_range=N_range, survivors=ntw,
                        killed_fraction=1 - ntw / N_range)
    log(f"  [win] top machine kills {1 - ntw / N_range:.4f} of the bottom machine's "
        f"{N_range} candidates over the period")

    # ---------- 6. the shared origin: aperiodicity ----------
    # the bottom machine repeats every P columns; the top machine's period is the
    # product of ALL its gears.  Test agreement of the top pattern with its own
    # translate by a bottom subperiod, on a mid-period window, against the
    # agreement expected from the two segments' own densities.
    per = []
    L = min(P // 8, 200000)
    start = P // 2
    base = t_open[start:start + L]
    d0 = float(base.mean())
    dd = 1
    for g in bottom:
        dd *= g
        if start + dd + L >= P:
            break
        seg = t_open[start + dd: start + dd + L]
        d1 = float(seg.mean())
        exp = d0 * d1 + (1 - d0) * (1 - d1)
        per.append([dd, float((seg == base).mean()), exp])
    out["top_periodicity"] = per
    log(f"  [origin] top pattern vs its translate by a bottom subperiod "
        f"(observed agreement, independent expectation): "
        f"{[(a, round(b_, 4), round(c, 4)) for a, b_, c in per]}")
    out["seconds"] = time.time() - t0
    with open(os.path.join(RES, f"patterns_q{q}.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"  done in {out['seconds']:.1f}s")


def main():
    qs = [int(x) for x in sys.argv[1:]] or [11, 13, 17, 19, 23]
    fh = open(os.path.join(RES, "patterns.log"), "a", encoding="utf-8")

    def log(msg):
        print(msg)
        fh.write(msg + "\n")
        fh.flush()

    for q in qs:
        run(q, log)
    fh.close()


if __name__ == "__main__":
    main()
