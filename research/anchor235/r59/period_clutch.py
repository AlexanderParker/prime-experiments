"""R4.a - THE CLUTCH: bottom machine and top machine on the same track.

Both machines are built the same way and independently: a gear per prime, teeth at
k = +- 6^{-1} (mod g), every gear starting at column 0, no exemptions.

  BOTTOM = primes in [5, q].  Period P = prod g columns.
  TOP    = primes in (q, Z],  Z = isqrt(6P+1)  (the gears that can act below P).

Part A: the top machine's OWN pattern over [0, P) - density, drift (non-periodicity),
        run structure, kills per gear.
Part B: the clutch - every column below P classified by (bottom state, top state), the
        four cells against the independence prediction, the sub-split of the
        bottom-open/top-closed cell into home-column strikes and proper strikes, run
        structure inside each cell, correlation by block.
Part C: the shared origin - column 0, mirror symmetry, the neighbourhood of zero.

Usage: uv run python research/anchor235/r59/period_clutch.py 11 13 17 19 23
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


def runs(b):
    """(longest run of True, longest run of False, spectrum dict for True runs)."""
    if b.size == 0:
        return 0, 0, {}
    d = np.diff(b.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if b[0]:
        starts = np.concatenate(([0], starts))
    if b[-1]:
        ends = np.concatenate((ends, [b.size]))
    tr = ends - starts
    tot = b.size
    ntrue = int(b.sum())
    # false runs
    fstarts = ends[:-1] if b[0] else np.concatenate(([0], ends))
    # simpler: complement
    c = ~b
    dc = np.diff(c.astype(np.int8))
    cs = np.flatnonzero(dc == 1) + 1
    ce = np.flatnonzero(dc == -1) + 1
    if c[0]:
        cs = np.concatenate(([0], cs))
    if c[-1]:
        ce = np.concatenate((ce, [c.size]))
    fr = ce - cs
    spec = {}
    if tr.size:
        u, cnt = np.unique(tr, return_counts=True)
        spec = {int(a): int(b_) for a, b_ in zip(u[:20], cnt[:20])}
    return (int(tr.max()) if tr.size else 0,
            int(fr.max()) if fr.size else 0, spec, int(tot), ntrue)


def run(q, log):
    t0 = time.time()
    allp = primes_upto(10 ** 6)
    bottom = [int(p) for p in allp if 5 <= p <= q]
    m = len(bottom)
    P = 1
    for g in bottom:
        P *= g
    six_P = 6 * P
    Z = math.isqrt(six_P + 1)
    tops = [int(p) for p in primes_upto(Z) if p > q]
    out = dict(q=q, m=m, P=P, six_P=six_P, Z=Z, n_top=len(tops))
    log(f"=== CLUTCH q={q}  P={P}  6P={six_P}  Z={Z}  bottom {m} gears  top {len(tops)} gears")

    # ---------- masks ----------
    b_lo = np.zeros(P, dtype=bool)   # bottom gear divides 6k-1
    b_up = np.zeros(P, dtype=bool)
    for g in bottom:
        u = pow(6, -1, g)
        b_lo[u % g:: g] = True
        b_up[(g - u) % g:: g] = True
    b_open = ~(b_lo | b_up)
    N = int(b_open.sum())

    t_lo = np.zeros(P, dtype=bool)
    t_up = np.zeros(P, dtype=bool)
    home = np.zeros(P, dtype=bool)   # column is the home column of some top gear
    kills_per_gear = []
    for g in tops:
        u = pow(6, -1, g)
        a1 = u % g
        a2 = (g - u) % g
        t_lo[a1::g] = True
        t_up[a2::g] = True
        n1 = (P - a1 + g - 1) // g
        n2 = (P - a2 + g - 1) // g
        kills_per_gear.append((g, int(n1 + n2)))
        home[home_col(g)] = True
    t_open = ~(t_lo | t_up)
    T = int(t_open.sum())

    # ---------- A. the top machine's own pattern ----------
    prod_top = 1.0
    for g in tops:
        prod_top *= (1.0 - 2.0 / g)
    d_top = T / P
    d_bot = N / P
    out["A"] = dict(top_open=T, top_density=d_top, prod_top=prod_top,
                    ratio_density_over_prod=d_top / prod_top,
                    bottom_open=N, bottom_density=d_bot)
    log(f"  [A] top-open columns over [0,P): {T}  density {d_top:.6f}  "
        f"vs prod(1-2/g) = {prod_top:.6g}  ratio {d_top/prod_top:.4f}")
    log(f"      bottom-open {N}  density {d_bot:.6f} (exact = prod(g-2)/prod(g))")

    # drift across the range: 20 blocks
    NB = 20
    bl = P // NB
    drift = []
    for i in range(NB):
        sl = slice(i * bl, (i + 1) * bl if i < NB - 1 else P)
        drift.append([i, float(t_open[sl].mean()), float(b_open[sl].mean())])
    out["A_drift"] = drift
    log("      top-open density by block (20 blocks of the period): "
        + " ".join(f"{d[1]:.4f}" for d in drift))
    log("      bottom-open density by block: "
        + " ".join(f"{d[2]:.4f}" for d in drift))

    # run structure of the top machine
    tmax_open, tmax_closed, tspec, _, _ = runs(t_open)
    bmax_open, bmax_closed, bspec, _, _ = runs(b_open)
    out["A_runs"] = dict(top_longest_open_run=tmax_open,
                         top_longest_closed_run=tmax_closed,
                         top_open_run_spectrum=tspec,
                         bottom_longest_open_run=bmax_open,
                         bottom_longest_closed_run=bmax_closed,
                         bottom_open_run_spectrum=bspec)
    log(f"      top runs: longest open {tmax_open}, longest closed {tmax_closed}; "
        f"open-run spectrum {tspec}")
    log(f"      bottom runs: longest open {bmax_open}, longest closed {bmax_closed} "
        f"(= F({{5..{q}}}) + 1 as a run of columns)")

    # kills per gear
    kg = np.array([k for _, k in kills_per_gear], dtype=np.int64)
    gg = np.array([g for g, _ in kills_per_gear], dtype=np.int64)
    pred = 2.0 * P / gg
    out["A_kills"] = dict(min=int(kg.min()), max=int(kg.max()),
                          max_abs_dev=float(np.max(np.abs(kg - pred))),
                          smallest_gear=int(gg[0]), largest_gear=int(gg[-1]),
                          kills_smallest=int(kg[0]), kills_largest=int(kg[-1]))
    log(f"      kills per top gear over [0,P): from {int(kg.max())} (g={int(gg[0])}) "
        f"down to {int(kg.min())} (g={int(gg[-1])}); max |kills - 2P/g| = "
        f"{float(np.max(np.abs(kg - pred))):.2f}")

    # ---------- B. the clutch ----------
    cell_oo = b_open & t_open
    cell_oc = b_open & ~t_open
    cell_co = ~b_open & t_open
    cell_cc = ~b_open & ~t_open
    n_oo, n_oc, n_co, n_cc = (int(cell_oo.sum()), int(cell_oc.sum()),
                              int(cell_co.sum()), int(cell_cc.sum()))
    indep_oo = d_bot * d_top * P
    out["B_cells"] = dict(both_open=n_oo, bottom_open_top_closed=n_oc,
                          bottom_closed_top_open=n_co, both_closed=n_cc,
                          indep_both_open=indep_oo,
                          coupling=n_oo / indep_oo,
                          indep_oc=d_bot * (1 - d_top) * P,
                          coupling_oc=n_oc / (d_bot * (1 - d_top) * P),
                          indep_co=(1 - d_bot) * d_top * P,
                          coupling_co=n_co / ((1 - d_bot) * d_top * P),
                          indep_cc=(1 - d_bot) * (1 - d_top) * P,
                          coupling_cc=n_cc / ((1 - d_bot) * (1 - d_top) * P))
    log(f"  [B] cells: both open {n_oo}, bottom-open/top-closed {n_oc}, "
        f"bottom-closed/top-open {n_co}, both closed {n_cc}  (sum {n_oo+n_oc+n_co+n_cc} = P)")
    log(f"      independence prediction for both-open: {indep_oo:.1f}; "
        f"coupling = {n_oo/indep_oo:.4f}")
    log(f"      couplings (oc, co, cc): {out['B_cells']['coupling_oc']:.4f}, "
        f"{out['B_cells']['coupling_co']:.4f}, {out['B_cells']['coupling_cc']:.4f}")

    # sub-split of bottom-open / top-closed: home strike only vs proper strike
    # a home strike at column k means a top gear's own prime is a member of k.
    # build "proper top strike" masks: same as t_lo/t_up but skipping the home column
    p_lo = np.zeros(P, dtype=bool)
    p_up = np.zeros(P, dtype=bool)
    for g in tops:
        u = pow(6, -1, g)
        a1 = u % g
        a2 = (g - u) % g
        hg = home_col(g)
        i1 = np.arange(a1, P, g, dtype=np.int64)
        i2 = np.arange(a2, P, g, dtype=np.int64)
        if g % 6 == 5:
            p_lo[i1[i1 != hg]] = True
            p_up[i2] = True
        else:
            p_lo[i1] = True
            p_up[i2[i2 != hg]] = True
    proper = p_lo | p_up
    only_home = cell_oc & ~proper
    n_only_home = int(only_home.sum())
    n_proper_oc = n_oc - n_only_home
    out["B_oc_split"] = dict(home_only=n_only_home, has_proper=n_proper_oc)
    log(f"      bottom-open/top-closed split: killed only by a home-column strike "
        f"(the member IS the top prime, so the column is a twin) {n_only_home}; "
        f"with at least one proper strike {n_proper_oc}")
    twins_total = n_oo + n_only_home
    out["twins_total"] = twins_total
    log(f"      twins in [0,P) = both-open + home-only = {n_oo} + {n_only_home} "
        f"= {twins_total}")

    # both-closed sub-structure: which machine strikes which member
    def side(lo, up):
        return (lo.astype(np.int8) + 2 * up.astype(np.int8))
    sb = side(b_lo, b_up)
    st = side(t_lo, t_up)
    del b_lo, b_up, t_lo, t_up, p_lo, p_up, proper
    cc_tab = {}
    idx = np.flatnonzero(cell_cc)
    if idx.size:
        key = sb[idx] * 4 + st[idx]
        u, c = np.unique(key, return_counts=True)
        for a, b_ in zip(u, c):
            cc_tab[f"b{int(a)//4}_t{int(a)%4}"] = int(b_)
    del sb, st, idx
    out["B_cc_sides"] = cc_tab
    log(f"      both-closed by (bottom side, top side) 1=lower 2=upper 3=both: {cc_tab}")

    # run structure inside each cell
    cellruns = {}
    for name, arr in (("both_open", cell_oo), ("bopen_tclosed", cell_oc),
                      ("bclosed_topen", cell_co), ("both_closed", cell_cc)):
        a, b_, spec, _, _ = runs(arr)
        cellruns[name] = dict(longest_run=a, longest_gap=b_, spectrum=spec)
    out["B_cellruns"] = cellruns
    for k_, v in cellruns.items():
        log(f"      cell {k_}: longest consecutive run {v['longest_run']}, "
            f"longest gap {v['longest_gap']}")

    # correlation by block
    corr = []
    for i in range(NB):
        sl = slice(i * bl, (i + 1) * bl if i < NB - 1 else P)
        db = float(b_open[sl].mean())
        dt = float(t_open[sl].mean())
        doo = float(cell_oo[sl].mean())
        corr.append([i, db, dt, doo, doo / (db * dt) if db * dt else 0.0])
    out["B_corr_block"] = corr
    log("      coupling by block: " + " ".join(f"{c[4]:.3f}" for c in corr))

    # ---------- C. the shared origin ----------
    # mirror: column -k has members -(6k+1), -(6k-1): the same pair with the members
    # swapped, so every state is an even function of k.  Verified on the masks.
    K = min(P // 2, 2_000_000)
    kk = np.arange(1, K, dtype=np.int64)
    mir_b = int(np.count_nonzero(b_open[kk] != b_open[(P - kk) % P]))
    # top mirror: build the top state at columns -1..-(K-1) directly
    tn_lo = np.zeros(K, dtype=bool)
    tn_up = np.zeros(K, dtype=bool)
    for g in tops:
        u = pow(6, -1, g)
        # column -k :  g | 6(-k)-1  iff  -k = u  iff  k = -u
        a1 = (-u) % g
        a2 = u % g
        tn_lo[a1::g] = True
        tn_up[a2::g] = True
    tn_open = ~(tn_lo | tn_up)
    mir_t = int(np.count_nonzero(t_open[:K] != tn_open[:K]))
    out["C_mirror"] = dict(checked=int(K), bottom_mismatches=mir_b,
                           top_mismatches=mir_t)
    log(f"  [C] mirror about column 0: bottom mismatches {mir_b}, top mismatches "
        f"{mir_t} over {K} columns each side")

    # neighbourhood of the origin
    nb = []
    for L in (10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, P):
        L = min(L, P)
        nb.append([int(L), float(t_open[:L].mean()), float(b_open[:L].mean()),
                   float(cell_oo[:L].mean())])
    out["C_origin"] = nb
    for r in nb:
        log(f"      first {r[0]:>9} columns: top-open {r[1]:.6f}  bottom-open "
            f"{r[2]:.6f}  both-open {r[3]:.6f}")
    out["C_col0"] = dict(bottom_open=bool(b_open[0]), top_open=bool(t_open[0]))
    log(f"      column 0: bottom-open {bool(b_open[0])}, top-open {bool(t_open[0])}")

    out["seconds"] = time.time() - t0
    with open(os.path.join(RES, f"clutch_q{q}.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"  done in {out['seconds']:.1f}s")
    return out


def main():
    qs = [int(x) for x in sys.argv[1:]] or [11, 13, 17, 19, 23]
    fh = open(os.path.join(RES, "clutch.log"), "a", encoding="utf-8")

    def log(msg):
        print(msg)
        fh.write(msg + "\n")
        fh.flush()

    for q in qs:
        run(q, log)
    fh.close()


if __name__ == "__main__":
    main()
