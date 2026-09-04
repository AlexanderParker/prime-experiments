"""Branch 7b: the anchor pattern in the window, measured literally.

Column k = (6k-1, 6k+1).  Gear g >= 5 strikes k iff k = +-u_g (mod g), u_g = 6^{-1} mod g.
Level Q (prime), Q' next prime.  Window columns k_lo = floor((Q+1)/6)+1 .. k_hi = (Q'^2-1)/6 - 1.
Section columns k_s = (Q^2-1)/6 + 1 .. k_hi.

For an anchor A = {5..a} and every gear g in (a, Q] ascending:
  raw_g   = anchor openings in the window struck by g
  fresh_g = surviving openings struck by g (then removed)
  N_cur   = survivors before g
and the same on the section and on the left half of the window.  Survivors at the end must equal
the twin prime pairs in the window (primality sieve) - the correctness test.

Also per gear: the percentile of |D_fresh| for the real teeth among all symmetric tooth pairs
+-v mod g on the same survivor set, and the full-period deviation delta_{M}(g) of the lower
machine M = {5..g-1} when its period fits in the window.

Usage: uv run python research/anchor235/r34/anchor_window.py [QMAX] [anchors]
  anchors: comma list of 'min', '13', '19' (default all three).  QMAX default 5000.
Writes research/anchor235/r34/results/sweep_<tag>.tsv, gears_<tag>.npz, gears_<tag>_Q<Q>.tsv
for the named levels, and prints a bounded summary.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

NAMED_Q = [17, 59, 173, 499, 997, 1499, 2999, 4999]


def primes_upto(n):
    s = np.ones(n + 1, dtype=np.uint8)
    s[:2] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = 0
    return np.nonzero(s)[0]


def prime_mask_upto(n):
    s = np.ones(n + 1, dtype=np.uint8)
    s[:2] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = 0
    return s


def tooth(g):
    return pow(6, -1, g)


def full_period_openings(gears):
    """Columns k in [0, P) open under the given gears (sorted array)."""
    P = 1
    for q in gears:
        P *= q
    ks = np.arange(P, dtype=np.int64)
    m = np.ones(P, dtype=bool)
    for q in gears:
        r = ks % q
        u = tooth(q)
        m &= (r != u) & (r != q - u)
    return ks[m], P


def anchor_min(W, primes):
    P = 1
    for q in primes:
        if q < 5:
            continue
        P *= q
        if P >= W:
            return q
    raise ValueError("no anchor")


def run_level(Q, Qn, a, primes, twin, delta_tab, want_detail):
    """One level, one anchor.  Returns (summary dict, per-gear rows list)."""
    k_lo = (Q + 1) // 6 + 1
    k_hi = (Qn * Qn - 1) // 6 - 1
    k_s = (Q * Q - 1) // 6 + 1
    W = k_hi - k_lo + 1
    S = k_hi - k_s + 1
    k_mid = k_lo + W // 2  # left half = k < k_mid
    ks = np.arange(k_lo, k_hi + 1, dtype=np.int64)
    anchor_gears = [q for q in primes if 5 <= q <= a]
    amask = np.ones(W, dtype=bool)
    for q in anchor_gears:
        r = ks % q
        u = tooth(q)
        amask &= (r != u) & (r != q - u)
    N_A = int(amask.sum())
    N_A_sec = int(amask[k_s - k_lo:].sum())
    alive = ks[amask]  # sorted survivors
    P_lower = 1
    for q in anchor_gears:
        P_lower *= q
    lower_gears = list(anchor_gears)
    rows = []
    E_win = 0.0
    E_sec = 0.0
    sum_fresh = 0
    sum_fresh_sec = 0
    gears = [g for g in primes if a < g <= Q]
    for g in gears:
        u = tooth(g)
        i_plus = (u - k_lo) % g
        i_minus = (-u - k_lo) % g
        raw = int(amask[i_plus::g].sum() + amask[i_minus::g].sum())
        i_ps = (u - k_s) % g
        i_ms = (-u - k_s) % g
        sec_view = amask[k_s - k_lo:]
        raw_sec = int(sec_view[i_ps::g].sum() + sec_view[i_ms::g].sum())
        N_cur = alive.size
        r = alive % g
        hit = (r == u) | (r == g - u)
        fresh = int(hit.sum())
        i_sec = int(np.searchsorted(alive, k_s))
        i_half = int(np.searchsorted(alive, k_mid))
        N_cur_sec = N_cur - i_sec
        N_cur_left = i_half
        fresh_sec = int(hit[i_sec:].sum())
        fresh_left = int(hit[:i_half].sum())
        # live zone: columns k >= (g^2-1)/6.  The real teeth strike no survivor below it
        # (a struck member g*m with m < g was struck by a prime factor of m already).
        k_live = (g * g - 1) // 6
        i_live = int(np.searchsorted(alive, k_live))
        N_live = N_cur - i_live
        below = int(hit[:i_live].sum())
        if below != 0:
            raise AssertionError(f"fresh strike below g^2/6 at Q={Q} g={g}: {below}")
        D_live = fresh - 2.0 * N_live / g
        # counterfactual teeth on the same survivor set
        hist = np.bincount(r, minlength=g)
        half = np.arange(1, (g - 1) // 2 + 1)
        cnt = hist[half] + hist[g - half]
        Dv = np.abs(cnt - 2.0 * N_cur / g)
        D_real = abs(fresh - 2.0 * N_cur / g)
        pct = (np.sum(Dv < D_real) + 0.5 * np.sum(Dv == D_real)) / Dv.size
        if N_live > 0:
            hist_l = np.bincount(r[i_live:], minlength=g)
            cnt_l = hist_l[half] + hist_l[g - half]
            Dv_l = np.abs(cnt_l - 2.0 * N_live / g)
            pct_l = (np.sum(Dv_l < abs(D_live)) + 0.5 * np.sum(Dv_l == abs(D_live))) / Dv_l.size
        else:
            pct_l = np.nan
        # section counterfactual
        if N_cur_sec > 0:
            hist_s = np.bincount(r[i_sec:], minlength=g)
            cnt_s = hist_s[half] + hist_s[g - half]
            Dv_s = np.abs(cnt_s - 2.0 * N_cur_sec / g)
            D_real_s = abs(fresh_sec - 2.0 * N_cur_sec / g)
            pct_s = (np.sum(Dv_s < D_real_s) + 0.5 * np.sum(Dv_s == D_real_s)) / Dv_s.size
        else:
            pct_s = np.nan
        # full-period deviation of the lower machine at this gear, if its period fits
        fits = 1 if P_lower <= W else 0
        key = (tuple(lower_gears), g)
        delta = delta_tab.get(key, np.nan)
        D_fresh = fresh - 2.0 * N_cur / g
        D_raw = raw - 2.0 * N_A / g
        D_fresh_sec = fresh_sec - 2.0 * N_cur_sec / g
        D_raw_sec = raw_sec - 2.0 * N_A_sec / g
        if N_cur > 0:
            E_win += max(D_fresh / N_cur, 0.0)
        if N_cur_sec > 0:
            E_sec += max(D_fresh_sec / N_cur_sec, 0.0)
        sum_fresh += fresh
        sum_fresh_sec += fresh_sec
        rows.append((Q, g, N_cur, raw, fresh, D_raw, D_fresh, N_cur_sec, raw_sec, fresh_sec,
                     D_raw_sec, D_fresh_sec, N_cur_left, fresh_left, pct, pct_s, fits, delta,
                     hist.max() - hist.min(), hist.std(), N_live, D_live, pct_l))
        alive = alive[~hit]
        P_lower *= g
        lower_gears.append(g)
    surv = alive.size
    surv_sec = int(np.sum(alive >= k_s))
    tw = int(twin[k_lo:k_hi + 1].sum())
    tw_sec = int(twin[k_s:k_hi + 1].sum())
    ok = (surv == tw) and (surv_sec == tw_sec) and (N_A - sum_fresh == surv)
    prod = 1.0
    for g in gears:
        prod *= (1.0 - 2.0 / g)
    R = tw / (N_A * prod) if N_A > 0 else np.nan
    R_sec = tw_sec / (N_A_sec * prod) if N_A_sec > 0 else np.nan
    arr = np.array([(r_[6], r_[1], r_[2], r_[4]) for r_ in rows])
    j = int(np.argmax(np.abs(arr[:, 0])))
    arr_r = np.array([(r_[5], r_[1]) for r_ in rows])
    jr = int(np.argmax(np.abs(arr_r[:, 0])))
    arr_s = np.array([(r_[11], r_[1], r_[7], r_[9]) for r_ in rows])
    js = int(np.argmax(np.abs(arr_s[:, 0])))
    arr_l = np.array([(r_[21], r_[1], r_[20]) for r_ in rows])
    jl = int(np.argmax(np.abs(arr_l[:, 0])))
    E_live = float(sum(max(r_[21] / r_[20], 0.0) for r_ in rows if r_[20] > 0))
    top = [r_ for r_ in rows if r_[1] > Q / 2]
    top_max = max(abs(r_[6]) for r_ in top) if top else 0.0
    summary = dict(Q=Q, Qn=Qn, a=a, k_lo=k_lo, k_hi=k_hi, W=W, S=S, N_A=N_A, N_A_sec=N_A_sec,
                   twins=tw, twins_sec=tw_sec, surv=surv, surv_sec=surv_sec, ok=int(ok),
                   sum_fresh=sum_fresh, R=R, R_sec=R_sec, prod=prod,
                   maxDf=arr[j, 0], gDf=int(arr[j, 1]), NcurDf=int(arr[j, 2]), freshDf=int(arr[j, 3]),
                   maxDr=arr_r[jr, 0], gDr=int(arr_r[jr, 1]),
                   maxDfs=arr_s[js, 0], gDfs=int(arr_s[js, 1]), NcurDfs=int(arr_s[js, 2]),
                   freshDfs=int(arr_s[js, 3]),
                   E_win=E_win, E_sec=E_sec, top_max=top_max,
                   room=np.log(max(N_A * prod, 1.0)),
                   pct_mean=float(np.nanmean([r_[14] for r_ in rows])),
                   pct_sec_mean=float(np.nanmean([r_[15] for r_ in rows])),
                   maxDl=arr_l[jl, 0], gDl=int(arr_l[jl, 1]), NliveDl=int(arr_l[jl, 2]),
                   pct_live_mean=float(np.nanmean([r_[22] for r_ in rows])),
                   E_live=E_live)
    return summary, rows


ROW_COLS = ["Q", "g", "N_cur", "raw", "fresh", "D_raw", "D_fresh", "N_cur_sec", "raw_sec",
            "fresh_sec", "D_raw_sec", "D_fresh_sec", "N_cur_left", "fresh_left", "pct", "pct_sec",
            "fits", "delta_lower", "hist_range", "hist_std", "N_live", "D_live", "pct_live"]
SUM_COLS = ["Q", "Qn", "a", "k_lo", "k_hi", "W", "S", "N_A", "N_A_sec", "twins", "twins_sec",
            "surv", "surv_sec", "ok", "sum_fresh", "R", "R_sec", "prod", "maxDf", "gDf", "NcurDf",
            "freshDf", "maxDr", "gDr", "maxDfs", "gDfs", "NcurDfs", "freshDfs", "E_win", "E_sec",
            "top_max", "room", "pct_mean", "pct_sec_mean", "maxDl", "gDl", "NliveDl",
            "pct_live_mean", "E_live"]


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    QMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    anchors = sys.argv[2].split(",") if len(sys.argv) > 2 else ["min", "13", "19"]
    t0 = time.time()
    primes = primes_upto(QMAX + 200)
    plist = [int(p) for p in primes]
    Qs = [p for p in plist if 17 <= p <= QMAX]
    Qn_of = {plist[i]: plist[i + 1] for i in range(len(plist) - 1)}
    nmax = Qn_of[Qs[-1]] ** 2
    pm = prime_mask_upto(nmax)
    kmax = (nmax - 1) // 6
    kk = np.arange(1, kmax + 1, dtype=np.int64)
    twin = np.zeros(kmax + 1, dtype=np.uint8)
    twin[1:] = pm[6 * kk - 1] & pm[6 * kk + 1]
    del pm
    print(f"primes and twins ready in {time.time() - t0:.1f}s; kmax = {kmax}")
    # full-period deviations delta_M(g) for the lower machines whose period can fit a window
    delta_tab = {}
    for gears in ([5], [5, 7], [5, 7, 11], [5, 7, 11, 13], [5, 7, 11, 13, 17],
                  [5, 7, 11, 13, 17, 19]):
        op, P = full_period_openings(gears)
        N = op.size
        for g in plist:
            if g <= gears[-1] or g > QMAX:
                continue
            h = np.bincount(op % g, minlength=g)
            u = tooth(g)
            delta_tab[(tuple(gears), g)] = float(h[u] + h[g - u] - 2.0 * N / g)
    print(f"delta table ready in {time.time() - t0:.1f}s")
    for tag in anchors:
        sums = []
        allrows = []
        for Q in Qs:
            Qn = Qn_of[Q]
            k_lo = (Q + 1) // 6 + 1
            k_hi = (Qn * Qn - 1) // 6 - 1
            W = k_hi - k_lo + 1
            if tag == "min":
                a = anchor_min(W, plist)
            else:
                a = int(tag)
                if a >= Q:
                    continue
            s, rows = run_level(Q, Qn, a, plist, twin, delta_tab, Q in NAMED_Q)
            sums.append(s)
            allrows.extend(rows)
            if Q in NAMED_Q:
                with open(os.path.join(RES, f"gears_{tag}_Q{Q}.tsv"), "w") as f:
                    f.write("\t".join(ROW_COLS) + "\n")
                    for r_ in rows:
                        f.write("\t".join(fmt(v) for v in r_) + "\n")
            if not s["ok"]:
                print(f"CORRECTNESS FAIL tag={tag} Q={Q}: surv={s['surv']} twins={s['twins']} "
                      f"surv_sec={s['surv_sec']} twins_sec={s['twins_sec']}")
        with open(os.path.join(RES, f"sweep_{tag}.tsv"), "w") as f:
            f.write("\t".join(SUM_COLS) + "\n")
            for s in sums:
                f.write("\t".join(fmt(s[c]) for c in SUM_COLS) + "\n")
        arr = np.array(allrows, dtype=np.float64)
        np.savez_compressed(os.path.join(RES, f"gears_{tag}.npz"), rows=arr, cols=np.array(ROW_COLS))
        nfail = sum(1 for s in sums if not s["ok"])
        print(f"anchor {tag}: {len(sums)} levels, correctness failures {nfail}, "
              f"{time.time() - t0:.1f}s")
        for s in sums:
            if s["Q"] in NAMED_Q:
                print(f"  Q={s['Q']:5d} a={s['a']:3d} W={s['W']:8d} N_A={s['N_A']:8d} twins={s['twins']:6d} "
                      f"R={s['R']:.3f} maxDf={s['maxDf']:8.1f}@g={s['gDf']:5d}(Ncur={s['NcurDf']},fresh={s['freshDf']}) "
                      f"maxDr={s['maxDr']:8.1f}@{s['gDr']} sec:maxDf={s['maxDfs']:6.1f}@{s['gDfs']} "
                      f"R_sec={s['R_sec']:.3f} E={s['E_win']:.3f} E_sec={s['E_sec']:.3f} room={s['room']:.2f} "
                      f"top_max={s['top_max']:.1f} pct={s['pct_mean']:.3f} | live: maxDl={s['maxDl']:.1f}@{s['gDl']}"
                      f"(Nlive={s['NliveDl']}) pct_live={s['pct_live_mean']:.3f} E_live={s['E_live']:.3f}")
        # extremes over Q
        jmax = int(np.argmax([abs(s["maxDf"]) for s in sums]))
        s = sums[jmax]
        print(f"  extreme window |D_fresh|: Q={s['Q']} g={s['gDf']} D={s['maxDf']:.1f} Ncur={s['NcurDf']} fresh={s['freshDf']}")
        jmax = int(np.argmax([abs(s["maxDfs"]) for s in sums]))
        s = sums[jmax]
        print(f"  extreme section |D_fresh|: Q={s['Q']} g={s['gDfs']} D={s['maxDfs']:.1f} Ncur={s['NcurDfs']} fresh={s['freshDfs']}")
        print(f"  R range {min(s['R'] for s in sums):.3f}..{max(s['R'] for s in sums):.3f}; "
              f"R_sec range {np.nanmin([s['R_sec'] for s in sums]):.3f}..{np.nanmax([s['R_sec'] for s in sums]):.3f}; "
              f"E_win max {max(s['E_win'] for s in sums):.3f}; E_sec max {max(s['E_sec'] for s in sums):.3f}; "
              f"top_max max {max(s['top_max'] for s in sums):.1f}")


if __name__ == "__main__":
    main()
