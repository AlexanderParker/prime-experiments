"""Branch 7b, part two: mechanism at the extremes and the pre-registered tests.

Reads results/gears_<tag>.npz and results/sweep_<tag>.tsv written by anchor_window.py.

  M1  growth law of max |D_fresh| (fixed anchor: linear in W at the first gear above the anchor,
      slope delta_A(g)/P_A; A_min: C sqrt(2 N_cur/g)); top-gear sizes.
  M2  the (b) ratio R(Q) and the (e) budget E(Q) against the room ln(N_A prod).
  M3  sign structure: halves (mirror), g mod 30, sign vs delta_lower, tooth percentiles.
  M4  residue histogram n_W(r) at the extreme gears (rebuilt state), spread, rank of the teeth,
      struck columns by anchor class; live zone vs whole window.
  M5  exact interval discrepancy of the anchor {5..13} and of all 180 re-toothings: the proved
      bound for D_raw at the first gear over a fixed anchor, in-window.

Usage: uv run python research/anchor235/r34/mechanism.py
Writes results/mechanism.txt (all output is also printed, bounded).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


def primes_upto(n):
    s = np.ones(n + 1, dtype=np.uint8)
    s[:2] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = 0
    return [int(p) for p in np.nonzero(s)[0]]


def tooth(g):
    return pow(6, -1, g)


def load(tag):
    z = np.load(os.path.join(RES, f"gears_{tag}.npz"))
    cols = [str(c) for c in z["cols"]]
    rows = z["rows"]
    R = {c: rows[:, i] for i, c in enumerate(cols)}
    sw = {}
    with open(os.path.join(RES, f"sweep_{tag}.tsv")) as f:
        hdr = f.readline().split()
        for line in f:
            v = line.split()
            d = {h: float(x) for h, x in zip(hdr, v)}
            sw[int(d["Q"])] = d
    return R, sw


def state_before(Q, Qn, a, g_star, primes):
    """Survivor columns of {5..g_star-1} in the window at level Q, plus anchor mask and ks."""
    k_lo = (Q + 1) // 6 + 1
    k_hi = (Qn * Qn - 1) // 6 - 1
    ks = np.arange(k_lo, k_hi + 1, dtype=np.int64)
    amask = np.ones(ks.size, dtype=bool)
    for q in primes:
        if 5 <= q <= a:
            r = ks % q
            u = tooth(q)
            amask &= (r != u) & (r != q - u)
    alive = ks[amask]
    for g in primes:
        if a < g < g_star:
            r = alive % g
            u = tooth(g)
            alive = alive[(r != u) & (r != g - u)]
    return ks, amask, alive, k_lo, k_hi


def m1_growth(tag, R, sw, primes):
    say(f"\n== M1 growth law, anchor {tag} ==")
    Qs = sorted(sw)
    # max |D_fresh| against W, and against sqrt(2 N_cur/g) at the argmax gear
    say("Q, W, max|D_fresh| @g (N_cur, fresh), ratio D/sqrt(2Ncur/g), max|D_live| @g, max|D_raw| @g, top_max")
    for Q in Qs:
        if Q in (17, 59, 173, 499, 997, 1499, 1999, 2999, 3999, 4999) or Q == Qs[-1]:
            d = sw[Q]
            sc = np.sqrt(2.0 * d["NcurDf"] / d["gDf"]) if d["NcurDf"] > 0 else np.nan
            say(f"  {Q:5d} {int(d['W']):8d} {d['maxDf']:8.1f} @{int(d['gDf']):5d} ({int(d['NcurDf'])}, {int(d['freshDf'])})"
                f"  C={d['maxDf'] / sc:6.2f}  live {d['maxDl']:8.1f} @{int(d['gDl'])}  raw {d['maxDr']:8.1f} @{int(d['gDr'])}"
                f"  top {d['top_max']:5.1f}")
    # fixed anchor: the first gear above the anchor, D_raw and D_fresh vs W
    a = int(sw[Qs[-1]]["a"])
    if tag != "min":
        g1 = [g for g in primes if g > a][0]
        g2 = [g for g in primes if g > a][1]
        for g in (g1, g2):
            m = (R["g"] == g)
            Qg = R["Q"][m]
            Dr = R["D_raw"][m]
            Df = R["D_fresh"][m]
            Dl = R["D_live"][m]
            Ws = np.array([sw[int(q)]["W"] for q in Qg])
            NA = np.array([sw[int(q)]["N_A"] for q in Qg])
            delta = R["delta_lower"][m]
            say(f"  gear {g}: D_raw/W over Q (x1e3): first {Dr[:3] / Ws[:3] * 1e3}, last {Dr[-3:] / Ws[-3:] * 1e3};"
                f" delta_lower(g) = {delta[-1]:.3f}; slope prediction delta/P: {delta[-1] / np.prod([q for q in primes if 5 <= q <= a]) * 1e3:.4f} x1e-3")
            say(f"     D_raw at Q = {[int(q) for q in Qg[-5:]]}: {np.round(Dr[-5:], 1)};  D_fresh: {np.round(Df[-5:], 1)};"
                f"  D_live: {np.round(Dl[-5:], 1)}")
            say(f"     relative D_fresh/N_cur (x1e3) at those Q: {np.round(Df[-5:] / R['N_cur'][m][-5:] * 1e3, 3)}")
            # linear fit D_raw = s W + c
            A = np.vstack([Ws, np.ones_like(Ws)]).T
            sol, res, _, _ = np.linalg.lstsq(A, Dr, rcond=None)
            resid = Dr - A @ sol
            say(f"     linear fit D_raw = {sol[0] * 1e3:.4f}e-3 W + {sol[1]:.2f}; max |resid| = {np.abs(resid).max():.1f}, "
                f"std resid = {resid.std():.2f}")
    # A_min: C at the first gear above the anchor, over Q
    if tag == "min":
        Cs = []
        for Q in Qs:
            d = sw[Q]
            a_ = int(d["a"])
            g1 = [g for g in primes if g > a_][0]
            m = (R["Q"] == Q) & (R["g"] == g1)
            if not m.any():
                continue
            Nc = R["N_cur"][m][0]
            Df = R["D_fresh"][m][0]
            Cs.append((Q, a_, g1, Df, Df / np.sqrt(2 * Nc / g1)))
        Cs = np.array(Cs)
        say("  A_min first gear: Q, a, g1, D_fresh, C = D/sqrt(2N/g) - summary by anchor:")
        for a_ in sorted(set(Cs[:, 1])):
            mm = Cs[:, 1] == a_
            say(f"    a={int(a_):3d} n={mm.sum():3d}  C mean {Cs[mm, 4].mean():6.2f}  |C| max {np.abs(Cs[mm, 4]).max():6.2f}"
                f"  D range {Cs[mm, 3].min():7.1f}..{Cs[mm, 3].max():7.1f}  Q {int(Cs[mm, 0].min())}..{int(Cs[mm, 0].max())}")
        # the max over all gears: C at the argmax gear
        Call = np.array([sw[Q]["maxDf"] / np.sqrt(2 * sw[Q]["NcurDf"] / sw[Q]["gDf"]) for Q in Qs if sw[Q]["NcurDf"] > 0])
        say(f"  A_min argmax gear: C = maxD/sqrt(2Ncur/g): mean {Call.mean():.2f}, max {np.abs(Call).max():.2f}, "
            f"quantiles 10/50/90: {np.percentile(np.abs(Call), [10, 50, 90]).round(2)}")
    # top gears: fresh_g distribution
    for Q in (997, 1499, 4999):
        if Q not in sw:
            continue
        m = (R["Q"] == Q) & (R["g"] > Q / 2)
        fr = R["fresh"][m].astype(int)
        say(f"  top gears (g > Q/2) at Q={Q}: n={m.sum()}, fresh values histogram {np.bincount(fr)}, "
            f"max |D_fresh| {np.abs(R['D_fresh'][m]).max():.2f}, max |D_live| {np.abs(R['D_live'][m]).max():.2f}, "
            f"mean 2N_cur/g {np.mean(2 * R['N_cur'][m] / R['g'][m]):.2f}, mean 2N_live/g {np.mean(2 * R['N_live'][m] / R['g'][m]):.2f}")


def m2_ratio_budget(tag, sw):
    say(f"\n== M2 ratio R(Q) and budget E(Q), anchor {tag} ==")
    Qs = sorted(sw)
    say("Q, a, N_A, twins, R, R_sec, E_win, E_live, E_sec, room=ln(N_A prod), sum_fresh/N_A")
    for Q in Qs:
        if Q in (17, 59, 173, 499, 997, 1499, 1999, 2999, 3999, 4999):
            d = sw[Q]
            say(f"  {Q:5d} {int(d['a']):3d} {int(d['N_A']):8d} {int(d['twins']):6d} {d['R']:.4f} {d['R_sec']:.3f}"
                f" {d['E_win']:.3f} {d['E_live']:.3f} {d['E_sec']:.3f} {d['room']:.2f} {d['sum_fresh'] / d['N_A']:.4f}")
    Rv = np.array([sw[Q]["R"] for Q in Qs])
    Rs = np.array([sw[Q]["R_sec"] for Q in Qs])
    Qa = np.array(Qs)
    for lo, hi in ((17, 100), (100, 500), (500, 1500), (1500, 3000), (3000, 5001)):
        m = (Qa >= lo) & (Qa < hi)
        if m.any():
            say(f"  Q in [{lo},{hi}): R min {Rv[m].min():.4f} mean {Rv[m].mean():.4f} max {Rv[m].max():.4f};"
                f" R_sec min {np.nanmin(Rs[m]):.3f} mean {np.nanmean(Rs[m]):.3f} max {np.nanmax(Rs[m]):.3f};"
                f" E_win max {max(sw[Q]['E_win'] for Q in Qa[m]):.3f} E_live max {max(sw[Q]['E_live'] for Q in Qa[m]):.3f}"
                f" E_sec max {max(sw[Q]['E_sec'] for Q in Qa[m]):.3f}")
    say(f"  overall: R in [{Rv.min():.4f}, {Rv.max():.4f}], outside [0.75,1.05] for Q>=100: "
        f"{int(((Rv < 0.75) | (Rv > 1.05))[Qa >= 100].sum())} levels; E_win max {max(sw[Q]['E_win'] for Q in Qs):.3f} at "
        f"Q={max(Qs, key=lambda Q: sw[Q]['E_win'])}; E_sec max {max(sw[Q]['E_sec'] for Q in Qs):.3f} at "
        f"Q={max(Qs, key=lambda Q: sw[Q]['E_sec'])}; min room/E_win = "
        f"{min(sw[Q]['room'] / max(sw[Q]['E_win'], 1e-9) for Q in Qs if Q >= 59):.1f}")


def m3_signs(tag, R, sw):
    say(f"\n== M3 sign structure, anchor {tag} ==")
    Q = R["Q"]
    g = R["g"]
    Df = R["D_fresh"]
    Dl = R["D_live"]
    Nc = R["N_cur"]
    fits = R["fits"] == 1
    # halves
    Dleft = R["fresh_left"] - 2.0 * R["N_cur_left"] / g
    Dright = (R["fresh"] - R["fresh_left"]) - 2.0 * (Nc - R["N_cur_left"]) / g
    for name, m in (("P_lower <= W", fits), ("P_lower > W", ~fits), ("P_lower > W and g > Q/2", (~fits) & (g > Q / 2)),
                    ("P_lower > W and g <= Q/2", (~fits) & (g <= Q / 2))):
        if m.sum() < 5:
            continue
        r = np.corrcoef(Dleft[m], Dright[m])[0, 1]
        agree = np.mean(np.sign(Dleft[m]) == np.sign(Dright[m]))
        say(f"  halves, {name}: n={m.sum()}, corr(D_left, D_right) = {r:+.3f}, sign agreement {agree:.3f}")
    # sign balance and mod 30 - on every fifth level only (rows of one gear at consecutive
    # levels are the same count re-measured, so pooling all levels understates the SE)
    Qlist = sorted(set(Q.astype(int)))
    sparse = np.isin(Q, Qlist[::5])
    for name, arr in (("D_fresh", Df), ("D_live", Dl)):
        m = (~fits) & sparse
        neg = np.mean(arr[m] < 0)
        say(f"  {name} (P_lower > W): fraction negative {neg:.3f}, n={m.sum()}; mean D/N_cur x1e3 = {np.mean(arr[m] / Nc[m]) * 1e3:+.3f}")
        say(f"    by g mod 30 (class: n, mean D, SE, frac negative):")
        for c in (1, 7, 11, 13, 17, 19, 23, 29):
            mc = m & (g % 30 == c)
            if mc.sum() < 3:
                continue
            v = arr[mc]
            say(f"      {c:2d}: n={mc.sum():5d} mean {v.mean():+7.3f} SE {v.std() / np.sqrt(mc.sum()):.3f} neg {np.mean(v < 0):.3f}"
                f"  z = {v.mean() / (v.std() / np.sqrt(mc.sum())):+.2f}")
    # sign vs delta_lower where the lower period fits
    m = fits & np.isfinite(R["delta_lower"]) & (R["delta_lower"] != 0)
    if m.sum():
        ag = np.mean(np.sign(Df[m]) == np.sign(R["delta_lower"][m]))
        say(f"  sign(D_fresh) == sign(delta_lower(g)) where P_lower <= W: {ag:.3f} of n={m.sum()}"
            f" (gears present: {sorted(set(g[m].astype(int)))[:12]})")
        # also the magnitude: D_fresh vs (W/P_lower) delta
        Ws = np.array([sw[int(q)]["W"] for q in Q[m]])
        # P_lower per row: product of primes < g up to anchor... recover from the gear list
    # percentiles
    for name in ("pct", "pct_live", "pct_sec"):
        v = R[name]
        for band, mb in (("all", np.ones_like(v, bool)), ("g<=Q/4", g <= Q / 4), ("Q/4<g<=Q/2", (g > Q / 4) & (g <= Q / 2)),
                         ("g>Q/2", g > Q / 2)):
            mm = mb & np.isfinite(v) & (Q >= 173)
            if mm.sum():
                say(f"  {name:9s} {band:12s}: mean {np.nanmean(v[mm]):.3f}, n={mm.sum()}, frac >0.9: {np.mean(v[mm] > 0.9):.3f}, frac <0.1: {np.mean(v[mm] < 0.1):.3f}")
    # size of the structural bias: 2 N_cur_below/g vs D_fresh, at Q=1499
    for Qx in (997, 1499, 4999):
        m = Q == Qx
        if not m.any():
            continue
        bias = -2.0 * (Nc[m] - R["N_live"][m]) / g[m]
        resid = Df[m] - bias
        say(f"  Q={Qx}: D_fresh = structural bias -2(N_cur-N_live)/g + D_live; bias range {bias.min():.1f}..{bias.max():.1f},"
            f" corr(D_fresh, bias) = {np.corrcoef(Df[m], bias)[0, 1]:+.3f}; |D_live| max {np.abs(Dl[m]).max():.1f},"
            f" |D_fresh| max {np.abs(Df[m]).max():.1f}; sum D_fresh {Df[m].sum():.1f}, sum bias {bias.sum():.1f}, sum D_live {Dl[m].sum():.1f}")


def m6_bands(tag, R, sw):
    say(f"\n== M6 live-zone strike ratio by band t = ln g / ln Q', anchor {tag} ==")
    Q = R["Q"]
    g = R["g"]
    Qn = np.array([sw[int(q)]["Qn"] for q in Q])
    t = np.log(g) / np.log(Qn)
    fair_live = 2.0 * R["N_live"] / g
    fair_win = 2.0 * R["N_cur"] / g
    fair_sec = 2.0 * R["N_cur_sec"] / g
    say("  pooled over levels Q >= 500: band, n rows, sum fresh / sum(2 N_live/g), sum fresh / sum(2 N_cur/g),"
        " sum fresh_sec / sum(2 N_cur_sec/g), mean pct_live, mean pct_sec")
    edges = [0.0, 0.4, 0.5, 0.6, 0.667, 0.75, 0.85, 0.95, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t >= lo) & (t < hi) & (Q >= 500)
        if m.sum() == 0:
            continue
        say(f"  t in [{lo:.3f},{hi:.3f}): n={m.sum():6d}  live {R['fresh'][m].sum() / max(fair_live[m].sum(), 1e-9):.3f}"
            f"  window {R['fresh'][m].sum() / max(fair_win[m].sum(), 1e-9):.3f}"
            f"  section {R['fresh_sec'][m].sum() / max(fair_sec[m].sum(), 1e-9):.3f}"
            f"  pct_live {np.nanmean(R['pct_live'][m]):.3f}  pct_sec {np.nanmean(R['pct_sec'][m]):.3f}"
            f"  (e^gamma = 1.781)")
    # by level, the top band ratio and the whole-window ratio
    say("  per named level: ratio live (t>0.75), ratio live (t<0.6), ratio section (all gears), E_live")
    for Qx in (173, 499, 997, 1499, 2999, 4999):
        if Qx not in sw:
            continue
        m = Q == Qx
        top = m & (t > 0.75)
        low = m & (t < 0.6)
        say(f"    Q={Qx:5d}: live top {R['fresh'][top].sum() / max(fair_live[top].sum(), 1e-9):.3f} (n={top.sum()}),"
            f" live low {R['fresh'][low].sum() / max(fair_live[low].sum(), 1e-9):.3f} (n={low.sum()}),"
            f" section {R['fresh_sec'][m].sum() / max(fair_sec[m].sum(), 1e-9):.3f}, E_live {sw[Qx]['E_live']:.3f},"
            f" sum D_live/N_live over gears {np.sum(R['D_live'][m] / np.maximum(R['N_live'][m], 1)):+.3f},"
            f" sum D_fresh/N_cur {np.sum(R['D_fresh'][m] / np.maximum(R['N_cur'][m], 1)):+.3f}, -ln R = {-np.log(sw[Qx]['R']):+.3f}")


def m4_histogram(tag, R, sw, primes):
    say(f"\n== M4 residue histograms at the extreme gears, anchor {tag} ==")
    Qs = sorted(sw)
    picks = []
    # window extreme, live extreme, and the named levels' argmax
    Qx = max(Qs, key=lambda Q: abs(sw[Q]["maxDf"]))
    picks.append((Qx, int(sw[Qx]["gDf"]), "window extreme |D_fresh|"))
    Ql = max(Qs, key=lambda Q: abs(sw[Q]["maxDl"]))
    picks.append((Ql, int(sw[Ql]["gDl"]), "live-zone extreme |D_live|"))
    Qsx = max(Qs, key=lambda Q: abs(sw[Q]["maxDfs"]))
    picks.append((Qsx, int(sw[Qsx]["gDfs"]), "section extreme |D_fresh_sec|"))
    for Q in (173, 1499):
        if Q in sw:
            picks.append((Q, int(sw[Q]["gDf"]), f"argmax at Q={Q}"))
            picks.append((Q, int(sw[Q]["gDl"]), f"live argmax at Q={Q}"))
    seen = set()
    for Q, gs, label in picks:
        if (Q, gs) in seen:
            continue
        seen.add((Q, gs))
        Qn = [p for p in primes if p > Q][0]
        a = int(sw[Q]["a"])
        ks, amask, alive, k_lo, k_hi = state_before(Q, Qn, a, gs, primes)
        u = tooth(gs)
        W = ks.size
        k_live = (gs * gs - 1) // 6
        k_s = (Q * Q - 1) // 6 + 1
        for zone, sel in (("window", alive), ("live", alive[alive >= k_live]), ("section", alive[alive >= k_s])):
            if sel.size == 0:
                continue
            h = np.bincount(sel % gs, minlength=gs)
            fair = sel.size / gs
            pair = h[1:(gs + 1) // 2] + h[gs - 1: (gs - 1) // 2: -1]  # h[v] + h[g-v], v=1..(g-1)/2
            real = h[u] + h[gs - u]
            rank = (np.sum(pair < real) + 0.5 * np.sum(pair == real)) / pair.size
            say(f"  {label}: Q={Q} g={gs} u={u} zone={zone}: N={sel.size}, W/g={W / gs:.0f}, n(r) mean {fair:.2f} min {h.min()} max {h.max()}"
                f" std {h.std():.2f} (sqrt(mean) {np.sqrt(fair):.2f}); teeth n(u)={h[u]} n(-u)={h[gs - u]}, pair {real} vs 2N/g {2 * fair:.1f},"
                f" D={real - 2 * fair:+.1f}; pair counts mean {pair.mean():.1f} std {pair.std():.2f}; rank of real pair {rank:.3f}")
        # struck columns: which anchor classes.  Compare struck-survivor residues mod 5, 7 with survivors' overall.
        hit = alive[(alive % gs == u) | (alive % gs == gs - u)]
        say(f"    struck survivors: {hit.size}; first 12 columns {hit[:12].tolist()}; below k_live: {int(np.sum(hit < k_live))}")
        for q in (5, 7, 11, 13):
            if q > a:
                break
            hs = np.bincount(hit % q, minlength=q)
            ha = np.bincount(alive % q, minlength=q)
            exp = ha / max(alive.size, 1) * hit.size
            say(f"    mod {q}: struck by class {hs.tolist()} vs expected from survivors {np.round(exp, 1).tolist()}")
        # the two teeth separately, by position: first/second/third third of the window
        thirds = np.searchsorted(hit, [k_lo + W // 3, k_lo + 2 * W // 3])
        al3 = np.searchsorted(alive, [k_lo + W // 3, k_lo + 2 * W // 3])
        n3 = np.diff(np.concatenate([[0], thirds, [hit.size]]))
        a3 = np.diff(np.concatenate([[0], al3, [alive.size]]))
        say(f"    by window third: struck {n3.tolist()} vs fair 2N/g per third {np.round(2 * a3 / gs, 1).tolist()}")


def m5_anchor_discrepancy(primes):
    say("\n== M5 exact interval discrepancy of the anchor {5..13} and its 180 re-toothings ==")
    gears = [5, 7, 11, 13]
    P = 5005
    ks = np.arange(P)
    # all symmetric tooth vectors: for each gear q a half-width v in 1..(q-1)/2 (teeth +-v)
    import itertools
    choices = [list(range(1, (q - 1) // 2 + 1)) for q in gears]
    real = tuple(min(tooth(q), q - tooth(q)) for q in gears)
    results = []
    for vs in itertools.product(*choices):
        m = np.ones(P, dtype=bool)
        for q, v in zip(gears, vs):
            r = ks % q
            m &= (r != v) & (r != q - v)
        N = int(m.sum())
        rho = N / P
        # prefix sums over two periods for wrap-around intervals
        c = np.concatenate([[0], np.cumsum(np.concatenate([m, m]).astype(np.int64))])
        best = 0.0
        bestL = 0
        prof = {}
        for L in range(1, P // 2 + 1):
            cnt = c[L:L + P] - c[:P]
            d = np.abs(cnt - L * rho).max()
            if d > best:
                best, bestL = d, L
            if L in (50, 100, 200, 300, 500, 1000, 2000, 2502):
                prof[L] = d
        results.append((vs, N, best, bestL, prof))
        if vs == real:
            say(f"  REAL teeth {vs}: N={N}, max interval discrepancy {best:.2f} at L={bestL}; profile by L: "
                + ", ".join(f"L={L}:{d:.2f}" for L, d in prof.items()))
    bests = np.array([r[2] for r in results])
    rank = np.mean(bests < [r[2] for r in results if r[0] == real][0])
    say(f"  180 re-toothings: max discrepancy min {bests.min():.2f} median {np.median(bests):.2f} max {bests.max():.2f};"
        f" real at percentile {rank:.3f}")
    say(f"  so for a FIXED anchor {{5..13}} and ANY gear g, |D_raw(g)| <= 2 x {bests.max():.2f} + (W/5005)|delta_A(g)| for every Q:"
        f" the sub-period part is bounded by the worst re-toothing's interval discrepancy.")
    for L in (50, 100, 200, 500, 1000, 2502):
        v = np.array([r[4][L] for r in results])
        say(f"    profile L={L}: discrepancy over 180 re-toothings min {v.min():.2f} median {np.median(v):.2f} max {v.max():.2f};"
            f" sqrt(L rho) ~ {np.sqrt(L * 0.2967):.2f}")


def main():
    primes = primes_upto(5300)
    tags = sys.argv[1].split(",") if len(sys.argv) > 1 else ["13", "min", "19"]
    for tag in tags:
        if not os.path.exists(os.path.join(RES, f"gears_{tag}.npz")):
            say(f"missing results for anchor {tag}")
            continue
        R, sw = load(tag)
        m1_growth(tag, R, sw, primes)
        m2_ratio_budget(tag, sw)
        m3_signs(tag, R, sw)
        m6_bands(tag, R, sw)
        m4_histogram(tag, R, sw, primes)
    m5_anchor_discrepancy(primes)
    with open(os.path.join(RES, "mechanism.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
