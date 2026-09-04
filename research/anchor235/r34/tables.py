"""Branch 7b, part three: tables for the write-up, the Buchstab check, and the residual.

  T1  per-gear excerpts at the named levels (anchor {5..13} and A_min): the first gears above
      the anchor, the argmax gears of |D_fresh| and |D_live|, the gear nearest Q/2, the top gear.
  T2  the band law fresh_g / (2 N_live / g) as a function of t = ln g / ln Q' in 20 bins pooled
      over Q >= 500, against the first-order Buchstab prediction omega(2/t - 1) / omega(2/t).
  T3  residual after the band law: r_g = fresh_g - beta(t) 2 N_live / g; its size against
      sqrt(2 N_live / g) and the per-level sum of positive parts E'' = sum max(r_g / N_live, 0).

Usage: uv run python research/anchor235/r34/tables.py
Writes results/tables.txt.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


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


def omega_table(umax=8.0, h=1e-3):
    """Buchstab omega on a grid: omega(u) = 1/u on [1,2], (u omega(u))' = omega(u-1)."""
    n = int(umax / h) + 1
    u = np.arange(n) * h
    w = np.zeros(n)
    i1 = int(1 / h)
    i2 = int(2 / h)
    w[i1:i2 + 1] = 1.0 / u[i1:i2 + 1]
    uw = u * w
    for i in range(i2 + 1, n):
        # uw(u_i) = uw(u_{i-1}) + h * (omega(u_{i-1}-1) + omega(u_i - 1)) / 2
        uw[i] = uw[i - 1] + h * (w[i - 1 - i1] + w[i - i1]) / 2
        w[i] = uw[i] / u[i]
    return u, w


def omega(x, u, w):
    return np.interp(x, u, w)


def t1_excerpts(tag, R, sw):
    say(f"\n== T1 per-gear excerpts, anchor {tag} ==")
    say("  columns: g, t=ln g/ln Q', N_cur, N_live, raw, fresh, D_raw, D_fresh, D_live, fresh/(2N_live/g), pct_live, N_sec, fresh_sec, D_sec")
    for Q in (17, 59, 173, 499, 997, 1499, 2999, 4999):
        if Q not in sw:
            continue
        d = sw[Q]
        m = R["Q"] == Q
        g = R["g"][m]
        order = np.argsort(g)
        idx = np.nonzero(m)[0][order]
        g = R["g"][idx]
        Qn = d["Qn"]
        say(f"  Q={Q} Q'={int(Qn)} a={int(d['a'])} window k in [{int(d['k_lo'])}, {int(d['k_hi'])}] W={int(d['W'])} S={int(d['S'])}"
            f" N_A={int(d['N_A'])} N_A_sec={int(d['N_A_sec'])} twins={int(d['twins'])} twins_sec={int(d['twins_sec'])}"
            f" sum fresh={int(d['sum_fresh'])} check={'OK' if d['ok'] == 1 else 'FAIL'} R={d['R']:.3f} R_sec={d['R_sec']:.3f}")
        picks = set()
        picks.update(idx[:3].tolist())
        jf = idx[int(np.argmax(np.abs(R["D_fresh"][idx])))]
        jl = idx[int(np.argmax(np.abs(R["D_live"][idx])))]
        js = idx[int(np.argmax(np.abs(R["D_fresh_sec"][idx])))]
        jh = idx[int(np.argmin(np.abs(g - Q / 2)))]
        jq = idx[int(np.argmin(np.abs(g - Q ** (2 / 3))))]
        picks.update([jf, jl, js, jh, jq, idx[-1]])
        for j in sorted(picks, key=lambda j: R["g"][j]):
            gg = int(R["g"][j])
            tag_ = []
            if j == jf:
                tag_.append("argmax|D_fresh|")
            if j == jl:
                tag_.append("argmax|D_live|")
            if j == js:
                tag_.append("argmax|D_sec|")
            if j == jh:
                tag_.append("~Q/2")
            if j == jq:
                tag_.append("~Q^(2/3)")
            if j == idx[-1]:
                tag_.append("top")
            if j in idx[:3]:
                tag_.append("above anchor")
            fl = 2 * R["N_live"][j] / gg
            say(f"    {gg:5d} {np.log(gg) / np.log(Qn):.3f} {int(R['N_cur'][j]):8d} {int(R['N_live'][j]):8d} {int(R['raw'][j]):7d} {int(R['fresh'][j]):6d}"
                f" {R['D_raw'][j]:+8.1f} {R['D_fresh'][j]:+8.1f} {R['D_live'][j]:+8.1f} {R['fresh'][j] / fl if fl > 0 else float('nan'):6.3f}"
                f" {R['pct_live'][j]:5.3f} {int(R['N_cur_sec'][j]):6d} {int(R['fresh_sec'][j]):4d} {R['D_fresh_sec'][j]:+6.1f}  {' '.join(tag_)}")


def t2_band_law(R, sw, u, w):
    say("\n== T2 band law fresh/(2 N_live/g) vs t = ln g / ln Q', pooled over Q >= 500, against Buchstab omega(2/t-1)/omega(2/t) ==")
    Q = R["Q"]
    g = R["g"]
    Qn = np.array([sw[int(q)]["Qn"] for q in Q])
    t = np.log(g) / np.log(Qn)
    fair = 2.0 * R["N_live"] / g
    fair_sec = 2.0 * R["N_cur_sec"] / g
    edges = np.linspace(0.3, 1.0, 15)
    edges = np.concatenate([[0.0], edges])
    say("  t band, n, measured ratio (live), measured ratio (section), Buchstab first order, integrated first order")
    beta = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t >= lo) & (t < hi + (1e-9 if hi >= 1.0 else 0)) & (Q >= 500)
        if m.sum() == 0:
            continue
        tc = 0.5 * (lo + hi) if lo > 0 else 0.5 * hi
        ux = 2.0 / tc
        um = ux - 1.0
        first = omega(um, u, w) / omega(ux, u, w)
        # integrated: the m-range (g, x/g) weighted by dy = e^{v L} L dv with L = ln g at the band centre
        # use a representative ln g = t_c * ln Q' with ln Q' ~ mean over rows in the band
        L = np.mean(np.log(g[m]))
        v = np.linspace(1.0, um, 400)
        wv = np.exp((v - um) * L)
        integ = np.trapz(omega(v, u, w) * wv, v) / max(np.trapz(wv, v), 1e-12)
        integrated = integ / omega(ux, u, w)
        meas = R["fresh"][m].sum() / fair[m].sum()
        meas_s = R["fresh_sec"][m].sum() / max(fair_sec[m].sum(), 1e-9)
        beta[(lo, hi)] = meas
        say(f"  [{lo:.3f},{hi:.3f}) n={m.sum():6d}  live {meas:.3f}  section {meas_s:.3f}  omega ratio {first:.3f}  integrated {integrated:.3f}")
    return t, beta, edges


def t3_residual(R, sw, t, beta, edges):
    say("\n== T3 residual after the band law ==")
    Q = R["Q"]
    g = R["g"]
    fair = 2.0 * R["N_live"] / g
    b = np.ones_like(t)
    for (lo, hi), val in beta.items():
        m = (t >= lo) & (t < hi + (1e-9 if hi >= 1.0 else 0))
        b[m] = val
    resid = R["fresh"] - b * fair
    scale = np.sqrt(np.maximum(fair, 1e-9))
    z = resid / scale
    for lo, hi in ((17, 500), (500, 1500), (1500, 3000), (3000, 5001)):
        m = (Q >= lo) & (Q < hi) & (fair > 20)
        if m.sum():
            say(f"  Q in [{lo},{hi}), gears with fair share > 20: n={m.sum()}, z = resid/sqrt(fair): mean {z[m].mean():+.3f}, std {z[m].std():.3f},"
                f" max |z| {np.abs(z[m]).max():.2f}, quantiles 1/50/99: {np.percentile(z[m], [1, 50, 99]).round(2)}")
    say("  per named level: max |resid| @g, max |D_live| @g, E'' = sum max(resid/N_live,0), E_live, -ln R")
    for Qx in (173, 499, 997, 1499, 2999, 4999):
        m = Q == Qx
        if not m.any():
            continue
        j = np.nonzero(m)[0][int(np.argmax(np.abs(resid[m])))]
        jl = np.nonzero(m)[0][int(np.argmax(np.abs(R["D_live"][m])))]
        Epp = float(np.sum(np.maximum(resid[m] / np.maximum(R["N_live"][m], 1), 0)))
        say(f"    Q={Qx:5d}: max|resid| {resid[j]:+7.1f} @g={int(g[j])} (z={z[j]:+.2f}); max|D_live| {R['D_live'][jl]:+7.1f} @g={int(g[jl])};"
            f" E''={Epp:.3f}  E_live={sw[Qx]['E_live']:.3f}  -ln R={-np.log(sw[Qx]['R']):+.3f}  room={sw[Qx]['room']:.2f}")


def main():
    u, w = omega_table()
    say(f"Buchstab omega check: omega(2)={omega(2.0, u, w):.4f} omega(3)={omega(3.0, u, w):.4f} omega(4)={omega(4.0, u, w):.4f} (limit e^-gamma = 0.5615)")
    for tag in ("13", "min"):
        R, sw = load(tag)
        t1_excerpts(tag, R, sw)
        if tag == "13":
            t, beta, edges = t2_band_law(R, sw, u, w)
            t3_residual(R, sw, t, beta, edges)
    with open(os.path.join(RES, "tables.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
