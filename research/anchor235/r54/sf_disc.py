"""sf_disc.py - the obstruction named and measured: the cross-fibre discrepancy.

Branch R2.c.ii.  With the partition frozen at the classes mod Q_s the engine's term for a big
gear is

    M^(2)_g = E_f[alpha_g(f)^2] = (E_f alpha)^2 + Var_f(alpha) ,
    alpha_g(f) = (strikes of g on fibre f's survivors) / (fibre f's survivors) ,

and over a FULL PERIOD of the gears below g every fibre gives alpha = 2/g exactly (CRT), so
Var = 0 and eta = sum 4/g^2 < 0.36455.  In the window Var_f is a finite object per q: the L2
discrepancy of g's strikes across the residue classes of Q_s.  This script measures it,
prices a uniform bound |D_f| <= C against it, and sweeps every tooth-pair of every gear to see
how much a single adversarial phase can buy.

  D_f(g) = (strikes of g on f's survivors) - 2 s_f / g          (per-fibre discrepancy)

7b (anchor_window.md) proved/measured the WINDOW-aggregate discrepancy of a gear on the
anchor's openings; this is the same object resolved per residue class of the anchor, and
squared.

Usage: uv run python research/anchor235/r54/sf_disc.py [qmax]
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i:: i] = bytearray(len(sieve[i * i:: i]))
    return [i for i in range(n + 1) if sieve[i]]


PR = primes_upto(20000)


def next_prime(q):
    return next(p for p in PR if p > q)


def window(q):
    qp = next_prime(q)
    return (q + 1) // 6 + 1, (qp * qp - 1) // 6


def gears_of(q):
    return [p for p in PR if 5 <= p <= q]


def exact_cut(q):
    """largest gear-primorial Q_s with Q_s * q <= W(q): every fibre then holds whole
    classes mod every gear, so every per-fibre first moment is exact up to the ceiling."""
    lo, hi = window(q)
    L = hi - lo + 1
    gs = gears_of(q)
    Qs, t, bt, bQ = 1, 0, 0, 1
    for t in range(len(gs) + 1):
        if Qs * q <= L:
            bt, bQ = t, Qs
        if t < len(gs):
            Qs *= gs[t]
    return bt, bQ, L


def analyse(q, phase_sweep=True):
    t, Qs, L = exact_cut(q)
    lo, hi = window(q)
    k = np.arange(lo, hi + 1, dtype=np.int64)
    gs = gears_of(q)
    part = k % Qs
    w = np.full(L, 1.0 / L)
    alive = np.ones(L, dtype=bool)
    Qlt = 1
    rows = []
    eta_real = eta_maxphase = eta_apriori = 0.0
    for i, g in enumerate(gs):
        mod = Qlt if i < t else Qs
        pid = (k % mod) if mod > 1 else np.zeros(L, dtype=np.int64)
        npart = mod
        u = pow(6, -1, g)
        r = k % g
        struck = (r == u) | (r == (g - u) % g)
        tot = np.bincount(pid, weights=w, minlength=npart)
        num = np.bincount(pid[struck], weights=w[struck], minlength=npart)
        nz = tot > 0
        a = np.zeros(npart)
        a[nz] = num[nz] / tot[nz]
        M1 = float(num.sum())
        M2 = float((tot * a * a).sum())
        eta_real += M2
        # --- the per-fibre discrepancy and the phase sweep, frozen stages only
        if i >= t:
            s_f = np.bincount(pid, weights=alive.astype(np.float64), minlength=npart)
            fr = np.bincount(pid[struck & alive], minlength=npart).astype(np.float64)
            live = s_f > 0
            D = fr[live] - 2.0 * s_f[live] / g
            maxD = float(np.abs(D).max())
            rmsD = float(np.sqrt((D * D).mean()))
            smin = float(s_f[live].min())
            if phase_sweep and npart * g <= 4_000_000:
                A = np.bincount(pid * g + r, weights=w, minlength=npart * g).reshape(npart, g)
                T = A.sum(axis=1)
                good = T > 0
                vs = np.arange(1, (g + 1) // 2)
                al = (A[:, vs] + A[:, (g - vs) % g])
                al = al[good] / T[good, None]
                m2 = (T[good, None] * al * al).sum(axis=0)
                mx = float(m2.max())
                rank = float((m2 <= M2 + 1e-15).mean())
            else:
                mx, rank = float("nan"), float("nan")
        else:
            maxD = rmsD = 0.0
            smin = L / max(npart, 1)
            mx, rank = M2, float("nan")
        eta_maxphase += mx if mx == mx else M2
        # a priori: every strike of g lands on a survivor of the fibre
        pi = float(alive.sum()) / L
        eta_apriori += min(1.0, (2.0 / g) / max(pi, 1e-12)) ** 2
        rows.append(dict(g=g, M1=M1, M2=M2, V=M2 - M1 * M1, ideal=4.0 / (g * g),
                         maxD=maxD, rmsD=rmsD, smin=smin, mx=mx, rank=rank,
                         eta_real=eta_real, eta_mx=eta_maxphase, eta_ap=eta_apriori, pi=pi))
        af = a[pid]
        fac_on = np.where(af > 0, np.maximum(0.0, 2.0 - 1.0 / np.where(af > 0, af, 1.0)), 0.0)
        fac_off = np.minimum(1.0 / np.maximum(1e-300, 1.0 - af), 2.0)
        w = np.where(struck, w * fac_on, w * fac_off)
        alive &= ~struck
    return dict(q=q, t=t, Qs=Qs, L=L, rows=rows, gs=gs)


def main():
    qmax = int(sys.argv[1]) if len(sys.argv) > 1 else 1999
    QS = [q for q in (59, 97, 199, 499, 997, 1999) if q <= qmax]
    say("=" * 100)
    say("A. The cross-fibre term, the per-fibre discrepancy, and the single-phase adversary")
    say("=" * 100)
    say("   Q_s is the exactness cut (largest gear-primorial with Q_s*q <= W, so every fibre")
    say("   holds whole classes mod every gear).  eta_real = the machine's own budget;")
    say("   eta_maxphase = each gear's tooth pair replaced by its WORST pair on the same")
    say("   survivors; eta_apriori = every strike of g lands on a survivor ((2/g)/Pi)^2.")
    say("")
    say("    q     Q_s   fibres   sum V_g   eta_real  eta_maxphase  eta_apriori  max|D_f| @g   mean rank")
    store = {}
    for q in QS:
        res = analyse(q)
        store[q] = res
        rows = res["rows"]
        tail = [r for r in rows if r["g"] > (res["gs"][res["t"] - 1] if res["t"] else 0)]
        sumV = sum(r["V"] for r in tail)
        mD = max(tail, key=lambda r: r["maxD"])
        ranks = [r["rank"] for r in tail if r["rank"] == r["rank"]]
        say("  %5d %6d %7d  %8.5f  %8.5f  %11.5f  %11.4f  %6.1f @%-5d %8.3f"
            % (q, res["Qs"], res["Qs"], sumV, rows[-1]["eta_real"], rows[-1]["eta_mx"],
               rows[-1]["eta_ap"], mD["maxD"], mD["g"],
               float(np.mean(ranks)) if ranks else float("nan")))
    say()

    say("=" * 100)
    say("B. Pricing a uniform per-fibre discrepancy bound |D_f(g)| <= C")
    say("=" * 100)
    say("   With |D_f| <= C, alpha_f = 2/g + D_f/s_f, so")
    say("       M^(2)_g <= (2/g + C/s_min)^2   and   eta <= sum_g (2/g + C/s_min(g))^2 .")
    say("   C* = the largest C for which that stays below 1.  7b's proved anchor rigidity")
    say("   gives the WINDOW-aggregate |D| below 30 (max |D_raw| 26.3 over 663 levels).")
    say("")
    say("    q     Q_s   room=1-sum4/g^2      C*    max|D_f|  rms|D_f|  C*/max   eta at C=30")
    for q in QS:
        res = store[q]
        rows = res["rows"]
        tail = [r for r in rows if r["g"] > (res["gs"][res["t"] - 1] if res["t"] else 0)]
        head = sum(r["M2"] for r in rows if r not in tail)
        ideal = sum(4.0 / (g * g) for g in res["gs"])

        def eta_of(C):
            e = head
            for r in tail:
                e += min(1.0, 2.0 / r["g"] + C / max(r["smin"], 1.0)) ** 2
            return e
        def bisect(f):
            lo_, hi_ = 0.0, 1e7
            for _ in range(80):
                mid = (lo_ + hi_) / 2
                if f(mid) < 1.0:
                    lo_ = mid
                else:
                    hi_ = mid
            return lo_
        Cinf = bisect(eta_of)
        mx = max(r["maxD"] for r in tail)
        rm = max(r["rmsD"] for r in tail)
        say("  %5d %6d      %8.5f   %8.3f  %8.1f  %8.2f  %7.2f      %8.4f"
            % (q, res["Qs"], 1 - ideal, Cinf, mx, rm, Cinf / max(mx, 1e-9), eta_of(30.0)))
    say()

    say("=" * 100)
    say("C. Per-gear detail at q = 997 (cut Q_s = 35): where the variance sits")
    say("=" * 100)
    if 997 in store:
        res = store[997]
        say("      g   Pi_<g    M1        M2       4/g^2      V_g     max|D_f|  rms|D_f|  min surv  worst-pair M2  rank")
        rows = res["rows"]
        sel = rows[:6] + rows[6::12]
        seen = set()
        for r in sel:
            if r["g"] in seen:
                continue
            seen.add(r["g"])
            say("  %5d  %.5f %9.6f %9.6f %9.6f %9.6f %8.1f %8.2f %9.0f  %12.6f %6.3f"
                % (r["g"], r["pi"], r["M1"], r["M2"], r["ideal"], r["V"], r["maxD"],
                   r["rmsD"], r["smin"], r["mx"], r["rank"]))
    say()

    say("=" * 100)
    say("D. Full-period control: the same gear on a full period of the gears below it")
    say("=" * 100)
    say("   If the interval is a full period of {5..g^-} the CRT makes every D_f exactly 0.")
    for gtest, low in ((11, [5, 7]), (13, [5, 7, 11]), (17, [5, 7, 11, 13])):
        P = gtest
        for g in low:
            P *= g
        kk = np.arange(P, dtype=np.int64)
        alive = np.ones(P, dtype=bool)
        for g in low:
            u = pow(6, -1, g)
            rr = kk % g
            alive &= ~((rr == u) | (rr == (g - u) % g))
        Qs = 35
        pid = kk % Qs
        u = pow(6, -1, gtest)
        rr = kk % gtest
        st = (rr == u) | (rr == (gtest - u) % gtest)
        s_f = np.bincount(pid, weights=alive.astype(float), minlength=Qs)
        fr = np.bincount(pid[st & alive], minlength=Qs).astype(float)
        live = s_f > 0
        D = fr[live] - 2.0 * s_f[live] / gtest
        say("   gears below %2d on the full period of {5..%d} = %8d, Q_s = 35: "
            "max|D_f| = %.3e  (fibres %d, %d columns each)"
            % (gtest, gtest, P, float(np.abs(D).max()), int(live.sum()), P // Qs))

    say()
    say("=" * 100)
    say("E. Growth of the needed per-fibre discrepancy bound C*(q) (no phase sweep)")
    say("=" * 100)
    say("   the requirement is |D_f(g)| <= C*(q) for every fibre and every gear; the")
    say("   square-root heuristic gives |D_f| ~ sqrt(2 s_f/g), the measured max is in the table")
    say("    q     Q_s    #gears   min surv/fibre     C*     measured max|D_f|   C*/max   C*/q^1.5")
    for q in (43, 59, 97, 149, 199, 307, 499, 701, 997, 1499, 1999):
        if q > qmax:
            continue
        res = store.get(q) or analyse(q, phase_sweep=False)
        rows = res["rows"]
        tail = [r for r in rows if r["g"] > (res["gs"][res["t"] - 1] if res["t"] else 0)]
        head = sum(r["M2"] for r in rows if r not in tail)

        def eta_of(C, tail=tail, head=head):
            e = head
            for r in tail:
                e += min(1.0, 2.0 / r["g"] + C / max(r["smin"], 1.0)) ** 2
            return e
        lo_, hi_ = 0.0, 1e7
        for _ in range(80):
            mid = (lo_ + hi_) / 2
            if eta_of(mid) < 1.0:
                lo_ = mid
            else:
                hi_ = mid
        mx = max(r["maxD"] for r in tail)
        say("  %5d %6d %7d %14.0f %9.3f %14.1f %11.2f %10.5f"
            % (q, res["Qs"], len(res["gs"]), min(r["smin"] for r in tail), lo_, mx,
               lo_ / max(mx, 1e-9), lo_ / q ** 1.5))

    with open(os.path.join(OUT, "sf_disc.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
