"""sf_engine.py - the sub-machine fibre budget, exact, on the real window.

Branch R2.c.ii (research/proof/submachine_fibres.md).

The engine is BBMST Invent. math. 228 (2022) Theorem 3.1 with a stage-dependent partition
(valid for any partition, r52 section 2.2).  Gears in increasing order, delta = 1/2.

  * small gears (g <= a, the SUB-MACHINE S = {5..a}, Q_s = prod S):
        the parts at stage i are the classes mod Q_{<i} = g_1...g_{i-1}   (refining)
  * big gears (g > a):
        the parts are FROZEN at the classes mod Q_s                        (the fibres)

A fibre is an arithmetic progression of step Q_s with m = L/Q_s columns.  Cut index t = |S|;
t = 0 is r52's one-block engine (Q_s = 1), t = infinity is r51's fully refining fibre engine.

Everything is computed exactly from the real teeth u_g = 6^{-1} mod g on the real window.

Usage: uv run python research/anchor235/r54/sf_engine.py [qmax]
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
    for p in PR:
        if p > q:
            return p
    raise ValueError


def window(q):
    """r51/r52 convention: k_lo = floor((q+1)/6)+1, k_hi = (q'^2-1)/6."""
    qp = next_prime(q)
    lo = (q + 1) // 6 + 1
    hi = (qp * qp - 1) // 6
    return lo, hi


def gears_of(q):
    return [p for p in PR if 5 <= p <= q]


def run(q, t, collect=False):
    """Exact sub-machine fibre budget at cut t (t small gears).  Returns a dict."""
    lo, hi = window(q)
    L = hi - lo + 1
    k = np.arange(lo, hi + 1, dtype=np.int64)
    gs = gears_of(q)
    Qs = 1
    for g in gs[:t]:
        Qs *= g

    w = np.full(L, 1.0 / L)
    alive = np.ones(L, dtype=bool)
    Qlt = 1                      # product of the gears already processed (head only)
    eta = 0.0
    rows = []
    for i, g in enumerate(gs):
        # ---- the partition at this stage
        if i < t:
            mod = Qlt                       # refining
        else:
            mod = Qs                        # frozen at the sub-machine period
        if mod > L:
            part = np.arange(L, dtype=np.int64)
            npart = L
        else:
            part = k % mod
            npart = mod
        # ---- the gear's strikes
        u = pow(6, -1, g)
        r = k % g
        struck = (r == u) | (r == (g - u) % g)
        # BBMST's B_i is the whole strike set of the new modulus.  Columns already covered
        # carry zero weight whenever every earlier alpha was <= 1/2, so this agrees with
        # "newly covered" there, and is the conservative reading when some alpha > 1/2.
        fresh = struck
        # ---- the moments under P_{i-1}
        tot = np.bincount(part, weights=w, minlength=npart)
        num = np.bincount(part[fresh], weights=w[fresh], minlength=npart)
        nz = tot > 0
        a = np.zeros(npart)
        a[nz] = num[nz] / tot[nz]
        M1 = float(num.sum())
        M2 = float((tot * a * a).sum())
        term = min(M1, M2)
        eta += term
        # ---- diagnostics on the frozen stages
        if i >= t:
            s_f = np.bincount(part, weights=alive.astype(np.float64), minlength=npart)
            live = s_f > 0
            nlive = int(live.sum())
            fr_f = np.bincount(part[fresh], minlength=npart).astype(np.float64)
            D = np.zeros(npart)
            D[live] = fr_f[live] - 2.0 * s_f[live] / g
            maxD = float(np.abs(D[live]).max()) if nlive else 0.0
            meansurv = float(s_f[live].mean()) if nlive else 0.0
            minsurv = float(s_f[live].min()) if nlive else 0.0
        else:
            nlive, maxD, meansurv, minsurv = npart, 0.0, L / max(npart, 1), L / max(npart, 1)
        rows.append(dict(g=g, mod=mod, npart=npart, nlive=nlive, m=L / max(mod, 1),
                         M1=M1, M2=M2, term=term, V=M2 - M1 * M1, eta=eta,
                         amax=float(a.max()), maxD=maxD,
                         meansurv=meansurv, minsurv=minsurv,
                         ideal=4.0 / (g * g)))
        # ---- reweight (delta = 1/2)
        af = a[part]
        with np.errstate(divide="ignore", invalid="ignore"):
            fac_on = np.where(af > 0, np.maximum(0.0, 2.0 - 1.0 / np.where(af > 0, af, 1.0)), 0.0)
        fac_off = np.minimum(1.0 / np.maximum(1e-300, 1.0 - af), 2.0)
        w = np.where(fresh, w * fac_on, w * fac_off)
        alive &= ~struck
        if i < t:
            Qlt *= g
    res = dict(q=q, t=t, Qs=Qs, L=L, eta=eta, rows=rows,
               nsurv=int(alive.sum()), amax=max(r["amax"] for r in rows),
               sumV=sum(r["V"] for r in rows
                        if r["g"] > (gs[min(t, len(gs)) - 1] if t else 0)))
    return res


def run_refining(q):
    """r51's fibre engine: refine at every gear (collapse once Q_{<i} >= L)."""
    return run(q, t=10 ** 6)


def main():
    qmax = int(sys.argv[1]) if len(sys.argv) > 1 else 1999
    QS = [q for q in (59, 97, 199, 499, 997, 1999, 4999) if q <= qmax]
    say("=" * 100)
    say("A. The exact sub-machine fibre budget eta_SF by cut, real teeth, real window")
    say("=" * 100)
    say("  Q_s = 1 is r52's one-block budget; the last row of each block is r51's fully")
    say("  refining fibre budget.  m = L/Q_s = columns per fibre.  sum4/g^2 = the ideal.")
    say("")
    store = {}
    for q in QS:
        lo, hi = window(q)
        L = hi - lo + 1
        gs = gears_of(q)
        ideal = sum(4.0 / (g * g) for g in gs)
        say("  q = %d   L = W(q) = %d   gears = %d   sum 4/g^2 = %.5f" % (q, L, len(gs), ideal))
        say("    t  sub-machine      Q_s        m    #fibres  eta_SF   sum V_g  max alpha  first g with Q_s*g>L")
        cuts = []
        Qs = 1
        tt = 0
        while Qs <= L and tt <= len(gs):
            cuts.append((tt, Qs))
            if tt == len(gs):
                break
            Qs *= gs[tt]
            tt += 1
        for (t, Qs) in cuts:
            res = run(q, t)
            store[(q, t)] = res
            firstbad = next((g for g in gs[t:] if Qs * g > L), None)
            sm = "{5..%d}" % gs[t - 1] if t else "{}"
            say("    %2d  %-12s %8d  %7.1f  %8d  %7.4f  %7.4f  %8.4f   %s"
                % (t, sm, Qs, L / Qs, res["rows"][min(t, len(res["rows"]) - 1)]["nlive"],
                   res["eta"], res["sumV"], res["amax"], str(firstbad)))
        ref = run_refining(q)
        store[(q, "ref")] = ref
        say("    -- fully refining (r51 fibre engine): eta = %.4f" % ref["eta"])
        say("")

    say("=" * 100)
    say("B. Per-gear detail at the exactness-preserving cut (largest Q_s with Q_s*q <= L)")
    say("=" * 100)
    for q in QS:
        lo, hi = window(q)
        L = hi - lo + 1
        gs = gears_of(q)
        best_t, best_Qs = 0, 1
        Qs = 1
        for t in range(len(gs) + 1):
            if Qs * q <= L:
                best_t, best_Qs = t, Qs
            if t < len(gs):
                Qs *= gs[t]
        res = store.get((q, best_t)) or run(q, best_t)
        say("  q = %d  cut t = %d  Q_s = %d  m = %.1f  eta_SF = %.5f  (r51 fibre %.4f, r52 block %.4f)"
            % (q, best_t, best_Qs, L / best_Qs, res["eta"],
               store[(q, "ref")]["eta"], store[(q, 0)]["eta"]))
        say("      g   parts   #live  mean surv/fibre    M1        M2      4/g^2     V_g      cum eta   max|D_f|")
        rows = res["rows"]
        show = rows[:8] + rows[8::max(1, len(rows) // 10)]
        seen = set()
        for r in show:
            if r["g"] in seen:
                continue
            seen.add(r["g"])
            say("   %6d %7d %7d %10.1f  %9.6f %9.6f %9.6f %9.6f %8.4f %9.1f"
                % (r["g"], r["npart"], r["nlive"], r["meansurv"], r["M1"], r["M2"],
                   r["ideal"], r["V"], r["eta"], r["maxD"]))
        say("")

    with open(os.path.join(OUT, "sf_engine.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    say("wrote " + os.path.join(OUT, "sf_engine.txt"))


if __name__ == "__main__":
    main()
