"""r65 / position-length frontier, part 2: the prefix [1, W] at every rung q = 23..19997.

REDUCTION (used, and gated by direct sieving at four rungs).  For 1 <= k <= W = (q'^2-1)/6 a
member 6k+-1 is 1, a prime <= q (a gear, struck), a prime > q (not struck), or composite -- and
a composite below q'^2 has least prime factor <= q unless it is q'^2 itself.  So the openings of
{5..q} in [1, W] are exactly the twin-prime columns with 6k-1 > q, plus the column W when
q'^2 - 2 is prime (the square-gate rider).  One prime sieve to 20011^2 serves every rung.

Self-contained, numpy only.  Peak memory about 700 MB.
Run: uv run python research/anchor235/r65/pf_window.py
"""
import os
import sys
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

QMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 19997


def sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def gate(q, qp, ops):
    """Direct sieve of machine {5..q} over [0, W]; compare opening sets."""
    Wc = (qp * qp - 1) // 6
    b = np.zeros(Wc + 1, dtype=bool)
    small = sieve(q)
    for g in range(5, q + 1):
        if not small[g]:
            continue
        u = pow(6, -1, g)
        b[u::g] = True
        b[(g - u) % g::g] = True
    direct = np.flatnonzero(~b)
    direct = direct[direct >= 1]
    return np.array_equal(direct, np.asarray(ops))


def main():
    lim = QMAX + 400
    sm = sieve(lim)
    primes_small = np.flatnonzero(sm).tolist()
    rungs = [p for p in primes_small if 23 <= p <= QMAX]
    qmax_next = min(p for p in primes_small if p > QMAX)
    NMAX = qmax_next * qmax_next + 10
    isp = sieve(NMAX)
    KMAX = (qmax_next * qmax_next - 1) // 6
    twincols_l = []
    CH = 1 << 22
    for a in range(1, KMAX + 1, CH):
        bnd = min(a + CH, KMAX + 1)
        k = np.arange(a, bnd, dtype=np.int64)
        t = isp[6 * k - 1] & isp[6 * k + 1]
        twincols_l.append(k[t].astype(np.int64))
    twincols = np.concatenate(twincols_l)
    del twincols_l

    lines = []
    W_ = lines.append
    W_("rungs %d..%d, %d of them; sieve to %d; twin columns to %d: %d"
       % (rungs[0], rungs[-1], len(rungs), NMAX, KMAX, twincols.size))
    W_("")
    W_("    q    q'          W     d_0   init   F_pre  x(F_pre)   F_win  x(F_win)   x/W"
       "   minratio[d0,Fpre]  atL  3L-ok  pareto  init_longest")
    rows = []
    viol3 = []
    gates = []
    for q in rungs:
        qp = min(p for p in primes_small if p > q)
        Wc = (qp * qp - 1) // 6
        lo = int(np.searchsorted(twincols, (q + 2) // 6))
        while lo < twincols.size and 6 * int(twincols[lo]) - 1 <= q:
            lo += 1
        hi = int(np.searchsorted(twincols, Wc, side="right"))
        o = twincols[lo:hi]
        rider = bool(isp[qp * qp - 2])
        ops = o.tolist()
        if rider and (not ops or ops[-1] != Wc):
            ops.append(Wc)
        if q in (23, 59, 97, 211):
            gates.append((q, gate(q, qp, ops)))
        ops = np.array(ops, dtype=np.int64)
        d0 = int(ops[0])
        starts = np.empty(ops.size, dtype=np.int64)
        lens = np.empty(ops.size, dtype=np.int64)
        starts[0] = 1
        lens[0] = ops[0] - 1
        starts[1:] = ops[:-1] + 1
        lens[1:] = np.diff(ops) - 1
        m = lens > 0
        starts, lens = starts[m], lens[m]
        Fpre = int(lens.max())
        xpre = int(starts[int(np.argmax(lens))])
        wm = starts > q // 6
        Fwin = int(lens[wm].max())
        xwin = int(starts[wm][int(np.argmax(lens[wm]))])
        rmv = np.maximum.accumulate(lens)
        keep = np.empty(lens.size, dtype=bool)
        keep[0] = True
        keep[1:] = rmv[1:] > rmv[:-1]
        px, pl = starts[keep], lens[keep]

        def rmin(L):
            return int(px[int(np.searchsorted(pl, L, side="left"))])

        rng = list(range(d0, Fpre + 1))
        if rng:
            rd, rdL = min((rmin(L) / L, L) for L in rng)
            ok3 = all(rmin(L) >= 3 * L for L in rng)
        else:
            rd, rdL, ok3 = None, None, None
        if ok3 is False:
            viol3.append(q)
        rows.append(dict(q=q, qp=qp, W=Wc, d0=d0, init=d0 - 1, Fpre=Fpre, xpre=xpre,
                         Fwin=Fwin, xwin=xwin, frac=xwin / Wc, rd=rd, rdL=rdL, ok3=ok3,
                         par=int(px.size), initlong=(xpre == 1)))
        if q <= 150 or q in (211, 401, 601, 809, 997, 1201, 1409, 1423, 1427, 1601,
                             2003, 3001, 4999, 7001, 9973, 14999, 19997):
            W_("%5d %5d %10d %7d %6d %7d %9d %7d %9d %6.3f %14s %8s %6s %6d %6s"
               % (q, qp, Wc, d0, d0 - 1, Fpre, xpre, Fwin, xwin, xwin / Wc,
                  ("%.3f" % rd) if rd else "-", str(rdL), str(ok3), px.size,
                  "YES" if xpre == 1 else "no"))
    W_("")
    W_("GATE (direct sieve of {5..q} over [0,W] vs the twin reduction): %s" % gates)
    W_("")
    W_("SUMMARY over %d rungs q = 23..%d" % (len(rows), rungs[-1]))
    nz = [r for r in rows if r["rd"] is not None]
    W_("  rungs where some run in [1,W] is LONGER than the initial run: %d of %d"
       % (len(nz), len(rows)))
    W_("  of those, rungs where R_min^>=(L) >= 3L fails for some L in [d_0, F_pre]: %d %s"
       % (len(viol3), viol3[:20]))
    if nz:
        b = min(nz, key=lambda r: r["rd"])
        W_("  min over those rungs of min_{L in [d_0,F_pre]} R_min(L)/L: %.4f at q=%d, L=%d"
           % (b["rd"], b["q"], b["rdL"]))
    il = [r["q"] for r in rows if r["initlong"]]
    W_("  rungs where the longest run of [1,W] IS the initial run: %d of %d"
       % (len(il), len(rows)))
    last_not = max((r["q"] for r in rows if not r["initlong"]), default=None)
    W_("  LAST rung whose longest prefix run is not the initial one: q = %s" % last_not)
    W_("  after it, initial-longest at %d of %d rungs"
       % (sum(1 for r in rows if r["q"] > (last_not or 0) and r["initlong"]),
          sum(1 for r in rows if r["q"] > (last_not or 0))))
    fr = np.array([r["frac"] for r in rows])
    W_("  x(F_win)/W: median %.3f; above 0.5 at %.1f%%; above 0.75 at %.1f%%"
       % (float(np.median(fr)), 100 * float(np.mean(fr > 0.5)), 100 * float(np.mean(fr > 0.75))))
    W_("  (d_0-1)/(q/6): min %.3f, median %.3f, max %.3f"
       % (min(r["init"] / (r["q"] / 6) for r in rows),
          float(np.median([r["init"] / (r["q"] / 6) for r in rows])),
          max(r["init"] / (r["q"] / 6) for r in rows)))
    W_("  init/F_win by band (q, mean init, mean F_win, mean ratio):")
    bands = [(23, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10000), (10000, 20000)]
    for a, bnd in bands:
        sel = [r for r in rows if a <= r["q"] < bnd]
        if sel:
            W_("    %6d-%-6d  n=%-4d init %8.1f   F_win %7.1f   init/F_win %6.3f  "
               "initial-longest %5.1f%%"
               % (a, bnd, len(sel), np.mean([r["init"] for r in sel]),
                  np.mean([r["Fwin"] for r in sel]),
                  np.mean([r["init"] / r["Fwin"] for r in sel]),
                  100 * np.mean([r["initlong"] for r in sel])))
    W_("  pareto staircase size: median %.1f, max %d"
       % (float(np.median([r["par"] for r in rows])), max(r["par"] for r in rows)))
    W_("  the window statement in this coordinate: init/W = d_0/W")
    W_("    max (d_0-1)/W = %.5f at q=%d ; min = %.6f at q=%d"
       % (max(r["init"] / r["W"] for r in rows),
          max(rows, key=lambda r: r["init"] / r["W"])["q"],
          min(r["init"] / r["W"] for r in rows),
          min(rows, key=lambda r: r["init"] / r["W"])["q"]))
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "pf_window.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
