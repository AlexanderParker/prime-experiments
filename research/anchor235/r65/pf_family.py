"""r65 / position-length frontier, part 5: the tooth-counterfactual family.

A member of the family keeps the gears {5..q} and gives gear g teeth at +-d_g for a d_g drawn
uniformly from [1, (g-1)/2] instead of the real d_g = (g -+ 1)/6.  Column 0 stays open.

Part A: full periods m13, m17, m19 -- 20 members each -- the frontier constant
        c = min over L >= d_0 of R_min^>=(L)/L, and the initial run's length.
Part B: the prefix [1, W] at rungs 211, 401, 997 -- 20 members each -- whether the run
        starting at column 1 is the longest run of the prefix (the real machine's anomaly).

Self-contained, numpy only.  Run: uv run python research/anchor235/r65/pf_family.py
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
SEED = 20260906


def sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def runs_from_bool(b):
    a = np.empty(b.size + 2, dtype=np.int8)
    a[0] = 0
    a[-1] = 0
    a[1:-1] = b
    d = np.diff(a)
    st = np.flatnonzero(d == 1)
    en = np.flatnonzero(d == -1)
    return st, en - st


def frontier_stats(starts, lens):
    mx = int(lens.max())
    o = np.argsort(starts, kind="stable")
    ss, ll = starts[o], lens[o]
    rm = np.maximum.accumulate(ll)
    keep = np.empty(ss.size, dtype=bool)
    keep[0] = True
    keep[1:] = rm[1:] > rm[:-1]
    px, pl = ss[keep], ll[keep]

    def rmin(L):
        return int(px[int(np.searchsorted(pl, L, side="left"))])

    d0 = int(ss[0] + ll[0]) if ss[0] == 1 else 1   # first opening
    init = int(ll[0]) if ss[0] == 1 else 0
    rng = list(range(init + 1, mx + 1))
    c = min((rmin(L) / L, L) for L in rng) if rng else (float("inf"), 0)
    return mx, init, c, px.size


def main():
    rng = np.random.default_rng(SEED)
    lines = []
    W_ = lines.append
    isp = sieve(1_100_000)
    primes = [p for p in range(5, 1000) if isp[p]]

    W_("=" * 84)
    W_("A. FULL PERIODS: the frontier constant c on 20 counterfactual members")
    W_("=" * 84)
    for q in [13, 17, 19]:
        gears = [p for p in primes if p <= q]
        P = 1
        for g in gears:
            P *= g
        # real member
        real_d = [(g - 1) // 6 if (g - 1) % 6 == 0 else (g + 1) // 6 for g in gears]
        rows = []
        for mi in range(21):
            if mi == 0:
                dd = real_d
                tag = "REAL"
            else:
                dd = [int(rng.integers(1, (g - 1) // 2 + 1)) for g in gears]
                tag = "fam%02d" % mi
            b = np.zeros(P, dtype=bool)
            for g, d in zip(gears, dd):
                b[d % g::g] = True
                b[(g - d) % g::g] = True
            st, ln = runs_from_bool(b)
            mx, init, c, npar = frontier_stats(st, ln)
            rows.append((tag, mx, init, c[0], c[1], npar, dd))
        W_("")
        W_("m%d  gears=%s  P=%d   real teeth=%s" % (q, gears, P, real_d))
        W_("  member   F-1   init   c=min R_min/L (L>init)   atL   pareto   teeth")
        for tag, mx, init, c, cl, npar, dd in rows:
            W_("  %-7s %4d %6d %22.3f %6d %7d   %s"
               % (tag, mx, init, c, cl, npar, dd))
        fam = [r for r in rows if r[0] != "REAL"]
        real = rows[0]
        W_("  REAL c = %.3f ; family c: min %.3f, median %.3f, max %.3f ; "
           "members with c < REAL: %d of 20"
           % (real[3], min(r[3] for r in fam), float(np.median([r[3] for r in fam])),
              max(r[3] for r in fam), sum(1 for r in fam if r[3] < real[3])))
        W_("  REAL init = %d ; family init: median %.1f, max %d ; members with init >= REAL: %d of 20"
           % (real[2], float(np.median([r[2] for r in fam])), max(r[2] for r in fam),
              sum(1 for r in fam if r[2] >= real[2])))

    W_("")
    W_("=" * 84)
    W_("B. THE PREFIX [1, W]: is the run at column 1 the longest? (the real anomaly)")
    W_("=" * 84)
    for q in [211, 401, 997]:
        qp = q + 1
        while not isp[qp]:
            qp += 1
        Wc = (qp * qp - 1) // 6
        gears = [p for p in primes if p <= q]
        real_d = [(g - 1) // 6 if (g - 1) % 6 == 0 else (g + 1) // 6 for g in gears]
        W_("")
        W_("rung q=%d  q'=%d  W=%d  gears=%d" % (q, qp, Wc, len(gears)))
        W_("  member    init   F_pre   x(F_pre)   init longest?   pareto")
        stats = []
        for mi in range(21):
            if mi == 0:
                dd, tag = real_d, "REAL"
            else:
                dd = [int(rng.integers(1, (g - 1) // 2 + 1)) for g in gears]
                tag = "fam%02d" % mi
            b = np.zeros(Wc + 1, dtype=bool)
            for g, d in zip(gears, dd):
                b[d % g::g] = True
                b[(g - d) % g::g] = True
            b[0] = False
            st, ln = runs_from_bool(b)
            # drop a truncated tail
            if st[-1] + ln[-1] == Wc + 1:
                st, ln = st[:-1], ln[:-1]
            init = int(ln[0]) if st[0] == 1 else 0
            fpre = int(ln.max())
            xpre = int(st[int(np.argmax(ln))])
            o = np.argsort(st, kind="stable")
            rm = np.maximum.accumulate(ln[o])
            npar = 1 + int((rm[1:] > rm[:-1]).sum())
            stats.append((tag, init, fpre, xpre, xpre == 1, npar))
            if mi <= 6 or mi == 20:
                W_("  %-7s %6d %7d %10d %14s %8d"
                   % (tag, init, fpre, xpre, "YES" if xpre == 1 else "no", npar))
        fam = stats[1:]
        W_("  REAL: init=%d, F_pre=%d, init longest=%s"
           % (stats[0][1], stats[0][2], stats[0][4]))
        W_("  family: init median %.1f (max %d) against REAL %d ; init-longest at %d of 20 ;"
           " F_pre median %.1f"
           % (float(np.median([r[1] for r in fam])), max(r[1] for r in fam), stats[0][1],
              sum(1 for r in fam if r[4]), float(np.median([r[2] for r in fam]))))
        W_("  real init / family median init = %.2f"
           % (stats[0][1] / max(float(np.median([r[1] for r in fam])), 1e-9)))

    txt = "\n".join(lines)
    with open(os.path.join(OUT, "pf_family.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
