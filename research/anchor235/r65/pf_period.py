"""r65 / position-length frontier, part 1: full periods m7..m23.

For each machine M = {5..q} over its whole period P:
  - the list of maximal blocked runs (start, length);
  - R_min^>=(L), R_min^=(L), R_max^=(L) for every realised L;
  - the mirror identity R_max^=(L) = P - R_min^=(L) - L + 1;
  - the PARETO STAIRCASE: the runs that are longer than every run starting earlier
    (these determine R_min^>= completely);
  - the ratio R_min^>=(L)/L and its minimum over L >= 6 and over L >= d_0;
  - the effective-machine bound  x >= ceil((y_L^2-1)/6) - L + 1  and its slack.

Self-contained, numpy only.  Peak memory: about 250 MB at m23.
Run: uv run python research/anchor235/r65/pf_period.py
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31]
FLADDER = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}


def blocked_period(gears):
    P = 1
    for g in gears:
        P *= g
    b = np.zeros(P, dtype=bool)
    for g in gears:
        u = pow(6, -1, g)
        b[u::g] = True
        b[(g - u) % g::g] = True
    return b, P


def maximal_runs(b):
    a = np.empty(b.size + 2, dtype=np.int8)
    a[0] = 0
    a[-1] = 0
    a[1:-1] = b
    d = np.diff(a)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, (ends - starts)


def y_for_L(L):
    for p in PRIMES:
        if FLADDER[p] >= L + 1:
            return p
    return None


def bound_for_L(L):
    y = y_for_L(L)
    if y is None:
        return None
    return -(-(y * y - 1) // 6) - L + 1


def main():
    lines = []
    W = lines.append
    for q in [7, 11, 13, 17, 19, 23]:
        gears = [p for p in PRIMES if p <= q]
        b, P = blocked_period(gears)
        starts, lens = maximal_runs(b)
        maxlen = int(lens.max())
        F = maxlen + 1
        assert F == FLADDER[q]
        d0 = int(starts[0] + lens[0])
        W("")
        W("=" * 84)
        W("machine m%d  gears=%s  P=%d  F=%d  longest run=%d  d_0=%d  runs=%d"
          % (q, gears, P, F, maxlen, d0, starts.size))
        W("=" * 84)

        rmin_eq, rmax_eq = {}, {}
        for s, L in zip(starts.tolist(), lens.tolist()):
            if L not in rmin_eq:
                rmin_eq[L] = s
            rmax_eq[L] = s
        rmin_ge = {}
        for L in range(1, maxlen + 1):
            rmin_ge[L] = int(starts[lens >= L].min())

        W("")
        W("  L   count   R_min>=(L)  R_min=(L)   R_max=(L)  mirror  ratio>=   bound  slack")
        mirror_bad = 0
        viol_L = []
        for L in range(1, maxlen + 1):
            cnt = int((lens == L).sum())
            rg = rmin_ge[L]
            re_ = rmin_eq.get(L)
            rx = rmax_eq.get(L)
            bd = bound_for_L(L)
            if bd is not None and rg < bd:
                viol_L.append(L)
            sl = ("%.2f" % (rg / bd)) if bd else "-"
            if re_ is None:
                W("  %2d  %7d  %11d  %9s  %10s  %6s  %8.3f  %5s  %6s"
                  % (L, cnt, rg, "-", "-", "-", rg / L, bd, sl))
                continue
            ok = (rx == P - re_ - L + 1)
            if not ok:
                mirror_bad += 1
            W("  %2d  %7d  %11d  %9d  %10d  %6s  %8.3f  %5s  %6s"
              % (L, cnt, rg, re_, rx, "yes" if ok else "NO", rg / L, bd, sl))
        W("")
        W("  mirror mismatches: %d of %d realised lengths" % (mirror_bad, len(rmin_eq)))
        W("  effective-machine bound violated at L = %s   (d_0 - 1 = %d)"
          % (viol_L, d0 - 1))
        real = sorted(rmin_eq)
        inv = [(a, c) for a, c in zip(real, real[1:]) if rmin_eq[a] > rmin_eq[c]]
        W("  R_min^= inversions: %d  %s" % (len(inv), inv[:10]))
        W("  R_min^>= monotone: %s"
          % all(rmin_ge[L] <= rmin_ge[L + 1] for L in range(1, maxlen)))
        r6 = [(rmin_ge[L] / L, L) for L in range(6, maxlen + 1)]
        rd = [(rmin_ge[L] / L, L) for L in range(d0, maxlen + 1)]
        if r6:
            W("  min ratio over L>=6:    %.4f at L=%d" % min(r6))
        if rd:
            W("  min ratio over L>=d_0:  %.4f at L=%d" % min(rd))
        W("  min ratio over all L:   %.4f at L=%d"
          % min([(rmin_ge[L] / L, L) for L in range(1, maxlen + 1)]))
        W("  initial run: start 1, length %d" % (d0 - 1))
        W("  R_min^>=(F-1) = %d = %.4f P" % (rmin_ge[maxlen], rmin_ge[maxlen] / P))

        # Pareto staircase
        o = np.argsort(starts, kind="stable")
        ss, ll = starts[o], lens[o]
        run_max = np.maximum.accumulate(ll)
        keep = np.empty(ss.size, dtype=bool)
        keep[0] = True
        keep[1:] = run_max[1:] > run_max[:-1]
        px, pl = ss[keep], ll[keep]
        W("")
        W("  PARETO STAIRCASE (x, L, L/x, effective machine {5..y} with y=sqrt(6(x+L)-5)):")
        for x, L in zip(px.tolist(), pl.tolist()):
            top = 6 * (x + L - 1) + 1
            y = max([p for p in PRIMES if p * p <= top], default=0)
            W("    x=%-10d L=%-3d  x/L=%8.3f   6(x+L)-5=%-12d y=%-3d F({5..y})=%s"
              % (x, L, x / L, top, y, FLADDER.get(y, "-")))
        del b
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "pf_period.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
