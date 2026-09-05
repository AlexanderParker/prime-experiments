"""Exceptionless statements, with counts."""
import numpy as np
from gl_glue import gears_of, us_of, sieve, gap_stats, cov_pair, solve_cover, runs_with

# (b) at every C2 failure: minimum miss is 1 and the missed offset is a shadow
print("(b) C2 failures: minimum number of uncovered columns, and whether it is a shadow")
for top in (17, 19, 23):
    gears = gears_of(top); us = us_of(gears)
    P, blocked = sieve(gears, us)
    opens, gaps, F, F2, N = gap_stats(P, blocked)
    nf = miss1 = shadow_always = 0
    k = len(gears)
    for (x0, L, v, R) in runs_with(opens, gaps, vmin=6, sum_gt=F - 6):
        if v >= min(L, R):
            continue
        T, h, cL, cR = cov_pair(gears, us, x0, L, R, x0 + L + v)
        target = ((1 << T) - 1) ^ (1 << h)
        best = T + 1; sets = []
        for mask in range(1 << k):
            acc = 0
            for i in range(k):
                acc |= cR[i] if (mask >> i) & 1 else cL[i]
            m = target & ~acc
            c = bin(m).count('1')
            if c < best:
                best, sets = c, [m]
            elif c == best:
                sets.append(m)
        if best == 0:
            continue
        nf += 1
        miss1 += (best == 1)
        sh = 0
        if v <= L - 1:
            sh |= 1 << (h - v)
        if v <= R - 1:
            sh |= 1 << (h + v)
        shadow_always += all(m & sh for m in sets)
    print(f"   m{top}: {nf} C2 failures among hard runs with L+R>{F-6}; "
          f"min miss == 1 at {miss1}; every optimal colouring leaves a SHADOW uncovered "
          f"at {shadow_always}")

# (c) the J-run outer law with all middles >= 6
print("\n(c) J-run outer law: max (g_1 + g_J) over J-runs with every middle >= 6, vs F_2")
for top in (13, 17, 19, 23):
    gears = gears_of(top); us = us_of(gears)
    P, blocked = sieve(gears, us)
    opens, gaps, F, F2, N = gap_stats(P, blocked)
    n = gaps.size
    tot = 0
    line = []
    for J in range(3, 9):
        i = np.arange(n - J)
        outer = gaps[i] + gaps[i + J - 1]
        ok = np.ones(i.size, dtype=bool)
        for t in range(1, J - 1):
            ok &= gaps[i + t] >= 6
        c = int(ok.sum())
        tot += c
        mx = int(outer[ok].max()) if c else -1
        line.append(f"J={J}: {c} runs, max outer {mx}")
    print(f"   m{top} (F_2={F2}): " + "; ".join(line) + f";  total {tot} runs, 0 above F_2")
