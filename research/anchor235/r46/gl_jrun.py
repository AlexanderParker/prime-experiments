"""G8 / item 5: the J-run outer law.  For J consecutive gaps the glue construction (right base
shifted by the middle span) bounds g_1 + g_J by F_2 when the colouring works.  Measured here:
M_J = max over J-runs of (g_1 + g_J), with every middle >= 6 and, separately, with no condition,
against F_2 and 2F."""
import numpy as np
from gl_glue import gears_of, us_of, sieve, gap_stats, cov_pair, solve_cover

for top in (13, 17, 19, 23):
    gears = gears_of(top); us = us_of(gears)
    P, blocked = sieve(gears, us)
    opens, gaps, F, F2, N = gap_stats(P, blocked)
    n = gaps.size
    print(f"\nm{top}: F={F} F_2={F2} 2F={2*F}")
    for J in range(3, 9):
        # J-run: gaps[i..i+J-1]; outer = gaps[i] + gaps[i+J-1]; middles gaps[i+1..i+J-2]
        i = np.arange(n - J)
        outer = gaps[i] + gaps[i + J - 1]
        okmid = np.ones(i.size, dtype=bool)
        for t in range(1, J - 1):
            okmid &= gaps[i + t] >= 6
        m_all = int(outer.max())
        m_mid = int(outer[okmid].max()) if okmid.any() else -1
        # a witness at the constrained max, and the C2 verdict there
        verdict = "-"
        if m_mid > 0:
            k = int(np.flatnonzero(okmid & (outer == m_mid))[0])
            x0 = int(opens[k]); L = int(gaps[k])
            S = int(gaps[k + 1:k + J - 1].sum())
            y = x0 + L + S; RR = int(gaps[k + J - 1])
            T, h, cL, cR = cov_pair(gears, us, x0, L, RR, y)
            verdict = "OK" if solve_cover(cL, cR, T, h) else "FAIL"
        print(f"   J={J}: max outer (any middles) {m_all:3d}; with all middles >=6 {m_mid:3d} "
              f"({'<=' if m_mid <= F2 else '> '} F_2)   C2 at that witness: {verdict}")
