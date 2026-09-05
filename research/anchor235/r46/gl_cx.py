"""The two residual runs whose cross glue was not found in the first sample: a wider search."""
import numpy as np
from gl_glue import gears_of, us_of, sieve, gap_stats, cov_pair, solve_cover
gears = gears_of(23); us = us_of(gears)
P, blocked = sieve(gears, us)
opens, gaps, F, F2, N = gap_stats(P, blocked)
rng = np.random.default_rng(11)
for (x0, L, v, R) in ((15578190, 25, 7, 10), (28701300, 25, 7, 10)):
    cand = np.flatnonzero(gaps >= R)
    order = rng.permutation(cand.size)[:60000]
    best = None
    for t in order.tolist():
        i = int(cand[t]); y = int(opens[i]); Rp = int(gaps[i])
        T, h, cL, cR = cov_pair(gears, us, x0, L, Rp, y)
        if solve_cover(cL, cR, T, h) is not None:
            best = (y, Rp)
            if Rp >= R:
                break
    print(f"({L},{v},{R}) x0={x0}: cross glue over {min(60000,cand.size)} partners -> "
          f"{'F_2 >= %d at y=%d (R\'=%d), loss %d' % (L+best[1], best[0], best[1], max(0,(L+R)-(L+best[1]))) if best else 'none found'}")
