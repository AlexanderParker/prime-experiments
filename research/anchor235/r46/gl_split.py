"""The trivial/hard split of the attaining 3-runs: how much of the 95.5% is the peel bound."""
import sys
from gl_glue import gears_of, us_of, sieve, gap_stats, glue, runs_with
from gl_shadow import min_moves

o = sys.stdout
for top in (13, 17, 19, 23):
    gears = gears_of(top); us = us_of(gears)
    P, blocked = sieve(gears, us)
    opens, gaps, F, F2, N = gap_stats(P, blocked)
    att = {v: s for v, s in N.items() if v >= 6}
    tr = trok = hd = hdok = 0; mv = {}
    for (x0, L, v, R) in runs_with(opens, gaps, vmin=6, only_attaining=att):
        good, bl, br = min_moves(gears, us, x0, L, v, R)
        if v >= min(L, R):
            tr += 1; trok += good
        else:
            hd += 1; hdok += good
        if good:
            m = min(bl, br); mv[m] = mv.get(m, 0) + 1
    o.write(f"m{top}: attaining v>=6: {tr+hd};  trivial (v>=min(L,R)) {trok}/{tr}; "
            f"HARD (v<min(L,R)) {hdok}/{hd} = {100*hdok/max(hd,1):.1f}%;  moves {dict(sorted(mv.items()))}\n")
