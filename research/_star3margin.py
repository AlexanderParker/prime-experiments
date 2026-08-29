"""round 26 LP thread: STAR-k conditional row margins at k = 3 (hold 5,7,11)."""
import os, sys, time
from itertools import product
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from star_case import case_margin
from lp_degree_range import gears_of, budget

for y in (41, 53):
    g = gears_of(y); W = budget(y)
    for k in (3,):
        held = g[:k]
        t = time.time(); vals = []
        for ws in product(*[range(q) for q in held]):
            v, npos, inex = case_margin(g, W, held, ws)
            vals.append((ws, v))
        mn = min(vals, key=lambda x: x[1]); mx = max(vals, key=lambda x: x[1])
        mean = sum(v for _, v in vals) / len(vals)
        nbad = sum(1 for _, v in vals if v <= 0)
        print("m%d W=%d hold %s: %d cases, min %+.4f at %s, max %+.4f at %s,"
              " mean %+.4f, %d cases <= 0  [%.0fs]"
              % (y, W, held, len(vals), float(mn[1]), mn[0], float(mx[1]),
                 mx[0], float(mean), nbad, time.time() - t), flush=True)
