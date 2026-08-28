"""Round 25: the exact CONSISTENT degree-2 threshold W*_cons(19).

Round 23 stopped its consistent bisection at machine 17, leaving the m19 cell
blank; round 24's composition certified width 33 there and asked whether the
recursion or consistency owns that width.  This bisects the consistent-only
threshold exactly, with BOTH verdicts exact (a certificate on the infeasible
side, an exhibited completable point on the feasible side) and every feasible
witness saved to disk - the round-24 process rule.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cw_decide25 import decideC25                                 # noqa: E402
from lp_degree_range import gears_of, F_EXACT                     # noqa: E402

g = gears_of(19)
lo, hi = F_EXACT[19], 33            # 25 must be feasible; 33 is certified
print("bisecting W*_cons(19) in [%d, %d]" % (lo, hi), flush=True)
seen = {}
while hi - lo > 0:
    mid = (lo + hi) // 2
    t0 = time.time()
    v, info = decideC25(g, mid, 2, verbose=False)
    seen[mid] = v
    print("  W = %d: %s  (%d rows, %d its) [%.0fs]"
          % (mid, v, info['rows'], info['its'], time.time() - t0), flush=True)
    assert v in ('CERTIFIED', 'REFUTED'), ("undecided at W = %d" % mid)
    if v == 'CERTIFIED':
        hi = mid
    else:
        lo = mid + 1
print("W*_cons(19) = %d   (verdicts: %s)" % (lo, sorted(seen.items())),
      flush=True)
