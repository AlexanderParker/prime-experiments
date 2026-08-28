"""Round 25 diagnostic: WHERE does the machine-29 decider's 1.35 GB of
committed memory come from?  Measure the bit-lengths of the exact rationals
that flow into the separation oracle, iteration by iteration.  No verdict is
taken here - this is instrumentation only."""
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cw_consistent import RelaxCF                                 # noqa: E402
from lp_degree_range import gears_of, separate, ZERO             # noqa: E402


def bits(fr):
    return max(fr.numerator.bit_length(), fr.denominator.bit_length())


y, W, iters = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
R = RelaxCF(gears_of(y), W, 2)
t0 = time.time()
for it in range(iters):
    t, z, res = R._solve_float()
    den = 10 ** (4 + min(it // 40, 4))
    zr = R.rationalise(z, den)
    zex = R.repair_consistency(zr)
    bz_pre = max(bits(v) for v in zr)
    bz = max(bits(v) for v in zex)
    bm = 0
    added = 0
    for i in range(W):
        mom = R.moments_at(zex, i)
        bm = max(bm, max(bits(v) for v in mom.values()))
        lam = separate(mom, R.n, 2, Fraction(1, 10 ** 5))
        if lam is not None:
            R.rows.append((i, lam))
            added += 1
    print("it %d: t=%+.4f cuts=%d rows=%d | max bits: z(rounded)=%d "
          "z(repaired)=%d moments=%d  [%.0fs]"
          % (it, t, added, len(R.rows), bz_pre, bz, bm, time.time() - t0),
          flush=True)
