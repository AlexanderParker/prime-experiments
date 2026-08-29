"""
LATERAL round 27 - the mechanism probe for tooth-counterfactual-percentile.md.

Every symmetric tooth vector (v_q) is v_q = m^{-1} mod q for some integer m (by
CRT), and the TWIN machine is m = 6.  P13 asked whether the twin's low F is the
feature "m is small".  Sweep m and look.

Usage: python tooth_msweep.py [--upto 60]
"""
import argparse
import sys

import numpy as np

from tooth_counterfactual import maxgap

GEARS = [5, 7, 11, 13, 17, 19]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=60)
    a = ap.parse_args()
    P = 1
    for q in GEARS:
        P *= q
    rows = []
    for m in range(1, a.upto + 1):
        if any(m % q == 0 for q in GEARS):
            continue
        v = [min(pow(m, -1, q) % q, (-pow(m, -1, q)) % q) for q in GEARS]
        F, _ = maxgap(GEARS, v, P)
        rows.append((m, F, tuple(v)))
    vals = [r[1] for r in rows]
    print("m19 (gears %s, P = %d): teeth +-m^{-1} mod q, m = 1..%d coprime to the gears"
          % (GEARS, P, a.upto))
    print("  samples %d   min %d  median %.1f  max %d"
          % (len(vals), min(vals), float(np.median(vals)), max(vals)))
    print("  full family V(19) for comparison: min 20  median 28  max 43")
    twin = dict((r[0], r[1]) for r in rows)[6]
    print("  TWIN (m = 6): F = %d ; argmin over the sweep: m = %s"
          % (twin, [r[0] for r in rows if r[1] == min(vals)]))
    for m, F, v in rows:
        print("   m=%-3d F=%-3d v=%s" % (m, F, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
