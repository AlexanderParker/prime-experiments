"""Round 28 (mechanic): IS THE DEPTH-0 LEMMA'S HYPOTHESIS q' > 2(m+1) SHARP?

The lemma says D_m(M) subset D_m(M + q') whenever q' > 2(m+1) - the proof needs
an admissible phase for the new gear and the pattern forbids at most 2(m+1)
residues.  A lemma with an unnecessary hypothesis is a worse lemma, and a lemma
whose hypothesis is exactly where it starts failing is a better one.  This
sweeps m upward at every small step and records where monotonicity first
breaks.

Prediction registered before running: monotonicity survives somewhat past the
proof's threshold (the bound 2(m+1) counts residues with multiplicity and the
exposed set collides mod small q'), so the first failure should sit ABOVE
q' = 2(m+1), not at it.

Usage:  <venv>/python research/depth0_sharp_r28.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from onset_ladder_r28 import gaps_cyclic, ktuples      # noqa: E402

# machines 19 and below only: an (N, m) int64 array at m23 with m = 12 is
# 763 MB before np.unique copies it, and MEMORY IS THE BINDING CONSTRAINT
# on this box.  m19's is 36 MB.  (Learned by launching the m23 version and
# killing it.)
STEPS = [(7, 11), (11, 13), (13, 17), (17, 19)]
MS = list(range(2, 13))


def main():
    print("D_m(M) subset D_m(M + q') ?   ('.' = holds, 'X' = FAILS)")
    print("  the proof covers q' > 2(m+1); the threshold column marks where "
          "that stops applying\n")
    header = "  step     " + "".join("%4d" % m for m in MS)
    print(header)
    print("  m ->" + " " * 6 + "".join("%4s" % "" for m in MS))
    cache = {}
    for M, qp in STEPS:
        row = []
        first_fail = None
        for m in MS:
            for y in (M, qp):
                if (y, m) not in cache:
                    cache[(y, m)] = ktuples(gaps_cyclic(y), m)
            bad = cache[(M, m)] - cache[(qp, m)]
            ok = not bad
            row.append("." if ok else "X")
            if not ok and first_fail is None:
                first_fail = (m, sorted(bad, key=sum)[0])
        thr = max(m for m in MS if qp > 2 * (m + 1))
        print("  %2d->%2d   " % (M, qp) + "".join("%4s" % c for c in row)
              + "     proof covers m <= %d;  first failure at m = %s"
              % (thr, first_fail[0] if first_fail else "none in range"))
        if first_fail:
            print("           first missing %d-tuple: %s (span %d)"
                  % (first_fail[0], first_fail[1], sum(first_fail[1])))


if __name__ == "__main__":
    main()
