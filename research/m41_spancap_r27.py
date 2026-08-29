"""Round 27 (mechanic): the F_4(41) SPAN CAP applied to the m41 4-tuple
superset - a whole band of the census decided with no solver call.

A realised 4-tuple of machine 41 is four CONSECUTIVE gaps, so its span is a
sum of four consecutive gaps and therefore at most F_4(41), BY DEFINITION.
The standing bound was the deletion-ladder one F_4(41) <= F(53) = 145 (C11),
which is exactly the superset's own maximum span - i.e. it pruned nothing.
Round 27 computes F_4(41) EXACTLY by the floor-1 lap-phase transfer (K2) and
uses it, so the whole band (F_4(41), 145] of the superset is ZERO outright.

Usage: python research/m41_spancap_r27.py F4
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m41_census_r27 import load_arr, SCREENED, OUT           # noqa: E402

TRANSFER = os.path.join(HERE, "data", "gap_tuples_41_4_transfer.csv")


def main():
    F4 = int(sys.argv[1])
    os.makedirs(OUT, exist_ok=True)
    for label, path in (("dict_transfer superset (K4)", TRANSFER),
                        ("phase-saturation screened (C31)", SCREENED)):
        a = load_arr(path)
        sp = a.sum(axis=1)
        keep = sp <= F4
        print("%-34s %9d tuples -> %9d survive span <= %d  "
              "(%9d ZERO BY DEFINITION, %.2f%%)"
              % (label, len(a), int(keep.sum()), F4,
                 int((~keep).sum()), 100.0 * (~keep).mean()))
        if path == SCREENED:
            out = os.path.join(OUT, "gap_tuples_41_4_screened_spancap.csv")
            b = a[keep]
            order = np.lexsort((b[:, 3], b[:, 2], b[:, 1], b[:, 0]))
            with open(out, "w") as fh:
                fh.write("g1,g2,g3,g4\n")
                for row in b[order]:
                    fh.write("%d,%d,%d,%d\n" % tuple(int(x) for x in row))
            print("   wrote %s" % out)
            # induced lower-arity dictionaries
            for m in (1, 2, 3):
                s = set()
                for row in b:
                    r = [int(x) for x in row]
                    for i in range(0, 4 - m + 1):
                        s.add(tuple(r[i:i + m]))
                print("   induced %d-tuple dictionary: %d" % (m, len(s)))


if __name__ == "__main__":
    main()
