"""Round 26 (constructor): AN A-PRIORI CAP ON THE CEGAR QUERY COUNT.

R67 item (ii): "THE QUERY COUNT.  181, 90, 955, 3,399 at the four certified
rungs - growing, and bounded by NOTHING PROVEN."  That sentence is too weak,
and this script says by how much.

THE CAP (proved, trivially, and it is the point that the loop MEMOISES).  The
refinement loop only ever asks two kinds of question:
  * "is this value 4-tuple realised?"  - and the tuple is the value label of a
    LIVE MF_4 EDGE, so it lies in the set of distinct value 4-tuples carried by
    the machine-free system at (F, q');
  * "is this value pair realised?"     - the (flank, base) pair of a live MF_4
    STATE, so it lies in the set of distinct such pairs.
Every answer is memoised, so no tuple is ever asked twice.  Hence for ANY
refinement strategy whatever

    queries  <=  T_4(F, q')  +  T_2(F, q')   <=  F^4 + F^2,

where T_4, T_2 are the distinct value-tuple counts of MF_4 - both computable
in advance from the two integers (F, q') alone, with no machine in them.  The
bound is uniform in y once F(M) is: the query count is NOT unbounded, it is
bounded by a machine-free function of the step's own two parameters.

WHAT IS OPEN, stated precisely.  The measured counts are 10^-2 to 10^-5 of the
cap, so the cap is true and enormously loose; what is not bounded by anything
proven is the RATIO - how much of the machine-free tuple space a certificate
actually has to visit.  That ratio is what this script measures.

Usage:  python research/query_cap.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain_cegar                                       # noqa: E402
from machinefree_cert import build_mf_edges              # noqa: E402

DDIR = os.path.join(HERE, "data")
NEG = -(1 << 40)


def caps(F, q1, mod=35, m=4):
    S, esrc, edst, ew, Rs, Ls, tup = build_mf_edges(F, q1, mod, m)
    t4 = len({tuple(int(x) for x in r) for r in tup})
    ok = Ls > NEG // 2
    t2 = len({(int(a), int(b)) for a, b in zip(Ls[ok], Rs[ok])})
    return S, len(esrc), t4, t2


def main():
    rows = json.load(open(os.path.join(DDIR, "r26_query_law.json")))
    print("THE QUERY-COUNT CAP  (T_4 + T_2 = distinct value tuples of MF_4;\n"
          " the loop memoises, so NO strategy can exceed it)\n")
    print("  M    q'   F   queries   T_4      T_2     cap = T_4+T_2   "
          "used %     F^4")
    out = []
    for r in rows:
        S, E, t4, t2 = caps(r["F"], r["q1"])
        cap = t4 + t2
        assert r["q"] <= cap, (r["y"], r["q"], cap)
        print("  %-4d %-4d %-4d %7d  %7d  %7d  %13d   %7.4f  %10d"
              % (r["y"], r["q1"], r["F"], r["q"], t4, t2, cap,
                 100.0 * r["q"] / cap, r["F"] ** 4))
        out.append(dict(r, t4=t4, t2=t2, cap=cap,
                        used=100.0 * r["q"] / cap))
    json.dump(out, open(os.path.join(DDIR, "r26_query_cap.json"), "w"))
    print("\nall assertions passed (every measured count is under its cap)")


if __name__ == "__main__":
    main()
