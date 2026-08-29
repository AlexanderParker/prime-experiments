"""Round 26 (constructor): HOW MANY REALISABILITY QUERIES DOES A RUNG COST?

R67 item (ii): the chain generates every INTEGER it needs, but not the QUERY
COUNT - 181, 90, 955, 3,399 at the four rungs round 25 certified, "growing,
and bounded by nothing proven".  This script measures the count on the WHOLE
ladder under ONE fixed strategy so the numbers are comparable, and reports the
correlates.

FIXED STRATEGY (the round-25 canonical one, so the four published counts are
reproduced rather than re-defined): MF_4 mod 35 machine-free start, topk = 1
(refine along the single arg-max walk), NO given F_2 integer (R58: the loop
needs none), node budget 2e6, oracle = the exact realised-tuple predicate.

ORACLE EQUIVALENCE.  The loop is deterministic given the ANSWERS, and the two
oracles available - the scan-free CRT decision and the exact full-period
census - decide the same predicate.  So the query COUNT is oracle-independent,
and that is checked here rather than assumed: machines 23, 29, 31 are run with
the census oracle and must reproduce round 25's CRT counts 90, 955, 3,399.

Usage:  python research/query_law.py [--steps 11,13,17,19,23,29,31,37]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain_cegar                                       # noqa: E402
import chain_dict_oracle                                 # noqa: E402
from machinefree_cert import build_mf_edges              # noqa: E402

DDIR = os.path.join(HERE, "data")
# round-25 published counts under the same strategy (CRT oracle)
R25 = {19: 181, 23: 90, 29: 955, 31: 3399}


def main():
    args = sys.argv[1:]
    ys = ([int(x) for x in args[args.index("--steps") + 1].split(",")]
          if "--steps" in args else [11, 13, 17, 19, 23, 29, 31, 37])
    nb = 2_000_000
    rows = []
    for y in ys:
        F, Q1, EXACT = chain_cegar.STEPS[y]
        if y in chain_dict_oracle.DICT_CSV:
            orc = chain_dict_oracle.ExactDictOracle(y)
            kind = "census"
        else:
            orc = chain_cegar.CRTOracle(y, nb)
            kind = "CRT"
        print("\n=== %d -> %d   F = %d  budget %d   oracle = %s"
              % (y, Q1, F, F + Q1, kind), flush=True)
        t0 = time.time()
        r = chain_cegar.run_step(y, orc, topk=1, f2=0, verbose=False)
        secs = time.time() - t0
        S, esrc, _, _, _, _, _ = build_mf_edges(F, Q1, 35, 4)
        q = r.get("q4", 0) + r.get("q2", 0)
        chk = ""
        if y in R25:
            chk = "vs round-25 %d %s" % (R25[y],
                                         "MATCH" if q == R25[y] else "DIFFER")
            assert q == R25[y], (y, q, R25[y])
        print("   %s  bound %s  budget %d   queries %d (%d arity-4 + %d "
              "arity-2)  it %d  %.0fs  %s"
              % (r["status"], r.get("bound"), F + Q1, q, r.get("q4", 0),
                 r.get("q2", 0), r["it"], secs, chk), flush=True)
        assert r["status"] == "CERTIFIED", (y, r["status"], r.get("bound"))
        rows.append(dict(y=y, q1=Q1, F=F, budget=F + Q1, exact=EXACT,
                         q=q, q4=r.get("q4", 0), q2=r.get("q2", 0),
                         k4=r.get("k4", 0), k2=r.get("k2", 0),
                         it=r["it"], states=S, edges=len(esrc),
                         secs=secs, oracle=kind))
    print("\n\nQUERY COUNT ALONG THE LADDER (topk = 1, no given integer)")
    print("  M    q'   F   budget  MF_4 states  MF_4 edges  queries   a4  "
          "  a2   iters   q/F^2   q/edges")
    for r in rows:
        print("  %-4d %-4d %-4d %5d  %11d %11d  %7d %5d %5d %7d  %6.3f  "
              "%8.5f"
              % (r["y"], r["q1"], r["F"], r["budget"], r["states"],
                 r["edges"], r["q"], r["q4"], r["q2"], r["it"],
                 r["q"] / r["F"] ** 2, r["q"] / max(1, r["edges"])))
    json.dump(rows, open(os.path.join(DDIR, "r26_query_law.json"), "w"))
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
