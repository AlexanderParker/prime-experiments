"""Round 25 (constructor): the one-gap refutation cost at machine 41 - the
first machine where the scan-free decision procedure does NOT decide within a
sane budget.  F(41) = 91 is known (corpus / Mechanic COV-SAT), so the question
'is there a gap of exactly 92?' has answer NO; this measures what it costs the
CRT decider to prove it."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crt_dict


def main():
    for v in (91, 92, 93):
        t0 = time.time()
        try:
            ok, wit, nodes = crt_dict.realised_nodes(41, (v,),
                                                     node_budget=400_000_000)
            print("m41 gap %d: %s  (%d nodes, %.1f s)"
                  % (v, "REALISED" if ok else "refuted", nodes,
                     time.time() - t0), flush=True)
        except crt_dict.Budget:
            print("m41 gap %d: UNDECIDED at 4e8 nodes (%.1f s)"
                  % (v, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
