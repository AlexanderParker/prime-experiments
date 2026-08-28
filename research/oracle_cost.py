"""Round 25 (constructor): THE COST CURVE OF THE SCAN-FREE DICTIONARY, by
ARITY - the measurement that decides where the chain stops.

research/crt_dict.py answers "is this gap tuple realised?" by an exact CRT
cover search.  The chain's own queries are arity 2 and 4 (states and edges of
MF_4).  The ONE-GAP question - "is there a gap of exactly v?" - is the same
machinery at arity 1, and it is the question that pins F(M) exactly.

This script measures both, per machine, separately for WITNESS (realised - the
search stops at the first solution) and REFUTATION (unrealised - the search
must exhaust the tree).  Refutations are the whole cost: a certificate deletes
only on a refutation.

Usage: python research/oracle_cost.py 19 23 29 31 37 [--budget 60000000]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crt_dict                                       # noqa: E402

# the previous rung's certified budget F(prev) + q, i.e. what the chain KNOWS
# about F(M) before it pins it
PREV_BUDGET = {13: 20, 17: 28, 19: 37, 23: 48, 29: 63, 31: 74, 37: 95,
               41: 129, 43: 134}


def sweep(y, nb):
    """Pin F(M) exactly, starting from the previous rung's certified bound."""
    hi = PREV_BUDGET[y]
    t0 = time.time()
    nref = 0
    refsec = 0.0
    worst = (0.0, None)
    for v in range(hi, 0, -1):
        t1 = time.time()
        try:
            ok, _, nodes = crt_dict.realised_nodes(y, (v,), node_budget=nb)
        except crt_dict.Budget:
            print("    v = %d: UNDECIDED at %d nodes (%.0f s) - the sweep "
                  "cannot pin F(%d)" % (v, nb, time.time() - t1, y),
                  flush=True)
            return None, time.time() - t0, nref, worst
        dt = time.time() - t1
        if ok:
            return v, time.time() - t0, nref, worst
        nref += 1
        refsec += dt
        if dt > worst[0]:
            worst = (dt, v)
    return None, time.time() - t0, nref, worst


def main():
    argv = sys.argv[1:]
    nb = 60_000_000
    if "--budget" in argv:
        i = argv.index("--budget")
        nb = int(argv[i + 1])
        del argv[i:i + 2]
    ys = [int(a) for a in argv if a.isdigit()] or [19, 23, 29, 31, 37]
    print("PINNING F(M) FROM THE PREVIOUS RUNG'S CERTIFIED BOUND\n")
    print("  M    F-bar  F   refutations  worst single   total")
    for y in ys:
        F, tot, nref, worst = sweep(y, nb)
        known = crt_dict.KNOWN_F.get(y)
        if F is not None and known is not None:
            assert F == known, (y, F, known)
        print("  %-4d %5d  %-4s %8d   %8.1f s (v=%s)  %8.1f s"
              % (y, PREV_BUDGET[y], F, nref, worst[0], worst[1], tot),
              flush=True)
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
