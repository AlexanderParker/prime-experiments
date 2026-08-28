"""Round 25 (constructor): the realised-gap-tuple DICTIONARY of a machine,
built level by level with NO period scan.

R49/R57: the whole machine input of the history certificate A_m (plain) and of
the survivor system (which needs one order more) is the DICTIONARY of realised
m-tuples of consecutive gaps.  Round 24 got it from a full-period scan.  Here it
is built from the gear list alone:

    D_1 = {v : some gap of M equals v}                    (F queries)
    D_m = {t + (v,) : t in D_{m-1}, t[1:]+(v,) in D_{m-1}, realised}   (overlap
          lemma, R45 - every contiguous sub-tuple of a realised tuple is
          realised, so level m only tests candidates whose two contiguous
          (m-1)-sub-tuples both survived level m-1)

Each membership test is one exact CRT decision (research/crt_dict.py).  So

    F(M)   = max span in D_1,      F_2(M) = max span in D_2,
    F_j(M) = max span in D_j,

all scan-free, and D_4 / D_5 are exactly the inputs A_4 / A_5 need.

GATE: at machines 19, 23, 29 the level-4 dictionary is compared against the
round-24 full-period dumps (research/data/tuples4_*.txt), restricted to the
T3-legal tuples those dumps contain.

Usage:  python research/scanfree_dict.py 23 --levels 4 --workers 6
        python research/scanfree_dict.py gate
"""
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crt_dict                                     # noqa: E402

DDIR = os.path.join(HERE, "data")
_Y = None
_NB = 20_000_000


def _init(y, nb):
    global _Y, _NB
    _Y, _NB = y, nb


def _work(t):
    try:
        return t, crt_dict.realised(_Y, t, _NB)
    except crt_dict.Budget:
        return t, None


class _SeqPool:
    """No-subprocess fallback.  ROUND-25 NOTE: on a memory-pressed box the
    multiprocessing workers were killed under us and pool.map() then blocked
    for ever - a silent hang, not a crash.  workers <= 1 avoids the pool
    entirely."""

    def __init__(self, y, nb):
        _init(y, nb)

    def map(self, fn, it, chunksize=1):
        return [fn(t) for t in it]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def build(y, levels=4, workers=6, cap=None, verbose=True):
    """Returns {m: sorted list of realised m-tuples} and the F_j ladder."""
    F_hi = cap if cap else 5 * y
    D = {}
    Fj = {}
    und = []
    ctx = (_SeqPool(y, _NB) if workers <= 1 else
           Pool(workers, initializer=_init, initargs=(y, _NB)))
    with ctx as pool:
        t0 = time.time()
        cands = [(v,) for v in range(1, F_hi + 1)]
        got = pool.map(_work, cands, chunksize=1)
        D[1] = sorted(t for t, r in got if r)
        und += [t for t, r in got if r is None]
        Fj[1] = max(sum(t) for t in D[1])
        if verbose:
            print("  level 1: %d candidates -> %d realised, F_1 = %d  (%.0fs)"
                  % (len(cands), len(D[1]), Fj[1], time.time() - t0),
                  flush=True)
        for m in range(2, levels + 1):
            t0 = time.time()
            prev = set(D[m - 1])
            cands = []
            for t in D[m - 1]:
                for v in D[1]:
                    if m == 2 or (t[1:] + (v[0],)) in prev:
                        cands.append(t + (v[0],))
            got = pool.map(_work, cands, chunksize=8)
            D[m] = sorted(t for t, r in got if r)
            und += [t for t, r in got if r is None]
            Fj[m] = max(sum(t) for t in D[m]) if D[m] else 0
            if verbose:
                print("  level %d: %d candidates -> %d realised, F_%d = %d  "
                      "(%.0fs)" % (m, len(cands), len(D[m]), m, Fj[m],
                                   time.time() - t0), flush=True)
            if not D[m]:
                break
    return D, Fj, und


def gate():
    print("SCAN-FREE DICTIONARY vs the round-24 full-period dumps\n")
    KNOWN = {19: [25, 31, 37, 38], 23: [34, 39, 50, 55], 29: [43, 55, 65, 70]}
    for y in (19, 23, 29):
        print("machine %d" % y)
        D, Fj, und = build(y, 4, workers=6, cap=crt_dict.KNOWN_F[y] + 20)
        assert not und, ("undecided queries", und[:5])
        assert Fj[1] == crt_dict.KNOWN_F[y], (y, Fj[1])
        assert Fj[2] == crt_dict.KNOWN_F2[y], (y, Fj[2])
        dump = set()
        for line in open(os.path.join(DDIR, "tuples4_%d.txt" % y)):
            dump.add(tuple(int(x) for x in line.split()))
        d4 = set(D[4])
        missing = dump - d4
        assert not missing, ("dump tuples the scan-free dictionary MISSED",
                             sorted(missing)[:10])
        print("  F_1..F_4 = %s   (corpus F = %d, F_2 = %d)"
              % ([Fj[j] for j in sorted(Fj)], crt_dict.KNOWN_F[y],
                 crt_dict.KNOWN_F2[y]))
        print("  level-4 dictionary %d tuples SUPERSET of the %d dumped "
              "(T3-filtered) tuples: 0 missing" % (len(d4), len(dump)))
        # LEVEL-2 CROSS-CHECK against Mechanic's full-period lag-1 pair census.
        # Lateral (round 25) reported that gap_pair_hist.csv's ghist rows are
        # short the cyclic wrap-around gap, which is what the seam-recovery in
        # that loader uses - so this comparison is a direct test of whether the
        # census pair set (as loaded) equals the truth.
        import chain_cegar
        o2 = chain_cegar.load_pairs(y)
        d2 = set(D[2])
        print("  level-2: scan-free %d pairs, census %d pairs;  census-only "
              "%s ;  scan-free-only %s"
              % (len(d2), len(o2), sorted(o2 - d2)[:6], sorted(d2 - o2)[:6]))
        assert not (o2 - d2), ("the census claims a pair the CRT decision "
                               "refutes", sorted(o2 - d2)[:6])
        assert max(sum(p) for p in d2) == crt_dict.KNOWN_F2[y]
        print()
    print("all assertions passed")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    if args[0] == "gate":
        gate()
        return
    y = int(args[0])
    levels = int(args[args.index("--levels") + 1]) if "--levels" in args else 4
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args \
        else 6
    cap = int(args[args.index("--cap") + 1]) if "--cap" in args else None
    t0 = time.time()
    D, Fj, und = build(y, levels, workers, cap)
    print("\nmachine %d, scan-free spectrum: %s   (%.0fs)"
          % (y, {("F_%d" % j): Fj[j] for j in sorted(Fj)}, time.time() - t0))
    if und:
        print("UNDECIDED at the node budget: %d tuples, e.g. %s"
              % (len(und), und[:5]))
    out = os.path.join(DDIR, "sfdict_%d.txt" % y)
    with open(out, "w") as f:
        for m in sorted(D):
            for t in D[m]:
                f.write(" ".join(str(x) for x in t) + "\n")
    print("dictionary written to %s" % out)


if __name__ == "__main__":
    main()
