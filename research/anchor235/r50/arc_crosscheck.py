"""Branch 5d.ii.i.a: independent cross-check of the MILP's infeasibility claims.

The MILP (arc_milp) says no K gears block L columns, quantified over ALL primes via the
type reduction.  This script re-asks the same question with the r48 tool (cover_core,
a different implementation and a different algorithm) by brute force over every
K-subset of the primes 5..POOL.  Agreement is a cross-check, not a second proof: the
enumeration omits gears above POOL, which at these L are bare dominoes of arc >= 26.

Usage: uv run python .../arc_crosscheck.py K L POOL
"""
import itertools, os, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r48"))
sys.path.insert(0, HERE)
from cover_core import coverable
from arc_core import primes_upto, RESULTS

K = int(sys.argv[1]); L = int(sys.argv[2]); POOL = int(sys.argv[3])

def job(gs):
    return gs, coverable(L, list(gs))

def main():
    pool = [p for p in primes_upto(POOL) if p >= 5]
    subs = list(itertools.combinations(pool, K))
    t = time.time(); found = []
    with Pool(3) as pl:
        for gs, ok in pl.imap_unordered(job, subs, chunksize=8):
            if ok:
                found.append(gs)
    msg = (f"K={K} L={L} pool primes 5..{POOL} ({len(pool)} gears, {len(subs)} subsets): "
           f"{len(found)} sets cover {L} columns, {time.time()-t:.0f}s")
    print(msg, flush=True)
    with open(os.path.join(RESULTS, f"arc_crosscheck_K{K}_L{L}.txt"), "w") as f:
        f.write(msg + "\n")
        for g in found[:20]:
            f.write("   " + str(list(g)) + "\n")

if __name__ == "__main__":
    main()
