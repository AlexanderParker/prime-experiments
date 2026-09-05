"""Branch 5d.ii.i.a: is F a function of the SHORT-ARC multiset alone?

Every gear is g = 3a + e with a = a_g the short arc and e = +-1; the long arc is
g - a = 2a + e.  So a gear's full arc content is the PAIR (a, 2a+e), and the two
members of a twin pair share a but differ in e, hence in the LONG arc, by 2.
This script groups every K-subset of a prime pool by its short-arc multiset and
reports the spread of F inside each group: the part of F the short arcs decide, and
the part the signs decide.
"""
import itertools, json, os, sys, time
from collections import defaultdict
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arc_core import arc, F_of, primes_upto, RESULTS

def job(gs):
    return gs, F_of(list(gs))

def main():
    log = open(os.path.join(RESULTS, "arc_signature.txt"), "w")
    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()
    out = {}
    for K, POOL in ((3, 61), (4, 61), (5, 47)):
        pool = [p for p in primes_upto(POOL) if p >= 5]
        subs = list(itertools.combinations(pool, K))
        by = defaultdict(list)
        t = time.time()
        with Pool(3) as pl:
            for gs, f in pl.imap_unordered(job, subs, chunksize=16):
                by[tuple(sorted(arc(g) for g in gs))].append((f, gs))
        multi = {k: v for k, v in by.items() if len(v) > 1}
        const = sum(1 for v in multi.values() if len(set(x[0] for x in v)) == 1)
        spreads = sorted(((max(x[0] for x in v) - min(x[0] for x in v), k, v)
                          for k, v in multi.items()), reverse=True)
        say(f"K={K}, pool 5..{POOL}: {len(subs)} sets, {len(by)} short-arc multisets, "
            f"{len(multi)} of them realised by more than one gear set, "
            f"{time.time()-t:.0f}s")
        say(f"    multisets on which F is CONSTANT: {const}/{len(multi)}")
        say(f"    largest spreads of F within one short-arc multiset:")
        for sp, k, v in spreads[:8]:
            lo = min(v); hi = max(v)
            say(f"      arcs {str(list(k)):18s} spread {sp:3d}   "
                f"min F={lo[0]} at {list(lo[1])}   max F={hi[0]} at {list(hi[1])}")
        out[K] = {"n_sets": len(subs), "n_multisets": len(by), "n_multi": len(multi),
                  "n_const": const,
                  "max_spread": spreads[0][0] if spreads else 0}
    json.dump(out, open(os.path.join(RESULTS, "arc_signature.json"), "w"))
    log.close()

if __name__ == "__main__":
    main()
