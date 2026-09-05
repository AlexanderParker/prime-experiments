"""Branch 5d.ii.i.a, item 3/6: F at FIXED gear count, sorted by how many arcs the set
duplicates (= how many twin pairs it contains).  Exhaustive over all K-subsets of a
prime pool.  Usage: uv run python .../arc_famtax.py"""
import itertools, json, os, sys, time
from collections import defaultdict
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arc_core import arc, F_of, primes_upto, RESULTS

def twin_pairs(gears):
    seen = defaultdict(int)
    for g in gears:
        seen[arc(g)] += 1
    return sum(v - 1 for v in seen.values())

def job(gs):
    return gs, F_of(list(gs))

def main():
    log = open(os.path.join(RESULTS, "arc_famtax.txt"), "w")
    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()
    fam = {}
    for K, POOL in ((3, 61), (4, 61), (5, 47), (6, 43)):
        pool = [p for p in primes_upto(POOL) if p >= 5]
        subs = list(itertools.combinations(pool, K))
        t = time.time()
        by = defaultdict(list)
        with Pool(3) as pl:
            for gs, f in pl.imap_unordered(job, subs, chunksize=16):
                by[twin_pairs(gs)].append((f, gs))
        say(f"K={K}, pool primes 5..{POOL} ({len(pool)} gears, {len(subs)} subsets), "
            f"{time.time()-t:.0f}s")
        say("    dup arcs   sets     max F   argmax                     mean F")
        for d in sorted(by):
            v = by[d]
            mx = max(v)
            say(f"      {d:3d}    {len(v):6d}    {mx[0]:5d}   {str(list(mx[1])):25s} "
                f"{sum(x[0] for x in v)/len(v):7.2f}")
        fam[K] = {str(d): {"n": len(v), "max": max(v)[0], "argmax": list(max(v)[1]),
                           "mean": sum(x[0] for x in v)/len(v)} for d, v in by.items()}
    json.dump(fam, open(os.path.join(RESULTS, "arc_famtax.json"), "w"))
    log.close()

if __name__ == "__main__":
    main()
