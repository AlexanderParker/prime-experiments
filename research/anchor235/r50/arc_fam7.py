"""Branch 5d.ii.i.a: the duplicated-arc test at K = 7 (arc_famtax.py covers K = 3..6).
Exhaustive over every 7-subset of the primes 5..43."""
import sys, os, itertools, time
from collections import defaultdict
sys.path.insert(0, os.path.abspath('research/anchor235/r50'))
from arc_core import arc, F_of, primes_upto
def tp(gs):
    s=defaultdict(int)
    for g in gs: s[arc(g)]+=1
    return sum(v-1 for v in s.values())
pool=[p for p in primes_upto(43) if p>=5]
subs=list(itertools.combinations(pool,7))
by=defaultdict(list); t=time.time()
for gs in subs:
    by[tp(gs)].append((F_of(list(gs)), gs))
print(f"K=7, pool primes 5..43 ({len(pool)} gears, {len(subs)} subsets), {time.time()-t:.0f}s")
print("    dup arcs   sets     max F   argmax                     mean F")
for d in sorted(by):
    v=by[d]; mx=max(v)
    print(f"      {d:3d}    {len(v):6d}    {mx[0]:5d}   {str(list(mx[1])):25s} {sum(x[0] for x in v)/len(v):7.2f}")
