"""Branch 5d.ii.i.a, item 4: the level-2 hole-distance dictionary of the small part.

For an initial segment of gears (the small/tiler part) and each run length L: the
minimum number of holes it can leave, and - when that minimum is 2 - which distances
those two holes can be at.  The big part must supply exactly those distances, so this
dictionary is what a bound on A(K) would have to control.
"""
import itertools, json, os, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arc_core import RESULTS

def dictionary(gears, Lmax):
    rows = []
    for L in range(2, Lmax + 1):
        masks = []
        for g in gears:
            d = pow(3, -1, g)
            ms = set()
            for o in range(g):
                m = 0
                for i in range(o, L, g):
                    m |= 1 << i
                for i in range((o + d) % g, L, g):
                    m |= 1 << i
                ms.add(m)
            masks.append(sorted(ms))
        best = L + 1
        dists = Counter()
        for combo in itertools.product(*masks):
            cov = 0
            for m in combo:
                cov |= m
            h = L - bin(cov).count("1")
            if h < best:
                best = h; dists = Counter()
            if h == best == 2:
                holes = [i for i in range(L) if not (cov >> i & 1)]
                dists[holes[1] - holes[0]] += 1
        rows.append((L, best, dict(sorted(dists.items()))))
    return rows

def main():
    log = open(os.path.join(RESULTS, "arc_dict.txt"), "w")
    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()
    OUT = {}
    for gears, Lmax in (([5, 7], 12), ([5, 7, 11], 18), ([5, 7, 11, 13], 20),
                        ([5, 7, 11, 13, 17], 22)):
        say(f"small part {gears}")
        say("     L   min holes   hole-pair distances at the 2-hole optimum (count)")
        rows = dictionary(gears, Lmax)
        for L, best, dists in rows:
            say(f"    {L:3d}     {best:3d}       {dists if dists else ''}")
        OUT[str(gears)] = rows
    json.dump(OUT, open(os.path.join(RESULTS, "arc_dict.json"), "w"))
    log.close()

if __name__ == "__main__":
    main()
