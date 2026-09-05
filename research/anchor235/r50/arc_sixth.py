"""Branch 5d.ii.i.a: the hole-distance dictionary of {5,7,11,13,17} past L = 22, and the
sixth-gear sweep it predicts.

At L = 27 the small part {5..17} can leave its two holes only at distances 5, 7 or 18.
Every short arc is even, so a single big gear can span only 18, and 53 = 3*18 - 1 is the
only prime with arc 18 (55 is not prime).  The sweep confirms it: F({5..17, g}) = 28 only
at g = 53.  This is r48's K = 4 mechanism one level up, predicted before it was measured.
"""
import itertools, os, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r48"))
sys.path.insert(0, HERE)
from cover_core import F_of
from arc_core import arc, primes_upto, RESULTS

GEARS = [5, 7, 11, 13, 17]

def main():
    log = open(os.path.join(RESULTS, "arc_sixth.txt"), "w")
    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()
    say(f"dictionary of {GEARS}")
    for L in range(18, 33):
        masks = []
        for g in GEARS:
            d = pow(3, -1, g); ms = set()
            for o in range(g):
                m = 0
                for i in range(o, L, g): m |= 1 << i
                for i in range((o + d) % g, L, g): m |= 1 << i
                ms.add(m)
            masks.append(sorted(ms))
        best, dists = L + 1, Counter()
        for combo in itertools.product(*masks):
            cov = 0
            for m in combo: cov |= m
            h = L - bin(cov).count("1")
            if h < best: best, dists = h, Counter()
            if h == best == 2:
                holes = [i for i in range(L) if not (cov >> i & 1)]
                dists[holes[1] - holes[0]] += 1
        say(f"  L={L:3d}  min holes {best}  2-hole distances {dict(sorted(dists.items()))}")
    say("")
    say("sixth-gear sweep, F({5,7,11,13,17,g}):")
    say("   g     " + "".join(f"{g:4d}" for g in primes_upto(89) if g >= 23))
    gs = [g for g in primes_upto(89) if g >= 23]
    say("   a_g   " + "".join(f"{arc(g):4d}" for g in gs))
    say("   F     " + "".join(f"{F_of(GEARS + [g]):4d}" for g in gs))
    log.close()

if __name__ == "__main__":
    main()
