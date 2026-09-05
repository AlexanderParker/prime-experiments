"""Branch 5d.ii.i.a, item 4: the refined matching bound on A(K), as a ladder of
relaxations, and the level-2 hole-distance dictionary of the small part.

Every gear whose long arc does not fit in the run is a bare domino of size a_g.  Relax
what the adversary may buy for those, keeping the concrete (small) gears real:

  B1  hole count only  : each big gear = ANY two columns of the run (or one)
  B2  + parity         : any two columns at an EVEN distance (a_g = (g -+ 1)/3 is
                         always even, since 3 a_g = g -+ 1 = 0 mod 6)
  B3  + realisable arcs: any two columns at a distance a with 3a - 1 or 3a + 1 prime,
                         UNLIMITED multiplicity
  A   the truth        : arcs of primes that are actually big at this L, multiplicity
                         1 or 2 (2 exactly at a twin pair)

B1 >= B2 >= B3 >= A, all exhaustive.  The gaps say how much of A(K) is decided by the
hole COUNT (a counting quantity, which face A forbids using), by parity, by which arcs
the primes offer, and by how many times they offer them.

Also: the level-2 dictionary of the small part - for the initial segments {5,7},
{5,7,11}, {5,7,11,13}, {5..17} - which hole-pair distances occur at each L.

Usage: uv run python research/anchor235/r50/arc_bound.py [KMAX]
"""
import itertools
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arc_core import Level, arc, is_prime, primes_upto, RESULTS  # noqa: E402


def realisable_arcs(amax):
    return [a for a in range(2, amax + 1, 2)
            if is_prime(3 * a - 1) or is_prime(3 * a + 1)]


class Relax(Level):
    """Level with the big gears replaced by a relaxed 'super domino'."""

    def __init__(self, L, K, mode):
        self.mode = mode
        Level.__init__(self, L, K)
        conc = [(k, key, m) for k, key, m in self.items if k == 'p']
        self.items = conc + [('R', mode, K)]
        self.cap = self.cap[:len(conc)] + [2]
        self.pmask = self.pmask[:len(conc)] + [None]
        for w in self.wins:
            self.wcap[w] = self.wcap[w][:len(conc)] + [2]
        self.arcset = set(realisable_arcs(2 * L + 4))

    def options(self, idx, pos):
        kind, key, _m = self.items[idx]
        if kind != 'R':
            return Level.options(self, idx, pos)
        L = self.L
        outs = [1 << pos]
        for j in range(pos + 1, L):
            d = j - pos
            if key == 'B1' or (key == 'B2' and d % 2 == 0) or \
               (key == 'B3' and d in self.arcset):
                outs.append((1 << pos) | (1 << j))
        return outs


def ladder(K, mode, L0):
    L = L0
    while True:
        lv = Relax(L, K, mode) if mode else Level(L, K)
        if not lv.coverable():
            return L, lv.nodes
        L += 1


def dictionary(gears, Lmax):
    """For each L, the hole-count minimum of this small part, and the hole-pair
    distances that occur when exactly two holes are left."""
    rows = []
    for L in range(2, Lmax + 1):
        full = (1 << L) - 1
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
        singles = 0
        for combo in itertools.product(*masks):
            cov = 0
            for m in combo:
                cov |= m
            h = L - bin(cov).count("1")
            if h < best:
                best = h
                dists = Counter()
                singles = 0
            if h == best and h == 2:
                holes = [i for i in range(L) if not (cov >> i & 1)]
                dists[holes[1] - holes[0]] += 1
            if h == best and h == 1:
                singles += 1
        rows.append((L, best, dict(sorted(dists.items()))))
    return rows


def main():
    KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    os.makedirs(RESULTS, exist_ok=True)
    log = open(os.path.join(RESULTS, "arc_bound.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    KNOWN = {4: 16, 5: 22, 6: 28}
    say("=== the relaxation ladder ===")
    say("  K    B1 (count)   B2 (+parity)   B3 (+prime arcs)   A(K)   "
        "B1/A   B2/A   B3/A   window W(p_{K+1})")
    ps = primes_upto(500)
    out = []
    for K in range(2, KMAX + 1):
        A = KNOWN.get(K)
        row = {}
        for mode in ("B3", "B2", "B1"):
            t = time.time()
            v, nodes = ladder(K, mode, max(2, (A or 2)))
            row[mode] = v
            say(f"    K={K} {mode} = {v}  ({nodes} nodes, {time.time()-t:.1f}s)")
        qn = ps[K + 2]
        W = (qn * qn - 1) // 6
        if A:
            say(f"  {K:2d}   {row['B1']:6d}      {row['B2']:6d}         "
                f"{row['B3']:6d}         {A:4d}   {row['B1']/A:5.2f}  "
                f"{row['B2']/A:5.2f}  {row['B3']/A:5.2f}    {W:8d}")
        out.append({"K": K, "A": A, **row, "W": W})

    say("")
    say("=== the level-2 dictionary of the small part ===")
    DICT = {}
    for gears, Lmax in (([5, 7], 14), ([5, 7, 11], 20), ([5, 7, 11, 13], 22),
                        ([5, 7, 11, 13, 17], 24)):
        say(f"  small part {gears}")
        say("     L   min holes   hole-pair distances at the 2-hole optimum")
        rows = dictionary(gears, Lmax)
        for L, best, dists in rows:
            say(f"    {L:3d}     {best:3d}       {dists}")
        DICT[str(gears)] = rows
    json.dump({"ladder": out, "dictionary": DICT},
              open(os.path.join(RESULTS, "arc_bound.json"), "w"))
    log.close()


if __name__ == "__main__":
    main()
