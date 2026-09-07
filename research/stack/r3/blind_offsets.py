"""Blindness score of offsets relative to prime squares, against where first twins land.

Relative to a prime square q^2, gear g (g not dividing q) strikes offset i, the slot
(q^2 + 6i - 2, q^2 + 6i), iff -6i or 2 - 6i is a nonzero square mod g.  So whether g CAN strike
offset i at any square is a property of i alone: g is blind at i iff both -6i and 2 - 6i are
non-residues (or zero handled separately).  Score(i) = the number of gears g <= G blind at i.
This script tabulates, for offsets i <= IMAX: the score, and the number of primes q <= QMAX
whose first twin above q^2 sits at offset i; then the first-twin rate per score class against
the availability-corrected expectation.
usage: uv run python research/stack/r3/blind_offsets.py QMAX IMAX G
"""
import os
import sys
from collections import Counter, defaultdict

from sympy import isprime, primerange

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def main():
    QMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    IMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    gears = list(primerange(5, G + 1))
    sq = {g: {(x * x) % g for x in range(1, g)} for g in gears}
    # blind[g][i] : g can never strike offset i relative to any prime square coprime to g
    def blind(g, i):
        a = (-6 * i) % g
        b = (2 - 6 * i) % g
        return (a not in sq[g]) and (b not in sq[g]) and a != 0 and b != 0
    score = [sum(1 for g in gears if blind(g, i)) for i in range(IMAX + 1)]
    # first twins
    first = Counter()
    nq = 0
    for q in primerange(7, QMAX + 1):
        nq += 1
        base = q * q
        for i in range(1, IMAX + 1):
            a = base + 6 * i - 2
            if isprime(a) and isprime(a + 2):
                first[i] += 1
                break
    # per score class: offsets, first-twin landings, and the availability weight
    # availability of offset i at a random prime square = prod over gears of (fraction of q with g not striking i)
    # exact fraction for gear g: g strikes i iff q^2 = -6i or 2-6i mod g; each nonzero square is hit by 2 of the g-1 classes of q
    def avail(i):
        w = 1.0
        for g in gears:
            a = (-6 * i) % g
            b = (2 - 6 * i) % g
            hits = set()
            if a in sq[g]:
                hits.add(a)
            if b in sq[g]:
                hits.add(b)
            w *= 1 - 2 * len(hits) / (g - 1)
        return w
    bycls = defaultdict(lambda: [0, 0, 0.0])
    for i in range(1, IMAX + 1):
        c = bycls[score[i]]
        c[0] += 1
        c[1] += first[i]
        c[2] += avail(i)
    out = [f"# offsets i <= {IMAX}, gears 5..{G} ({len(gears)} gears), first twins above q^2 for primes q <= {QMAX} ({nq} primes; {sum(first.values())} landed below {IMAX})"]
    out.append("score = number of small gears blind at the offset; availability = expected fraction of prime squares at which no small gear strikes the offset")
    out.append("score | offsets | first-twin landings | landings per offset | mean availability | landings / availability")
    for s in sorted(bycls):
        n, f, a = bycls[s]
        out.append(f"{s:5d} | {n:7d} | {f:19d} | {f/n:19.3f} | {a/n:17.5f} | {(f/n)/(a/n) if a else 0:22.3f}")
    top = sorted(range(1, IMAX + 1), key=lambda i: -first[i])[:20]
    out.append("top 20 offsets by first-twin landings (i, landings, score, availability): " + ", ".join(f"({i},{first[i]},{score[i]},{avail(i):.4f})" for i in top))
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(RES, f"blind_offsets_{QMAX}_{IMAX}_{G}.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
